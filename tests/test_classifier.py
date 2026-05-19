"""End-to-end classifier/guardrail tests with the LLM mocked.

The OpenAI boundary (``src.guardrails.classify_via_llm``) is patched, so the
real classifier, guardrail, and observability code runs but no network call is
made (CLAUDE.md Phase 9: never hit the real API in CI). RAG is disabled so no
ChromaDB/embeddings are touched either.
"""

from unittest.mock import patch

import pytest

from src.classifier import (
    ClassifiedResult,
    ClassifierError,
    classify_ticket,
    classify_ticket_detailed,
)
from src.guardrails import (
    GuardrailError,
    apply_confidence_floor,
    classify_with_schema_fallback,
    redact_pii,
)
from src.llm import LLMCallMetadata, LLMError, LLMResult
from src.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from src.schema import ClassificationOutput

_TARGET = "src.guardrails.classify_via_llm"


def _output(**overrides) -> ClassificationOutput:
    base = {
        "category": "Billing",
        "subcategory": "Duplicate charge",
        "priority": "High",
        "sentiment": "Frustrated",
        "entities": {},
        "summary": "Customer was billed twice and wants it fixed.",
        "suggested_response": "Sorry about that — the duplicate charge will be reversed.",
        "confidence": 0.9,
        "rag_sources": ["bogus.md"],
    }
    base.update(overrides)
    return ClassificationOutput(**base)


def _result(output: ClassificationOutput | None = None, **meta) -> LLMResult:
    metadata = LLMCallMetadata(
        model="gpt-4o-mini",
        latency_ms=120.0,
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=0.0001,
    )
    for key, value in meta.items():
        setattr(metadata, key, value)
    return LLMResult(output=output or _output(), metadata=metadata)


def _classified() -> ClassifiedResult:
    r = _result()
    return ClassifiedResult(
        output=r.output, metadata=r.metadata, needs_human_review=False
    )


def test_classify_ticket_clears_rag_sources_when_rag_off():
    with patch(_TARGET, return_value=_result(_output(rag_sources=["bogus.md"]))):
        out = classify_ticket("My card was charged twice", use_rag=False)
    assert isinstance(out, ClassificationOutput)
    # rag_sources is authoritative: no RAG run -> empty, model's echo dropped.
    assert out.rag_sources == []


def test_classify_ticket_detailed_exposes_metadata():
    with patch(_TARGET, return_value=_result()):
        res = classify_ticket_detailed("billed twice", use_rag=False)
    assert isinstance(res, ClassifiedResult)
    assert res.metadata.model == "gpt-4o-mini"
    assert res.needs_human_review is False


def test_confidence_floor_forces_other_and_flags_review():
    low = _output(category="Billing", confidence=0.2)
    with patch(_TARGET, return_value=_result(low)):
        res = classify_ticket_detailed("vague ticket", use_rag=False)
    assert res.output.category == "Other"
    assert res.needs_human_review is True


@pytest.mark.parametrize("text", ["", "   "])
def test_blank_input_raises_classifier_error(text):
    with patch(_TARGET, return_value=_result()):
        with pytest.raises(ClassifierError):
            classify_ticket(text, use_rag=False)


def test_schema_fallback_retries_once_then_succeeds():
    with patch(_TARGET, side_effect=[LLMError("bad json"), _result()]) as mock:
        res = classify_with_schema_fallback(SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, "t")
    assert mock.call_count == 2
    assert res.metadata.schema_retried is True


def test_schema_fallback_double_failure_raises_guardrail_error():
    with patch(_TARGET, side_effect=[LLMError("one"), LLMError("two")]):
        with pytest.raises(GuardrailError):
            classify_with_schema_fallback(SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, "t")


def test_redact_pii_masks_email_and_phone_but_not_short_numbers():
    redacted = redact_pii("mail me@x.io or call +1 415 555 1234 re order 12")
    assert "me@x.io" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "order 12" in redacted  # short id must survive redaction


def test_apply_confidence_floor_does_not_mutate_original():
    confident = _output(confidence=0.9)
    out, flagged = apply_confidence_floor(confident)
    assert flagged is False and out.category == "Billing"

    weak = _output(category="Shipping", confidence=0.1)
    out2, flagged2 = apply_confidence_floor(weak)
    assert flagged2 is True and out2.category == "Other"
    assert weak.category == "Shipping"  # copy, original untouched


def test_input_cap_rejects_oversize_before_calling_model():
    # >4000 chars must raise a clear ClassifierError *before* any API call.
    with patch(_TARGET) as mock:
        with pytest.raises(ClassifierError, match="4000-character limit"):
            classify_ticket("x" * 4001, use_rag=False)
    mock.assert_not_called()


def test_session_rate_limit_blocks_at_cap_without_calling_model():
    import app

    with patch.object(app, "classify_ticket_detailed") as mock:
        # At the cap: rejected, model not called, counter unchanged.
        card, raw, _, used = app._classify("hi", app._SESSION_LIMIT)
        assert (
            "Session limit reached" in card and raw == {} and used == app._SESSION_LIMIT
        )
        mock.assert_not_called()


def test_session_rate_limit_increments_under_cap():
    import app

    with patch.object(app, "classify_ticket_detailed", return_value=_classified()):
        _, _, _, used = app._classify("a billing question", 0)
    assert used == 1
