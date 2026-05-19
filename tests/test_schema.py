"""Schema-contract tests: JSON round-trip, enum rejection, bound rejection.

These guard the §6 contract. They never touch the network — the schema is
pure — so they are safe to run in CI.
"""

import pytest
from pydantic import ValidationError

from src.prompts import FEW_SHOT_EXAMPLES
from src.schema import MAX_TICKET_CHARS, ClassificationOutput, TicketInput


def _valid_output() -> ClassificationOutput:
    return ClassificationOutput(
        category="Billing",
        subcategory="Duplicate charge",
        priority="High",
        sentiment="Frustrated",
        entities={"amount": "$29.99"},
        summary="Customer was billed twice and wants one charge reversed.",
        suggested_response=(
            "We're sorry for the duplicate charge and have reversed it; "
            "it should clear within 5-7 business days."
        ),
        confidence=0.9,
    )


def test_json_round_trip():
    original = _valid_output()
    restored = ClassificationOutput.model_validate_json(original.model_dump_json())
    assert restored == original


def test_rejects_invalid_category():
    with pytest.raises(ValidationError):
        ClassificationOutput(**{**_valid_output().model_dump(), "category": "Plumbing"})


def test_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        ClassificationOutput(**{**_valid_output().model_dump(), "confidence": 1.5})


def test_rejects_confidence_below_zero():
    with pytest.raises(ValidationError):
        ClassificationOutput(**{**_valid_output().model_dump(), "confidence": -0.1})


def test_rejects_too_short_summary():
    with pytest.raises(ValidationError):
        ClassificationOutput(**{**_valid_output().model_dump(), "summary": "short"})


def test_rejects_unknown_field():
    # extra="forbid" must reject model-invented fields, not silently drop them.
    with pytest.raises(ValidationError):
        ClassificationOutput(**{**_valid_output().model_dump(), "urgency": "now"})


def test_ticket_input_rejects_empty_body():
    with pytest.raises(ValidationError):
        TicketInput(body="")


def test_ticket_input_allows_near_empty_body():
    assert TicketInput(body=".").combined_text == "."


def test_ticket_input_rejects_oversize_body():
    TicketInput(body="x" * MAX_TICKET_CHARS)  # exactly at cap is allowed
    with pytest.raises(ValidationError):
        TicketInput(body="x" * (MAX_TICKET_CHARS + 1))


def test_ticket_input_combines_subject_and_body():
    ticket = TicketInput(subject="Refund", body="Where is my money")
    assert ticket.combined_text == "Refund\nWhere is my money"


def test_exactly_five_valid_few_shots():
    # Phase 2 requires exactly five; each must satisfy the contract by being a
    # real ClassificationOutput instance (enforced at import, asserted here).
    assert len(FEW_SHOT_EXAMPLES) == 5
    for example in FEW_SHOT_EXAMPLES:
        assert isinstance(example.output, ClassificationOutput)
