"""End-to-end classification pipeline.

Wires the pieces together: validate input -> retrieve SOPs (optional) ->
build the prompt with SOPs injected -> call the LLM behind the schema-fallback
guard -> apply the confidence floor -> set rag_sources authoritatively -> log
-> return. The only public entry point is ``classify_ticket``; everything
else (prompting, retrieval, the model call, logging) lives in its own module.

If RAG is enabled but the store is unavailable, classification degrades to a
no-SOP run with a logged warning rather than failing the user's request.
"""

import json
import logging
from typing import Final

from pydantic import BaseModel, ValidationError

from src.guardrails import apply_confidence_floor, classify_with_schema_fallback
from src.llm import LLMCallMetadata
from src.observability import log_classification
from src.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from src.rag import RAGError, SOPChunk, retrieve_sops
from src.schema import MAX_TICKET_CHARS, ClassificationOutput, TicketInput

_RAG_K: Final[int] = 5

# Reuses the logger observability already configured (imported above), so the
# degradation notice lands in the same structured stream.
_logger: Final[logging.Logger] = logging.getLogger("ticket_classifier")


class ClassifierError(RuntimeError):
    """The ticket could not be classified — invalid input, or the model
    failed even after the guardrail retry (§4.4: no bare exceptions)."""


class ClassifiedResult(BaseModel):
    """The §6 output plus the call telemetry and review flag the UI needs.
    ``classify_ticket`` returns only ``output`` to keep its §4.1 contract;
    ``classify_ticket_detailed`` returns this for callers (the Gradio sidebar)
    that also need latency/tokens/cost and the human-review flag."""

    output: ClassificationOutput
    metadata: LLMCallMetadata
    needs_human_review: bool


def _build_prompt_with_sops(chunks: list[SOPChunk]) -> str:
    sop_block = "".join(f"### {chunk.filename}\n{chunk.text}\n\n" for chunk in chunks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "## Relevant standard operating procedures\n"
        "Ground the suggested_response in these procedures and do not state "
        "any policy they do not support.\n\n"
        f"{sop_block}"
    )


def classify_ticket_detailed(text: str, *, use_rag: bool = True) -> ClassifiedResult:
    """Run the full pipeline and return the output plus call telemetry.

    Same behaviour as :func:`classify_ticket`; returns a ``ClassifiedResult``
    so the UI can show latency/tokens/cost and the human-review flag.

    Raises:
        ClassifierError: The input was empty/blank.
        GuardrailError: The model could not produce schema-valid output even
            after the single re-prompt.
    """
    # Explicit checks first so the user sees *why* (empty vs too long) rather
    # than a generic Pydantic message; TicketInput stays the contract backstop.
    if len(text) > MAX_TICKET_CHARS:
        raise ClassifierError(
            f"Ticket exceeds the {MAX_TICKET_CHARS}-character limit "
            f"({len(text)} chars). Please shorten it."
        )
    try:
        ticket = TicketInput(body=text)
    except ValidationError as exc:
        raise ClassifierError("Ticket text is empty.") from exc
    if not ticket.body.strip():
        raise ClassifierError("Ticket text is blank.")
    ticket_text = ticket.body

    retrieved: list[SOPChunk] = []
    if use_rag:
        try:
            retrieved = retrieve_sops(ticket_text, k=_RAG_K)
        except RAGError as exc:
            # Infra problem, not a user problem: proceed without grounding
            # rather than failing the request, but make the gap visible.
            _logger.warning(json.dumps({"event": "rag_degraded", "reason": str(exc)}))

    prompt = _build_prompt_with_sops(retrieved) if retrieved else SYSTEM_PROMPT
    result = classify_with_schema_fallback(prompt, FEW_SHOT_EXAMPLES, ticket_text)

    output, needs_human_review = apply_confidence_floor(result.output)
    # Authoritative: trust the retrieval layer, not the model, for provenance.
    output = output.model_copy(
        update={"rag_sources": [chunk.filename for chunk in retrieved]}
    )

    log_classification(ticket_text, output, result.metadata, needs_human_review)
    return ClassifiedResult(
        output=output,
        metadata=result.metadata,
        needs_human_review=needs_human_review,
    )


def classify_ticket(text: str, *, use_rag: bool = True) -> ClassificationOutput:
    """Classify a single support ticket and draft a suggested response.

    Args:
        text: Raw ticket body. Subject and body should be concatenated by the
            caller with a newline separator.
        use_rag: When True, retrieve SOPs from ChromaDB and inject them into
            the prompt. Disable for latency-sensitive paths or prompt-only
            eval.

    Returns:
        A validated ClassificationOutput. ``confidence`` below 0.3 forces the
        category to "Other" and is flagged for human review in the logs.
        ``rag_sources`` is set by this function to the SOPs actually injected,
        not whatever the model echoed.

    Raises:
        ClassifierError: The input was empty/blank.
        GuardrailError: The model could not produce schema-valid output even
            after the single re-prompt.
    """
    return classify_ticket_detailed(text, use_rag=use_rag).output
