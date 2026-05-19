"""Safety rails around the raw LLM call.

Three independent guards, all pure except the one LLM re-call:

* PII redaction — strip emails/phones from text *before it is logged* (the
  returned answer is untouched; it may legitimately echo a customer email).
* Confidence floor — a result the model is barely sure of is downgraded to
  "Other" and flagged for human review (§7 step 3). §6 has no review field,
  so the flag is returned to the caller for the observability log, not added
  to the contract.
* Schema fallback — if the model's output cannot be coerced into the schema,
  re-prompt exactly once with the error, then give up loudly.

This module does not log or retrieve; it only validates and re-asks.
"""

import re
from typing import Final

from src.llm import LLMError, LLMResult, classify_via_llm
from src.prompts import FewShotExample
from src.schema import ClassificationOutput

# Below this confidence the classification is too weak to action automatically
# (CLAUDE.md §7 step 3).
_CONFIDENCE_FLOOR: Final[float] = 0.3

# Deliberately broad for emails. The phone pattern requires 8+ digits with
# common separators so it does not eat short order ids; over-redacting a log
# is safer than leaking a number, but eating every "#1234" would gut the log.
_EMAIL_RE: Final[re.Pattern[str]] = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE: Final[re.Pattern[str]] = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")


class GuardrailError(RuntimeError):
    """The model could not produce schema-valid output even after the single
    re-prompt (§4.4: no bare exceptions)."""


def redact_pii(text: str) -> str:
    """Replace emails and phone numbers with placeholders. Email is redacted
    first so an email's digits cannot later be mistaken for a phone number."""
    redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    return _PHONE_RE.sub("[REDACTED_PHONE]", redacted)


def apply_confidence_floor(
    output: ClassificationOutput,
) -> tuple[ClassificationOutput, bool]:
    """Downgrade a low-confidence result.

    Returns:
        The (possibly rewritten) output and ``needs_human_review``. When
        confidence is below the floor the category is forced to "Other" via a
        copy — the original is not mutated — and the flag is True.
    """
    if output.confidence >= _CONFIDENCE_FLOOR:
        return output, False
    return output.model_copy(update={"category": "Other"}), True


def classify_with_schema_fallback(
    prompt: str, examples: list[FewShotExample], ticket: str
) -> LLMResult:
    """Call the LLM; on any ``LLMError`` re-prompt once with the error text.

    A second failure is unrecoverable here (re-prompting again would just burn
    tokens), so it is surfaced as ``GuardrailError``.

    Raises:
        GuardrailError: The model failed twice to return schema-valid output.
    """
    try:
        return classify_via_llm(prompt, examples, ticket)
    except LLMError as first_error:
        # Show the model exactly why it was rejected so the retry can correct
        # the specific field rather than guess.
        corrected_prompt = (
            f"{prompt}\n\nYour previous response was rejected with: "
            f"{first_error}\nReturn a response that exactly satisfies the "
            f"required schema."
        )
        try:
            retried = classify_via_llm(corrected_prompt, examples, ticket)
            # Mark it so Phase 8's eval can count schema-fallback re-prompts.
            retried.metadata.schema_retried = True
            return retried
        except LLMError as second_error:
            raise GuardrailError(
                f"Model failed to return schema-valid output after one "
                f"retry: {second_error}"
            ) from second_error
