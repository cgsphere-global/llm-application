"""OpenAI client wrapper: structured-output classification.

Exposes one public function, ``classify_via_llm``. This module owns all
OpenAI I/O for classification and nothing else — retrieval, prompt assembly,
and re-prompting guardrails live in their own modules. It retries only
transient transport failures; it never re-prompts on bad content (that is the
Phase 6 guardrail's job).

OpenAI strict structured outputs cannot encode a free-form ``dict[str, str]``,
which is what §6 specifies for ``entities``. So the wire shape sent to / parsed
from the model uses a list of ``{name, value}`` pairs (a private model that
never leaves this file); it is mapped to the §6 ``ClassificationOutput``
contract on the way out and back for few-shot demonstrations on the way in.
"""

import time
from typing import Final

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, ValidationError

from src.config import cost_usd, settings
from src.prompts import FewShotExample
from src.schema import (
    CategoryLiteral,
    ClassificationOutput,
    PriorityLiteral,
    SentimentLiteral,
)


class LLMError(RuntimeError):
    """An OpenAI call failed unrecoverably (after retries / on a non-retryable
    error) or returned output that could not be coerced into the schema
    (§4.4: never raise bare exceptions)."""


class _WireEntity(BaseModel):
    """One extracted fact as a name/value pair. The strict-structured-output
    encoding of a single ``entities`` map entry."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: str


class _WireOutput(BaseModel):
    """The shape actually exchanged with the model.

    Mirrors ``ClassificationOutput`` except ``entities`` is a list of pairs
    (strict structured outputs can't express an open-ended object) and no
    field carries length/range constraints — strict mode rejects several such
    keywords, and the §6 bounds are re-imposed when we build the real
    contract in :func:`_from_wire`. Every field is required and extra keys are
    forbidden, as strict mode demands.
    """

    model_config = ConfigDict(extra="forbid")

    category: CategoryLiteral
    subcategory: str
    priority: PriorityLiteral
    sentiment: SentimentLiteral
    entities: list[_WireEntity]
    summary: str
    suggested_response: str
    confidence: float
    rag_sources: list[str]


class LLMCallMetadata(BaseModel):
    """Cost and latency telemetry for a single classification call.

    ``model`` is recorded because cost is model-dependent and the Phase 6
    observability layer logs which model actually served the request.
    """

    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    # Failure-rate telemetry for Phase 8's eval report. ``retries`` is the
    # number of transient-error backoff retries before this call succeeded;
    # ``schema_retried`` is set by the guardrail when it had to re-prompt
    # once because the first response failed schema validation.
    retries: int = 0
    schema_retried: bool = False


class LLMResult(BaseModel):
    """The parsed classification plus the telemetry of the call that produced
    it. A model rather than a tuple so call sites read self-documentingly and
    observability can serialize it directly."""

    output: ClassificationOutput
    metadata: LLMCallMetadata


# 3 retries on transient transport failures only, with the 1s/2s/4s schedule
# mandated by CLAUDE.md Phase 3. 400s and auth errors are bad input or
# misconfig — they never succeed on retry, so we fail fast on them instead.
_BACKOFF_SECONDS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0)
_TRANSIENT_ERRORS: Final[tuple[type[OpenAIError], ...]] = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

# max_retries=0: the SDK's own retry layer would compound with ours and make
# the observed backoff non-deterministic. We want exactly the spec schedule.
_client = OpenAI(api_key=settings.openai_api_key.get_secret_value(), max_retries=0)


def _to_wire(output: ClassificationOutput) -> _WireOutput:
    """Render a §6 output in the model-facing shape, used for few-shot
    assistant turns so demonstrations match the response schema exactly."""
    return _WireOutput(
        category=output.category,
        subcategory=output.subcategory,
        priority=output.priority,
        sentiment=output.sentiment,
        entities=[
            _WireEntity(name=key, value=value) for key, value in output.entities.items()
        ],
        summary=output.summary,
        suggested_response=output.suggested_response,
        confidence=output.confidence,
        rag_sources=output.rag_sources,
    )


def _from_wire(wire: _WireOutput) -> ClassificationOutput:
    """Map the model-facing shape back to the §6 contract, re-applying every
    §6 length/range bound here (those are not enforced on the wire model).

    Raises:
        ValidationError: The model's content violates a §6 bound.
    """
    # The wire model carries no length bounds, so the model can emit an empty
    # subcategory on terse tickets. Rather than fail a customer request on a
    # blank descriptive label, fall back to the category name (still §6-valid).
    subcategory = wire.subcategory.strip() or wire.category
    return ClassificationOutput(
        category=wire.category,
        subcategory=subcategory,
        priority=wire.priority,
        sentiment=wire.sentiment,
        # On a duplicate entity name, last value wins — acceptable because the
        # contract is a map and a repeated key has no meaning there anyway.
        entities={entity.name: entity.value for entity in wire.entities},
        summary=wire.summary,
        suggested_response=wire.suggested_response,
        confidence=wire.confidence,
        rag_sources=wire.rag_sources,
    )


def _build_messages(
    system_prompt: str, examples: list[FewShotExample], ticket: str
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for example in examples:
        messages.append({"role": "user", "content": example.ticket})
        # Assistant turns use the wire shape so the demonstrations agree with
        # the response schema the model is bound to; a contract-shaped (dict
        # entities) example would contradict it.
        messages.append(
            {
                "role": "assistant",
                "content": _to_wire(example.output).model_dump_json(),
            }
        )
    messages.append({"role": "user", "content": ticket})
    return messages


def classify_via_llm(
    prompt: str, examples: list[FewShotExample], ticket: str
) -> LLMResult:
    """Classify one ticket via OpenAI structured outputs.

    Args:
        prompt: The system instruction block. The classifier passes
            ``SYSTEM_PROMPT`` here, with retrieved SOPs already appended when
            RAG is enabled.
        examples: Few-shot demonstrations, rendered as user/assistant turns.
        ticket: The raw ticket text to classify.

    Returns:
        An ``LLMResult`` carrying the validated ``ClassificationOutput`` and
        the call's latency/token/cost metadata.

    Raises:
        LLMError: The API failed after all retries, refused the request, or
            returned output that could not be coerced into the schema.
    """
    messages = _build_messages(prompt, examples, ticket)
    start = time.perf_counter()

    attempt = 0
    while True:
        try:
            completion = _client.beta.chat.completions.parse(
                model=settings.openai_model,
                messages=messages,
                response_format=_WireOutput,
                # Deterministic: consistent classification and reproducible
                # eval matter more than response variety here.
                temperature=0,
            )
            break
        except _TRANSIENT_ERRORS as exc:
            if attempt >= len(_BACKOFF_SECONDS):
                raise LLMError(
                    f"OpenAI call failed after {attempt} retries: {exc}"
                ) from exc
            time.sleep(_BACKOFF_SECONDS[attempt])
            attempt += 1
        except BadRequestError as exc:
            raise LLMError(f"OpenAI rejected the request (not retried): {exc}") from exc
        except OpenAIError as exc:
            raise LLMError(f"OpenAI call failed (not retried): {exc}") from exc

    # Latency intentionally spans backoff sleeps: it is the user-perceived
    # cost we report in eval p95, not just the successful attempt.
    latency_ms = (time.perf_counter() - start) * 1000.0
    message = completion.choices[0].message

    if message.refusal:
        raise LLMError(f"Model refused to classify the ticket: {message.refusal}")

    wire = message.parsed
    if wire is None:
        # OpenAI returns parsed=None when server-side schema binding fails; the
        # raw JSON is still in content, so re-validate locally before giving up.
        if not message.content:
            raise LLMError("OpenAI returned neither parsed output nor content.")
        try:
            wire = _WireOutput.model_validate_json(message.content)
        except ValidationError as exc:
            raise LLMError(
                f"Model output failed wire-schema validation: {exc}"
            ) from exc

    try:
        output = _from_wire(wire)
    except ValidationError as exc:
        raise LLMError(f"Model output violated the §6 contract: {exc}") from exc

    usage = completion.usage
    if usage is None:
        raise LLMError("OpenAI response carried no token usage; cannot price it.")

    metadata = LLMCallMetadata(
        model=settings.openai_model,
        latency_ms=latency_ms,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cost_usd=cost_usd(
            settings.openai_model,
            usage.prompt_tokens,
            usage.completion_tokens,
        ),
        retries=attempt,
    )
    return LLMResult(output=output, metadata=metadata)
