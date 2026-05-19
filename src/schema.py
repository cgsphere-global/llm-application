"""Pydantic data contracts that cross module boundaries.

Defines the only two shapes that flow between the UI, classifier, and
evaluator: ``TicketInput`` (a raw inbound ticket) and ``ClassificationOutput``
(the validated result — the §6 contract in CLAUDE.md, reproduced verbatim).
This module is pure: no I/O and no model calls. Because changing
``ClassificationOutput`` ripples into prompts, the evaluator, and the UI, treat
it as a versioned contract and edit it deliberately.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CategoryLiteral = Literal[
    "Billing", "Technical", "Account", "Refund", "Shipping", "Other"
]
PriorityLiteral = Literal["Urgent", "High", "Medium", "Low"]
SentimentLiteral = Literal["Angry", "Frustrated", "Neutral", "Positive"]

# Phase 10 hardening: a hard input cap so a pasted megabyte can't run up an
# unbounded prompt bill. Single source of truth — the classifier reuses it for
# a clear pre-validation message.
MAX_TICKET_CHARS = 4000


class TicketInput(BaseModel):
    """A raw customer-support ticket before classification.

    Attributes:
        subject: Optional short subject line. Many channels — and the Gradio
            single-textbox UI — carry no subject, so it defaults to empty.
        body: The ticket text. Must be non-empty, but near-empty tickets like
            "." are intentionally accepted: handling them well is an edge case
            the classifier must cover, not a reason to reject input here.
    """

    subject: str = Field(default="", max_length=200)
    body: str = Field(min_length=1, max_length=MAX_TICKET_CHARS)

    @property
    def combined_text(self) -> str:
        """Subject and body joined by a newline — the single string the prompt
        layer consumes (the caller contract described in CLAUDE.md §4.1)."""
        if not self.subject:
            return self.body
        return f"{self.subject}\n{self.body}".strip()


class ClassificationOutput(BaseModel):
    """The validated classification result — the CLAUDE.md §6 contract.

    A ``confidence`` below ~0.5 means the model was uncertain and the caller
    should consider human review; the Phase 6 guardrail enforces a hard floor.
    """

    # extra="forbid" becomes additionalProperties:false in the generated JSON
    # schema, which OpenAI's strict structured-output mode requires, and it
    # also rejects any model-invented field that would silently drift from
    # this contract instead of letting it pass through unnoticed.
    model_config = ConfigDict(extra="forbid")

    category: CategoryLiteral
    subcategory: str = Field(min_length=1, max_length=80)
    priority: PriorityLiteral
    sentiment: SentimentLiteral
    entities: dict[str, str] = Field(default_factory=dict)
    summary: str = Field(min_length=10, max_length=200)
    suggested_response: str = Field(min_length=20, max_length=1000)
    confidence: float = Field(ge=0.0, le=1.0)
    rag_sources: list[str] = Field(default_factory=list)
