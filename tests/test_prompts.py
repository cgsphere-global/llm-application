"""Prompt invariants — the structural "snapshot" CLAUDE.md §7 (Phase 2) asks
for. Asserted as invariants, not a brittle full-text match, so legitimate
wording edits don't fail the suite while drift from the schema still does.
"""

from typing import get_args

from src.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, FewShotExample
from src.schema import (
    CategoryLiteral,
    ClassificationOutput,
    PriorityLiteral,
    SentimentLiteral,
)


def test_prompt_lists_every_schema_enum_value():
    # Anti-drift: the allowed values are rendered from the schema Literals, so
    # every one must appear verbatim in the prompt.
    for value in (
        *get_args(CategoryLiteral),
        *get_args(PriorityLiteral),
        *get_args(SentimentLiteral),
    ):
        assert value in SYSTEM_PROMPT


def test_prompt_defines_every_category():
    for category in get_args(CategoryLiteral):
        assert f"{category}:" in SYSTEM_PROMPT


def test_prompt_keeps_injection_guard_and_subcategory_rule():
    assert "ignore previous instructions" in SYSTEM_PROMPT.lower()
    assert "subcategory" in SYSTEM_PROMPT
    # The defect-fix rule that subcategory must never be empty.
    assert "non-empty" in SYSTEM_PROMPT


def test_exactly_five_distinct_valid_few_shots():
    assert len(FEW_SHOT_EXAMPLES) == 5
    for example in FEW_SHOT_EXAMPLES:
        assert isinstance(example, FewShotExample)
        assert example.ticket.strip()
        assert isinstance(example.output, ClassificationOutput)
    categories = {e.output.category for e in FEW_SHOT_EXAMPLES}
    assert len(categories) == 5  # five distinct categories demonstrated
