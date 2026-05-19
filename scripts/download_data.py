"""Download the real Bitext support dataset and build the Bitext eval slice.

Pulls ``bitext/Bitext-customer-support-llm-chatbot-training-dataset`` from
Hugging Face, remaps its native taxonomy onto the six §6 categories, takes a
1000-row sample stratified proportionally by the mapped category, and writes
``data/eval/eval_set.csv``. Run once. ``scripts/generate_edge_cases.py`` later
appends the 30 synthetic edge cases to this file.

This script only handles real Bitext data — per CLAUDE.md §9 we never
substitute synthetic Bitext-like rows here.
"""

import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

_HF_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
_OUTPUT = Path("data/eval/eval_set.csv")
_SAMPLE_SIZE = 1000

# Fixed so reviewers reproduce the exact sample (CLAUDE.md §5.1).
_SEED = 42

# User-approved Bitext -> §6 taxonomy remap, reconciled to the dataset's real
# 11 categories. CANCEL/SUBSCRIPTION -> Account per the user's decision. Refund
# *intents* override this map (see _map_to_taxonomy): Bitext files many refunds
# under non-REFUND categories, the remap CLAUDE.md §4.1 cites.
#
# 2026-05-19 revision (Phase 8): ORDER changed Other -> Shipping. The Phase-8
# diagnostic showed ORDER tickets ("where is my order", "track order", "ETA")
# are semantically identical to DELIVERY tickets (already Shipping) but had the
# opposite gold label, an irreducible conflict no prompt can resolve. This is a
# methodology correction, not metric-gaming.
_CATEGORY_MAP: dict[str, str] = {
    "ACCOUNT": "Account",
    "CANCEL": "Account",
    "CONTACT": "Other",
    "DELIVERY": "Shipping",
    "FEEDBACK": "Other",
    "INVOICE": "Billing",
    "ORDER": "Shipping",
    "PAYMENT": "Billing",
    "REFUND": "Refund",
    "SHIPPING": "Shipping",
    "SUBSCRIPTION": "Account",
}

# Final eval schema shared with the edge-case rows so Phase 8 iterates
# uniformly. ``expected_category`` is the only column that must never be null.
_COLUMNS = [
    "text",
    "expected_category",
    "source",
    "edge_type",
    "bitext_category",
    "bitext_intent",
]


class DataError(RuntimeError):
    """Dataset download or remapping failed in a way we must not paper over
    (§4.4: no bare exceptions, fail loudly on unmapped categories)."""


def _map_to_taxonomy(bitext_category: str, intent: str) -> str:
    # A refund intent wins regardless of category: Bitext tags many refund
    # requests under ORDER, so keying on category alone would mislabel them.
    if "refund" in intent.lower():
        return "Refund"
    try:
        return _CATEGORY_MAP[bitext_category.strip().upper()]
    except KeyError as exc:
        raise DataError(
            f"Bitext category {bitext_category!r} is not in the approved "
            f"mapping. Update _CATEGORY_MAP before re-running."
        ) from exc


def _stratified_sample(frame: pd.DataFrame, size: int, by: str) -> pd.DataFrame:
    """Sample ``size`` rows with per-group counts proportional to each group's
    share, summing to exactly ``size`` (largest groups absorb rounding)."""
    shares = frame[by].value_counts(normalize=True)
    alloc = (shares * size).round().astype(int)

    drift = size - int(alloc.sum())
    # Hand the rounding remainder to the largest groups so totals hit `size`
    # exactly without starving small categories.
    for group in shares.index[: abs(drift)]:
        alloc[group] += 1 if drift > 0 else -1

    parts = [
        group.sample(n=int(alloc[name]), random_state=_SEED)
        for name, group in frame.groupby(by)
        if alloc[name] > 0
    ]
    return pd.concat(parts).sample(frac=1, random_state=_SEED).reset_index(drop=True)


def _fetch_bitext() -> pd.DataFrame:
    """Download Bitext and return the columns CLAUDE.md §5.1 names."""
    dataset = load_dataset(_HF_DATASET, split="train")
    frame = dataset.to_pandas()
    required = {"instruction", "category", "intent"}
    missing = required - set(frame.columns)
    if missing:
        raise DataError(f"Bitext is missing expected columns: {sorted(missing)}")
    return frame


def main() -> None:
    frame = _fetch_bitext()
    print(f"Downloaded {len(frame):,} Bitext rows.")

    frame["expected_category"] = [
        _map_to_taxonomy(category, intent)
        for category, intent in zip(frame["category"], frame["intent"])
    ]

    sample = _stratified_sample(frame, _SAMPLE_SIZE, by="expected_category")
    sample = pd.DataFrame(
        {
            "text": sample["instruction"],
            "expected_category": sample["expected_category"],
            "source": "bitext",
            "edge_type": "",
            "bitext_category": sample["category"],
            "bitext_intent": sample["intent"],
        },
        columns=_COLUMNS,
    )

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(_OUTPUT, index=False)

    print(f"Wrote {len(sample)} rows -> {_OUTPUT}")
    print("Mapped-category distribution:")
    print(sample["expected_category"].value_counts().to_string())


if __name__ == "__main__":
    try:
        main()
    except DataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
