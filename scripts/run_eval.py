"""Evaluate the classifier over the eval set and write a report.

Runs ``classify_ticket`` with RAG **off** (CLAUDE.md §7 step 1: category-only
fairness) over every row of ``data/eval/eval_set.csv`` — or a fixed stratified
subset for fast prompt-iteration rounds — under bounded concurrency, then
computes accuracy, per-category P/R/F1, latency (mean/p95), cost, and the
schema-fallback/retry failure rate. Results are written to
``reports/eval_<timestamp>[_label].{json,md}``.

The subset is seeded so every iteration round scores the *same* tickets,
making before/after prompt comparisons fair.
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_EVAL_SET = Path("data/eval/eval_set.csv")
_REPORT_DIR = Path("reports")
_CATEGORIES = ["Billing", "Technical", "Account", "Refund", "Shipping", "Other"]
_SEED = 42

# Acceptance targets (CLAUDE.md §7 Phase 8 step 3).
_TARGET_ACCURACY = 0.80
_TARGET_P95_MS = 5000.0
_TARGET_COST_PER_TICKET = 0.005


class DataError(RuntimeError):
    """Eval could not run (missing/!malformed eval set) — §4.4: no bare
    exceptions."""


def _stratified(frame: pd.DataFrame, size: int) -> pd.DataFrame:
    """Fixed proportional subsample by expected_category (seeded, so rounds
    score identical rows)."""
    shares = frame["expected_category"].value_counts(normalize=True)
    alloc = (shares * size).round().astype(int)
    drift = size - int(alloc.sum())
    for group in shares.index[: abs(drift)]:
        alloc[group] += 1 if drift > 0 else -1
    parts = [
        g.sample(n=int(alloc[name]), random_state=_SEED)
        for name, g in frame.groupby("expected_category")
        if alloc[name] > 0
    ]
    return pd.concat(parts).reset_index(drop=True)


def _classify_row(text: str) -> dict:
    """Classify one row, capturing metrics. A single ticket's failure must
    not abort a 1000-row batch, so unexpected errors are recorded, not raised."""
    from src.classifier import ClassifierError, classify_ticket_detailed
    from src.guardrails import GuardrailError

    try:
        result = classify_ticket_detailed(text, use_rag=False)
    except (ClassifierError, GuardrailError) as exc:
        return {"failed": True, "error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # batch resilience; the error is reported, not hidden
        return {"failed": True, "error": f"{type(exc).__name__}: {exc}"}
    meta = result.metadata
    return {
        "failed": False,
        "predicted": result.output.category,
        "latency_ms": meta.latency_ms,
        "cost_usd": meta.cost_usd,
        "prompt_tokens": meta.prompt_tokens,
        "completion_tokens": meta.completion_tokens,
        "retries": meta.retries,
        "schema_retried": meta.schema_retried,
    }


def _per_category(expected: list[str], predicted: list[str | None]) -> dict:
    table = {}
    for category in _CATEGORIES:
        tp = sum(p == category and e == category for e, p in zip(expected, predicted))
        fp = sum(p == category and e != category for e, p in zip(expected, predicted))
        fn = sum(e == category and p != category for e, p in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        table[category] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(e == category for e in expected),
        }
    return table


def _metrics(rows: list[dict], records: list[dict], elapsed_s: float) -> dict:
    total = len(rows)
    failures = [r for r in records if r["failed"]]
    ok = [r for r in records if not r["failed"]]
    expected = [row["expected_category"] for row in rows]
    predicted = [rec.get("predicted") if not rec["failed"] else None for rec in records]
    correct = sum(e == p for e, p in zip(expected, predicted))
    latencies = [r["latency_ms"] for r in ok]
    return {
        "total": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "per_category": _per_category(expected, predicted),
        "macro_f1": round(
            float(
                np.mean([v["f1"] for v in _per_category(expected, predicted).values()])
            ),
            4,
        ),
        "latency_ms_mean": round(float(np.mean(latencies)), 1) if latencies else 0.0,
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 1)
        if latencies
        else 0.0,
        "cost_usd_total": round(sum(r["cost_usd"] for r in ok), 6),
        "cost_usd_per_ticket": round(sum(r["cost_usd"] for r in ok) / total, 6)
        if total
        else 0.0,
        "failure_rate": round(len(failures) / total, 4) if total else 0.0,
        "rows_with_transient_retry": sum(r.get("retries", 0) > 0 for r in ok),
        "rows_with_schema_refrompt": sum(r.get("schema_retried", False) for r in ok),
        "wall_clock_s": round(elapsed_s, 1),
    }


def _targets(metrics: dict) -> dict:
    checks = {
        "accuracy>=0.80": metrics["accuracy"] >= _TARGET_ACCURACY,
        "p95_latency<5s": metrics["latency_ms_p95"] < _TARGET_P95_MS,
        "cost/ticket<$0.005": metrics["cost_usd_per_ticket"] < _TARGET_COST_PER_TICKET,
    }
    return {"checks": checks, "all_passed": all(checks.values())}


def _write_reports(payload: dict, label: str) -> Path:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    base = _REPORT_DIR / f"eval_{stamp}{suffix}"

    base.with_suffix(".json").write_text(json.dumps(payload, indent=2))

    m, t = payload["metrics"], payload["targets"]
    lines = [
        f"# Eval report — {payload['label'] or 'run'} ({payload['timestamp']})",
        "",
        f"- Rows: **{m['total']}**  ·  Subset: {payload['subset'] or 'full'}",
        f"- Accuracy: **{m['accuracy']:.1%}**  ·  Macro-F1: {m['macro_f1']:.3f}",
        f"- Latency mean/p95: {m['latency_ms_mean']:.0f} / "
        f"**{m['latency_ms_p95']:.0f} ms**",
        f"- Cost total / per ticket: ${m['cost_usd_total']:.4f} / "
        f"**${m['cost_usd_per_ticket']:.6f}**",
        f"- Failure rate: {m['failure_rate']:.2%}  ·  schema re-prompts: "
        f"{m['rows_with_schema_refrompt']}  ·  transient retries: "
        f"{m['rows_with_transient_retry']}",
        "",
        "## Targets",
        "",
        "| Target | Result |",
        "|---|---|",
    ]
    lines += [
        f"| {name} | {'✅ PASS' if ok else '❌ FAIL'} |"
        for name, ok in t["checks"].items()
    ]
    lines += [
        "",
        "## Per-category",
        "",
        "| Category | P | R | F1 | n |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {c} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | "
        f"{v['support']} |"
        for c, v in m["per_category"].items()
    ]
    if payload.get("notes"):
        lines += ["", "## Methodology & known limitations", ""]
        lines += [f"- {note}" for note in payload["notes"]]
    base.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=0, help="stratified N rows")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--label", type=str, default="", help="report tag")
    parser.add_argument(
        "--note", action="append", default=[], help="methodology note (repeatable)"
    )
    args = parser.parse_args()

    if not _EVAL_SET.exists():
        raise DataError(f"{_EVAL_SET} missing. Run the Phase 4 scripts first.")
    frame = pd.read_csv(_EVAL_SET, keep_default_na=False)
    if args.subset:
        frame = _stratified(frame, args.subset)
    rows = frame.to_dict("records")

    # One JSON line per ticket would bury eval progress; quiet that logger.
    logging.getLogger("ticket_classifier").setLevel(logging.WARNING)

    print(f"Evaluating {len(rows)} rows (RAG off, {args.workers} workers)...")
    start = datetime.now(timezone.utc)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda r: _classify_row(r["text"]), rows))
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()

    metrics = _metrics(rows, records, elapsed)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "subset": args.subset or None,
        "notes": args.note,
        "metrics": metrics,
        "targets": _targets(metrics),
    }
    base = _write_reports(payload, args.label)

    print(json.dumps(metrics, indent=2))
    print("Targets:", payload["targets"])
    print(f"Report: {base}.json / .md")


if __name__ == "__main__":
    try:
        main()
    except DataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
