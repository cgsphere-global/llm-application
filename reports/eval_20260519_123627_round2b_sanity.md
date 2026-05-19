# Eval report — round2b_sanity (2026-05-19T12:36:27.052066+00:00)

- Rows: **200**  ·  Subset: 200
- Accuracy: **54.0%**  ·  Macro-F1: 0.572
- Latency mean/p95: 2605 / **3612 ms**
- Cost total / per ticket: $0.0682 / **$0.000341**
- Failure rate: 0.00%  ·  schema re-prompts: 0  ·  transient retries: 0

## Targets

| Target | Result |
|---|---|
| accuracy>=0.80 | ❌ FAIL |
| p95_latency<5s | ✅ PASS |
| cost/ticket<$0.005 | ✅ PASS |

## Per-category

| Category | P | R | F1 | n |
|---|---|---|---|---|
| Billing | 0.74 | 0.57 | 0.64 | 30 |
| Technical | 0.50 | 1.00 | 0.67 | 1 |
| Account | 0.90 | 0.47 | 0.61 | 58 |
| Refund | 1.00 | 0.30 | 0.47 | 23 |
| Shipping | 1.00 | 0.45 | 0.62 | 58 |
| Other | 0.27 | 1.00 | 0.42 | 30 |
