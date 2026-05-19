---
title: Customer Ticket Classifier
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
# HF Spaces now defaults to Python 3.13, where stdlib `audioop` was removed
# (PEP 594) and pydub->gradio fails to import. Pin 3.11 to match the tested
# local environment (CLAUDE.md §2).
python_version: "3.11"
pinned: false
---

# Customer Ticket Classifier

An LLM application that classifies a customer-support ticket and drafts a
grounded suggested reply. Given a raw ticket it returns a validated JSON
object with category, subcategory, priority, sentiment, extracted entities, a
summary, a suggested response grounded in retrieved SOPs, a confidence score,
and the SOP sources used.

It is the capstone for an *LLM Applications System Design* module and is built
to make each taught concept visible in its own module: prompting
(`src/prompts.py`), structured outputs (`src/schema.py`, `src/llm.py`), RAG
with ChromaDB (`src/rag.py`), guardrails (`src/guardrails.py`), evaluation
(`scripts/run_eval.py`), observability (`src/observability.py`), and token
economics (price table in `src/config.py`). `CLAUDE.md` is the authoritative
spec.

## Features

- OpenAI structured outputs validated against a Pydantic contract
- ChromaDB RAG over 13 SOP documents (`text-embedding-3-small`)
- Guardrails: schema-fallback re-prompt, confidence floor, PII redaction
- Retry with exponential backoff; per-call latency/token/cost accounting
- Structured one-line-per-ticket JSON logs
- Gradio UI with rendered card, raw JSON, and a live metrics sidebar
- Per-session rate limit (30) and a 4000-character input cap

## Local setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then set OPENAI_API_KEY in .env

python scripts/download_data.py        # Bitext sample -> data/eval/
python scripts/generate_edge_cases.py  # 30 edge cases, assembles eval_set.csv
python scripts/ingest_sops.py          # embed SOPs into ChromaDB

python app.py                          # http://localhost:7860
```

## Evaluation

```bash
python scripts/run_eval.py             # full eval -> reports/eval_*.{json,md}
python scripts/run_eval.py --subset 200 --label dev   # fast iteration
```

**Honest status (v1):** latency p95 and cost-per-ticket targets pass with wide
margin and the failure rate is 0%, but category accuracy is **~0.54**, below
the 0.80 target. This is a documented dataset/taxonomy ceiling (terse,
templated Bitext text vs. natural tickets), analysed in the
`reports/eval_*_final.md` "Methodology & known limitations" section — reported
honestly rather than tuned around.

## Quality gates

```bash
ruff check . && ruff format --check .
pytest                                 # LLM mocked; no network in CI
```

## Deployment

Deployed to Hugging Face Spaces (Gradio SDK) via GitHub Actions on push to
`main`. Required secrets/variables (see `CLAUDE.md` §8, Phases 11–14):

- HF Space: `OPENAI_API_KEY` (secret) plus the non-secret vars from
  `.env.example`
- GitHub repo: `HF_TOKEN` (write scope), `HF_USERNAME`, `HF_SPACE`

## Screenshots

_Placeholder — add UI and eval-report screenshots here before submission._

## Data & attribution

Primary data: the **Bitext Customer Support LLM Chatbot Training Dataset**
(`bitext/Bitext-customer-support-llm-chatbot-training-dataset`), licensed
**CDLA-Sharing-1.0**, which permits commercial use with attribution. We sample
1,000 rows stratified by a remapped taxonomy and add 30 synthetic edge cases.
SOP documents are generic and synthetic (no real company data).

## License

Application code: MIT (or as the course specifies). The Bitext dataset retains
its CDLA-Sharing-1.0 license and attribution requirement above.
