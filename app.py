"""Gradio entry point.

Thin presentation layer: it calls classify_ticket_detailed and formats the
result. No classification logic lives here (CLAUDE.md §4.2); the only logic is
the §10 per-session rate limit, which §10 explicitly mandates live in app.py.
"""

import gradio as gr

from src.classifier import ClassifiedResult, ClassifierError, classify_ticket_detailed
from src.guardrails import GuardrailError

_EXAMPLES = [
    "I was charged twice for my Pro plan this month, order #99812. Refund the extra $20.",
    "My order #ZX-5567 was due five days ago and tracking hasn't moved since the warehouse.",
    "I can't log into my account - it says locked after attempts I never made.",
]

# §10 hardening: cap classifications per browser session to bound abuse/cost.
_SESSION_LIMIT = 30


def _card(result: ClassifiedResult) -> str:
    o = result.output
    review = (
        "> ⚠️ **Low confidence — flagged for human review**\n\n"
        if result.needs_human_review
        else ""
    )
    entities = "\n".join(f"- **{k}**: {v}" for k, v in o.entities.items()) or "_none_"
    return (
        f"{review}### {o.category} · {o.subcategory}\n"
        f"**Priority:** {o.priority} | **Sentiment:** {o.sentiment} | "
        f"**Confidence:** {o.confidence:.2f}\n\n"
        f"**Summary:** {o.summary}\n\n"
        f"**Suggested response:**\n\n{o.suggested_response}\n\n"
        f"**Entities:**\n{entities}\n\n"
        f"**SOPs used:** {', '.join(o.rag_sources) or '_none_'}"
    )


def _metrics(result: ClassifiedResult) -> str:
    m = result.metadata
    return (
        "### Last request\n"
        f"- **Model:** {m.model}\n"
        f"- **Latency:** {m.latency_ms:.0f} ms\n"
        f"- **Prompt tokens:** {m.prompt_tokens}\n"
        f"- **Completion tokens:** {m.completion_tokens}\n"
        f"- **Cost:** ${m.cost_usd:.6f}"
    )


def _classify(ticket: str, used: int) -> tuple[str, dict, str, int]:
    if used >= _SESSION_LIMIT:
        return (
            f"🚫 Session limit reached ({_SESSION_LIMIT} classifications). "
            "Refresh the page to start a new session.",
            {},
            "",
            used,
        )
    try:
        result = classify_ticket_detailed(ticket)
    except ClassifierError as exc:
        return f"⚠️ {exc}", {}, "", used + 1
    except GuardrailError as exc:
        return f"❌ Could not produce a valid result: {exc}", {}, "", used + 1
    return _card(result), result.output.model_dump(), _metrics(result), used + 1


with gr.Blocks(title="Ticket Classifier") as demo:
    gr.Markdown("# Customer Ticket Classifier")
    with gr.Row():
        with gr.Column(scale=2):
            inp = gr.Textbox(label="Customer ticket", lines=8)
            gr.Examples(_EXAMPLES, inputs=inp)
            btn = gr.Button("Classify", variant="primary")
        with gr.Column(scale=3):
            card = gr.Markdown()
            raw = gr.JSON(label="Raw output")
        with gr.Column(scale=1):
            metrics = gr.Markdown()
    # gr.State is per-session, so the counter naturally resets on refresh.
    used = gr.State(0)
    btn.click(_classify, inputs=[inp, used], outputs=[card, raw, metrics, used])

if __name__ == "__main__":
    # Bind localhost for local runs; Hugging Face Spaces overrides the host
    # via the GRADIO_SERVER_NAME env var it sets automatically.
    demo.launch(server_port=7860)
