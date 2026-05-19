"""Structured, one-line-per-ticket logging.

Emits a single JSON object per classification to stdout (so Hugging Face
Spaces captures it) carrying timestamp, latency, tokens, cost, category, and
confidence. The ticket preview is PII-redacted here, at the logging boundary,
because that is the last place text is written out. This module only logs; it
makes no model or store calls.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Final

from src.config import settings
from src.guardrails import redact_pii
from src.llm import LLMCallMetadata
from src.schema import ClassificationOutput

_LOGGER_NAME: Final[str] = "ticket_classifier"


def _build_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(settings.log_level)
    # Re-importing this module must not stack duplicate handlers, which would
    # print every line N times.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # We emit pre-serialized JSON, so the formatter must not wrap it.
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    # Don't also bubble to the root logger's default handler (double print).
    logger.propagate = False
    return logger


_logger: Final[logging.Logger] = _build_logger()


def log_classification(
    ticket_text: str,
    output: ClassificationOutput,
    metadata: LLMCallMetadata,
    needs_human_review: bool,
) -> None:
    """Write one structured JSON log line for a completed classification."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": metadata.model,
        "latency_ms": round(metadata.latency_ms, 1),
        "prompt_tokens": metadata.prompt_tokens,
        "completion_tokens": metadata.completion_tokens,
        "cost_usd": round(metadata.cost_usd, 6),
        "category": output.category,
        "confidence": output.confidence,
        "needs_human_review": needs_human_review,
        "rag_sources": output.rag_sources,
        # Preview only, redacted: logs should never carry a full PII-laden body.
        "ticket_preview": redact_pii(ticket_text)[:200],
    }
    _logger.info(json.dumps(payload))
