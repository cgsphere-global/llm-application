"""Single source of configuration truth.

Loads environment variables (via a ``.env`` file in development, or real
process env on Hugging Face Spaces) into a validated, immutable ``Settings``
object. Importing this module constructs the ``settings`` singleton and fails
loudly if anything required is missing — there is no silent default for the
API key. This module does not perform any network or model I/O; it only reads
and validates configuration.
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, field_validator


class ConfigError(RuntimeError):
    """Configuration is missing or invalid. Raised at import time so the app
    cannot start in a half-configured state (§4.4: no bare exceptions)."""


class Settings(BaseModel):
    """Validated application configuration.

    Attributes:
        openai_api_key: OpenAI credential. Stored as ``SecretStr`` so it is
            masked in ``repr``/logs — the acceptance check prints ``settings``
            and we never want the raw key on a console or in structured logs.
        openai_model: Default chat model for classification.
        embedding_model: Model used for SOP embedding and query embedding.
        chroma_path: Filesystem location of the persisted ChromaDB store.
        log_level: Verbosity for structured logging.
    """

    # frozen: settings are loaded once at startup and never mutated (§4.2).
    # extra="forbid": we build the input dict ourselves, so an unexpected key
    # means a wiring bug, not user input — fail rather than ignore it.
    model_config = ConfigDict(frozen=True, extra="forbid")

    openai_api_key: SecretStr
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chroma_path: Path = Path("./data/chroma_db")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("openai_api_key")
    @classmethod
    def _reject_blank_key(cls, value: SecretStr) -> SecretStr:
        # An empty string is a valid SecretStr, so `OPENAI_API_KEY=` in a
        # copied-but-unfilled .env would pass silently. Reject it explicitly
        # so "missing key" always fails loudly, not just an absent variable.
        if not value.get_secret_value().strip():
            raise ValueError(
                "OPENAI_API_KEY is empty. Copy .env.example to .env and set it."
            )
        return value


def _load_settings() -> Settings:
    """Read env (with .env overlay) and build the validated Settings singleton.

    Raises:
        ConfigError: A required variable is missing or a value failed
            validation. The underlying Pydantic error is chained for detail.
    """
    # Searches upward from the cwd for a .env; a no-op when none exists, which
    # is the expected case on Hugging Face Spaces where vars are set in the UI.
    load_dotenv()

    # Only forward variables that are actually set. Passing None for an unset
    # optional would override the model default with None and fail validation;
    # omitting it lets the field default apply.
    raw: dict[str, str] = {
        field: value
        for field, value in {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "openai_model": os.getenv("OPENAI_MODEL"),
            "embedding_model": os.getenv("EMBEDDING_MODEL"),
            "chroma_path": os.getenv("CHROMA_PATH"),
            "log_level": os.getenv("LOG_LEVEL"),
        }.items()
        if value is not None
    }

    try:
        return Settings(**raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid or missing configuration:\n{exc}") from exc


settings: Settings = _load_settings()


class ModelPrice(BaseModel):
    """USD list price per 1M tokens for one model."""

    model_config = ConfigDict(frozen=True)

    input_usd_per_1m: float
    output_usd_per_1m: float


# Public OpenAI list prices (USD per 1M tokens). Centralised here so cost is
# computed from one place and a price change is a single-line edit, not a hunt
# through the codebase. Embeddings have no output tokens, hence 0.0.
MODEL_PRICES: dict[str, ModelPrice] = {
    "gpt-4o-mini": ModelPrice(input_usd_per_1m=0.15, output_usd_per_1m=0.60),
    "gpt-4o": ModelPrice(input_usd_per_1m=2.50, output_usd_per_1m=10.00),
    "text-embedding-3-small": ModelPrice(input_usd_per_1m=0.02, output_usd_per_1m=0.0),
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Dollar cost of one API call, derived from ``MODEL_PRICES``.

    Raises:
        ConfigError: ``model`` has no price entry. We fail loudly rather than
            report $0 and silently hide real spend.
    """
    try:
        price = MODEL_PRICES[model]
    except KeyError as exc:
        raise ConfigError(f"No price table entry for model {model!r}.") from exc
    return (
        prompt_tokens / 1_000_000 * price.input_usd_per_1m
        + completion_tokens / 1_000_000 * price.output_usd_per_1m
    )
