"""Embed the SOP markdown files into the persisted ChromaDB collection.

Thin CLI over ``src.rag.sync_sop_collection(recreate=True)`` — the embedding
and collection logic lives in one place so the app's boot-time ingest and this
script can never drift. Run once after editing the SOPs; re-running rebuilds
the collection so stale vectors never linger.
"""

import sys
from pathlib import Path

# Run as `python scripts/ingest_sops.py` (CLAUDE.md §10) puts scripts/ on
# sys.path, not the repo root. Add the repo root; the `src` import stays
# function-local so this file has no out-of-order module imports (no E402).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DataError(RuntimeError):
    """SOP ingestion failed unrecoverably (§4.4: no bare exceptions)."""


def main() -> None:
    # Function-local so the sys.path bootstrap above runs first.
    from src.config import settings
    from src.rag import COLLECTION_NAME, RAGError, sync_sop_collection

    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    try:
        count = sync_sop_collection(recreate=True)
    except RAGError as exc:
        raise DataError(str(exc)) from exc

    print(
        f"Ingested {count} SOPs into '{COLLECTION_NAME}' at " f"{settings.chroma_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except DataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
