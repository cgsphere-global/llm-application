"""Embed the SOP markdown files into the persisted ChromaDB collection.

Reads every ``data/sop_documents/*.md`` file, embeds each whole file as one
chunk (CLAUDE.md §7: SOPs are short, one chunk per SOP), and writes them to
the ChromaDB collection at ``CHROMA_PATH``. Run once after editing the SOPs.
Re-running is idempotent: the collection is recreated so stale vectors never
linger.
"""

import sys
from pathlib import Path

import chromadb

# Run as `python scripts/ingest_sops.py` (CLAUDE.md §10) puts scripts/ on
# sys.path, not the repo root. Add the repo root; the `src` imports stay
# function-local so this file has no out-of-order module imports (no E402).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SOP_DIR = Path("data/sop_documents")


class DataError(RuntimeError):
    """SOP ingestion failed unrecoverably (§4.4: no bare exceptions)."""


def main() -> None:
    # Function-local so the sys.path bootstrap above runs first.
    from src.config import settings
    from src.rag import COLLECTION_NAME, DISTANCE_SPACE, RAGError, embed_texts

    files = sorted(_SOP_DIR.glob("*.md"))
    if not files:
        raise DataError(f"No SOP markdown files found in {_SOP_DIR}.")

    names = [path.name for path in files]
    texts = [path.read_text(encoding="utf-8") for path in files]

    try:
        embeddings = embed_texts(texts)  # one batched embedding call
    except RAGError as exc:
        raise DataError(str(exc)) from exc

    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    # Telemetry off: its posthog version mismatch spams the console.
    client = chromadb.PersistentClient(
        path=str(settings.chroma_path),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )

    # Check-then-delete (not except-pass) so re-runs replace cleanly while
    # staying within §4.4's "catch narrowly" rule.
    if COLLECTION_NAME in {c.name for c in client.list_collections()}:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": DISTANCE_SPACE}
    )

    collection.add(
        ids=names,
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"filename": name} for name in names],
    )

    print(
        f"Ingested {len(names)} SOPs into '{COLLECTION_NAME}' at "
        f"{settings.chroma_path}"
    )
    print("Files:", ", ".join(names))


if __name__ == "__main__":
    try:
        main()
    except DataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
