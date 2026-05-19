"""ChromaDB-backed SOP retrieval and store lifecycle.

Owns the vector-store side of RAG: embedding text via OpenAI, building the
persisted ChromaDB collection of SOPs, and querying it. ``scripts/ingest_sops``
is now a thin CLI over :func:`sync_sop_collection`; the app calls the same
function on boot so a fresh deployment (Spaces ships no ``chroma_db/``)
self-populates. Importing this module still performs no network or disk I/O —
the Chroma client is created lazily inside the functions.
"""

import logging
from pathlib import Path
from typing import Final

import chromadb
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

from src.config import settings

# chromadb 0.5.x calls posthog with a signature its bundled version rejects,
# logging an error per telemetry event even with telemetry "disabled". We
# don't use telemetry; mute that logger so it never pollutes the console
# (Phase 7's acceptance requires a clean console). Process-wide: this module
# is imported by the app and the ingest script alike.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# One ChromaDB collection holds every SOP, one document per file (the SOPs are
# short enough that CLAUDE.md §7 specifies one chunk per SOP).
COLLECTION_NAME: Final[str] = "sop_documents"

# Cosine space so distances map cleanly to a 0..1-ish similarity score that
# the classifier can surface; set at collection-creation time.
DISTANCE_SPACE: Final[str] = "cosine"

_SOP_DIR: Final[Path] = Path("data/sop_documents")


class RAGError(RuntimeError):
    """Retrieval/ingestion failed: the store is missing/empty, no SOP files,
    or embedding failed (§4.4: no bare exceptions)."""


class SOPChunk(BaseModel):
    """One retrieved SOP. ``filename`` populates ``rag_sources`` in the §6
    output; ``score`` is cosine similarity (higher is closer)."""

    filename: str
    text: str
    score: float


def _chroma_client() -> chromadb.ClientAPI:
    # Telemetry off: chromadb 0.5.x's posthog call errors per event otherwise.
    return chromadb.PersistentClient(
        path=str(settings.chroma_path),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the configured OpenAI embedding model.

    Shared by ingestion and the query path so both use the exact same model;
    a mismatched embedding model would silently wreck retrieval.

    Raises:
        RAGError: The embedding API call failed.
    """
    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
    try:
        response = client.embeddings.create(model=settings.embedding_model, input=texts)
    except OpenAIError as exc:
        raise RAGError(f"Embedding call failed: {exc}") from exc
    return [item.embedding for item in response.data]


def sync_sop_collection(recreate: bool = False) -> int:
    """Ensure the SOP collection exists, embedding the markdown files if not.

    Idempotent: with ``recreate=False`` this is a no-op when the collection is
    already populated, so it is safe to call on every app start (a fresh
    Hugging Face Space ships no ``chroma_db/`` and self-populates here). With
    ``recreate=True`` it always rebuilds — used by ``scripts/ingest_sops.py``.

    Returns:
        Number of SOPs ingested (0 when it was a no-op).

    Raises:
        RAGError: No SOP markdown files were found, or embedding failed.
    """
    client = _chroma_client()
    exists = COLLECTION_NAME in {c.name for c in client.list_collections()}
    if exists and not recreate and client.get_collection(COLLECTION_NAME).count():
        return 0

    files = sorted(_SOP_DIR.glob("*.md"))
    if not files:
        raise RAGError(f"No SOP markdown files found in {_SOP_DIR}.")
    names = [path.name for path in files]
    texts = [path.read_text(encoding="utf-8") for path in files]
    embeddings = embed_texts(texts)  # one batched embedding call

    if exists:
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
    return len(names)


def retrieve_sops(query: str, k: int = 5) -> list[SOPChunk]:
    """Retrieve the top-``k`` SOPs most relevant to ``query``.

    Args:
        query: Free text — typically the ticket body.
        k: Number of SOPs to return. Default 5 because SOPs are short and the
            suggested-response step favors recall over precision (§4.1).

    Returns:
        Up to ``k`` ``SOPChunk``s, most similar first.

    Raises:
        RAGError: The collection is missing/empty (ingest not run) or the
            embedding/query call failed.
    """
    client = _chroma_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        # chromadb raises a typed error when the collection is absent;
        # translate it into our contract with an actionable message.
        raise RAGError(
            f"SOP collection {COLLECTION_NAME!r} not found at "
            f"{settings.chroma_path}. Run scripts/ingest_sops.py first."
        ) from exc

    if collection.count() == 0:
        raise RAGError("SOP collection is empty. Run scripts/ingest_sops.py.")

    query_embedding = embed_texts([query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        SOPChunk(
            filename=str(metadata["filename"]),
            text=document,
            # Cosine distance -> similarity; higher means closer.
            score=1.0 - float(distance),
        )
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]
