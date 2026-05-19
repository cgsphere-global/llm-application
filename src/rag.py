"""ChromaDB-backed SOP retrieval.

Owns the vector store side of RAG: turning text into OpenAI embeddings and
querying the persisted ChromaDB collection of standard operating procedures.
Ingestion (reading the markdown files, writing the collection) is the job of
``scripts/ingest_sops.py``; this module only embeds and retrieves. It holds
no module-level client so importing it performs no network or disk I/O.
"""

import logging
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
# the classifier can surface; set at creation time by the ingest script.
DISTANCE_SPACE: Final[str] = "cosine"


class RAGError(RuntimeError):
    """Retrieval failed: the store is missing/empty or embedding failed
    (§4.4: no bare exceptions)."""


class SOPChunk(BaseModel):
    """One retrieved SOP. ``filename`` populates ``rag_sources`` in the §6
    output; ``score`` is cosine similarity (higher is closer)."""

    filename: str
    text: str
    score: float


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the configured OpenAI embedding model.

    Shared by the ingest script and query path so both use the exact same
    model; mismatched embedding models would silently wreck retrieval.

    Raises:
        RAGError: The embedding API call failed.
    """
    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
    try:
        response = client.embeddings.create(model=settings.embedding_model, input=texts)
    except OpenAIError as exc:
        raise RAGError(f"Embedding call failed: {exc}") from exc
    return [item.embedding for item in response.data]


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
    # Disable ChromaDB's posthog telemetry: its version mismatch spams the
    # console and would trip Phase 7's "no warnings" acceptance.
    client = chromadb.PersistentClient(
        path=str(settings.chroma_path),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        # chromadb raises a bare/typed error when the collection is absent;
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
