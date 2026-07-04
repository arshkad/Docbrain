"""
Vector store (ChromaDB) setup and embedding utilities.
Uses ChromaDB's built-in sentence-transformers embeddings (no OpenAI needed).
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from app.config import settings


# Persistent local ChromaDB — data survives restarts
chroma_client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=ChromaSettings(anonymized_telemetry=False),
)

# Use sentence-transformers (free, runs locally)
# For production swap with OpenAI or Cohere embeddings
embedding_fn = embedding_functions.DefaultEmbeddingFunction()


def get_or_create_collection(collection_name: str) -> chromadb.Collection:
    """Get existing collection or create a new one."""
    return chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    
def list_collections() -> list[dict]:
    """List all collections with their document counts."""
    collections = chroma_client.list_collections()
    result = []
    for col in collections:
        c = chroma_client.get_collection(col.name, embedding_function=embedding_fn)
        result.append({
            "name": col.name,
            "document_chunks": c.count(),
            "metadata": col.metadata or {},
        })
    return result


def delete_collection(collection_name: str) -> bool:
    """Delete a collection and all its documents."""
    try:
        chroma_client.delete_collection(collection_name)
        return True
    except Exception:
        return False


def semantic_search(
    collection_name: str,
    query: str,
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """
    Run semantic similarity search against a collection.
    Returns ranked chunks with metadata and distances.
    """
    try:
        collection = chroma_client.get_collection(
            collection_name, embedding_function=embedding_fn
        )
    except Exception:
        raise ValueError(f"Collection '{collection_name}' not found")

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "score": round(1 - results["distances"][0][i], 4),  # cosine → similarity
        })

    return sorted(chunks, key=lambda x: x["score"], reverse=True)
