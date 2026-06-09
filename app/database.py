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