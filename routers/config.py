"""Configuration settings for DocBrain."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str = "YOUR_ANTHROPIC_API_KEY"
    claude_model: str = "claude-opus-4-5"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_persist_dir: str = "./data/chroma"

    # RAG tuning
    chunk_size: int = 800          # tokens per chunk
    chunk_overlap: int = 150       # overlap between chunks
    top_k_results: int = 5         # chunks retrieved per query
    max_context_tokens: int = 4000 # max tokens sent to Claude

    # Upload limits
    max_file_size_mb: int = 20
    allowed_extensions: list[str] = [".pdf", ".txt", ".md", ".csv"]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()