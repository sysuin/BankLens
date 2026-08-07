"""
Configuration management for BankLens.

Uses Pydantic Settings to load and validate all environment variables
from the .env file. A single 'settings' instance is imported and shared
across the entire application — no config values are hardcoded anywhere.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded and validated from environment variables.

    All fields map directly to keys in the .env file. Fields with a
    default value are optional; fields without one are required and
    will raise a ValidationError if missing.
    """

    # ── OpenAI ────────────────────────────────────────────────────────────────
    # Required: your OpenAI API key — set via OPENAI_API_KEY in .env
    openai_api_key: str

    # LLM model used for profile generation (GPT-4o gives the best JSON output)
    openai_model: str = "gpt-4o"

    # Embedding model — text-embedding-3-small is fast and cost-effective
    openai_embedding_model: str = "text-embedding-3-small"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    # Local directory where ChromaDB persists its index between runs
    chroma_persist_dir: str = "./chroma_db"

    # Name of the ChromaDB collection that holds the product embeddings
    chroma_collection_name: str = "banking_products"

    # ── RAG Tuning ────────────────────────────────────────────────────────────
    # Number of characters per chunk when splitting knowledge base documents
    chunk_size: int = 500

    # Number of overlapping characters between adjacent chunks.
    # Overlap ensures context is not lost at chunk boundaries.
    chunk_overlap: int = 50

    # Number of product chunks to retrieve for each customer query
    retrieval_k: int = 4

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore any extra keys in .env that are not defined above
        extra="ignore",
    )


# ── Singleton ─────────────────────────────────────────────────────────────────
# Import this instance in every module that needs config values.
# Example: from app.core.config import settings
settings = Settings()
