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
    openai_api_key: str = ""

    # LLM model used for profile generation (GPT-4o gives the best JSON output)
    openai_model: str = "gpt-4o"
    openai_mini_model: str = "gpt-4o-mini"

    # Embedding model — text-embedding-3-small is fast and cost-effective
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Google Gemini (reserved, not yet wired up) ─────────────────────────
    # Kept as a placeholder for adding a second provider. Nothing reads these
    # today, and langchain-google-genai is deliberately not a dependency, so
    # setting them has no effect until a Gemini code path exists.
    google_api_key: str = ""
    google_model: str = "gemini-1.5-flash"

    # ── LangSmith Tracing (Optional) ───────────────────────────────────────
    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "BankLens"

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

    # Number of product chunks passed to the LLM after reranking
    retrieval_k: int = 4

    # Number of candidates pulled from each retriever *before* reranking.
    # Retrieving wide and then reranking narrow is the whole point: the fused
    # dense+BM25 ordering is cheap but coarse, so it is used to assemble a
    # generous candidate pool rather than to pick the final passages.
    retrieval_candidate_k: int = 15

    # ── Reranking ─────────────────────────────────────────────────────────────
    # "llm"           — listwise rerank with the mini model. No extra
    #                   dependencies, reuses the existing API key, ~1s latency.
    # "cross_encoder" — local cross-encoder. Better latency per query and no API
    #                   cost, but requires sentence-transformers, which pulls in
    #                   torch. That is a poor trade on a 1 GB t2.micro, so it is
    #                   opt-in rather than the default.
    # "none"          — skip reranking and use the fusion order directly.
    #
    # Defaults to "none" on the evidence, not on principle. Measured over the 65
    # golden queries at the shipped retrieval_k of 4 (evals/retrieval.py,
    # `--retrieval-ab`), LLM reranking improves ordering — nDCG +0.061, MRR
    # +0.054, precision +0.065 — but regresses the two things that matter more:
    #
    #   hit rate  1.000 -> 0.923   it discards the relevant document outright
    #                              on ~8% of queries fusion got right
    #   harmful   0.400 -> 0.908   averaged over all queries; concentrated in
    #                              the 26 cashflow-deficit cases that is 1.0 ->
    #                              2.3 chunks per at-risk customer, so credit
    #                              card and personal loan content grows from a
    #                              quarter to over half of their context window
    #
    # Feeding more unsecured-credit material to customers already in deficit is
    # the exact failure the credit guardrail exists to prevent. The prompt-level
    # guardrail still holds, so this is a narrowed safety margin rather than a
    # live defect — but it is not something to enable by default until the
    # reranker is made aware of the cashflow flag. Re-run the A/B after any
    # change to RERANK_SYSTEM_PROMPT or the retrieval query.
    rerank_backend: str = "none"

    # Only consulted when rerank_backend is "cross_encoder".
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Characters of each candidate shown to the reranker. Chunks are 500 chars,
    # so this passes them whole while bounding cost if chunk_size is raised.
    rerank_passage_chars: int = 600

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
