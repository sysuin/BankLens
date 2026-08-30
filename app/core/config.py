"""
Configuration management for BankLens.

Uses Pydantic Settings to load and validate all environment variables
from the .env file. A single 'settings' instance is imported and shared
across the entire application — no config values are hardcoded anywhere.
"""

import os

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

    # Modern spelling of the same credential. A present key is treated as
    # intent to trace — nobody puts a LangSmith key in .env hoping for
    # nothing, which is what used to happen when only the legacy LANGCHAIN_*
    # names were recognised.
    langsmith_api_key: str = ""

    # LangSmith is regional and a key is only valid in the region it was
    # created in — a key from another region authenticates as 403, which reads
    # like a bad credential and is not. Default is the US data plane; set
    # LANGSMITH_ENDPOINT to https://eu.api.smith.langchain.com or
    # https://apac.api.smith.langchain.com to match where the key was issued.
    langsmith_endpoint: str = "https://api.smith.langchain.com"

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

    # ── Multi-Query Retrieval ────────────────────────────────────────────────
    # Expands the retrieval query into facet-specific rewrites (savings vs
    # credit vs liquidity) and RRF-fuses their result lists.
    #
    # Defaults ON by the same evidence standard that keeps the reranker OFF.
    # Fusion is *anchored*: the original query gets full RRF weight, rewrites
    # get half. The first, unweighted attempt cut harmful credit content for
    # deficit customers by 92% — but the grounded eval layer then caught the
    # same dilution removing credit_card.md for a Medium-band customer whose
    # correct recommendation it was. Anchoring keeps diversity subordinate to
    # what the actual query matches.
    #
    # A/B over the 65 golden queries (`--multi-query-ab`, top-4, no reranker),
    # anchored fusion — nothing regresses:
    #
    #   hit      1.000 -> 1.000   unchanged
    #   mrr      0.785 -> 0.838   first relevant document ranks higher
    #   ndcg     0.626 -> 0.635   slightly better
    #   prec     0.419 -> 0.419   unchanged
    #   harmful  0.400 -> 0.354   modestly less credit content for deficit
    #                             customers
    #
    # And retrieval_supports_recommendation holds 6/6 in the grounded layer.
    # Cost per analysis: one mini-model rewrite call + two extra hybrid
    # retrievals (~1s). Re-run BOTH the A/B and `--with-llm` after changing
    # the rewrite prompt or the fusion weights.
    multi_query_enabled: bool = True
    multi_query_count: int = 2

    # ── Profile Response Cache ───────────────────────────────────────────────
    # Exact-key cache over (metrics, retrieved chunks, system prompt, model).
    # Correctness note: this is deliberately NOT semantic caching — similar
    # statements are different customers. The key hashes the prompt file and
    # model name, so edits invalidate automatically.
    profile_cache_enabled: bool = True
    profile_cache_dir: str = "./.profile_cache"

    # ── Vision OCR (Engine 3) ────────────────────────────────────────────────
    # Off by default: the sanitizer masks PII in *text* before it reaches an
    # external model, but a rendered page image cannot be masked. Enabling
    # this trades that guarantee for the ability to read scanned statements.
    vision_ocr_enabled: bool = False
    vision_ocr_max_pages: int = 4

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

# LangChain reads tracing configuration from the process environment, not from
# this Settings object. Exporting here is what actually turns tracing on when
# .env asks for it; setdefault so a real environment variable still wins.
_tracing_key = settings.langsmith_api_key or settings.langchain_api_key
_tracing_wanted = settings.langchain_tracing_v2.strip().lower() == "true" or bool(
    settings.langsmith_api_key
)
if _tracing_wanted and _tracing_key:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", _tracing_key)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", _tracing_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)
    os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.langsmith_endpoint)
