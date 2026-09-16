"""
Hybrid RAG pipeline for BankLens (High Performance & Self-Healing Vector Index).

Handles the full retrieval-augmented generation workflow:

    1. Load — reads all markdown files from the knowledge_base/ directory
    2. Chunk — splits documents into overlapping chunks using LangChain's
               RecursiveCharacterTextSplitter
    3. Embed — converts chunks to dense vector embeddings via OpenAI
    4. Index — creates dense (ChromaDB) and sparse (BM25) indexes cached at startup
    5. Retrieve — fast ~0.4s Reciprocal Rank Fusion (RRF) Hybrid Search combining
                  dense semantic search and sparse lexical keyword search
"""

import hashlib
import os
import re
import shutil
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.context import current_tenant
from app.core.logger import get_logger
from app.pipeline.reranker import rerank

logger = get_logger(__name__)

# ── Tenancy ───────────────────────────────────────────────────────────────────
#
# Each tenant (a bank) has its own product catalogue under
# knowledge_base/<tenant>/, its own Chroma collection persisted under
# <chroma_persist_dir>/<tenant>/, and its own BM25 index. Nothing below the
# public functions takes a tenant that defaults to "whatever is in scope" —
# the tenant is resolved once at the top and passed down explicitly, so a
# cross-tenant read has to be written on purpose.

KNOWLEDGE_BASE_ROOT = Path(__file__).resolve().parent.parent.parent / "knowledge_base"

# The default tenant's catalogue. Kept as a module constant because the
# evaluation harness and the tests address the catalogue directly.
KNOWLEDGE_BASE_DIR = KNOWLEDGE_BASE_ROOT / settings.default_tenant


# A tenant slug becomes a directory name (knowledge_base/<slug>, the index
# directory) and part of a collection name. Anything but this shape is
# refused before it reaches a path, so "../" or "/" can never walk out of
# those roots, and the index rebuild's rmtree can never point elsewhere.
_TENANT_SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def validate_tenant_slug(tenant: str) -> str:
    if not isinstance(tenant, str) or not _TENANT_SLUG.fullmatch(tenant):
        raise ValueError(
            f"invalid tenant {tenant!r}: use lowercase letters, digits, '_' or '-'"
        )
    return tenant


def _tenant_or_default(tenant: str | None) -> str:
    return validate_tenant_slug(tenant or current_tenant())


def kb_dir(tenant: str | None = None) -> Path:
    """The knowledge-base directory for a tenant."""
    resolved = _tenant_or_default(tenant)
    if resolved == settings.default_tenant:
        # Honour monkeypatched KNOWLEDGE_BASE_DIR in tests for the default tenant.
        return KNOWLEDGE_BASE_DIR
    return KNOWLEDGE_BASE_ROOT / resolved


def list_tenants() -> list[str]:
    """Tenants that have a catalogue on disk."""
    if not KNOWLEDGE_BASE_ROOT.exists():
        return []
    return sorted(p.name for p in KNOWLEDGE_BASE_ROOT.iterdir() if p.is_dir())


def persist_dir_for(tenant: str | None = None) -> str:
    """One index per tenant per embedding provider: vectors never mix."""
    from app.platform import gateway

    provider = gateway.embedding_signature().split(":", 1)[0]
    return str(
        Path(settings.chroma_persist_dir) / _tenant_or_default(tenant) / provider
    )


def collection_name_for(tenant: str | None = None) -> str:
    return f"{settings.chroma_collection_name}_{_tenant_or_default(tenant)}"


# Per-tenant BM25 retrievers, built once per process.
_CACHED_BM25_RETRIEVERS: dict[str, BM25Retriever] = {}

# Written inside the persist directory alongside the Chroma index. Records what
# the index was built from, so a stale one can be detected on the next startup.
INDEX_FINGERPRINT_FILE = ".kb_fingerprint"


# ── Step 1: Load ─────────────────────────────────────────────────────────────


def _load_documents(tenant: str | None = None) -> list:
    """Load all markdown files from the tenant's knowledge_base directory."""
    directory = kb_dir(tenant)
    if not directory.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found at: {directory}\n"
            "Each tenant needs a directory under knowledge_base/ at the project root."
        )

    documents = []
    md_files = sorted(directory.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(
            f"No .md files found in {directory}. "
            "The knowledge base must contain at least one markdown file."
        )

    for md_file in md_files:
        loader = TextLoader(str(md_file), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = md_file.name
        documents.extend(docs)
        logger.info("Loaded: %s", md_file.name)

    logger.info("Total knowledge base files loaded: %d", len(md_files))
    return documents


# ── Step 2: Chunk ─────────────────────────────────────────────────────────────


def _chunk_documents(documents: list) -> list:
    """Split documents into smaller, overlapping chunks for precise retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Chunked %d documents into %d chunks (chunk_size=%d, chunk_overlap=%d).",
        len(documents),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks


# ── Index Freshness ───────────────────────────────────────────────────────────
#
# A persisted index is only reusable if it was built from the same inputs. It
# loads without error whether or not the knowledge base has moved on, so
# "loads successfully" is not the same as "is correct" — a new product document
# would simply never appear in dense results, silently and indefinitely.
#
# BM25 is rebuilt from disk on every startup, so it would see the new document
# while the dense half did not. Fusion would still return *something*, which is
# what makes this worth detecting explicitly rather than trusting to look wrong.


def compute_kb_fingerprint(tenant: str | None = None) -> str:
    """
    Hash everything the persisted index depends on.

    Covers the knowledge base contents and the parameters that determine how
    those contents become vectors. Any change to chunking or the embedding
    model invalidates an existing index just as surely as editing a document
    does, so all three go into the same hash.
    """
    digest = hashlib.sha256()

    for md_file in sorted(kb_dir(tenant).glob("*.md")):
        digest.update(md_file.name.encode("utf-8"))
        digest.update(md_file.read_bytes())

    from app.platform import gateway

    digest.update(
        f"{settings.chunk_size}|{settings.chunk_overlap}|"
        f"{gateway.embedding_signature()}".encode("utf-8")
    )

    return digest.hexdigest()


def read_stored_fingerprint(persist_dir: str) -> str | None:
    """Read the fingerprint recorded when the index was built, if any."""
    path = Path(persist_dir) / INDEX_FINGERPRINT_FILE
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def write_fingerprint(persist_dir: str, fingerprint: str) -> None:
    """
    Record the fingerprint of the inputs an index was just built from.

    Never fatal: an index that works but cannot be verified next startup is a
    far better outcome than refusing to serve because a marker file could not
    be written.
    """
    try:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        (Path(persist_dir) / INDEX_FINGERPRINT_FILE).write_text(
            fingerprint, encoding="utf-8"
        )
    except OSError as e:
        logger.warning("Could not write index fingerprint to '%s': %s", persist_dir, e)


# ── Steps 3–4: Embed and Persist Vector Store & BM25 Cache ──────────────────


def get_cached_bm25_retriever(tenant: str | None = None) -> BM25Retriever | None:
    """Get or initialize the cached BM25 retriever for a tenant."""
    resolved = _tenant_or_default(tenant)
    if resolved not in _CACHED_BM25_RETRIEVERS:
        try:
            documents = _load_documents(resolved)
            chunks = _chunk_documents(documents)
            retriever = BM25Retriever.from_documents(chunks)
            retriever.k = settings.retrieval_k
            _CACHED_BM25_RETRIEVERS[resolved] = retriever
            logger.info("Initialized and cached BM25 index for tenant '%s'.", resolved)
        except Exception as e:
            logger.warning("Failed to initialize BM25 index for '%s': %s", resolved, e)
            return None
    return _CACHED_BM25_RETRIEVERS[resolved]


def reset_bm25_cache() -> None:
    """Drop cached BM25 indexes (tests and catalogue reloads)."""
    _CACHED_BM25_RETRIEVERS.clear()


def _rebuild_vector_store(
    persist_dir: str,
    embeddings,
    fingerprint: str,
    tenant: str | None = None,
) -> Chroma:
    """Discard any existing index and rebuild it from the knowledge base."""
    shutil.rmtree(persist_dir, ignore_errors=True)

    documents = _load_documents(tenant)
    chunks = _chunk_documents(documents)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name_for(tenant),
        persist_directory=persist_dir,
    )

    write_fingerprint(persist_dir, fingerprint)
    logger.info("ChromaDB built and persisted at '%s'.", persist_dir)
    return vector_store


def build_vector_store(tenant: str | None = None) -> Chroma:
    """
    Build or load the ChromaDB vector store for a tenant.

    An existing index is reused only when its fingerprint matches the current
    knowledge base and indexing settings. It is rebuilt when the inputs have
    changed, when it was built before fingerprinting existed, or when it fails
    to load at all.
    """
    from app.platform import gateway

    resolved = _tenant_or_default(tenant)
    embeddings = gateway.embeddings()

    persist_dir = persist_dir_for(resolved)
    fingerprint = compute_kb_fingerprint(resolved)

    if not (os.path.exists(persist_dir) and os.listdir(persist_dir)):
        logger.info(
            "No persisted index for tenant '%s'. Building ChromaDB from knowledge base.",
            resolved,
        )
        return _finalize(
            _rebuild_vector_store(persist_dir, embeddings, fingerprint, resolved),
            resolved,
        )

    stored = read_stored_fingerprint(persist_dir)
    if stored != fingerprint:
        logger.info(
            "Persisted index at '%s' is stale (%s). Rebuilding from knowledge base.",
            persist_dir,
            "no fingerprint recorded" if stored is None else "knowledge base changed",
        )
        return _finalize(
            _rebuild_vector_store(persist_dir, embeddings, fingerprint, resolved),
            resolved,
        )

    logger.info(
        "Persisted ChromaDB found at '%s' and fingerprint matches. Loading existing index.",
        persist_dir,
    )
    try:
        vector_store = Chroma(
            collection_name=collection_name_for(resolved),
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    except Exception as e:
        logger.warning(
            "Persisted ChromaDB index incompatible or corrupted (%s). Self-healing and rebuilding.",
            e,
        )
        vector_store = _rebuild_vector_store(
            persist_dir, embeddings, fingerprint, resolved
        )

    return _finalize(vector_store, resolved)


def _finalize(vector_store: Chroma, tenant: str | None = None) -> Chroma:
    """Warm the BM25 cache at startup so the first query is not slow."""
    get_cached_bm25_retriever(tenant)
    return vector_store


# ── Step 5: Reciprocal Rank Fusion (RRF) Fast Hybrid Retrieve ─────────────


def reciprocal_rank_fusion(dense_docs: list, bm25_docs: list, k: int = 4) -> list:
    """Combine Dense and BM25 search results using Reciprocal Rank Fusion (RRF)."""
    scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense_docs):
        doc_id = doc.page_content
        scores[doc_id] = scores.get(doc_id, 0.0) + (0.6 / (60 + rank))
        doc_map[doc_id] = doc

    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.page_content
        scores[doc_id] = scores.get(doc_id, 0.0) + (0.4 / (60 + rank))
        doc_map[doc_id] = doc

    sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_doc_ids[:k]]


def build_retrieval_query(metrics) -> str:
    """
    Render computed metrics into the natural language retrieval query.

    Defined once and shared by the app and the evaluation harness. If the evals
    built their own query string, they would be measuring a retrieval path that
    does not exist in production — a quietly useless eval.

    Args:
        metrics: A FinancialMetrics instance.
    """
    categories = ", ".join(c["category"] for c in metrics.top_categories)
    return (
        f"Customer with monthly income {metrics.total_income:,.0f}, "
        f"expenses {metrics.total_expenses:,.0f}, "
        f"savings rate {metrics.savings_rate_pct:.1f}%, "
        f"expense-to-income ratio {metrics.expense_to_income_ratio:.2f}, "
        f"credit risk {metrics.risk_profile}. "
        f"Top spending categories: {categories}."
    )


def _fused_candidates(
    query: str, vector_store: Chroma, candidate_k: int, tenant: str | None = None
) -> list[dict]:
    """One query's hybrid candidate pool: dense + BM25, RRF-fused."""
    dense_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": candidate_k},
    )

    bm25 = get_cached_bm25_retriever(tenant)
    if bm25 is not None:
        try:
            original_bm25_k = bm25.k
            bm25.k = candidate_k
            try:
                dense_docs = dense_retriever.invoke(query)
                bm25_docs = bm25.invoke(query)
            finally:
                bm25.k = original_bm25_k

            results = reciprocal_rank_fusion(dense_docs, bm25_docs, k=candidate_k)
            logger.info(
                "Executed RRF hybrid search over %d candidates (Dense + Cached BM25).",
                len(results),
            )
        except Exception as e:
            logger.warning("BM25 retrieval error, fallback to Dense search: %s", e)
            results = dense_retriever.invoke(query)
    else:
        results = dense_retriever.invoke(query)

    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
        }
        for doc in results[:candidate_k]
    ]


class QueryVariants(BaseModel):
    """Rewritten retrieval queries produced by the multi-query expander."""

    variants: list[str] = Field(
        description="Rewrites of the query, each emphasising a different facet."
    )


MULTI_QUERY_PROMPT = """You rewrite a bank customer summary into alternative retrieval queries for a
banking product knowledge base.

Produce {count} rewrites, each emphasising a DIFFERENT facet of the customer's
position — for example one focused on savings and deposit products, one on
credit and lending suitability, one on liquidity and cashflow management.
Keep every numeric fact from the original. Do not invent facts.
"""


def _generate_query_variants(query: str, count: int) -> list[str]:
    """
    Rewrite the query `count` ways with the mini model.

    Temperature 0: the variants differ because the prompt demands different
    facets, not because of sampling — so retrieval stays reproducible, which
    the evaluation harness depends on. Any failure returns [] and retrieval
    proceeds single-query; query expansion must never break retrieval.
    """
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from app.platform import gateway

    if not gateway.available():
        return []

    try:
        parser = PydanticOutputParser(pydantic_object=QueryVariants)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", MULTI_QUERY_PROMPT + "\n\n{format_instructions}"),
                ("human", "<query>\n{query}\n</query>"),
            ]
        )
        llm = gateway.chat_model("mini", temperature=0.0)
        result: QueryVariants = (prompt | llm | parser).invoke(
            {
                "query": query,
                "count": count,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        variants = [v.strip() for v in result.variants if v.strip()][:count]
        logger.info("Multi-query expansion produced %d variant(s).", len(variants))
        return variants
    except Exception as exc:  # noqa: BLE001 - degrade to single-query retrieval
        logger.warning("Multi-query expansion failed (%s); using original only.", exc)
        return []


def _rrf_merge_chunk_lists(
    ranked_lists: list[list[dict]], k: int, weights: list[float] | None = None
) -> list[dict]:
    """
    Weighted RRF across per-variant chunk rankings, keyed on (source, content).

    The original query gets full weight and rewrites get half. Unweighted
    fusion let the rewrites outvote the original: measured on the golden
    queries it cut harmful credit content for deficit customers by 92%, but it
    also diluted credit_card.md out of the top-k for a Medium-band customer —
    where a credit card is the *correct* recommendation — and the grounding
    check caught the recommendation's document missing from retrieval.
    Anchoring on the original keeps the diversity benefit without letting the
    rewrites override what the actual query matches.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[tuple, float] = {}
    first_seen: dict[tuple, dict] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for position, chunk in enumerate(ranked):
            key = (chunk["source"], chunk["content"])
            scores[key] = scores.get(key, 0.0) + weight / (60 + position + 1)
            first_seen.setdefault(key, chunk)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [first_seen[key] for key in ordered[:k]]


def retrieve(
    query: str,
    vector_store: Chroma,
    use_reranker: bool = True,
    use_multi_query: bool | None = None,
    tenant: str | None = None,
) -> list[dict]:
    """
    Retrieve product context: hybrid fusion over a wide candidate pool, then rerank.

    Both retrievers are asked for retrieval_candidate_k results rather than the
    final retrieval_k. Fusion is used to assemble candidates, not to choose the
    final passages.

    Args:
        query: Natural language summary of the customer's financial position.
        vector_store: The Chroma collection holding product embeddings.
        use_reranker: Set False to get the raw fusion ordering. Used by the
                      retrieval evaluation to measure the reranker's effect.
        use_multi_query: Expand the query into facet rewrites and fuse their
                         result lists. None defers to settings.multi_query_enabled;
                         the A/B evaluation passes explicit True/False.
        tenant: Whose BM25 index to fuse with. Must match the tenant the
                vector store was built for; None means the tenant in scope.

    Returns:
        Up to retrieval_k chunks as {"content": ..., "source": ...}.
    """
    candidate_k = max(settings.retrieval_candidate_k, settings.retrieval_k)

    multi = settings.multi_query_enabled if use_multi_query is None else use_multi_query
    queries = [query]
    if multi:
        queries += _generate_query_variants(query, settings.multi_query_count)

    if len(queries) > 1:
        per_query = [
            _fused_candidates(q, vector_store, candidate_k, tenant) for q in queries
        ]
        anchor_weights = [1.0] + [0.5] * (len(per_query) - 1)
        candidates = _rrf_merge_chunk_lists(per_query, candidate_k, anchor_weights)
    else:
        candidates = _fused_candidates(query, vector_store, candidate_k, tenant)

    if use_reranker:
        retrieved = rerank(query, candidates, top_k=settings.retrieval_k)
    else:
        retrieved = candidates[: settings.retrieval_k]

    sources = [r["source"] for r in retrieved]
    logger.info(
        "Retrieved %d chunks from %d candidates. Sources: %s",
        len(retrieved),
        len(candidates),
        sources,
    )
    return retrieved
