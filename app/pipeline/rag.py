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

import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# Absolute path to the knowledge base directory
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_base"

# Global module cache for BM25 retriever to avoid re-reading files on every query
_CACHED_BM25_RETRIEVER = None


# ── Step 1: Load ─────────────────────────────────────────────────────────────


def _load_documents() -> list:
    """Load all markdown files from the knowledge_base/ directory."""
    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found at: {KNOWLEDGE_BASE_DIR}\n"
            "Make sure the knowledge_base/ directory exists at the project root."
        )

    documents = []
    md_files = sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(
            f"No .md files found in {KNOWLEDGE_BASE_DIR}. "
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


# ── Steps 3–4: Embed and Persist Vector Store & BM25 Cache ──────────────────


def get_cached_bm25_retriever() -> BM25Retriever | None:
    """Get or initialize the cached BM25 retriever instance."""
    global _CACHED_BM25_RETRIEVER
    if _CACHED_BM25_RETRIEVER is None:
        try:
            documents = _load_documents()
            chunks = _chunk_documents(documents)
            _CACHED_BM25_RETRIEVER = BM25Retriever.from_documents(chunks)
            _CACHED_BM25_RETRIEVER.k = settings.retrieval_k
            logger.info("Initialized and cached BM25 index at startup.")
        except Exception as e:
            logger.warning("Failed to initialize BM25 index: %s", e)
            _CACHED_BM25_RETRIEVER = None
    return _CACHED_BM25_RETRIEVER


def build_vector_store() -> Chroma:
    """Build or load the ChromaDB vector store with self-healing fallback."""
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key or "dummy_key",
    )

    persist_dir = settings.chroma_persist_dir

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        logger.info(
            "Persisted ChromaDB found at '%s'. Loading existing index.", persist_dir
        )
        try:
            vector_store = Chroma(
                collection_name=settings.chroma_collection_name,
                embedding_function=embeddings,
                persist_directory=persist_dir,
            )
        except Exception as e:
            logger.warning(
                "Persisted ChromaDB index incompatible or corrupted (%s). Self-healing and rebuilding.",
                e,
            )
            shutil.rmtree(persist_dir, ignore_errors=True)
            documents = _load_documents()
            chunks = _chunk_documents(documents)
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=settings.chroma_collection_name,
                persist_directory=persist_dir,
            )
    else:
        logger.info("No persisted index found. Building ChromaDB from knowledge base.")
        documents = _load_documents()
        chunks = _chunk_documents(documents)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=settings.chroma_collection_name,
            persist_directory=persist_dir,
        )
        logger.info("ChromaDB built and persisted at '%s'.", persist_dir)

    # Warm up BM25 cache at startup
    get_cached_bm25_retriever()

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


def retrieve(query: str, vector_store: Chroma) -> list[dict]:
    """
    Fast RRF Hybrid Search using cached BM25 and ChromaDB.
    """
    dense_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_k},
    )

    bm25 = get_cached_bm25_retriever()
    if bm25 is not None:
        try:
            dense_docs = dense_retriever.invoke(query)
            bm25_docs = bm25.invoke(query)
            results = reciprocal_rank_fusion(
                dense_docs, bm25_docs, k=settings.retrieval_k
            )
            logger.info("Executed Fast RRF Hybrid Search (Dense + Cached BM25).")
        except Exception as e:
            logger.warning("BM25 retrieval error, fallback to Dense search: %s", e)
            results = dense_retriever.invoke(query)
    else:
        results = dense_retriever.invoke(query)

    retrieved = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
        }
        for doc in results[: settings.retrieval_k]
    ]

    sources = [r["source"] for r in retrieved]
    logger.info(
        "Retrieved %d chunks from Hybrid RAG. Sources: %s",
        len(retrieved),
        sources,
    )
    return retrieved
