"""
RAG pipeline for BankLens.

Handles the full retrieval-augmented generation workflow:

    1. Load — reads all markdown files from the knowledge_base/ directory
    2. Chunk — splits documents into overlapping chunks using LangChain's
               RecursiveCharacterTextSplitter
    3. Embed — converts chunks to vector embeddings via OpenAI
    4. Persist — stores embeddings in a local ChromaDB collection
    5. Retrieve — given a query string, returns the top-k most similar
                  chunks along with their source filenames

On the first run, steps 1–4 are executed and the index is persisted to disk.
On subsequent runs, the persisted index is loaded directly — no re-embedding.
This makes the app fast to restart and keeps API costs minimal.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# Absolute path to the knowledge base directory
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_base"


# ── Step 1: Load ─────────────────────────────────────────────────────────────


def _load_documents() -> list:
    """
    Load all markdown files from the knowledge_base/ directory.

    Each file is loaded as a LangChain Document and tagged with its
    source filename so we can display which products were retrieved.

    Returns:
        A list of LangChain Document objects, one per .md file.

    Raises:
        FileNotFoundError: If the knowledge_base/ directory does not exist.
    """
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
        # Attach the filename as metadata so it surfaces in the UI
        for doc in docs:
            doc.metadata["source"] = md_file.name
        documents.extend(docs)
        logger.info("Loaded: %s", md_file.name)

    logger.info("Total knowledge base files loaded: %d", len(md_files))
    return documents


# ── Step 2: Chunk ─────────────────────────────────────────────────────────────


def _chunk_documents(documents: list) -> list:
    """
    Split documents into smaller, overlapping chunks for precise retrieval.

    Uses RecursiveCharacterTextSplitter, which tries to split on natural
    boundaries (section headers, paragraphs, lines) before falling back
    to splitting on individual characters.

    The chunk size and overlap are controlled by settings.chunk_size and
    settings.chunk_overlap so they can be tuned without code changes.

    Args:
        documents: List of LangChain Document objects from _load_documents().

    Returns:
        A list of smaller Document chunks, each preserving the source metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        # Try to split on markdown headers and paragraphs first
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Chunked %d documents into %d chunks " "(chunk_size=%d, chunk_overlap=%d).",
        len(documents),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks


# ── Steps 3–4: Embed and Persist ─────────────────────────────────────────────


def build_vector_store() -> Chroma:
    """
    Build or load the ChromaDB vector store.

    If the persist directory already contains data, the existing index
    is loaded without re-embedding. Otherwise, the full build pipeline
    (load → chunk → embed → persist) is executed.

    Returns:
        A Chroma vector store instance ready for similarity search.
    """
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )

    persist_dir = settings.chroma_persist_dir

    # Check for an existing, non-empty persisted index
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        logger.info(
            "Persisted ChromaDB found at '%s'. Loading existing index.", persist_dir
        )
        vector_store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
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
        logger.info(
            "ChromaDB built and persisted at '%s'. "
            "Subsequent runs will load from disk.",
            persist_dir,
        )

    return vector_store


# ── Step 5: Retrieve ──────────────────────────────────────────────────────────


def retrieve(query: str, vector_store: Chroma) -> list[dict]:
    """
    Retrieve the top-k most semantically similar product chunks for a query.

    The query is embedded using the same model as the knowledge base, then
    ChromaDB performs a cosine similarity search and returns the nearest chunks.

    Args:
        query: A natural language string that describes the customer's
               financial situation. Used as the semantic search query.
        vector_store: An initialised Chroma vector store instance.

    Returns:
        A list of dicts, each containing:
            'content' (str)  — the chunk text passed to the LLM as context
            'source'  (str)  — the knowledge base filename (e.g. 'fixed_deposit.md')
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_k},
    )

    results = retriever.invoke(query)

    retrieved = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
        }
        for doc in results
    ]

    sources = [r["source"] for r in retrieved]
    logger.info(
        "Retrieved %d chunks from ChromaDB. Sources: %s",
        len(retrieved),
        sources,
    )
    return retrieved
