"""
Unit tests for app.pipeline.rag.

Tests verify that:
    - The knowledge base directory loads all expected .md files
    - Chunking produces a non-empty list of chunks from the documents
    - Chunks preserve the 'source' metadata from their parent document
    - The retrieve() function returns exactly k results

Note: These tests do NOT call the OpenAI embeddings API. The RAG tests
that require a real vector store are integration tests and are tagged
with @pytest.mark.integration so they can be skipped in CI if no
API key is available.
"""

from unittest.mock import MagicMock

from app.pipeline.rag import (
    _load_documents,
    _chunk_documents,
    retrieve,
    KNOWLEDGE_BASE_DIR,
)


class TestLoadDocuments:
    """Tests for the _load_documents() internal function."""

    def test_knowledge_base_directory_exists(self):
        """The knowledge_base/ directory must exist in the project root."""
        assert KNOWLEDGE_BASE_DIR.exists(), (
            f"knowledge_base/ directory not found at {KNOWLEDGE_BASE_DIR}. "
            "Make sure you are running tests from the project root."
        )

    def test_all_six_product_files_are_present(self):
        """All six knowledge base markdown files must be present."""
        expected_files = {
            "fixed_deposit.md",
            "personal_loan.md",
            "credit_card.md",
            "savings_account.md",
            "recurring_deposit.md",
            "sweep_account.md",
        }
        actual_files = {f.name for f in KNOWLEDGE_BASE_DIR.glob("*.md")}
        missing = expected_files - actual_files
        assert not missing, f"Missing knowledge base files: {missing}"

    def test_load_documents_returns_six_documents(self):
        """_load_documents() should return exactly one document per .md file."""
        documents = _load_documents()
        assert len(documents) == 6

    def test_each_document_has_source_metadata(self):
        """Each loaded document must have a 'source' key in its metadata."""
        documents = _load_documents()
        for doc in documents:
            assert (
                "source" in doc.metadata
            ), f"Document from '{doc.metadata}' is missing 'source' metadata."

    def test_source_metadata_is_filename(self):
        """The 'source' metadata should be the .md filename, not a full path."""
        documents = _load_documents()
        for doc in documents:
            source = doc.metadata["source"]
            assert source.endswith(
                ".md"
            ), f"Expected a .md filename in source metadata, got: '{source}'"
            assert (
                "/" not in source and "\\" not in source
            ), f"Source should be a filename, not a path: '{source}'"


class TestChunkDocuments:
    """Tests for the _chunk_documents() internal function."""

    def test_chunking_produces_more_chunks_than_documents(self):
        """
        Chunking should split documents into more pieces than the original count.
        (Each knowledge base file is large enough to produce multiple chunks.)
        """
        documents = _load_documents()
        chunks = _chunk_documents(documents)
        assert len(chunks) > len(documents)

    def test_chunks_preserve_source_metadata(self):
        """Every chunk must inherit the 'source' metadata from its parent document."""
        documents = _load_documents()
        chunks = _chunk_documents(documents)
        for chunk in chunks:
            assert (
                "source" in chunk.metadata
            ), f"Chunk is missing 'source' metadata: {chunk.page_content[:50]}"

    def test_chunk_content_is_non_empty(self):
        """No chunk should have empty page content."""
        documents = _load_documents()
        chunks = _chunk_documents(documents)
        for chunk in chunks:
            assert chunk.page_content.strip(), "Found a chunk with empty content."


class TestRetrieve:
    """Tests for the retrieve() function using a mocked vector store."""

    def test_retrieve_returns_correct_number_of_results(self):
        """
        retrieve() should return exactly k results as configured.
        Uses a mock vector store to avoid API calls.
        """
        # Create mock LangChain Document objects
        mock_doc_1 = MagicMock()
        mock_doc_1.page_content = "Fixed Deposit: guaranteed returns for savers."
        mock_doc_1.metadata = {"source": "fixed_deposit.md"}

        mock_doc_2 = MagicMock()
        mock_doc_2.page_content = "Recurring Deposit: monthly savings discipline."
        mock_doc_2.metadata = {"source": "recurring_deposit.md"}

        mock_doc_3 = MagicMock()
        mock_doc_3.page_content = "Savings Account: high yield for idle balance."
        mock_doc_3.metadata = {"source": "savings_account.md"}

        # Mock the retriever to return 3 documents
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc_1, mock_doc_2, mock_doc_3]

        mock_vector_store = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever

        results = retrieve(
            query="Customer with high savings rate and low expenses.",
            vector_store=mock_vector_store,
        )

        assert len(results) == 3

    def test_retrieve_result_has_content_and_source_keys(self):
        """Each result dict must have 'content' and 'source' keys."""
        mock_doc = MagicMock()
        mock_doc.page_content = "Some product description."
        mock_doc.metadata = {"source": "fixed_deposit.md"}

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]

        mock_vector_store = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever

        results = retrieve(query="test query", vector_store=mock_vector_store)

        assert len(results) == 1
        assert "content" in results[0]
        assert "source" in results[0]
        assert results[0]["source"] == "fixed_deposit.md"
