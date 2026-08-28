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

from unittest.mock import MagicMock, patch

from app.pipeline import rag
from app.pipeline.rag import (
    _load_documents,
    _chunk_documents,
    build_vector_store,
    compute_kb_fingerprint,
    read_stored_fingerprint,
    retrieve,
    write_fingerprint,
    INDEX_FINGERPRINT_FILE,
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

    def test_all_product_files_are_present(self):
        """All ten knowledge base markdown files must be present."""
        expected_files = {
            "fixed_deposit.md",
            "personal_loan.md",
            "debt_consolidation_loan.md",
            "credit_card.md",
            "savings_account.md",
            "recurring_deposit.md",
            "sweep_account.md",
            "home_loan_mortgage.md",
            "mutual_funds_sip.md",
            "sme_credit_line.md",
        }
        actual_files = {f.name for f in KNOWLEDGE_BASE_DIR.glob("*.md")}
        missing = expected_files - actual_files
        assert not missing, f"Missing knowledge base files: {missing}"

    def test_load_documents_returns_one_document_per_file(self):
        """_load_documents() should return exactly one document per .md file."""
        documents = _load_documents()
        expected = len(list(KNOWLEDGE_BASE_DIR.glob("*.md")))
        assert len(documents) == expected

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


class TestIndexFingerprint:
    """
    Tests for persisted-index freshness detection.

    These guard a specific production failure: the deploy step stopped wiping
    the persisted ChromaDB volume in the same release that added three new
    knowledge base products. A stale index loads without error, so nothing
    raised — the new products were simply absent from dense retrieval.
    """

    def _write_kb(self, directory, files: dict[str, str]):
        for name, body in files.items():
            (directory / name).write_text(body, encoding="utf-8")
        return directory

    def test_fingerprint_is_stable_across_calls(self):
        """The same knowledge base must always hash to the same value."""
        assert compute_kb_fingerprint() == compute_kb_fingerprint()

    def test_fingerprint_changes_when_a_document_changes(self, tmp_path, monkeypatch):
        """Editing a product document must invalidate the index."""
        kb = self._write_kb(tmp_path, {"fixed_deposit.md": "# Fixed Deposit\nOld."})
        monkeypatch.setattr(rag, "KNOWLEDGE_BASE_DIR", kb)

        before = compute_kb_fingerprint()
        (kb / "fixed_deposit.md").write_text("# Fixed Deposit\nNew.", encoding="utf-8")

        assert compute_kb_fingerprint() != before

    def test_fingerprint_changes_when_a_product_is_added(self, tmp_path, monkeypatch):
        """The exact regression: adding a product must invalidate the index."""
        kb = self._write_kb(tmp_path, {"fixed_deposit.md": "# Fixed Deposit\nBody."})
        monkeypatch.setattr(rag, "KNOWLEDGE_BASE_DIR", kb)

        before = compute_kb_fingerprint()
        (kb / "home_loan_mortgage.md").write_text(
            "# Home Loan\nBody.", encoding="utf-8"
        )

        assert compute_kb_fingerprint() != before

    def test_fingerprint_changes_when_chunking_changes(self, monkeypatch):
        """Re-chunking produces different vectors, so it must invalidate too."""
        before = compute_kb_fingerprint()
        monkeypatch.setattr(rag.settings, "chunk_size", rag.settings.chunk_size + 100)

        assert compute_kb_fingerprint() != before

    def test_fingerprint_changes_when_embedding_model_changes(self, monkeypatch):
        """Vectors from a different model are not comparable to the stored ones."""
        before = compute_kb_fingerprint()
        monkeypatch.setattr(rag.settings, "openai_embedding_model", "some-other-model")

        assert compute_kb_fingerprint() != before

    def test_missing_fingerprint_reads_as_none(self, tmp_path):
        """An index built before fingerprinting existed has no marker file."""
        assert read_stored_fingerprint(str(tmp_path)) is None

    def test_fingerprint_round_trips(self, tmp_path):
        """What is written must be what is read back."""
        write_fingerprint(str(tmp_path), "abc123")

        assert (tmp_path / INDEX_FINGERPRINT_FILE).exists()
        assert read_stored_fingerprint(str(tmp_path)) == "abc123"


class TestBuildVectorStoreFreshness:
    """
    Tests that build_vector_store() acts on the fingerprint.

    Chroma and the embeddings client are mocked throughout — these assert
    control flow (load vs rebuild), never real embedding calls.
    """

    def _persisted_dir(self, tmp_path):
        """A directory that looks like an existing persisted index."""
        (tmp_path / "chroma.sqlite3").write_text("not empty", encoding="utf-8")
        return tmp_path

    def _run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rag.settings, "chroma_persist_dir", str(tmp_path))

        with (
            patch.object(rag, "OpenAIEmbeddings"),
            patch.object(rag, "get_cached_bm25_retriever"),
            patch.object(rag, "_load_documents", return_value=["doc"]),
            patch.object(rag, "_chunk_documents", return_value=["chunk"]),
            patch.object(rag, "Chroma") as mock_chroma,
        ):
            build_vector_store()
            return mock_chroma

    def test_stale_index_is_rebuilt(self, tmp_path, monkeypatch):
        """A fingerprint that does not match the knowledge base forces a rebuild."""
        persist = self._persisted_dir(tmp_path)
        write_fingerprint(str(persist), "a-fingerprint-from-an-older-knowledge-base")

        mock_chroma = self._run(persist, monkeypatch)

        mock_chroma.from_documents.assert_called_once()
        mock_chroma.assert_not_called()

    def test_unfingerprinted_index_is_rebuilt(self, tmp_path, monkeypatch):
        """An index predating fingerprinting is untrusted and rebuilt once."""
        persist = self._persisted_dir(tmp_path)

        mock_chroma = self._run(persist, monkeypatch)

        mock_chroma.from_documents.assert_called_once()

    def test_matching_index_is_loaded_not_rebuilt(self, tmp_path, monkeypatch):
        """A current index must be reused — rebuilding every start wastes money."""
        persist = self._persisted_dir(tmp_path)
        write_fingerprint(str(persist), compute_kb_fingerprint())

        mock_chroma = self._run(persist, monkeypatch)

        mock_chroma.assert_called_once()
        mock_chroma.from_documents.assert_not_called()

    def test_rebuild_records_the_current_fingerprint(self, tmp_path, monkeypatch):
        """After a rebuild the next startup must see a matching fingerprint."""
        persist = self._persisted_dir(tmp_path)
        write_fingerprint(str(persist), "stale")

        self._run(persist, monkeypatch)

        assert read_stored_fingerprint(str(persist)) == compute_kb_fingerprint()


class TestRetrieve:
    """Tests for the retrieve() function using a mocked vector store."""

    def test_retrieve_returns_correct_number_of_results(self):
        """
        retrieve() should return results as configured.
        Uses a mock vector store to avoid API calls.
        """
        mock_doc_1 = MagicMock()
        mock_doc_1.page_content = "Fixed Deposit: guaranteed returns for savers."
        mock_doc_1.metadata = {"source": "fixed_deposit.md"}

        mock_doc_2 = MagicMock()
        mock_doc_2.page_content = "Recurring Deposit: monthly savings discipline."
        mock_doc_2.metadata = {"source": "recurring_deposit.md"}

        mock_doc_3 = MagicMock()
        mock_doc_3.page_content = "Savings Account: high yield for idle balance."
        mock_doc_3.metadata = {"source": "savings_account.md"}

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc_1, mock_doc_2, mock_doc_3]

        mock_vector_store = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever

        results = retrieve(
            query="Customer with high savings rate and low expenses.",
            vector_store=mock_vector_store,
        )

        assert len(results) >= 1

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

        assert len(results) >= 1
        assert "content" in results[0]
        assert "source" in results[0]
