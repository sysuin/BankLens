"""
Unit tests for app.pipeline.reranker and the retrieval metrics.

The reranker's most important property is not that it ranks well — that is
measured empirically by evals/retrieval.py — but that it never breaks
retrieval. Every failure path here asserts a graceful fall back to the fusion
ordering, because a slightly worse passage order is vastly preferable to an
error where a customer profile should be.

No network access is required: the LLM backend is monkeypatched.
"""

import pytest

from app.core.config import settings
from app.pipeline import reranker as reranker_module
from app.pipeline.reranker import _repair_ranking, rerank
from evals.retrieval import mean_metrics, score_ranking


def _candidates(n: int) -> list[dict]:
    return [{"content": f"passage {i}", "source": f"doc_{i}.md"} for i in range(n)]


@pytest.fixture
def default_backend(monkeypatch):
    """Reset rerank settings to a known state for each test."""
    monkeypatch.setattr(settings, "rerank_backend", "llm")
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "retrieval_k", 4)
    return settings


class TestRepairRanking:
    """A model-proposed ordering must be made safe before it is trusted."""

    def test_clean_ranking_is_preserved(self):
        assert _repair_ranking([2, 0, 1], candidate_count=3, top_k=3) == [2, 0, 1]

    def test_out_of_range_indices_are_dropped(self):
        assert _repair_ranking([9, 1, -4, 0], candidate_count=3, top_k=2) == [1, 0]

    def test_duplicates_are_dropped(self):
        assert _repair_ranking([1, 1, 1, 0], candidate_count=3, top_k=2) == [1, 0]

    def test_short_ranking_is_padded_from_original_order(self):
        """A model returning too few indices must not silently shrink the context."""
        assert _repair_ranking([3], candidate_count=5, top_k=3) == [3, 0, 1]

    def test_empty_ranking_falls_back_to_original_order(self):
        assert _repair_ranking([], candidate_count=4, top_k=2) == [0, 1]

    def test_never_returns_more_than_top_k(self):
        assert len(_repair_ranking([0, 1, 2, 3, 4], candidate_count=5, top_k=2)) == 2


class TestRerankFallbacks:
    """Every failure path must degrade to the fusion order, never raise."""

    def test_backend_none_returns_fusion_order(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_backend", "none")
        candidates = _candidates(6)
        assert rerank("q", candidates, top_k=3) == candidates[:3]

    def test_unknown_backend_returns_fusion_order(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_backend", "quantum")
        candidates = _candidates(6)
        assert rerank("q", candidates, top_k=3) == candidates[:3]

    def test_missing_api_key_returns_fusion_order(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_backend", "llm")
        monkeypatch.setattr(settings, "openai_api_key", "")
        candidates = _candidates(6)
        assert rerank("q", candidates, top_k=2) == candidates[:2]

    def test_backend_exception_returns_fusion_order(self, monkeypatch, default_backend):
        def boom(*_args, **_kwargs):
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(reranker_module, "_rerank_with_llm", boom)
        candidates = _candidates(6)
        assert rerank("q", candidates, top_k=3) == candidates[:3]

    def test_missing_cross_encoder_dependency_returns_fusion_order(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_backend", "cross_encoder")

        def missing(*_args, **_kwargs):
            raise ImportError("No module named 'sentence_transformers'")

        monkeypatch.setattr(reranker_module, "_rerank_with_cross_encoder", missing)
        candidates = _candidates(5)
        assert rerank("q", candidates, top_k=2) == candidates[:2]

    def test_empty_candidates(self, default_backend):
        assert rerank("q", [], top_k=4) == []

    def test_single_candidate_skips_the_backend(self, monkeypatch, default_backend):
        def boom(*_args, **_kwargs):
            raise AssertionError("backend must not be called for one candidate")

        monkeypatch.setattr(reranker_module, "_rerank_with_llm", boom)
        candidates = _candidates(1)
        assert rerank("q", candidates, top_k=4) == candidates


class TestRerankReorders:
    """With a working backend the order actually changes."""

    def test_reordering_is_applied(self, monkeypatch, default_backend):
        def fake(_query, candidates, top_k):
            return list(reversed(candidates))[:top_k]

        monkeypatch.setattr(reranker_module, "_rerank_with_llm", fake)
        candidates = _candidates(5)
        result = rerank("q", candidates, top_k=3)
        assert [c["source"] for c in result] == [
            "doc_4.md",
            "doc_3.md",
            "doc_2.md",
        ]

    def test_defaults_to_settings_retrieval_k(self, monkeypatch, default_backend):
        monkeypatch.setattr(settings, "rerank_backend", "none")
        assert len(rerank("q", _candidates(10))) == settings.retrieval_k


class TestRetrievalMetrics:
    """The metrics used to judge the reranker must themselves be correct."""

    RELEVANT = {"fixed_deposit.md", "sweep_account.md"}

    def test_perfect_ranking(self):
        m = score_ranking(["fixed_deposit.md", "sweep_account.md"], self.RELEVANT)
        assert m.hit is True
        assert m.mrr == 1.0
        assert m.precision == 1.0
        assert m.ndcg == pytest.approx(1.0)

    def test_no_relevant_results(self):
        m = score_ranking(["credit_card.md", "personal_loan.md"], self.RELEVANT)
        assert m.hit is False
        assert m.mrr == 0.0
        assert m.precision == 0.0
        assert m.ndcg == 0.0

    def test_mrr_reflects_first_relevant_position(self):
        m = score_ranking(
            ["credit_card.md", "credit_card.md", "fixed_deposit.md"], self.RELEVANT
        )
        assert m.mrr == pytest.approx(1 / 3)

    def test_ranking_relevant_higher_scores_better(self):
        """The property that makes these metrics able to detect reranking at all."""
        good = score_ranking(["fixed_deposit.md", "credit_card.md"], self.RELEVANT)
        bad = score_ranking(["credit_card.md", "fixed_deposit.md"], self.RELEVANT)
        assert good.ndcg > bad.ndcg
        assert good.mrr > bad.mrr
        assert good.precision == bad.precision  # same set, different order

    def test_harmful_documents_are_counted(self):
        m = score_ranking(
            ["credit_card.md", "fixed_deposit.md"],
            self.RELEVANT,
            harmful={"credit_card.md"},
        )
        assert m.harmful == 1

    def test_empty_results(self):
        m = score_ranking([], self.RELEVANT)
        assert m.hit is False and m.ndcg == 0.0

    def test_mean_metrics_averages_each_key(self):
        rows = [
            score_ranking(["fixed_deposit.md"], self.RELEVANT),
            score_ranking(["credit_card.md"], self.RELEVANT),
        ]
        assert mean_metrics(rows)["hit"] == pytest.approx(0.5)

    def test_mean_of_nothing_is_empty(self):
        assert mean_metrics([]) == {}


class TestRetrieveWiring:
    """retrieve() must actually route candidates through the reranker."""

    class _FakeDoc:
        def __init__(self, content, source):
            self.page_content = content
            self.metadata = {"source": source}

    class _FakeRetriever:
        def __init__(self, docs):
            self._docs = docs

        def invoke(self, _query):
            return list(self._docs)

    class _FakeStore:
        def __init__(self, docs):
            self._docs = docs
            self.requested_k = None

        def as_retriever(self, search_type=None, search_kwargs=None):  # noqa: ARG002
            self.requested_k = (search_kwargs or {}).get("k")
            return TestRetrieveWiring._FakeRetriever(self._docs)

    def _store(self, n=12):
        docs = [self._FakeDoc(f"dense passage {i}", f"dense_{i}.md") for i in range(n)]
        return self._FakeStore(docs)

    def test_candidate_pool_is_wider_than_final_k(self, monkeypatch):
        """The whole point: fetch wide, then narrow."""
        from app.pipeline.rag import retrieve

        monkeypatch.setattr(settings, "rerank_backend", "none")
        store = self._store()
        retrieve("savings rate 75%", store, use_reranker=False)
        assert store.requested_k == settings.retrieval_candidate_k
        assert store.requested_k > settings.retrieval_k

    def test_reranker_is_invoked_and_output_respected(self, monkeypatch):
        from app.pipeline import rag as rag_module

        monkeypatch.setattr(settings, "rerank_backend", "llm")
        monkeypatch.setattr(settings, "openai_api_key", "test-key")

        seen = {}

        def fake_rerank(query, candidates, top_k=None):
            seen["query"] = query
            seen["candidate_count"] = len(candidates)
            return list(reversed(candidates))[:top_k]

        monkeypatch.setattr(rag_module, "rerank", fake_rerank)
        result = rag_module.retrieve("savings rate 75%", self._store())

        assert seen["query"] == "savings rate 75%"
        assert seen["candidate_count"] > settings.retrieval_k
        assert len(result) == settings.retrieval_k

    def test_use_reranker_false_bypasses_it(self, monkeypatch):
        from app.pipeline import rag as rag_module

        def boom(*_args, **_kwargs):
            raise AssertionError("reranker must not run when disabled")

        monkeypatch.setattr(rag_module, "rerank", boom)
        result = rag_module.retrieve("q", self._store(), use_reranker=False)
        assert len(result) == settings.retrieval_k

    def test_bm25_k_is_restored_after_retrieval(self, monkeypatch):
        """Widening BM25's k must not leak into subsequent queries."""
        from app.pipeline.rag import get_cached_bm25_retriever, retrieve

        monkeypatch.setattr(settings, "rerank_backend", "none")
        bm25 = get_cached_bm25_retriever()
        before = bm25.k
        retrieve("savings rate 75%", self._store(), use_reranker=False)
        assert bm25.k == before
