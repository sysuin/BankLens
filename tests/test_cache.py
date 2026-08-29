"""Unit tests for the profile response cache. Fully offline."""

from app.core.config import settings
from app.pipeline import cache as cache_module
from app.pipeline.agent import CustomerProfile
from app.pipeline.analyzer import FinancialMetrics
from app.pipeline.cache import (
    cached_build_profile,
    profile_cache_key,
    read_cached_profile,
    write_cached_profile,
)


def _metrics(**overrides) -> FinancialMetrics:
    base = dict(
        total_income=100000.0,
        total_expenses=50000.0,
        savings_amount=50000.0,
        savings_rate_pct=50.0,
        expense_to_income_ratio=0.5,
        top_categories=[{"category": "Food", "total_spent": 10000.0}],
        transaction_count=10,
        credit_count=2,
        debit_count=8,
        period="2024-01",
        essential_expenses=40000.0,
        discretionary_expenses=8000.0,
        unclassified_expenses=2000.0,
        risk_profile="Low",
        financial_health_score=85,
        is_cashflow_negative=False,
    )
    base.update(overrides)
    return FinancialMetrics(**base)


def _profile() -> CustomerProfile:
    return CustomerProfile(
        financial_persona="Steady Saver",
        income_stability_analysis="a",
        spending_pattern_breakdown="b",
        credit_risk_assessment="c",
        primary_product="Fixed Deposit",
        primary_reason="r",
        secondary_product="Savings Account",
        secondary_reason="s",
        rm_hook_points=["1", "2", "3"],
        financial_health_score=85,
        risk_profile="Low",
        retrieved_sources=["fixed_deposit.md"],
    )


CHUNKS = [{"source": "fixed_deposit.md", "content": "FD offers guaranteed returns."}]


class TestCacheKey:
    def test_key_is_stable(self):
        assert profile_cache_key(_metrics(), CHUNKS) == profile_cache_key(
            _metrics(), CHUNKS
        )

    def test_key_changes_with_metrics(self):
        assert profile_cache_key(_metrics(), CHUNKS) != profile_cache_key(
            _metrics(total_income=100001.0), CHUNKS
        )

    def test_key_changes_with_retrieved_context(self):
        other = [{"source": "credit_card.md", "content": "different"}]
        assert profile_cache_key(_metrics(), CHUNKS) != profile_cache_key(
            _metrics(), other
        )

    def test_key_changes_with_model(self, monkeypatch):
        before = profile_cache_key(_metrics(), CHUNKS)
        monkeypatch.setattr(settings, "openai_model", "some-other-model")
        assert profile_cache_key(_metrics(), CHUNKS) != before


class TestRoundTrip:
    def test_write_then_read(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "profile_cache_dir", str(tmp_path))
        write_cached_profile("k1", _profile())
        loaded = read_cached_profile("k1")
        assert loaded is not None and loaded.financial_persona == "Steady Saver"

    def test_miss_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "profile_cache_dir", str(tmp_path))
        assert read_cached_profile("nope") is None

    def test_corrupt_entry_is_a_miss_and_is_removed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "profile_cache_dir", str(tmp_path))
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
        assert read_cached_profile("bad") is None
        assert not (tmp_path / "bad.json").exists()


class TestCachedBuildProfile:
    def test_second_call_is_served_from_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "profile_cache_dir", str(tmp_path))
        monkeypatch.setattr(settings, "profile_cache_enabled", True)
        calls = {"n": 0}

        def fake_build(metrics, chunks):
            calls["n"] += 1
            return _profile()

        monkeypatch.setattr(cache_module, "build_profile", fake_build)

        _, from_cache_1 = cached_build_profile(_metrics(), CHUNKS)
        _, from_cache_2 = cached_build_profile(_metrics(), CHUNKS)

        assert (from_cache_1, from_cache_2) == (False, True)
        assert calls["n"] == 1

    def test_disabled_cache_always_generates(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "profile_cache_dir", str(tmp_path))
        monkeypatch.setattr(settings, "profile_cache_enabled", False)
        calls = {"n": 0}

        def fake_build(metrics, chunks):
            calls["n"] += 1
            return _profile()

        monkeypatch.setattr(cache_module, "build_profile", fake_build)

        cached_build_profile(_metrics(), CHUNKS)
        _, from_cache = cached_build_profile(_metrics(), CHUNKS)

        assert from_cache is False
        assert calls["n"] == 2
