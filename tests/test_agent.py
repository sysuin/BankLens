"""
Unit tests for app.pipeline.agent.

These cover the output guardrails rather than the LLM call itself:

    - product names are validated against the knowledge base, so an invented
      product cannot reach the UI as a real recommendation
    - the health score is bounded
    - exactly three RM hook points are required
    - the risk rating and health score are NOT fields the LLM can set

No network access is required — every test constructs models directly.
"""

import pytest
from pydantic import ValidationError

from app.pipeline.agent import (
    CustomerProfile,
    ProfileNarrative,
    get_product_catalogue,
    resolve_product,
)


def _narrative_kwargs(**overrides) -> dict:
    """A valid ProfileNarrative payload, with selective overrides."""
    payload = {
        "financial_persona": "Disciplined High Saver",
        "income_stability_analysis": "Salary credits arrive monthly without interruption.",
        "spending_pattern_breakdown": "Essential outlays dominate; discretionary spend is modest.",
        "credit_risk_assessment": "Surplus cashflow supports a Low risk rating.",
        "primary_product": "Fixed Deposit",
        "primary_reason": "Idle surplus cash can be locked at a guaranteed rate.",
        "secondary_product": "Savings Account",
        "secondary_reason": "A high-yield account keeps the buffer liquid.",
        "rm_hook_points": ["Observation", "Value proposition", "Call to action"],
    }
    payload.update(overrides)
    return payload


class TestProductCatalogue:
    """The catalogue is built from the knowledge base on disk."""

    def test_catalogue_covers_every_knowledge_base_file(self):
        catalogue = get_product_catalogue()
        assert len(catalogue) == 10
        assert all(name.endswith(".md") for name in catalogue)

    @pytest.mark.parametrize(
        "name",
        [
            "Fixed Deposit",
            "Fixed Deposit (FD)",
            "Sweep-In Fixed Deposit",
            "High-Yield Savings Account",
            "Recurring Deposit",
            "Credit Card",
            "Debt Consolidation Loan",
            "Home Loan",
            "Personal Loan",
            "Systematic Investment Plan",
            "SIP Mutual Fund",
            "SME Credit Line",
        ],
    )
    def test_real_product_variants_resolve(self, name):
        """Legitimate phrasings of real products must be accepted."""
        assert resolve_product(name) is not None

    @pytest.mark.parametrize(
        "name",
        [
            "Platinum Rewards Card",
            "Gold Loan",
            "Crypto Savings Vault",
            "BankLens Premier Wealth Bundle",
            "",
            "   ",
        ],
    )
    def test_invented_products_are_rejected(self, name):
        """Plausible-sounding inventions must not resolve to a real product."""
        assert resolve_product(name) is None

    @pytest.mark.parametrize(
        "name,expected",
        [
            # The regression: sweep_account.md is titled "Sweep-In Fixed
            # Deposit (Sweep Account)", so this name scores 1.0 against it and
            # against fixed_deposit.md, whose alias it also fully contains.
            # The tie used to be settled alphabetically, sending it to the
            # wrong document while still resolving to something — which is why
            # an is-not-None assertion never caught it.
            ("Sweep-In Fixed Deposit", "sweep_account.md"),
            ("Sweep Account", "sweep_account.md"),
            ("Fixed Deposit", "fixed_deposit.md"),
            ("Fixed Deposit (FD)", "fixed_deposit.md"),
            ("Recurring Deposit", "recurring_deposit.md"),
            ("High-Yield Savings Account", "savings_account.md"),
            ("Systematic Investment Plan", "mutual_funds_sip.md"),
            ("Home Loan", "home_loan_mortgage.md"),
            ("Personal Loan", "personal_loan.md"),
            ("Debt Consolidation Loan", "debt_consolidation_loan.md"),
            ("Credit Card", "credit_card.md"),
        ],
    )
    def test_products_resolve_to_the_right_document(self, name, expected):
        """
        Resolving to *a* product is not enough; it must be the right one.

        check_credit_guardrail decides whether a forbidden product was
        recommended by resolving the name, so a plausible mis-resolution would
        make a safety check answer the wrong question.
        """
        assert resolve_product(name) == expected


class TestProfileNarrativeValidation:
    """Semantic validation of the LLM's output."""

    def test_valid_narrative_is_accepted(self):
        narrative = ProfileNarrative(**_narrative_kwargs())
        assert narrative.primary_product == "Fixed Deposit"

    def test_hallucinated_primary_product_is_rejected(self):
        """The bug this guardrail exists to prevent."""
        with pytest.raises(ValidationError) as excinfo:
            ProfileNarrative(
                **_narrative_kwargs(primary_product="Platinum Rewards Card")
            )
        assert "not a product in the knowledge base" in str(excinfo.value)

    def test_hallucinated_secondary_product_is_rejected(self):
        with pytest.raises(ValidationError):
            ProfileNarrative(
                **_narrative_kwargs(secondary_product="Crypto Savings Vault")
            )

    @pytest.mark.parametrize(
        "hooks",
        [
            ["only one"],
            ["one", "two"],
            ["one", "two", "three", "four"],
            [],
        ],
    )
    def test_hook_points_must_number_exactly_three(self, hooks):
        with pytest.raises(ValidationError):
            ProfileNarrative(**_narrative_kwargs(rm_hook_points=hooks))

    def test_narrative_cannot_set_risk_profile(self):
        """Risk rating is computed, not generated — it is not on the narrative model."""
        assert "risk_profile" not in ProfileNarrative.model_fields
        assert "financial_health_score" not in ProfileNarrative.model_fields


class TestCustomerProfileComposition:
    """CustomerProfile carries the computed fields on top of the narrative."""

    def _profile(self, **overrides) -> CustomerProfile:
        payload = _narrative_kwargs()
        payload.update(
            {
                "financial_health_score": 92,
                "risk_profile": "Low",
                "retrieved_sources": ["fixed_deposit.md"],
            }
        )
        payload.update(overrides)
        return CustomerProfile(**payload)

    def test_composition_succeeds(self):
        profile = self._profile()
        assert profile.risk_profile == "Low"
        assert profile.financial_health_score == 92

    @pytest.mark.parametrize("score", [-1, 101, 150, 1000])
    def test_health_score_is_bounded(self, score):
        with pytest.raises(ValidationError):
            self._profile(financial_health_score=score)

    def test_invalid_risk_band_is_rejected(self):
        with pytest.raises(ValidationError):
            self._profile(risk_profile="Severe")

    def test_backward_compatible_ui_aliases(self):
        """components.py still reads these aliases."""
        profile = self._profile()
        assert profile.recommended_product == profile.primary_product
        assert profile.recommendation_reason == profile.primary_reason
        assert profile.rm_hook == "Observation | Value proposition | Call to action"


class TestCorrectiveRetry:
    """A rejected product name should trigger one corrective retry, not a hard failure."""

    @staticmethod
    def _metrics():
        from app.pipeline.analyzer import FinancialMetrics

        return FinancialMetrics(
            total_income=2400000.0,
            total_expenses=582000.0,
            savings_amount=1818000.0,
            savings_rate_pct=75.75,
            expense_to_income_ratio=0.2425,
            top_categories=[{"category": "Rent & Housing", "total_spent": 360000.0}],
            transaction_count=120,
            credit_count=12,
            debit_count=108,
            period="2024-01 to 2024-12",
            essential_expenses=520000.0,
            discretionary_expenses=50000.0,
            unclassified_expenses=12000.0,
            risk_profile="Low",
            financial_health_score=92,
            is_cashflow_negative=False,
        )

    @staticmethod
    def _response(primary_product: str) -> str:
        import json

        return json.dumps(_narrative_kwargs(primary_product=primary_product))

    def _run(self, monkeypatch, responses):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        import app.pipeline.agent as agent_module

        made = []

        def fake_chat_openai(**_kwargs):
            model = FakeListChatModel(responses=list(responses))
            made.append(model)
            return model

        monkeypatch.setattr(agent_module, "ChatOpenAI", fake_chat_openai)
        return agent_module.build_profile(
            self._metrics(),
            [{"source": "fixed_deposit.md", "content": "Fixed Deposit details."}],
        )

    def test_invalid_product_recovers_on_retry(self, monkeypatch):
        profile = self._run(
            monkeypatch,
            [self._response("Platinum Rewards Card"), self._response("Fixed Deposit")],
        )
        assert profile.primary_product == "Fixed Deposit"

    def test_risk_fields_come_from_metrics_not_the_model(self, monkeypatch):
        """Even a compliant model response cannot influence the rating."""
        profile = self._run(monkeypatch, [self._response("Fixed Deposit")] * 2)
        assert profile.risk_profile == "Low"
        assert profile.financial_health_score == 92
        assert profile.retrieved_sources == ["fixed_deposit.md"]

    def test_two_consecutive_failures_raise(self, monkeypatch):
        """If the retry also fails, fail loudly rather than surface a bad product."""
        from langchain_core.exceptions import OutputParserException

        with pytest.raises(OutputParserException) as excinfo:
            self._run(monkeypatch, [self._response("Gold Loan")] * 2)
        assert "not a product in the knowledge base" in str(excinfo.value)
