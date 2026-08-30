"""
Unit tests for app.pipeline.categorizer.

Tests verify that:
    - Known merchant keywords map to the correct category
    - Unknown descriptions fall back to 'Others'
    - Matching is case-insensitive
    - categorize_dataframe() adds a 'category' column without mutating the input
"""

import pandas as pd
import pytest

from app.pipeline.categorizer import categorize, categorize_dataframe


class TestCategorize:
    """Tests for the categorize() function."""

    def test_zomato_maps_to_food(self):
        """A Zomato order should be categorised as Food."""
        assert categorize("Zomato Order #98123") == "Food"

    def test_swiggy_maps_to_food(self):
        """A Swiggy order should be categorised as Food."""
        assert categorize("Swiggy Lunch Delivery") == "Food"

    def test_ola_maps_to_transport(self):
        """An Ola cab booking should be categorised as Transport."""
        assert categorize("Ola Cab - Office Drop") == "Transport"

    def test_uber_maps_to_transport(self):
        """An Uber ride should be categorised as Transport."""
        assert categorize("Uber Cab - Airport") == "Transport"

    def test_netflix_maps_to_subscriptions(self):
        """A Netflix payment should be categorised as Subscriptions."""
        assert categorize("Netflix Subscription Monthly") == "Subscriptions"

    def test_salary_maps_to_income(self):
        """A salary credit should be categorised as Income."""
        assert categorize("Salary Credit - March 2024") == "Income"

    def test_electricity_maps_to_utilities(self):
        """An electricity bill should be categorised as Utilities."""
        assert categorize("Electricity Bill Payment") == "Utilities"

    def test_amazon_maps_to_shopping(self):
        """An Amazon purchase should be categorised as Shopping."""
        assert categorize("Amazon Order - Headphones") == "Shopping"

    def test_pharmacy_maps_to_health(self):
        """A pharmacy purchase should be categorised as Health."""
        assert categorize("Apollo Pharmacy Purchase") == "Health"

    def test_fixed_deposit_maps_to_savings(self):
        """An FD opening transfer should be categorised as Savings."""
        assert categorize("Fixed Deposit Opening Transfer") == "Savings"

    def test_unknown_description_returns_others(self):
        """A description with no matching keyword should return 'Others'."""
        assert categorize("Random Merchant XYZ 9999") == "Others"

    def test_empty_string_returns_others(self):
        """An empty string should return 'Others' without raising an error."""
        assert categorize("") == "Others"

    def test_case_insensitive_matching(self):
        """
        Matching should be case-insensitive.
        'ZOMATO' and 'zomato' and 'Zomato' should all map to Food.
        """
        assert categorize("ZOMATO NIGHT ORDER") == "Food"
        assert categorize("zomato evening") == "Food"
        assert categorize("Zomato Breakfast") == "Food"


class TestCategorizeDataframe:
    """Tests for the categorize_dataframe() function."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """A minimal DataFrame with two transactions."""
        return pd.DataFrame(
            {
                "date": ["2024-03-01", "2024-03-02"],
                "description": ["Salary Credit", "Zomato Order"],
                "amount": [50000, 350],
                "type": ["Credit", "Debit"],
            }
        )

    def test_adds_category_column(self, sample_df):
        """categorize_dataframe() should add a 'category' column."""
        result = categorize_dataframe(sample_df)
        assert "category" in result.columns

    def test_does_not_mutate_input(self, sample_df):
        """The original DataFrame should not be modified."""
        original_columns = list(sample_df.columns)
        categorize_dataframe(sample_df)
        assert list(sample_df.columns) == original_columns

    def test_categories_are_correct(self, sample_df):
        """Each row should be categorised correctly."""
        result = categorize_dataframe(sample_df)
        assert result.iloc[0]["category"] == "Income"
        assert result.iloc[1]["category"] == "Food"

    def test_row_count_unchanged(self, sample_df):
        """The number of rows should remain the same after categorization."""
        result = categorize_dataframe(sample_df)
        assert len(result) == len(sample_df)


class TestDebtPayments:
    """
    Loan servicing must be recognised by the rules layer.

    Production regression: "Personal Loan EMI Debit" reached the stage-2 LLM
    (no rule matched) and came back as "Income" — the word "Loan" reads like
    an inflow — so a debt repayment was counted as money coming in.
    """

    @pytest.mark.parametrize(
        "description",
        [
            "Personal Loan EMI Debit",
            "Home Loan EMI",
            "HDFC Loan Repayment",
            "Credit Card Minimum Payment",
            "Credit Card Bill Auto Debit",
            "Loan Installment Auto Pay",
        ],
    )
    def test_loan_servicing_is_categorized_by_rules(self, description):
        assert categorize(description) == "Debt Payments"

    def test_loan_disbursement_is_not_debt_payment(self):
        """An incoming loan is not a repayment; rules must not overreach."""
        assert categorize("Loan Disbursement Credit") != "Debt Payments"

    def test_debt_payments_is_classified_as_essential(self):
        """Servicing debt is a fixed obligation, so it counts as essential."""
        from app.pipeline.analyzer import (
            DISCRETIONARY_CATEGORIES,
            ESSENTIAL_CATEGORIES,
        )

        assert "Debt Payments" in ESSENTIAL_CATEGORIES
        assert "Debt Payments" not in DISCRETIONARY_CATEGORIES


class TestCreditOnlyGuard:
    """A debit can never be Income, whatever stage 2 decides."""

    def test_debit_labelled_income_by_llm_is_corrected(self, monkeypatch):
        import pandas as pd

        from app.core.config import settings
        from app.pipeline import categorizer as categorizer_module

        # Force stage 2 to run and to return the exact mistake seen in prod.
        monkeypatch.setattr(settings, "openai_api_key", "test-key")
        monkeypatch.setattr(
            categorizer_module,
            "batch_llm_categorize_others",
            lambda descs: {d: "Income" for d in descs},
        )

        df = pd.DataFrame(
            [
                ("2024-01-05", "Mystery Outflow XYZ", 14500.0, "Debit"),
                ("2024-01-06", "Mystery Inflow XYZ", 50000.0, "Credit"),
            ],
            columns=["date", "description", "amount", "type"],
        )
        out = categorizer_module.categorize_dataframe(df)

        # The debit is rescued; the credit is legitimately left as Income.
        assert out.loc[0, "category"] == "Others"
        assert out.loc[1, "category"] == "Income"
