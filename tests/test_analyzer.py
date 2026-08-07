"""
Unit tests for app.pipeline.analyzer.

Tests verify that:
    - Financial metrics are computed correctly from a known input
    - Edge cases (zero income, empty DataFrame, missing columns) raise errors
    - The savings rate and expense ratio formulas are correct
    - The top_categories list is ordered correctly by spend
    - The period field is set correctly for single-month and multi-month data
"""

import pandas as pd
import pytest

from app.pipeline.analyzer import compute_metrics, FinancialMetrics

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def standard_df() -> pd.DataFrame:
    """
    A DataFrame with known values so we can assert exact metric outputs.

    Income: 100,000 (one salary credit)
    Expenses: 40,000 (two debits: Food 10k + Transport 30k)
    Savings: 60,000
    Savings Rate: 60%
    Expense Ratio: 0.4
    """
    return pd.DataFrame(
        {
            "date": ["2024-03-01", "2024-03-15", "2024-03-20"],
            "description": ["Salary Credit", "Zomato Food", "Petrol Fuel"],
            "amount": [100000.0, 10000.0, 30000.0],
            "type": ["Credit", "Debit", "Debit"],
            "category": ["Income", "Food", "Transport"],
        }
    )


@pytest.fixture
def multi_month_df() -> pd.DataFrame:
    """A DataFrame that spans two months — to test the period field."""
    return pd.DataFrame(
        {
            "date": ["2024-03-01", "2024-04-01"],
            "description": ["Salary March", "Salary April"],
            "amount": [50000.0, 50000.0],
            "type": ["Credit", "Credit"],
            "category": ["Income", "Income"],
        }
    )


# ── Tests: correct computation ────────────────────────────────────────────────


class TestComputeMetrics:
    """Tests for the compute_metrics() function."""

    def test_returns_financial_metrics_instance(self, standard_df):
        """compute_metrics() should return a FinancialMetrics Pydantic model."""
        result = compute_metrics(standard_df)
        assert isinstance(result, FinancialMetrics)

    def test_total_income_is_correct(self, standard_df):
        """Total income should equal the sum of all Credit rows."""
        result = compute_metrics(standard_df)
        assert result.total_income == 100000.0

    def test_total_expenses_is_correct(self, standard_df):
        """Total expenses should equal the sum of all Debit rows."""
        result = compute_metrics(standard_df)
        assert result.total_expenses == 40000.0

    def test_savings_amount_is_correct(self, standard_df):
        """Savings amount should be income minus expenses."""
        result = compute_metrics(standard_df)
        assert result.savings_amount == 60000.0

    def test_savings_rate_is_correct(self, standard_df):
        """Savings rate should be 60% for this fixture."""
        result = compute_metrics(standard_df)
        assert result.savings_rate_pct == 60.0

    def test_expense_ratio_is_correct(self, standard_df):
        """Expense-to-income ratio should be 0.4 for this fixture."""
        result = compute_metrics(standard_df)
        assert result.expense_to_income_ratio == pytest.approx(0.4, rel=1e-3)

    def test_transaction_count(self, standard_df):
        """Transaction count should match the total rows in the DataFrame."""
        result = compute_metrics(standard_df)
        assert result.transaction_count == 3

    def test_credit_and_debit_counts(self, standard_df):
        """Credit and debit counts should split correctly."""
        result = compute_metrics(standard_df)
        assert result.credit_count == 1
        assert result.debit_count == 2

    def test_top_categories_ordered_by_spend(self, standard_df):
        """Top categories should be sorted descending by total amount spent."""
        result = compute_metrics(standard_df)
        # Transport (30k) should rank above Food (10k)
        assert result.top_categories[0]["category"] == "Transport"
        assert result.top_categories[1]["category"] == "Food"

    def test_single_month_period(self, standard_df):
        """Period should be a single month string when all dates are in March."""
        result = compute_metrics(standard_df)
        assert result.period == "2024-03"

    def test_multi_month_period(self, multi_month_df):
        """Period should show a range when data spans multiple months."""
        result = compute_metrics(multi_month_df)
        assert result.period == "2024-03 to 2024-04"


# ── Tests: edge cases ─────────────────────────────────────────────────────────


class TestComputeMetricsEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe_raises_value_error(self):
        """An empty DataFrame should raise a ValueError."""
        empty_df = pd.DataFrame(
            columns=["date", "description", "amount", "type", "category"]
        )
        with pytest.raises(ValueError, match="no transactions"):
            compute_metrics(empty_df)

    def test_missing_column_raises_value_error(self):
        """A DataFrame missing required columns should raise a ValueError."""
        bad_df = pd.DataFrame({"date": ["2024-03-01"], "amount": [100.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            compute_metrics(bad_df)

    def test_zero_income_does_not_raise(self):
        """
        A statement with no Credit rows should not raise a ZeroDivisionError.
        Savings rate and expense ratio should default to 0.0.
        """
        debit_only_df = pd.DataFrame(
            {
                "date": ["2024-03-01"],
                "description": ["Some Expense"],
                "amount": [5000.0],
                "type": ["Debit"],
                "category": ["Shopping"],
            }
        )
        result = compute_metrics(debit_only_df)
        assert result.total_income == 0.0
        assert result.savings_rate_pct == 0.0
        assert result.expense_to_income_ratio == 0.0
