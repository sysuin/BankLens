"""
Financial health metrics analyzer for BankLens.

Computes core banking financial health KPIs from a transactions DataFrame:
    - Total Income (sum of Credit transactions)
    - Total Expenses (sum of Debit transactions excluding internal Savings transfers)
    - Savings Amount (Total Income - Total Expenses)
    - Savings Rate % (Savings Amount / Total Income * 100)
    - Expense-to-Income Ratio (Total Expenses / Total Income)
    - Top 3 Spending Categories (by total debit outlay)
    - Period covered by statement (e.g. '2024-01 to 2024-12')
"""

from typing import Any
import pandas as pd
from pydantic import BaseModel

from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Output Data Model ─────────────────────────────────────────────────────────


class FinancialMetrics(BaseModel):
    """
    Validated financial metrics container passed downstream to RAG and LLM.

    Percentage fields are expressed on a 0–100 scale (e.g. 35.5 means 35.5%).
    """

    total_income: float
    total_expenses: float
    savings_amount: float
    savings_rate_pct: float
    expense_to_income_ratio: float

    # List of dicts: [{"category": "Food", "total_spent": 4500.0}, ...]
    top_categories: list[dict[str, Any]]

    transaction_count: int
    credit_count: int
    debit_count: int

    # Human-readable period string, e.g. "2024-03" or "2024-03 to 2024-05"
    period: str


# ── Main Function ─────────────────────────────────────────────────────────────


def compute_metrics(df: pd.DataFrame) -> FinancialMetrics:
    """
    Compute financial health metrics from a categorized transactions DataFrame.

    Args:
        df: A pandas DataFrame with columns:
                date        (str or datetime) — transaction date
                description (str)             — merchant / memo
                amount      (float)           — always positive
                type        (str)             — 'Credit' or 'Debit'
                category    (str)             — assigned by categorizer

    Returns:
        A FinancialMetrics Pydantic model with all computed values.

    Raises:
        ValueError: If required columns are missing or the DataFrame is empty.
    """
    # ── Validate input ────────────────────────────────────────────────────────
    required_columns = {"date", "description", "amount", "type", "category"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in uploaded CSV: {missing}. "
            "Expected: date, description, amount, type, category."
        )

    if df.empty:
        raise ValueError("The uploaded CSV contains no transactions.")

    # ── Split credits and debits ──────────────────────────────────────────────
    type_lower = df["type"].str.strip().str.lower()
    credits = df[type_lower == "credit"]
    debits = df[type_lower == "debit"]

    total_income = float(credits["amount"].sum())

    # Exclude internal savings/investment outlays from consumption expenses
    consumption_debits = debits[debits["category"] != "Savings"]
    total_expenses = (
        float(consumption_debits["amount"].sum())
        if not consumption_debits.empty
        else float(debits["amount"].sum())
    )
    savings_amount = total_income - total_expenses

    # Avoid division by zero for edge cases (e.g. statement with no income)
    if total_income > 0:
        savings_rate_pct = round(savings_amount / total_income * 100, 2)
        expense_to_income_ratio = round(total_expenses / total_income, 4)
    else:
        savings_rate_pct = 0.0
        expense_to_income_ratio = 0.0

    # ── Top 3 spending categories ─────────────────────────────────────────────
    category_totals = (
        debits.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .reset_index()
        .rename(columns={"amount": "total_spent"})
    )
    top_categories: list[dict[str, Any]] = category_totals.to_dict(orient="records")

    # ── Determine the period covered by the statement ─────────────────────────
    df_temp = df.copy()
    df_temp["date"] = pd.to_datetime(df_temp["date"], errors="coerce")
    min_month = df_temp["date"].min().strftime("%Y-%m")
    max_month = df_temp["date"].max().strftime("%Y-%m")
    period = min_month if min_month == max_month else f"{min_month} to {max_month}"

    metrics = FinancialMetrics(
        total_income=total_income,
        total_expenses=total_expenses,
        savings_amount=savings_amount,
        savings_rate_pct=savings_rate_pct,
        expense_to_income_ratio=expense_to_income_ratio,
        top_categories=top_categories,
        transaction_count=len(df),
        credit_count=len(credits),
        debit_count=len(debits),
        period=period,
    )

    logger.info(
        "Metrics computed | Period: %s | Income: %.2f | Expenses: %.2f | "
        "Savings Rate: %.2f%% | Expense Ratio: %.4f",
        period,
        total_income,
        total_expenses,
        savings_rate_pct,
        expense_to_income_ratio,
    )
    return metrics
