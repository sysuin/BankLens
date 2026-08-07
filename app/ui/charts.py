"""
Chart rendering functions for BankLens.

Wraps Streamlit's native charting so that chart logic stays out of main.py
and can be updated independently of the application layout.
"""

import pandas as pd
import streamlit as st


def render_spending_by_category(df: pd.DataFrame) -> None:
    """
    Render a bar chart of total spending grouped by category.

    Only Debit transactions are included. Categories are sorted in
    descending order so the highest spend appears at the top.

    Args:
        df: The categorized transactions DataFrame with columns:
            amount (float), type (str), category (str).
    """
    debits = df[df["type"].str.strip().str.lower() == "debit"]

    if debits.empty:
        st.info("No debit transactions found to chart.")
        return

    category_totals = (
        debits.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"category": "Category", "amount": "Total Spent (₹)"})
        .set_index("Category")
    )

    st.markdown("#### Spending by Category")
    st.bar_chart(category_totals, use_container_width=True)


def render_income_vs_expense(total_income: float, total_expenses: float) -> None:
    """
    Render an income vs expense comparison using Streamlit metric widgets.

    Streamlit does not natively support pie charts, so this renders
    a clear two-column metric comparison as a readable alternative.

    Args:
        total_income: Total sum of all Credit transactions.
        total_expenses: Total sum of all Debit transactions.
    """
    st.markdown("#### Income vs Expenses")

    total = total_income + total_expenses
    income_pct = (total_income / total * 100) if total > 0 else 0.0
    expense_pct = (total_expenses / total * 100) if total > 0 else 0.0

    col1, col2 = st.columns(2)
    col1.metric(
        label="💚 Income",
        value=f"₹{total_income:,.0f}",
        delta=f"{income_pct:.1f}% of total flow",
    )
    col2.metric(
        label="🔴 Expenses",
        value=f"₹{total_expenses:,.0f}",
        delta=f"{expense_pct:.1f}% of total flow",
        delta_color="inverse",
    )
