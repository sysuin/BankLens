"""
Reusable Streamlit UI components for BankLens.

Each function renders one self-contained UI block. Keeping UI rendering
separate from pipeline logic means:
    - Pipeline functions can be unit-tested without Streamlit running
    - UI blocks can be rearranged or restyled without touching business logic

Functions in this module:
    render_header()          — page title and tagline
    render_metric_cards()    — four KPI cards in a row
    render_transaction_table() — styled transaction DataFrame
    render_profile_card()    — customer persona and risk badge
    render_recommendation()  — product recommendation, reason, RM hook, RAG sources
"""

import pandas as pd
import streamlit as st

from app.pipeline.analyzer import FinancialMetrics
from app.pipeline.agent import CustomerProfile

# ── Colour Maps ───────────────────────────────────────────────────────────────

# Maps risk level to a colour used in the profile card badge
RISK_COLOURS: dict[str, str] = {
    "Low": "#22c55e",  # green
    "Medium": "#f59e0b",  # amber
    "High": "#ef4444",  # red
}


# ── Components ────────────────────────────────────────────────────────────────


def render_header() -> None:
    """Render the BankLens page header with title and tagline."""
    st.markdown(
        """
        <h1 style='text-align:center; color:#1e3a5f; font-size:2.5rem;'>
            🏦 BankLens
        </h1>
        <p style='text-align:center; color:#64748b; font-size:1.1rem;
                  margin-top:-0.5rem;'>
            AI-powered bank statement analysis &amp; product recommendation
        </p>
        <hr style='border:1px solid #e2e8f0; margin-top:1rem;'>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: FinancialMetrics) -> None:
    """
    Render four KPI metric cards in a single row.

    Displays income, expenses, savings amount, and savings rate so the
    user immediately sees the most important numbers from their statement.

    Args:
        metrics: The computed FinancialMetrics for the uploaded statement.
    """
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="💰 Total Income",
        value=f"₹{metrics.total_income:,.0f}",
    )
    col2.metric(
        label="💸 Total Expenses",
        value=f"₹{metrics.total_expenses:,.0f}",
    )
    col3.metric(
        label="🏦 Net Savings",
        value=f"₹{metrics.savings_amount:,.0f}",
        # Delta shows whether the customer saved (positive) or overspent (negative)
        delta=f"{metrics.savings_rate_pct:.1f}% of income",
        delta_color="normal" if metrics.savings_amount >= 0 else "inverse",
    )
    col4.metric(
        label="📊 Expense Ratio",
        value=f"{metrics.expense_to_income_ratio:.0%}",
        help=(
            "Total expenses divided by total income. "
            "Lower is healthier. Below 70% is considered good."
        ),
    )


def render_transaction_table(df: pd.DataFrame) -> None:
    """
    Render the categorized transactions as a styled Streamlit table.

    Formats the amount column as currency and displays only the
    columns relevant to the user.

    Args:
        df: The categorized transactions DataFrame with columns:
            date, description, amount, type, category.
    """
    display_df = df.copy()

    # Format amount as a readable currency string
    display_df["amount"] = display_df["amount"].apply(lambda x: f"₹{x:,.2f}")

    st.dataframe(
        display_df[["date", "description", "amount", "type", "category"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "description": "Description",
            "amount": "Amount",
            "type": st.column_config.TextColumn("Type"),
            "category": "Category",
        },
    )


def render_profile_card(profile: CustomerProfile) -> None:
    """
    Render the AI-generated customer profile summary card.

    Shows the financial persona label, a colour-coded risk badge,
    and the spending and savings insights from the LLM.

    Args:
        profile: The CustomerProfile Pydantic model from the LLM agent.
    """
    risk_colour = RISK_COLOURS.get(profile.risk_profile, "#94a3b8")

    st.markdown(
        f"""
        <div style='background:#f8fafc; border-radius:12px; padding:1.5rem;
                    border:1px solid #e2e8f0; margin-bottom:1rem;'>
            <div style='display:flex; align-items:center; gap:0.75rem;
                        flex-wrap:wrap;'>
                <h3 style='margin:0; color:#1e3a5f;'>
                    {profile.financial_persona}
                </h3>
                <span style='background:{risk_colour}; color:white;
                             font-size:0.75rem; font-weight:600;
                             padding:3px 12px; border-radius:20px;'>
                    {profile.risk_profile} Risk
                </span>
            </div>
            <p style='color:#475569; margin-top:0.75rem; margin-bottom:0.25rem;'>
                📈 <strong>Spending:</strong> {profile.spending_insight}
            </p>
            <p style='color:#475569; margin-bottom:0;'>
                💵 <strong>Savings:</strong> {profile.savings_insight}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation(profile: CustomerProfile) -> None:
    """
    Render the product recommendation, reason, RM hook, and RAG sources.

    This is the final output section of the app — the actionable
    intelligence that a Bank Relationship Manager would use.

    Args:
        profile: The CustomerProfile Pydantic model from the LLM agent.
    """
    # ── Recommended product ───────────────────────────────────────────────────
    st.markdown("### 🎯 Recommended Product")
    st.success(f"**{profile.recommended_product}**")
    st.markdown(profile.recommendation_reason)

    st.markdown("---")

    # ── RM opening line ───────────────────────────────────────────────────────
    st.markdown("### 💬 Relationship Manager Opening Line")
    st.caption(
        "Copy this line to open a conversation with the customer. "
        "It references a specific detail from their statement."
    )
    st.info(f'*"{profile.rm_hook}"*')

    st.markdown("---")

    # ── RAG transparency block ────────────────────────────────────────────────
    st.markdown("### 📚 Knowledge Base Sources")
    st.caption(
        "The recommendation above is grounded in the following documents "
        "retrieved from the banking product knowledge base via semantic search. "
        "No product outside this list was considered."
    )
    for source in profile.retrieved_sources:
        # Display each source as a monospace tag — clean and interview-friendly
        st.markdown(f"- `{source}`")
