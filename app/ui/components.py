"""
Reusable Streamlit UI components for BankLens.

Renders polished UI blocks with enterprise-grade styling:
    - inject_custom_css()      — master CSS for font, gradients, light mode lock, dark sidebar, mobile responsiveness
    - render_header()          — sleek header with gradient title & live badge
    - render_metric_cards()    — formatted KPI cards
    - render_transaction_table() — styled DataFrame display
    - render_profile_card()    — customer persona, health score & risk badge
    - render_recommendation()  — Primary & Secondary products, bulleted RM hooks, RAG sources
    - render_footer()          — clean footer with contact info
"""

import textwrap
import pandas as pd
import streamlit as st

from app.pipeline.analyzer import FinancialMetrics
from app.pipeline.agent import CustomerProfile

# ── Colour Maps ───────────────────────────────────────────────────────────────

RISK_COLOURS: dict[str, str] = {
    "Low": "#10b981",  # emerald green
    "Medium": "#f59e0b",  # amber
    "High": "#ef4444",  # rose red
}


def inject_custom_css() -> None:
    """Inject custom CSS rules for a premium enterprise fintech aesthetic locked in Light Mode."""
    st.markdown(
        textwrap.dedent("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Force Main Container Light Mode */
        [data-testid="stAppViewContainer"], .stApp {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }

        [data-testid="stHeader"] {
            background-color: rgba(255, 255, 255, 0.9) !important;
        }

        /* Main Workspace Typography */
        .stApp p, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp li {
            color: #0f172a;
        }

        /* Metric card styling */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: #0f172a !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            color: #475569 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Glass card container */
        .glass-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
            margin-bottom: 1.25rem;
        }

        /* Sidebar compact spacing & dark theme styling */
        [data-testid="stSidebar"] {
            background-color: #0f172a !important;
            color: #f8fafc !important;
            padding-top: 1rem !important;
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stRadio label {
            color: #cbd5e1 !important;
            font-weight: 600;
        }

        /* Dark file uploader styling inside sidebar */
        [data-testid="stFileUploader"] {
            background-color: #1e293b !important;
            border: 1px dashed #475569 !important;
            border-radius: 10px !important;
            padding: 0.75rem !important;
        }

        [data-testid="stFileUploader"] label {
            color: #f8fafc !important;
            font-weight: 600 !important;
        }

        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] [data-testid="stCaptionContainer"],
        [data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] p,
        [data-testid="stFileUploader"] span {
            color: #94a3b8 !important;
            font-size: 0.78rem !important;
        }

        [data-testid="stFileUploader"] button {
            background-color: #334155 !important;
            color: #ffffff !important;
            border: 1px solid #475569 !important;
            border-radius: 8px !important;
        }

        /* Compact sidebar element spacing */
        [data-testid="stSidebar"] .element-container {
            margin-bottom: 0.35rem !important;
        }

        [data-testid="stSidebar"] hr {
            margin: 0.5rem 0 !important;
            border-color: #334155 !important;
        }

        /* Tech badge pill */
        .tech-pill {
            display: inline-flex;
            align-items: center;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #38bdf8 !important;
            padding: 3px 9px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            margin: 2px;
        }

        .stButton>button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease-in-out !important;
        }

        /* Clean footer styling */
        .app-footer {
            text-align: center;
            padding: 1.5rem 0 1rem 0;
            color: #64748b;
            font-size: 0.88rem;
            border-top: 1px solid #e2e8f0;
            margin-top: 2rem;
        }

        .app-footer a {
            color: #2563eb;
            text-decoration: none;
            font-weight: 600;
        }

        /* Mobile Device Responsiveness */
        @media (max-width: 768px) {
            .grid-3col {
                grid-template-columns: 1fr !important;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.4rem !important;
            }
        }
        </style>
        """),
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the main header banner with title and subtitle."""
    st.markdown(
        textwrap.dedent("""
        <div style='text-align: center; padding: 0.5rem 0 1.2rem 0;'>
            <div style='display: inline-flex; align-items: center; gap: 8px; background: #e0f2fe; color: #0284c7; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.75rem;'>
                ⚡ Enterprise AI Financial Intelligence Platform
            </div>
            <h1 style='color: #0f172a; font-weight: 800; font-size: 2.6rem; margin: 0; letter-spacing: -0.02em;'>
                🏦 BankLens
            </h1>
            <p style='color: #64748b; font-size: 1.1rem; margin-top: 0.4rem; font-weight: 400;'>
                Automated Financial Profiling, RAG Multi-Product Pitch Engine &amp; Default Guardrails
            </p>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: FinancialMetrics) -> None:
    """Render 4 formatted metric cards in a single row."""
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
        delta=f"{metrics.savings_rate_pct:.1f}% savings rate",
        delta_color="normal" if metrics.savings_amount >= 0 else "inverse",
    )
    col4.metric(
        label="📊 Expense Ratio",
        value=f"{metrics.expense_to_income_ratio:.0%}",
        help="Total Expenses ÷ Total Income. Below 70% is considered healthy.",
    )


def render_transaction_table(df: pd.DataFrame) -> None:
    """Render categorized transactions as a clean interactive dataframe."""
    display_df = df.copy()
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
            "category": st.column_config.TextColumn("Category"),
        },
    )


def render_profile_card(profile: CustomerProfile) -> None:
    """Render the AI customer persona, health score & risk profile summary card."""
    risk_colour = RISK_COLOURS.get(profile.risk_profile, "#94a3b8")
    score = profile.financial_health_score
    score_color = (
        "#10b981" if score >= 75 else ("#f59e0b" if score >= 50 else "#ef4444")
    )

    card_html = f"""
    <div class='glass-card'>
        <div style='display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 1rem; margin-bottom: 1rem;'>
            <div>
                <span style='font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em; color:#64748b; font-weight:700;'>Customer Archetype</span>
                <h2 style='margin:0; color:#0f172a; font-weight:800; font-size:1.7rem;'>
                    👤 {profile.financial_persona}
                </h2>
            </div>
            <div style='display:flex; gap:12px; align-items:center;'>
                <div style='background:#f1f5f9; border:1px solid #cbd5e1; padding:6px 16px; border-radius:30px; text-align:center;'>
                    <span style='font-size:0.75rem; color:#64748b; font-weight:700; text-transform:uppercase;'>Health Score</span>
                    <div style='font-size:1.2rem; font-weight:800; color:{score_color};'>{score}/100</div>
                </div>
                <div style='background:{risk_colour}; color:white; font-size:0.85rem; font-weight:700; padding:10px 18px; border-radius:30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                    🛡️ {profile.risk_profile} Risk Profile
                </div>
            </div>
        </div>
        <div class='grid-3col' style='display:grid; grid-template-columns: 1fr 1fr 1fr; gap:1rem; margin-top:1rem;'>
            <div style='background:#ffffff; padding:1rem; border-radius:12px; border:1px solid #e2e8f0;'>
                <span style='font-weight:700; color:#0284c7; font-size:0.88rem;'>📈 Income Stability</span>
                <p style='color:#334155; margin-top:0.4rem; margin-bottom:0; font-size:0.9rem; line-height:1.5;'>
                    {profile.income_stability_analysis}
                </p>
            </div>
            <div style='background:#ffffff; padding:1rem; border-radius:12px; border:1px solid #e2e8f0;'>
                <span style='font-weight:700; color:#059669; font-size:0.88rem;'>🛒 Spending Breakdown</span>
                <p style='color:#334155; margin-top:0.4rem; margin-bottom:0; font-size:0.9rem; line-height:1.5;'>
                    {profile.spending_pattern_breakdown}
                </p>
            </div>
            <div style='background:#ffffff; padding:1rem; border-radius:12px; border:1px solid #e2e8f0;'>
                <span style='font-weight:700; color:#d97706; font-size:0.88rem;'>⚠️ Credit Risk Rationale</span>
                <p style='color:#334155; margin-top:0.4rem; margin-bottom:0; font-size:0.9rem; line-height:1.5;'>
                    {profile.credit_risk_assessment}
                </p>
            </div>
        </div>
    </div>
    """
    st.markdown(textwrap.dedent(card_html), unsafe_allow_html=True)


def render_recommendation(profile: CustomerProfile) -> None:
    """Render Primary & Secondary product pitch cards, bulleted RM hooks, and RAG sources."""
    st.markdown("### 🎯 AI Multi-Product Pitch Recommendation")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        card1_html = f"""
        <div style='background: #eff6ff; border: 2px solid #2563eb; border-radius: 14px; padding: 1.25rem; height: 100%;'>
            <div style='background:#2563eb; color:white; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; display:inline-block; margin-bottom:0.6rem;'>
                🥇 PRIMARY PRODUCT OFFER
            </div>
            <h3 style='color:#1e3a5f; margin:0 0 0.5rem 0; font-weight:800;'>
                🏆 {profile.primary_product}
            </h3>
            <p style='color:#334155; font-size:0.95rem; line-height:1.55; margin:0;'>
                {profile.primary_reason}
            </p>
        </div>
        """
        st.markdown(textwrap.dedent(card1_html), unsafe_allow_html=True)

    with col_p2:
        card2_html = f"""
        <div style='background: #f0fdf4; border: 2px solid #16a34a; border-radius: 14px; padding: 1.25rem; height: 100%;'>
            <div style='background:#16a34a; color:white; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; display:inline-block; margin-bottom:0.6rem;'>
                🥈 SECONDARY CROSS-SELL OFFER
            </div>
            <h3 style='color:#14532d; margin:0 0 0.5rem 0; font-weight:800;'>
                💎 {profile.secondary_product}
            </h3>
            <p style='color:#334155; font-size:0.95rem; line-height:1.55; margin:0;'>
                {profile.secondary_reason}
            </p>
        </div>
        """
        st.markdown(textwrap.dedent(card2_html), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Relationship Manager Bulleted Call Pitch ──────────────────────────────
    col_hook_title, col_download = st.columns([3, 1])
    with col_hook_title:
        st.markdown("### 💬 Relationship Manager Structured Pitch Talking Points")
        st.caption("Copy these structured bullet points during your client call:")

    # Generate text format for download
    summary_txt = f"""BANKLENS AI CUSTOMER PROFILE REPORT
===================================
Persona: {profile.financial_persona}
Health Score: {profile.financial_health_score}/100
Risk Profile: {profile.risk_profile} Risk

RECOMMENDED PRODUCTS:
1. Primary Offer: {profile.primary_product}
   Reason: {profile.primary_reason}

2. Secondary Offer: {profile.secondary_product}
   Reason: {profile.secondary_reason}

TALKING POINTS FOR RM CALL:
- {profile.rm_hook_points[0] if len(profile.rm_hook_points) > 0 else ''}
- {profile.rm_hook_points[1] if len(profile.rm_hook_points) > 1 else ''}
- {profile.rm_hook_points[2] if len(profile.rm_hook_points) > 2 else ''}

RESOURCES: {', '.join(profile.retrieved_sources)}
"""

    with col_download:
        st.download_button(
            label="📥 Download Pitch Summary (.txt)",
            data=summary_txt,
            file_name="banklens_rm_pitch_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

    points_html = "".join(
        [
            f"<li style='margin-bottom: 0.6rem; color: #1e293b; font-size: 0.96rem; line-height: 1.5;'>{pt}</li>"
            for pt in profile.rm_hook_points
        ]
    )

    hook_box_html = f"""
    <div style='background: #f8fafc; border-left: 5px solid #0284c7; border-radius: 8px; padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;'>
        <ul style='margin: 0; padding-left: 1.2rem;'>
            {points_html}
        </ul>
    </div>
    """
    st.markdown(textwrap.dedent(hook_box_html), unsafe_allow_html=True)

    st.markdown("---")

    # ── RAG Grounding Sources ────────────────────────────────────────────────
    st.markdown("### 📚 Grounded Knowledge Base Sources (RAG Transparency)")
    st.caption(
        "Semantic search retrieved the following vector chunks from ChromaDB to ground this recommendation:"
    )

    cols = st.columns(len(profile.retrieved_sources))
    for idx, source in enumerate(profile.retrieved_sources):
        with cols[idx % len(cols)]:
            tag_html = f"""
            <div style='background:#f1f5f9; border:1px solid #cbd5e1; padding:0.6rem 1rem; border-radius:8px; text-align:center;'>
                📄 <code style='font-weight:600; color:#0f172a;'>{source}</code>
            </div>
            """
            st.markdown(textwrap.dedent(tag_html), unsafe_allow_html=True)


def render_footer() -> None:
    """Render the application footer with author and website details."""
    footer_html = """
    <div class='app-footer'>
        <p>
            Developed by <strong>Sunny Singh</strong> |
            🌐 <a href='https://sysuin.com' target='_blank'>sysuin.com</a> |
            ✉️ <a href='mailto:sunnysinghnitb@gmail.com'>sunnysinghnitb@gmail.com</a>
        </p>
    </div>
    """
    st.markdown(textwrap.dedent(footer_html), unsafe_allow_html=True)
