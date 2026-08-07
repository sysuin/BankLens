"""
BankLens — Main Streamlit Application.

Orchestrates the complete BankLens AI pipeline:
    1. Dataset Selection / Upload (3 built-in statements or custom CSV)
    2. NLP Keyword Categorization (categorizer.py)
    3. Financial Health Analysis (analyzer.py)
    4. RAG Retrieval from ChromaDB Knowledge Base (rag.py)
    5. Customer Profiling & RM Pitch Synthesis with GPT-4o (agent.py)
    6. Interactive Multi-Tab Presentation (Streamlit UI)

Run locally:
    streamlit run app/main.py
"""

import textwrap
import time
import pandas as pd
import streamlit as st

from app.core.logger import get_logger
from app.pipeline.analyzer import compute_metrics
from app.pipeline.categorizer import categorize_dataframe
from app.pipeline.rag import build_vector_store, retrieve
from app.pipeline.agent import build_profile, CustomerProfile
from app.ui.charts import render_income_vs_expense, render_spending_by_category
from app.ui.components import (
    inject_custom_css,
    render_footer,
    render_header,
    render_metric_cards,
    render_profile_card,
    render_recommendation,
    render_transaction_table,
)

logger = get_logger(__name__)

# ── Page config — must be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="BankLens | AI Bank Statement Analyzer & Product Recommendation Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached resources ──────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading product knowledge base into ChromaDB...")
def get_vector_store():
    """Build or load the ChromaDB vector store, cached for the session."""
    return build_vector_store()


# ── Helper: CSV Loader & Validator ───────────────────────────────────────────


def load_and_validate_csv(file_or_path) -> pd.DataFrame:
    """Parse CSV and validate required columns: date, description, amount, type."""
    df = pd.read_csv(file_or_path)
    df.columns = [col.strip().lower() for col in df.columns]

    required = {"date", "description", "amount", "type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}\n"
            "Expected columns: date, description, amount, type (Credit/Debit)."
        )
    return df


# ── Helper: Run AI Pipeline with Stepper ──────────────────────────────────────


def run_ai_pipeline(metrics, categorized_df) -> CustomerProfile | None:
    """Execute the 6-step AI profiling pipeline with interactive progress status."""
    with st.status("⚙️ Executing BankLens GenAI Pipeline...", expanded=True) as status:
        # Step 1
        st.write(
            "🔹 **Step 1/6:** Ingesting & Categorizing Transactions via NLP (`categorizer.py`)..."
        )
        time.sleep(0.3)

        # Step 2
        st.write(
            "🔹 **Step 2/6:** Computing Financial Health Metrics & Savings Ratios (`analyzer.py`)..."
        )
        time.sleep(0.3)

        # Step 3
        st.write(
            f"🔹 **Step 3/6:** Vectorizing Search Query & Embedding (`text-embedding-3-small`)...\n"
            f"   * Query Metrics: Income ₹{metrics.total_income:,.0f}, Savings Rate {metrics.savings_rate_pct:.1f}%"
        )
        time.sleep(0.3)

        # Step 4
        st.write(
            "🔹 **Step 4/6:** Searching Product Knowledge Base Vector Store (`ChromaDB`)...\n"
            "   * Performing Cosine Similarity Search..."
        )
        try:
            vector_store = get_vector_store()
            query = (
                f"Customer with monthly income ₹{metrics.total_income:,.0f}, "
                f"expenses ₹{metrics.total_expenses:,.0f}, "
                f"savings rate {metrics.savings_rate_pct:.1f}%, "
                f"expense-to-income ratio {metrics.expense_to_income_ratio:.2f}. "
                f"Top spending categories: "
                f"{', '.join(c['category'] for c in metrics.top_categories)}."
            )
            retrieved_chunks = retrieve(query, vector_store)
            sources = [c["source"] for c in retrieved_chunks]
            st.write(
                f"   * ✅ Retrieved **{len(retrieved_chunks)} vector chunks** from sources: `{sources}`"
            )
        except Exception as e:
            status.update(label="❌ Pipeline Failed at Vector Search", state="error")
            st.error(f"Vector retrieval error: {e}")
            return None

        time.sleep(0.3)

        # Step 5
        st.write(
            "🔹 **Step 5/6:** Formatting Chat Prompt & Enforcing Guardrails (`PydanticOutputParser`)...\n"
            "   * Injecting credit default risk guardrails & schema constraints..."
        )
        time.sleep(0.3)

        # Step 6
        st.write(
            "🔹 **Step 6/6:** Synthesizing Multi-Product Offers & Pitch Talking Points (`OpenAI GPT-4o`)...\n"
            "   * Generating grounded response..."
        )
        try:
            profile = build_profile(metrics, retrieved_chunks)
        except Exception as e:
            status.update(label="❌ Pipeline Failed at LLM Synthesis", state="error")
            st.error(f"Profile synthesis error: {e}")
            return None

        status.update(
            label="✨ BankLens AI Pipeline Execution Complete!",
            state="complete",
            expanded=False,
        )
        return profile


# ── Main Application Controller ───────────────────────────────────────────────


def main() -> None:
    """Main application controller."""
    inject_custom_css()
    render_header()

    # Session state initialization for active tab navigation
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "profiler"
    if "ai_profile" not in st.session_state:
        st.session_state.ai_profile = None

    about_html = """
    **BankLens** is an enterprise AI financial intelligence platform designed for **Bank Relationship Managers (RMs)**.

    #### 🚩 The Problem:
    Relationship Managers manually review raw bank statements to identify sales opportunities and assess customer credit health. This manual process is time-consuming, prone to human bias, and often misses key financial signals.

    #### ⚡ The Solution:
    1. **Automated Ingestion & NLP Categorization:** BankLens ingests statement CSVs, auto-labeling every transaction (Income, Rent, Food, Utilities, Health, Education, etc.) using custom NLP keyword matching.
    2. **Financial Health Analytics:** Computes key financial metrics including Total Income, Total Expenses, Net Savings Rate (%), Expense-to-Income Ratio, and Top Spending Categories.
    3. **RAG Vector Search (ChromaDB):** Embeds the customer's financial health summary and performs semantic vector search across the bank's product knowledge base (`Fixed Deposit`, `Sweep Account`, `Credit Card`, `Personal Loan`, `Debt Consolidation Loan`).
    4. **Grounded AI Profiling (GPT-4o + Pydantic):** Synthesizes a structured customer profile, credit risk rating, grounded multi-product recommendations (Primary + Secondary), and structured RM pitch talking points — strictly protected against default risk & hallucinations by Pydantic guardrail schemas.
    """
    with st.expander(
        "💡 **What is BankLens & How It Works** (Click to expand)",
        expanded=False,
    ):
        st.markdown(textwrap.dedent(about_html))

    # ── Sidebar ───────────────────────────────────────────────────────────────
    sidebar_generate_clicked = False

    with st.sidebar:
        st.markdown("## 📊 Statement Selector")
        st.caption("Select a demo statement or upload your own CSV:")

        data_choice = st.radio(
            "Data Source",
            options=[
                "💼 Statement 1: High Income / Surplus Saver",
                "💳 Statement 2: Active Lifestyle Spender",
                "🚑 Statement 3: High Expense Pressure",
                "📁 Upload Custom CSV Statement",
            ],
            index=0,
            label_visibility="collapsed",
        )

        uploaded_file = None
        selected_csv_path = None

        if "Statement 1" in data_choice:
            selected_csv_path = "data/sample_1_high_saver.csv"
        elif "Statement 2" in data_choice:
            selected_csv_path = "data/sample_2_active_spender.csv"
        elif "Statement 3" in data_choice:
            selected_csv_path = "data/sample_3_cashflow_stressed.csv"
        else:
            uploaded_file = st.file_uploader(
                "Upload Statement CSV",
                type=["csv"],
                help="Columns needed: date, description, amount, type",
            )

        # ── Sidebar Action Button ─────────────────────────────────────────────
        if st.button(
            "🚀 Generate AI Profile",
            type="primary",
            use_container_width=True,
            key="sidebar_gen_btn",
        ):
            sidebar_generate_clicked = True
            st.session_state.active_tab = "profiler"

        st.markdown("---")

        # ── Project GitHub Link ───────────────────────────────────────────────
        st.markdown("### 🔗 Project Repository")
        github_html = """
        <a href='https://github.com/sysuin/BankLens' target='_blank' style='text-decoration:none;'>
            <div style='background:#1e293b; color:#38bdf8; padding:8px 12px; border-radius:8px; text-align:center; font-weight:600; border:1px solid #334155;'>
                ⭐ GitHub: sysuin/BankLens
            </div>
        </a>
        """
        st.markdown(textwrap.dedent(github_html), unsafe_allow_html=True)

        st.markdown("---")

        # ── Expanded Tech Stack Architecture ─────────────────────────────────
        st.markdown("### 🛠️ System Architecture")

        tech_html = """
        <div style='font-size:0.78rem; color:#cbd5e1;'>
            <p style='margin-bottom:2px;'><strong>🧠 AI &amp; LLM Orchestration</strong></p>
            <div style='display:flex; flex-wrap:wrap; gap:2px; margin-bottom:6px;'>
                <span class='tech-pill'>OpenAI GPT-4o</span>
                <span class='tech-pill'>LangChain LCEL</span>
                <span class='tech-pill'>Pydantic Guardrails</span>
            </div>
            <p style='margin-bottom:2px;'><strong>🗄️ RAG &amp; Vector Store</strong></p>
            <div style='display:flex; flex-wrap:wrap; gap:2px; margin-bottom:6px;'>
                <span class='tech-pill'>ChromaDB (Persisted)</span>
                <span class='tech-pill'>text-embedding-3-small</span>
                <span class='tech-pill'>TextSplitter Chunking</span>
            </div>
            <p style='margin-bottom:2px;'><strong>📊 Analytics &amp; Frontend</strong></p>
            <div style='display:flex; flex-wrap:wrap; gap:2px; margin-bottom:6px;'>
                <span class='tech-pill'>Python 3.11</span>
                <span class='tech-pill'>pandas Engine</span>
                <span class='tech-pill'>Streamlit Framework</span>
                <span class='tech-pill'>Custom Glassmorphic CSS</span>
            </div>
            <p style='margin-bottom:2px;'><strong>🐳 DevOps &amp; Cloud</strong></p>
            <div style='display:flex; flex-wrap:wrap; gap:2px;'>
                <span class='tech-pill'>Docker Multi-stage</span>
                <span class='tech-pill'>AWS EC2 Linux 2023</span>
                <span class='tech-pill'>Amazon ECR</span>
                <span class='tech-pill'>Nginx Reverse Proxy</span>
                <span class='tech-pill'>Certbot SSL (HTTPS)</span>
                <span class='tech-pill'>GitHub Actions CI/CD</span>
            </div>
        </div>
        """
        st.markdown(textwrap.dedent(tech_html), unsafe_allow_html=True)

        st.markdown("---")
        st.caption("BankLens Enterprise v1.0 | RAG-Grounded Intelligence")

    # ── Load Data ─────────────────────────────────────────────────────────────
    if selected_csv_path:
        raw_df = load_and_validate_csv(selected_csv_path)
    elif uploaded_file:
        try:
            raw_df = load_and_validate_csv(uploaded_file)
        except ValueError as e:
            st.error(f"❌ **Invalid CSV format:** {e}")
            return
    else:
        st.info("👈 Please select a Statement or upload a CSV from the left sidebar.")
        return

    # ── Pipeline Step 1 & 2: Categorization & Analytics ───────────────────────
    categorized_df = categorize_dataframe(raw_df)

    try:
        metrics = compute_metrics(categorized_df)
    except ValueError as e:
        st.error(f"❌ **Error computing financial metrics:** {e}")
        return

    # ── Custom Styled Tab Navigation Bar ─────────────────────────────────────
    col_nav1, col_nav2, col_nav3 = st.columns(3)

    with col_nav1:
        if st.button(
            "📋 Transaction Ledger",
            use_container_width=True,
            type=(
                "primary" if st.session_state.active_tab == "ledger" else "secondary"
            ),
        ):
            st.session_state.active_tab = "ledger"
            st.rerun()

    with col_nav2:
        if st.button(
            "📊 Financial Analytics",
            use_container_width=True,
            type=(
                "primary" if st.session_state.active_tab == "analytics" else "secondary"
            ),
        ):
            st.session_state.active_tab = "analytics"
            st.rerun()

    with col_nav3:
        if st.button(
            "🤖 AI Profiler & RAG Pitch",
            use_container_width=True,
            type=(
                "primary" if st.session_state.active_tab == "profiler" else "secondary"
            ),
        ):
            st.session_state.active_tab = "profiler"
            st.rerun()

    st.markdown(
        "<hr style='margin-top:0.2rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True
    )

    # ── View 1: Transaction Ledger ───────────────────────────────────────────
    if st.session_state.active_tab == "ledger":
        st.markdown(
            f"**{metrics.transaction_count} ledger entries** | "
            f"Period: `{metrics.period}` | "
            f"Credits: **{metrics.credit_count}** | "
            f"Debits: **{metrics.debit_count}**"
        )

        all_categories = sorted(categorized_df["category"].unique().tolist())
        selected_category = st.selectbox(
            "Filter transactions by category",
            options=["All Categories"] + all_categories,
        )

        filtered_df = (
            categorized_df
            if selected_category == "All Categories"
            else categorized_df[categorized_df["category"] == selected_category]
        )

        render_transaction_table(filtered_df)

    # ── View 2: Financial Analytics ───────────────────────────────────────────
    elif st.session_state.active_tab == "analytics":
        render_metric_cards(metrics)
        st.markdown("---")

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            render_spending_by_category(categorized_df)
        with col_chart2:
            render_income_vs_expense(metrics.total_income, metrics.total_expenses)

        st.markdown("---")
        st.markdown("#### 🔝 Top Spending Categories")
        if metrics.top_categories:
            top_df = pd.DataFrame(metrics.top_categories)
            top_df["total_spent"] = top_df["total_spent"].apply(lambda x: f"₹{x:,.2f}")
            st.dataframe(
                top_df.rename(
                    columns={
                        "category": "Category",
                        "total_spent": "Total Spent",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    # ── View 3: AI Profiling & RAG Product Pitch ───────────────────────────────
    elif st.session_state.active_tab == "profiler":
        st.markdown(
            "Click **Generate AI Customer Profile** below to execute the live multi-stage GenAI pipeline:"
        )

        tab_generate_clicked = st.button(
            "🚀 Generate AI Customer Profile & Pitch",
            type="primary",
            use_container_width=True,
            key="tab3_generate_btn",
        )

        if sidebar_generate_clicked or tab_generate_clicked:
            st.session_state.ai_profile = run_ai_pipeline(metrics, categorized_df)

        if st.session_state.ai_profile:
            prof = st.session_state.ai_profile
            render_profile_card(prof)
            st.markdown("---")
            render_recommendation(prof)

    # Render application footer
    render_footer()


if __name__ == "__main__":
    main()
