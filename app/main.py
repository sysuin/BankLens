"""
BankLens — Main Streamlit Application.

This is the entry point for the app. It orchestrates the full pipeline:

    Step 1  Upload       — user uploads a bank statement CSV via sidebar
    Step 2  Parse        — CSV is read and validated for required columns
    Step 3  Categorize   — NLP keyword categorizer labels each transaction
    Step 4  Analyze      — financial metrics are computed from the DataFrame
    Step 5  Retrieve     — ChromaDB finds the most relevant product chunks
    Step 6  Profile      — LangChain + GPT-4o generates a CustomerProfile
    Step 7  Display      — results shown across three tabs in the Streamlit UI

Run this file with:
    streamlit run app/main.py
"""

import pandas as pd
import streamlit as st

from app.core.config import settings
from app.core.logger import get_logger
from app.pipeline.analyzer import compute_metrics
from app.pipeline.categorizer import categorize_dataframe
from app.pipeline.rag import build_vector_store, retrieve
from app.pipeline.agent import build_profile
from app.ui.charts import render_income_vs_expense, render_spending_by_category
from app.ui.components import (
    render_header,
    render_metric_cards,
    render_profile_card,
    render_recommendation,
    render_transaction_table,
)

logger = get_logger(__name__)

# ── Page config — must be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="BankLens | AI Bank Statement Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached resources ──────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading knowledge base into ChromaDB...")
def get_vector_store():
    """
    Build or load the ChromaDB vector store, cached for the session.

    @st.cache_resource ensures this runs only once per Streamlit session,
    not on every user interaction or file upload. The ChromaDB index is
    either loaded from disk (fast) or built from the knowledge base (once).

    Returns:
        A Chroma vector store instance ready for similarity search.
    """
    return build_vector_store()


# ── CSV loader with validation ────────────────────────────────────────────────


def load_and_validate_csv(uploaded_file) -> pd.DataFrame:
    """
    Parse the uploaded CSV file and validate it has the required columns.

    Column names are normalised to lowercase and stripped of whitespace so
    the app handles minor formatting differences in real-world exports.

    Args:
        uploaded_file: The Streamlit UploadedFile object from st.file_uploader.

    Returns:
        A clean pandas DataFrame with normalised column names.

    Raises:
        ValueError: If required columns are missing from the uploaded file.
    """
    df = pd.read_csv(uploaded_file)

    # Normalise column names: lowercase and strip whitespace
    df.columns = [col.strip().lower() for col in df.columns]

    required = {"date", "description", "amount", "type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Your CSV is missing these required columns: {missing}\n\n"
            "Expected columns: **date**, **description**, **amount**, **type** "
            "(Credit / Debit). Download the sample CSV from the sidebar to see "
            "the expected format."
        )

    return df


# ── Main controller ───────────────────────────────────────────────────────────


def main() -> None:
    """
    Main application controller.

    Renders the sidebar, handles file upload, runs the pipeline steps
    in order, and renders the results across three tabs.
    """
    render_header()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📂 Upload Bank Statement")
        st.caption(
            "Upload a CSV with columns: `date`, `description`, `amount`, `type`. "
            "The `type` column should contain **Credit** or **Debit**."
        )

        # Allow the user to download the sample CSV to try the app immediately
        try:
            with open("data/sample_statement.csv", "rb") as f:
                st.download_button(
                    label="⬇️ Download Sample CSV",
                    data=f,
                    file_name="sample_statement.csv",
                    mime="text/csv",
                    help="Download a 30-row demo statement to try the app right now.",
                )
        except FileNotFoundError:
            pass  # Sample CSV not critical — skip silently

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**🔧 Active Configuration**")
        st.markdown(
            f"- Model: `{settings.openai_model}`\n"
            f"- Embeddings: `{settings.openai_embedding_model}`\n"
            f"- Vector DB: ChromaDB (local)\n"
            f"- RAG k: `{settings.retrieval_k}`\n"
            f"- Chunk size: `{settings.chunk_size}`"
        )

    # ── No file uploaded yet ──────────────────────────────────────────────────
    if uploaded_file is None:
        st.info(
            "👈 **Upload a bank statement CSV** from the sidebar to get started.\n\n"
            "Download the sample CSV if you want to try the app immediately."
        )
        return

    # ── Step 2: Parse and validate CSV ────────────────────────────────────────
    try:
        raw_df = load_and_validate_csv(uploaded_file)
        logger.info(
            "CSV uploaded: %d rows, %d columns.",
            len(raw_df),
            len(raw_df.columns),
        )
    except ValueError as e:
        st.error(f"❌ **Invalid CSV format**\n\n{e}")
        return

    # ── Step 3: Categorize transactions ───────────────────────────────────────
    with st.spinner("Categorising transactions..."):
        categorized_df = categorize_dataframe(raw_df)

    # ── Step 4: Compute financial metrics ─────────────────────────────────────
    with st.spinner("Computing financial metrics..."):
        try:
            metrics = compute_metrics(categorized_df)
        except ValueError as e:
            st.error(f"❌ **Could not compute metrics**\n\n{e}")
            return

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        ["📋 Transactions", "📊 Financial Health", "🤖 AI Profile & Recommendation"]
    )

    # ── Tab 1: Transactions ───────────────────────────────────────────────────
    with tab1:
        st.markdown(
            f"**{metrics.transaction_count} transactions** found | "
            f"Period: `{metrics.period}` | "
            f"Credits: **{metrics.credit_count}** | "
            f"Debits: **{metrics.debit_count}**"
        )

        # Category filter dropdown
        all_categories = sorted(categorized_df["category"].unique().tolist())
        selected_category = st.selectbox(
            "Filter by category",
            options=["All"] + all_categories,
            help="Filter the table to show only transactions in a specific category.",
        )

        filtered_df = (
            categorized_df
            if selected_category == "All"
            else categorized_df[categorized_df["category"] == selected_category]
        )

        st.caption(f"Showing {len(filtered_df)} transactions.")
        render_transaction_table(filtered_df)

    # ── Tab 2: Financial Health ───────────────────────────────────────────────
    with tab2:
        render_metric_cards(metrics)
        st.markdown("---")

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            render_spending_by_category(categorized_df)
        with col_chart2:
            render_income_vs_expense(metrics.total_income, metrics.total_expenses)

        # Top categories breakdown table
        st.markdown("---")
        st.markdown("#### Top Spending Categories")
        if metrics.top_categories:
            top_df = pd.DataFrame(metrics.top_categories)
            top_df["total_spent"] = top_df["total_spent"].apply(lambda x: f"₹{x:,.2f}")
            st.dataframe(
                top_df.rename(
                    columns={"category": "Category", "total_spent": "Total Spent"}
                ),
                use_container_width=True,
                hide_index=True,
            )

    # ── Tab 3: AI Profile & Recommendation ───────────────────────────────────
    with tab3:
        st.markdown(
            "Click the button below to run the full AI pipeline:\n"
            "1. Your financial metrics are embedded and sent to ChromaDB\n"
            "2. The top 3 most relevant banking products are retrieved\n"
            "3. GPT-4o generates a customer profile grounded in the retrieved context"
        )

        if st.button(
            "🔍 Generate AI Profile",
            type="primary",
            use_container_width=True,
        ):
            # ── Step 5: RAG retrieval ─────────────────────────────────────────
            with st.spinner("Searching knowledge base for relevant products..."):
                try:
                    vector_store = get_vector_store()

                    # Build a natural language query from the computed metrics
                    query = (
                        f"Customer with monthly income ₹{metrics.total_income:,.0f}, "
                        f"expenses ₹{metrics.total_expenses:,.0f}, "
                        f"savings rate {metrics.savings_rate_pct:.1f}%, "
                        f"expense-to-income ratio "
                        f"{metrics.expense_to_income_ratio:.2f}. "
                        f"Top spending categories: "
                        f"{', '.join(c['category'] for c in metrics.top_categories)}."
                    )
                    retrieved_chunks = retrieve(query, vector_store)
                    logger.info(
                        "RAG retrieval complete. %d chunks retrieved.",
                        len(retrieved_chunks),
                    )
                except Exception as e:
                    logger.error("RAG retrieval failed: %s", e)
                    st.error(f"❌ **RAG retrieval failed:** {e}")
                    return

            # ── Step 6: LLM agent ─────────────────────────────────────────────
            with st.spinner("GPT-4o is building your customer profile..."):
                try:
                    profile = build_profile(metrics, retrieved_chunks)
                except Exception as e:
                    logger.error("Agent failed to build profile: %s", e)
                    st.error(f"❌ **Could not generate profile:** {e}")
                    return

            # ── Step 7: Display profile ───────────────────────────────────────
            render_profile_card(profile)
            st.markdown("---")
            render_recommendation(profile)


if __name__ == "__main__":
    main()
