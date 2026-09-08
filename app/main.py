"""
BankLens — Main Streamlit Enterprise Application v2.0.

Orchestrates the complete BankLens Enterprise GenAI pipeline:
    1. Dual Dataset Selection / Upload (CSV & PDF bank statement parsing)
    2. Data Privacy & PII Sanitization Guard (Account & ID Masking)
    3. 2-Stage Hybrid NLP Transaction Categorization (categorizer.py)
    4. Financial Health Analysis & Metrics Computation (analyzer.py)
    5. High Performance Hybrid RAG Retrieval (ChromaDB + BM25 Ensemble RRF)
    6. Grounded AI Profiling & Pitch Synthesis with GPT-4o & Pydantic Guardrails (agent.py)
    7. Interactive Multi-Tab Presentation & Executive Report Export (Streamlit UI)

Run locally:
    streamlit run app/main.py
"""

import textwrap
import time
import pandas as pd
import streamlit as st

from app.core.config import settings
from app.core.logger import get_logger
from app.pipeline.analyzer import compute_metrics
from app.pipeline.categorizer import categorize_dataframe
from app.pipeline.pdf_parser import parse_pdf_statement
from app.pipeline.sanitizer import sanitize_dataframe
from app.pipeline.rag import build_retrieval_query, build_vector_store, retrieve
from app.pipeline.agent import CustomerProfile
from app.pipeline.cache import cached_build_profile
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


@st.cache_resource(
    show_spinner="Loading product knowledge base into ChromaDB & BM25..."
)
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
    with st.status(
        "⚙️ Executing BankLens Enterprise GenAI Pipeline...", expanded=True
    ) as status:
        # Step 1
        st.write(
            "🔹 **Step 1/6:** Ingesting & Categorizing Transactions via 2-Stage Hybrid NLP (`categorizer.py`)..."
        )
        time.sleep(0.3)

        # Step 2
        st.write(
            "🔹 **Step 2/6:** Computing Financial Health Metrics & Savings Ratios (`analyzer.py`)..."
        )
        time.sleep(0.3)

        # Step 3
        st.write(
            f"🔹 **Step 3/6:** Vectorizing Search Query & Embeddings (`text-embedding-3-small`)...\n"
            f"   * Query Metrics: Income ₹{metrics.total_income:,.0f}, Savings Rate {metrics.savings_rate_pct:.1f}%"
        )
        time.sleep(0.3)

        # Step 4
        st.write(
            "🔹 **Step 4/6:** Searching Product Knowledge Base via Fast Hybrid RAG (`ChromaDB + BM25 Ensemble`)...\n"
            "   * Executing ~20ms Reciprocal Rank Fusion (RRF) Hybrid Search..."
        )
        try:
            vector_store = get_vector_store()
            # Query construction lives in rag.py so the evaluation harness
            # measures the same retrieval path this screen uses.
            retrieved_chunks = retrieve(build_retrieval_query(metrics), vector_store)
            sources = [c["source"] for c in retrieved_chunks]
            st.write(
                f"   * ✅ Retrieved **{len(retrieved_chunks)} Hybrid RAG chunks** from sources: `{sources}`"
            )
        except Exception as e:
            status.update(label="❌ Pipeline Failed at Vector Search", state="error")
            st.error(f"Vector retrieval error: {e}")
            return None

        time.sleep(0.3)

        # Step 5
        st.write(
            "🔹 **Step 5/6:** Applying Data Privacy PII Masking & Prompt Injection Boundaries...\n"
            "   * Account & ID numbers masked. Encapsulating inputs inside XML tags..."
        )
        time.sleep(0.3)

        # Step 6
        st.write(
            "🔹 **Step 6/6:** Synthesizing Multi-Product Offers & Pitch Talking Points (`OpenAI GPT-4o`)...\n"
            "   * Generating grounded response with fallback retry protection..."
        )
        try:
            profile, from_cache = cached_build_profile(metrics, retrieved_chunks)
            if from_cache:
                st.write(
                    "   * ⚡ Served from **response cache** — identical inputs, "
                    "no LLM call spent"
                )
        except Exception as e:
            status.update(label="❌ Pipeline Failed at LLM Synthesis", state="error")
            st.error(f"Profile synthesis error: {e}")
            return None

        status.update(
            label="✨ BankLens Enterprise AI Pipeline Execution Complete!",
            state="complete",
            expanded=False,
        )
        return profile


# ── Statement views (shared by direct mode and API mode) ─────────────────────


def render_statement_views(
    metrics,
    categorized_df,
    *,
    generate_profile,
    chat_stream,
    sidebar_generate_clicked: bool,
    allow_generate: bool = True,
) -> None:
    """
    The four views over one analysed statement.

    `generate_profile()` returns a CustomerProfile (or None on failure) and
    `chat_stream(question)` yields answer tokens. Direct mode passes the
    pipeline functions; API mode passes HTTP calls. The views do not know
    which.
    """
    # ── Custom Styled Tab Navigation Bar ─────────────────────────────────────
    # The wrapper div lets CSS style the whole row as one segmented control
    # rather than four loose buttons (see .bl-navwrap in components.py).
    st.markdown("<div class='bl-navwrap'>", unsafe_allow_html=True)
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)

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

    with col_nav4:
        if st.button(
            "💬 Ask BankLens",
            use_container_width=True,
            type=("primary" if st.session_state.active_tab == "chat" else "secondary"),
        ):
            st.session_state.active_tab = "chat"
            st.rerun()

    st.markdown(
        "<hr style='margin-top:0.2rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True
    )

    # ── View 1: Transaction Ledger ───────────────────────────────────────────
    if st.session_state.active_tab == "ledger":
        st.success(
            "🔒 **PII Privacy Guard Active:** Account & Sensitive ID numbers masked."
        )
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

        tab_generate_clicked = allow_generate and st.button(
            "🚀 Generate AI Customer Profile & Pitch",
            type="primary",
            use_container_width=True,
            key="tab3_generate_btn",
        )
        if not allow_generate:
            st.info(
                "Reviewers read profiles; generating one is a Relationship Manager action."
            )

        if sidebar_generate_clicked or tab_generate_clicked:
            st.session_state.ai_profile = generate_profile()

        if st.session_state.ai_profile:
            prof = st.session_state.ai_profile
            render_profile_card(prof)
            st.markdown("---")
            render_recommendation(prof)

    # Render application footer
    # ── View 4: Ask BankLens (agentic chat over the analyzed statement) ──────
    elif st.session_state.active_tab == "chat":
        st.markdown("#### 💬 Ask BankLens about this statement")
        st.caption(
            "A tool-calling assistant: it can fetch the computed metrics, search "
            "the product knowledge base, and inspect category spending. Answers "
            "stream in live."
        )

        # History is keyed to the loaded statement so switching statements
        # cannot leak one customer's conversation into another's.
        statement_key = st.session_state.get("loaded_statement_key")
        current_key = (
            f"{metrics.period}|{metrics.transaction_count}|{metrics.total_income}"
        )
        if statement_key != current_key:
            st.session_state.chat_history = []
            st.session_state.loaded_statement_key = current_key

        from langchain_core.messages import AIMessage as _AI

        for message in st.session_state.chat_history:
            role = "assistant" if isinstance(message, _AI) else "user"
            with st.chat_message(role):
                st.markdown(message.content)

        if question := st.chat_input("e.g. Why is this customer rated Low risk?"):
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    st.write_stream(chat_stream(question))
                except Exception as e:
                    st.error(f"Chat error: {e}")


def _direct_chat_stream(question: str, metrics, categorized_df):
    """Direct mode: run the tool-calling chat in-process."""
    from app.pipeline.chat import run_chat_turn

    return run_chat_turn(
        question, st.session_state.chat_history, metrics, categorized_df
    )


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
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    about_html = """
    **BankLens Enterprise v2.0** is a commercial-grade AI financial intelligence platform designed for **Bank Relationship Managers (RMs)**.

    #### 🚩 The Problem:
    Relationship Managers manually review raw bank statements to identify sales opportunities and assess customer credit health. This manual process is time-consuming, prone to human bias, and often misses key financial signals.

    #### ⚡ The Solution:
    1. **Dual PDF & CSV Ingestion + PII Sanitization:** Ingests bank statements in CSV or PDF format with automatic PII masking (Account/ID masking) for GDPR/RBI compliance.
    2. **2-Stage Hybrid Categorization:** Auto-labels transactions using fast keyword rules + LLM batch classification for unknown merchant names.
    3. **Financial Health Analytics:** Computes Net Savings Rate (%), Expense-to-Income Ratio, and Top Spending Categories.
    4. **Fast Hybrid RAG Vector Search:** Combines Dense Embeddings (ChromaDB) + Lexical Keyword Search (BM25) with cached indexing for ~20ms search speed.
    5. **Grounded AI Profiling:** Synthesizes structured customer profiles, credit risk ratings, and RM call scripts using GPT-4o reasoning, prompt injection boundaries, & Pydantic output guardrails.
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
        st.caption("Select a demo statement or upload your own CSV/PDF:")

        data_choice = st.radio(
            "Data Source",
            options=[
                "💼 Statement 1 (CSV): High Income / Surplus Saver",
                "💳 Statement 2 (CSV): Active Lifestyle Spender",
                "🚑 Statement 3 (PDF): High Expense Pressure",
                "📁 Upload Custom CSV / PDF Statement",
            ],
            index=0,
            label_visibility="collapsed",
        )

        uploaded_file = None
        selected_file_path = None

        if "Statement 1" in data_choice:
            selected_file_path = "data/sample_1_high_saver.csv"
        elif "Statement 2" in data_choice:
            selected_file_path = "data/sample_2_active_spender.csv"
        elif "Statement 3" in data_choice:
            selected_file_path = "data/sample_3_cashflow_stressed.pdf"
        else:
            uploaded_file = st.file_uploader(
                "Upload Statement (CSV or PDF)",
                type=["csv", "pdf"],
                help="Columns needed for CSV/PDF: date, description, amount, type",
            )

        st.markdown("---")

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
            <div style='background:var(--bl-sb-surface); color:var(--bl-sb-accent); padding:8px 12px; border-radius:8px; text-align:center; font-weight:600; border:1px solid var(--bl-sb-border);'>
                ⭐ GitHub: sysuin/BankLens
            </div>
        </a>
        """
        st.markdown(textwrap.dedent(github_html), unsafe_allow_html=True)

        st.markdown("---")

        # ── Expanded Tech Stack Architecture ─────────────────────────────────
        st.markdown("### 🛠️ System Architecture")

        tech_html = """
        <div style='font-size:0.78rem; color:var(--bl-sb-muted);'>
            <p style='margin-bottom:2px;'><strong>🧠 AI &amp; LLM Orchestration</strong></p>
            <div style='display:flex; flex-wrap:wrap; gap:2px; margin-bottom:6px;'>
                <span class='tech-pill'>OpenAI GPT-4o</span>
                <span class='tech-pill'>Pydantic Guardrails</span>
                <span class='tech-pill'>LangChain LCEL</span>
            </div>
            <p style='margin-bottom:2px;'><strong>🗄️ Hybrid RAG &amp; Vector Store</strong></p>
            <div style='display:flex; flex-wrap:wrap; gap:2px; margin-bottom:6px;'>
                <span class='tech-pill'>ChromaDB + BM25</span>
                <span class='tech-pill'>text-embedding-3-small</span>
                <span class='tech-pill'>RRF Hybrid Search</span>
            </div>
            <p style='margin-bottom:2px;'><strong>🔒 Security &amp; Analytics</strong></p>
            <div style='display:flex; flex-wrap:wrap; gap:2px; margin-bottom:6px;'>
                <span class='tech-pill'>PII Regex Masking</span>
                <span class='tech-pill'>pdfplumber Engine</span>
                <span class='tech-pill'>pandas Analytics</span>
                <span class='tech-pill'>Streamlit UI</span>
            </div>
            <p style='margin-bottom:2px;'><strong>🐳 DevOps &amp; Cloud</strong></p>
            <div style='display:flex; flex-wrap:wrap; gap:2px;'>
                <span class='tech-pill'>Docker Multi-stage</span>
                <span class='tech-pill'>AWS EC2 Linux 2023</span>
                <span class='tech-pill'>Amazon ECR</span>
                <span class='tech-pill'>Nginx &amp; Certbot SSL</span>
                <span class='tech-pill'>GitHub Actions CI/CD</span>
            </div>
        </div>
        """
        st.markdown(textwrap.dedent(tech_html), unsafe_allow_html=True)

        st.markdown("---")
        st.caption("BankLens Enterprise v2.0 | Grounded Financial Intelligence")

    # ── Load & Parse Data ──────────────────────────────────────────────────────
    try:
        if selected_file_path:
            if selected_file_path.endswith(".pdf"):
                raw_df = parse_pdf_statement(selected_file_path)
            else:
                raw_df = load_and_validate_csv(selected_file_path)
        elif uploaded_file:
            if uploaded_file.name.endswith(".pdf"):
                raw_df = parse_pdf_statement(uploaded_file)
            else:
                raw_df = load_and_validate_csv(uploaded_file)
        else:
            st.markdown(
                textwrap.dedent("""
                <div class='bl-empty'>
                    <div class='bl-empty-mark'>&#8592;</div>
                    <h3 class='bl-empty-title'>Pick a statement to begin</h3>
                    <p class='bl-empty-sub'>
                        Choose one of the three demo statements in the sidebar,
                        or upload your own CSV or PDF. Everything below runs on
                        the statement you select.
                    </p>
                    <div class='bl-empty-steps'>
                        <div class='bl-step'><b>1</b> Parse &amp; mask PII</div>
                        <div class='bl-step'><b>2</b> Categorise &amp; score</div>
                        <div class='bl-step'><b>3</b> Retrieve products</div>
                        <div class='bl-step'><b>4</b> Draft the RM pitch</div>
                    </div>
                </div>
                """),
                unsafe_allow_html=True,
            )
            return
    except Exception as e:
        st.error(f"❌ **Error parsing bank statement:** {e}")
        return

    # Apply PII Sanitization Guard
    raw_df = sanitize_dataframe(raw_df)

    # ── Pipeline Step 1 & 2: Categorization & Analytics ───────────────────────
    categorized_df = categorize_dataframe(raw_df)

    try:
        metrics = compute_metrics(categorized_df)
    except ValueError as e:
        st.error(f"❌ **Error computing financial metrics:** {e}")
        return

    render_statement_views(
        metrics,
        categorized_df,
        generate_profile=lambda: run_ai_pipeline(metrics, categorized_df),
        chat_stream=lambda question: _direct_chat_stream(
            question, metrics, categorized_df
        ),
        sidebar_generate_clicked=sidebar_generate_clicked,
    )
    render_footer()


# ── API mode ──────────────────────────────────────────────────────────────────
#
# With BANKLENS_API_URL set, this console is a client of the BankLens API:
# sign in as a bank user, pick or upload a statement, and every view renders
# from what the API returns. The pipeline never runs in this process.


def _render_review_queue(client, tenant: str) -> None:
    """Reviewer console: pending income-verification decisions for this bank."""
    from app.ui.api_client import ApiError

    st.markdown("#### 🛂 Review queue")
    try:
        pending = client.reviews("pending")
    except ApiError as exc:
        st.error(exc.detail)
        return
    if not pending:
        st.success("Nothing waiting for review.")
        return
    for d in pending:
        with st.container(border=True):
            st.markdown(
                f"**{d['customer_name']}** (`{d['customer_ref']}`) · declared "
                f"₹{d['declared_monthly_income']:,.0f} vs observed "
                f"₹{d['observed_monthly_income']:,.0f} · **{d['discrepancy_pct']:.1f}%** "
                f"off (threshold {d['threshold_pct']:.0f}%)"
            )
            note = st.text_input(
                "Note", key=f"note_{d['id']}", placeholder="e.g. payslips received"
            )
            col_a, col_b = st.columns(2)
            action = None
            if col_a.button(
                "Approve",
                key=f"approve_{d['id']}",
                type="primary",
                use_container_width=True,
            ):
                action = "approve"
            if col_b.button(
                "Reject", key=f"reject_{d['id']}", use_container_width=True
            ):
                action = "reject"
            if action:
                with st.status(
                    f"Resuming run from its checkpoint ({action})…", expanded=True
                ) as status:
                    try:
                        for ev in client.decide(d["id"], action, note or None):
                            if ev.get("event") == "node":
                                st.write(f"✅ {ev.get('node')}")
                            elif ev.get("event") == "error":
                                st.write(f"❌ {ev.get('data')}")
                        status.update(
                            label="Run resumed and finished",
                            state="complete",
                            expanded=False,
                        )
                    except ApiError as exc:
                        status.update(label="Resume failed", state="error")
                        st.error(exc.detail)
                ss = st.session_state
                ss.ai_profile = None
                ss.profile_for = None
                st.rerun()


def main_api() -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    from app.ui.api_client import (
        ApiError,
        BankLensClient,
        dataframe_from_detail,
        metrics_from_detail,
        profile_from_response,
    )

    inject_custom_css()
    render_header()
    ss = st.session_state
    for key, default in (
        ("active_tab", "profiler"),
        ("ai_profile", None),
        ("chat_history", []),
        ("api_token", None),
        ("api_user", None),
        ("selected_statement_id", None),
        ("profile_for", None),
    ):
        if key not in ss:
            ss[key] = default

    client = BankLensClient(settings.banklens_api_url, ss.api_token)

    def sign_out() -> None:
        ss.api_token = None
        ss.api_user = None
        ss.selected_statement_id = None
        ss.ai_profile = None
        ss.chat_history = []
        st.rerun()

    sidebar_generate_clicked = False
    with st.sidebar:
        if not ss.api_token:
            st.markdown("## 🔐 Sign in")
            try:
                tenants = client.health().get("tenants_on_disk", [])
            except Exception as exc:  # noqa: BLE001 - shown to the user
                st.error(f"API unreachable at {settings.banklens_api_url}: {exc}")
                return
            tenant = st.selectbox("Bank", tenants or ["meridian"])
            email = st.text_input("Email", value=f"rm@{tenant}.example")
            password = st.text_input("Password", type="password", value="banklens-demo")
            if st.button("Sign in", type="primary", use_container_width=True):
                try:
                    body = client.login(tenant, email, password)
                    ss.api_token = body["access_token"]
                    ss.api_user = body
                    st.rerun()
                except ApiError as exc:
                    st.error(exc.detail)
            st.caption(
                "Demo users: rm@<bank>.example and reviewer@<bank>.example, "
                "password banklens-demo."
            )
            return

        user = ss.api_user
        is_rm = user["role"] == "rm"
        st.markdown(
            f"**{user['full_name']}**  \n`{user['tenant']}` · role **{user['role']}**"
        )
        if st.button("Sign out", use_container_width=True):
            sign_out()
        st.markdown("---")

        st.markdown("## 📊 Statements")
        try:
            statements = client.statements()
            customers = client.customers() if is_rm else []
        except ApiError as exc:
            if exc.status_code == 401:
                sign_out()
            st.error(exc.detail)
            return

        options = {
            f"{s['customer_name']} · {s['filename']} · {s['period']}": s["id"]
            for s in statements
        }
        if options:
            ids = list(options.values())
            index = (
                ids.index(ss.selected_statement_id)
                if ss.selected_statement_id in ids
                else 0
            )
            choice = st.selectbox("Choose a statement", list(options), index=index)
            ss.selected_statement_id = options[choice]
        else:
            st.info("No statements yet." + (" Upload one below." if is_rm else ""))

        if is_rm:
            st.markdown("---")
            st.markdown("### 📁 Upload a statement")
            customer_map = {
                f"{c['external_ref']} · {c['full_name']}": c["id"] for c in customers
            }
            customer_choice = (
                st.selectbox("Customer", list(customer_map)) if customer_map else None
            )
            uploaded = st.file_uploader("CSV or PDF", type=["csv", "pdf"])
            if (
                uploaded is not None
                and customer_choice
                and st.button(
                    "Upload & analyse", type="primary", use_container_width=True
                )
            ):
                try:
                    with st.spinner("Parsing, masking, categorising, computing…"):
                        summary = client.upload_statement(
                            customer_map[customer_choice],
                            uploaded.name,
                            uploaded.getvalue(),
                        )
                    ss.selected_statement_id = summary["id"]
                    ss.ai_profile = None
                    ss.chat_history = []
                    st.rerun()
                except ApiError as exc:
                    st.error(exc.detail)
            st.markdown("---")
            if st.button(
                "🚀 Generate AI Profile",
                type="primary",
                use_container_width=True,
                key="sidebar_gen_btn_api",
            ):
                sidebar_generate_clicked = True
                ss.active_tab = "profiler"
        st.markdown("---")
        try:
            gw = client.gateway()
            use = gw.get("would_use") or {}
            budget = gw.get("budget") or {}
            provider_line = (
                f"model gateway → **{use.get('provider')}** ({use.get('reason')})"
            )
            if budget:
                provider_line += (
                    f" · spent today ${budget['spent_today_usd']:.4f} "
                    f"of ${budget['daily_limit_usd']:.2f}"
                )
            circuits = ", ".join(
                f"{p['name']}: {p['circuit']}" for p in gw.get("providers", [])
            )
            st.caption(provider_line)
            st.caption(f"circuits — {circuits}")
        except ApiError:
            pass
        st.caption(f"API: {settings.banklens_api_url}")

    if not ss.selected_statement_id:
        st.markdown(
            "<div class='bl-empty'><h3 class='bl-empty-title'>Pick a statement to begin</h3>"
            "<p class='bl-empty-sub'>Choose one in the sidebar"
            + (" or upload a new one." if is_rm else ".")
            + "</p></div>",
            unsafe_allow_html=True,
        )
        return

    try:
        detail = client.statement(ss.selected_statement_id)
    except ApiError as exc:
        st.error(exc.detail)
        return

    metrics = metrics_from_detail(detail)
    categorized_df = dataframe_from_detail(detail)
    tenant = user["tenant"]

    if ss.profile_for != detail["id"]:
        ss.profile_for = detail["id"]
        ss.ai_profile = None
        try:
            existing = client.latest_profile(detail["id"])
        except ApiError:
            existing = None
        if existing:
            ss.ai_profile = profile_from_response(existing, tenant)

    NODE_LABELS = {
        "load_context": "Loaded metrics and the declared income",
        "verify_income": "Compared declared vs observed income",
        "await_review": "Reviewer decided",
        "retrieve": "Retrieved product passages",
        "narrate": "Model wrote the narrative",
        "guardrails": "Guardrails checked the recommendation",
        "finalize": "Profile stored",
        "finalize_rejected": "Run ended: income verification rejected",
        "finalize_blocked": "Run ended: guardrail blocked the recommendation",
    }

    def _render_events(events, box) -> str | None:
        """Render graph events as they arrive; return the run's final state."""
        outcome = None
        for ev in events:
            kind = ev.get("event")
            data = ev.get("data", {}) if isinstance(ev.get("data"), dict) else {}
            if kind == "node":
                node = ev.get("node", "?")
                line = f"✅ **{NODE_LABELS.get(node, node)}**"
                if node == "verify_income":
                    line += (
                        f" — declared ₹{ss.get('_declared', 0):,.0f}, "
                        f"discrepancy {data.get('discrepancy_pct', 0):.1f}%"
                        + (
                            " → **review required**"
                            if data.get("review_required")
                            else " → cleared"
                        )
                    )
                elif node == "retrieve":
                    line += f" — {', '.join(data.get('sources', []))}"
                elif node == "narrate":
                    line += (
                        f" — {data.get('tokens_in', 0)} in / {data.get('tokens_out', 0)} out tokens"
                        + (" (cache)" if data.get("from_cache") else "")
                    )
                elif node == "guardrails":
                    if data.get("guardrail_violations"):
                        line += " — **blocked**: " + "; ".join(
                            data["guardrail_violations"]
                        )
                    elif data.get("guardrail_warnings"):
                        line += " — warnings: " + "; ".join(data["guardrail_warnings"])
                elif node == "await_review":
                    line += f" — {data.get('review_decision')}"
                if data.get("outcome"):
                    outcome = data["outcome"]
                box.write(line)
            elif kind == "interrupt":
                outcome = "awaiting_review"
                box.write(
                    "⏸️ **Paused for a human.** Observed income differs from the declared "
                    "income by more than the threshold. A reviewer must approve or reject "
                    "before any recommendation is written. The run is checkpointed in "
                    "Postgres and resumes from here, even after a restart."
                )
            elif kind == "error":
                outcome = "error"
                box.write(f"❌ {ev.get('data')}")
        return outcome

    def generate_profile():
        ss["_declared"] = detail["declared_monthly_income"]
        with st.status("Running the decision graph…", expanded=True) as status:
            try:
                outcome = _render_events(client.run_graph(detail["id"]), st)
            except ApiError as exc:
                status.update(label="❌ Run failed", state="error")
                st.error(exc.detail)
                return None
            if outcome == "completed":
                status.update(
                    label="✨ Profile stored", state="complete", expanded=False
                )
                latest = client.latest_profile(detail["id"])
                return profile_from_response(latest, tenant) if latest else None
            if outcome == "awaiting_review":
                status.update(
                    label="⏸️ Awaiting reviewer", state="complete", expanded=True
                )
                st.info(
                    "Sign in as the reviewer to decide. The run resumes from its checkpoint."
                )
                return None
            status.update(label=f"Run ended: {outcome}", state="error", expanded=True)
            return None

    def chat_stream(question: str):
        history = [
            {
                "role": "assistant" if isinstance(m, AIMessage) else "user",
                "content": m.content,
            }
            for m in ss.chat_history
        ]
        collected: list[str] = []
        for token in client.chat(detail["id"], question, history):
            collected.append(token)
            yield token
        ss.chat_history.append(HumanMessage(content=question))
        ss.chat_history.append(AIMessage(content="".join(collected)))

    st.caption(
        f"Customer **{detail['customer_name']}** · declared monthly income "
        f"₹{detail['declared_monthly_income']:,.0f} · statement `{detail['id'][:8]}`"
    )

    if not is_rm:
        _render_review_queue(client, tenant)

    with st.expander(
        "🧾 Decision runs and audit trail for this statement", expanded=False
    ):
        try:
            runs = client.runs(detail["id"])
            trail = client.audit(detail["id"])
        except ApiError as exc:
            runs, trail = [], []
            st.error(exc.detail)
        if runs:
            st.dataframe(
                pd.DataFrame(runs)[
                    ["id", "status", "current_node", "error", "created_at"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No runs yet.")
        if trail:
            frame = pd.DataFrame(trail)[
                [
                    "created_at",
                    "node",
                    "event",
                    "actor",
                    "model",
                    "prompt_version",
                    "tokens_in",
                    "tokens_out",
                    "duration_ms",
                    "inputs_hash",
                ]
            ]
            st.dataframe(frame, use_container_width=True, hide_index=True)

    with st.expander("📊 Numbers the chat answers without a model", expanded=False):
        st.caption(
            "Questions matching these run a vetted SQL template as a read-only "
            "database role over tenant-filtered views. No model call, and every "
            "run is logged."
        )
        try:
            tpls = client.warehouse_templates()
            log = client.query_log(20)
        except ApiError as exc:
            tpls, log = [], []
            st.error(exc.detail)
        if tpls:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "template": t["name"],
                            "answers": t["description"],
                            "try asking": ", ".join(t["examples"]),
                        }
                        for t in tpls
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        st.markdown("**Query log** (what actually ran, newest first)")
        if log:
            st.dataframe(
                pd.DataFrame(log)[
                    [
                        "created_at",
                        "template",
                        "params",
                        "role",
                        "actor",
                        "rows",
                        "duration_ms",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("Nothing yet. Ask the chat for the savings rate.")

    with st.expander("⏱️ Traces: where the time and the money went", expanded=False):
        try:
            trace_rows = client.traces(detail["id"])
        except ApiError as exc:
            trace_rows = []
            st.error(exc.detail)
        if not trace_rows:
            st.caption(
                "No traces yet. Upload, run the graph or ask the chat something."
            )
        else:
            labels = {
                f"{t['started'][11:19]} · {t['root']} · {t['duration_ms']:.0f} ms · "
                f"${t['cost_usd']:.4f} · {t['spans']} spans": t["trace_id"]
                for t in trace_rows
            }
            picked = st.selectbox("Trace", list(labels), key="trace_pick")
            try:
                tr = client.trace(labels[picked])
            except ApiError as exc:
                tr = None
                st.error(exc.detail)
            if tr:
                st.markdown(
                    f"**{tr['total_ms']:.0f} ms** · {tr['tokens_in']} in / {tr['tokens_out']} out tokens · "
                    f"**${tr['cost_usd']:.4f}** · trace `{tr['trace_id'][:12]}`"
                )
                st.code("\n".join(tr["waterfall"]), language="text")
                rows = [
                    {
                        "span": ("  " * sp["depth"]) + sp["name"],
                        "start ms": sp["offset_ms"],
                        "duration ms": sp["duration_ms"],
                        "model": sp["model"],
                        "tokens in": sp["tokens_in"],
                        "tokens out": sp["tokens_out"],
                        "cost $": sp["cost_usd"],
                        "status": sp["status"],
                    }
                    for sp in tr["spans"]
                ]
                st.dataframe(
                    pd.DataFrame(rows), use_container_width=True, hide_index=True
                )

    render_statement_views(
        metrics,
        categorized_df,
        generate_profile=generate_profile,
        chat_stream=chat_stream,
        sidebar_generate_clicked=sidebar_generate_clicked,
        allow_generate=is_rm,
    )
    render_footer()


if __name__ == "__main__":
    if settings.banklens_api_url:
        main_api()
    else:
        main()
