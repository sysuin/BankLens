"""
Reusable Streamlit UI components for BankLens.

Renders polished UI blocks with enterprise-grade styling & SEO metadata:
    - inject_custom_css()      — theme tokens, master CSS, SEO meta tags, mobile responsiveness
    - render_header()          — sleek header with gradient title & live badge
    - render_metric_cards()    — formatted KPI cards
    - render_transaction_table() — styled DataFrame display
    - render_profile_card()    — customer persona, health score & risk badge
    - render_recommendation()  — Primary & Secondary products, bulleted RM hooks, RAG sources
    - render_footer()          — clean footer with contact info

Theming
-------
Streamlit themes its own widgets from .streamlit/config.toml, which defines a
light and a dark scheme. The custom HTML in this module cannot read that theme:
Streamlit exposes no CSS custom properties and stamps no theme attribute on the
DOM — only unstable emotion class hashes. So this module publishes its own
token palette and every custom colour is drawn from it.

The palette is resolved twice, deliberately:

  1. `prefers-color-scheme` handles the common case where the viewer is on the
     default "System" setting, and keeps working even if a rerun never happens.
  2. `st.context.theme` reports the theme Streamlit actually settled on, which
     is the only thing that is correct when the viewer picks Light or Dark
     explicitly from the menu and overrides their OS. It is emitted last so it
     wins over the media query at equal specificity.

Neither alone is sufficient, which is why both are emitted.
"""

import textwrap
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app.pipeline.analyzer import FinancialMetrics
from app.pipeline.agent import CustomerProfile

# ── Colour Maps ───────────────────────────────────────────────────────────────

# Risk badges carry white text on a solid fill, so the fill has to clear 4.5:1
# against white in *both* schemes. These shades do (5.3:1, 4.9:1, 6.5:1) and
# read correctly on a white card and a dark one alike, so unlike everything
# else in this module they are deliberately not theme-dependent. The previous
# values (#10b981, #f59e0b, #ef4444) sat near 2.3:1 and failed in either.
RISK_COLOURS: dict[str, str] = {
    "Low": "#047857",
    "Medium": "#b45309",
    "High": "#b91c1c",
}

# ── Theme Tokens ──────────────────────────────────────────────────────────────

# `*-border` is the card outline and wants to be bright enough to see.
# `*-chip` is the fill behind white label text and must clear 4.5:1 against
# white, which the border shades do not. They are separate for that reason.
LIGHT_TOKENS: dict[str, str] = {
    "--bl-surface": "#ffffff",
    "--bl-surface-2": "#f8fafc",
    "--bl-surface-3": "#f1f5f9",
    "--bl-border": "#e2e8f0",
    "--bl-border-strong": "#cbd5e1",
    "--bl-text": "#0f172a",
    "--bl-text-muted": "#475569",
    # Slate-500 (#64748b) measured 4.34:1 on the --bl-surface-3 score pill,
    # just under AA. This shade clears it there (4.9:1) and on white (5.4:1).
    "--bl-text-subtle": "#5b6b81",
    "--bl-body": "#334155",
    "--bl-accent": "#2563eb",
    "--bl-accent-text": "#0369a1",
    "--bl-info-bg": "#e0f2fe",
    "--bl-card-grad-a": "#ffffff",
    "--bl-card-grad-b": "#f8fafc",
    "--bl-shadow": "rgba(15, 23, 42, 0.05)",
    "--bl-primary-bg": "#eff6ff",
    "--bl-primary-border": "#2563eb",
    "--bl-primary-chip": "#2563eb",
    "--bl-primary-title": "#1e3a5f",
    "--bl-secondary-bg": "#f0fdf4",
    "--bl-secondary-border": "#16a34a",
    "--bl-secondary-chip": "#15803d",
    "--bl-secondary-title": "#14532d",
    "--bl-hook-bg": "#f8fafc",
    "--bl-hook-border": "#0284c7",
    "--bl-hook-text": "#1e293b",
    "--bl-success-text": "#047857",
    "--bl-warning-text": "#b45309",
    "--bl-score-good": "#047857",
    "--bl-score-mid": "#b45309",
    "--bl-score-bad": "#b91c1c",
    # Sidebar is deep navy in light mode, so its tokens are dark-surface tokens.
    "--bl-sb-text": "#f8fafc",
    "--bl-sb-muted": "#cbd5e1",
    "--bl-sb-surface": "#1e293b",
    "--bl-sb-border": "#334155",
    "--bl-sb-accent": "#38bdf8",
}

DARK_TOKENS: dict[str, str] = {
    # Cards sit above the #0b1220 page background rather than matching it, so
    # panels keep an edge without needing a heavier border.
    "--bl-surface": "#131c2e",
    "--bl-surface-2": "#1a2436",
    "--bl-surface-3": "#1f2a3d",
    "--bl-border": "#2a3648",
    "--bl-border-strong": "#3b4a63",
    "--bl-text": "#e8eef7",
    "--bl-text-muted": "#a9b6c9",
    "--bl-text-subtle": "#8494ab",
    "--bl-body": "#c3cedd",
    "--bl-accent": "#60a5fa",
    "--bl-accent-text": "#7dd3fc",
    "--bl-info-bg": "#12283d",
    "--bl-card-grad-a": "#131c2e",
    "--bl-card-grad-b": "#172033",
    "--bl-shadow": "rgba(0, 0, 0, 0.45)",
    "--bl-primary-bg": "#14243c",
    "--bl-primary-border": "#3b82f6",
    "--bl-primary-chip": "#1d4ed8",
    "--bl-primary-title": "#bfdbfe",
    "--bl-secondary-bg": "#0f2a1e",
    "--bl-secondary-border": "#22c55e",
    "--bl-secondary-chip": "#15803d",
    "--bl-secondary-title": "#bbf7d0",
    "--bl-hook-bg": "#1a2436",
    "--bl-hook-border": "#38bdf8",
    "--bl-hook-text": "#dbe6f3",
    "--bl-success-text": "#34d399",
    "--bl-warning-text": "#fbbf24",
    "--bl-score-good": "#34d399",
    "--bl-score-mid": "#fbbf24",
    "--bl-score-bad": "#f87171",
    "--bl-sb-text": "#e8eef7",
    "--bl-sb-muted": "#a9b6c9",
    "--bl-sb-surface": "#1f2a3d",
    "--bl-sb-border": "#2f3d52",
    "--bl-sb-accent": "#7dd3fc",
}


def _active_theme() -> str | None:
    """
    Return "light" / "dark" as Streamlit resolved it, or None if unavailable.

    None is a normal outcome, not an error: the theme is reported by the
    browser, so it can be absent on the very first script run. Callers fall
    back to the prefers-color-scheme rules, which is why this never raises.
    """
    try:
        theme = st.context.theme
    except Exception:  # noqa: BLE001 - theme reporting must never break the page
        return None

    theme_type = getattr(theme, "type", None) if theme is not None else None
    return theme_type if theme_type in ("light", "dark") else None


def _token_block(tokens: dict[str, str], indent: str = "        ") -> str:
    """Render a token mapping as CSS declarations."""
    return "\n".join(f"{indent}{name}: {value};" for name, value in tokens.items())


def _theme_token_css() -> str:
    """
    Build the token stylesheet: light base, dark media query, explicit override.

    Order matters. The explicit block is emitted last and at the same
    specificity as the media query, so when Streamlit tells us the real theme
    it overrides whatever the OS preference implied.
    """
    css = [
        ":root {",
        _token_block(LIGHT_TOKENS),
        "}",
        "@media (prefers-color-scheme: dark) {",
        "    :root {",
        _token_block(DARK_TOKENS, indent="            "),
        "    }",
        "}",
    ]

    resolved = _active_theme()
    if resolved is not None:
        css += [
            f"/* Streamlit reported theme: {resolved} */",
            ":root {",
            _token_block(DARK_TOKENS if resolved == "dark" else LIGHT_TOKENS),
            "}",
        ]

    return "\n".join(css)


def inject_custom_css() -> None:
    """Inject theme tokens, custom CSS rules, and SEO meta tags."""
    # ── SEO Meta Tags Injection ───────────────────────────────────────────────
    st.markdown(
        textwrap.dedent("""
        <head>
            <title>BankLens | AI Bank Statement Analyzer &amp; Product Recommendation Engine</title>
            <meta name="description" content="BankLens is an enterprise AI financial intelligence platform that converts bank statements into financial health profiles, credit risk ratings, and grounded banking product recommendations." />
            <meta name="keywords" content="Bank Statement Analyzer, AI Financial Profiling, RAG Banking, Credit Risk Assessment, Relationship Manager Tool, FinTech AI, LangChain, Streamlit, ChromaDB" />
            <meta name="author" content="Sunny Singh" />
            <meta name="robots" content="index, follow" />
            <link rel="canonical" href="https://banklens.sysuin.com" />
            <meta name="color-scheme" content="light dark" />
            <meta property="og:type" content="website" />
            <meta property="og:url" content="https://banklens.sysuin.com" />
            <meta property="og:title" content="BankLens | AI Bank Statement Analyzer &amp; Product Recommendation Engine" />
            <meta property="og:description" content="Automated financial health analysis, RAG product recommendations, and relationship manager pitch generation powered by GPT-4o and ChromaDB." />
            <meta property="twitter:card" content="summary_large_image" />
            <meta property="twitter:url" content="https://banklens.sysuin.com" />
            <meta property="twitter:title" content="BankLens | AI Bank Statement Analyzer" />
            <meta property="twitter:description" content="Automated financial health analysis and RAG product recommendations for retail banking." />
        </head>
        """),
        unsafe_allow_html=True,
    )

    # ── Theme Tokens ─────────────────────────────────────────────────────────
    st.markdown(f"<style>\n{_theme_token_css()}\n</style>", unsafe_allow_html=True)

    # ── Master CSS Styling ───────────────────────────────────────────────────
    #
    # Nothing here paints the app background or forces body text colour any
    # more. Those are Streamlit's to set from config.toml, and overriding them
    # with !important was what pinned the whole app to light mode regardless of
    # the viewer's preference.
    st.markdown(
        textwrap.dedent("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Metric card styling */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: var(--bl-text) !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            color: var(--bl-text-muted) !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Glass card container */
        .glass-card {
            background: linear-gradient(135deg, var(--bl-card-grad-a) 0%, var(--bl-card-grad-b) 100%);
            border: 1px solid var(--bl-border);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 4px 20px -2px var(--bl-shadow);
            margin-bottom: 1.25rem;
        }

        /* Sidebar compact spacing. Sidebar colours come from
           [theme.light.sidebar] / [theme.dark.sidebar] in config.toml, so the
           blanket `[data-testid="stSidebar"] *` colour override is gone — it
           used to repaint Streamlit's own widget text in both schemes. */
        [data-testid="stSidebar"] .element-container {
            margin-bottom: 0.35rem !important;
        }

        [data-testid="stSidebar"] hr {
            margin: 0.5rem 0 !important;
            border-color: var(--bl-sb-border) !important;
        }

        /* File uploader — lives in the dark sidebar in both schemes */
        [data-testid="stFileUploader"] {
            background-color: var(--bl-sb-surface) !important;
            border: 1px dashed var(--bl-sb-border) !important;
            border-radius: 10px !important;
            padding: 0.75rem !important;
        }

        [data-testid="stFileUploader"] label {
            color: var(--bl-sb-text) !important;
            font-weight: 600 !important;
        }

        /* Previously #0f172a — near-black text on the near-black uploader
           panel, about 1.2:1 and effectively invisible. */
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] [data-testid="stCaptionContainer"] p,
        [data-testid="stFileUploader"] span:not(button span) {
            color: var(--bl-sb-muted) !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
        }

        [data-testid="stFileUploader"] button {
            background-color: var(--bl-sb-surface) !important;
            color: var(--bl-sb-text) !important;
            border: 1px solid var(--bl-sb-border) !important;
            border-radius: 8px !important;
        }

        /* Tech badge pill */
        .tech-pill {
            display: inline-flex;
            align-items: center;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: var(--bl-sb-accent) !important;
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
            color: var(--bl-text-subtle);
            font-size: 0.88rem;
            border-top: 1px solid var(--bl-border);
            margin-top: 2rem;
        }

        .app-footer a {
            color: var(--bl-accent);
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

    # ── JS: Replace file uploader limit text ─────────────────────────────────
    # st.markdown strips <script> tags in production. The only reliable way to
    # execute JS in Streamlit is components.html(), which creates an iframe.
    # We use window.parent.document to reach the main app's DOM from the iframe.
    components.html(
        """
        <script>
        (function() {
            function fixUploaderText() {
                try {
                    var els = window.parent.document.querySelectorAll(
                        '[data-testid="stFileUploader"] small'
                    );
                    els.forEach(function(el) {
                        if (el.innerText && el.innerText.indexOf('5MB') === -1) {
                            el.innerText = 'Limit 5MB per file • CSV';
                        }
                    });
                } catch(e) {}
            }
            fixUploaderText();
            var observer = new MutationObserver(fixUploaderText);
            observer.observe(window.parent.document.body, { childList: true, subtree: true });
        })();
        </script>
        """,
        height=0,
    )


def render_header() -> None:
    """Render the main header banner with title and subtitle."""
    st.markdown(
        textwrap.dedent("""
        <div style='text-align: center; padding: 0.5rem 0 1.2rem 0;'>
            <div style='display: inline-flex; align-items: center; gap: 8px; background: var(--bl-info-bg); color: var(--bl-accent-text); padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.75rem;'>
                ⚡ Enterprise AI Financial Intelligence Platform
            </div>
            <h1 style='color: var(--bl-text); font-weight: 800; font-size: 2.6rem; margin: 0; letter-spacing: -0.02em;'>
                🏦 BankLens
            </h1>
            <p style='color: var(--bl-text-subtle); font-size: 1.1rem; margin-top: 0.4rem; font-weight: 400;'>
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
    risk_colour = RISK_COLOURS.get(profile.risk_profile, "#64748b")
    score = profile.financial_health_score
    score_token = (
        "var(--bl-score-good)"
        if score >= 75
        else ("var(--bl-score-mid)" if score >= 50 else "var(--bl-score-bad)")
    )

    card_html = f"""
    <div class='glass-card'>
        <div style='display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem; border-bottom: 1px solid var(--bl-border); padding-bottom: 1rem; margin-bottom: 1rem;'>
            <div>
                <span style='font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em; color:var(--bl-text-subtle); font-weight:700;'>Customer Archetype</span>
                <h2 style='margin:0; color:var(--bl-text); font-weight:800; font-size:1.7rem;'>
                    👤 {profile.financial_persona}
                </h2>
            </div>
            <div style='display:flex; gap:12px; align-items:center;'>
                <div style='background:var(--bl-surface-3); border:1px solid var(--bl-border-strong); padding:6px 16px; border-radius:30px; text-align:center;'>
                    <span style='font-size:0.75rem; color:var(--bl-text-subtle); font-weight:700; text-transform:uppercase;'>Health Score</span>
                    <div style='font-size:1.2rem; font-weight:800; color:{score_token};'>{score}/100</div>
                </div>
                <div style='background:{risk_colour}; color:#ffffff; font-size:0.85rem; font-weight:700; padding:10px 18px; border-radius:30px; box-shadow: 0 2px 10px var(--bl-shadow);'>
                    🛡️ {profile.risk_profile} Risk Profile
                </div>
            </div>
        </div>
        <div class='grid-3col' style='display:grid; grid-template-columns: 1fr 1fr 1fr; gap:1rem; margin-top:1rem;'>
            <div style='background:var(--bl-surface); padding:1rem; border-radius:12px; border:1px solid var(--bl-border);'>
                <span style='font-weight:700; color:var(--bl-accent-text); font-size:0.88rem;'>📈 Income Stability</span>
                <p style='color:var(--bl-body); margin-top:0.4rem; margin-bottom:0; font-size:0.9rem; line-height:1.5;'>
                    {profile.income_stability_analysis}
                </p>
            </div>
            <div style='background:var(--bl-surface); padding:1rem; border-radius:12px; border:1px solid var(--bl-border);'>
                <span style='font-weight:700; color:var(--bl-success-text); font-size:0.88rem;'>🛒 Spending Breakdown</span>
                <p style='color:var(--bl-body); margin-top:0.4rem; margin-bottom:0; font-size:0.9rem; line-height:1.5;'>
                    {profile.spending_pattern_breakdown}
                </p>
            </div>
            <div style='background:var(--bl-surface); padding:1rem; border-radius:12px; border:1px solid var(--bl-border);'>
                <span style='font-weight:700; color:var(--bl-warning-text); font-size:0.88rem;'>⚠️ Credit Risk Rationale</span>
                <p style='color:var(--bl-body); margin-top:0.4rem; margin-bottom:0; font-size:0.9rem; line-height:1.5;'>
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
        <div style='background: var(--bl-primary-bg); border: 2px solid var(--bl-primary-border); border-radius: 14px; padding: 1.25rem; height: 100%;'>
            <div style='background:var(--bl-primary-chip); color:#ffffff; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; display:inline-block; margin-bottom:0.6rem;'>
                🥇 PRIMARY PRODUCT OFFER
            </div>
            <h3 style='color:var(--bl-primary-title); margin:0 0 0.5rem 0; font-weight:800;'>
                🏆 {profile.primary_product}
            </h3>
            <p style='color:var(--bl-body); font-size:0.95rem; line-height:1.55; margin:0;'>
                {profile.primary_reason}
            </p>
        </div>
        """
        st.markdown(textwrap.dedent(card1_html), unsafe_allow_html=True)

    with col_p2:
        card2_html = f"""
        <div style='background: var(--bl-secondary-bg); border: 2px solid var(--bl-secondary-border); border-radius: 14px; padding: 1.25rem; height: 100%;'>
            <div style='background:var(--bl-secondary-chip); color:#ffffff; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; display:inline-block; margin-bottom:0.6rem;'>
                🥈 SECONDARY CROSS-SELL OFFER
            </div>
            <h3 style='color:var(--bl-secondary-title); margin:0 0 0.5rem 0; font-weight:800;'>
                💎 {profile.secondary_product}
            </h3>
            <p style='color:var(--bl-body); font-size:0.95rem; line-height:1.55; margin:0;'>
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
            f"<li style='margin-bottom: 0.6rem; color: var(--bl-hook-text); font-size: 0.96rem; line-height: 1.5;'>{pt}</li>"
            for pt in profile.rm_hook_points
        ]
    )

    hook_box_html = f"""
    <div style='background: var(--bl-hook-bg); border-left: 5px solid var(--bl-hook-border); border-radius: 8px; padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;'>
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
            <div style='background:var(--bl-surface-3); border:1px solid var(--bl-border-strong); padding:0.6rem 1rem; border-radius:8px; text-align:center;'>
                📄 <code style='font-weight:600; color:var(--bl-text);'>{source}</code>
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
