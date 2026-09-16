"""
BankLens as an MCP server.

Exposes the analysis pipeline over the Model Context Protocol, so any MCP
client — Claude Desktop, an IDE, another agent — can call BankLens as a tool
instead of driving the Streamlit UI.

This wrapper is deliberately thin: it re-exports the same functions the app
uses, in the same order the app calls them. No pipeline logic lives here, so
the MCP surface can never drift from what the product actually does.

Run standalone (stdio transport, the default for local clients):

    python mcp_server.py

Claude Desktop config entry:

    "banklens": {
        "command": "/path/to/.venv/bin/python",
        "args": ["/path/to/BankLens/mcp_server.py"]
    }

The OPENAI_API_KEY from .env is loaded by pydantic-settings as usual.
"""

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

# MCP hosts (Claude Desktop included) launch servers from an arbitrary working
# directory, but the pipeline resolves .env — and therefore the API key —
# relative to CWD. Anchor to the project root before any app import so the
# server behaves identically however it is launched.
os.chdir(Path(__file__).resolve().parent)

# mcp 2.x renamed FastMCP to MCPServer; our surface (tool decorator, stdio
# run) is unchanged across the major. Pinned >=2,<3 after migrating.
mcp = MCPServer("banklens")


def _resolve_tenant(tenant: str) -> str | None:
    """
    Empty means the configured default. Anything else must be a bank with a
    catalogue on disk; the error names the ones that exist, so the calling
    agent can correct itself.
    """
    if not tenant:
        return None
    from app.pipeline.rag import list_tenants, validate_tenant_slug

    validate_tenant_slug(tenant)
    available = list_tenants()
    if tenant not in available:
        raise ValueError(f"unknown tenant {tenant!r}; available: {available}")
    return tenant


def _analyze(csv_path: str, tenant: str | None = None) -> tuple:
    """Run the full pipeline on a statement CSV. Shared by both tools."""
    import pandas as pd

    from app.pipeline.analyzer import compute_metrics
    from app.pipeline.categorizer import categorize_dataframe
    from app.pipeline.rag import build_retrieval_query, build_vector_store, retrieve
    from app.pipeline.sanitizer import sanitize_dataframe

    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"No statement file at: {path}")

    df = pd.read_csv(path)
    df = sanitize_dataframe(df)
    df = categorize_dataframe(df)
    metrics = compute_metrics(df)

    chunks = retrieve(
        build_retrieval_query(metrics), build_vector_store(tenant), tenant=tenant
    )
    return metrics, chunks


@mcp.tool()
def analyze_statement(csv_path: str, tenant: str = "") -> dict:
    """
    Analyze a bank statement CSV and return the full customer profile.

    Runs sanitization, categorization, metric computation, hybrid RAG
    retrieval and grounded LLM profiling — the same pipeline as the web app.
    The CSV needs columns: date, description, amount, type (Credit/Debit).
    `tenant` selects which bank's product catalogue is used (a directory under
    knowledge_base/); empty means the configured default tenant.

    Returns the computed metrics and the AI-generated profile, including the
    two product recommendations and the RM pitch hooks.
    """
    from app.pipeline.agent import build_profile

    resolved = _resolve_tenant(tenant)
    metrics, chunks = _analyze(csv_path, resolved)
    profile = build_profile(metrics, chunks, tenant=resolved)

    return {
        "metrics": metrics.model_dump(),
        "profile": profile.model_dump(),
    }


@mcp.tool()
def compute_statement_metrics(csv_path: str) -> dict:
    """
    Compute deterministic financial metrics for a statement CSV — no LLM call.

    Free and instant: income, expenses, savings rate, risk band, health score,
    the essential/discretionary split, and top spending categories. Use this
    when the numbers are enough and a full narrative profile is not needed.
    """
    import pandas as pd

    from app.pipeline.analyzer import compute_metrics
    from app.pipeline.categorizer import categorize_dataframe
    from app.pipeline.sanitizer import sanitize_dataframe

    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"No statement file at: {path}")

    df = pd.read_csv(path)
    df = sanitize_dataframe(df)
    df = categorize_dataframe(df)
    return compute_metrics(df).model_dump()


@mcp.tool()
def search_products(query: str, tenant: str = "") -> list[dict]:
    """
    Search a bank's product knowledge base.

    Hybrid retrieval (dense embeddings + BM25, RRF-fused) over the tenant's
    product documents. Returns the most relevant passages with their source
    filenames. `tenant` is a directory under knowledge_base/; empty means the
    configured default tenant.
    """
    from app.pipeline.rag import build_vector_store, retrieve

    resolved = _resolve_tenant(tenant)
    return retrieve(query, build_vector_store(resolved), tenant=resolved)


if __name__ == "__main__":
    # Under stdio transport, stdout is the JSON-RPC wire. Pipeline logs must
    # therefore go to stderr, where MCP clients (Claude Desktop included)
    # collect them as server logs instead of choking on them as bad frames.
    from app.core.logger import route_logs_to_stderr

    route_logs_to_stderr()
    mcp.run(transport="stdio")
