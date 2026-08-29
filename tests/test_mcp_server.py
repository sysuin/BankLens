"""Unit tests for the MCP server wrapper. Offline: tool registration + errors."""

import asyncio

import pytest

import mcp_server


class TestToolRegistration:
    def test_all_three_tools_are_registered(self):
        tools = asyncio.run(mcp_server.mcp.list_tools())
        names = {t.name for t in tools}
        assert names == {
            "analyze_statement",
            "compute_statement_metrics",
            "search_products",
        }

    def test_tools_carry_descriptions(self):
        tools = asyncio.run(mcp_server.mcp.list_tools())
        for t in tools:
            assert t.description and len(t.description) > 40


class TestErrorPaths:
    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="No statement file"):
            mcp_server.compute_statement_metrics("/nowhere/statement.csv")
