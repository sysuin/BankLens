"""
Unit tests for the "Ask BankLens" tool-calling chat. Fully offline.

The LLM is faked: the point here is the agent loop's control flow — tools
execute, results feed back, the loop terminates, history updates — none of
which needs a real model to verify.
"""

import pandas as pd
import pytest

from app.pipeline.analyzer import compute_metrics
from app.pipeline.chat import MAX_TOOL_ROUNDS, make_tools, run_chat_turn


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("2024-01-01", "Salary Credit", 100000.0, "Credit", "Income"),
            ("2024-01-03", "Amazon Shopping", 16500.0, "Debit", "Shopping"),
            ("2024-01-04", "Local Store", 500.0, "Debit", "Shopping"),
            ("2024-01-05", "Rent", 30000.0, "Debit", "Rent & Housing"),
        ],
        columns=["date", "description", "amount", "type", "category"],
    )


@pytest.fixture
def metrics():
    return compute_metrics(_df())


class TestTools:
    def test_three_tools_are_built(self, metrics):
        names = {t.name for t in make_tools(metrics, _df())}
        assert names == {
            "get_customer_metrics",
            "search_products",
            "get_category_spending",
        }

    def test_metrics_tool_returns_computed_values(self, metrics):
        tool = {t.name: t for t in make_tools(metrics, _df())}["get_customer_metrics"]
        result = tool.invoke({})
        assert result["risk_profile"] == metrics.risk_profile
        assert result["total_income"] == 100000.0

    def test_category_tool_sums_and_ranks(self, metrics):
        tool = {t.name: t for t in make_tools(metrics, _df())}["get_category_spending"]
        result = tool.invoke({"category": "shopping"})
        assert result["total_spent"] == 17000.0
        assert result["transaction_count"] == 2
        assert result["largest_transactions"][0]["amount"] == 16500.0

    def test_unknown_category_lists_known_ones(self, metrics):
        tool = {t.name: t for t in make_tools(metrics, _df())}["get_category_spending"]
        result = tool.invoke({"category": "Yachts"})
        assert "error" in result and "Shopping" in result["known_categories"]


class _FakeChunk:
    """Stands in for AIMessageChunk: addable, with content and tool_calls."""

    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def __add__(self, other):
        return _FakeChunk(
            self.content + other.content, self.tool_calls + other.tool_calls
        )


class _FakeLLM:
    """Scripted model: first turn calls a tool, second turn answers in text."""

    def __init__(self, script):
        self.script = list(script)
        self.seen_messages = []

    def bind_tools(self, tools):
        return self

    def stream(self, messages):
        self.seen_messages.append(list(messages))
        yield from self.script.pop(0)


class TestAgentLoop:
    def _patch_llm(self, monkeypatch, fake):
        import langchain_openai

        monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kw: fake)

    def test_tool_round_then_streamed_answer(self, monkeypatch, metrics):
        fake = _FakeLLM(
            [
                [
                    _FakeChunk(
                        tool_calls=[
                            {"name": "get_customer_metrics", "args": {}, "id": "1"}
                        ]
                    )
                ],
                [_FakeChunk("The risk "), _FakeChunk("is Low.")],
            ]
        )
        self._patch_llm(monkeypatch, fake)

        history = []
        chunks = list(run_chat_turn("What's the risk?", history, metrics, _df()))

        assert "".join(chunks) == "The risk is Low."
        # Round 2 must contain the tool result the model asked for.
        final_messages = fake.seen_messages[-1]
        assert any(type(m).__name__ == "ToolMessage" for m in final_messages)
        # History carries the completed exchange.
        assert len(history) == 2
        assert history[1].content == "The risk is Low."

    def test_unknown_tool_is_reported_not_fatal(self, monkeypatch, metrics):
        fake = _FakeLLM(
            [
                [
                    _FakeChunk(
                        tool_calls=[{"name": "launch_missiles", "args": {}, "id": "1"}]
                    )
                ],
                [_FakeChunk("Done.")],
            ]
        )
        self._patch_llm(monkeypatch, fake)

        chunks = list(run_chat_turn("q", [], metrics, _df()))
        assert "".join(chunks) == "Done."
        tool_messages = [
            m for m in fake.seen_messages[-1] if type(m).__name__ == "ToolMessage"
        ]
        assert "Unknown tool" in tool_messages[0].content

    def test_loop_terminates_at_max_rounds(self, monkeypatch, metrics):
        endless = [
            [
                _FakeChunk(
                    tool_calls=[
                        {"name": "get_customer_metrics", "args": {}, "id": str(i)}
                    ]
                )
            ]
            for i in range(MAX_TOOL_ROUNDS + 3)
        ]
        fake = _FakeLLM(endless)
        self._patch_llm(monkeypatch, fake)

        answer = "".join(run_chat_turn("q", [], metrics, _df()))
        assert "maximum number of tool rounds" in answer
        assert len(fake.seen_messages) == MAX_TOOL_ROUNDS
