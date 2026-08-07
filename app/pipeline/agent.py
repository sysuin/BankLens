"""
LangChain agent for BankLens customer profiling.

Builds a LangChain chain that:
    1. Formats a prompt from the customer's financial metrics and the
       RAG-retrieved product context
    2. Calls OpenAI GPT-4o with temperature=0.2 for consistent output
    3. Parses the response into a validated CustomerProfile Pydantic model

Using PydanticOutputParser ensures the LLM output always matches the
expected schema. If the model returns malformed JSON, LangChain raises
a descriptive error rather than silently failing downstream.

The agent is grounded by design — the system prompt explicitly instructs
the model to recommend only products that appear in the retrieved context
and to never invent financial figures.
"""

from pathlib import Path
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logger import get_logger
from app.pipeline.analyzer import FinancialMetrics

logger = get_logger(__name__)

# Path to the system prompt file — loaded once at module import time
SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "prompts" / "system_prompt.txt"
)


# ── Output Schema ─────────────────────────────────────────────────────────────


class CustomerProfile(BaseModel):
    """
    Structured output produced by the LangChain agent.

    Every field is grounded in the customer's actual transaction data
    and the RAG-retrieved banking product context. No field should
    contain invented or hallucinated information.
    """

    financial_persona: str = Field(
        description=(
            "A short label describing the customer's financial archetype. "
            "Examples: 'Disciplined Saver', 'Active Lifestyle Spender', "
            "'Growth-Oriented Professional', 'Cautious Budgeter'."
        )
    )

    risk_profile: Literal["Low", "Medium", "High"] = Field(
        description=(
            "Credit and financial risk rating based on savings rate and "
            "expense-to-income ratio. "
            "Low = savings rate > 30%, Medium = 10–30%, High = < 10%."
        )
    )

    spending_insight: str = Field(
        description=(
            "One concise sentence summarising the customer's most notable "
            "spending pattern, citing the actual top category and amount."
        )
    )

    savings_insight: str = Field(
        description=(
            "One concise sentence describing the customer's savings behaviour "
            "and what it signals about their financial discipline."
        )
    )

    recommended_product: str = Field(
        description=(
            "The specific banking product name from the retrieved context "
            "that best fits this customer's profile. "
            "Must be an exact product name from the knowledge base."
        )
    )

    recommendation_reason: str = Field(
        description=(
            "Two to three sentences explaining why this product suits the "
            "customer, grounded in the retrieved knowledge base context. "
            "Reference the customer's actual metrics."
        )
    )

    rm_hook: str = Field(
        description=(
            "A personalised, one-sentence opening line a Bank Relationship "
            "Manager can use to start a call with this customer. It should "
            "reference a specific detail from their statement."
        )
    )

    retrieved_sources: list[str] = Field(
        description=(
            "List of knowledge base filenames that were retrieved and used "
            "as context for this recommendation. "
            "Example: ['fixed_deposit.md', 'savings_account.md']."
        )
    )


# ── Agent ─────────────────────────────────────────────────────────────────────


def build_profile(
    metrics: FinancialMetrics,
    retrieved_chunks: list[dict],
) -> CustomerProfile:
    """
    Generate a structured CustomerProfile using GPT-4o and RAG context.

    Constructs a LangChain chain:
        ChatPromptTemplate → ChatOpenAI → PydanticOutputParser

    The prompt includes:
        - The system prompt (persona, rules, tone) from system_prompt.txt
        - The customer's financial metrics as JSON
        - The RAG-retrieved product context (text + source filenames)
        - Pydantic format instructions injected automatically by the parser

    Args:
        metrics: The validated FinancialMetrics for this customer.
        retrieved_chunks: List of dicts from rag.retrieve(), each with
                          'content' (str) and 'source' (str) keys.

    Returns:
        A validated CustomerProfile Pydantic model.

    Raises:
        FileNotFoundError: If system_prompt.txt does not exist.
        ValidationError: If GPT-4o returns output that does not match
                         the CustomerProfile schema.
    """
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"System prompt not found at: {SYSTEM_PROMPT_PATH}")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    parser = PydanticOutputParser(pydantic_object=CustomerProfile)

    # ── Format the RAG context block ──────────────────────────────────────────
    # Each retrieved chunk is prefixed with its source filename so the LLM
    # can cite it in the recommendation_reason and retrieved_sources fields.
    rag_context_block = "\n\n---\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['content']}" for chunk in retrieved_chunks
    )

    # Collect unique source filenames for injection into retrieved_sources
    source_filenames = sorted({chunk["source"] for chunk in retrieved_chunks})

    # ── Build the prompt template ─────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages(
        [
            # System message: establishes the persona and output rules
            ("system", system_prompt + "\n\n{format_instructions}"),
            # Human message: provides the actual customer data
            (
                "human",
                "## Customer Financial Metrics\n\n"
                "{metrics}\n\n"
                "## Retrieved Banking Product Context\n\n"
                "{rag_context}\n\n"
                "## Available Source Documents\n\n"
                "{sources}",
            ),
        ]
    )

    # ── Initialise the LLM ────────────────────────────────────────────────────
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.2,  # Low temperature for consistent, structured output
        openai_api_key=settings.openai_api_key,
    )

    # ── Compose the chain ─────────────────────────────────────────────────────
    # LangChain's pipe operator: prompt → LLM → parser
    chain = prompt | llm | parser

    logger.info(
        "Invoking %s | RAG chunks: %d | Sources: %s",
        settings.openai_model,
        len(retrieved_chunks),
        source_filenames,
    )

    profile: CustomerProfile = chain.invoke(
        {
            "metrics": metrics.model_dump_json(indent=2),
            "rag_context": rag_context_block,
            "sources": str(source_filenames),
            "format_instructions": parser.get_format_instructions(),
        }
    )

    # Overwrite retrieved_sources with the actual filenames used
    # (the LLM may summarise them differently in its output)
    profile.retrieved_sources = source_filenames

    logger.info(
        "Profile built | Persona: '%s' | Risk: %s | Product: '%s'",
        profile.financial_persona,
        profile.risk_profile,
        profile.recommended_product,
    )
    return profile
