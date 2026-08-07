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
    Structured analytical output produced by the LangChain agent.

    Every field is grounded in the customer's actual transaction data
    and the RAG-retrieved banking product context.
    """

    financial_persona: str = Field(
        description="Short label describing the customer archetype, e.g. 'Disciplined High-Net Saver'."
    )

    financial_health_score: int = Field(
        description="Overall financial health score from 0 to 100 based on savings rate and expense stability."
    )

    risk_profile: Literal["Low", "Medium", "High"] = Field(
        description="Credit and default risk rating based on expense-to-income ratio and cashflow deficit."
    )

    income_stability_analysis: str = Field(
        description="Detailed analysis of income credits, salary stability, and secondary inflows."
    )

    spending_pattern_breakdown: str = Field(
        description="Detailed breakdown of essential vs discretionary spending based on actual category totals."
    )

    credit_risk_assessment: str = Field(
        description="In-depth credit default risk assessment explaining why the customer is rated Low/Medium/High risk."
    )

    primary_product: str = Field(
        description="The primary banking product recommended from the retrieved context."
    )

    primary_reason: str = Field(
        description="2-3 detailed sentences explaining why the primary product fits the customer's metrics."
    )

    secondary_product: str = Field(
        description="A complementary secondary product from the retrieved context to cross-sell."
    )

    secondary_reason: str = Field(
        description="1-2 sentences explaining why the secondary product adds value to the customer."
    )

    rm_hook_points: list[str] = Field(
        description="Exactly 3 structured bullet points for the RM call pitch: 1. Opening observation, 2. Value proposition, 3. Call-to-action."
    )

    retrieved_sources: list[str] = Field(
        description="List of knowledge base filenames retrieved for this recommendation."
    )

    @property
    def recommended_product(self) -> str:
        """Alias for backward compatibility with UI components."""
        return self.primary_product

    @property
    def recommendation_reason(self) -> str:
        """Alias for backward compatibility with UI components."""
        return self.primary_reason

    @property
    def rm_hook(self) -> str:
        """Alias for backward compatibility — joins bullet points into a clean string."""
        return " | ".join(self.rm_hook_points)


# ── Agent ─────────────────────────────────────────────────────────────────────


def build_profile(
    metrics: FinancialMetrics,
    retrieved_chunks: list[dict],
) -> CustomerProfile:
    """Generate a structured CustomerProfile using GPT-4o and RAG context."""
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"System prompt not found at: {SYSTEM_PROMPT_PATH}")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    parser = PydanticOutputParser(pydantic_object=CustomerProfile)

    rag_context_block = "\n\n---\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['content']}" for chunk in retrieved_chunks
    )

    source_filenames = sorted({chunk["source"] for chunk in retrieved_chunks})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n\n{format_instructions}"),
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

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.2,
        openai_api_key=settings.openai_api_key,
    )

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

    profile.retrieved_sources = source_filenames

    logger.info(
        "Profile built | Persona: '%s' | Score: %d | Risk: %s | Primary: '%s' | Secondary: '%s'",
        profile.financial_persona,
        profile.financial_health_score,
        profile.risk_profile,
        profile.primary_product,
        profile.secondary_product,
    )
    return profile
