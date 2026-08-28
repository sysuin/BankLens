"""
Customer profiling engine for BankLens Enterprise v2.0.

Builds a structured CustomerProfile using:
    1. Grounded RAG context & financial health metrics
    2. Indirect Prompt Injection Defense via XML boundary encapsulation
    3. OpenAI GPT-4o with temperature=0.2 for deterministic precision
    4. Pydantic output guardrails — including *semantic* validation of product
       names against the knowledge base, not merely type validation
    5. A single corrective retry when validation fails, before giving up

The risk rating and health score are NOT produced here. They are computed
deterministically in analyzer.py and passed in; this module only generates the
narrative that explains them.
"""

import re
from functools import lru_cache
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.core.logger import get_logger
from app.pipeline.analyzer import FinancialMetrics, RiskProfile
from app.pipeline.rag import KNOWLEDGE_BASE_DIR

logger = get_logger(__name__)

# Path to the system prompt file
SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "prompts" / "system_prompt.txt"
)


# ── Product Catalogue & Name Resolution ───────────────────────────────────────
#
# Pydantic validates that primary_product is a string. It cannot know whether
# that string names a product the bank actually sells. Without the check below,
# an invented product passes validation and is rendered to a Relationship
# Manager as a real recommendation to make to a customer.

# Words carrying no product meaning, dropped before comparison.
_STOPWORDS = {"the", "a", "an", "and", "of", "for", "in", "with", "to"}

# Minimum overlap between a proposed name and a catalogue entry for the name to
# be accepted. 0.6 admits real variants ("High-Yield Savings Account",
# "Sweep-In Fixed Deposit") while rejecting plausible inventions
# ("Platinum Rewards Card", "Gold Loan").
_PRODUCT_MATCH_THRESHOLD = 0.6


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, drop stopwords, and crudely singularize."""
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    tokens = set()
    for word in words:
        if word in _STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        tokens.add(word)
    return tokens


@lru_cache(maxsize=1)
def get_product_catalogue() -> dict[str, tuple[frozenset[str], ...]]:
    """
    Build {knowledge base filename: (alias token sets,)} from disk.

    Each product contributes two aliases — its filename stem and its markdown
    H1 title — because the LLM may echo either ("mutual_funds_sip.md" is titled
    "Systematic Investment Plan (SIP) & Wealth Mutual Funds", and both phrasings
    are legitimate).
    """
    catalogue: dict[str, tuple[frozenset[str], ...]] = {}

    for md_file in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        aliases = [frozenset(_tokenize(md_file.stem.replace("_", " ")))]

        first_line = md_file.read_text(encoding="utf-8").splitlines()[0]
        if first_line.startswith("#"):
            aliases.append(frozenset(_tokenize(first_line.lstrip("# "))))

        catalogue[md_file.name] = tuple(a for a in aliases if a)

    return catalogue


def resolve_product(name: str) -> str | None:
    """
    Map a proposed product name onto a knowledge base file, or None.

    Scored symmetrically — overlap is measured against both the catalogue entry
    and the proposed name, so neither a terse name ("Home Loan" against
    "home_loan_mortgage") nor a verbose one ("Systematic Investment Plan"
    against "mutual_funds_sip") is unfairly penalised.
    """
    candidate = _tokenize(name)
    if not candidate:
        return None

    # Ranked on (score, coverage). Score alone ties more often than it looks:
    # "Sweep-In Fixed Deposit" scores 1.0 against both fixed_deposit.md, whose
    # whole alias it contains, and sweep_account.md, whose H1 is literally
    # "Sweep-In Fixed Deposit (Sweep Account)". A strict > comparison then
    # settled it alphabetically and picked the wrong document.
    #
    # Coverage — the share of the *proposed name* the alias accounts for —
    # breaks that tie toward the document that leaves nothing unexplained.
    # fixed_deposit.md covers 2 of 3 tokens and ignores "sweep"; sweep_account
    # covers all 3. Score stays primary so a verbose but correct name like
    # "Systematic Investment Plan" still resolves against a longer alias.
    best_file, best_rank = None, (0.0, 0.0)

    for filename, aliases in get_product_catalogue().items():
        for alias in aliases:
            shared = len(alias & candidate)
            if not shared:
                continue
            score = max(shared / len(alias), shared / len(candidate))
            rank = (score, shared / len(candidate))
            if rank > best_rank:
                best_file, best_rank = filename, rank

    return best_file if best_rank[0] >= _PRODUCT_MATCH_THRESHOLD else None


def _validate_product_name(value: str) -> str:
    """Field validator shared by primary_product and secondary_product."""
    if resolve_product(value) is None:
        known = sorted(get_product_catalogue())
        raise ValueError(
            f"'{value}' is not a product in the knowledge base. "
            f"Use a product from: {known}"
        )
    return value


# ── Output Schema ─────────────────────────────────────────────────────────────


class ProfileNarrative(BaseModel):
    """
    The part of the profile the LLM is actually asked to write.

    Analysis and recommendation prose, grounded in the customer's transaction
    metrics and the RAG-retrieved product context. Deliberately excludes the
    risk rating and health score: those are arithmetic, they are computed in
    analyzer.py, and a credit decision must be reproducible.
    """

    financial_persona: str = Field(
        description="Short label describing the customer archetype, e.g. 'Disciplined High-Net Saver'."
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
        min_length=3,
        max_length=3,
        description="Exactly 3 structured bullet points for the RM call pitch: 1. Opening observation, 2. Value proposition, 3. Call-to-action.",
    )

    @field_validator("primary_product", "secondary_product")
    @classmethod
    def product_must_exist(cls, value: str) -> str:
        """Reject product names that do not correspond to a knowledge base entry."""
        return _validate_product_name(value)


class CustomerProfile(ProfileNarrative):
    """
    The complete profile handed to the UI.

    Extends the LLM-generated narrative with fields the application owns:
    the risk rating and health score computed in analyzer.py, and the list of
    sources actually retrieved. Keeping these off ProfileNarrative is what
    prevents the model from being asked to decide them.
    """

    financial_health_score: int = Field(
        ge=0,
        le=100,
        description="Financial health score computed from savings rate in analyzer.py.",
    )

    risk_profile: RiskProfile = Field(
        description="Credit risk band computed from savings rate in analyzer.py."
    )

    retrieved_sources: list[str] = Field(
        default_factory=list,
        description="List of knowledge base filenames retrieved for this recommendation.",
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
    """Generate a structured CustomerProfile using GPT-4o, prompt injection defense, and RAG context."""
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"System prompt not found at: {SYSTEM_PROMPT_PATH}")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    parser = PydanticOutputParser(pydantic_object=ProfileNarrative)

    rag_context_block = "\n\n---\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['content']}" for chunk in retrieved_chunks
    )

    source_filenames = sorted({chunk["source"] for chunk in retrieved_chunks})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n\n{format_instructions}"),
            (
                "human",
                "<customer_metrics>\n"
                "{metrics}\n"
                "</customer_metrics>\n\n"
                "<rag_context>\n"
                "{rag_context}\n"
                "</rag_context>\n\n"
                "## Available Source Documents\n"
                "{sources}\n\n"
                "## Assigned Risk Rating (already computed — explain it, do not change it)\n"
                "risk_profile={risk_profile}, financial_health_score={health_score}, "
                "cashflow_negative={cashflow_negative}"
                "{correction}",
            ),
        ]
    )

    primary_llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.2,
        openai_api_key=settings.openai_api_key or "dummy_key",
    )

    backup_llm = ChatOpenAI(
        model=settings.openai_mini_model,
        temperature=0.2,
        openai_api_key=settings.openai_api_key or "dummy_key",
    )

    # Attach fallback to handle transient API downtime
    llm = primary_llm.with_fallbacks([backup_llm])
    chain = prompt | llm | parser

    logger.info(
        "Invoking Profiling Pipeline (%s) | RAG chunks: %d | Sources: %s",
        settings.openai_model,
        len(retrieved_chunks),
        source_filenames,
    )

    payload = {
        "metrics": metrics.model_dump_json(indent=2),
        "rag_context": rag_context_block,
        "sources": str(source_filenames),
        "format_instructions": parser.get_format_instructions(),
        "risk_profile": metrics.risk_profile,
        "health_score": metrics.financial_health_score,
        "cashflow_negative": metrics.is_cashflow_negative,
        "correction": "",
    }

    try:
        narrative: ProfileNarrative = chain.invoke(payload)
    except (OutputParserException, ValueError) as first_error:
        # One corrective retry with the validation error fed back. A rejected
        # product name is usually a near miss the model can fix when told
        # exactly what was wrong — failing the whole request on the first
        # attempt would be needlessly brittle.
        logger.warning(
            "Profile validation failed, retrying once with corrective feedback: %s",
            first_error,
        )
        payload["correction"] = (
            "\n\n## Correction Required\n"
            "Your previous response was rejected with this error:\n"
            f"{first_error}\n"
            "Return a corrected response that satisfies the schema exactly."
        )
        narrative = chain.invoke(payload)

    profile = CustomerProfile(
        **narrative.model_dump(),
        financial_health_score=metrics.financial_health_score,
        risk_profile=metrics.risk_profile,
        retrieved_sources=source_filenames,
    )

    logger.info(
        "Profile built | Persona: '%s' | Score: %d | Risk: %s | Primary: '%s' | Secondary: '%s'",
        profile.financial_persona,
        profile.financial_health_score,
        profile.risk_profile,
        profile.primary_product,
        profile.secondary_product,
    )
    return profile
