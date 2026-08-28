"""
LLM-as-judge groundedness scoring for BankLens.

Deterministic checks can confirm that a recommended product exists and that its
document was retrieved. They cannot tell whether the *prose* asserts things the
retrieved context does not support — that requires reading, which is what a
judge model is for.

Design choices worth defending:

  - The judge runs on gpt-4o-mini, not on the model under test. Asking a model
    to grade its own output invites self-preference bias, and the judging task
    (does claim X appear in context Y) is far easier than the generation task.
  - Temperature 0, and the verdict is a structured object, so a run is
    reproducible and machine-readable.
  - The rubric asks for specific unsupported claims, not a score out of ten.
    A number gives you no way to act; a quoted sentence does.
  - The judge is told to ignore style entirely. Judges drift toward rewarding
    fluent writing if you let them.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logger import get_logger
from app.pipeline.agent import CustomerProfile
from app.pipeline.analyzer import (
    FinancialMetrics,
    RISK_LOW_MIN_SAVINGS_RATE,
    RISK_MEDIUM_MIN_SAVINGS_RATE,
)

logger = get_logger(__name__)


class GroundednessVerdict(BaseModel):
    """A judge's reading of whether a profile's claims are supported."""

    grounded: bool = Field(
        description="True only if every factual claim is supported by the metrics or the retrieved context."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Verbatim quotes of any claim not supported by the supplied evidence.",
    )
    contradicts_assigned_risk: bool = Field(
        description="True if the narrative argues for a different risk rating than the one assigned."
    )
    reasoning: str = Field(description="One or two sentences explaining the verdict.")


JUDGE_SYSTEM_PROMPT = """\
You are evaluating a bank customer profile for factual grounding. You are not
assessing whether it is well written, persuasive, or useful — ignore style
entirely, and do not reward fluency.

You are given three things: the customer's computed metrics, the product
documentation that was retrieved, and the generated profile.

You are also given a <derived_figures> block. It restates the same metrics in
the forms a profile actually quotes them in — ratios as percentages, category
totals pulled out of their nested structure. Anything appearing there is
supported, by definition. Do not re-derive it and do not second-guess it.

Mark a claim unsupported if it:
  - states a figure that appears neither in the metrics nor in the derived
    figures, and cannot be reached from them by simple arithmetic
  - describes a product feature absent from the retrieved documentation
  - asserts something about the customer's behaviour that the transaction
    metrics do not evidence

A ratio written as a percentage is the same figure, not a new one: an
expense_to_income_ratio of 0.88 fully supports "88%". A category total quoted
from the derived figures is supported however it is phrased. Flagging either of
these is the most common way this evaluation goes wrong.

Do NOT mark a claim unsupported merely because it is an interpretation or a
judgement, provided the underlying figures are real. "Disciplined saver" is a
fair reading of a 75% savings rate. The same applies to "balanced",
"significant", "essential", "discretionary", "disciplined" and similar
characterisations: these are readings of real figures, not factual claims, and
flagging them is out of scope. Splitting spending into essential and
discretionary is an expected part of the task, not an invented statistic.

A sentence that pairs a real figure with an interpretation is supported if the
figure is real. Judge the figure, not the framing.

If you find yourself writing that a figure is correct and then listing it as
unsupported, it is supported. Report it as supported.

Also report whether the narrative argues against the risk rating it was
assigned. The rating is computed by the application and is not the model's to
dispute; a narrative that reasons toward a different rating is a defect.

The rating is banded on savings rate alone, by this exact rule:

    Low     savings rate >= LOW_MIN%
    Medium  savings rate >= MEDIUM_MIN% and < LOW_MIN%
    High    savings rate <  MEDIUM_MIN%

Expenses exclude internal savings transfers, so the savings rate and the
expense-to-income ratio are the same measurement:
savings_rate = 100 - (expense_to_income_ratio * 100). An 88% expense ratio is a
12% savings rate, which is Medium — not High, however stressful 88% sounds.

Apply that rule and nothing else. Do not substitute your own view of what
counts as risky: a narrative describing an 88% expense ratio as stretched but
solvent is explaining a Medium rating correctly, not contradicting it. Set
contradicts_assigned_risk only when the narrative explicitly argues for a band
different from the one it was given.

Quote unsupported claims verbatim so they can be traced.
"""


def _risk_banded_prompt() -> str:
    """
    Fill the banding thresholds into the rubric from the analyzer's constants.

    The judge was reading an 88% expense ratio as High risk because it had been
    told to check the rating without being told what the rating means. Sourcing
    the numbers from RISK_LOW_MIN_SAVINGS_RATE / RISK_MEDIUM_MIN_SAVINGS_RATE
    keeps the rubric correct if the bands are ever retuned.
    """
    return JUDGE_SYSTEM_PROMPT.replace(
        "LOW_MIN", f"{RISK_LOW_MIN_SAVINGS_RATE:g}"
    ).replace("MEDIUM_MIN", f"{RISK_MEDIUM_MIN_SAVINGS_RATE:g}")


def _derived_figures(metrics: FinancialMetrics) -> str:
    """
    Restate the metrics in the forms a generated profile actually quotes.

    The judge kept rejecting true claims for two reasons, and both are
    presentation rather than reasoning. Ratios are stored as decimals but
    written as percentages, so "88%" looked absent next to 0.88. And
    top_categories is a list of dicts, so an exact category total looked
    invented next to its own nested value.

    Handing the judge these forms directly removes the derivation step instead
    of asking it to be better at arithmetic.
    """
    lines = [
        f"total_income = {metrics.total_income:,.2f}",
        f"total_expenses = {metrics.total_expenses:,.2f}",
        f"savings_amount = {metrics.savings_amount:,.2f}",
        f"savings_rate = {metrics.savings_rate_pct:.2f}%",
        f"expense_to_income_ratio = {metrics.expense_to_income_ratio:.4f}"
        f" = {metrics.expense_to_income_ratio * 100:.2f}% of income spent",
        f"share_of_income_saved = {metrics.savings_rate_pct:.2f}%",
        f"cashflow_negative = {metrics.is_cashflow_negative}",
        f"financial_health_score = {metrics.financial_health_score}",
        "top spending categories (exact totals):",
    ]
    for category in metrics.top_categories:
        lines.append(f"  - {category['category']} = {category['total_spent']:,.2f}")
    return "\n".join(lines)


def judge_groundedness(
    profile: CustomerProfile,
    metrics: FinancialMetrics,
    retrieved_chunks: list[dict],
) -> GroundednessVerdict:
    """
    Score one generated profile for groundedness.

    Raises:
        RuntimeError: If no API key is configured — the caller is expected to
                      skip the judged layer rather than silently pass it.
    """
    if not settings.openai_api_key:
        raise RuntimeError("judge_groundedness requires OPENAI_API_KEY to be set")

    parser = PydanticOutputParser(pydantic_object=GroundednessVerdict)

    context_block = "\n\n---\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['content']}" for chunk in retrieved_chunks
    )

    derived_block = _derived_figures(metrics)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _risk_banded_prompt() + "\n\n{format_instructions}"),
            (
                "human",
                "<computed_metrics>\n{metrics}\n</computed_metrics>\n\n"
                "<derived_figures>\n{derived}\n</derived_figures>\n\n"
                "<assigned_risk>{risk}</assigned_risk>\n\n"
                "<retrieved_context>\n{context}\n</retrieved_context>\n\n"
                "<generated_profile>\n{profile}\n</generated_profile>",
            ),
        ]
    )

    judge = ChatOpenAI(
        model=settings.openai_mini_model,
        temperature=0.0,
        openai_api_key=settings.openai_api_key,
    )

    verdict: GroundednessVerdict = (prompt | judge | parser).invoke(
        {
            "metrics": metrics.model_dump_json(indent=2),
            "derived": derived_block,
            "risk": metrics.risk_profile,
            "context": context_block,
            "profile": profile.model_dump_json(indent=2),
            "format_instructions": parser.get_format_instructions(),
        }
    )

    logger.info(
        "Judge verdict | grounded=%s | contradicts_risk=%s | unsupported=%d",
        verdict.grounded,
        verdict.contradicts_assigned_risk,
        len(verdict.unsupported_claims),
    )
    return verdict
