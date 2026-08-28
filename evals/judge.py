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


# The prose fields a juror scores. Everything else on the profile is either a
# product name (already validated against the knowledge base) or a number the
# application computed, so neither is the judge's to second-guess.
NARRATIVE_FIELDS: tuple[str, ...] = (
    "income_stability_analysis",
    "spending_pattern_breakdown",
    "credit_risk_assessment",
    "primary_reason",
    "secondary_reason",
)

# Jurors per profile. Odd, so a majority always exists.
JUDGE_PANEL_SIZE = 3


class FieldVerdict(BaseModel):
    """One juror's reading of one narrative field."""

    field: str = Field(description="The profile field being judged.")
    grounded: bool = Field(
        description="True if every factual claim in this field is supported."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Verbatim quotes from this field that the evidence does not support.",
    )


class JurorVerdict(BaseModel):
    """One juror's full reading of a profile."""

    fields: list[FieldVerdict] = Field(
        description="One entry per narrative field you were given."
    )
    contradicts_assigned_risk: bool = Field(
        description="True if the narrative argues for a different risk rating than the one assigned."
    )
    reasoning: str = Field(description="One or two sentences explaining the verdict.")


class GroundednessVerdict(BaseModel):
    """
    The panel's result after majority vote.

    Keeps the shape the callers already consume — `grounded`,
    `unsupported_claims`, `contradicts_assigned_risk`, `reasoning` — and adds
    the per-field detail that makes a failure actionable.
    """

    grounded: bool = Field(
        description="True when no field was condemned by a majority of jurors."
    )
    unsupported_claims: list[str] = Field(default_factory=list)
    contradicts_assigned_risk: bool = Field(
        description="True when a majority of jurors saw the narrative dispute its rating."
    )
    reasoning: str = Field(description="Summary of how the panel voted.")
    condemned_fields: list[str] = Field(
        default_factory=list,
        description="Fields a majority of jurors found ungrounded.",
    )
    juror_count: int = Field(
        default=0, description="How many jurors returned a verdict."
    )


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

Judge each narrative field separately and return one entry per field, using the
field names exactly as given. A weak sentence in one field says nothing about
the others, and lumping them together loses the only information that would
tell someone what to fix.
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


def _run_juror(
    profile: CustomerProfile,
    metrics: FinancialMetrics,
    context_block: str,
    derived_block: str,
    seed_hint: str,
) -> JurorVerdict:
    """Ask one juror for a per-field verdict."""
    parser = PydanticOutputParser(pydantic_object=JurorVerdict)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _risk_banded_prompt() + "\n\n{format_instructions}"),
            (
                "human",
                "<computed_metrics>\n{metrics}\n</computed_metrics>\n\n"
                "<derived_figures>\n{derived}\n</derived_figures>\n\n"
                "<assigned_risk>{risk}</assigned_risk>\n\n"
                "<retrieved_context>\n{context}\n</retrieved_context>\n\n"
                "<generated_profile>\n{profile}\n</generated_profile>\n\n"
                "<fields_to_judge>\n{fields}\n</fields_to_judge>\n\n"
                "{seed_hint}",
            ),
        ]
    )

    # Jurors must disagree to be worth polling. At temperature 0 three calls on
    # identical input return the same answer three times, which is a single
    # opinion wearing a rosette. Reproducibility is deliberately traded here;
    # every deterministic check in the suite stays exactly reproducible.
    judge = ChatOpenAI(
        model=settings.openai_mini_model,
        temperature=0.4,
        openai_api_key=settings.openai_api_key,
    )

    return (prompt | judge | parser).invoke(
        {
            "metrics": metrics.model_dump_json(indent=2),
            "derived": derived_block,
            "risk": metrics.risk_profile,
            "context": context_block,
            "profile": profile.model_dump_json(indent=2),
            "fields": "\n".join(f"- {name}" for name in NARRATIVE_FIELDS),
            "seed_hint": seed_hint,
            "format_instructions": parser.get_format_instructions(),
        }
    )


def judge_groundedness(
    profile: CustomerProfile,
    metrics: FinancialMetrics,
    retrieved_chunks: list[dict],
    panel_size: int = JUDGE_PANEL_SIZE,
) -> GroundednessVerdict:
    """
    Score one generated profile for groundedness, by majority of a juror panel.

    A single juror was measured condemning claims it had just restated as
    correct — calling a verbatim "88%" unsupported in the same verdict that
    said the ratio was 88%. Those errors are not reproducible across samples,
    which is exactly what a panel exploits: a false positive has to be invented
    by two independent jurors to survive, while a real one usually is not hard
    to see twice.

    Scoring is per field, so a shaky sentence in one field cannot condemn the
    other four, and a failure names the field to look at.

    Raises:
        RuntimeError: If no API key is configured — the caller is expected to
                      skip the judged layer rather than silently pass it.
    """
    if not settings.openai_api_key:
        raise RuntimeError("judge_groundedness requires OPENAI_API_KEY to be set")

    context_block = "\n\n---\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['content']}" for chunk in retrieved_chunks
    )
    derived_block = _derived_figures(metrics)

    verdicts: list[JurorVerdict] = []
    for index in range(panel_size):
        hint = f"You are juror {index + 1} of {panel_size}, reviewing independently."
        try:
            verdicts.append(
                _run_juror(profile, metrics, context_block, derived_block, hint)
            )
        except Exception as exc:  # noqa: BLE001 - a juror who cannot vote abstains
            logger.warning("Juror %d failed to return a verdict (%s).", index + 1, exc)

    if not verdicts:
        raise RuntimeError("no juror returned a verdict")

    majority = len(verdicts) // 2 + 1

    # Tally condemnations per field. A juror who omits a field is treated as
    # having no objection to it rather than as condemning it.
    against: dict[str, list[str]] = {name: [] for name in NARRATIVE_FIELDS}
    counts: dict[str, int] = {name: 0 for name in NARRATIVE_FIELDS}
    for verdict in verdicts:
        for field_verdict in verdict.fields:
            name = field_verdict.field
            if name not in counts or field_verdict.grounded:
                continue
            counts[name] += 1
            against[name].extend(field_verdict.unsupported_claims)

    condemned = [name for name, count in counts.items() if count >= majority]
    disputes_risk = sum(1 for v in verdicts if v.contradicts_assigned_risk) >= majority

    tally = ", ".join(f"{name} {counts[name]}/{len(verdicts)}" for name in condemned)
    reasoning = f"{len(verdicts)} jurors, majority {majority}. " + (
        f"Condemned by majority: {tally}."
        if condemned
        else "No field condemned by majority."
    )

    result = GroundednessVerdict(
        grounded=not condemned,
        unsupported_claims=[claim for name in condemned for claim in against[name]],
        contradicts_assigned_risk=disputes_risk,
        reasoning=reasoning,
        condemned_fields=condemned,
        juror_count=len(verdicts),
    )

    logger.info(
        "Panel verdict | jurors=%d | grounded=%s | contradicts_risk=%s | condemned=%s",
        result.juror_count,
        result.grounded,
        result.contradicts_assigned_risk,
        result.condemned_fields or "none",
    )
    return result
