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
from app.pipeline.analyzer import FinancialMetrics

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

Mark a claim unsupported if it:
  - states a figure that does not appear in the metrics and cannot be derived
    from them by simple arithmetic
  - describes a product feature absent from the retrieved documentation
  - asserts something about the customer's behaviour that the transaction
    metrics do not evidence

Do NOT mark a claim unsupported merely because it is an interpretation or a
judgement, provided the underlying figures are real. "Disciplined saver" is a
fair reading of a 75% savings rate.

Also report whether the narrative argues against the risk rating it was
assigned. The rating is computed by the application and is not the model's to
dispute; a narrative that reasons toward a different rating is a defect.

Quote unsupported claims verbatim so they can be traced.
"""


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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JUDGE_SYSTEM_PROMPT + "\n\n{format_instructions}"),
            (
                "human",
                "<computed_metrics>\n{metrics}\n</computed_metrics>\n\n"
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
