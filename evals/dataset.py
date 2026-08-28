"""
Golden dataset for BankLens evaluation.

Three real statements is a smoke test, not an evaluation set. This module
synthesizes a spread of cases from those three by rescaling consumption
spending to hit a *target savings rate*, which works because the metric is
exactly determined:

    savings_rate_pct = 100 * (1 - total_expenses / total_income)

so scaling every non-Savings debit by

    k = total_income * (1 - target/100) / current_consumption_expenses

lands the statement on any savings rate we choose. Controlling the input that
way is what lets the expected outcome be written down by hand rather than
recomputed with the same code under test.

Expected risk bands live in RISK_SWEEP as an explicit literal table. That
duplication is deliberate: an eval that derives its expectation by calling
compute_risk_profile() proves nothing. Stating the rule a second time, by
hand, is what catches an accidental change to the first.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BASE_STATEMENTS = {
    "high_saver": "sample_1_high_saver.csv",
    "active_spender": "sample_2_active_spender.csv",
    "cashflow_stressed": "sample_3_cashflow_stressed.csv",
}

# (target savings rate %, expected risk band). Hand-written from the business
# rule — Low at 30% and above, Medium from 10% up to 30%, High below 10% —
# including both sides of each boundary.
RISK_SWEEP: list[tuple[float, str]] = [
    (-60.0, "High"),
    (-40.0, "High"),
    (-25.0, "High"),
    (-10.0, "High"),
    (-1.0, "High"),
    (0.0, "High"),
    (5.0, "High"),
    (9.9, "High"),
    (10.0, "Medium"),
    (12.0, "Medium"),
    (15.0, "Medium"),
    (20.0, "Medium"),
    (25.0, "Medium"),
    (29.9, "Medium"),
    (30.0, "Low"),
    (35.0, "Low"),
    (45.0, "Low"),
    (60.0, "Low"),
    (75.0, "Low"),
    (90.0, "Low"),
]

# A customer in deficit must never be steered into more unsecured credit,
# whatever else the recommendation says. This is the one assertion in the set
# that encodes a business rule rather than a formula.
CREDIT_PRODUCTS_FORBIDDEN_IN_DEFICIT = {"credit_card.md", "personal_loan.md"}


@dataclass(frozen=True)
class EvalCase:
    """One golden case: an input statement plus what we expect from it."""

    case_id: str
    archetype: str
    target_savings_rate: float | None
    expected_risk: str

    # Populated only for the sampled subset that the paid eval layer runs.
    include_in_llm_eval: bool = False

    # Knowledge base files this customer must never be recommended.
    forbidden_products: frozenset[str] = field(default_factory=frozenset)

    # Set for handcrafted cases that bypass synthesis.
    literal_frame: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.literal_frame is None and self.target_savings_rate is None:
            raise ValueError(f"{self.case_id}: needs a target rate or a literal frame")


def load_base(archetype: str) -> pd.DataFrame:
    """Load one of the three reference statements by archetype name."""
    filename = BASE_STATEMENTS[archetype]
    return pd.read_csv(DATA_DIR / filename)


def categorize_rules_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize using the keyword rules alone, skipping the Stage 2 LLM pass.

    Evaluation inputs must be identical on every run and must not cost money to
    construct, so the golden set is built against the deterministic half of the
    categorizer. The LLM stage is exercised separately in the paid layer.
    """
    from app.pipeline.categorizer import categorize

    result = df.copy()
    result["category"] = result["description"].map(categorize)
    return result


def retarget_savings_rate(
    df: pd.DataFrame,
    categorized: pd.DataFrame,
    target_savings_rate: float,
) -> pd.DataFrame:
    """
    Rescale consumption debits so the statement lands on a target savings rate.

    Args:
        df: The raw statement.
        categorized: The same statement after categorization — needed because
                     debits categorized as Savings are internal transfers and
                     are excluded from consumption expenses.
        target_savings_rate: Desired savings rate on a 0-100 scale. May be
                             negative to model a customer in deficit.

    Returns:
        A copy of the statement with consumption debit amounts scaled.
    """
    is_credit = categorized["type"].str.strip().str.lower() == "credit"
    is_consumption_debit = ~is_credit & (categorized["category"] != "Savings")

    total_income = float(categorized.loc[is_credit, "amount"].sum())
    current_expenses = float(categorized.loc[is_consumption_debit, "amount"].sum())

    if total_income <= 0 or current_expenses <= 0:
        raise ValueError("Cannot retarget a statement with no income or no expenses")

    target_expenses = total_income * (1.0 - target_savings_rate / 100.0)
    scale = target_expenses / current_expenses

    result = df.copy()
    result.loc[is_consumption_debit.values, "amount"] = (
        result.loc[is_consumption_debit.values, "amount"] * scale
    ).round(2)
    return result


# ── Handcrafted edge cases ────────────────────────────────────────────────────


def _frame(rows: list[tuple[str, str, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "description", "amount", "type"])


EDGE_CASES: list[EvalCase] = [
    EvalCase(
        case_id="edge_no_income",
        archetype="synthetic",
        target_savings_rate=None,
        # No income means no meaningful savings rate; the analyzer defaults it
        # to 0.0, which must band as High rather than crash or read as healthy.
        expected_risk="High",
        literal_frame=_frame(
            [
                ("2024-03-01", "Apartment Rent Payment Landlord", 25000.0, "Debit"),
                ("2024-03-05", "Swiggy Food Delivery", 1800.0, "Debit"),
            ]
        ),
    ),
    EvalCase(
        case_id="edge_all_income_no_spend",
        archetype="synthetic",
        target_savings_rate=None,
        expected_risk="Low",
        literal_frame=_frame(
            [
                ("2024-03-01", "TechCorp Salary Credit", 90000.0, "Credit"),
                ("2024-03-15", "Freelance Payment Received", 20000.0, "Credit"),
            ]
        ),
    ),
    EvalCase(
        case_id="edge_single_transaction",
        archetype="synthetic",
        target_savings_rate=None,
        expected_risk="Low",
        literal_frame=_frame(
            [("2024-03-01", "TechCorp Salary Credit", 75000.0, "Credit")]
        ),
    ),
    EvalCase(
        case_id="edge_breakeven",
        archetype="synthetic",
        target_savings_rate=None,
        # Spends exactly what is earned: savings rate 0%, unambiguously High.
        expected_risk="High",
        literal_frame=_frame(
            [
                ("2024-03-01", "TechCorp Salary Credit", 50000.0, "Credit"),
                ("2024-03-02", "Apartment Rent Payment Landlord", 50000.0, "Debit"),
            ]
        ),
    ),
    EvalCase(
        case_id="edge_savings_transfer_is_not_expense",
        archetype="synthetic",
        target_savings_rate=None,
        # Money moved into an FD is not consumption. Income 100k, consumption
        # 10k, so the savings rate is 90% and the rating must be Low despite
        # total debits equalling 70k.
        expected_risk="Low",
        literal_frame=_frame(
            [
                ("2024-03-01", "TechCorp Salary Credit", 100000.0, "Credit"),
                ("2024-03-02", "Auto Transfer to FD Fixed Deposit", 60000.0, "Debit"),
                ("2024-03-05", "Swiggy Food Delivery", 10000.0, "Debit"),
            ]
        ),
    ),
]


# ── Case set assembly ─────────────────────────────────────────────────────────

# Sampled for the paid layer: one clear case per risk band, plus both sides of
# the Medium/Low boundary and a deficit case for the credit guardrail.
_LLM_EVAL_SAMPLE = {
    "high_saver@75.0",
    "active_spender@29.9",
    "active_spender@30.0",
    "cashflow_stressed@-40.0",
    "cashflow_stressed@12.0",
    "high_saver@5.0",
}


def build_cases() -> list[EvalCase]:
    """Assemble the full golden set: the swept grid plus the edge cases."""
    cases: list[EvalCase] = []

    for archetype in BASE_STATEMENTS:
        for target, expected_risk in RISK_SWEEP:
            case_id = f"{archetype}@{target}"
            cases.append(
                EvalCase(
                    case_id=case_id,
                    archetype=archetype,
                    target_savings_rate=target,
                    expected_risk=expected_risk,
                    include_in_llm_eval=case_id in _LLM_EVAL_SAMPLE,
                    forbidden_products=(
                        frozenset(CREDIT_PRODUCTS_FORBIDDEN_IN_DEFICIT)
                        if target < 0
                        else frozenset()
                    ),
                )
            )

    cases.extend(EDGE_CASES)
    return cases


def materialize(case: EvalCase) -> pd.DataFrame:
    """
    Produce the raw transaction frame for a case.

    Synthesized cases are rescaled from their base statement; handcrafted edge
    cases are returned as written.
    """
    if case.literal_frame is not None:
        return case.literal_frame.copy()

    base = load_base(case.archetype)
    categorized = categorize_rules_only(base)
    return retarget_savings_rate(base, categorized, case.target_savings_rate)
