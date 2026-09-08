"""
Business policy that both the graph and the evals must agree on.

Kept free of database and framework imports so the eval suite can load it
without booting anything. The graph's guardrail node reads the same dict.
"""

from __future__ import annotations

# A customer in deficit must never be steered into more unsecured credit,
# whatever else the recommendation says. Per tenant catalogue: anything
# unsecured that adds debt.
CREDIT_PRODUCTS_FORBIDDEN_IN_DEFICIT: dict[str, frozenset[str]] = {
    "meridian": frozenset({"credit_card.md", "personal_loan.md"}),
    "harbor": frozenset(
        {"cashback_credit_card.md", "small_business_line.md", "auto_loan.md"}
    ),
}


def forbidden_in_deficit(tenant: str) -> frozenset[str]:
    return CREDIT_PRODUCTS_FORBIDDEN_IN_DEFICIT.get(tenant, frozenset())
