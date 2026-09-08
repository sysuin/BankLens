"""
Bias check: the numbers must not depend on who the customer seems to be.

    python -m evals.bias_check          # exit 1 on any divergence
    (runs in CI on every push)

Every golden statement is rewritten for several demographic groups: a
name token from the group is appended to salary and transfer lines, and
merchant spellings are swapped for that group's local forms. Amounts,
dates and types are untouched. The deterministic pipeline then must give
**identical** risk bands and health scores across groups, and the guardrail
scan must flag no group more than another (a group's merchant names must
not look like injection).

This is a regression test for the doctrine "the numbers come from code",
and a real detector for the way bias creeps in: a merchant keyword that
collides with a name, a category rule that matches one language's word for
"rent" and not another's.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import pandas as pd

from app.pipeline.analyzer import compute_metrics
from app.platform.guardrails import scan_statement
from evals.dataset import build_cases, categorize_rules_only, materialize

# Group -> (name tokens, merchant respellings). Synthetic, illustrative.
GROUPS: dict[str, tuple[list[str], dict[str, str]]] = {
    "baseline": ([], {}),
    "hindi": (
        ["Priya", "Rohan", "Sunita"],
        {"Grocery": "Kirana", "Rent": "Kiraya", "Pharmacy": "Dawai", "Food": "Khana"},
    ),
    "tamil": (
        ["Karthik", "Meena", "Arun"],
        {"Grocery": "Maligai", "Rent": "Vaadagai", "Pharmacy": "Marundhu"},
    ),
    "arabic": (
        ["Fatima", "Omar", "Layla"],
        {"Grocery": "Baqala", "Rent": "Ijar", "Pharmacy": "Saydaliya"},
    ),
    "western": (
        ["Emma", "Liam", "Sophie"],
        {"Grocery": "Supermarket", "Rent": "Lease", "Pharmacy": "Chemist"},
    ),
}


def rewrite(
    df: pd.DataFrame, names: list[str], respell: dict[str, str]
) -> pd.DataFrame:
    out = df.copy()
    descs = []
    for i, d in enumerate(out["description"].astype(str)):
        text = d
        for old, new in respell.items():
            text = text.replace(old, new).replace(old.lower(), new.lower())
        if names and ("Salary" in d or "Transfer" in d or "Credit" in d):
            text = f"{text} - {names[i % len(names)]}"
        descs.append(text)
    out["description"] = descs
    return out


def main() -> int:
    cases = build_cases()
    divergences: list[str] = []
    bands: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    flagged: dict[str, int] = defaultdict(int)

    for case in cases:
        base = materialize(case)
        results = {}
        for group, (names, respell) in GROUPS.items():
            df = rewrite(base, names, respell)
            df, scan = scan_statement(df)
            flagged[group] += scan.flagged_rows
            # Rules only: deterministic and free, so CI needs no key and a
            # divergence can only come from the rules or the numbers.
            metrics = compute_metrics(categorize_rules_only(df))
            results[group] = (
                str(metrics.risk_profile),
                int(metrics.financial_health_score),
            )
            bands[group][str(metrics.risk_profile)] += 1
        reference = results["baseline"]
        for group, value in results.items():
            if value != reference:
                divergences.append(
                    f"{case.case_id}: {group} -> {value} vs baseline {reference}"
                )

    print(f"\nBias check: {len(cases)} golden statements x {len(GROUPS)} groups\n")
    print(f"{'group':<10}{'Low':>6}{'Medium':>8}{'High':>6}{'flagged rows':>14}")
    print("-" * 44)
    for group in GROUPS:
        row = bands[group]
        print(
            f"{group:<10}{row.get('Low', 0):>6}{row.get('Medium', 0):>8}"
            f"{row.get('High', 0):>6}{flagged[group]:>14}"
        )
    print("-" * 44)
    flag_spread = max(flagged.values()) - min(flagged.values()) if flagged else 0
    if divergences:
        print(f"\n{len(divergences)} divergence(s):")
        for line in divergences[:20]:
            print("  " + line)
    if flag_spread:
        print(f"\nguardrail scan flags differ across groups by {flag_spread} row(s)")
    ok = not divergences and flag_spread == 0
    print(
        "\n"
        + (
            "Risk bands and scores identical across groups; guardrail flags equal."
            if ok
            else "Bias check FAILED."
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
