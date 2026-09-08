"""
Intent router: which template, with which parameters, how confident.

Deterministic on purpose (the Ops Copilot pattern): trigger phrases per
template, scored by how much of the question they explain, with a
confidence floor. Above the floor the question is answered from the
warehouse without a model; below it the question goes to the tool-calling
chat, which can still fetch metrics and search the catalogue. A role that
is not allowed a matched template gets a refusal, not a downgrade, so a
crafted RM question about the review queue is recorded as denied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.warehouse.semantic import Template, load_layer

_NUMBER = re.compile(
    r"\b(top|first|last|largest|biggest)\s+(\d{1,2})\b|\b(\d{1,2})\s+(largest|biggest|top|rows|items)\b",
    re.I,
)


@dataclass(frozen=True)
class Route:
    kind: str  # "template" | "chat" | "denied"
    template: Template | None = None
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""


def _score(question: str, template: Template) -> float:
    """
    Share of the question's words explained by the template's triggers.

    Every distinct matching trigger counts, so "declared vs observed income
    discrepancy" (three triggers of declared_vs_observed) beats the single
    word "income" of income_vs_expenses. Digits are dropped before matching
    so "top 3 categories" still hits "top categories".
    """
    q = " " + re.sub(r"[^a-z ]+", " ", re.sub(r"\d+", " ", question.lower())) + " "
    q = re.sub(r"\s+", " ", q)
    words = max(1, len(q.split()))
    explained = 0
    for trigger in template.triggers:
        if f" {trigger} " in q:
            explained += len(trigger.split())
    if not explained:
        return 0.0
    return round(min(explained / words + 0.5, 1.0), 3)


def _extract_limit(question: str) -> int | None:
    m = _NUMBER.search(question)
    if not m:
        return None
    for group in m.groups():
        if group and group.isdigit():
            return int(group)
    return None


def _extract_category(question: str, categories: list[str]) -> str | None:
    q = question.lower()
    # Longest category name first so "Rent & Housing" beats "Rent".
    for cat in sorted(categories, key=len, reverse=True):
        needle = cat.lower().replace("&", "and")
        if cat.lower() in q or needle in q or cat.lower().split()[0] in q.split():
            return cat
    return None


def route(
    question: str,
    *,
    role: str,
    statement_id: str | None,
    categories: list[str] | None = None,
) -> Route:
    layer = load_layer()
    scored = sorted(
        ((_score(question, t), t) for t in layer.templates.values()),
        key=lambda pair: pair[0],
        reverse=True,
    )
    confidence, template = scored[0] if scored else (0.0, None)
    if template is None or confidence < settings.warehouse_intent_floor:
        return Route(
            "chat", confidence=confidence, reason="no template above the floor"
        )
    if not template.allows(role):
        return Route(
            "denied",
            template=template,
            confidence=confidence,
            reason=f"'{template.name}' is for {sorted(template.roles)}, not '{role}'",
        )

    params: dict[str, Any] = {}
    if "statement_id" in template.params:
        if not statement_id:
            return Route("chat", confidence=confidence, reason="no statement in scope")
        params["statement_id"] = statement_id
    if "limit" in template.params:
        params["limit"] = _extract_limit(question) or 10
    if "category" in template.params:
        category = _extract_category(question, categories or [])
        if category is None:
            return Route(
                "chat",
                confidence=confidence,
                reason="category question without a known category",
            )
        params["category"] = category
    return Route("template", template=template, params=params, confidence=confidence)
