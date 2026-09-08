"""
Guardrails as running code.

Four guards, all deterministic, all explainable, all cheap enough to run on
every input before any model is involved:

    scan_injection(text)      instruction-like text: override, exfiltration,
                              impersonated authority, role markers, encoding
                              tricks. Weighted pattern families; a score at or
                              above the block score is a block.
    scan_statement(df)        the uploaded statement is attacker-controlled
                              text. Flagged descriptions are replaced with a
                              marker so the model never sees them; amounts and
                              dates are kept so the numbers do not change.
    scope_gate(question, tenant)   the chat abstains on questions whose
                              content words are not in the bank's vocabulary,
                              measured as coverage, not similarity.
    guard_sql(sql, allowed)   only single SELECTs over allow-listed relations,
                              no side effects, no session tricks. Used by the
                              Phase 6 warehouse path; tested now.

Plus `scan_output(text)`: a model's narrative must not echo an injected
instruction or leak a PII shape. The graph's guardrails node calls it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

NEUTRALISED = "[removed: instruction-like text]"

# ── Injection patterns ───────────────────────────────────────────────────────
#
# (family, weight, regex). Weights add up per text; families do not double
# count. 1.0 = one clear phrase of that family is enough to block on its own.

_FAMILIES: list[tuple[str, float, re.Pattern]] = [
    (
        "override",
        1.0,
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instructions?|"
            r"rules?|guidelines?|prompt|policy|policies|guardrails?|previous|prior|above)\b",
            re.I,
        ),
    ),
    (
        "override",
        1.0,
        re.compile(
            r"\b(new|updated|real)\s+(system\s+)?(instructions?|prompt)\s*:", re.I
        ),
    ),
    (
        "role_hijack",
        1.0,
        re.compile(
            r"\b(you are now|from now on|act as|pretend (to be|you are)|"
            r"developer mode|jailbreak|do anything now|\bDAN\b)\b",
            re.I,
        ),
    ),
    (
        "role_marker",
        1.0,
        re.compile(
            r"(<\|im_start\|>|<\|system\|>|\[INST\]|<<SYS>>|###\s*(system|assistant)|"
            r"^\s*(system|assistant)\s*:)",
            re.I | re.M,
        ),
    ),
    (
        "authority",
        1.0,
        re.compile(
            r"\b(approve|approved|mark|set|record)\b.{0,30}\b(the\s+)?(loan|credit|"
            r"application|income|risk|review|decision|customer)\b.{0,30}\b(approved|"
            r"low risk|verified|cleared|as approved)\b",
            re.I,
        ),
    ),
    (
        "authority",
        0.7,
        re.compile(
            r"\b(as|i am|this is)\s+(the|your|a)\s+(bank|reviewer|admin|administrator|"
            r"manager|compliance|system|developer|auditor)\b",
            re.I,
        ),
    ),
    (
        "exfiltration",
        1.0,
        re.compile(
            r"\b(reveal|print|show|output|repeat|leak|dump|display)\b.{0,30}\b(system\s+"
            r"prompt|instructions|api key|secret|password|token|credentials?)\b",
            re.I,
        ),
    ),
    (
        # Cross-tenant or bulk data requests block on their own.
        "exfiltration",
        1.0,
        re.compile(
            r"\b(other|all)\s+(customers?|tenants?|banks?)\b.{0,30}\b(data|"
            r"statements?|balances?|records?|accounts?)\b"
            r"|\b(statements?|data|records?|balances?)\s+(for|of|from)\s+"
            r"(other|all)\s+(customers?|tenants?|banks?)\b",
            re.I,
        ),
    ),
    (
        "tool_abuse",
        1.0,
        re.compile(
            r"\b(call|invoke|run|execute|use)\s+(the\s+)?(tool|function|sql|query|"
            r"command)\b.{0,40}\b(delete|drop|update|insert|grant|transfer|approve)\b",
            re.I,
        ),
    ),
    (
        "encoding",
        0.6,
        re.compile(
            r"\b(base64|rot13|decode (this|the following)|translate the following "
            r"instructions?)\b",
            re.I,
        ),
    ),
    (
        "encoding",
        0.6,
        re.compile(r"[​‌‍⁠﻿]"),  # zero-width characters
    ),
    (
        # A long base64-looking token in a bank statement description is never
        # legitimate. No trailing \b: '=' padding is not a word character.
        "encoding",
        1.0,
        re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{60,}={0,2}(?![A-Za-z0-9+/])"),
    ),
]

_PII_SHAPES = re.compile(
    r"(\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"  # 16-digit card
    r"|\b\d{3}-\d{2}-\d{4}\b"  # SSN
    r"|\b[A-Z]{5}\d{4}[A-Z]\b"  # PAN
    r"|\b\d{9,18}\b"  # long account number
    r"|\b[\w.+-]+@[\w-]+\.[\w.]+\b)",  # email
)


@dataclass(frozen=True)
class Verdict:
    blocked: bool
    score: float
    families: tuple[str, ...]
    matched: tuple[str, ...]  # short excerpts, safe to log

    def as_dict(self) -> dict:
        return {
            "blocked": self.blocked,
            "score": round(self.score, 2),
            "families": list(self.families),
            "matched": list(self.matched),
        }


def scan_injection(text: str) -> Verdict:
    """Score one text for instruction-like content."""
    if not text:
        return Verdict(False, 0.0, (), ())
    score = 0.0
    families: list[str] = []
    matched: list[str] = []
    seen: set[str] = set()
    for family, weight, pattern in _FAMILIES:
        m = pattern.search(text)
        if not m:
            continue
        if family not in seen:
            seen.add(family)
            score += weight
            families.append(family)
        excerpt = m.group(0)[:60].replace("\n", " ")
        matched.append(excerpt)
    blocked = score >= settings.guardrail_injection_block_score
    return Verdict(blocked, score, tuple(families), tuple(matched[:5]))


@dataclass
class StatementScan:
    flagged_rows: int = 0
    total_rows: int = 0
    families: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "flagged_rows": self.flagged_rows,
            "total_rows": self.total_rows,
            "families": sorted(set(self.families)),
            "examples": self.examples[:5],
        }


def scan_statement(df: pd.DataFrame) -> tuple[pd.DataFrame, StatementScan]:
    """
    Neutralise instruction-like descriptions in a statement.

    The statement is untrusted input: a PDF can carry text that tries to
    instruct the model. Flagged descriptions are replaced with a marker; the
    row's date, amount and type are untouched, so income, expenses and the
    risk band are exactly what they would have been. The category for a
    neutralised row falls to the rules' default.
    """
    result = df.copy()
    scan = StatementScan(total_rows=int(len(result)))
    if "description" not in result.columns:
        return result, scan
    for idx, raw in result["description"].items():
        verdict = scan_injection(str(raw))
        if verdict.blocked:
            scan.flagged_rows += 1
            scan.families.extend(verdict.families)
            scan.examples.append(str(raw)[:80])
            result.at[idx, "description"] = NEUTRALISED
    if scan.flagged_rows:
        logger.warning(
            "Statement scan neutralised %d/%d rows (%s)",
            scan.flagged_rows,
            scan.total_rows,
            ", ".join(sorted(set(scan.families))),
        )
    return result, scan


# ── Chat input ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChatGuard:
    allowed: bool
    reason: str  # "ok" | "injection" | "out_of_scope"
    question: str  # what the model may see (PII redacted)
    detail: dict


def guard_chat_question(
    question: str, tenant: str, *, check_scope: bool = True
) -> ChatGuard:
    """
    Injection first, then scope, then PII redaction of what passes.

    The chat route runs the intent router between injection and scope: a
    question that maps to a vetted template is in scope by definition, so
    it calls this with check_scope=False after the router has spoken.
    """
    from app.pipeline.sanitizer import sanitize_text

    verdict = scan_injection(question)
    if verdict.blocked:
        return ChatGuard(False, "injection", "", verdict.as_dict())
    in_scope, coverage, unknown = (
        scope_gate(question, tenant) if check_scope else (True, 1.0, [])
    )
    if not in_scope:
        return ChatGuard(
            False,
            "out_of_scope",
            "",
            {"coverage": round(coverage, 2), "unknown_terms": unknown[:8]},
        )
    return ChatGuard(
        True, "ok", sanitize_text(question), {"coverage": round(coverage, 2)}
    )


# ── Scope gate ───────────────────────────────────────────────────────────────

_STOPWORDS = frozenset(
    """a an the and or but if then of for to in on at by with from as is are was were be been
    being this that these those it its i me my we our you your he she they them his her their
    what which who whom whose when where why how do does did done can could should would will
    shall may might must have has had having not no yes so than too very just about into over
    under again further once here there all any both each few more most other some such only
    own same s t don now please tell show give me explain describe list""".split()
)

_DOMAIN_TERMS = frozenset(
    """income expense expenses spending spend savings saving rate ratio risk band score health
    statement statements transaction transactions customer customers credit debit deficit surplus
    monthly month year period balance cash cashflow flow category categories top largest total
    average recommend recommendation recommendations product products suitable suit eligible
    eligibility fee fees interest rate rates loan loans deposit deposits account accounts card
    cards savings investment invest emi rent salary bill bills food shopping travel utilities
    medical insurance declared observed discrepancy verification review reviewer approve approved
    rejected pitch hook why what explain rated rating low medium high essential discretionary
    bank banking rm relationship manager profile persona liquidity budget computed
    metric metrics number numbers figure figures summary totals amount amounts""".split()
)


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z][a-z0-9'-]{1,}", text.lower())
    out = []
    for w in words:
        w = w.strip("'-")
        if len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        if w and w not in _STOPWORDS:
            out.append(w)
    return out


@lru_cache(maxsize=16)
def tenant_vocabulary(tenant: str) -> frozenset[str]:
    """Content words of the tenant's catalogue plus the shared finance vocabulary."""
    from app.pipeline.rag import kb_dir

    words: set[str] = set(_DOMAIN_TERMS)
    directory: Path = kb_dir(tenant)
    for md in directory.glob("*.md") if directory.exists() else []:
        try:
            words.update(_tokens(md.read_text(encoding="utf-8")))
        except OSError:
            continue
    return frozenset(words)


def scope_gate(question: str, tenant: str) -> tuple[bool, float, list[str]]:
    """
    (in_scope, coverage, unknown terms).

    Coverage = share of the question's content words found in the tenant's
    vocabulary. Similarity scores do not separate on- from off-topic
    questions (measured: 0.125 vs 0.126); vocabulary coverage does. The
    floor is strict: "capital of Australia" is half finance vocabulary and
    must still abstain.
    """
    toks = _tokens(question)
    if not toks:
        return True, 1.0, []  # nothing to judge; let the model handle pleasantries
    vocab = tenant_vocabulary(tenant)
    known = [
        t
        for t in toks
        if t in vocab or any(t.startswith(v) for v in vocab if len(v) > 4)
    ]
    coverage = len(known) / len(toks)
    unknown = [t for t in toks if t not in known]
    return coverage > settings.guardrail_scope_floor, coverage, unknown


# ── SQL allow-list ───────────────────────────────────────────────────────────

_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|vacuum|"
    r"analyze|call|do|execute|prepare|listen|notify|lock|set|reset|show|explain|"
    r"set_config|current_setting|pg_[a-z_]+|information_schema|lo_[a-z_]+)\b",
    re.I,
)
_SQL_RELATIONS = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][\w.]*)", re.I)
_SQL_CTES = re.compile(r"(?:\bwith\s+|,\s*)([a-zA-Z_]\w*)\s+as\s*\(", re.I)


def guard_sql(sql: str, allowed_relations: set[str], max_limit: int = 500) -> Verdict:
    """
    Accept only one read-only SELECT over allow-listed relations.

    Deterministic checks, no parser: single statement, starts with SELECT or
    WITH, no forbidden keywords, every FROM/JOIN target allow-listed (names
    defined by WITH count as allowed), and a LIMIT at or below the cap.
    """
    text = sql.strip().rstrip(";").strip()
    reasons: list[str] = []
    if ";" in text:
        reasons.append("multiple statements")
    if not re.match(r"^(select|with)\b", text, re.I):
        reasons.append("not a SELECT")
    forbidden = sorted({m.group(0).lower() for m in _SQL_FORBIDDEN.finditer(text)})
    if forbidden:
        reasons.append("forbidden: " + ", ".join(forbidden))
    relations = {m.group(1).lower() for m in _SQL_RELATIONS.finditer(text)}
    ctes = {m.group(1).lower() for m in _SQL_CTES.finditer(text)}
    permitted = {a.lower() for a in allowed_relations} | ctes
    bad = sorted(r for r in relations if r not in permitted)
    if bad:
        reasons.append("relation not allowed: " + ", ".join(bad))
    if not relations:
        reasons.append("no relation")
    # A literal LIMIT is capped here; a bound LIMIT (:limit) is capped when
    # the parameter is bound (see app.warehouse.query.bind_params).
    limit = re.search(r"\blimit\s+(\d+|:[a-zA-Z_]\w*)\b", text, re.I)
    if limit is None:
        reasons.append("no LIMIT")
    elif limit.group(1).isdigit() and int(limit.group(1)) > max_limit:
        reasons.append(f"LIMIT above {max_limit}")
    if "--" in text or "/*" in text:
        reasons.append("comment")
    blocked = bool(reasons)
    return Verdict(blocked, float(len(reasons)), tuple(reasons), (text[:60],))


# ── Output ───────────────────────────────────────────────────────────────────


def scan_output(text: str) -> Verdict:
    """A narrative must not carry an instruction or a PII shape back out."""
    verdict = scan_injection(text)
    pii = _PII_SHAPES.search(text or "")
    if pii:
        return Verdict(
            True,
            verdict.score + 1.0,
            verdict.families + ("pii",),
            verdict.matched + (pii.group(0)[:4] + "…",),
        )
    return verdict
