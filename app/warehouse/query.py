"""
Run a vetted template as the read-only chat role, log it, format the answer.

The only SQL that ever executes here is the text of a template from the
semantic layer, with named bind parameters. The connection is the
`banklens_chat` role, which can read tenant-filtered views and nothing
else, and the tenant is pinned with SET LOCAL inside the transaction. Every
run writes a `query_log` row (template name, parameters, rows, duration,
actor, role) so "what did the chat query?" is a table, not a guess.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from app.core.logger import get_logger
from app.db.models import QueryLog
from app.db.session import chat_engine, tenant_session
from app.platform.tracing import ATTR_STATEMENT_ID, span
from app.warehouse.semantic import DEFAULT_LIMIT, MAX_LIMIT, Template, load_layer

logger = get_logger(__name__)


class TemplateNotAllowed(PermissionError):
    """The role may not run this template."""


@dataclass
class QueryResult:
    template: str
    params: dict[str, Any]
    columns: list[str]
    rows: list[dict[str, Any]]
    duration_ms: int

    def as_dict(self) -> dict:
        return {
            "template": self.template,
            "params": self.params,
            "columns": self.columns,
            "rows": self.rows,
            "duration_ms": self.duration_ms,
        }


def _clean(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def bind_params(template: Template, supplied: dict[str, Any]) -> dict[str, Any]:
    """Only declared parameters, with the limit clamped."""
    bound: dict[str, Any] = {}
    for name in template.params:
        if name == "limit":
            try:
                limit = int(supplied.get("limit", DEFAULT_LIMIT))
            except (TypeError, ValueError):
                limit = DEFAULT_LIMIT
            bound["limit"] = max(1, min(limit, MAX_LIMIT))
        elif name == "statement_id":
            bound["statement_id"] = str(supplied["statement_id"])
        else:
            if name not in supplied:
                raise ValueError(f"template '{template.name}' needs '{name}'")
            bound[name] = str(supplied[name])[:64]
    return bound


async def run_template(
    name: str,
    *,
    tenant_id: uuid.UUID,
    role: str,
    actor: str,
    params: dict[str, Any],
    question: str | None = None,
) -> QueryResult:
    template = load_layer().templates.get(name)
    if template is None:
        raise KeyError(f"unknown template '{name}'")
    if not template.allows(role):
        raise TemplateNotAllowed(
            f"template '{name}' is for {sorted(template.roles)}, not '{role}'"
        )
    bound = bind_params(template, params)

    started = time.perf_counter()
    with span(
        "warehouse.query",
        **{
            "warehouse.template": name,
            "warehouse.role": role,
            ATTR_STATEMENT_ID: bound.get("statement_id"),
        },
    ) as current:
        async with chat_engine().connect() as conn:
            async with conn.begin():
                await conn.execute(
                    text("SELECT set_config('app.tenant_id', :t, true)"),
                    {"t": str(tenant_id)},
                )
                result = await conn.execute(text(template.sql), bound)
                columns = list(result.keys())
                rows = [
                    {col: _clean(val) for col, val in zip(columns, row)}
                    for row in result.fetchall()
                ]
        duration_ms = int((time.perf_counter() - started) * 1000)
        current.set_attribute("warehouse.rows", len(rows))

    async with tenant_session(tenant_id) as session:
        session.add(
            QueryLog(
                tenant_id=tenant_id,
                template=name,
                params={k: v for k, v in bound.items() if k != "statement_id"},
                actor=actor,
                role=role,
                statement_id=(
                    uuid.UUID(bound["statement_id"])
                    if "statement_id" in bound
                    else None
                ),
                rows=len(rows),
                duration_ms=duration_ms,
                question_hash=(
                    hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
                    if question
                    else None
                ),
            )
        )
    logger.info(
        "warehouse template=%s role=%s rows=%d %d ms",
        name,
        role,
        len(rows),
        duration_ms,
    )
    return QueryResult(name, bound, columns, rows, duration_ms)


# ── Formatting: the numbers come from the rows, never from a model ──────────


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}" if abs(value) < 1000 else f"{value:,.0f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "—"
    return str(value)


def format_answer(result: QueryResult) -> str:
    template = load_layer().templates[result.template]
    if not result.rows:
        return f"No rows for {template.description.lower().rstrip('.')}."
    if template.answer != "table":
        try:
            return " ".join(template.answer.format(**result.rows[0]).split())
        except (KeyError, ValueError, TypeError):
            pass  # fall through to a table
    header = " | ".join(result.columns)
    divider = " | ".join("---" for _ in result.columns)
    body = "\n".join(
        " | ".join(_fmt(row[c]) for c in result.columns) for row in result.rows
    )
    return f"**{template.description}**\n\n| {header} |\n| {divider} |\n" + "\n".join(
        f"| {line} |" for line in body.splitlines()
    )


async def query_log(tenant_id: uuid.UUID, limit: int = 100) -> list[dict]:
    from sqlalchemy import select

    async with tenant_session(tenant_id) as session:
        rows = (
            (
                await session.execute(
                    select(QueryLog).order_by(QueryLog.id.desc()).limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return [
            {
                "id": r.id,
                "template": r.template,
                "params": r.params,
                "actor": r.actor,
                "role": r.role,
                "statement_id": r.statement_id,
                "rows": r.rows,
                "duration_ms": r.duration_ms,
                "created_at": r.created_at,
            }
            for r in rows
        ]


def params_json(params: dict[str, Any]) -> str:
    return json.dumps(params, default=str, sort_keys=True)
