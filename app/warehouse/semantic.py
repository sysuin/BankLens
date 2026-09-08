"""
The semantic layer, loaded and validated.

`semantic_layer.yaml` is the single definition of every number the chat may
quote and every query it may run. Loading it validates each template against
the SQL allow-list (`guard_sql`) and the declared relations, so a template
that could read a base table, forget its LIMIT, or slip in a second
statement fails at import, long before a question arrives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from app.platform.guardrails import guard_sql

LAYER_PATH = Path(__file__).resolve().parent / "semantic_layer.yaml"

DEFAULT_LIMIT = 10
MAX_LIMIT = 50


class SemanticLayerError(ValueError):
    """The layer file is unsafe or inconsistent."""


@dataclass(frozen=True)
class Template:
    name: str
    roles: frozenset[str]
    description: str
    triggers: tuple[str, ...]
    params: tuple[str, ...]
    sql: str
    answer: str  # a format string over the first row, or "table"

    def allows(self, role: str) -> bool:
        return role in self.roles


@dataclass(frozen=True)
class Metric:
    name: str
    view: str
    expression: str
    unit: str
    description: str


@dataclass(frozen=True)
class Layer:
    relations: frozenset[str]
    metrics: dict[str, Metric]
    templates: dict[str, Template]


_BIND = re.compile(r"(?<!:):([a-zA-Z_]\w*)")


def _validate_template(name: str, tpl: Template, relations: frozenset[str]) -> None:
    verdict = guard_sql(tpl.sql, set(relations), max_limit=MAX_LIMIT)
    if verdict.blocked:
        raise SemanticLayerError(
            f"template '{name}' rejected by the SQL allow-list: {'; '.join(verdict.families)}"
        )
    binds = set(_BIND.findall(tpl.sql))
    declared = set(tpl.params)
    if binds - declared:
        raise SemanticLayerError(
            f"template '{name}' binds {sorted(binds - declared)} that it does not declare"
        )
    if declared - binds:
        raise SemanticLayerError(
            f"template '{name}' declares {sorted(declared - binds)} it never binds"
        )
    if not tpl.roles:
        raise SemanticLayerError(f"template '{name}' allows no role")
    if "limit" in tpl.params and not re.search(r"\blimit\s+:limit\b", tpl.sql, re.I):
        raise SemanticLayerError(f"template '{name}' must apply :limit as its LIMIT")


@lru_cache(maxsize=1)
def load_layer(path: Path | None = None) -> Layer:
    raw = yaml.safe_load((path or LAYER_PATH).read_text(encoding="utf-8"))
    relations = frozenset(raw.get("relations", []))
    metrics = {
        name: Metric(
            name=name,
            view=spec["view"],
            expression=spec["expression"],
            unit=spec.get("unit", ""),
            description=spec.get("description", ""),
        )
        for name, spec in (raw.get("metrics") or {}).items()
    }
    for metric in metrics.values():
        if metric.view not in relations:
            raise SemanticLayerError(
                f"metric '{metric.name}' points at '{metric.view}', not a declared relation"
            )
    templates: dict[str, Template] = {}
    for name, spec in (raw.get("templates") or {}).items():
        tpl = Template(
            name=name,
            roles=frozenset(spec.get("roles", [])),
            description=spec.get("description", ""),
            triggers=tuple(t.lower() for t in spec.get("triggers", [])),
            params=tuple(spec.get("params", [])),
            sql=spec["sql"].strip(),
            answer=str(spec.get("answer", "table")).strip(),
        )
        _validate_template(name, tpl, relations)
        templates[name] = tpl
    if not templates:
        raise SemanticLayerError("no templates defined")
    return Layer(relations=relations, metrics=metrics, templates=templates)


def templates_for(role: str) -> list[Template]:
    return [t for t in load_layer().templates.values() if t.allows(role)]


def describe(role: str) -> list[dict]:
    """What the console lists as 'questions I can answer from the numbers'."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "examples": list(t.triggers[:3]),
            "params": [p for p in t.params if p != "statement_id"],
        }
        for t in templates_for(role)
    ]
