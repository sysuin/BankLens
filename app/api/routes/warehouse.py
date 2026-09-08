"""
The warehouse from the outside: what the role may ask, and what was run.

    GET /warehouse/templates    templates this role may run (name, description, examples)
    GET /warehouse/metrics      the semantic layer's metric definitions
    GET /warehouse/query-log    every template run for this tenant, newest first
    POST /warehouse/query       run one template by name with parameters (role-checked)
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.deps import Principal, current_user
from app.warehouse import query as wq
from app.warehouse.semantic import describe, load_layer

router = APIRouter(prefix="/warehouse", tags=["warehouse"])


class QueryRequest(BaseModel):
    template: str = Field(max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)


@router.get("/templates")
async def templates(principal: Principal = Depends(current_user)) -> list[dict]:
    return describe(principal.role)


@router.get("/metrics")
async def metrics(principal: Principal = Depends(current_user)) -> list[dict]:
    return [
        {
            "name": m.name,
            "view": m.view,
            "expression": m.expression,
            "unit": m.unit,
            "description": m.description,
        }
        for m in load_layer().metrics.values()
    ]


@router.get("/query-log")
async def query_log(
    limit: int = Query(100, ge=1, le=500),
    principal: Principal = Depends(current_user),
) -> list[dict]:
    return await wq.query_log(principal.tenant_id, limit=limit)


@router.post("/query")
async def run_query(
    body: QueryRequest, principal: Principal = Depends(current_user)
) -> dict:
    try:
        result = await wq.run_template(
            body.template,
            tenant_id=principal.tenant_id,
            role=principal.role,
            actor=principal.email,
            params=body.params,
        )
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown template")
    except wq.TemplateNotAllowed as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc))
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    payload = result.as_dict()
    payload["answer"] = wq.format_answer(result)
    payload["statement_id"] = payload["params"].get("statement_id")
    if payload["statement_id"]:
        payload["statement_id"] = str(uuid.UUID(payload["statement_id"]))
    return payload
