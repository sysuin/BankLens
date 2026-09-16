"""
The model gateway: one interface, several providers, no vendor lock-in.

    chat_model(role)   a LangChain chat model for "primary" or "mini" work,
                       with retries (exponential backoff + jitter), a circuit
                       breaker per provider, cost-aware fallback, and a span
                       per call carrying model, tokens and cost
    embeddings()       the matching embeddings client
    status()           what the gateway would do right now, and why

Providers are OpenAI-compatible endpoints, which is why Ollama needs no
extra dependency: it speaks the same wire protocol on /v1. Selection order
is settings.llm_provider first ("auto" = OpenAI when a key exists, otherwise
Ollama), then the other provider as fallback. A provider is skipped when its
circuit is open (too many recent failures) or, for hosted providers, when
the tenant in scope has spent its daily budget. Budget is read from the
`spans` table, so what the gateway enforces is exactly what tracing
recorded, not a parallel counter that can drift.

Everything here is synchronous on purpose: the pipeline runs in worker
threads and LangChain's sync API is what the pipeline modules use.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from opentelemetry import context as otel_context
from opentelemetry import trace

from app.core import context as request_context
from app.core.config import settings
from app.core.logger import get_logger
from app.platform.pricing import cost_usd

logger = get_logger(__name__)

OPENAI = "openai"
OLLAMA = "ollama"


class GatewayUnavailable(RuntimeError):
    """No provider can serve this call right now (and this is why)."""


class BudgetExceeded(GatewayUnavailable):
    """The tenant spent its daily budget and no cost-free provider exists."""


# ── Providers ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    model: str
    mini_model: str
    embedding_model: str
    api_key: str
    base_url: str | None
    cost_free: bool

    def model_for(self, role: str) -> str:
        return self.mini_model if role == "mini" else self.model


def providers() -> dict[str, ProviderSpec]:
    """Every provider that is configured, keyed by name."""
    out: dict[str, ProviderSpec] = {}
    if settings.openai_api_key:
        out[OPENAI] = ProviderSpec(
            name=OPENAI,
            model=settings.openai_model,
            mini_model=settings.openai_mini_model,
            embedding_model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            base_url=None,
            cost_free=False,
        )
    if settings.ollama_base_url:
        out[OLLAMA] = ProviderSpec(
            name=OLLAMA,
            model=settings.ollama_model,
            mini_model=settings.ollama_mini_model,
            embedding_model=settings.ollama_embedding_model,
            api_key="ollama",
            base_url=settings.ollama_base_url,
            cost_free=True,
        )
    return out


def primary_name() -> str:
    wanted = settings.llm_provider.strip().lower()
    if wanted in (OPENAI, OLLAMA):
        return wanted
    return OPENAI if settings.openai_api_key else OLLAMA


def provider_order() -> list[ProviderSpec]:
    """Primary first, then the others as fallbacks (if enabled)."""
    available = providers()
    order = []
    first = primary_name()
    if first in available:
        order.append(available[first])
    if settings.llm_fallback_enabled:
        order += [spec for name, spec in available.items() if name != first]
    return order


# ── Circuit breaker ──────────────────────────────────────────────────────────


@dataclass
class _Circuit:
    failures: int = 0
    opened_at: float | None = None
    last_error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if time.monotonic() - self.opened_at >= settings.gateway_cooldown_s:
            return "half_open"
        return "open"


_circuits: dict[str, _Circuit] = {}
_circuits_lock = threading.Lock()


def circuit(name: str) -> _Circuit:
    with _circuits_lock:
        if name not in _circuits:
            _circuits[name] = _Circuit()
        return _circuits[name]


def record_success(name: str) -> None:
    c = circuit(name)
    with c.lock:
        c.failures = 0
        c.opened_at = None
        c.last_error = None


def record_failure(name: str, error: str) -> None:
    c = circuit(name)
    with c.lock:
        c.failures += 1
        c.last_error = error[:200]
        if c.failures >= settings.gateway_failure_threshold and c.opened_at is None:
            c.opened_at = time.monotonic()
            logger.warning(
                "circuit OPEN for provider %s after %d failures", name, c.failures
            )
        elif c.opened_at is not None:
            # A failed half-open probe re-arms the cooldown.
            c.opened_at = time.monotonic()


def reset_circuits() -> None:
    with _circuits_lock:
        _circuits.clear()


# ── Budget (read from the spans table) ───────────────────────────────────────

_budget_cache: dict[str, tuple[float, float, float]] = (
    {}
)  # tenant -> (checked, spent, limit)
_BUDGET_TTL_S = 10.0
_budgets_enabled = False


def enable_budgets(enabled: bool = True) -> None:
    """Called by the API and the worker at startup: platform mode has a database."""
    global _budgets_enabled
    _budgets_enabled = enabled


def _tenant_row(tenant_slug: str) -> tuple[uuid.UUID, float] | None:
    """(tenant id, daily budget) for a slug, via the API role with the tenant pinned."""
    import psycopg

    from app.db.session import _resolve_urls

    app_url, _ = _resolve_urls()
    conninfo = app_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    with psycopg.connect(conninfo) as conn:
        row = conn.execute(
            "SELECT id, daily_budget_usd FROM tenants WHERE slug = %s", (tenant_slug,)
        ).fetchone()
        if row is None:
            return None
        tenant_id, limit = row
        conn.execute("SELECT set_config('app.tenant_id', %s, true)", (str(tenant_id),))
        spent = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM spans "
            "WHERE tenant_id = %s AND start_time >= date_trunc('day', now())",
            (tenant_id,),
        ).fetchone()[0]
    return (
        uuid.UUID(str(tenant_id)),
        float(limit or settings.tenant_daily_budget_usd),
        float(spent),
    )


def budget_for(tenant_slug: str | None) -> tuple[float, float] | None:
    """(spent today, daily limit) for the tenant, cached for a few seconds."""
    if not tenant_slug:
        return None
    if not (_budgets_enabled or settings.database_url or settings.database_admin_url):
        # Direct (single-process) mode has no platform database: no budgets.
        return None
    now = time.monotonic()
    cached = _budget_cache.get(tenant_slug)
    if cached and now - cached[0] < _BUDGET_TTL_S:
        return cached[1], cached[2]
    try:
        row = _tenant_row(tenant_slug)
    except Exception as exc:  # noqa: BLE001 - no database, no budget enforcement
        logger.debug("budget lookup skipped: %s", exc)
        return None
    if row is None:
        return None
    _, limit, spent = row
    _budget_cache[tenant_slug] = (now, spent, limit)
    return spent, limit


def invalidate_budget_cache() -> None:
    _budget_cache.clear()


# ── Selection ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Choice:
    spec: ProviderSpec
    reason: str
    fallbacks: tuple[ProviderSpec, ...]


def choose(role: str = "primary") -> Choice:
    """Pick the provider for a call right now, and remember why."""
    order = provider_order()
    if not order:
        raise GatewayUnavailable(
            "no model provider configured: set OPENAI_API_KEY or run Ollama"
        )
    tenant = request_context._tenant.get()
    budget = budget_for(tenant)
    over_budget = budget is not None and budget[0] >= budget[1]

    usable: list[tuple[ProviderSpec, str]] = []
    skipped: list[str] = []
    for spec in order:
        state = circuit(spec.name).state()
        if state == "open":
            skipped.append(f"{spec.name}: circuit open")
            continue
        if over_budget and not spec.cost_free:
            skipped.append(f"{spec.name}: tenant budget spent")
            continue
        reason = "primary" if spec is order[0] else "fallback"
        if state == "half_open":
            reason += " (half-open probe)"
        if over_budget and spec.cost_free:
            reason += " (budget)"
        usable.append((spec, reason))

    if not usable:
        if over_budget:
            raise BudgetExceeded(
                f"tenant '{tenant}' spent its daily budget "
                f"(${budget[0]:.4f} of ${budget[1]:.2f}); skipped: {'; '.join(skipped)}"
            )
        raise GatewayUnavailable("every provider is unavailable: " + "; ".join(skipped))

    first, reason = usable[0]
    return Choice(spec=first, reason=reason, fallbacks=tuple(s for s, _ in usable[1:]))


# ── Per-call span + breaker bookkeeping ──────────────────────────────────────


class GatewayCallback(BaseCallbackHandler):
    """Opens an `llm.call` span per model call; feeds the circuit breaker."""

    def __init__(self, spec: ProviderSpec, model: str, reason: str) -> None:
        self.spec = spec
        self.model = model
        self.reason = reason
        self._spans: dict[Any, tuple[Any, Any]] = {}

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        self._start(run_id)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._start(run_id)

    def _start(self, run_id) -> None:
        from app.platform.tracing import inherited_attributes

        attrs = inherited_attributes()
        attrs.update(
            {
                "llm.provider": self.spec.name,
                "llm.model": self.model,
                "gateway.reason": self.reason,
            }
        )
        span = trace.get_tracer("banklens").start_span("llm.call", attributes=attrs)
        token = otel_context.attach(trace.set_span_in_context(span))
        self._spans[run_id] = (span, token)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        span, token = self._spans.pop(run_id, (None, None))
        tokens_in = tokens_out = 0
        usage = (response.llm_output or {}).get("token_usage") or {}
        if usage:
            tokens_in = int(usage.get("prompt_tokens", 0) or 0)
            tokens_out = int(usage.get("completion_tokens", 0) or 0)
        else:
            for generation in response.generations:
                for g in generation:
                    meta = (
                        getattr(getattr(g, "message", None), "usage_metadata", None)
                        or {}
                    )
                    tokens_in += int(meta.get("input_tokens", 0) or 0)
                    tokens_out += int(meta.get("output_tokens", 0) or 0)
        record_success(self.spec.name)
        if span is not None:
            span.set_attribute("llm.tokens.in", tokens_in)
            span.set_attribute("llm.tokens.out", tokens_out)
            span.set_attribute(
                "llm.cost_usd",
                (
                    0.0
                    if self.spec.cost_free
                    else cost_usd(self.model, tokens_in, tokens_out)
                ),
            )
            span.end()
            otel_context.detach(token)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        span, token = self._spans.pop(run_id, (None, None))
        record_failure(self.spec.name, f"{type(error).__name__}: {error}")
        if span is not None:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)[:200]))
            span.end()
            otel_context.detach(token)


# ── Public API ───────────────────────────────────────────────────────────────


def _client(
    spec: ProviderSpec, model: str, temperature: float, reason: str
) -> ChatOpenAI:
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "api_key": spec.api_key,
        "timeout": settings.llm_timeout_s,
        "max_retries": 0,  # retries are the gateway's job, so they are visible
        "callbacks": [GatewayCallback(spec, model, reason)],
    }
    if spec.base_url:
        kwargs["base_url"] = spec.base_url
    return ChatOpenAI(**kwargs)


def chat_model(
    role: str = "primary", temperature: float = 0.0, *, streaming: bool = False
):
    """
    A chat model for `role` ("primary" or "mini").

    Non-streaming callers get retries with exponential backoff and jitter,
    then fallbacks: the same provider's mini model (for primary work), then
    the other providers. Streaming callers (the chat) get the chosen
    provider's raw client, because retry wrappers do not stream tokens; the
    breaker and the span still apply through the callback.
    """
    choice = choose(role)
    spec = choice.spec
    model = spec.model_for(role)
    llm = _client(spec, model, temperature, choice.reason)
    if streaming:
        return llm

    fallbacks = []
    if role == "primary" and spec.mini_model != model:
        fallbacks.append(_client(spec, spec.mini_model, temperature, "fallback (mini)"))
    for other in choice.fallbacks:
        fallbacks.append(_client(other, other.model_for(role), temperature, "fallback"))

    resilient = llm.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": 0.5, "max": 8.0},
    )
    if fallbacks:
        resilient = resilient.with_fallbacks(
            [
                f.with_retry(stop_after_attempt=2, wait_exponential_jitter=True)
                for f in fallbacks
            ]
        )
    return resilient


def embeddings() -> OpenAIEmbeddings:
    """Embeddings from the primary provider (never falls back: vectors must not mix)."""
    order = provider_order()
    if not order:
        raise GatewayUnavailable("no model provider configured for embeddings")
    spec = order[0]
    kwargs: dict[str, Any] = {"model": spec.embedding_model, "api_key": spec.api_key}
    if spec.base_url:
        kwargs["base_url"] = spec.base_url
        # Ollama has no tiktoken-based length check.
        kwargs["check_embedding_ctx_length"] = False
    return OpenAIEmbeddings(**kwargs)


def embedding_signature() -> str:
    """Goes into the vector index fingerprint: switching provider = new index."""
    order = provider_order()
    if not order:
        return "none"
    return f"{order[0].name}:{order[0].embedding_model}"


def available() -> bool:
    return bool(provider_order())


def status(tenant_slug: str | None = None) -> dict:
    """What the gateway would do right now, for /platform/gateway."""
    order = provider_order()
    budget = budget_for(tenant_slug)
    rows = []
    for spec in order:
        c = circuit(spec.name)
        rows.append(
            {
                "name": spec.name,
                "model": spec.model,
                "mini_model": spec.mini_model,
                "embedding_model": spec.embedding_model,
                "cost_free": spec.cost_free,
                "circuit": c.state(),
                "consecutive_failures": c.failures,
                "last_error": c.last_error,
            }
        )
    try:
        choice = choose("primary")
        would_use = {"provider": choice.spec.name, "reason": choice.reason}
    except GatewayUnavailable as exc:
        would_use = {"provider": None, "reason": str(exc)}
    return {
        "primary": primary_name(),
        "fallback_enabled": settings.llm_fallback_enabled,
        "providers": rows,
        "would_use": would_use,
        "budget": (
            {"spent_today_usd": round(budget[0], 4), "daily_limit_usd": budget[1]}
            if budget
            else None
        ),
        "checked_at": datetime.now(timezone.utc),
    }
