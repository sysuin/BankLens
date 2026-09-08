"""
Profile response cache.

Generation runs at temperature 0.2, so the same statement can produce a
slightly different narrative on every run — and costs a GPT-4o call each time.
For a given (metrics, retrieved context, prompt, model) the profile is a pure
function in intent, so it is cached on exactly that key.

This is exact-key response caching, not semantic caching, and the name is kept
honest on purpose: semantic caching matches *similar* inputs via embeddings,
which would be wrong here — two statements differing by one transaction are
different customers and must not share a profile. The inputs are already
canonical (computed metrics), so exact keying is both correct and free.

Every input that could change the output is hashed into the key — including
the system prompt file and the model name — so editing the prompt or switching
models invalidates the cache automatically. Same philosophy as the vector
index fingerprint: never serve results whose inputs have moved.
"""

import hashlib
import json
from pathlib import Path

from app.core.config import settings
from app.core.context import current_tenant, tenant_scope
from app.core.logger import get_logger
from app.pipeline.agent import SYSTEM_PROMPT_PATH, CustomerProfile, build_profile
from app.pipeline.analyzer import FinancialMetrics

logger = get_logger(__name__)


def profile_cache_key(
    metrics: FinancialMetrics, retrieved_chunks: list[dict], tenant: str | None = None
) -> str:
    """Hash every input that determines the generated profile."""
    digest = hashlib.sha256()
    # The tenant is part of the key: two banks can retrieve identical text
    # and still must never share a cached recommendation.
    digest.update((tenant or current_tenant()).encode("utf-8"))
    digest.update(metrics.model_dump_json().encode("utf-8"))
    for chunk in retrieved_chunks:
        digest.update(chunk["source"].encode("utf-8"))
        digest.update(chunk["content"].encode("utf-8"))
    try:
        digest.update(SYSTEM_PROMPT_PATH.read_bytes())
    except OSError:
        pass
    from app.platform import gateway

    digest.update(settings.openai_model.encode("utf-8"))
    digest.update(gateway.primary_name().encode("utf-8"))
    return digest.hexdigest()


def _cache_path(key: str) -> Path:
    return Path(settings.profile_cache_dir) / f"{key}.json"


def read_cached_profile(key: str) -> CustomerProfile | None:
    """Load a cached profile, or None on miss or any corruption."""
    path = _cache_path(key)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CustomerProfile.model_validate(data)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - a corrupt entry is a miss, not an error
        logger.warning(
            "Discarding corrupt profile cache entry %s (%s).", path.name, exc
        )
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def write_cached_profile(key: str, profile: CustomerProfile) -> None:
    """Persist a profile under its key. Never fatal."""
    try:
        Path(settings.profile_cache_dir).mkdir(parents=True, exist_ok=True)
        _cache_path(key).write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write profile cache entry (%s).", exc)


# ── Postgres backend (shared by the API and the worker, tenant-scoped) ──────


def _pg_conn(tenant: str):
    """App-role connection with the tenant pinned, or None if no database."""
    import psycopg

    from app.db.session import _resolve_urls

    try:
        app_url, _ = _resolve_urls()
    except Exception:  # noqa: BLE001 - no database configured
        return None
    conninfo = app_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    conn = psycopg.connect(conninfo)
    tenant_id = conn.execute(
        "SELECT id FROM tenants WHERE slug = %s", (tenant,)
    ).fetchone()
    if tenant_id is None:
        conn.close()
        return None
    conn.execute("SELECT set_config('app.tenant_id', %s, false)", (str(tenant_id[0]),))
    return conn, tenant_id[0]


def _backend() -> str:
    backend = settings.profile_cache_backend.strip().lower()
    if backend in ("postgres", "file"):
        return backend
    return (
        "postgres" if (settings.database_url or settings.database_admin_url) else "file"
    )


def read_cached_profile_pg(tenant: str, key: str) -> CustomerProfile | None:
    try:
        handle = _pg_conn(tenant)
        if handle is None:
            return None
        conn, _ = handle
        with conn:
            row = conn.execute(
                "UPDATE profile_cache SET hits = hits + 1, last_hit_at = now() "
                "WHERE cache_key = %s RETURNING profile_json",
                (key,),
            ).fetchone()
        conn.close()
        if row is None:
            return None
        return CustomerProfile.model_validate(row[0])
    except Exception as exc:  # noqa: BLE001 - a broken cache is a miss
        logger.warning("Postgres profile cache read failed (%s).", exc)
        return None


def write_cached_profile_pg(tenant: str, key: str, profile: CustomerProfile) -> None:
    try:
        handle = _pg_conn(tenant)
        if handle is None:
            return
        conn, tenant_id = handle
        with conn:
            conn.execute(
                "INSERT INTO profile_cache (tenant_id, cache_key, model, prompt_version, "
                "profile_json) VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (tenant_id, cache_key) DO UPDATE SET profile_json = "
                "EXCLUDED.profile_json, last_hit_at = now()",
                (
                    tenant_id,
                    key,
                    settings.openai_model,
                    hashlib.sha256(SYSTEM_PROMPT_PATH.read_bytes()).hexdigest()[:12],
                    json.dumps(profile.model_dump(mode="json")),
                ),
            )
        conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Postgres profile cache write failed (%s).", exc)


def cached_build_profile(
    metrics: FinancialMetrics,
    retrieved_chunks: list[dict],
    tenant: str | None = None,
) -> tuple[CustomerProfile, bool]:
    """
    build_profile with an exact-key response cache in front.

    Returns:
        (profile, from_cache) — the flag lets the UI say a result was cached
        rather than silently pretending a fresh generation happened.
    """
    resolved = tenant or current_tenant()
    if not settings.profile_cache_enabled:
        return build_profile(metrics, retrieved_chunks, tenant=resolved), False

    key = profile_cache_key(metrics, retrieved_chunks, resolved)
    backend = _backend()
    # Validation of a cached profile checks product names against the
    # tenant's catalogue, so the tenant must be in scope while reading.
    with tenant_scope(resolved):
        cached = (
            read_cached_profile_pg(resolved, key)
            if backend == "postgres"
            else read_cached_profile(key)
        )
    if cached is not None:
        logger.info(
            "Profile cache HIT (%s, %s…) — skipping LLM call.", backend, key[:12]
        )
        return cached, True

    profile = build_profile(metrics, retrieved_chunks, tenant=resolved)
    if backend == "postgres":
        write_cached_profile_pg(resolved, key, profile)
    else:
        write_cached_profile(key, profile)
    logger.info("Profile cache MISS (%s…) — generated and stored.", key[:12])
    return profile, False
