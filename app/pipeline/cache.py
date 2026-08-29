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
from app.core.logger import get_logger
from app.pipeline.agent import SYSTEM_PROMPT_PATH, CustomerProfile, build_profile
from app.pipeline.analyzer import FinancialMetrics

logger = get_logger(__name__)


def profile_cache_key(metrics: FinancialMetrics, retrieved_chunks: list[dict]) -> str:
    """Hash every input that determines the generated profile."""
    digest = hashlib.sha256()
    digest.update(metrics.model_dump_json().encode("utf-8"))
    for chunk in retrieved_chunks:
        digest.update(chunk["source"].encode("utf-8"))
        digest.update(chunk["content"].encode("utf-8"))
    try:
        digest.update(SYSTEM_PROMPT_PATH.read_bytes())
    except OSError:
        pass
    digest.update(settings.openai_model.encode("utf-8"))
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


def cached_build_profile(
    metrics: FinancialMetrics, retrieved_chunks: list[dict]
) -> tuple[CustomerProfile, bool]:
    """
    build_profile with an exact-key response cache in front.

    Returns:
        (profile, from_cache) — the flag lets the UI say a result was cached
        rather than silently pretending a fresh generation happened.
    """
    if not settings.profile_cache_enabled:
        return build_profile(metrics, retrieved_chunks), False

    key = profile_cache_key(metrics, retrieved_chunks)
    cached = read_cached_profile(key)
    if cached is not None:
        logger.info("Profile cache HIT (%s…) — skipping LLM call.", key[:12])
        return cached, True

    profile = build_profile(metrics, retrieved_chunks)
    write_cached_profile(key, profile)
    logger.info("Profile cache MISS (%s…) — generated and stored.", key[:12])
    return profile, False
