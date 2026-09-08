"""
Prompt and model version registry.

Every profile already records the prompt hash and model it was made with.
This table is the other direction: which prompt versions and models this
deployment has actually run, since when, and how often. Insert-or-bump on
every model call, so the registry is never a document someone forgot to
update.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from sqlalchemy import select

from app.core.config import settings
from app.db.models import PromptVersion
from app.db.session import unscoped_app_session
from app.pipeline.agent import SYSTEM_PROMPT_PATH

PROMPT_NAME = "system_prompt"


def prompt_fingerprint() -> tuple[str, str, int]:
    """(short version, full sha256, chars) of the system prompt file."""
    try:
        raw = SYSTEM_PROMPT_PATH.read_bytes()
    except OSError:
        raw = b""
    sha = hashlib.sha256(raw).hexdigest()
    return sha[:12], sha, len(raw)


async def register_use(model: str | None = None) -> str:
    """Record that the current prompt ran against `model`. Returns the version."""
    version, sha, chars = prompt_fingerprint()
    model = model or settings.openai_model
    async with unscoped_app_session() as session:
        row = (
            await session.execute(
                select(PromptVersion).where(
                    PromptVersion.prompt_name == PROMPT_NAME,
                    PromptVersion.version == version,
                    PromptVersion.model == model,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            session.add(
                PromptVersion(
                    prompt_name=PROMPT_NAME,
                    version=version,
                    sha256=sha,
                    chars=chars,
                    model=model,
                    uses=1,
                )
            )
        else:
            row.uses += 1
            row.last_seen_at = datetime.now(timezone.utc)
    return version


async def list_versions() -> list[dict]:
    async with unscoped_app_session() as session:
        rows = (
            (
                await session.execute(
                    select(PromptVersion).order_by(PromptVersion.first_seen_at.desc())
                )
            )
            .scalars()
            .all()
        )
        return [
            {
                "prompt_name": r.prompt_name,
                "version": r.version,
                "sha256": r.sha256,
                "chars": r.chars,
                "model": r.model,
                "first_seen_at": r.first_seen_at,
                "last_seen_at": r.last_seen_at,
                "uses": r.uses,
            }
            for r in rows
        ]
