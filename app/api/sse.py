"""Server-Sent Events over an async iterator of small dicts."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi.responses import StreamingResponse


async def _encode(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    async for item in events:
        kind = item.get("event", "message")
        body = {k: v for k, v in item.items() if k != "event"}
        yield f"event: {kind}\ndata: {json.dumps(body, default=str)}\n\n"


def sse_response(events: AsyncIterator[dict[str, Any]]) -> StreamingResponse:
    return StreamingResponse(
        _encode(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
