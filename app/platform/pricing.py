"""
Token prices, so every span can carry a cost.

USD per one million tokens (input, output), OpenAI list prices as of
September 2026. A model not listed costs 0 and is reported as unpriced, so
a missing row shows up as a gap in the numbers rather than as a silent zero.
Local models (Phase 4) are priced at 0 on purpose: their cost is the box
they run on, which is not a per-token number.
"""

from __future__ import annotations

PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def is_priced(model: str) -> bool:
    return _key(model) in PRICES_PER_MILLION


def _key(model: str) -> str:
    """Strip dated suffixes: gpt-4o-2024-08-06 → gpt-4o."""
    name = model.lower()
    for known in sorted(PRICES_PER_MILLION, key=len, reverse=True):
        if name == known or name.startswith(known + "-"):
            return known
    return name


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Cost of one call, in USD, rounded to a millionth of a dollar."""
    price_in, price_out = PRICES_PER_MILLION.get(_key(model), (0.0, 0.0))
    return round((tokens_in * price_in + tokens_out * price_out) / 1_000_000, 6)
