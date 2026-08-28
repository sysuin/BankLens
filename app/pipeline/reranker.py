"""
Reranking stage for BankLens retrieval.

Hybrid fusion gives a cheap, coarse ordering: dense search scores a query
against chunk embeddings computed independently of each other, and BM25 scores
term overlap. Neither ever looks at the query and a passage *together*. A
reranker does, which is why moving from "retrieve 4" to "retrieve 15, rerank to
4" is usually the largest single quality gain available to a RAG system for the
least code.

Two backends, because the right answer depends on where this runs:

  - "llm" (default): one listwise call to the mini model. No new dependencies,
    reuses the existing API key, ~1s of added latency.
  - "cross_encoder": a local sentence-transformers cross-encoder. Faster per
    query and free to run, but it pulls in torch — several hundred MB in the
    image and more memory than a t2.micro has spare. Opt-in only.

Reranking never breaks retrieval. Every failure path falls back to the fusion
order, because slightly worse ordering is a far better outcome than an error
page where a customer profile should be.
"""

from functools import lru_cache

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class RerankRanking(BaseModel):
    """Ordered candidate indices returned by the listwise LLM reranker."""

    ranking: list[int] = Field(
        description="Candidate indices, most relevant first. Use the numbers shown against each passage."
    )


RERANK_SYSTEM_PROMPT = """\
You rank banking product documentation by how well it answers a customer's
financial situation.

You are given a customer summary and a numbered list of candidate passages.
Return the indices of the most relevant passages, most relevant first.

Rank on whether the passage helps decide what to recommend to THIS customer:
  - a passage describing a product whose eligibility profile matches the
    customer's savings rate and expense ratio is highly relevant
  - a passage about a product the customer plainly does not qualify for, or
    that would be irresponsible for their position, is not relevant
  - generic boilerplate present in every product document is not relevant

Return only indices from the list you were given, with no duplicates.
"""


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _repair_ranking(ranking: list[int], candidate_count: int, top_k: int) -> list[int]:
    """
    Turn a model-proposed ordering into a usable one.

    Drops out-of-range and duplicate indices, then pads from the original order
    so the caller always receives exactly top_k results (or every candidate, if
    there are fewer). A reranker that silently returns three passages when the
    prompt was built for four is a subtle way to lose context.
    """
    seen: set[int] = set()
    cleaned: list[int] = []

    for index in ranking:
        if 0 <= index < candidate_count and index not in seen:
            seen.add(index)
            cleaned.append(index)

    for index in range(candidate_count):
        if len(cleaned) >= top_k:
            break
        if index not in seen:
            seen.add(index)
            cleaned.append(index)

    return cleaned[:top_k]


# ── Backend: listwise LLM ─────────────────────────────────────────────────────


def _rerank_with_llm(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Rerank via a single listwise call to the mini model."""
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    parser = PydanticOutputParser(pydantic_object=RerankRanking)

    passages = "\n\n".join(
        f"[{index}] (source: {chunk.get('source', 'unknown')})\n"
        f"{_truncate(chunk['content'], settings.rerank_passage_chars)}"
        for index, chunk in enumerate(candidates)
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RERANK_SYSTEM_PROMPT + "\n\n{format_instructions}"),
            (
                "human",
                "<customer_summary>\n{query}\n</customer_summary>\n\n"
                "<candidate_passages>\n{passages}\n</candidate_passages>\n\n"
                "Return the top {top_k} indices, most relevant first.",
            ),
        ]
    )

    llm = ChatOpenAI(
        model=settings.openai_mini_model,
        temperature=0.0,
        openai_api_key=settings.openai_api_key,
    )

    result: RerankRanking = (prompt | llm | parser).invoke(
        {
            "query": query,
            "passages": passages,
            "top_k": top_k,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    order = _repair_ranking(result.ranking, len(candidates), top_k)
    return [candidates[index] for index in order]


# ── Backend: local cross-encoder ──────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_cross_encoder():
    """Load and cache the cross-encoder. Import is local so torch stays optional."""
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder model '%s'.", settings.cross_encoder_model)
    return CrossEncoder(settings.cross_encoder_model)


def _rerank_with_cross_encoder(
    query: str, candidates: list[dict], top_k: int
) -> list[dict]:
    """Score every (query, passage) pair jointly and keep the best."""
    model = _load_cross_encoder()
    pairs = [
        (query, _truncate(chunk["content"], settings.rerank_passage_chars))
        for chunk in candidates
    ]
    scores = model.predict(pairs)

    ordered = sorted(
        zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True
    )
    return [chunk for chunk, _score in ordered[:top_k]]


# ── Public entry point ────────────────────────────────────────────────────────


def rerank(query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
    """
    Reorder candidate chunks by relevance to the query and keep the best top_k.

    Args:
        query: The customer summary used as the retrieval query.
        candidates: Chunks as {"content": str, "source": str}, in fusion order.
        top_k: How many to keep. Defaults to settings.retrieval_k.

    Returns:
        At most top_k chunks. On any failure — backend disabled, missing
        dependency, missing API key, model error — returns the first top_k
        candidates in their original fusion order.
    """
    limit = top_k if top_k is not None else settings.retrieval_k
    fallback = candidates[:limit]

    if not candidates:
        return []

    backend = (settings.rerank_backend or "none").strip().lower()

    if backend == "none":
        return fallback

    if len(candidates) <= 1:
        return fallback

    try:
        if backend == "llm":
            if not settings.openai_api_key:
                logger.info("No API key configured; skipping LLM rerank.")
                return fallback
            reranked = _rerank_with_llm(query, candidates, limit)
        elif backend == "cross_encoder":
            reranked = _rerank_with_cross_encoder(query, candidates, limit)
        else:
            logger.warning("Unknown rerank_backend '%s'; using fusion order.", backend)
            return fallback
    except ImportError as exc:
        logger.warning(
            "Rerank backend '%s' unavailable (%s). Install sentence-transformers "
            "or set RERANK_BACKEND=llm. Using fusion order.",
            backend,
            exc,
        )
        return fallback
    except Exception as exc:  # noqa: BLE001 - never fail retrieval on rerank
        logger.warning("Reranking failed (%s). Falling back to fusion order.", exc)
        return fallback

    moved = sum(
        1
        for position, chunk in enumerate(reranked)
        if position < len(fallback) and chunk is not fallback[position]
    )
    logger.info(
        "Reranked %d candidates to %d via '%s' backend (%d position changes).",
        len(candidates),
        len(reranked),
        backend,
        moved,
    )
    return reranked
