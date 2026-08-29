"""
Retrieval quality measurement for BankLens.

Adding a reranker without measuring it is a claim, not an improvement. This
module supplies the ground truth and the metrics needed to say whether it
helped, and an A/B entry point that runs the same queries with reranking on and
off.

Relevance is judged at document level rather than chunk level. With ten product
documents and 500-character chunks, several chunks from the same document are
near-interchangeable, so "did we surface the right *product*" is the question
that matters to a Relationship Manager.
"""

import math
from dataclasses import dataclass

# Which product documents are a defensible recommendation for each risk band.
# Written from the product strategy in prompts/system_prompt.txt, by hand.
RELEVANT_BY_RISK: dict[str, set[str]] = {
    "Low": {
        "fixed_deposit.md",
        "sweep_account.md",
        "savings_account.md",
        "mutual_funds_sip.md",
    },
    "Medium": {
        "credit_card.md",
        "recurring_deposit.md",
        "savings_account.md",
    },
    "High": {
        "debt_consolidation_loan.md",
        "recurring_deposit.md",
    },
}

# Documents that must never be surfaced as a recommendation to a customer in
# deficit. Retrieval showing them is not itself a failure — the guardrail lives
# in the prompt — but a spike here is worth knowing about.
HARMFUL_FOR_DEFICIT: set[str] = {"credit_card.md", "personal_loan.md"}


@dataclass(frozen=True)
class RetrievalMetrics:
    """Ranking quality for one query."""

    hit: bool  # at least one relevant document in the results
    mrr: float  # 1 / rank of the first relevant chunk
    precision: float  # share of returned chunks from a relevant document
    ndcg: float  # rank-weighted, binary relevance
    harmful: int  # count of chunks from a product barred for this customer

    def as_row(self) -> dict:
        return {
            "hit": float(self.hit),
            "mrr": self.mrr,
            "precision": self.precision,
            "ndcg": self.ndcg,
            "harmful": float(self.harmful),
        }


def score_ranking(
    sources_in_order: list[str],
    relevant: set[str],
    harmful: set[str] | None = None,
) -> RetrievalMetrics:
    """
    Score one ranked list of chunk sources against a relevant document set.

    Args:
        sources_in_order: Source filename per returned chunk, best rank first.
        relevant: Documents that would be a sound recommendation.
        harmful: Documents that would be unsound for this customer.
    """
    harmful = harmful or set()

    if not sources_in_order:
        return RetrievalMetrics(False, 0.0, 0.0, 0.0, 0)

    flags = [source in relevant for source in sources_in_order]

    mrr = 0.0
    for position, is_relevant in enumerate(flags, start=1):
        if is_relevant:
            mrr = 1.0 / position
            break

    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, is_relevant in enumerate(flags, start=1)
        if is_relevant
    )
    ideal_hits = min(len(relevant), len(flags))
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))

    return RetrievalMetrics(
        hit=any(flags),
        mrr=mrr,
        precision=sum(flags) / len(flags),
        ndcg=(dcg / idcg) if idcg else 0.0,
        harmful=sum(1 for source in sources_in_order if source in harmful),
    )


def mean_metrics(rows: list[RetrievalMetrics]) -> dict[str, float]:
    """Average a set of per-query metrics."""
    if not rows:
        return {}
    keys = rows[0].as_row().keys()
    return {key: sum(row.as_row()[key] for row in rows) / len(rows) for key in keys}


# ── Offline: BM25 headroom ────────────────────────────────────────────────────


def bm25_sources(query: str, k: int) -> list[str]:
    """
    Rank sources using the BM25 half of retrieval alone.

    Purely lexical, so it needs no API key and no embeddings. Useful for
    establishing how much headroom a reranker has: if the relevant document is
    inside the candidate pool but outside the final cut, reranking has
    something real to fix. If it is missing from the pool entirely, no amount
    of reranking will help and the fix belongs upstream in chunking or query
    construction.
    """
    from app.pipeline.rag import get_cached_bm25_retriever

    retriever = get_cached_bm25_retriever()
    if retriever is None:
        return []

    original_k = retriever.k
    retriever.k = k
    try:
        docs = retriever.invoke(query)
    finally:
        retriever.k = original_k

    return [doc.metadata.get("source", "unknown") for doc in docs]


def measure_bm25_headroom(
    queries_by_risk: list[tuple[str, str]],
    final_k: int,
    candidate_k: int,
) -> dict[str, dict[str, float]]:
    """
    Compare BM25 at the final cut against BM25 over the wider candidate pool.

    Args:
        queries_by_risk: (query, risk_band) pairs.
        final_k: How many chunks reach the LLM.
        candidate_k: How many are pulled before reranking.

    Returns:
        {"final_k": means, "candidate_k": means} — the gap between them is the
        ceiling on what reranking alone can recover.
    """
    narrow, wide = [], []

    for query, risk in queries_by_risk:
        relevant = RELEVANT_BY_RISK[risk]
        narrow.append(score_ranking(bm25_sources(query, final_k), relevant))
        wide.append(score_ranking(bm25_sources(query, candidate_k), relevant))

    return {"final_k": mean_metrics(narrow), "candidate_k": mean_metrics(wide)}


# ── Online: reranker A/B ──────────────────────────────────────────────────────


def compare_multi_query(queries_by_risk: list[tuple[str, str]]) -> dict[str, dict]:
    """
    Run retrieval with multi-query expansion off and on over the same queries.

    Reranking is held off in both arms so this measures expansion alone.
    Requires an API key (embeddings + the variant-generation call).
    """
    from app.core.config import settings
    from app.pipeline.rag import build_vector_store, retrieve

    store = build_vector_store()
    baseline, expanded = [], []

    for query, risk in queries_by_risk:
        relevant = RELEVANT_BY_RISK[risk]
        harmful = HARMFUL_FOR_DEFICIT if risk == "High" else set()

        single = retrieve(query, store, use_reranker=False, use_multi_query=False)
        multi = retrieve(query, store, use_reranker=False, use_multi_query=True)

        baseline.append(score_ranking([c["source"] for c in single], relevant, harmful))
        expanded.append(score_ranking([c["source"] for c in multi], relevant, harmful))

    base_means = mean_metrics(baseline)
    multi_means = mean_metrics(expanded)

    return {
        "variant_count": settings.multi_query_count,
        "final_k": settings.retrieval_k,
        "baseline": base_means,
        "multi_query": multi_means,
        "delta": {key: multi_means[key] - base_means[key] for key in base_means},
    }


def compare_reranker(queries_by_risk: list[tuple[str, str]]) -> dict[str, dict]:
    """
    Run the full hybrid pipeline with reranking off and on over the same queries.

    Requires an API key: the dense half needs embeddings and the default
    reranker backend is an LLM.

    Returns:
        {"baseline": means, "reranked": means, "delta": per-metric change}
    """
    from app.core.config import settings
    from app.pipeline.rag import build_vector_store, retrieve

    store = build_vector_store()
    baseline, reranked = [], []

    for query, risk in queries_by_risk:
        relevant = RELEVANT_BY_RISK[risk]
        harmful = HARMFUL_FOR_DEFICIT if risk == "High" else set()

        without = retrieve(query, store, use_reranker=False)
        with_rerank = retrieve(query, store, use_reranker=True)

        baseline.append(
            score_ranking([c["source"] for c in without], relevant, harmful)
        )
        reranked.append(
            score_ranking([c["source"] for c in with_rerank], relevant, harmful)
        )

    base_means = mean_metrics(baseline)
    rerank_means = mean_metrics(reranked)

    return {
        "backend": settings.rerank_backend,
        "candidate_k": settings.retrieval_candidate_k,
        "final_k": settings.retrieval_k,
        "baseline": base_means,
        "reranked": rerank_means,
        "delta": {key: rerank_means[key] - base_means[key] for key in base_means},
    }
