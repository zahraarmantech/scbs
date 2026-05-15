"""
Standard IR evaluation metrics.

Implementations of NDCG@k, MAP, MRR, P@k, Recall@k for benchmarking.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence


def ndcg_at_k(
    results: Sequence[str],
    qrels: Mapping[str, int],
    k: int = 10,
) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    Parameters
    ----------
    results : sequence of str
        Ranked list of document IDs.
    qrels : mapping from doc_id to relevance
        Ground truth relevance judgments.
    k : int, default=10
        Cutoff position.

    Returns
    -------
    float in [0, 1]
    """
    dcg = 0.0
    for i, doc_id in enumerate(results[:k]):
        rel = qrels.get(doc_id, 0)
        dcg += rel / math.log2(i + 2)
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(
    results: Sequence[str],
    qrels: Mapping[str, int],
) -> float:
    """Average precision over a single query."""
    rel_docs = {d for d, r in qrels.items() if r > 0}
    if not rel_docs:
        return 0.0
    score = 0.0
    num_rel = 0
    for i, doc_id in enumerate(results):
        if doc_id in rel_docs:
            num_rel += 1
            score += num_rel / (i + 1)
    return score / len(rel_docs)


def reciprocal_rank(
    results: Sequence[str],
    qrels: Mapping[str, int],
) -> float:
    """Reciprocal rank of the first relevant document."""
    rel_docs = {d for d, r in qrels.items() if r > 0}
    for i, doc_id in enumerate(results):
        if doc_id in rel_docs:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(
    results: Sequence[str],
    qrels: Mapping[str, int],
    k: int,
) -> float:
    """Precision at cutoff k."""
    rel_docs = {d for d, r in qrels.items() if r > 0}
    top = results[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in rel_docs) / len(top)


def recall_at_k(
    results: Sequence[str],
    qrels: Mapping[str, int],
    k: int,
) -> float:
    """Recall at cutoff k."""
    rel_docs = {d for d, r in qrels.items() if r > 0}
    if not rel_docs:
        return 0.0
    return sum(1 for d in results[:k] if d in rel_docs) / len(rel_docs)
