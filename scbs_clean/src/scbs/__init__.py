"""
SCBS — Sparse Compositional Basis Search

A retrieval system using sparse overcomplete dictionary learning with
late interaction. Matches BM25 NDCG@10 and beats it on P@1, with no
learned embeddings, no GPU, and no external services.
"""
from scbs.retriever import Retriever, SearchResult
from scbs.bm25 import BM25, BM25Result
from scbs import metrics

__version__ = "1.0.0"

__all__ = [
    "Retriever",
    "SearchResult",
    "BM25",
    "BM25Result",
    "metrics",
]
