"""Smoke tests — verify the library imports and basic operations work."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_imports():
    from scbs import Retriever, BM25, SearchResult, BM25Result, metrics
    from scbs.retriever import Retriever as R2
    from scbs.bm25 import BM25 as B2
    assert R2 is Retriever
    assert B2 is BM25


def test_retriever_fit_and_search():
    from scbs import Retriever
    corpus = [
        "Kafka consumer lag in production",
        "Database migration completed successfully",
        "Authentication service errors elevated",
        "Customer reported duplicate charge on statement",
        "Fraud detection flagged wire transfer",
        "Performance review scheduled for next month",
    ] * 5  # 30 docs - enough for the system to function
    r = Retriever(n_atoms=16, top_k_query=3, top_m_doc=4)
    r.fit(corpus)
    results = r.search("authentication problem", top_k=3)
    assert isinstance(results, list)
    # Should return at least one result
    assert len(results) >= 1


def test_bm25_fit_and_search():
    from scbs import BM25
    corpus = [
        "Kafka consumer lag in production",
        "Database migration completed",
        "Authentication service errors",
    ]
    b = BM25()
    b.fit(corpus, doc_ids=["d1", "d2", "d3"])
    results = b.search("kafka lag", top_k=3)
    assert len(results) > 0
    assert results[0].doc_id == "d1"


def test_metrics():
    from scbs import metrics
    qrels = {"d1": 1, "d2": 0, "d3": 1}
    results = ["d1", "d3", "d2"]
    ndcg = metrics.ndcg_at_k(results, qrels, k=3)
    assert ndcg > 0.9  # near-perfect ranking
    ap = metrics.average_precision(results, qrels)
    assert ap > 0.9
    mrr = metrics.reciprocal_rank(results, qrels)
    assert mrr == 1.0


if __name__ == "__main__":
    test_imports()
    print("✓ imports")
    test_retriever_fit_and_search()
    print("✓ retriever")
    test_bm25_fit_and_search()
    print("✓ bm25")
    test_metrics()
    print("✓ metrics")
    print("\nAll smoke tests passed.")
