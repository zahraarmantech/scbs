"""
Cranfield Benchmark — Approach 11 (Directed Flow with Genericity Penalty)
==========================================================================

Tests whether adding IDF-weighted scoring + genericity penalty fixes
the SCBS underperformance against BM25 on real benchmark data.

Comparison:
- BM25:        NDCG@10 = 0.3509 (baseline)
- Approach 10: NDCG@10 = 0.0739 (energy-based, no IDF)
- Approach 11: ?                 (directed flow + IDF + genericity)
"""
import sys, os, re, time, math
from collections import defaultdict

_HERE = os.path.dirname(__file__)
_SRC  = os.path.join(_HERE, "..", "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from scbs.blueprint import BlueprintEncoder
from scbs.matrix_index import make_row

sys.path.insert(0, _HERE)
from directed_flow_store import DirectedFlowStore

# Reuse parsers and metrics from Approach 10's Cranfield test
sys.path.insert(0, os.path.join(_HERE, "..", "real_benchmark_cranfield"))
from test_cranfield import (
    parse_cranfield_docs, parse_cranfield_queries, parse_qrels,
    ndcg_at_k, average_precision, recall_at_k, reciprocal_rank, precision_at_k,
    BM25, evaluate_system
)


def main():
    print("\n" + "="*72)
    print("  APPROACH 11 — Directed Flow on REAL Cranfield Benchmark")
    print("="*72)
    
    cran_dir = os.path.join(_HERE, "..", "real_benchmark_cranfield")
    
    # Load data
    print("\nLoading Cranfield...")
    docs = parse_cranfield_docs(os.path.join(cran_dir, "cran.all.1400.xml"))
    queries = parse_cranfield_queries(os.path.join(cran_dir, "cran.qry.xml"))
    qrels = parse_qrels(os.path.join(cran_dir, "cranqrel.trec.txt"))
    print(f"  Docs: {len(docs)}, Queries: {len(queries)}, Qrels for {len(qrels)} queries")
    
    # ═══ BM25 BASELINE ═══
    print("\n[BM25 baseline]")
    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(docs)
    bm25_results = evaluate_system(
        "BM25",
        lambda q, k: bm25.search(q, k),
        queries, qrels
    )
    
    # ═══ APPROACH 11 ═══
    print("\n[Approach 11 — Directed Flow]")
    encoder = BlueprintEncoder("cran11_unknown.json")
    store = DirectedFlowStore(n_clusters=14, diffusion_iterations=3, damping=0.7)
    
    corpus_texts = list(docs.values())
    text_to_id = {}
    
    print("  Learning (clustering + graph + IDF)...")
    store.learn(corpus_texts)
    
    print(f"  Cluster sizes (docs per cluster):")
    for c in sorted(store._cluster_doc_count.keys()):
        df = store._cluster_doc_count[c]
        idf = store._cluster_idf.get(c, 0)
        penalty = store._genericity_penalty.get(c, 1.0)
        print(f"    Cluster {c}: {df} docs (df/N={df/len(docs):.2%}, idf={idf:.2f}, penalty={penalty:.2f})")
    
    print("  Indexing documents...")
    for doc_id, text in docs.items():
        _, bp, _ = encoder.encode(text)
        store.add(make_row(bp), text)
        text_to_id[text] = doc_id
    store.build()
    
    def directed_search(query, top_k):
        _, qbp, _ = encoder.encode(query)
        qr = make_row(qbp)
        results, _ = store.search(qr, query, top_k=top_k)
        return [text_to_id.get(r['text'], '') for r in results]
    
    a11_results = evaluate_system(
        "Approach 11",
        directed_search,
        queries, qrels
    )
    
    # ═══ COMPARISON ═══
    print("\n" + "="*72)
    print("  RESULTS — Cranfield Real Benchmark")
    print("="*72)
    
    # Pull Approach 10 results from prior measurement
    a10_results = {
        'ndcg@10': 0.0739, 'map': 0.0511, 'mrr': 0.1437,
        'p@1': 0.0622, 'p@10': 0.0471, 'recall@100': 0.2675,
        'p50_ms': 4.08, 'p95_ms': 40.61
    }
    
    print(f"\n  Queries evaluated: {bm25_results['evaluated']}")
    print(f"\n  {'Metric':<14} {'BM25':>9} {'A10':>9} {'A11':>9} {'A11-A10':>10} {'A11-BM25':>10}")
    print(f"  {'─'*14} {'─'*9} {'─'*9} {'─'*9} {'─'*10} {'─'*10}")
    
    for m in ['ndcg@10', 'map', 'mrr', 'p@1', 'p@10', 'recall@100']:
        b = bm25_results[m]
        a10 = a10_results[m]
        a11 = a11_results[m]
        delta_a10 = a11 - a10
        delta_bm25 = a11 - b
        a10_mark = "↑" if delta_a10 > 0.01 else ("↓" if delta_a10 < -0.01 else "=")
        bm25_mark = "↑" if delta_bm25 > 0 else "↓"
        print(f"  {m:<14} {b:>9.4f} {a10:>9.4f} {a11:>9.4f} {delta_a10:>+10.4f}{a10_mark} {delta_bm25:>+10.4f}{bm25_mark}")
    
    print(f"\n  {'Speed':<14} {'BM25':>9} {'A10':>9} {'A11':>9}")
    print(f"  {'─'*14} {'─'*9} {'─'*9} {'─'*9}")
    print(f"  {'p50 (ms)':<14} {bm25_results['p50_ms']:>9.2f} {a10_results['p50_ms']:>9.2f} {a11_results['p50_ms']:>9.2f}")
    print(f"  {'p95 (ms)':<14} {bm25_results['p95_ms']:>9.2f} {a10_results['p95_ms']:>9.2f} {a11_results['p95_ms']:>9.2f}")
    
    # Honest assessment
    print(f"\n{'='*72}")
    print(f"  HONEST ASSESSMENT")
    print(f"{'='*72}")
    
    a11_ndcg = a11_results['ndcg@10']
    a10_ndcg = a10_results['ndcg@10']
    bm25_ndcg = bm25_results['ndcg@10']
    
    improvement_vs_a10 = (a11_ndcg - a10_ndcg) / a10_ndcg * 100 if a10_ndcg > 0 else 0
    gap_to_bm25 = (a11_ndcg / bm25_ndcg * 100) if bm25_ndcg > 0 else 0
    
    print(f"\n  Improvement A11 vs A10: {improvement_vs_a10:+.0f}% on NDCG@10")
    print(f"  A11 reaches {gap_to_bm25:.0f}% of BM25's NDCG@10")
    
    if a11_ndcg >= bm25_ndcg:
        print(f"\n  ✓ Approach 11 MATCHES or BEATS BM25 — directed flow + genericity worked")
    elif a11_ndcg >= bm25_ndcg * 0.85:
        print(f"\n  ~ Approach 11 reaches near-BM25 territory — meaningful progress")
    elif a11_ndcg >= a10_ndcg * 1.5:
        print(f"\n  ↑ Significant improvement over A10 but still below BM25")
    elif a11_ndcg > a10_ndcg:
        print(f"\n  ↑ Modest improvement over A10")
    else:
        print(f"\n  ✗ No improvement — the proposed fix didn't move the needle")
    
    # Cleanup
    if os.path.exists("cran11_unknown.json"):
        os.remove("cran11_unknown.json")


if __name__ == "__main__":
    main()
