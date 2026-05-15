"""
Cranfield Benchmark — Approach 12 (Sparse Directed Semantic Flow)
==================================================================

Testing whether sparse NMF basis + directed flow + genericity penalty
fixes the cluster-collapse problem and beats BM25.

Tests multiple n_basis values to find the sweet spot.

Comparison:
- BM25:        NDCG@10 = 0.3509
- Approach 10: NDCG@10 = 0.0739 (energy-based, broken clustering)
- Approach 11: NDCG@10 = 0.0739 (directed flow on broken clustering)
- Approach 12: ?                 (sparse basis + directed flow)
"""
import sys, os, time

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
from sdsf_store import SDSFStore

# Reuse parsers/metrics from Cranfield test
sys.path.insert(0, os.path.join(_HERE, "..", "real_benchmark_cranfield"))
from test_cranfield import (
    parse_cranfield_docs, parse_cranfield_queries, parse_qrels,
    ndcg_at_k, average_precision, recall_at_k, reciprocal_rank, precision_at_k,
    BM25, evaluate_system
)


def main():
    print("\n" + "="*72)
    print("  APPROACH 12 — Sparse Directed Semantic Flow on Cranfield")
    print("="*72)
    
    cran_dir = os.path.join(_HERE, "..", "real_benchmark_cranfield")
    
    print("\nLoading Cranfield...")
    docs = parse_cranfield_docs(os.path.join(cran_dir, "cran.all.1400.xml"))
    queries = parse_cranfield_queries(os.path.join(cran_dir, "cran.qry.xml"))
    qrels = parse_qrels(os.path.join(cran_dir, "cranqrel.trec.txt"))
    print(f"  {len(docs)} docs, {len(queries)} queries, qrels for {len(qrels)} queries")
    
    # ═══ BM25 BASELINE ═══
    print("\n[BM25 baseline]")
    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(docs)
    bm25_results = evaluate_system(
        "BM25",
        lambda q, k: bm25.search(q, k),
        queries, qrels
    )
    
    # ═══ SDSF at multiple n_basis values ═══
    doc_ids = list(docs.keys())
    corpus_texts = [docs[d] for d in doc_ids]
    
    sdsf_results_all = {}
    
    for n_basis in [64, 128, 256]:
        print(f"\n[SDSF n_basis={n_basis}]")
        store = SDSFStore(
            n_basis=n_basis,
            top_k_basis_per_doc=10,
            diffusion_iterations=2,
            damping=0.6,
        )
        
        t0 = time.perf_counter()
        store.learn_and_build(corpus_texts)
        store.add_doc_lookup(doc_ids)
        print(f"  Build time: {time.perf_counter() - t0:.1f}s")
        
        def make_search(s):
            def search_fn(query, top_k):
                results = s.search(query, top_k=top_k)
                return [r['id'] for r in results]
            return search_fn
        
        results = evaluate_system(
            f"SDSF-{n_basis}",
            make_search(store),
            queries, qrels
        )
        sdsf_results_all[n_basis] = results
    
    # ═══ COMPARISON TABLE ═══
    print("\n" + "="*72)
    print("  RESULTS — Cranfield Real Benchmark")
    print("="*72)
    
    print(f"\n  Queries evaluated: {bm25_results['evaluated']}")
    
    # Reference numbers from earlier runs
    a10_ndcg, a11_ndcg = 0.0739, 0.0739
    
    print(f"\n  {'Metric':<14} {'BM25':>9} {'A10':>9} {'A11':>9}", end="")
    for n in sdsf_results_all:
        print(f" {f'SDSF-{n}':>9}", end="")
    print()
    print(f"  {'─'*14} {'─'*9} {'─'*9} {'─'*9}", end="")
    for _ in sdsf_results_all:
        print(f" {'─'*9}", end="")
    print()
    
    a10_full = {'ndcg@10': 0.0739, 'map': 0.0511, 'mrr': 0.1437,
                 'p@1': 0.0622, 'p@10': 0.0471, 'recall@100': 0.2675}
    
    for m in ['ndcg@10', 'map', 'mrr', 'p@1', 'p@10', 'recall@100']:
        print(f"  {m:<14} {bm25_results[m]:>9.4f} {a10_full[m]:>9.4f} {a10_full[m]:>9.4f}", end="")
        for n, r in sdsf_results_all.items():
            print(f" {r[m]:>9.4f}", end="")
        print()
    
    print(f"\n  {'Speed':<14} {'BM25':>9} {'A10':>9} {'A11':>9}", end="")
    for n in sdsf_results_all:
        print(f" {f'SDSF-{n}':>9}", end="")
    print()
    print(f"  {'p50 (ms)':<14} {bm25_results['p50_ms']:>9.2f} {'4.08':>9} {'2.43':>9}", end="")
    for n, r in sdsf_results_all.items():
        print(f" {r['p50_ms']:>9.2f}", end="")
    print()
    
    # ═══ HONEST ASSESSMENT ═══
    print(f"\n{'='*72}")
    print(f"  HONEST ASSESSMENT")
    print(f"{'='*72}")
    
    best_n = max(sdsf_results_all.keys(), key=lambda n: sdsf_results_all[n]['ndcg@10'])
    best_result = sdsf_results_all[best_n]
    best_ndcg = best_result['ndcg@10']
    
    print(f"\n  Best SDSF config: n_basis={best_n}")
    print(f"    NDCG@10: {best_ndcg:.4f}")
    print(f"    vs BM25 ({bm25_results['ndcg@10']:.4f}): {(best_ndcg/bm25_results['ndcg@10']*100):.0f}%")
    print(f"    vs Approach 10 ({a10_ndcg:.4f}): {(best_ndcg/a10_ndcg*100):.0f}%")
    
    if best_ndcg >= bm25_results['ndcg@10']:
        print(f"\n  ✓✓ SDSF MATCHES or BEATS BM25 — architectural pivot worked.")
    elif best_ndcg >= bm25_results['ndcg@10'] * 0.85:
        print(f"\n  ✓ SDSF reaches near-BM25 territory — meaningful progress.")
    elif best_ndcg >= a10_ndcg * 3:
        print(f"\n  ↑ Substantial improvement over A10 (>3×) but still below BM25.")
    elif best_ndcg >= a10_ndcg * 1.5:
        print(f"\n  ↑ Meaningful improvement over A10 (>1.5×).")
    elif best_ndcg > a10_ndcg:
        print(f"\n  ↗ Modest improvement over A10.")
    else:
        print(f"\n  ✗ No improvement over A10. The pivot did not help.")


if __name__ == "__main__":
    main()
