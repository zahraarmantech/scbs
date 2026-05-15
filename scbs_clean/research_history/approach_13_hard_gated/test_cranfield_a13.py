"""
Cranfield Benchmark — Approach 13 (Hard-Gated Sparse Scoring)
==============================================================

Tests whether top-k query truncation + doc max-gating + IDF + no-diffusion
closes the gap to BM25.

Also runs ABLATION to see which change matters most.
"""
import sys, os, time

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
from hard_gated_store import HardGatedStore

sys.path.insert(0, os.path.join(_HERE, "..", "real_benchmark_cranfield"))
from test_cranfield import (
    parse_cranfield_docs, parse_cranfield_queries, parse_qrels,
    BM25, evaluate_system
)


def main():
    print("\n" + "="*72)
    print("  APPROACH 13 — Hard-Gated Sparse Scoring on Cranfield")
    print("="*72)
    
    cran_dir = os.path.join(_HERE, "..", "real_benchmark_cranfield")
    docs = parse_cranfield_docs(os.path.join(cran_dir, "cran.all.1400.xml"))
    queries = parse_cranfield_queries(os.path.join(cran_dir, "cran.qry.xml"))
    qrels = parse_qrels(os.path.join(cran_dir, "cranqrel.trec.txt"))
    print(f"\n  {len(docs)} docs, {len(queries)} queries")
    
    # BM25 baseline
    print("\n[BM25 baseline]")
    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(docs)
    bm25_results = evaluate_system("BM25", lambda q, k: bm25.search(q, k), queries, qrels)
    
    doc_ids = list(docs.keys())
    corpus_texts = [docs[d] for d in doc_ids]
    
    # Test configurations to find sweet spot + ablation
    configs = [
        # Full system at different n_basis
        {'label': 'A13 full n=128 topQ=5', 'n_basis': 128, 'top_k_query_basis': 5, 'use_doc_gating': True},
        {'label': 'A13 full n=256 topQ=5', 'n_basis': 256, 'top_k_query_basis': 5, 'use_doc_gating': True},
        {'label': 'A13 full n=256 topQ=10', 'n_basis': 256, 'top_k_query_basis': 10, 'use_doc_gating': True},
        {'label': 'A13 full n=256 topQ=20', 'n_basis': 256, 'top_k_query_basis': 20, 'use_doc_gating': True},
        {'label': 'A13 full n=512 topQ=10', 'n_basis': 512, 'top_k_query_basis': 10, 'use_doc_gating': True},
        # Ablations on best n_basis
        {'label': 'A13 NO gating n=256 topQ=5', 'n_basis': 256, 'top_k_query_basis': 5, 'use_doc_gating': False},
        {'label': 'A13 NO topQ n=256', 'n_basis': 256, 'top_k_query_basis': 256, 'use_doc_gating': True},
    ]
    
    results_all = {}
    for cfg in configs:
        label = cfg['label']
        print(f"\n[{label}]")
        store = HardGatedStore(
            n_basis=cfg['n_basis'],
            top_k_query_basis=cfg['top_k_query_basis'],
            top_k_basis_per_doc=10,
            use_doc_gating=cfg['use_doc_gating'],
        )
        t0 = time.perf_counter()
        store.learn_and_build(corpus_texts)
        store.add_doc_lookup(doc_ids)
        print(f"  Build time: {time.perf_counter() - t0:.1f}s")
        
        def make_search(s):
            return lambda q, k: [r['id'] for r in s.search(q, top_k=k)]
        
        results = evaluate_system(label, make_search(store), queries, qrels)
        results_all[label] = results
    
    # ═══ RESULTS TABLE ═══
    print("\n" + "="*72)
    print("  RESULTS — Cranfield Benchmark, Approach 13 (Hard-Gated)")
    print("="*72)
    
    # Prior results for comparison
    prior = {
        'BM25': bm25_results,
        'A10 (energy/clusters)': {'ndcg@10': 0.0739, 'map': 0.0511, 'mrr': 0.1437,
                                    'p@1': 0.0622, 'p@10': 0.0471, 'recall@100': 0.2675,
                                    'p50_ms': 4.08},
        'A12 (SDSF n=256)':      {'ndcg@10': 0.1753, 'map': 0.1327, 'mrr': 0.2800,
                                    'p@1': 0.1956, 'p@10': 0.1169, 'recall@100': 0.4726,
                                    'p50_ms': 60.52},
    }
    
    print(f"\n  {'Config':<30} {'NDCG@10':>9} {'MAP':>9} {'MRR':>9} {'P@1':>9} {'R@100':>9} {'p50':>9}")
    print(f"  {'─'*30} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")
    
    # Show prior baselines
    for name, r in prior.items():
        print(f"  {name:<30} {r['ndcg@10']:>9.4f} {r['map']:>9.4f} "
              f"{r['mrr']:>9.4f} {r['p@1']:>9.4f} {r['recall@100']:>9.4f} "
              f"{r.get('p50_ms', 0):>9.2f}")
    
    print(f"  {'─'*30}")
    
    # Show new approach
    for label, r in results_all.items():
        print(f"  {label:<30} {r['ndcg@10']:>9.4f} {r['map']:>9.4f} "
              f"{r['mrr']:>9.4f} {r['p@1']:>9.4f} {r['recall@100']:>9.4f} "
              f"{r['p50_ms']:>9.2f}")
    
    # Honest assessment
    print(f"\n{'='*72}")
    print(f"  HONEST ASSESSMENT")
    print(f"{'='*72}")
    
    best_label = max(results_all.keys(), key=lambda l: results_all[l]['ndcg@10'])
    best = results_all[best_label]
    
    bm25_ndcg = bm25_results['ndcg@10']
    
    print(f"\n  Best A13 config: {best_label}")
    print(f"    NDCG@10:    {best['ndcg@10']:.4f}")
    print(f"    vs BM25:    {best['ndcg@10']/bm25_ndcg*100:.0f}% of BM25's {bm25_ndcg:.4f}")
    print(f"    vs A10:     {best['ndcg@10']/0.0739*100:.0f}% (was 0.0739)")
    print(f"    vs A12:     {best['ndcg@10']/0.1753*100:.0f}% (was 0.1753)")
    
    if best['ndcg@10'] >= bm25_ndcg:
        print(f"\n  ✓✓ MATCHES or BEATS BM25 — Hard gating worked.")
    elif best['ndcg@10'] >= bm25_ndcg * 0.95:
        print(f"\n  ✓ Within 5% of BM25 — essentially competitive.")
    elif best['ndcg@10'] >= bm25_ndcg * 0.85:
        print(f"\n  ✓ Within 15% of BM25 — meaningful progress.")
    elif best['ndcg@10'] >= 0.1753 * 1.2:
        print(f"\n  ↑ >20% improvement over A12 but still below BM25.")
    else:
        print(f"\n  ↗ Modest or no improvement over A12.")


if __name__ == "__main__":
    main()
