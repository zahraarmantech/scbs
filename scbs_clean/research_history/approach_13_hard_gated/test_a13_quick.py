"""Approach 13 quick test - 3 key configurations only."""
import sys, os, time
import warnings
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
from hard_gated_store import HardGatedStore

sys.path.insert(0, os.path.join(_HERE, "..", "real_benchmark_cranfield"))
from test_cranfield import (
    parse_cranfield_docs, parse_cranfield_queries, parse_qrels,
    BM25, evaluate_system
)

print("\n" + "="*72)
print("  APPROACH 13 QUICK TEST — Cranfield")
print("="*72)

cran_dir = os.path.join(_HERE, "..", "real_benchmark_cranfield")
docs = parse_cranfield_docs(os.path.join(cran_dir, "cran.all.1400.xml"))
queries = parse_cranfield_queries(os.path.join(cran_dir, "cran.qry.xml"))
qrels = parse_qrels(os.path.join(cran_dir, "cranqrel.trec.txt"))
print(f"\n  {len(docs)} docs, {len(queries)} queries")

print("\n[BM25 baseline]")
bm25 = BM25(k1=1.5, b=0.75)
bm25.fit(docs)
bm25_r = evaluate_system("BM25", lambda q, k: bm25.search(q, k), queries, qrels)

doc_ids = list(docs.keys())
corpus_texts = [docs[d] for d in doc_ids]

# Just 3 key configs to see if approach helps
configs = [
    ('A13 n=128 topQ=5 (full)',  128, 5,  True),
    ('A13 n=256 topQ=10 (full)', 256, 10, True),
    ('A13 n=256 NO gating',      256, 10, False),
]

all_results = {}
for label, n_basis, top_q, gating in configs:
    print(f"\n[{label}]")
    store = HardGatedStore(
        n_basis=n_basis, top_k_query_basis=top_q,
        top_k_basis_per_doc=10, use_doc_gating=gating
    )
    t0 = time.perf_counter()
    store.learn_and_build(corpus_texts)
    store.add_doc_lookup(doc_ids)
    print(f"  Build: {time.perf_counter()-t0:.1f}s")
    
    r = evaluate_system(label, lambda q, k, s=store: [x['id'] for x in s.search(q, k)], queries, qrels)
    all_results[label] = r

print(f"\n{'='*72}")
print(f"  RESULTS")
print(f"{'='*72}")

prior = {
    'BM25':         bm25_r,
    'A10 (energy)': {'ndcg@10': 0.0739, 'map': 0.0511, 'mrr': 0.1437, 'p@1': 0.0622, 'p@10': 0.0471, 'recall@100': 0.2675, 'p50_ms': 4.08},
    'A12 SDSF-256': {'ndcg@10': 0.1753, 'map': 0.1327, 'mrr': 0.2800, 'p@1': 0.1956, 'p@10': 0.1169, 'recall@100': 0.4726, 'p50_ms': 60.52},
}

print(f"\n  {'Config':<30} {'NDCG':>8} {'MAP':>8} {'MRR':>8} {'P@1':>8} {'R@100':>8} {'p50ms':>8}")
print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
for name, r in prior.items():
    print(f"  {name:<30} {r['ndcg@10']:>8.4f} {r['map']:>8.4f} {r['mrr']:>8.4f} {r['p@1']:>8.4f} {r['recall@100']:>8.4f} {r.get('p50_ms',0):>8.2f}")
print(f"  {'─'*30}")
for label, r in all_results.items():
    print(f"  {label:<30} {r['ndcg@10']:>8.4f} {r['map']:>8.4f} {r['mrr']:>8.4f} {r['p@1']:>8.4f} {r['recall@100']:>8.4f} {r['p50_ms']:>8.2f}")

best = max(all_results.values(), key=lambda r: r['ndcg@10'])
print(f"\n  Best A13: NDCG@10={best['ndcg@10']:.4f}")
print(f"  vs BM25:  {best['ndcg@10']/bm25_r['ndcg@10']*100:.0f}% of BM25 ({bm25_r['ndcg@10']:.4f})")
print(f"  vs A10:   {best['ndcg@10']/0.0739*100:.0f}% (was 0.0739)")
print(f"  vs A12:   {best['ndcg@10']/0.1753*100:.0f}% (was 0.1753)")

if best['ndcg@10'] >= bm25_r['ndcg@10']:
    print(f"\n  ✓✓ MATCHES or BEATS BM25.")
elif best['ndcg@10'] >= bm25_r['ndcg@10'] * 0.85:
    print(f"\n  ✓ Within 15% of BM25.")
elif best['ndcg@10'] >= 0.1753 * 1.2:
    print(f"\n  ↑ >20% improvement over A12.")
elif best['ndcg@10'] > 0.1753:
    print(f"\n  ↗ Modest improvement over A12.")
else:
    print(f"\n  → No improvement over A12.")
