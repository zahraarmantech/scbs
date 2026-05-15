"""Approach 14 — single config test."""
import sys, os, time
import warnings
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
from late_interaction_store import LateInteractionStore

sys.path.insert(0, os.path.join(_HERE, "..", "real_benchmark_cranfield"))
from test_cranfield import (
    parse_cranfield_docs, parse_cranfield_queries, parse_qrels,
    BM25, evaluate_system
)

print("\n" + "="*72)
print("  APPROACH 14 — Late Interaction Cranfield Test")
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

print("\n[A14 n=256 topK=10 topM=15]")
store = LateInteractionStore(n_basis=256, top_k_query_basis=10, top_m_doc_basis=15)
t0 = time.perf_counter()
store.learn_and_build(corpus_texts)
store.add_doc_lookup(doc_ids)
print(f"  Build: {time.perf_counter()-t0:.1f}s")

r256 = evaluate_system("A14-256", lambda q, k, s=store: [x['id'] for x in s.search(q, k)], queries, qrels)

print(f"\n{'='*72}")
print(f"  RESULTS")
print(f"{'='*72}")
print(f"\n  {'Approach':<25} {'NDCG':>8} {'MAP':>8} {'MRR':>8} {'P@1':>8} {'R@100':>8} {'p50ms':>8}")
print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
print(f"  {'BM25 baseline':<25} {bm25_r['ndcg@10']:>8.4f} {bm25_r['map']:>8.4f} {bm25_r['mrr']:>8.4f} {bm25_r['p@1']:>8.4f} {bm25_r['recall@100']:>8.4f} {bm25_r['p50_ms']:>8.2f}")
print(f"  {'A10 (clusters)':<25} {0.0739:>8.4f} {0.0511:>8.4f} {0.1437:>8.4f} {0.0622:>8.4f} {0.2675:>8.4f} {4.08:>8.2f}")
print(f"  {'A12 SDSF-256':<25} {0.1753:>8.4f} {0.1327:>8.4f} {0.2800:>8.4f} {0.1956:>8.4f} {0.4726:>8.4f} {60.52:>8.2f}")
print(f"  {'A13 hard-gated':<25} {0.2360:>8.4f} {0.1862:>8.4f} {0.3359:>8.4f} {0.2000:>8.4f} {0.6227:>8.4f} {50.08:>8.2f}")
print(f"  {'A14 late interaction':<25} {r256['ndcg@10']:>8.4f} {r256['map']:>8.4f} {r256['mrr']:>8.4f} {r256['p@1']:>8.4f} {r256['recall@100']:>8.4f} {r256['p50_ms']:>8.2f}")

print(f"\n  A14 vs BM25: {r256['ndcg@10']/bm25_r['ndcg@10']*100:.0f}%")
print(f"  A14 vs A13:  {r256['ndcg@10']/0.2360*100:.0f}%")

if r256['ndcg@10'] >= bm25_r['ndcg@10']:
    print(f"\n  ✓✓ A14 MATCHES or BEATS BM25.")
elif r256['ndcg@10'] >= bm25_r['ndcg@10'] * 0.95:
    print(f"\n  ✓ Within 5% of BM25 — essentially competitive.")
elif r256['ndcg@10'] >= bm25_r['ndcg@10'] * 0.85:
    print(f"\n  ✓ Within 15% of BM25.")
elif r256['ndcg@10'] > 0.2360 * 1.1:
    print(f"\n  ↑ >10% improvement over A13.")
elif r256['ndcg@10'] > 0.2360:
    print(f"\n  ↗ Modest improvement over A13.")
else:
    print(f"\n  → No improvement over A13.")
