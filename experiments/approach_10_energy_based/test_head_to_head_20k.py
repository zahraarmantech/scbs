"""20K test - compares optimized vs what original A10 would do."""
import sys, os, time, random

_HERE = os.path.dirname(__file__)
_SRC  = os.path.join(_HERE, "..", "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from scbs.blueprint import BlueprintEncoder
from scbs.matrix_index import make_row

sys.path.insert(0, _HERE)
from approach_10_optimized import EnergyStoreOptimized
from test_approach_10 import EnergyStore  # original

sys.path.insert(0, os.path.join(_HERE, "..", "standard_benchmark"))
from test_5k_scale import generate_realistic_corpus, generate_queries, compute_ndcg_at_k

random.seed(42)

print("\n" + "="*70)
print("  HEAD-TO-HEAD @ 20K: Original vs Optimized")
print("="*70)

print("\nGenerating 20K corpus...")
corpus = generate_realistic_corpus(20000)
corpus_texts = [d['text'] for d in corpus]
print(f"  {len(corpus):,} docs")

print("\nPre-computing doc word sets...")
doc_word_sets = {d['id']: set(d['text'].lower().split()) for d in corpus}

def run_test(StoreClass, label):
    print(f"\nBuilding {label} index...")
    encoder = BlueprintEncoder(f"a20k_{label}.json")
    store = StoreClass(n_clusters=14, diffusion_iterations=3, damping=0.7)
    
    t0 = time.perf_counter()
    store.learn(corpus_texts)
    for doc in corpus:
        _, bp, _ = encoder.encode(doc['text'])
        store.add(make_row(bp), doc['text'])
    store.build()
    print(f"  Built in {time.perf_counter()-t0:.1f}s")
    
    queries = generate_queries()
    doc_lookup = {d['text']: d['id'] for d in corpus}
    
    ndcg_scores, recall_scores = [], []
    p10_scores, latencies = [], []
    
    for q, kw in queries:
        rel = {did for did, ws in doc_word_sets.items() if ws & kw}
        if not rel: continue
        _, qbp, _ = encoder.encode(q)
        qr = make_row(qbp)
        t0 = time.perf_counter()
        res, _ = store.search(qr, q, top_k=100)
        latencies.append((time.perf_counter()-t0)*1000)
        ids = [doc_lookup[r['text']] for r in res if r['text'] in doc_lookup]
        ndcg_scores.append(compute_ndcg_at_k(ids, rel, k=10))
        recall_scores.append(len(set(ids[:100]) & rel) / len(rel))
        t10 = ids[:10]
        p10_scores.append(sum(1 for d in t10 if d in rel)/len(t10) if t10 else 0)
    
    latencies.sort()
    fpath = f"a20k_{label}.json"
    if os.path.exists(fpath):
        os.remove(fpath)
    
    return {
        'p10': sum(p10_scores)/len(p10_scores)*100,
        'ndcg': sum(ndcg_scores)/len(ndcg_scores),
        'recall': sum(recall_scores)/len(recall_scores)*100,
        'p50': latencies[len(latencies)//2],
        'p95': latencies[int(len(latencies)*0.95)],
        'p99': latencies[int(len(latencies)*0.99)],
    }

orig = run_test(EnergyStore, "orig")
print(f"\n  Original @ 20K:")
print(f"    P@10={orig['p10']:.0f}%, NDCG={orig['ndcg']:.3f}, Recall={orig['recall']:.1f}%")
print(f"    p50={orig['p50']:.2f}ms, p95={orig['p95']:.2f}ms, p99={orig['p99']:.2f}ms")

opt = run_test(EnergyStoreOptimized, "opt")
print(f"\n  Optimized @ 20K:")
print(f"    P@10={opt['p10']:.0f}%, NDCG={opt['ndcg']:.3f}, Recall={opt['recall']:.1f}%")
print(f"    p50={opt['p50']:.2f}ms, p95={opt['p95']:.2f}ms, p99={opt['p99']:.2f}ms")

print(f"\n  Speedup: p50 {orig['p50']/opt['p50']:.1f}×, p95 {orig['p95']/opt['p95']:.1f}×, p99 {orig['p99']/opt['p99']:.1f}×")
print(f"  Quality delta: P@10 {opt['p10']-orig['p10']:+.0f}pp, NDCG {opt['ndcg']-orig['ndcg']:+.4f}")
