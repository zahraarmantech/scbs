"""50K test with efficient relevance computation."""
import sys, os, time, random
from collections import defaultdict

_HERE = os.path.dirname(__file__)
_SRC  = os.path.join(_HERE, "..", "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from scbs.blueprint import BlueprintEncoder
from scbs.matrix_index import make_row

sys.path.insert(0, _HERE)
from approach_10_optimized import EnergyStoreOptimized

sys.path.insert(0, os.path.join(_HERE, "..", "standard_benchmark"))
from test_5k_scale import generate_realistic_corpus, generate_queries, compute_ndcg_at_k

random.seed(42)

print("\n" + "="*70)
print("  APPROACH 10 OPTIMIZED @ 50K")
print("="*70)

print("\nGenerating 50K corpus...")
corpus = generate_realistic_corpus(50000)
print(f"  {len(corpus):,} docs")

print("\nPre-computing doc word sets (one-time cost)...")
t0 = time.perf_counter()
doc_word_sets = {d['id']: set(d['text'].lower().split()) for d in corpus}
print(f"  Done in {time.perf_counter()-t0:.1f}s")

print("\nBuilding optimized index...")
encoder = BlueprintEncoder("a10_opt_50k.json")
store = EnergyStoreOptimized(n_clusters=14, diffusion_iterations=3, damping=0.7)
corpus_texts = [d['text'] for d in corpus]

t0 = time.perf_counter()
store.learn(corpus_texts)
print(f"  Learning: {time.perf_counter()-t0:.1f}s")

t0 = time.perf_counter()
for i, doc in enumerate(corpus):
    _, bp, _ = encoder.encode(doc['text'])
    store.add(make_row(bp), doc['text'])
    if i > 0 and i % 10000 == 0:
        elapsed = time.perf_counter() - t0
        eta = (len(corpus) - i) * (elapsed / i)
        print(f"  {i:,}/{len(corpus):,} ({i/elapsed:.0f} doc/sec, ETA: {eta:.0f}s)")
store.build()
build_time = time.perf_counter() - t0
print(f"  Indexing: {build_time:.1f}s ({len(corpus)/build_time:.0f} doc/sec)")

queries = generate_queries()
doc_lookup = {d['text']: d['id'] for d in corpus}

print("\nRunning evaluation (efficient relevance check)...")
ndcg_scores, recall_scores, mrr_scores = [], [], []
p1_scores, p5_scores, p10_scores = [], [], []
latencies = []
thresh_stats = []

eval_start = time.perf_counter()
for q_idx, (query_text, keywords) in enumerate(queries):
    # Efficient relevance: iterate through pre-computed word sets
    relevant_ids = set()
    for doc_id, word_set in doc_word_sets.items():
        if word_set & keywords:
            relevant_ids.add(doc_id)
    
    if not relevant_ids:
        continue
    
    _, qbp, _ = encoder.encode(query_text)
    qr = make_row(qbp)
    t0 = time.perf_counter()
    results, stats = store.search(qr, query_text, top_k=100)
    latencies.append((time.perf_counter()-t0)*1000)
    thresh_stats.append(stats.get('above_threshold_concepts', 0))
    
    result_ids = [doc_lookup[r['text']] for r in results if r['text'] in doc_lookup]
    
    ndcg_scores.append(compute_ndcg_at_k(result_ids, relevant_ids, k=10))
    recall_scores.append(len(set(result_ids[:100]) & relevant_ids) / len(relevant_ids))
    mrr = 0
    for i, did in enumerate(result_ids):
        if did in relevant_ids:
            mrr = 1.0/(i+1)
            break
    mrr_scores.append(mrr)
    p1_scores.append(1.0 if result_ids and result_ids[0] in relevant_ids else 0.0)
    t5 = result_ids[:5]; t10 = result_ids[:10]
    p5_scores.append(sum(1 for d in t5 if d in relevant_ids)/len(t5) if t5 else 0)
    p10_scores.append(sum(1 for d in t10 if d in relevant_ids)/len(t10) if t10 else 0)
    
    if (q_idx+1) % 25 == 0:
        print(f"  Evaluated {q_idx+1}/{len(queries)}")

eval_time = time.perf_counter() - eval_start
print(f"  Evaluation: {eval_time:.1f}s")

latencies.sort()
print(f"\n{'='*70}")
print(f"  RESULTS @ 50K")
print(f"{'='*70}")
print(f"\n  Quality:")
print(f"    NDCG@10:  {sum(ndcg_scores)/len(ndcg_scores):.3f}")
print(f"    MRR:      {sum(mrr_scores)/len(mrr_scores):.3f}")
print(f"    Recall:   {sum(recall_scores)/len(recall_scores)*100:.2f}%")
print(f"    P@1:      {sum(p1_scores)/len(p1_scores)*100:.0f}%")
print(f"    P@10:     {sum(p10_scores)/len(p10_scores)*100:.0f}%")
print(f"\n  Speed:")
print(f"    p50:      {latencies[len(latencies)//2]:.2f}ms")
print(f"    p95:      {latencies[int(len(latencies)*0.95)]:.2f}ms")
print(f"    p99:      {latencies[int(len(latencies)*0.99)]:.2f}ms")
print(f"\n  Threshold stats: avg {sum(thresh_stats)/len(thresh_stats):.1f}/14 concepts kept")
print(f"\n  COMPARISON @ 50K:")
print(f"    Original:  P@10=97%, NDCG=0.969, p50=5.60ms, p95=22.60ms")
opt_p10 = sum(p10_scores)/len(p10_scores)*100
opt_ndcg = sum(ndcg_scores)/len(ndcg_scores)
opt_p50 = latencies[len(latencies)//2]
opt_p95 = latencies[int(len(latencies)*0.95)]
print(f"    Optimized: P@10={opt_p10:.0f}%, NDCG={opt_ndcg:.3f}, p50={opt_p50:.2f}ms, p95={opt_p95:.2f}ms")
speedup_p50 = 5.60 / opt_p50
speedup_p95 = 22.60 / opt_p95
print(f"    Speedup:   p50 {speedup_p50:.1f}×, p95 {speedup_p95:.1f}×")

if os.path.exists("a10_opt_50k.json"):
    os.remove("a10_opt_50k.json")
