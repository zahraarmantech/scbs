"""Original Approach 10 at 100K scale."""
import sys, os, time, random

_HERE = os.path.dirname(__file__)
_SRC  = os.path.join(_HERE, "..", "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from scbs.blueprint import BlueprintEncoder
from scbs.matrix_index import make_row

sys.path.insert(0, _HERE)
from test_approach_10 import EnergyStore

sys.path.insert(0, os.path.join(_HERE, "..", "standard_benchmark"))
from test_5k_scale import generate_realistic_corpus, generate_queries, compute_ndcg_at_k

random.seed(42)

print("APPROACH 10 ORIGINAL @ 100K")
print("="*70)

print("Generating 100K corpus...")
corpus = generate_realistic_corpus(100000)
corpus_texts = [d['text'] for d in corpus]
print(f"  {len(corpus):,} docs")

print("Pre-computing word sets for eval...")
doc_word_sets = {d['id']: set(d['text'].lower().split()) for d in corpus}

print("Building index (this takes ~10 min)...")
encoder = BlueprintEncoder("a10_100k.json")
store = EnergyStore(n_clusters=14, diffusion_iterations=3, damping=0.7)

t0 = time.perf_counter()
store.learn(corpus_texts)
print(f"  Learning: {time.perf_counter()-t0:.1f}s")

t0 = time.perf_counter()
for i, doc in enumerate(corpus):
    _, bp, _ = encoder.encode(doc['text'])
    store.add(make_row(bp), doc['text'])
    if i > 0 and i % 20000 == 0:
        elapsed = time.perf_counter() - t0
        eta = (len(corpus) - i) * (elapsed / i)
        print(f"  {i:,}/{len(corpus):,} ({i/elapsed:.0f}/sec, ETA: {eta:.0f}s)")
store.build()
print(f"  Indexing: {time.perf_counter()-t0:.1f}s")

queries = generate_queries()
doc_lookup = {d['text']: d['id'] for d in corpus}

print("Running evaluation...")
ndcg_scores, recall_scores, mrr_scores = [], [], []
p1_scores, p10_scores, latencies = [], [], []

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
    mrr = 0
    for i, did in enumerate(ids):
        if did in rel:
            mrr = 1.0/(i+1)
            break
    mrr_scores.append(mrr)
    p1_scores.append(1.0 if ids and ids[0] in rel else 0.0)
    t10 = ids[:10]
    p10_scores.append(sum(1 for d in t10 if d in rel)/len(t10) if t10 else 0)

latencies.sort()
print(f"\nRESULTS @ 100K:")
print(f"  NDCG@10:  {sum(ndcg_scores)/len(ndcg_scores):.3f}")
print(f"  MRR:      {sum(mrr_scores)/len(mrr_scores):.3f}")
print(f"  Recall:   {sum(recall_scores)/len(recall_scores)*100:.2f}%")
print(f"  P@1:      {sum(p1_scores)/len(p1_scores)*100:.0f}%")
print(f"  P@10:     {sum(p10_scores)/len(p10_scores)*100:.0f}%")
print(f"  p50:      {latencies[len(latencies)//2]:.2f}ms")
print(f"  p95:      {latencies[int(len(latencies)*0.95)]:.2f}ms")
print(f"  p99:      {latencies[int(len(latencies)*0.99)]:.2f}ms")

if os.path.exists("a10_100k.json"):
    os.remove("a10_100k.json")
