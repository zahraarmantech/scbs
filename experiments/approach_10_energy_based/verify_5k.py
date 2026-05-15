"""Quick verification at 5K - is the fix correct?"""
import sys, os, time, random

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

print("Quick 5K verification of optimized version...")
corpus = generate_realistic_corpus(5000)
encoder = BlueprintEncoder("verify.json")
store = EnergyStoreOptimized(n_clusters=14)

t0 = time.perf_counter()
store.learn([d['text'] for d in corpus])
for doc in corpus:
    _, bp, _ = encoder.encode(doc['text'])
    store.add(make_row(bp), doc['text'])
store.build()
print(f"Built in {time.perf_counter()-t0:.1f}s")

queries = generate_queries()
doc_lookup = {d['text']: d['id'] for d in corpus}

ndcg, p10, lats, thresh = [], [], [], []
for q, kw in queries:
    rel = {d['id'] for d in corpus if len(set(d['text'].lower().split()) & kw) >= 1}
    if not rel: continue
    _, qbp, _ = encoder.encode(q)
    qr = make_row(qbp)
    t0 = time.perf_counter()
    res, stats = store.search(qr, q, top_k=100)
    lats.append((time.perf_counter()-t0)*1000)
    thresh.append(stats.get('above_threshold_concepts', 0))
    ids = [doc_lookup[r['text']] for r in res if r['text'] in doc_lookup]
    ndcg.append(compute_ndcg_at_k(ids, rel, k=10))
    t10 = ids[:10]
    p10.append(sum(1 for d in t10 if d in rel)/len(t10) if t10 else 0)

lats.sort()
print(f"\n@ 5K:")
print(f"  Original:  P@10=98%, NDCG=0.979, p50=0.4ms, p95=1.8ms")
print(f"  Optimized: P@10={sum(p10)/len(p10)*100:.0f}%, NDCG={sum(ndcg)/len(ndcg):.3f}, p50={lats[len(lats)//2]:.2f}ms, p95={lats[int(len(lats)*0.95)]:.2f}ms")
print(f"  Avg concepts passing threshold: {sum(thresh)/len(thresh):.1f}/14")

if os.path.exists("verify.json"):
    os.remove("verify.json")
