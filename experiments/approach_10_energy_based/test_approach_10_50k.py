"""
50K Scale Test — Approach 10 Energy-Based Retrieval
====================================================

Pushing to find the breaking point.

Prior results:
- 1K:  P@10=86%, Recall=52%, p50=0.1ms
- 5K:  P@10=98%, Recall=14%, p50=0.4ms
- 10K: P@10=97%, Recall=7%,  p50=0.8ms

Question: At 50K, does precision still hold? Does speed stay sub-ms?
"""
import sys, os, time, random, math
from collections import defaultdict

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


def main():
    print("\n" + "="*70)
    print("  APPROACH 10 @ 50K SCALE — FINDING THE LIMIT")
    print("="*70)
    
    # Generate corpus
    print("\n  Generating realistic 50K document corpus...")
    t0 = time.perf_counter()
    corpus = generate_realistic_corpus(50000)
    gen_time = time.perf_counter() - t0
    print(f"    Generated {len(corpus):,} documents in {gen_time:.1f}s")
    print(f"    Avg length: {sum(len(d['text'].split()) for d in corpus) / len(corpus):.0f} words")
    
    # Build index
    print("\n  Building Approach 10 energy index at 50K scale...")
    print("    (This will take ~3-4 minutes)")
    encoder = BlueprintEncoder("approach10_50k_unknown.json")
    store = EnergyStore(
        n_clusters=14,
        diffusion_iterations=3,
        damping=0.7
    )
    
    corpus_texts = [d['text'] for d in corpus]
    
    t0 = time.perf_counter()
    store.learn(corpus_texts)
    learn_time = time.perf_counter() - t0
    print(f"    Learning phase: {learn_time:.1f}s")
    
    t0 = time.perf_counter()
    last_report = t0
    for i, doc in enumerate(corpus):
        _, bp, _ = encoder.encode(doc['text'])
        store.add(make_row(bp), doc['text'])
        if i % 10000 == 0 and i > 0:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed
            eta = (len(corpus) - i) / rate
            print(f"    Progress: {i:,}/{len(corpus):,} ({rate:.0f} doc/sec, ETA: {eta:.0f}s)")
    store.build()
    build_time = time.perf_counter() - t0
    print(f"    Indexing phase: {build_time:.1f}s ({len(corpus)/build_time:.0f} doc/sec)")
    print(f"    Total build:    {learn_time + build_time:.1f}s")
    
    # Generate queries
    print("\n  Generating 100 test queries...")
    queries = generate_queries()
    
    # Pre-build lookup for faster ID mapping at scale
    print("  Building doc lookup table...")
    doc_lookup = {doc['text']: doc['id'] for doc in corpus}
    
    # Evaluate
    print("\n  Running evaluation...")
    ndcg_scores = []
    recall_scores = []
    mrr_scores = []
    p1_scores = []
    p5_scores = []
    p10_scores = []
    latencies = []
    
    eval_start = time.perf_counter()
    for q_idx, (query_text, keywords) in enumerate(queries):
        # Find relevant docs (this is expensive at 50K)
        relevant_ids = set()
        for doc in corpus:
            if len(set(doc['text'].lower().split()) & keywords) >= 1:
                relevant_ids.add(doc['id'])
        
        if not relevant_ids:
            continue
        
        # Search
        _, qbp, _ = encoder.encode(query_text)
        qr = make_row(qbp)
        t0 = time.perf_counter()
        results, _ = store.search(qr, query_text, top_k=100)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        
        # Map results to doc IDs
        result_ids = []
        for r in results:
            doc_id = doc_lookup.get(r['text'])
            if doc_id is not None:
                result_ids.append(doc_id)
        
        # Metrics
        ndcg = compute_ndcg_at_k(result_ids, relevant_ids, k=10)
        recall = len(set(result_ids[:100]) & relevant_ids) / len(relevant_ids)
        
        mrr = 0.0
        for i, doc_id in enumerate(result_ids):
            if doc_id in relevant_ids:
                mrr = 1.0 / (i + 1)
                break
        
        p1 = 1.0 if result_ids and result_ids[0] in relevant_ids else 0.0
        top5 = result_ids[:5]
        p5 = sum(1 for d in top5 if d in relevant_ids) / len(top5) if top5 else 0.0
        top10 = result_ids[:10]
        p10 = sum(1 for d in top10 if d in relevant_ids) / len(top10) if top10 else 0.0
        
        ndcg_scores.append(ndcg)
        recall_scores.append(recall)
        mrr_scores.append(mrr)
        p1_scores.append(p1)
        p5_scores.append(p5)
        p10_scores.append(p10)
        
        if (q_idx + 1) % 25 == 0:
            print(f"    Evaluated {q_idx+1}/{len(queries)} queries...")
    
    eval_time = time.perf_counter() - eval_start
    print(f"    Evaluation completed in {eval_time:.1f}s")
    
    # Results
    latencies.sort()
    print(f"\n{'='*70}")
    print(f"  RESULTS — Approach 10 at 50K Scale")
    print(f"{'='*70}")
    print(f"\n  Scale:")
    print(f"    Corpus size:  {len(corpus):,} documents")
    print(f"    Queries:      {len(ndcg_scores)}")
    print(f"\n  Ranking Quality:")
    print(f"    NDCG@10:      {sum(ndcg_scores)/len(ndcg_scores):.3f}")
    print(f"    MRR:          {sum(mrr_scores)/len(mrr_scores):.3f}")
    print(f"    Recall@100:   {sum(recall_scores)/len(recall_scores):.3f}")
    print(f"\n  Top-K Precision:")
    print(f"    P@1:          {sum(p1_scores)/len(p1_scores)*100:.0f}%")
    print(f"    P@5:          {sum(p5_scores)/len(p5_scores)*100:.0f}%")
    print(f"    P@10:         {sum(p10_scores)/len(p10_scores)*100:.0f}%")
    print(f"\n  Speed:")
    print(f"    p50 latency:  {latencies[len(latencies)//2]:.1f}ms")
    print(f"    p95 latency:  {latencies[int(len(latencies)*0.95)]:.1f}ms")
    print(f"    p99 latency:  {latencies[int(len(latencies)*0.99)]:.1f}ms")
    print(f"    max latency:  {latencies[-1]:.1f}ms")
    
    print(f"\n{'='*70}")
    print(f"  COMPLETE SCALING TABLE — Approach 10")
    print(f"{'='*70}")
    print(f"\n  {'Scale':>8} {'P@10':>8} {'NDCG@10':>10} {'Recall':>8} {'p50':>10} {'p95':>10}")
    print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*10} {'─'*10}")
    print(f"  {'1K':>8} {'86%':>8} {'-':>10} {'52%':>8} {'0.1ms':>10} {'-':>10}")
    print(f"  {'5K':>8} {'98%':>8} {'0.979':>10} {'14%':>8} {'0.4ms':>10} {'1.8ms':>10}")
    print(f"  {'10K':>8} {'97%':>8} {'0.971':>10} {'7%':>8} {'0.8ms':>10} {'4.3ms':>10}")
    
    p10_50k = sum(p10_scores)/len(p10_scores)*100
    ndcg_50k = sum(ndcg_scores)/len(ndcg_scores)
    r_50k = sum(recall_scores)/len(recall_scores)*100
    p50_50k = latencies[len(latencies)//2]
    p95_50k = latencies[int(len(latencies)*0.95)]
    
    print(f"  {'50K':>8} {f'{p10_50k:.0f}%':>8} {f'{ndcg_50k:.3f}':>10} {f'{r_50k:.0f}%':>8} {f'{p50_50k:.1f}ms':>10} {f'{p95_50k:.1f}ms':>10}")
    
    # Clean up
    if os.path.exists("approach10_50k_unknown.json"):
        os.remove("approach10_50k_unknown.json")


if __name__ == "__main__":
    main()
