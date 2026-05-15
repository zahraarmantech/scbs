"""
5K Scale Test — Approach 10 Energy-Based Retrieval
====================================================

Testing energy diffusion at standard benchmark scale.

Comparison targets:
- Approach 3 @ 5K:  P@10=58%, Recall=0.1%, p50=3.7ms
- Approach 9 @ 5K:  P@10=92%, Recall=0.2%, p50=60ms
- Approach 10 @ 1K: P@10=86%, Recall=52%, p50=0.1ms

Question: Does energy-based retrieval maintain recall at scale?
"""
import sys, os, time, random, math
from collections import defaultdict

_HERE = os.path.dirname(__file__)
_SRC  = os.path.join(_HERE, "..", "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from scbs.blueprint import BlueprintEncoder
from scbs.matrix_index import make_row

# Import Approach 10's energy-based components
sys.path.insert(0, _HERE)
from test_approach_10 import EnergyStore

# Import the realistic corpus generator from standard_benchmark
sys.path.insert(0, os.path.join(_HERE, "..", "standard_benchmark"))
from test_5k_scale import generate_realistic_corpus, generate_queries, compute_ndcg_at_k

random.seed(42)


def main():
    print("\n" + "="*70)
    print("  APPROACH 10 @ 5K SCALE")
    print("  Energy-Based Retrieval via Graph Diffusion")
    print("="*70)
    
    # Generate corpus
    print("\n  Generating realistic 5K document corpus...")
    corpus = generate_realistic_corpus(5000)
    print(f"    Generated {len(corpus):,} documents")
    print(f"    Avg length: {sum(len(d['text'].split()) for d in corpus) / len(corpus):.0f} words")
    
    # Build index
    print("\n  Building Approach 10 energy index...")
    encoder = BlueprintEncoder("approach10_5k_unknown.json")
    store = EnergyStore(
        n_clusters=14,
        diffusion_iterations=3,
        damping=0.7
    )
    
    corpus_texts = [d['text'] for d in corpus]
    
    t0 = time.perf_counter()
    store.learn(corpus_texts)
    for doc in corpus:
        _, bp, _ = encoder.encode(doc['text'])
        store.add(make_row(bp), doc['text'])
    store.build()
    build_time = time.perf_counter() - t0
    print(f"    Built in {build_time:.1f}s ({len(corpus)/build_time:.0f} doc/sec)")
    
    # Generate queries
    print("\n  Generating 100 test queries...")
    queries = generate_queries()
    
    # Evaluate
    print("\n  Running evaluation...")
    ndcg_scores = []
    recall_scores = []
    mrr_scores = []
    p1_scores = []
    p5_scores = []
    p10_scores = []
    latencies = []
    
    for query_text, keywords in queries:
        # Find relevant docs
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
            for doc in corpus:
                if doc['text'] == r['text']:
                    result_ids.append(doc['id'])
                    break
        
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
    
    # Results
    latencies.sort()
    print(f"\n{'='*70}")
    print(f"  RESULTS — Approach 10 at 5K Scale")
    print(f"{'='*70}")
    print(f"\n  Scale:")
    print(f"    Corpus size:  {len(corpus):,} documents")
    print(f"    Queries:      {len(ndcg_scores)}")
    print(f"\n  Ranking Quality (standard IR metrics):")
    print(f"    NDCG@10:      {sum(ndcg_scores)/len(ndcg_scores):.3f}")
    print(f"    MRR:          {sum(mrr_scores)/len(mrr_scores):.3f}")
    print(f"    Recall@100:   {sum(recall_scores)/len(recall_scores):.3f}")
    print(f"\n  Top-K Precision (user-facing):")
    print(f"    P@1:          {sum(p1_scores)/len(p1_scores)*100:.0f}%")
    print(f"    P@5:          {sum(p5_scores)/len(p5_scores)*100:.0f}%")
    print(f"    P@10:         {sum(p10_scores)/len(p10_scores)*100:.0f}%")
    print(f"\n  Speed:")
    print(f"    p50 latency:  {latencies[len(latencies)//2]:.1f}ms")
    print(f"    p95 latency:  {latencies[int(len(latencies)*0.95)]:.1f}ms")
    print(f"    p99 latency:  {latencies[int(len(latencies)*0.99)]:.1f}ms")
    
    print(f"\n  Comparison Across All Approaches:")
    print(f"    Approach 3  @ 1K:  P@10=96%,  Recall=23%,  p50=0.4ms")
    print(f"    Approach 3  @ 5K:  P@10=58%,  Recall=0.1%, p50=3.7ms")
    print(f"    Approach 9  @ 1K:  P@10=87%,  Recall=81%,  p50=8.2ms")
    print(f"    Approach 9  @ 5K:  P@10=92%,  Recall=0.2%, p50=60ms")
    print(f"    Approach 10 @ 1K:  P@10=86%,  Recall=52%,  p50=0.1ms")
    print(f"    Approach 10 @ 5K:  P@10={sum(p10_scores)/len(p10_scores)*100:.0f}%,  Recall={sum(recall_scores)/len(recall_scores)*100:.0f}%,  p50={latencies[len(latencies)//2]:.1f}ms")
    
    # Clean up
    if os.path.exists("approach10_5k_unknown.json"):
        os.remove("approach10_5k_unknown.json")


if __name__ == "__main__":
    main()
