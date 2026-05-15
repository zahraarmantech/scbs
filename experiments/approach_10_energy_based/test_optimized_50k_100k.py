"""
Approach 10 Optimized — Scale Test (50K and 100K)
===================================================

Tests the generic energy-threshold fix at:
1. 50K — verify it matches original Approach 10's quality
2. 100K — see if the fix lets us scale further

Comparison:
- Approach 10 @ 50K: P@10=97%, NDCG@10=0.969, p50=5.6ms
- Approach 10 Optimized @ 50K: should match quality, be faster
- Approach 10 Optimized @ 100K: see if it holds
"""
import sys, os, time, random, math
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


def run_scale_test(corpus_size, label):
    print(f"\n{'='*70}")
    print(f"  APPROACH 10 OPTIMIZED @ {label}")
    print(f"{'='*70}")
    
    # Generate corpus
    print(f"\n  Generating realistic {corpus_size:,} document corpus...")
    t0 = time.perf_counter()
    corpus = generate_realistic_corpus(corpus_size)
    gen_time = time.perf_counter() - t0
    print(f"    Generated {len(corpus):,} documents in {gen_time:.1f}s")
    
    # Build index
    print(f"\n  Building optimized energy index...")
    encoder = BlueprintEncoder(f"approach10_opt_{label}_unknown.json")
    store = EnergyStoreOptimized(
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
    for i, doc in enumerate(corpus):
        _, bp, _ = encoder.encode(doc['text'])
        store.add(make_row(bp), doc['text'])
        if i > 0 and i % 20000 == 0:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed
            eta = (len(corpus) - i) / rate
            print(f"    Progress: {i:,}/{len(corpus):,} ({rate:.0f} doc/sec, ETA: {eta:.0f}s)")
    store.build()
    build_time = time.perf_counter() - t0
    print(f"    Indexing phase: {build_time:.1f}s ({len(corpus)/build_time:.0f} doc/sec)")
    
    # Generate queries
    queries = generate_queries()
    
    # Pre-build lookup
    doc_lookup = {doc['text']: doc['id'] for doc in corpus}
    
    # Evaluate
    print(f"\n  Running evaluation...")
    ndcg_scores = []
    recall_scores = []
    mrr_scores = []
    p1_scores = []
    p5_scores = []
    p10_scores = []
    latencies = []
    threshold_stats = []  # track how many concepts pass threshold
    
    eval_start = time.perf_counter()
    for q_idx, (query_text, keywords) in enumerate(queries):
        relevant_ids = set()
        for doc in corpus:
            if len(set(doc['text'].lower().split()) & keywords) >= 1:
                relevant_ids.add(doc['id'])
        
        if not relevant_ids:
            continue
        
        _, qbp, _ = encoder.encode(query_text)
        qr = make_row(qbp)
        t0 = time.perf_counter()
        results, stats = store.search(qr, query_text, top_k=100)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        threshold_stats.append(stats.get('above_threshold_concepts', 0))
        
        result_ids = []
        for r in results:
            doc_id = doc_lookup.get(r['text'])
            if doc_id is not None:
                result_ids.append(doc_id)
        
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
    
    eval_time = time.perf_counter() - eval_start
    
    # Results
    latencies.sort()
    print(f"\n  Results:")
    print(f"    Corpus size:        {len(corpus):,}")
    print(f"    Queries evaluated:  {len(ndcg_scores)}")
    print(f"    NDCG@10:            {sum(ndcg_scores)/len(ndcg_scores):.3f}")
    print(f"    MRR:                {sum(mrr_scores)/len(mrr_scores):.3f}")
    print(f"    Recall@100:         {sum(recall_scores)/len(recall_scores):.3f}")
    print(f"    P@1:                {sum(p1_scores)/len(p1_scores)*100:.0f}%")
    print(f"    P@5:                {sum(p5_scores)/len(p5_scores)*100:.0f}%")
    print(f"    P@10:               {sum(p10_scores)/len(p10_scores)*100:.0f}%")
    print(f"    p50 latency:        {latencies[len(latencies)//2]:.1f}ms")
    print(f"    p95 latency:        {latencies[int(len(latencies)*0.95)]:.1f}ms")
    print(f"    p99 latency:        {latencies[int(len(latencies)*0.99)]:.1f}ms")
    print(f"    avg concepts above threshold: {sum(threshold_stats)/len(threshold_stats):.1f}")
    
    # Clean up
    fpath = f"approach10_opt_{label}_unknown.json"
    if os.path.exists(fpath):
        os.remove(fpath)
    
    return {
        'p10': sum(p10_scores)/len(p10_scores)*100,
        'ndcg': sum(ndcg_scores)/len(ndcg_scores),
        'recall': sum(recall_scores)/len(recall_scores)*100,
        'p50': latencies[len(latencies)//2],
        'p95': latencies[int(len(latencies)*0.95)],
    }


def main():
    # Test at 50K to verify fix preserves quality
    results_50k = run_scale_test(50000, "50K")
    
    # Test at 100K to see if fix lets us scale
    results_100k = run_scale_test(100000, "100K")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"  COMPLETE SCALING TABLE")
    print(f"{'='*70}")
    print(f"\n  {'Version':>25} {'P@10':>8} {'NDCG@10':>10} {'p50':>10} {'p95':>10}")
    print(f"  {'─'*25} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Original A10 @ 5K':>25} {'98%':>8} {'0.979':>10} {'0.4ms':>10} {'1.8ms':>10}")
    print(f"  {'Original A10 @ 10K':>25} {'97%':>8} {'0.971':>10} {'0.8ms':>10} {'4.3ms':>10}")
    print(f"  {'Original A10 @ 50K':>25} {'97%':>8} {'0.969':>10} {'5.6ms':>10} {'22.6ms':>10}")
    print(f"  {'─'*25} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
    
    p10_50 = f"{results_50k['p10']:.0f}%"
    ndcg_50 = f"{results_50k['ndcg']:.3f}"
    p50_50 = f"{results_50k['p50']:.1f}ms"
    p95_50 = f"{results_50k['p95']:.1f}ms"
    print(f"  {'Optimized A10 @ 50K':>25} {p10_50:>8} {ndcg_50:>10} {p50_50:>10} {p95_50:>10}")
    
    p10_100 = f"{results_100k['p10']:.0f}%"
    ndcg_100 = f"{results_100k['ndcg']:.3f}"
    p50_100 = f"{results_100k['p50']:.1f}ms"
    p95_100 = f"{results_100k['p95']:.1f}ms"
    print(f"  {'Optimized A10 @ 100K':>25} {p10_100:>8} {ndcg_100:>10} {p50_100:>10} {p95_100:>10}")


if __name__ == "__main__":
    main()
