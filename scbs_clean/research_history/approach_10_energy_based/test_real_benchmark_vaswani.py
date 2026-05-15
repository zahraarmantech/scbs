"""
REAL BENCHMARK — Vaswani Collection
====================================

This is a genuine standard IR benchmark from 1979:
- 11,429 real physics/electronics abstracts
- 93 real informational queries
- 2,083 human-judged relevance judgments

Published baselines on Vaswani (from various papers):
- BM25:        MAP ≈ 0.30,  P@10 ≈ 0.36
- Vector space: MAP ≈ 0.28
- DFR-based:   MAP ≈ 0.32

We measure NDCG@10, MAP, MRR, Recall@100, P@1/5/10.
"""
import sys, os, re, time, math
from collections import defaultdict

_HERE = os.path.dirname(__file__)
_SRC = os.path.join(_HERE, "..", "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from scbs.blueprint import BlueprintEncoder
from scbs.matrix_index import make_row

sys.path.insert(0, _HERE)
from energy_store import EnergyStore


DATA_DIR = "/home/claude/benchmark_data/pyterrier/tests/fixtures/vaswani_npl"


# ════════════════════════════════════════════════════════════════
#  TREC FORMAT PARSERS
# ════════════════════════════════════════════════════════════════

def parse_trec_corpus(path):
    """Parse TREC-format corpus file. Returns dict[doc_id] -> text."""
    corpus = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Match <DOC>...<DOCNO>id</DOCNO>...</DOC>
    docs = re.findall(r'<DOC>\s*<DOCNO>(\S+)</DOCNO>\s*(.*?)</DOC>', content, re.DOTALL)
    for doc_id, text in docs:
        corpus[doc_id.strip()] = text.strip()
    return corpus


def parse_trec_queries(path):
    """Parse TREC-format query file. Returns dict[query_id] -> text."""
    queries = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Match <top><num>id</num><title>text</title></top>
    pattern = r'<top>\s*<num>(\S+)</num>\s*<title>\s*(.*?)\s*</title>\s*</top>'
    for match in re.finditer(pattern, content, re.DOTALL):
        qid = match.group(1).strip()
        text = match.group(2).strip()
        queries[qid] = text
    return queries


def parse_qrels(path):
    """Parse TREC qrels. Returns dict[query_id] -> dict[doc_id] -> relevance."""
    qrels = defaultdict(dict)
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, did, rel = parts[0], parts[1], parts[2], int(parts[3])
                qrels[qid][did] = rel
    return dict(qrels)


# ════════════════════════════════════════════════════════════════
#  IR METRICS
# ════════════════════════════════════════════════════════════════

def compute_ndcg(retrieved, relevant, k=10):
    """NDCG@k."""
    dcg = 0.0
    for i, did in enumerate(retrieved[:k]):
        rel = relevant.get(did, 0)
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal) if rel > 0)
    return dcg / idcg if idcg > 0 else 0.0


def compute_map(retrieved, relevant):
    """Mean Average Precision for a single query."""
    if not relevant:
        return 0.0
    ap = 0.0
    relevant_seen = 0
    for i, did in enumerate(retrieved):
        if relevant.get(did, 0) > 0:
            relevant_seen += 1
            ap += relevant_seen / (i + 1)
    total_relevant = sum(1 for r in relevant.values() if r > 0)
    return ap / total_relevant if total_relevant > 0 else 0.0


def compute_mrr(retrieved, relevant):
    """Reciprocal rank for first relevant document."""
    for i, did in enumerate(retrieved):
        if relevant.get(did, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def compute_precision_at(retrieved, relevant, k):
    """P@k."""
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for did in top if relevant.get(did, 0) > 0) / len(top)


def compute_recall_at(retrieved, relevant, k):
    """Recall@k."""
    top_relevant = sum(1 for did in retrieved[:k] if relevant.get(did, 0) > 0)
    total_relevant = sum(1 for r in relevant.values() if r > 0)
    return top_relevant / total_relevant if total_relevant > 0 else 0.0


# ════════════════════════════════════════════════════════════════
#  MAIN BENCHMARK
# ════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*70)
    print("  REAL BENCHMARK — Vaswani Collection (1979)")
    print("  Testing Approach 10 on standard IR test collection")
    print("="*70)
    
    # Load data
    print("\nLoading benchmark data...")
    corpus = parse_trec_corpus(os.path.join(DATA_DIR, "corpus", "doc-text.trec"))
    queries = parse_trec_queries(os.path.join(DATA_DIR, "query-text.trec"))
    qrels = parse_qrels(os.path.join(DATA_DIR, "qrels"))
    
    print(f"  Documents: {len(corpus):,}")
    print(f"  Queries:   {len(queries)}")
    print(f"  Qrels:     {sum(len(r) for r in qrels.values()):,} judgments across {len(qrels)} queries")
    
    # Filter to queries with qrels
    eval_queries = {qid: q for qid, q in queries.items() if qid in qrels}
    print(f"  Queries with relevance judgments: {len(eval_queries)}")
    
    if len(corpus) == 0:
        print("ERROR: corpus is empty!")
        return
    
    # Show sample data
    sample_doc_id = next(iter(corpus))
    sample_query_id = next(iter(eval_queries))
    print(f"\nSample document ({sample_doc_id}):")
    print(f"  {corpus[sample_doc_id][:200]}")
    print(f"\nSample query ({sample_query_id}):")
    print(f"  {eval_queries[sample_query_id]}")
    print(f"  Relevant docs: {list(qrels[sample_query_id].keys())[:5]}")
    
    # Build SCBS index
    print("\nBuilding SCBS Approach 10 index...")
    encoder = BlueprintEncoder("vaswani_vocab.json")
    store = EnergyStore(n_clusters=14, diffusion_iterations=3, damping=0.7)
    
    corpus_texts = list(corpus.values())
    doc_ids_ordered = list(corpus.keys())
    
    t0 = time.perf_counter()
    store.learn(corpus_texts)
    print(f"  Learning: {time.perf_counter()-t0:.1f}s")
    
    # Map text back to doc_id for retrieval
    text_to_doc_id = {}
    
    t0 = time.perf_counter()
    for doc_id, text in corpus.items():
        _, bp, _ = encoder.encode(text)
        store.add(make_row(bp), text)
        text_to_doc_id[text] = doc_id
        
    store.build()
    build_time = time.perf_counter() - t0
    print(f"  Indexing: {build_time:.1f}s ({len(corpus)/build_time:.0f} doc/sec)")
    
    # Run evaluation
    print(f"\nRunning evaluation on {len(eval_queries)} queries...")
    
    ndcg_10, ndcg_100 = [], []
    map_scores = []
    mrr_scores = []
    p1, p5, p10 = [], [], []
    r10, r100 = [], []
    latencies = []
    
    for qid, query_text in eval_queries.items():
        rel_judgments = qrels[qid]
        if not rel_judgments or not any(r > 0 for r in rel_judgments.values()):
            continue
        
        # Search
        _, qbp, _ = encoder.encode(query_text)
        qr = make_row(qbp)
        
        t0 = time.perf_counter()
        results, _ = store.search(qr, query_text, top_k=100)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        
        # Map results back to doc IDs
        retrieved_doc_ids = []
        for r in results:
            doc_id = text_to_doc_id.get(r['text'])
            if doc_id is not None:
                retrieved_doc_ids.append(doc_id)
        
        # Compute metrics
        ndcg_10.append(compute_ndcg(retrieved_doc_ids, rel_judgments, k=10))
        ndcg_100.append(compute_ndcg(retrieved_doc_ids, rel_judgments, k=100))
        map_scores.append(compute_map(retrieved_doc_ids, rel_judgments))
        mrr_scores.append(compute_mrr(retrieved_doc_ids, rel_judgments))
        p1.append(compute_precision_at(retrieved_doc_ids, rel_judgments, 1))
        p5.append(compute_precision_at(retrieved_doc_ids, rel_judgments, 5))
        p10.append(compute_precision_at(retrieved_doc_ids, rel_judgments, 10))
        r10.append(compute_recall_at(retrieved_doc_ids, rel_judgments, 10))
        r100.append(compute_recall_at(retrieved_doc_ids, rel_judgments, 100))
    
    latencies.sort()
    n = len(ndcg_10)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"  RESULTS — Approach 10 on Vaswani (real benchmark)")
    print(f"{'='*70}")
    print(f"\n  Evaluated on: {n} queries, {len(corpus):,} documents")
    print(f"\n  Quality Metrics:")
    print(f"    NDCG@10:    {sum(ndcg_10)/n:.3f}")
    print(f"    NDCG@100:   {sum(ndcg_100)/n:.3f}")
    print(f"    MAP:        {sum(map_scores)/n:.3f}")
    print(f"    MRR:        {sum(mrr_scores)/n:.3f}")
    print(f"\n  Precision:")
    print(f"    P@1:        {sum(p1)/n*100:.1f}%")
    print(f"    P@5:        {sum(p5)/n*100:.1f}%")
    print(f"    P@10:       {sum(p10)/n*100:.1f}%")
    print(f"\n  Recall:")
    print(f"    R@10:       {sum(r10)/n*100:.1f}%")
    print(f"    R@100:      {sum(r100)/n*100:.1f}%")
    print(f"\n  Latency:")
    print(f"    p50:        {latencies[len(latencies)//2]:.2f}ms")
    print(f"    p95:        {latencies[int(len(latencies)*0.95)]:.2f}ms")
    print(f"    p99:        {latencies[int(len(latencies)*0.99)]:.2f}ms")
    
    print(f"\n  Published baselines on Vaswani (typical values from literature):")
    print(f"    BM25:           MAP ≈ 0.30,  P@10 ≈ 0.36,  NDCG@10 ≈ 0.40-0.45")
    print(f"    Vector space:   MAP ≈ 0.28")
    print(f"    DFR (BM25-like): MAP ≈ 0.32")
    print(f"\n  Our results:    MAP = {sum(map_scores)/n:.2f}, P@10 = {sum(p10)/n:.2f}, NDCG@10 = {sum(ndcg_10)/n:.2f}")
    
    # Clean up
    if os.path.exists("vaswani_vocab.json"):
        os.remove("vaswani_vocab.json")


if __name__ == "__main__":
    main()
