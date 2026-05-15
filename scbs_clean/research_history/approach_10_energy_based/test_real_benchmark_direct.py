"""
REAL BENCHMARK — Vaswani Collection (Direct Encoding)
======================================================

The BlueprintEncoder was designed for specific domains and doesn't
generalize to arbitrary text (physics abstracts produce only 1-2 slots).

This version bypasses BlueprintEncoder and feeds word→cluster mappings
directly to the EnergyStore. The energy diffusion algorithm itself is
unchanged.

This tests the core ALGORITHM on real benchmark data, isolating it
from the encoder's domain bias.
"""
import sys, os, re, time, math
from collections import defaultdict

_HERE = os.path.dirname(__file__)
_SRC = os.path.join(_HERE, "..", "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from scbs.clustering import build_cooccurrence, cluster_words
sys.path.insert(0, _HERE)
from energy_store import build_semantic_graph, inject_energy, diffuse_energy

DATA_DIR = "/home/claude/benchmark_data/pyterrier/tests/fixtures/vaswani_npl"


def parse_trec_corpus(path):
    corpus = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    docs = re.findall(r'<DOC>\s*<DOCNO>(\S+)</DOCNO>\s*(.*?)</DOC>', content, re.DOTALL)
    for doc_id, text in docs:
        corpus[doc_id.strip()] = text.strip()
    return corpus


def parse_trec_queries(path):
    queries = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    pattern = r'<top>\s*<num>(\S+)</num>\s*<title>\s*(.*?)\s*</title>\s*</top>'
    for match in re.finditer(pattern, content, re.DOTALL):
        queries[match.group(1).strip()] = match.group(2).strip()
    return queries


def parse_qrels(path):
    qrels = defaultdict(dict)
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, did, rel = parts[0], parts[1], parts[2], int(parts[3])
                qrels[qid][did] = rel
    return dict(qrels)


def tokenize(text):
    """Simple tokenization: lowercase words, remove punctuation."""
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


def text_to_clusters(text, word_clusters):
    """Map text to a list of cluster IDs."""
    return [word_clusters[w] for w in tokenize(text) if w in word_clusters]


# ════════════════════════════════════════════════════════════════
#  DIRECT ENERGY STORE (no BlueprintEncoder)
# ════════════════════════════════════════════════════════════════

class DirectEnergyStore:
    """Energy-based store that works directly on word clusters."""
    
    def __init__(self, n_clusters=14, diffusion_iterations=3, damping=0.7):
        self.n_clusters = n_clusters
        self.diffusion_iterations = diffusion_iterations
        self.damping = damping
        self._word_clusters = {}
        self._graph = {}
        self._doc_clusters = {}  # doc_id -> set of clusters
        self._cluster_to_docs = defaultdict(list)
    
    def learn(self, corpus_texts, top_words=10000):
        """Build word clusters and semantic graph."""
        cooc = build_cooccurrence(corpus_texts)
        self._word_clusters = cluster_words(cooc, self.n_clusters, top_words=top_words)
        self._graph = build_semantic_graph(
            corpus_texts, self._word_clusters, self.n_clusters
        )
    
    def add(self, doc_id, text):
        """Add document. Build inverted index from clusters → docs."""
        clusters_in_doc = set(text_to_clusters(text, self._word_clusters))
        self._doc_clusters[doc_id] = clusters_in_doc
        for cid in clusters_in_doc:
            self._cluster_to_docs[cid].append(doc_id)
    
    def search(self, query_text, top_k=100):
        """Energy-based retrieval."""
        query_clusters = text_to_clusters(query_text, self._word_clusters)
        if not query_clusters:
            return [], {}
        
        # Build initial energy (each unique cluster gets unit energy, repeated gets more)
        initial_energy = defaultdict(float)
        for cid in query_clusters:
            initial_energy[cid] += 1.0
        initial_energy = dict(initial_energy)
        
        # Diffuse
        final_energy = diffuse_energy(
            initial_energy, self._graph,
            iterations=self.diffusion_iterations,
            damping=self.damping
        )
        
        # Score documents
        doc_scores = defaultdict(float)
        for cid, energy in final_energy.items():
            if cid in self._cluster_to_docs:
                for did in self._cluster_to_docs[cid]:
                    doc_scores[did] += energy
        
        # Rank
        ranked = sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]
        results = [{'doc_id': did, 'energy': score} for did, score in ranked]
        
        stats = {
            'query_clusters': len(set(query_clusters)),
            'activated_concepts': len(final_energy),
            'candidates_scored': len(doc_scores),
        }
        return results, stats


# ════════════════════════════════════════════════════════════════
#  IR METRICS
# ════════════════════════════════════════════════════════════════

def compute_ndcg(retrieved, relevant, k=10):
    dcg = 0.0
    for i, did in enumerate(retrieved[:k]):
        rel = relevant.get(did, 0)
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal) if rel > 0)
    return dcg / idcg if idcg > 0 else 0.0


def compute_map(retrieved, relevant):
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
    for i, did in enumerate(retrieved):
        if relevant.get(did, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def compute_p_at(retrieved, relevant, k):
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for did in top if relevant.get(did, 0) > 0) / len(top)


def compute_r_at(retrieved, relevant, k):
    top_relevant = sum(1 for did in retrieved[:k] if relevant.get(did, 0) > 0)
    total_relevant = sum(1 for r in relevant.values() if r > 0)
    return top_relevant / total_relevant if total_relevant > 0 else 0.0


# ════════════════════════════════════════════════════════════════
#  RUN BENCHMARK
# ════════════════════════════════════════════════════════════════

def run_benchmark(n_clusters):
    print(f"\n{'─'*70}")
    print(f"  n_clusters = {n_clusters}")
    print(f"{'─'*70}")
    
    store = DirectEnergyStore(n_clusters=n_clusters, diffusion_iterations=3, damping=0.7)
    
    print(f"  Learning clustering...")
    t0 = time.perf_counter()
    store.learn(list(CORPUS.values()))
    print(f"    {time.perf_counter()-t0:.1f}s, {len(store._word_clusters)} words clustered")
    
    print(f"  Indexing...")
    t0 = time.perf_counter()
    for did, text in CORPUS.items():
        store.add(did, text)
    print(f"    {time.perf_counter()-t0:.1f}s")
    
    print(f"  Evaluating on {len(EVAL_QUERIES)} queries...")
    
    ndcg_10, ndcg_100 = [], []
    map_s, mrr_s = [], []
    p1, p5, p10 = [], [], []
    r10, r100 = [], []
    latencies = []
    
    for qid, qtext in EVAL_QUERIES.items():
        rel = QRELS[qid]
        if not rel or not any(r > 0 for r in rel.values()):
            continue
        
        t0 = time.perf_counter()
        results, stats = store.search(qtext, top_k=100)
        latencies.append((time.perf_counter() - t0) * 1000)
        
        retrieved_ids = [r['doc_id'] for r in results]
        
        ndcg_10.append(compute_ndcg(retrieved_ids, rel, k=10))
        ndcg_100.append(compute_ndcg(retrieved_ids, rel, k=100))
        map_s.append(compute_map(retrieved_ids, rel))
        mrr_s.append(compute_mrr(retrieved_ids, rel))
        p1.append(compute_p_at(retrieved_ids, rel, 1))
        p5.append(compute_p_at(retrieved_ids, rel, 5))
        p10.append(compute_p_at(retrieved_ids, rel, 10))
        r10.append(compute_r_at(retrieved_ids, rel, 10))
        r100.append(compute_r_at(retrieved_ids, rel, 100))
    
    latencies.sort()
    n = len(ndcg_10)
    
    return {
        'n_clusters': n_clusters,
        'n_queries': n,
        'ndcg_10': sum(ndcg_10) / n,
        'ndcg_100': sum(ndcg_100) / n,
        'map': sum(map_s) / n,
        'mrr': sum(mrr_s) / n,
        'p1': sum(p1) / n,
        'p5': sum(p5) / n,
        'p10': sum(p10) / n,
        'r10': sum(r10) / n,
        'r100': sum(r100) / n,
        'p50_ms': latencies[len(latencies) // 2],
        'p95_ms': latencies[int(len(latencies) * 0.95)],
    }


def main():
    print("\n" + "="*70)
    print("  REAL BENCHMARK — Vaswani Collection (Direct Encoding)")
    print("  Bypasses BlueprintEncoder, tests energy diffusion directly")
    print("="*70)
    
    print("\nLoading benchmark data...")
    global CORPUS, EVAL_QUERIES, QRELS
    CORPUS = parse_trec_corpus(os.path.join(DATA_DIR, "corpus", "doc-text.trec"))
    queries = parse_trec_queries(os.path.join(DATA_DIR, "query-text.trec"))
    QRELS = parse_qrels(os.path.join(DATA_DIR, "qrels"))
    EVAL_QUERIES = {qid: q for qid, q in queries.items() if qid in QRELS}
    
    print(f"  Documents: {len(CORPUS):,}")
    print(f"  Queries (with qrels): {len(EVAL_QUERIES)}")
    print(f"  Total relevance judgments: {sum(len(r) for r in QRELS.values()):,}")
    
    # Test multiple cluster counts to see if granularity matters
    all_results = []
    for n_clusters in [50, 100, 200, 500]:
        r = run_benchmark(n_clusters)
        all_results.append(r)
    
    # Final comparison
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS ON VASWANI (real benchmark)")
    print(f"{'='*70}")
    print(f"\n  {'n_clusters':>10} {'NDCG@10':>10} {'MAP':>8} {'P@10':>8} {'MRR':>8} {'R@100':>8} {'p50 ms':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for r in all_results:
        print(f"  {r['n_clusters']:>10} {r['ndcg_10']:>10.3f} {r['map']:>8.3f} {r['p10']:>8.3f} {r['mrr']:>8.3f} {r['r100']:>8.3f} {r['p50_ms']:>10.2f}")
    
    print(f"\n  Published baselines on Vaswani (from IR literature):")
    print(f"    BM25:           MAP ≈ 0.30,    NDCG@10 ≈ 0.40-0.45")
    print(f"    TF-IDF:         MAP ≈ 0.27")
    print(f"    Vector space:   MAP ≈ 0.28")
    print(f"    DFR (BM25-like): MAP ≈ 0.32")
    
    best = max(all_results, key=lambda r: r['ndcg_10'])
    print(f"\n  SCBS Approach 10 best:  NDCG@10 = {best['ndcg_10']:.3f}, MAP = {best['map']:.3f}")


if __name__ == "__main__":
    main()
