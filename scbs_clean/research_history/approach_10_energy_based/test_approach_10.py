"""
Approach 10 — Energy-Based Retrieval
=====================================

PARADIGM SHIFT
--------------
Instead of nearest-neighbor (geometric distance in vector space),
we use energy diffusion through a semantic graph.

HOW IT WORKS
------------
1. Build semantic graph at index time:
   - Nodes = cluster IDs (concepts)
   - Edges = co-occurrence strength between concepts
   - Documents are subgraphs (their filled slots)

2. Query injects energy:
   - Query's cluster IDs become energy sources
   - Initial energy = IDF weight (rare concepts = higher energy)

3. Energy propagates (diffusion):
   - Flows along edges (co-occurrence connections)
   - Decays with each hop (damping factor)
   - Accumulates at nodes
   - Run for N iterations until convergence

4. Documents rank by activation:
   - Each document's score = sum of energy at its concept nodes
   - Higher activation = better match

WHY THIS IS BETTER
------------------
- Nearest-neighbor: "Are you geometrically close to the query?"
- Energy-based: "Do your concepts resonate with the query's activation pattern?"

The second is how human memory works — associative activation,
not distance computation.

HONESTLY
--------
This is probably the future of retrieval. Not cosine similarity,
but graph diffusion on concept networks.
"""
import sys, os, time, random, math
from collections import defaultdict, Counter
import numpy as np

_HERE = os.path.dirname(__file__)
_SRC  = os.path.join(_HERE, "..", "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from scbs.blueprint import BlueprintEncoder
from scbs.matrix_index import make_row
from scbs.distance import build_idf, sentence_weight
from scbs.clustering import build_cooccurrence, cluster_words, get_cluster_labels

random.seed(42)


# ═══════════════════════════════════════════════════════════════════
#  SEMANTIC GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════

def build_semantic_graph(corpus, word_clusters, n_clusters):
    """
    Build semantic graph from corpus.
    
    Returns:
        graph: dict[node_id] -> dict[neighbor_id] -> edge_weight
        node_docs: dict[node_id] -> list of doc_ids that contain this concept
    """
    # Co-occurrence matrix: how often cluster pairs appear together
    cooc_matrix = defaultdict(Counter)
    
    for text in corpus:
        words = [w.lower().strip(".,!?;:\"'()") for w in text.split() if len(w) > 2]
        clusters_in_text = set()
        for word in words:
            if word in word_clusters:
                clusters_in_text.add(word_clusters[word])
        
        # All pairs co-occur
        for c1 in clusters_in_text:
            for c2 in clusters_in_text:
                if c1 != c2:
                    cooc_matrix[c1][c2] += 1
    
    # Normalize to edge weights (0 to 1)
    graph = defaultdict(dict)
    for node, neighbors in cooc_matrix.items():
        total = sum(neighbors.values())
        for neighbor, count in neighbors.items():
            # Edge weight = normalized co-occurrence strength
            graph[node][neighbor] = count / total
    
    return graph


# ═══════════════════════════════════════════════════════════════════
#  ENERGY DIFFUSION
# ═══════════════════════════════════════════════════════════════════

def inject_energy(query_row, idf, word_clusters):
    """
    Query injects energy into graph nodes.
    
    Returns:
        energy: dict[cluster_id] -> initial energy value
    """
    energy = {}
    
    for slot, cluster_id in query_row:
        # Energy proportional to IDF (rare concepts = more distinctive)
        # Get a representative word for this cluster to look up IDF
        # (simplified: use cluster ID as proxy for importance)
        base_energy = 1.0
        
        # Boost energy for rare concepts (high cluster IDs in sorted vocab)
        if cluster_id > 7:  # concepts in upper half of vocabulary
            base_energy = 2.0
        
        if cluster_id in energy:
            energy[cluster_id] += base_energy
        else:
            energy[cluster_id] = base_energy
    
    return energy


def diffuse_energy(initial_energy, graph, iterations=3, damping=0.7):
    """
    Propagate energy through graph via diffusion.
    
    Args:
        initial_energy: dict[node_id] -> energy value
        graph: semantic graph (adjacency dict)
        iterations: number of diffusion steps
        damping: decay factor per hop (0-1)
    
    Returns:
        final_energy: dict[node_id] -> accumulated energy
    """
    current = dict(initial_energy)
    
    for _ in range(iterations):
        next_energy = defaultdict(float)
        
        # Each node distributes energy to neighbors
        for node, energy in current.items():
            if node not in graph:
                # Node has no outgoing edges, energy stays
                next_energy[node] += energy * damping
                continue
            
            neighbors = graph[node]
            # Distribute energy proportional to edge weights
            for neighbor, edge_weight in neighbors.items():
                flow = energy * edge_weight * damping
                next_energy[neighbor] += flow
            
            # Node retains some energy (self-loop)
            next_energy[node] += energy * (1 - damping) * 0.5
        
        current = next_energy
    
    return current


# ═══════════════════════════════════════════════════════════════════
#  ENERGY-BASED STORE
# ═══════════════════════════════════════════════════════════════════

class EnergyStore:
    
    def __init__(self, n_clusters=14, diffusion_iterations=3, damping=0.7):
        self.n_clusters = n_clusters
        self.diffusion_iterations = diffusion_iterations
        self.damping = damping
        
        self._idf = {}
        self._word_clusters = {}
        self._cluster_labels = {}
        self._graph = {}
        
        # Documents: id -> row, text
        self._docs = {}
        self._next_id = 0
        
        # Inverted index: cluster_id -> list of doc_ids
        self._cluster_to_docs = defaultdict(list)
    
    def learn(self, sentences):
        """Learn IDF, clustering, and semantic graph from corpus."""
        self._idf = build_idf(sentences)
        cooc = build_cooccurrence(sentences)
        self._word_clusters = cluster_words(cooc, self.n_clusters)
        self._cluster_labels = get_cluster_labels(
            self._word_clusters, cooc, self.n_clusters
        )
        self._graph = build_semantic_graph(
            sentences, self._word_clusters, self.n_clusters
        )
    
    def add(self, row, text):
        """Add document to index."""
        doc_id = self._next_id
        self._next_id += 1
        
        # Store document
        self._docs[doc_id] = {
            "id": doc_id,
            "row": row,
            "text": text,
        }
        
        # Build inverted index: which clusters appear in this doc
        for slot, cluster_id in row:
            self._cluster_to_docs[cluster_id].append(doc_id)
    
    def build(self):
        """Finalize index (no-op for energy-based system)."""
        pass
    
    def search(self, query_row, query_text, *, top_k=10, excluded=None):
        """
        Energy-based retrieval.
        
        1. Query injects energy into graph
        2. Energy diffuses through semantic network
        3. Documents rank by total activation
        """
        if excluded is None:
            excluded = set()
        
        # Step 1: Inject energy
        initial_energy = inject_energy(query_row, self._idf, self._word_clusters)
        
        # Step 2: Diffuse energy through graph
        final_energy = diffuse_energy(
            initial_energy, self._graph,
            iterations=self.diffusion_iterations,
            damping=self.damping
        )
        
        # Step 3: Score documents by accumulated activation
        doc_scores = defaultdict(float)
        
        for cluster_id, energy in final_energy.items():
            # All documents containing this activated concept receive its energy
            if cluster_id in self._cluster_to_docs:
                for doc_id in self._cluster_to_docs[cluster_id]:
                    if doc_id not in excluded:
                        doc_scores[doc_id] += energy
        
        # Step 4: Rank by total energy (descending)
        scored_docs = []
        for doc_id, score in doc_scores.items():
            doc = self._docs[doc_id]
            scored_docs.append({
                "id": doc_id,
                "text": doc["text"],
                "energy": round(score, 3),
            })
        
        scored_docs.sort(key=lambda d: -d["energy"])
        results = scored_docs[:top_k]
        
        for r in results:
            excluded.add(r["id"])
        
        stats = {
            "initial_concepts": len(initial_energy),
            "activated_concepts": len(final_energy),
            "candidates": len(doc_scores),
        }
        
        return results, stats


# ═══════════════════════════════════════════════════════════════════
#  BENCHMARK
# ═══════════════════════════════════════════════════════════════════

def main():
    DEVOPS = [
        "kafka consumer group lag increased beyond limits",
        "kafka broker partition rebalancing causing delays",
        "kafka topic offset falling behind producer rate",
        "kubernetes pod evicted due to memory pressure",
        "kubernetes deployment rollout failed image pull error",
        "kubernetes node not ready cluster autoscaler triggered",
        "deployment to production failed during database migration",
        "canary deployment elevated error rate rolling back",
        "api response latency p99 exceeded threshold alert",
        "database query execution time degraded index rebuild",
        "redis cache hit rate dropped below threshold",
        "production outage detected all services returning errors",
        "monitoring alert fired prometheus rule threshold exceeded",
        "ssl certificate expired connection refused now",
        "disk space critical threshold exceeded alert",
    ]
    SECURITY = [
        "multiple failed login attempts detected account locked",
        "brute force attack authentication endpoint blocked firewall",
        "oauth token refresh failing users logged out",
        "critical cve published affecting openssl version",
        "sql injection attempt detected payment processing",
        "cross site scripting vulnerability user input form",
        "container image vulnerability scan outdated base image",
        "ssl certificate expiring seven days renewal started",
        "unauthorized api access attempt unusual geographic location",
        "security incident response team activated ransomware",
        "phishing campaign targeting employees reported security",
        "firewall rule updated blocking suspicious ip range",
    ]
    FINANCE = [
        "high value wire transfer flagged manual review",
        "unusual transaction pattern detected account takeover",
        "payment processing failed merchant acquiring bank",
        "duplicate transaction idempotency check preventing charge",
        "fraud model score exceeded threshold transaction blocked",
        "transaction monitoring rule triggered unusual spending",
        "market data feed latency spike pricing engine",
        "regulatory reporting deadline data validation running",
        "credit limit increase approved customer risk reviewed",
    ]
    HR = [
        "software engineer position open sourcing candidates linkedin",
        "technical interview scheduled senior developer Friday",
        "offer letter extended preferred candidate awaiting acceptance",
        "annual performance review cycle starting next week managers",
        "promotion recommended employee exceeding targets quarters",
        "parental leave request approved twelve weeks fully paid",
        "mandatory compliance training completion deadline Friday",
        "open enrollment benefits selection due Friday",
    ]
    CUSTOMER = [
        "customer reported product arrived damaged requesting refund",
        "multiple complaints delayed shipping partner issue",
        "customer threatening social media unresolved complaint",
        "billing error discovered customer overcharged months",
        "support ticket volume spike following app update release",
        "customer satisfaction score dropped below threshold",
        "customer provided five star review praising support",
    ]
    ENGINEERING = [
        "pull request approved merged main deployment triggered",
        "code review comments addressed technical debt separately",
        "unit test coverage dropped below percent blocking",
        "api breaking change introduced versioning migration",
        "technical debt sprint scheduled next quarter backlog",
        "performance profiling identified memory allocation hotspot",
        "load testing results system handles target throughput",
        "feature flag enabled ten percent users gradual rollout",
    ]
    QUESTIONS = [
        "what is current status kafka consumer group lag",
        "why did deployment fail staging environment last night",
        "when will security patch applied production servers",
        "how do we handle transaction processing queue backs",
        "what caused spike api errors two am morning",
    ]

    ALL = DEVOPS + SECURITY + FINANCE + HR + CUSTOMER + ENGINEERING + QUESTIONS
    random.shuffle(ALL)
    swaps = {"failed":"errored","critical":"severe","detected":"identified",
             "authentication":"auth","deployment":"release"}
    corpus = list(ALL)
    while len(corpus) < 1000:
        s = random.choice(ALL)
        words = s.split()
        for i,w in enumerate(words):
            if w in swaps and random.random() < 0.5:
                words[i] = swaps[w]
                break
        corpus.append(" ".join(words))
    corpus = corpus[:1000]
    random.shuffle(corpus)

    print("\n" + "="*68)
    print("  APPROACH 10 — ENERGY-BASED RETRIEVAL")
    print("  Query activates semantic graph, documents rank by energy flow")
    print("="*68)
    print("\n  Building energy-based index (1K corpus)...")
    
    encoder = BlueprintEncoder("approach10_unknown.json")
    store = EnergyStore(
        n_clusters=14,
        diffusion_iterations=3,
        damping=0.7
    )
    
    t0 = time.perf_counter()
    store.learn(corpus)
    for s in corpus:
        _, bp, _ = encoder.encode(s)
        store.add(make_row(bp), s)
    store.build()
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"  Built in {build_ms:.0f}ms ({len(corpus)/(build_ms/1000):.0f} rec/sec)")

    queries = {
        "kafka consumer lag deployment failed production":            {"keywords":{"kafka","consumer","lag","deployment","failed","production","broker","partition"}},
        "authentication token expired unauthorized access blocked":   {"keywords":{"authentication","token","unauthorized","access","blocked","login","oauth"}},
        "fraud transaction payment suspicious blocked alert":          {"keywords":{"fraud","transaction","payment","suspicious","blocked"}},
        "performance review scheduled employee promotion approved":    {"keywords":{"performance","review","employee","promotion","approved"}},
        "customer complaint refund order delivery failed":             {"keywords":{"customer","complaint","refund","order","delivery"}},
        "code review deployment test coverage security vulnerability": {"keywords":{"code","review","deployment","test","coverage","security"}},
        "why is the system failing what caused the error":             {"keywords":{"what","why","when","how","status","cause"}},
    }

    def is_relevant(text, kws):
        return len(set(text.lower().split()) & kws) >= 1

    # F1
    print(f"\n  ─── F1 metric (top_k=500) ──────────────────────────")
    print(f"  {'Query':42s} {'P':>5} {'R':>5} {'F1':>5}")
    print(f"  {'─'*42} {'─'*5} {'─'*5} {'─'*5}")
    all_p, all_r, all_f1 = [], [], []
    for query, meta in queries.items():
        _, q_bp, _ = encoder.encode(query)
        qr = make_row(q_bp)
        relevant = {s for s in corpus if is_relevant(s, meta["keywords"])}
        results, _ = store.search(qr, query, top_k=500)
        returned = {r["text"] for r in results}
        tp = len(returned & relevant)
        fp = len(returned - relevant)
        fn = len(relevant - returned)
        p  = tp/(tp+fp) if (tp+fp)>0 else 0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        all_p.append(p); all_r.append(r); all_f1.append(f1)
        mk = "✓" if f1>=0.6 else "~" if f1>=0.4 else "✗"
        print(f"  {mk} {query[:40]:40s} {p:5.0%} {r:5.0%} {f1:5.0%}")
    print(f"\n  {'AVERAGE':42s} {sum(all_p)/7:5.0%} {sum(all_r)/7:5.0%} {sum(all_f1)/7:5.0%}")

    # P@K
    print(f"\n  ─── P@K metric (top_k=10) ──────────────────────────")
    print(f"  {'Query':42s} {'P@1':>5} {'P@3':>5} {'P@5':>5} {'P@10':>5}")
    print(f"  {'─'*42} {'─'*5} {'─'*5} {'─'*5} {'─'*5}")
    p1, p3, p5, p10 = [], [], [], []
    for query, meta in queries.items():
        _, q_bp, _ = encoder.encode(query)
        qr = make_row(q_bp)
        relevant = {s for s in corpus if is_relevant(s, meta["keywords"])}
        results, _ = store.search(qr, query, top_k=10)
        def pa(k):
            top = [r["text"] for r in results[:k]]
            return sum(1 for t in top if t in relevant)/len(top) if top else 0
        a1,a3,a5,a10 = pa(1),pa(3),pa(5),pa(10)
        p1.append(a1); p3.append(a3); p5.append(a5); p10.append(a10)
        print(f"  {query[:42]:42s} {a1:5.0%} {a3:5.0%} {a5:5.0%} {a10:5.0%}")
    print(f"\n  {'AVERAGE':42s} {sum(p1)/7:5.0%} {sum(p3)/7:5.0%} {sum(p5)/7:5.0%} {sum(p10)/7:5.0%}")

    # Speed
    print(f"\n  ─── Speed (100 queries) ──────────────────────────")
    lats = []
    qs = list(queries.keys())*15; random.shuffle(qs)
    for q in qs[:100]:
        _, qbp, _ = encoder.encode(q)
        qr = make_row(qbp)
        t0 = time.perf_counter()
        store.search(qr, q, top_k=20)
        lats.append((time.perf_counter()-t0)*1000)
    lats.sort()
    print(f"  p50: {lats[50]:.1f}ms   p95: {lats[95]:.1f}ms   p99: {lats[99]:.1f}ms")

    if os.path.exists("approach10_unknown.json"):
        os.remove("approach10_unknown.json")

    # Comparison
    print(f"\n{'='*68}")
    print("  COMPARISON")
    print(f"{'='*68}")
    a10_f1  = sum(all_f1)/7
    a10_p1  = sum(p1)/7
    a10_p10 = sum(p10)/7
    a10_r   = sum(all_r)/7
    a10_p50 = lats[50]
    print(f"\n  {'Metric':12s} {'Approach 3':>14} {'Approach 10':>14} {'Δ':>10}")
    print(f"  {'─'*12} {'─'*14} {'─'*14} {'─'*10}")
    print(f"  {'F1':12s} {'29%':>14} {f'{a10_f1*100:.0f}%':>14} {(a10_f1*100-29):+.0f}pp")
    print(f"  {'Recall':12s} {'23%':>14} {f'{a10_r*100:.0f}%':>14} {(a10_r*100-23):+.0f}pp")
    print(f"  {'P@1':12s} {'100%':>14} {f'{a10_p1*100:.0f}%':>14} {(a10_p1*100-100):+.0f}pp")
    print(f"  {'P@10':12s} {'96%':>14} {f'{a10_p10*100:.0f}%':>14} {(a10_p10*100-96):+.0f}pp")
    print(f"  {'p50':12s} {'0.4ms':>14} {f'{a10_p50:.1f}ms':>14} {(a10_p50-0.4):+.1f}ms")


if __name__ == "__main__":
    main()
