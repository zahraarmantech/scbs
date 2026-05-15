"""
Approach 11 — Directed Flow with Genericity Penalty
=====================================================

REFORMULATION (from symmetric distance to directed flow)
---------------------------------------------------------
Old:     score(q, d) = energy_at_doc_concepts  (no discrimination)
New:     score(q, d) = Φ(q → d) = Σ idf(c) · energy(c → d)

WHY THIS MIGHT FIX CRANFIELD UNDERPERFORMANCE
----------------------------------------------
Old system gave generic concepts equal weight as specific ones.
That meant docs containing common technical words ("system", "method",
"results") scored high regardless of relevance.

New system:
- Specific concepts (appearing in few docs) have HIGH idf
- Generic concepts (appearing in many docs) have LOW idf
- Energy gets weighted by idf when scoring documents
- Initial query injection is also idf-weighted

This is conceptually parallel to what BM25 does with its IDF term —
but applied to the graph diffusion framework.

DIRECTIONAL ASYMMETRY (already present, now used)
--------------------------------------------------
Graph edges W[c1][c2] = count(c1,c2) / total_from_c1 are non-Hermitian
by construction. We now leverage this:
- Outflow from a node = how much it spreads (already in diffusion)
- Inflow to a node = how concentrated its sources are
- Genericity = inverse of inflow concentration

Generic nodes get crushed; specific nodes get amplified.
"""
import math
from collections import defaultdict, Counter


def build_directed_semantic_graph(corpus, word_clusters, n_clusters):
    """
    Build directed co-occurrence graph with explicit outflow.
    Returns:
        graph: dict[from] -> dict[to] -> edge_weight (normalized by outflow)
        cluster_doc_count: dict[cluster_id] -> num_docs_containing
    """
    cooc_matrix = defaultdict(Counter)
    cluster_doc_count = defaultdict(int)
    
    for text in corpus:
        words = [w.lower().strip(".,!?;:\"'()") for w in text.split() if len(w) > 2]
        clusters_in_text = set()
        for word in words:
            if word in word_clusters:
                clusters_in_text.add(word_clusters[word])
        
        # Track document frequency per cluster
        for c in clusters_in_text:
            cluster_doc_count[c] += 1
        
        # Build co-occurrence (asymmetric by normalization)
        for c1 in clusters_in_text:
            for c2 in clusters_in_text:
                if c1 != c2:
                    cooc_matrix[c1][c2] += 1
    
    # Normalize: W[i][j] = P(j appears | i appears)
    graph = defaultdict(dict)
    for node, neighbors in cooc_matrix.items():
        total = sum(neighbors.values())
        if total > 0:
            for neighbor, count in neighbors.items():
                graph[node][neighbor] = count / total
    
    return graph, dict(cluster_doc_count)


def compute_cluster_idf(cluster_doc_count, total_docs):
    """
    Cluster-level IDF: how specific is each concept.
    
    cluster_idf[c] = log((N + 1) / (df + 1)) + 1
    
    Generic clusters (in many docs) get LOW idf.
    Specific clusters (in few docs) get HIGH idf.
    """
    cluster_idf = {}
    for c, df in cluster_doc_count.items():
        # Standard IDF formula with smoothing
        cluster_idf[c] = math.log((total_docs + 1) / (df + 1)) + 1
    return cluster_idf


def compute_genericity_penalty(graph, cluster_doc_count, total_docs):
    """
    Per-cluster penalty for being generic.
    
    Generic clusters have:
    - Many outgoing connections to varied targets (high entropy)
    - Appear in many documents (high doc_count)
    
    Returns multiplier in [0, 1]: 1 = specific (no penalty), <1 = generic
    """
    penalties = {}
    for c in range(len({**graph, **{c:None for c in cluster_doc_count}}.keys()) + 1):
        df = cluster_doc_count.get(c, 0)
        if df == 0:
            penalties[c] = 1.0
            continue
        
        # Fraction of documents containing this cluster
        doc_freq_ratio = df / total_docs
        
        # Penalty: linear decay
        # If cluster appears in 50%+ of docs, heavy penalty
        # If cluster appears in <5% of docs, no penalty
        if doc_freq_ratio < 0.05:
            penalty = 1.0
        elif doc_freq_ratio > 0.5:
            penalty = 0.1
        else:
            # Linear interpolation
            penalty = 1.0 - (doc_freq_ratio - 0.05) / 0.45 * 0.9
        
        penalties[c] = penalty
    
    return penalties


def inject_energy_idf_weighted(query_row, cluster_idf):
    """
    Inject energy weighted by cluster IDF.
    Specific concepts get more initial energy.
    """
    energy = {}
    for slot, cluster_id in query_row:
        # Weight by IDF — specific concepts inject more energy
        idf_weight = cluster_idf.get(cluster_id, 1.0)
        energy[cluster_id] = energy.get(cluster_id, 0) + idf_weight
    return energy


def diffuse_energy_directed(initial_energy, graph, iterations=3, damping=0.7):
    """
    Directed diffusion through the asymmetric graph.
    Same as before but emphasizing the flow direction.
    """
    current = dict(initial_energy)
    
    for _ in range(iterations):
        next_energy = defaultdict(float)
        
        for node, energy in current.items():
            if node not in graph or not graph[node]:
                next_energy[node] += energy * damping
                continue
            
            # Distribute along outgoing edges
            for neighbor, edge_weight in graph[node].items():
                next_energy[neighbor] += energy * edge_weight * damping
            
            # Retention
            next_energy[node] += energy * (1 - damping) * 0.5
        
        current = next_energy
    
    return current


# ═══════════════════════════════════════════════════════════════════
#  DIRECTED-FLOW ENERGY STORE
# ═══════════════════════════════════════════════════════════════════

class DirectedFlowStore:
    """
    Energy retrieval with directed flow and genericity penalty.
    
    Key changes from Approach 10:
    1. Initial energy injection weighted by cluster IDF
    2. Documents scored by energy weighted by cluster IDF
    3. Generic clusters penalized in final scoring
    """
    
    def __init__(self, n_clusters=14, diffusion_iterations=3, damping=0.7):
        self.n_clusters = n_clusters
        self.diffusion_iterations = diffusion_iterations
        self.damping = damping
        
        self._word_clusters = {}
        self._graph = {}
        self._cluster_doc_count = {}
        self._cluster_idf = {}
        self._genericity_penalty = {}
        self._total_docs = 0
        
        self._docs = {}
        self._next_id = 0
        self._cluster_to_docs = defaultdict(list)
    
    def learn(self, sentences):
        """Learn clustering, build directed graph, compute IDF."""
        from scbs.clustering import build_cooccurrence, cluster_words
        
        cooc = build_cooccurrence(sentences)
        self._word_clusters = cluster_words(cooc, self.n_clusters)
        self._graph, self._cluster_doc_count = build_directed_semantic_graph(
            sentences, self._word_clusters, self.n_clusters
        )
        self._total_docs = len(sentences)
        self._cluster_idf = compute_cluster_idf(
            self._cluster_doc_count, self._total_docs
        )
        self._genericity_penalty = compute_genericity_penalty(
            self._graph, self._cluster_doc_count, self._total_docs
        )
    
    def add(self, row, text):
        doc_id = self._next_id
        self._next_id += 1
        self._docs[doc_id] = {"id": doc_id, "row": row, "text": text}
        
        seen = set()
        for slot, cluster_id in row:
            if cluster_id not in seen:
                self._cluster_to_docs[cluster_id].append(doc_id)
                seen.add(cluster_id)
    
    def build(self):
        pass
    
    def search(self, query_row, query_text, *, top_k=10, excluded=None):
        if excluded is None:
            excluded = set()
        
        # IDF-weighted energy injection
        initial_energy = inject_energy_idf_weighted(query_row, self._cluster_idf)
        
        # Directed diffusion
        final_energy = diffuse_energy_directed(
            initial_energy, self._graph,
            iterations=self.diffusion_iterations,
            damping=self.damping
        )
        
        # Score documents — energy weighted by cluster specificity
        doc_scores = defaultdict(float)
        for cluster_id, energy in final_energy.items():
            if cluster_id not in self._cluster_to_docs:
                continue
            
            # Weight energy by IDF (specific concepts contribute more)
            # AND apply genericity penalty (generic concepts contribute less)
            idf = self._cluster_idf.get(cluster_id, 1.0)
            penalty = self._genericity_penalty.get(cluster_id, 1.0)
            weighted_energy = energy * idf * penalty
            
            for doc_id in self._cluster_to_docs[cluster_id]:
                if doc_id not in excluded:
                    doc_scores[doc_id] += weighted_energy
        
        # Rank
        scored = [
            {
                "id": doc_id,
                "text": self._docs[doc_id]["text"],
                "energy": round(score, 3),
            }
            for doc_id, score in doc_scores.items()
        ]
        scored.sort(key=lambda d: -d["energy"])
        results = scored[:top_k]
        
        for r in results:
            excluded.add(r["id"])
        
        stats = {
            "initial_concepts": len(initial_energy),
            "activated_concepts": len(final_energy),
            "candidates_scored": len(doc_scores),
        }
        
        return results, stats


if __name__ == "__main__":
    print("Approach 11 — Directed Flow with Genericity Penalty")
    print("Import DirectedFlowStore to use it.")
