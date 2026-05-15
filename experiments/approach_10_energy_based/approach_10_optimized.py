"""
Approach 10 Optimized — Generic Scoring Bottleneck Fix
========================================================

PROBLEM
-------
At 50K scale, scoring iterated through ALL docs in activated clusters.
With 14 clusters and ~3,500 docs each, that's costly.

GENERIC FIX
-----------
After diffusion, only score documents from concepts whose energy
exceeds the mean activated energy. No magic numbers.

WHY GENERIC
-----------
- Threshold derived from actual distribution (mean of energies)
- Self-adjusts to any corpus size (1K, 10K, 1M)
- Uses natural sparsity of diffusion
- If signal is sharp → few concepts above mean → fast
- If signal is diffuse → more concepts kept → correct behavior

NO CASE-BY-CASE TUNING
----------------------
- No hardcoded top-N per cluster
- No corpus-size-dependent thresholds  
- No domain-specific rules
- Works identically on any data
"""
import sys, os, time, random, math
from collections import defaultdict, Counter

_HERE = os.path.dirname(__file__)
_SRC  = os.path.join(_HERE, "..", "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from scbs.blueprint import BlueprintEncoder
from scbs.matrix_index import make_row
from scbs.distance import build_idf, sentence_weight
from scbs.clustering import build_cooccurrence, cluster_words, get_cluster_labels

# Reuse graph building and diffusion from original Approach 10
sys.path.insert(0, _HERE)
from test_approach_10 import (
    build_semantic_graph,
    inject_energy,
    diffuse_energy
)

random.seed(42)


# ═══════════════════════════════════════════════════════════════════
#  OPTIMIZED ENERGY STORE
# ═══════════════════════════════════════════════════════════════════

class EnergyStoreOptimized:
    """
    Same as EnergyStore but with generic energy-threshold filtering
    to fix the scoring bottleneck at scale.
    """
    
    def __init__(self, n_clusters=14, diffusion_iterations=3, damping=0.7):
        self.n_clusters = n_clusters
        self.diffusion_iterations = diffusion_iterations
        self.damping = damping
        
        self._idf = {}
        self._word_clusters = {}
        self._cluster_labels = {}
        self._graph = {}
        self._docs = {}
        self._next_id = 0
        self._cluster_to_docs = defaultdict(list)
    
    def learn(self, sentences):
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
        doc_id = self._next_id
        self._next_id += 1
        self._docs[doc_id] = {"id": doc_id, "row": row, "text": text}
        for slot, cluster_id in row:
            self._cluster_to_docs[cluster_id].append(doc_id)
    
    def build(self):
        pass
    
    def search(self, query_row, query_text, *, top_k=10, excluded=None):
        """
        Energy-based retrieval with generic threshold filtering.
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
        
        # ═══ GENERIC FIX ═══
        # Step 2.5: Filter to significantly activated concepts
        # Threshold = mean activated energy (data-driven, no magic numbers)
        if final_energy:
            energies = list(final_energy.values())
            mean_energy = sum(energies) / len(energies)
            # Only keep concepts with above-average activation
            active_concepts = {
                cid: e for cid, e in final_energy.items() 
                if e >= mean_energy
            }
        else:
            active_concepts = {}
        # ═══ END FIX ═══
        
        # Step 3: Score documents only from significantly activated concepts
        doc_scores = defaultdict(float)
        
        for cluster_id, energy in active_concepts.items():
            if cluster_id in self._cluster_to_docs:
                for doc_id in self._cluster_to_docs[cluster_id]:
                    if doc_id not in excluded:
                        doc_scores[doc_id] += energy
        
        # Step 4: Rank by total energy
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
            "above_threshold_concepts": len(active_concepts),
            "candidates": len(doc_scores),
        }
        
        return results, stats


if __name__ == "__main__":
    print("EnergyStoreOptimized — generic energy-threshold filtering")
    print("Import this module to use the optimized version.")
