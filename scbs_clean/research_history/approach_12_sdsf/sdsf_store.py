"""
Approach 12 — Sparse Directed Semantic Flow (SDSF)
====================================================

ARCHITECTURAL PIVOT
-------------------
Replace document clustering with sparse semantic basis decomposition.

Old:    X → {c_1, ..., c_k}        (partition: doc in 1-of-k clusters)
New:    X ≈ W·H                    (decomposition: doc as sparse mixture)

Where:
- W: basis matrix (n_basis x vocab)  — semantic basis vectors
- H: activation matrix (n_docs x n_basis) — sparse activations per doc

WHY THIS SHOULD HELP
--------------------
Previous problem on Cranfield:
- 14 clusters each contained 25-97% of docs (cluster collapse)
- No granularity to discriminate
- Genericity penalty couldn't help because everything was generic

With sparse basis (128-512 functions):
- Each doc activates only top-k basis (sparse)
- Basis functions overlap (compositional)
- Specific basis = activated by few docs (clean IDF signal)
- Generic basis = activated by many docs (heavy penalty)

PIPELINE
--------
1. TF-IDF sparse matrix (real values, not co-occurrence counts)
2. NMF decomposition into K basis functions
3. Build directed graph on basis functions (not docs/clusters)
4. Query → TF-IDF vector → project to basis space
5. Inject energy at top basis components, weighted by basis IDF
6. Diffuse through basis graph
7. Rank docs by their basis activation alignment

HONEST EXPECTATIONS
-------------------
NMF retrieval has been studied since the 2000s and typically doesn't
beat BM25 by large margins. What's novel here is combining sparse
decomposition with directed flow + basis-level genericity penalty.
Whether that combination beats prior NMF retrievers is what we measure.
"""
import math
import re
from collections import defaultdict, Counter
import numpy as np

try:
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
#  SPARSE DIRECTED SEMANTIC FLOW STORE
# ═══════════════════════════════════════════════════════════════════

class SDSFStore:
    """
    Retrieval via sparse semantic basis + directed flow.
    
    Index time:
      1. Build TF-IDF matrix of corpus
      2. NMF decomposition → basis functions + document activations
      3. Keep only top-K basis per document (sparsity)
      4. Build directed graph between basis functions
      5. Compute basis IDF and genericity
    
    Query time:
      1. Transform query to TF-IDF vector
      2. Project to basis space (sparse)
      3. Inject energy weighted by basis IDF
      4. Diffuse through basis graph (directed)
      5. Score docs by basis-activation alignment + genericity penalty
    """
    
    def __init__(self, n_basis=128, top_k_basis_per_doc=10,
                 diffusion_iterations=2, damping=0.6):
        if not SKLEARN_AVAILABLE:
            raise ImportError("SDSFStore requires scikit-learn and numpy")
        
        self.n_basis = n_basis
        self.top_k_basis_per_doc = top_k_basis_per_doc
        self.diffusion_iterations = diffusion_iterations
        self.damping = damping
        
        self._vectorizer = None
        self._nmf = None
        self._W = None          # basis matrix (n_basis x vocab)
        self._H_sparse = None   # doc activations (n_docs x n_basis), sparsified
        self._basis_idf = None  # IDF per basis function
        self._basis_genericity = None
        self._basis_graph = None  # directed: basis_i -> basis_j -> weight
        
        self._docs = []
        self._doc_texts = []
    
    def learn_and_build(self, sentences):
        """
        Single pass: build TF-IDF, do NMF, build basis graph.
        (Combined because NMF is the heaviest step.)
        """
        self._doc_texts = list(sentences)
        n_docs = len(sentences)
        
        # Step 1: TF-IDF
        print(f"    Step 1: TF-IDF vectorization...")
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=10000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            stop_words=None,
        )
        X = self._vectorizer.fit_transform(sentences)
        print(f"      Matrix: {X.shape[0]} docs × {X.shape[1]} terms")
        
        # Step 2: NMF decomposition
        print(f"    Step 2: NMF decomposition into {self.n_basis} basis functions...")
        self._nmf = NMF(
            n_components=self.n_basis,
            init='nndsvd',
            max_iter=200,
            tol=1e-4,
            random_state=42,
        )
        H = self._nmf.fit_transform(X)          # (n_docs, n_basis)
        self._W = self._nmf.components_         # (n_basis, vocab)
        print(f"      Reconstruction error: {self._nmf.reconstruction_err_:.2f}")
        
        # Step 3: Sparsify - keep top-K basis per document
        print(f"    Step 3: Sparsify to top-{self.top_k_basis_per_doc} basis per doc...")
        H_sparse = np.zeros_like(H)
        for i in range(H.shape[0]):
            top_indices = np.argsort(-H[i])[:self.top_k_basis_per_doc]
            H_sparse[i, top_indices] = H[i, top_indices]
        self._H_sparse = H_sparse
        
        # Step 4: Compute basis IDF (basis-level specificity)
        print(f"    Step 4: Compute basis-level IDF...")
        # Document frequency per basis: how many docs activate this basis
        basis_df = (H_sparse > 0).sum(axis=0)  # (n_basis,)
        self._basis_idf = np.log((n_docs + 1) / (basis_df + 1)) + 1
        
        # Genericity penalty: basis appearing in >50% docs gets penalized
        basis_freq_ratio = basis_df / n_docs
        self._basis_genericity = np.where(
            basis_freq_ratio < 0.05, 1.0,
            np.where(
                basis_freq_ratio > 0.5, 0.1,
                1.0 - (basis_freq_ratio - 0.05) / 0.45 * 0.9
            )
        )
        
        # Step 5: Build directed graph between basis functions
        print(f"    Step 5: Build directed basis graph...")
        # Co-activation matrix: how often basis i and j fire together
        # Use sparse activations
        H_bool = (H_sparse > 0).astype(np.float32)
        co_activation = H_bool.T @ H_bool  # (n_basis, n_basis)
        np.fill_diagonal(co_activation, 0)
        
        # Directed normalization: W[i][j] = P(j fires | i fires)
        row_sums = co_activation.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self._basis_graph = co_activation / row_sums
        
        print(f"    Done. Active basis stats:")
        print(f"      Min docs/basis:  {basis_df.min()}")
        print(f"      Max docs/basis:  {basis_df.max()}")
        print(f"      Median:          {int(np.median(basis_df))}")
        print(f"      Basis with df>50%: {(basis_freq_ratio > 0.5).sum()}/{self.n_basis}")
        print(f"      Basis with df<5%:  {(basis_freq_ratio < 0.05).sum()}/{self.n_basis}")
    
    def add_doc_lookup(self, doc_ids):
        """Store doc IDs in same order as sentences passed to learn_and_build."""
        self._docs = list(doc_ids)
    
    def search(self, query, top_k=10):
        """
        Energy-flow retrieval on basis graph.
        """
        # Step 1: Project query to basis space
        q_tfidf = self._vectorizer.transform([query])
        q_basis = self._nmf.transform(q_tfidf)[0]   # (n_basis,)
        
        # Step 2: Sparsify query (top-K basis)
        top_q = np.argsort(-q_basis)[:self.top_k_basis_per_doc]
        q_sparse = np.zeros_like(q_basis)
        q_sparse[top_q] = q_basis[top_q]
        
        # Step 3: IDF-weighted initial energy
        initial_energy = q_sparse * self._basis_idf
        
        # Step 4: Diffuse through directed basis graph
        current = initial_energy.copy()
        for _ in range(self.diffusion_iterations):
            # Outflow to neighbors + self-retention
            next_energy = current @ self._basis_graph * self.damping
            next_energy += current * (1 - self.damping) * 0.5
            current = next_energy
        
        final_energy = current
        
        # Step 5: Score docs by basis activation alignment
        # weighted by basis IDF and genericity
        basis_score_weight = final_energy * self._basis_idf * self._basis_genericity
        
        # doc_scores[d] = sum over basis of H[d][b] * basis_score_weight[b]
        doc_scores = self._H_sparse @ basis_score_weight  # (n_docs,)
        
        # Rank
        top_idx = np.argsort(-doc_scores)[:top_k]
        results = []
        for i in top_idx:
            if doc_scores[i] > 0:
                results.append({
                    "id": self._docs[i] if self._docs else i,
                    "text": self._doc_texts[i],
                    "score": float(doc_scores[i]),
                })
        
        return results
