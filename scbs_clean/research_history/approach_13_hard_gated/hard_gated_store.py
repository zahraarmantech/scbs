"""
Approach 13 — Hard-Gated Sparse Scoring
========================================

THREE TARGETED CHANGES FROM APPROACH 12
----------------------------------------

1. TOP-K QUERY TRUNCATION
   Only top-k strongest query basis contribute.
   "Strongest evidence decides ranking" — not smooth averaging.
   
   S(q,d) = Σ_{i ∈ TopK(q)} q_i · d_i

2. DOCUMENT MAX-GATING
   Only document activations above the document's own mean count.
   Weak document signals get crushed.
   
   S(q,d) = Σ q_i · max(d_i - μ_d, 0)

3. BASIS-LEVEL IDF
   Rare basis amplified, common basis suppressed.
   
   S(q,d) = Σ q_i · d_i · log(N / df_i)

4. DIFFUSION OFF IN RANKING
   Diffusion was smoothing the signal. Remove it from the scoring path.
   Scoring is now: project query, gate, weight by IDF, dot product.

WHY THIS SHOULD HELP
--------------------
Approach 12 averaged signals across all activated basis. Most basis
contribute weak signal that drowns the strong evidence.

ColBERT, BM25, and other strong retrievers all use some form of
"strongest evidence wins" — saturation, max-pooling, or top-k.
This approach borrows that principle for our sparse basis framework.
"""
import math
from collections import defaultdict
import numpy as np

try:
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class HardGatedStore:
    """
    Sparse basis retrieval with hard gating.
    
    No diffusion. No smoothing. Top-k query, max-gated docs, IDF-weighted.
    """
    
    def __init__(self, n_basis=256, top_k_query_basis=5,
                 top_k_basis_per_doc=10, use_doc_gating=True):
        if not SKLEARN_AVAILABLE:
            raise ImportError("Requires scikit-learn and numpy")
        
        self.n_basis = n_basis
        self.top_k_query_basis = top_k_query_basis
        self.top_k_basis_per_doc = top_k_basis_per_doc
        self.use_doc_gating = use_doc_gating
        
        self._vectorizer = None
        self._nmf = None
        self._H_sparse = None      # (n_docs, n_basis)
        self._H_gated = None       # (n_docs, n_basis) — max-gated
        self._basis_idf = None     # (n_basis,)
        self._docs = []
        self._doc_texts = []
    
    def learn_and_build(self, sentences):
        self._doc_texts = list(sentences)
        n_docs = len(sentences)
        
        print(f"    Step 1: TF-IDF vectorization...")
        self._vectorizer = TfidfVectorizer(
            lowercase=True, max_features=10000,
            min_df=2, max_df=0.95, sublinear_tf=True,
        )
        X = self._vectorizer.fit_transform(sentences)
        print(f"      Matrix: {X.shape[0]} docs × {X.shape[1]} terms")
        
        print(f"    Step 2: NMF decomposition (n_basis={self.n_basis})...")
        self._nmf = NMF(
            n_components=self.n_basis, init='nndsvd',
            max_iter=200, tol=1e-4, random_state=42,
        )
        H = self._nmf.fit_transform(X)
        print(f"      Reconstruction err: {self._nmf.reconstruction_err_:.2f}")
        
        # Sparsify: top-k basis per document
        print(f"    Step 3: Sparsify docs to top-{self.top_k_basis_per_doc} basis...")
        H_sparse = np.zeros_like(H)
        for i in range(H.shape[0]):
            top_indices = np.argsort(-H[i])[:self.top_k_basis_per_doc]
            H_sparse[i, top_indices] = H[i, top_indices]
        self._H_sparse = H_sparse
        
        # CHANGE 2: Document max-gating
        # Only activations above the document's own mean
        print(f"    Step 4: Apply document max-gating...")
        if self.use_doc_gating:
            doc_means = H_sparse.mean(axis=1, keepdims=True)
            self._H_gated = np.maximum(H_sparse - doc_means, 0)
        else:
            self._H_gated = H_sparse
        
        # CHANGE 3: Basis IDF
        print(f"    Step 5: Compute basis IDF...")
        basis_df = (H_sparse > 0).sum(axis=0)
        # Use the gated matrix for the actual scoring weights
        self._basis_idf = np.log((n_docs + 1) / (basis_df + 1)) + 1
        
        print(f"    Basis stats after gating:")
        gated_active = (self._H_gated > 0).sum(axis=0)
        print(f"      Basis active in (gated) docs - min/median/max: "
              f"{gated_active.min()}/{int(np.median(gated_active))}/{gated_active.max()}")
        print(f"      Sparsity: {(self._H_gated == 0).mean()*100:.1f}% zeros")
    
    def add_doc_lookup(self, doc_ids):
        self._docs = list(doc_ids)
    
    def search(self, query, top_k=10):
        # Project query into basis space
        q_tfidf = self._vectorizer.transform([query])
        q_basis = self._nmf.transform(q_tfidf)[0]
        
        # CHANGE 1: Top-k query truncation
        # Only strongest query basis contribute
        top_q_indices = np.argsort(-q_basis)[:self.top_k_query_basis]
        q_truncated = np.zeros_like(q_basis)
        q_truncated[top_q_indices] = q_basis[top_q_indices]
        
        # IDF-weight the truncated query
        q_weighted = q_truncated * self._basis_idf
        
        # NO DIFFUSION in ranking — direct dot product
        # Score: each doc scored by its gated activations · weighted query
        doc_scores = self._H_gated @ q_weighted  # (n_docs,)
        
        # Rank
        top_idx = np.argsort(-doc_scores)[:top_k]
        results = []
        for i in top_idx:
            if doc_scores[i] > 0:
                results.append({
                    "id": self._docs[i] if self._docs else int(i),
                    "text": self._doc_texts[i],
                    "score": float(doc_scores[i]),
                })
        
        return results
