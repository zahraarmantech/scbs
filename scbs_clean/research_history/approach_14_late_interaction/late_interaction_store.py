"""
Approach 14 — Late Interaction Sparse Retrieval
=================================================

ARCHITECTURAL FIX
-----------------
Replace early compression:
    S(q, d) = ⟨h_q, h_d⟩       (single dot product, signal lost)

With late interaction:
    S(q, d) = Σ_{i ∈ TopK(q)} max_{j ∈ TopM(d)} q_i · d_j · ⟨b_i, b_j⟩

INTUITION
---------
- Query selects what evidence matters (top-k basis)
- For each piece of evidence, doc presents its best match
- Inner product ⟨b_i, b_j⟩ measures basis-to-basis semantic affinity
- No early pooling — comparison happens late, after sparsification

WHY THIS MIGHT CLOSE THE BM25 GAP
----------------------------------
On Cranfield:
- Rare technical terms decide relevance
- Early compression (h_q · h_d) averages out discriminating signal
- Late interaction preserves which-basis-matched-what
- Each query concern can find its best evidence in the doc

This is ColBERT's late interaction principle applied to our NMF basis.

HONEST CAVEAT
-------------
ColBERT uses learned token embeddings. NMF basis are reconstruction-
optimal, not retrieval-optimal. Late interaction on NMF may or may not
help — we measure to find out.
"""
import math
import numpy as np

try:
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as sk_normalize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class LateInteractionStore:
    """
    Late-interaction retrieval on sparse NMF basis.
    
    Scoring (per doc):
        score = Σ_{i ∈ TopK(q)} max_{j ∈ TopM(d)} q_i · d_j · sim(b_i, b_j)
    
    Where sim(b_i, b_j) = cosine similarity between NMF basis vectors.
    """
    
    def __init__(self, n_basis=256, top_k_query_basis=10,
                 top_m_doc_basis=15):
        if not SKLEARN_AVAILABLE:
            raise ImportError("Requires scikit-learn and numpy")
        
        self.n_basis = n_basis
        self.top_k_query_basis = top_k_query_basis
        self.top_m_doc_basis = top_m_doc_basis
        
        self._vectorizer = None
        self._nmf = None
        self._W_norm = None       # (n_basis, vocab) — L2-normalized
        self._basis_sim = None    # (n_basis, n_basis) — basis-to-basis affinity
        self._basis_idf = None
        # Doc representation: top-M indices and their activations
        self._doc_top_basis = None    # (n_docs, top_m) — indices
        self._doc_top_values = None   # (n_docs, top_m) — values
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
        print(f"      Matrix: {X.shape[0]} × {X.shape[1]}")
        
        print(f"    Step 2: NMF (n_basis={self.n_basis})...")
        self._nmf = NMF(
            n_components=self.n_basis, init='nndsvd',
            max_iter=200, tol=1e-4, random_state=42,
        )
        H = self._nmf.fit_transform(X)         # (n_docs, n_basis)
        W = self._nmf.components_              # (n_basis, vocab)
        print(f"      Reconstruction err: {self._nmf.reconstruction_err_:.2f}")
        
        # Normalize basis vectors for cosine similarity
        print(f"    Step 3: Compute basis-to-basis affinity matrix...")
        self._W_norm = sk_normalize(W, axis=1)  # row-normalized
        # Basis similarity = cosine between basis vectors in vocab space
        self._basis_sim = self._W_norm @ self._W_norm.T  # (n_basis, n_basis)
        
        # Per-doc top-M sparse representation
        print(f"    Step 4: Extract top-{self.top_m_doc_basis} basis per doc...")
        self._doc_top_basis = np.zeros((n_docs, self.top_m_doc_basis), dtype=np.int32)
        self._doc_top_values = np.zeros((n_docs, self.top_m_doc_basis), dtype=np.float32)
        for i in range(n_docs):
            top_idx = np.argsort(-H[i])[:self.top_m_doc_basis]
            self._doc_top_basis[i] = top_idx
            self._doc_top_values[i] = H[i, top_idx]
        
        # Basis IDF (for query weighting)
        print(f"    Step 5: Compute basis IDF...")
        basis_active_count = np.zeros(self.n_basis)
        for i in range(n_docs):
            for j in self._doc_top_basis[i]:
                basis_active_count[j] += 1
        self._basis_idf = np.log((n_docs + 1) / (basis_active_count + 1)) + 1
        
        print(f"    Basis stats:")
        print(f"      Affinity matrix: shape={self._basis_sim.shape}")
        print(f"      Mean off-diagonal affinity: {(self._basis_sim - np.eye(self.n_basis)).mean():.3f}")
        print(f"      IDF range: [{self._basis_idf.min():.2f}, {self._basis_idf.max():.2f}]")
    
    def add_doc_lookup(self, doc_ids):
        self._docs = list(doc_ids)
    
    def search(self, query, top_k=10):
        """
        Late interaction scoring.
        
        For each query basis i (top-k):
            for each doc:
                find best matching doc basis j (in doc's top-m):
                    score_i = q_i * d_j * sim(b_i, b_j) * idf(b_i)
            sum over query basis
        """
        # Project query
        q_tfidf = self._vectorizer.transform([query])
        q_basis = self._nmf.transform(q_tfidf)[0]
        
        # Top-K query basis
        top_q_indices = np.argsort(-q_basis)[:self.top_k_query_basis]
        top_q_values = q_basis[top_q_indices]
        
        # Filter to non-zero query basis
        mask = top_q_values > 0
        top_q_indices = top_q_indices[mask]
        top_q_values = top_q_values[mask]
        
        if len(top_q_indices) == 0:
            return []
        
        # IDF-weight the query basis
        q_weights = top_q_values * self._basis_idf[top_q_indices]
        
        # For each query basis, get its affinity with all basis (n_basis,)
        q_affinity = self._basis_sim[top_q_indices]  # (top_k, n_basis)
        
        # For each doc, score = sum over query basis of max over doc basis
        n_docs = len(self._docs) if self._docs else self._doc_top_basis.shape[0]
        scores = np.zeros(n_docs)
        
        # Vectorized late interaction
        # doc_top_basis: (n_docs, top_m) — basis indices
        # doc_top_values: (n_docs, top_m) — basis activations
        # q_affinity[i, j] = similarity between query basis i and any basis j
        
        for qi_local, qi_global in enumerate(top_q_indices):
            # affinity from this query basis to every basis: (n_basis,)
            aff_row = q_affinity[qi_local]
            
            # For each doc, look up affinity at doc's top-m basis indices
            # doc_top_basis is (n_docs, top_m) → indexing gives (n_docs, top_m)
            doc_affinities = aff_row[self._doc_top_basis]  # (n_docs, top_m)
            
            # Per-doc interaction: d_j * sim(b_i, b_j) for each of top-m j
            per_basis_score = self._doc_top_values * doc_affinities  # (n_docs, top_m)
            
            # Max over doc's top-m basis (the "max" in MaxSim)
            best_match = per_basis_score.max(axis=1)  # (n_docs,)
            
            # Weighted contribution to score
            scores += q_weights[qi_local] * best_match
        
        # Rank
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for i in top_idx:
            if scores[i] > 0:
                results.append({
                    "id": self._docs[i] if self._docs else int(i),
                    "text": self._doc_texts[i],
                    "score": float(scores[i]),
                })
        return results
