"""
Approach 15 — Overlapping Sparse Dictionary + Late Interaction
================================================================

THE PIVOT
---------
A14 failed because NMF basis are near-orthogonal (mean affinity 0.035).
Late interaction needs overlap to work.

Replace NMF with sparse dictionary learning:

    NMF:           D^T D ≈ I       (orthogonal, no overlap)
    Dictionary:    D^T D ≠ I       (overlapping atoms)
    
The same X ≈ DH decomposition, but without the orthogonality constraint.
Overcomplete dictionaries naturally produce overlapping atoms — concepts
share vocabulary, just like in real language.

WHY THIS SHOULD FIX A14
-----------------------
- Off-diagonal basis affinity > 0 → late interaction has signal to work with
- "Best match" for each query basis is now meaningful (not just diagonal)
- Documents that share concept-overlap with query (but not same exact basis)
  can now be ranked correctly

PREDICTION (from theoretical analysis):
   NDCG@10 should land in 0.26–0.31 range
   vs A13's 0.236 and BM25's 0.351

HONEST CAVEAT
-------------
Dictionary learning is slower than NMF. May or may not produce semantically
meaningful overlap (vs random overlap). We measure to find out.
"""
import math
import numpy as np

try:
    from sklearn.decomposition import MiniBatchDictionaryLearning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as sk_normalize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class OverlappingDictionaryStore:
    """
    Retrieval using overlapping sparse dictionary + late interaction.
    
    Key difference from A14: dictionary atoms are NOT orthogonal,
    so basis-to-basis affinity matrix has meaningful off-diagonal entries.
    """
    
    def __init__(self, n_atoms=256, top_k_query=10, top_m_doc=15,
                 sparsity_alpha=0.1):
        if not SKLEARN_AVAILABLE:
            raise ImportError("Requires scikit-learn and numpy")
        
        self.n_atoms = n_atoms
        self.top_k_query = top_k_query
        self.top_m_doc = top_m_doc
        self.sparsity_alpha = sparsity_alpha
        
        self._vectorizer = None
        self._dict_learner = None
        self._D_norm = None       # (n_atoms, vocab) — L2-normalized
        self._basis_sim = None    # (n_atoms, n_atoms) — atom-to-atom affinity
        self._basis_idf = None
        self._doc_top_basis = None
        self._doc_top_values = None
        self._docs = []
        self._doc_texts = []
    
    def learn_and_build(self, sentences):
        self._doc_texts = list(sentences)
        n_docs = len(sentences)
        
        print(f"    Step 1: TF-IDF...")
        self._vectorizer = TfidfVectorizer(
            lowercase=True, max_features=10000,
            min_df=2, max_df=0.95, sublinear_tf=True,
        )
        X = self._vectorizer.fit_transform(sentences)
        X_dense = X.toarray().astype(np.float32)
        print(f"      Matrix: {X_dense.shape[0]} × {X_dense.shape[1]}")
        
        print(f"    Step 2: Sparse Dictionary Learning (n_atoms={self.n_atoms})...")
        print(f"      alpha={self.sparsity_alpha} (lower = denser codes, more overlap)")
        # MiniBatchDictionaryLearning is much faster than DictionaryLearning
        # Atoms are NOT constrained to be orthogonal → overlapping basis
        # positive_dict and positive_code ensure non-negative outputs (like NMF)
        self._dict_learner = MiniBatchDictionaryLearning(
            n_components=self.n_atoms,
            alpha=self.sparsity_alpha,
            max_iter=100,
            batch_size=64,
            random_state=42,
            fit_algorithm='cd',
            transform_algorithm='lasso_cd',
            transform_alpha=self.sparsity_alpha,
            positive_dict=True,
            positive_code=True,
        )
        H = self._dict_learner.fit_transform(X_dense)
        D = self._dict_learner.components_   # (n_atoms, vocab)
        print(f"      Done. H shape: {H.shape}")
        
        # With positive constraints, codes are already non-negative
        H_pos = H
        D_pos = D
        
        # Normalize atoms for cosine similarity
        print(f"    Step 3: Compute atom-to-atom affinity matrix...")
        self._D_norm = sk_normalize(D_pos, axis=1)
        self._basis_sim = self._D_norm @ self._D_norm.T   # (n_atoms, n_atoms)
        
        # Diagnostic: are atoms actually overlapping?
        off_diag = self._basis_sim - np.eye(self.n_atoms)
        print(f"      Mean off-diagonal affinity: {off_diag.mean():.3f}")
        print(f"      Max off-diagonal affinity:  {off_diag.max():.3f}")
        print(f"      Fraction > 0.1:             {(off_diag > 0.1).mean()*100:.1f}%")
        
        # Per-doc top-M sparse representation
        print(f"    Step 4: Extract top-{self.top_m_doc} atoms per doc...")
        self._doc_top_basis = np.zeros((n_docs, self.top_m_doc), dtype=np.int32)
        self._doc_top_values = np.zeros((n_docs, self.top_m_doc), dtype=np.float32)
        for i in range(n_docs):
            top_idx = np.argsort(-H_pos[i])[:self.top_m_doc]
            self._doc_top_basis[i] = top_idx
            self._doc_top_values[i] = H_pos[i, top_idx]
        
        # IDF per atom
        print(f"    Step 5: Compute atom-level IDF...")
        atom_active_count = np.zeros(self.n_atoms)
        for i in range(n_docs):
            for j in self._doc_top_basis[i]:
                atom_active_count[j] += 1
        self._basis_idf = np.log((n_docs + 1) / (atom_active_count + 1)) + 1
        print(f"      IDF range: [{self._basis_idf.min():.2f}, {self._basis_idf.max():.2f}]")
        print(f"      Atoms used in zero docs: {(atom_active_count == 0).sum()}")
    
    def add_doc_lookup(self, doc_ids):
        self._docs = list(doc_ids)
    
    def search(self, query, top_k=10):
        """Late interaction scoring on overlapping dictionary atoms."""
        q_tfidf = self._vectorizer.transform([query]).toarray().astype(np.float32)
        q_codes = self._dict_learner.transform(q_tfidf)[0]
        # With positive_code=True, codes are already non-negative
        
        # Top-K query atoms
        top_q_indices = np.argsort(-q_codes)[:self.top_k_query]
        top_q_values = q_codes[top_q_indices]
        
        mask = top_q_values > 0
        top_q_indices = top_q_indices[mask]
        top_q_values = top_q_values[mask]
        
        if len(top_q_indices) == 0:
            return []
        
        q_weights = top_q_values * self._basis_idf[top_q_indices]
        q_affinity = self._basis_sim[top_q_indices]   # (top_k, n_atoms)
        
        n_docs = self._doc_top_basis.shape[0]
        scores = np.zeros(n_docs)
        
        for qi_local in range(len(top_q_indices)):
            aff_row = q_affinity[qi_local]
            # For each doc, gather affinity at its top-m atoms: (n_docs, top_m)
            doc_affinities = aff_row[self._doc_top_basis]
            # Late interaction: d_j * sim(b_i, b_j)
            per_basis_score = self._doc_top_values * doc_affinities
            # Max over doc's atoms (MaxSim)
            best_match = per_basis_score.max(axis=1)
            scores += q_weights[qi_local] * best_match
        
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
