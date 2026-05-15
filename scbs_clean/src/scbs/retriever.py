"""
SCBS Retriever — Sparse Compositional Basis Search

A retrieval system using sparse overcomplete dictionary learning
with late interaction (MaxSim) scoring.

Architecture:
    Index:  TF-IDF → Sparse Dictionary → atom affinity matrix
    Query:  Project → Top-K gate → Late interaction → MaxSim ranking
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

try:
    from sklearn.decomposition import MiniBatchDictionaryLearning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as _l2_normalize
except ImportError as e:
    raise ImportError(
        "scbs requires scikit-learn. Install with: pip install scikit-learn"
    ) from e


@dataclass
class SearchResult:
    """A single search result."""
    doc_id: str
    text: str
    score: float


class Retriever:
    """
    Sparse compositional retrieval with late interaction.

    The retriever decomposes documents into a sparse overcomplete
    dictionary of semantic atoms, then ranks query-document pairs
    via late interaction (MaxSim) over their atom representations.

    Parameters
    ----------
    n_atoms : int, default=256
        Number of dictionary atoms. Should be overcomplete relative
        to the intrinsic dimensionality of the corpus.
    top_k_query : int, default=10
        Number of strongest query atoms used during scoring. Acts as
        a hard gate — weaker atoms are discarded.
    top_m_doc : int, default=15
        Number of strongest atoms stored per document.
    sparsity_alpha : float, default=0.1
        L1 regularization strength for sparse coding. Higher values
        produce sparser codes; lower values allow more atom overlap.
    max_features : int, default=10000
        Maximum vocabulary size for TF-IDF.
    min_df : int, default=2
        Minimum document frequency for vocabulary terms.
    max_df : float, default=0.95
        Maximum document frequency (filters very common terms).
    random_state : int, default=42
        Seed for reproducible dictionary learning.

    Examples
    --------
    >>> retriever = Retriever(n_atoms=256)
    >>> retriever.fit(corpus_texts, doc_ids)
    >>> results = retriever.search("my query", top_k=10)
    >>> for r in results:
    ...     print(r.doc_id, r.score, r.text[:80])
    """

    def __init__(
        self,
        n_atoms: int = 256,
        top_k_query: int = 10,
        top_m_doc: int = 15,
        sparsity_alpha: float = 0.1,
        max_features: int = 10000,
        min_df: int = 2,
        max_df: float = 0.95,
        random_state: int = 42,
    ):
        self.n_atoms = n_atoms
        self.top_k_query = top_k_query
        self.top_m_doc = top_m_doc
        self.sparsity_alpha = sparsity_alpha
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state

        # Set after fit
        self._vectorizer: TfidfVectorizer | None = None
        self._dict_learner: MiniBatchDictionaryLearning | None = None
        self._atom_affinity: np.ndarray | None = None
        self._atom_idf: np.ndarray | None = None
        self._doc_top_atoms: np.ndarray | None = None
        self._doc_top_values: np.ndarray | None = None
        self._doc_ids: list[str] = []
        self._doc_texts: list[str] = []

    def fit(
        self,
        documents: Sequence[str],
        doc_ids: Sequence[str] | None = None,
        verbose: bool = False,
    ) -> "Retriever":
        """
        Build the index from a corpus.

        Parameters
        ----------
        documents : sequence of str
            Document texts.
        doc_ids : sequence of str, optional
            Document identifiers. If None, integer indices are used.
        verbose : bool, default=False
            Print progress information.

        Returns
        -------
        self : Retriever
        """
        import warnings
        from sklearn.exceptions import ConvergenceWarning

        documents = list(documents)
        n_docs = len(documents)
        if doc_ids is None:
            doc_ids = [str(i) for i in range(n_docs)]
        else:
            doc_ids = [str(d) for d in doc_ids]
        if len(doc_ids) != n_docs:
            raise ValueError("documents and doc_ids must have the same length")

        self._doc_texts = documents
        self._doc_ids = doc_ids

        # Suppress lasso convergence warnings (low-level numerical noise,
        # does not affect retrieval quality at the tolerances used).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._fit_internal(documents, n_docs, verbose)

        return self

    def _fit_internal(self, documents, n_docs, verbose):
        """Internal fit logic — wrapped by fit() with warning suppression."""
        # 1. TF-IDF vectorization
        if verbose:
            print(f"  [1/5] TF-IDF vectorization ({n_docs} documents)...")
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=True,
        )
        X = self._vectorizer.fit_transform(documents)
        X_dense = X.toarray().astype(np.float32)
        if verbose:
            print(f"        Matrix: {X_dense.shape[0]} × {X_dense.shape[1]}")

        # 2. Sparse dictionary learning (overcomplete, overlapping atoms)
        if verbose:
            print(f"  [2/5] Sparse dictionary learning (n_atoms={self.n_atoms})...")
        self._dict_learner = MiniBatchDictionaryLearning(
            n_components=self.n_atoms,
            alpha=self.sparsity_alpha,
            max_iter=100,
            batch_size=64,
            random_state=self.random_state,
            fit_algorithm="cd",
            transform_algorithm="lasso_cd",
            transform_alpha=self.sparsity_alpha,
            positive_dict=True,
            positive_code=True,
        )
        H = self._dict_learner.fit_transform(X_dense)
        D = self._dict_learner.components_

        # 3. Atom-to-atom affinity (cosine similarity)
        if verbose:
            print(f"  [3/5] Building atom affinity matrix...")
        D_norm = _l2_normalize(D, axis=1)
        self._atom_affinity = D_norm @ D_norm.T

        # 4. Per-document sparse representation
        if verbose:
            print(f"  [4/5] Extracting top-{self.top_m_doc} atoms per document...")
        self._doc_top_atoms = np.zeros((n_docs, self.top_m_doc), dtype=np.int32)
        self._doc_top_values = np.zeros((n_docs, self.top_m_doc), dtype=np.float32)
        for i in range(n_docs):
            top_idx = np.argsort(-H[i])[: self.top_m_doc]
            self._doc_top_atoms[i] = top_idx
            self._doc_top_values[i] = H[i, top_idx]

        # 5. Per-atom IDF
        if verbose:
            print(f"  [5/5] Computing atom-level IDF...")
        atom_doc_count = np.zeros(self.n_atoms)
        for i in range(n_docs):
            for j in self._doc_top_atoms[i]:
                atom_doc_count[j] += 1
        self._atom_idf = np.log((n_docs + 1) / (atom_doc_count + 1)) + 1

        if verbose:
            print(f"  Index built. {n_docs} documents indexed.")
        return self

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Retrieve top-K documents for a query.

        Parameters
        ----------
        query : str
            Query text.
        top_k : int, default=10
            Number of results to return.

        Returns
        -------
        list of SearchResult
        """
        import warnings
        from sklearn.exceptions import ConvergenceWarning

        if self._dict_learner is None:
            raise RuntimeError("Retriever must be fit() before search()")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return self._search_internal(query, top_k)

    def _search_internal(self, query, top_k):
        """Internal search logic — wrapped by search() with warning suppression."""
        # Project query into atom space
        q_tfidf = self._vectorizer.transform([query]).toarray().astype(np.float32)
        q_codes = self._dict_learner.transform(q_tfidf)[0]

        # Top-K query atoms (hard gate)
        top_q_indices = np.argsort(-q_codes)[: self.top_k_query]
        top_q_values = q_codes[top_q_indices]

        mask = top_q_values > 0
        top_q_indices = top_q_indices[mask]
        top_q_values = top_q_values[mask]
        if len(top_q_indices) == 0:
            return []

        # IDF-weighted query atom contributions
        q_weights = top_q_values * self._atom_idf[top_q_indices]
        q_affinity = self._atom_affinity[top_q_indices]

        # Late interaction (MaxSim): for each query atom, find best matching doc atom
        n_docs = self._doc_top_atoms.shape[0]
        scores = np.zeros(n_docs)

        for qi_local in range(len(top_q_indices)):
            aff_row = q_affinity[qi_local]
            doc_affinities = aff_row[self._doc_top_atoms]
            per_atom_score = self._doc_top_values * doc_affinities
            best_match = per_atom_score.max(axis=1)
            scores += q_weights[qi_local] * best_match

        # Rank
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for i in top_idx:
            if scores[i] > 0:
                results.append(
                    SearchResult(
                        doc_id=self._doc_ids[i],
                        text=self._doc_texts[i],
                        score=float(scores[i]),
                    )
                )
        return results

    def get_atom_stats(self) -> dict:
        """Return diagnostic statistics about the learned atoms."""
        if self._atom_affinity is None:
            raise RuntimeError("Retriever must be fit() first")
        off_diag = self._atom_affinity - np.eye(self.n_atoms)
        return {
            "n_atoms": self.n_atoms,
            "mean_off_diagonal_affinity": float(off_diag.mean()),
            "max_off_diagonal_affinity": float(off_diag.max()),
            "fraction_affinity_above_0.1": float((off_diag > 0.1).mean()),
            "idf_range": (float(self._atom_idf.min()), float(self._atom_idf.max())),
        }
