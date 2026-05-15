"""
BM25 baseline implementation.

Included for benchmark comparison. The retriever module is the main
contribution; BM25 is here so users can reproduce the comparison
without external dependencies.

Reference: Robertson & Zaragoza (2009), "The Probabilistic Relevance
Framework: BM25 and Beyond"
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence


@dataclass
class BM25Result:
    """A single BM25 search result."""
    doc_id: str
    text: str
    score: float


class BM25:
    """
    Standard BM25 retrieval.

    Parameters
    ----------
    k1 : float, default=1.5
        Term frequency saturation parameter.
    b : float, default=0.75
        Document length normalization parameter.

    Examples
    --------
    >>> bm25 = BM25()
    >>> bm25.fit(corpus_texts, doc_ids)
    >>> results = bm25.search("my query", top_k=10)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_freqs: dict = {}
        self._doc_lens: dict = {}
        self._doc_terms: dict = {}
        self._doc_texts: dict = {}
        self._avg_dl: float = 0.0
        self._n_docs: int = 0
        self._idf: dict = {}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [w.lower().strip(".,!?;:\"'()") for w in text.split() if w.strip()]

    def fit(
        self,
        documents: Sequence[str],
        doc_ids: Sequence[str] | None = None,
    ) -> "BM25":
        """Build BM25 index from corpus."""
        documents = list(documents)
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(documents))]
        else:
            doc_ids = [str(d) for d in doc_ids]

        self._n_docs = len(documents)
        self._doc_freqs = defaultdict(int)
        self._doc_terms = {}
        self._doc_lens = {}
        self._doc_texts = {}
        total_len = 0

        for doc_id, text in zip(doc_ids, documents):
            terms = self._tokenize(text)
            term_counts = Counter(terms)
            self._doc_terms[doc_id] = term_counts
            self._doc_lens[doc_id] = len(terms)
            self._doc_texts[doc_id] = text
            total_len += len(terms)
            for term in set(terms):
                self._doc_freqs[term] += 1

        self._avg_dl = total_len / self._n_docs if self._n_docs > 0 else 0.0

        for term, df in self._doc_freqs.items():
            self._idf[term] = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1)
        return self

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        """Retrieve top-K documents for a query."""
        query_terms = self._tokenize(query)
        scores: dict = defaultdict(float)

        for term in query_terms:
            if term not in self._idf:
                continue
            idf = self._idf[term]
            for doc_id, term_counts in self._doc_terms.items():
                tf = term_counts.get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_lens[doc_id]
                norm = (
                    tf
                    * (self.k1 + 1)
                    / (tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl))
                )
                scores[doc_id] += idf * norm

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [
            BM25Result(doc_id=doc_id, text=self._doc_texts[doc_id], score=score)
            for doc_id, score in ranked
        ]
