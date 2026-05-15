"""
High-level Store interface for SCBS.

Provides a clean, simple API over the full encoding +
indexing + search pipeline. Most users only need this module.
"""

from typing import Optional
import time

from .blueprint    import BlueprintEncoder, SLOT_NAMES
from .matrix_index import make_row, MatrixStore, matrix_distance
from .distance     import (
    TFIDFClusterStore,
    weighted_matrix_distance,
)


class Encoder:
    """
    Text-to-blueprint encoder.

    Encodes natural language sentences into sparse
    semantic blueprints suitable for indexing and search.

    Example:
        >>> encoder = Encoder()
        >>> row = encoder.encode("kafka deployment failed")
        >>> print(row)
        [(1, 4297), (2, 3401), (4, 3804), ...]
    """

    def __init__(self, registry_path: str = "unknown_words.json"):
        """
        Initialise encoder.

        Args:
            registry_path: Path for persisting unknown words
                            discovered during encoding.
        """
        self._enc = BlueprintEncoder(registry_path)

    def encode(self, text: str) -> list:
        """
        Encode text into a matrix row.

        Args:
            text: Natural language sentence to encode.

        Returns:
            List of (slot, value) tuples sorted by slot.
            Each tuple represents one filled semantic slot.
        """
        _, blueprint, _ = self._enc.encode(text)
        return make_row(blueprint)

    def encode_detailed(self, text: str) -> dict:
        """
        Encode text and return detailed encoding information.

        Args:
            text: Natural language sentence.

        Returns:
            dict with:
                row:       matrix row (list of tuples)
                token_ids: raw token ID sequence
                blueprint: sparse slot dict
                trace:     step-by-step encoding trace
        """
        token_ids, blueprint, trace = self._enc.encode(text)
        return {
            "row":       make_row(blueprint),
            "token_ids": token_ids,
            "blueprint": blueprint,
            "trace":     trace,
        }


class Store:
    """
    Indexed semantic store with fast similarity search.

    Combines TF-IDF weighting, co-occurrence clustering,
    and hybrid index for sub-millisecond search at scale.

    Example:
        >>> store = Store()
        >>> encoder = Encoder()
        >>>
        >>> # Build index
        >>> for text in corpus:
        ...     row = encoder.encode(text)
        ...     store.add(row, text)
        >>> store.build(corpus_texts=corpus)
        >>>
        >>> # Search
        >>> query_row = encoder.encode("kafka error")
        >>> results, stats = store.search(query_row, "kafka error")
    """

    def __init__(self, n_clusters: int = 14):
        """
        Initialise store.

        Args:
            n_clusters: Number of co-occurrence clusters to learn
                         from corpus during build().
        """
        self._store     = TFIDFClusterStore(n_clusters=n_clusters)
        self._pending   = []     # records added before build()
        self._built     = False

    def add(self, row: list, text: str) -> None:
        """
        Add a record to the store.

        Must be called before build(). After build(), use
        the index for search.

        Args:
            row:  Encoded matrix row from Encoder.encode().
            text: Original sentence text.
        """
        self._pending.append((row, text))

    def build(self, corpus_texts: Optional[list] = None) -> None:
        """
        Build the search index. Call once after all add() calls.

        Args:
            corpus_texts: Optional list of all sentences for
                           IDF/cluster learning. If not provided,
                           uses texts from add() calls.
        """
        texts_for_learning = corpus_texts or [t for _, t in self._pending]
        self._store.learn(texts_for_learning)

        for row, text in self._pending:
            self._store.add(row, text)
        self._store.build()
        self._pending  = []
        self._built    = True

    def search(
        self,
        query_row:    list,
        query_text:   str,
        top_k:        int            = 10,
        threshold:    float          = 100.0,
        zone_radius:  int            = 150,
        excluded:     Optional[set]  = None,
    ) -> tuple:
        """
        Search for similar records.

        Args:
            query_row:    Encoded matrix row of the query.
            query_text:   Original query text (for IDF/cluster).
            top_k:        Maximum number of results to return.
            threshold:    Maximum distance for a match (lower = stricter).
            zone_radius:  Index zone search radius.
            excluded:     Set of record IDs to skip
                           (for puzzle-piece exclusion across calls).

        Returns:
            (results, stats) tuple where:
                results: list of dicts with id, text, distance, cluster
                stats:   dict with search statistics
        """
        if not self._built:
            raise RuntimeError(
                "Store not built. Call store.build() before search()."
            )
        return self._store.search(
            query_row, query_text,
            top_k=top_k,
            threshold=threshold,
            zone_radius=zone_radius,
            excluded=excluded,
        )

    def linear_search(
        self,
        query_row:  list,
        query_text: str,
        threshold:  float = 100.0,
    ) -> list:
        """
        Exhaustive linear search — for accuracy ground truth.

        Slower than search() but checks every record.

        Args:
            query_row:  Encoded matrix row.
            query_text: Original text.
            threshold:  Maximum distance threshold.

        Returns:
            list of result dicts.
        """
        if not self._built:
            raise RuntimeError("Store not built.")
        return self._store.linear_search(
            query_row, query_text, threshold=threshold,
        )

    def __len__(self) -> int:
        if self._built:
            return len(self._store)
        return len(self._pending)
