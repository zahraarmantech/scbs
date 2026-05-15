"""
SCBS — Semantic Chunk Buffer System
====================================

A deterministic, zero-dependency semantic encoding system
for fast classification, routing, and pre-filtering.

Quick start
-----------
>>> from scbs import Encoder, Store
>>>
>>> encoder = Encoder()
>>> store   = Store()
>>>
>>> for text in corpus:
...     row = encoder.encode(text)
...     store.add(row, text)
>>> store.build()
>>>
>>> query_row = encoder.encode("kafka deployment failed")
>>> results, _ = store.search(query_row, "kafka deployment failed")

Public API
----------
Encoder         — Text to semantic blueprint encoder
Store           — High-level indexed store with search
matrix_distance — Compare two encoded rows directly
SLOT_NAMES      — Map of slot index to human-readable name

For lower-level access:
    scbs.encoder       — raw text encoding
    scbs.blueprint     — sparse slot extraction
    scbs.matrix_index  — hybrid signature + zone index
    scbs.distance      — TF-IDF weighted distance
"""

from .store        import Store, Encoder
from .blueprint    import SLOT_NAMES
from .matrix_index import matrix_distance
from .distance     import weighted_matrix_distance

__version__ = "1.0.0"
__author__  = "Zahra"
__license__ = "MIT"

__all__ = [
    "Encoder",
    "Store",
    "SLOT_NAMES",
    "matrix_distance",
    "weighted_matrix_distance",
]
