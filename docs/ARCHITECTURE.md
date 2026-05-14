# SCBS Architecture

## Overview

SCBS is built in five independent layers. Each layer can be used standalone, but the full pipeline gives the best results.

```
       Text Input
           │
           ▼
    ┌─────────────┐
    │  Encoder    │  Greedy longest-match + unknown word resolver
    │  encoder.py │  Output: ordered list of integer cluster IDs
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │  Blueprint  │  Map cluster IDs to 10 semantic slots
    │ blueprint.py│  Output: sparse {slot: cluster_id} dict
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │ Matrix Row  │  Convert sparse dict to sorted (slot, id) tuples
    │matrix_index │  Output: fixed-format row ready for index
    └─────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────┐
    │ Index Layer                                     │
    │   - 10-bit signature (which slots are filled)   │
    │   - Per-cluster zone (sorted ID ranges)         │
    │   - TF-IDF weighting (rare words count more)    │
    │   - Co-occurrence clusters (corpus-driven)      │
    └─────────────────────────────────────────────────┘
           │
           ▼
        Search
```

## Layer 1: Encoder

**File**: `src/scbs/encoder.py`

Converts text to a sequence of cluster IDs using greedy longest-match against a 930-word vocabulary. Unknown words go through a 4-layer resolver:

1. Exact match in vocabulary
2. Prefix/suffix morphology hints (`cyber-` → Security, `-tion` → Action)
3. Edit-distance fallback to known words
4. Cluster-based n-gram voting

Result: every word gets a cluster ID. IDs are assigned so numeric proximity reflects semantic proximity.

## Layer 2: Blueprint

**File**: `src/scbs/blueprint.py`

Extracts a sparse 10-slot semantic record from the encoded sequence. Each slot captures one orthogonal dimension of meaning:

| Slot | Name | Examples |
|---|---|---|
| 0 | WHO | developer, customer, team |
| 1 | ACTION | deploy, fail, approve |
| 2 | TECH | kafka, python, postgres |
| 3 | EMOTION | happy, frustrated, critical |
| 4 | WHEN | today, scheduled, expired |
| 5 | SOCIAL | hello, welcome, goodbye |
| 6 | INTENT | what, why, when, how |
| 7 | MODIFIER | secure, scalable, deprecated |
| 8 | WORLD | server, region, count |
| 9 | DOMAIN | fraud, incident, breach |

A typical sentence fills 3–5 slots. Slots not filled are absent from the sparse dict.

## Layer 3: Matrix Index

**File**: `src/scbs/matrix_index.py`

Converts sparse blueprints to fixed-format rows for fast comparison.

A row is a sorted list of (slot, value) tuples:
```python
[(0, 3722), (1, 4314), (2, 3401), (9, 5020)]
```

Comparison is a merge-style walk over both rows in O(k) where k is the total number of filled slots.

The index has three parts:

1. **Signature filter** — 10-bit integer where bit i is set if slot i is filled. Bitwise AND eliminates 50–70% of candidates in one operation.

2. **Zone index** — per slot, a sorted list of (value, record_id). Binary search finds records whose slot value is within range of the query in O(log n).

3. **Distance computation** — runs only on records that pass both filters above.

## Layer 4: TF-IDF Distance

**File**: `src/scbs/distance.py`

Weights distance by how rare and discriminating the words are. Rare specific words pull similar sentences closer together. Common function words contribute less.

```
weighted_distance(A, B) = raw_distance(A, B) / harmonic_mean(weight_a, weight_b)
```

where `weight_x` is the average IDF of content words in sentence x.

## Layer 5: Co-occurrence Clusters

**File**: `src/scbs/clustering.py`

Learns natural sentence groups from corpus word co-occurrence patterns. No predefined domains, no labels. Sentences sharing word patterns cluster together.

Query sentences get tagged to their dominant cluster. Search runs within that cluster plus the general pool only — typically 10–15% of the corpus.

## Search Pipeline

```
Query text
   │
   ▼
Encode + extract blueprint
   │
   ▼
Tag with co-occurrence cluster ───────► Cluster pool (~10% of corpus)
   │
   ▼
Signature filter (10-bit AND) ────────► ~50% reduction
   │
   ▼
Zone binary search ───────────────────► ~80% further reduction
   │
   ▼
TF-IDF weighted distance ─────────────► Final ranking
   │
   ▼
Top-K results
```

At each stage the candidate set shrinks. The expensive distance computation runs only on records that survive every filter.

## Per-session Puzzle Exclusion

The `excluded` parameter to `search()` is a `set` of record IDs to skip. Pass the same set across multiple search calls and each call will skip records already returned. This forces diverse, non-repeating results.

Exclusion is per-session — passing a fresh set restarts fresh. Records are never permanently demoted, so query semantics remain stable across sessions.
