# Approach 3 — Baseline Snapshot

**Date saved:** May 14, 2026
**Status:** Frozen reference. Do not modify these files.

## Why this is preserved

This is the working state of SCBS after the Approach 3 domain-voted slot
weighting was integrated. It represents the best measured result of the
project so far on user-facing search quality metrics.

If any subsequent experiment damages F1, P@10, or speed, restore from
this folder back into `src/scbs/`.

## Architecture in this snapshot

1. Encoder — V2 vocabulary, 50 sub-clusters, 930 words
2. Blueprint — 10 slots (WHO, ACTION, TECH, EMOTION, WHEN, SOCIAL,
   INTENT, MODIFIER, WORLD, DOMAIN)
3. Matrix index — sparse (slot, value) tuples
4. Co-occurrence clustering — 14 buckets at index time
5. TF-IDF weighting on shared-slot distance
6. **Domain voting** — each record carries a domain hint (Finance, HR,
   DevOps, Security, etc.) computed at encode time from sub-cluster votes
7. **Domain-aware slot weights** — when query and record share a domain,
   that domain's slot importance profile is applied during distance
   computation

## Measured benchmark results (1,000 records, 7 queries)

| Metric | Value |
|--------|-------|
| F1 (top_k = 500)        | 29% |
| P@1                     | 100% |
| P@3                     | 100% |
| P@5                     | 100% |
| P@10                    | 96% |
| p50 latency             | 0.4 ms |
| p95 latency             | 0.7 ms |
| p99 latency             | 0.7 ms |
| Memory @ 1K             | 0.05 MB |
| Encode throughput       | 787 rec/sec |
| Monthly cost            | $0 |

## Why F1 stayed at 29%

The bucket filter (co-occurrence clustering) limits which records are
ever compared to the query. Documents about kafka and documents about
kubernetes land in different co-occurrence buckets at index time, so
they never compare against each other at query time. The domain voting
correctly identifies them both as DevOps, but it cannot reach across
the bucket boundary.

This is the limitation the next experiment attempts to address.

## Files in this snapshot

- `__init__.py`        — public API surface
- `encoder.py`         — text-to-blueprint encoder
- `vocabulary.py`      — V2 sub-cluster vocabulary
- `blueprint.py`       — 10-slot extraction
- `matrix_index.py`    — sparse row + zone index
- `distance.py`        — weighted_matrix_distance with Approach 3 hooks
- `domain_voting.py`   — sub-cluster → domain map, slot weight tables
- `clustering.py`      — co-occurrence bucket assignment
- `store.py`           — high-level public Store/Encoder wrappers

## To restore this snapshot

```bash
cp experiments/approach_3_baseline/*.py src/scbs/
```
