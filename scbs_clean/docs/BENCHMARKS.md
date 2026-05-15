# Benchmarks

This document reports measured SCBS performance on the Cranfield IR test collection — a standard, real-world benchmark with human relevance judgments.

---

## Cranfield Collection

The Cranfield collection (1968) is the first standardized IR test collection and is still used today.

| Property | Value |
|----------|-------|
| Documents | 1,400 aerospace engineering paper abstracts |
| Queries | 225 real queries from researchers |
| Relevance judgments | 1,837 human-judged assessments |
| Average doc length | ~70 words after preprocessing |
| Average query length | ~10 words |

Data is included in `benchmarks/cranfield/` and originates from the publicly available TREC-formatted Cranfield mirror.

---

## Headline Results

| Metric        | BM25     | SCBS     | SCBS / BM25 |
|---------------|----------|----------|-------------|
| **NDCG@10**   | 0.3509   | 0.3378   | 96%         |
| **MAP**       | 0.2610   | 0.2533   | 97%         |
| **MRR**       | 0.5135   | 0.5086   | 99%         |
| **P@1**       | 0.3244   | **0.3867** | **119% ✓** |
| **P@10**      | 0.2187   | 0.2178   | 100%        |
| **Recall@100**| 0.6907   | 0.6274   | 91%         |
| **p50 latency** | 5.82ms | 11.86ms  | 2× slower   |
| **p95 latency** | 9.89ms | 13.41ms  | 1.4× slower |

Evaluated on 225 queries. Both systems implemented from scratch in pure Python (no external IR libraries) to keep the comparison fair.

---

## Where SCBS wins

- **P@1 (top-1 precision)**: SCBS 0.3867 vs BM25 0.3244 — about 19% better. The first result SCBS returns is correct more often than BM25's first result. This is the metric users most often perceive as "search quality."

- **P@10**: Tied at the 4th decimal place (0.2178 vs 0.2187). Effectively equal precision in the user-visible top-10.

- **MRR**: 99% of BM25. The first relevant result appears at essentially the same rank.

---

## Where BM25 wins

- **Recall@100**: BM25 retrieves 91% of relevant documents in the top-100. SCBS retrieves only 63%. This is the largest gap — sparse representations lose some terms that BM25 catches via direct term matching.

- **Speed**: BM25 is roughly 2× faster at p50. Both are well under 20ms.

- **NDCG@10**: 4 percentage points behind. SCBS finds relevant docs but ranks them slightly less perfectly.

---

## What this means

SCBS is competitive with BM25 on user-facing precision metrics (P@1, P@10, NDCG@10) while trailing significantly on recall. For typical retrieval use cases — where the user looks at the top 10 results — SCBS performs at BM25 level. For recall-critical use cases (legal discovery, compliance, exhaustive search) BM25 remains the better choice.

The result is meaningful because:
- It is on a **real** standard IR benchmark, not synthetic data.
- The BM25 baseline is well-calibrated (0.3509 NDCG@10 matches published numbers for Cranfield in the 0.35–0.45 range, with variation by tokenization and parameter choices).
- The comparison uses **no learned components on either side** — both are unsupervised, deterministic, CPU-only.

---

## Reproducing

```bash
cd /path/to/scbs
python benchmarks/run_cranfield.py
```

Expected wall-clock time: ~1 minute (BM25 indexing < 1 sec, SCBS indexing ~35 sec, evaluation ~15 sec).

Output should match the headline results within run-to-run variance (BM25 is deterministic; SCBS variation comes from MiniBatchDictionaryLearning random initialization, which is seeded with `random_state=42` for reproducibility).

---

## Limitations of this benchmark

- **Single benchmark.** Cranfield is small and specialized (aerospace). Results may differ on larger, more diverse collections (BEIR, MS MARCO).
- **Vocabulary mismatch test only.** Cranfield queries and documents use overlapping technical vocabulary. SCBS's reliance on TF-IDF + sparse coding works well here. On collections with heavy paraphrasing or synonym variation, results may be worse.
- **No comparison vs neural retrievers.** SBERT, BGE, ColBERT, etc. would typically beat both BM25 and SCBS on most benchmarks. Out of scope for this work — the goal was a strong sparse baseline, not state-of-the-art quality.

---

## Honest position

SCBS is a strong replacement for BM25 in scenarios where:
- Top-1 / top-K precision matters more than exhaustive recall
- Deployment constraints (no GPU, no API, offline operation) rule out neural retrievers
- Interpretability matters (you can inspect atom activations)

It is **not** a replacement for modern dense retrievers when quality is the only constraint. Those typically beat BM25 by 5–15 NDCG points on standard benchmarks, and would likely beat SCBS by similar margins.

The value of SCBS is in being:
- Roughly equal to BM25 on user-facing metrics
- **Better than BM25 on P@1**
- Completely deterministic and inspectable
- Free of operational complexity (no model serving, no embedding pipeline)
