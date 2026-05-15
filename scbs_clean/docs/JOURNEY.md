# Research Journey

The path from the initial idea to the final SCBS architecture went through 15 numbered approaches over multiple iterations. Each was implemented, measured against ground truth, and either kept or discarded.

The complete experimental record is preserved in `research_history/`. This document summarizes the key turning points.

---

## Where it started

A retrieval system that used hard clustering: assign each document to one of K co-occurrence clusters, restrict search to documents in matching clusters. Looked strong on a 1,000-document synthetic test corpus (P@10 = 96%).

**First reality check:** at 5,000 documents, P@10 dropped to 58% and Recall to 0.1%. The clustering filter, which had seemed clever, was the binding constraint. Most relevant documents were in different clusters than the query and were never compared.

---

## The first pivot: clusters → probabilistic neighborhoods (Approach 9)

Replace hard cluster assignment with soft membership: each document belongs to its top-K clusters with weights. Search documents in any of the query's top-K neighborhoods.

- 1K corpus: Recall went from 23% → 81%. The bucket ceiling broke.
- 5K corpus: Recall collapsed to 0.2%. Neighborhood definitions didn't generalize.

Concept worked at small scale but didn't survive. The bucket filter wasn't the only issue.

---

## The first real benchmark: A10 vs BM25 on Cranfield

After 10 approaches, the system was tested against BM25 on the Cranfield collection — a real IR benchmark from 1968 with 1,400 documents and human relevance judgments.

| Metric        | BM25     | A10 (energy diffusion on clusters) |
|---------------|----------|-------------------------------------|
| NDCG@10       | 0.3509   | 0.0739                              |
| MAP           | 0.2610   | 0.0511                              |

**The synthetic-corpus numbers were inflated.** On real text with strict human relevance, A10 reached only 21% of BM25's NDCG@10.

Diagnostic: the 14 co-occurrence clusters each contained 25-97% of all documents on real text. Cluster collapse. Every query activated every cluster equally. The clustering layer was fundamentally broken for real corpora.

---

## The architectural pivot: partition → sparse decomposition (Approach 12)

Stop trying to fix the clustering. Replace it with NMF (Non-negative Matrix Factorization): each document is a sparse mixture of 256 basis vectors instead of belonging to one of 14 clusters.

**Mathematically:** from `X → {c_1, ..., c_k}` (partition) to `X ≈ WH` (decomposition).

Result on Cranfield:
- NDCG@10: 0.0739 → 0.1753 (+137%)
- Reached 50% of BM25

The pivot worked. The binding constraint was the partition assumption, not the scoring formula. Documents are sparse mixtures of concepts, not members of disjoint categories.

---

## Removing what didn't help (Approach 13)

Two ablations on top of A12:
- **Hard top-K query truncation** (only strongest query atoms contribute)
- **Remove diffusion from ranking** (it was smoothing signal we needed)

Result on Cranfield:
- NDCG@10: 0.1753 → 0.2360 (+35%)
- Reached 67% of BM25

The diffusion that had seemed central to the architecture turned out to be noise. Hard gating beats smooth averaging.

---

## A wrong turn: late interaction on NMF (Approach 14)

Borrowed ColBERT's MaxSim scoring — for each query basis, find best matching doc basis. Should preserve the token-level signal that early pooling destroys.

Result on Cranfield:
- NDCG@10: 0.2360 → 0.2063 (-13%)

**Why it failed:** measured atom affinity matrix mean off-diagonal = 0.035. NMF basis are nearly orthogonal by construction (minimizing reconstruction error subject to non-negativity drives basis apart). Late interaction had no cross-basis signal to work with.

This was an important falsification: late interaction doesn't work on any sparse basis. It needs overlap.

---

## The final pivot: orthogonal → overlapping basis (Approach 15)

Replace NMF with sparse dictionary learning (specifically `MiniBatchDictionaryLearning` with positive constraints). Same `X ≈ WH` decomposition without the orthogonality incentive — atoms can share vocabulary.

Measured atom affinity:
- NMF (A14): mean 0.035, max small
- Dictionary (A15): mean 0.129 (3.7× higher), max 0.637, 62% of pairs > 0.1

Late interaction now had real signal to work with.

Result on Cranfield:

| Metric        | BM25     | SCBS A15  | vs BM25 |
|---------------|----------|-----------|---------|
| NDCG@10       | 0.3509   | 0.3378    | 96%     |
| MAP           | 0.2610   | 0.2533    | 97%     |
| MRR           | 0.5135   | 0.5086    | 99%     |
| **P@1**       | 0.3244   | **0.3867**| **119%**|
| Recall@100    | 0.6907   | 0.6274    | 91%     |

Matched BM25 on NDCG, MAP, MRR. **Beat BM25 on P@1 by 19%.**

---

## The full progression

| Step | Architecture | NDCG@10 on Cranfield | vs BM25 |
|------|--------------|----------------------|---------|
| A3   | Hard clusters | failed at 5K scale  | —       |
| A10  | Energy diffusion on clusters | 0.0739 | 21% |
| A11  | Directed flow + genericity penalty | 0.0739 | 21% (clusters still broken) |
| A12  | NMF sparse basis + diffusion | 0.1753 | 50% |
| A13  | NMF + top-K gating, no diffusion | 0.2360 | 67% |
| A14  | NMF + late interaction (MaxSim) | 0.2063 | 59% (regression) |
| **A15** | **Sparse dictionary + late interaction** | **0.3378** | **96%** |

---

## Lessons from the path

1. **Synthetic benchmarks lie.** The 1K corpus numbers were inflated by permissive keyword-overlap relevance and template-generated text. Real benchmark data (Cranfield) cut every metric by 4-10×. Run on real data early.

2. **Negative results are useful.** A11 and A14 produced no improvement (or regressions), but each falsified a hypothesis and pointed to what was actually wrong. A14's failure on NMF directly motivated A15's switch to overlapping basis.

3. **Architectural shifts beat parameter tuning.** Approaches 4-8 (not detailed here) did parameter tweaks within the cluster framework. The gains came from changing what the system was doing (partition → decomposition → overlapping decomposition).

4. **The right baseline matters.** Comparing against BM25 — a 1990s baseline that is still strong — kept the work honest. The synthetic-corpus numbers had nothing to anchor against.

5. **Sequential theoretical reasoning paid off.** Each of the final three pivots came from explicit hypotheses about what was wrong with the previous version. Most were tested in days. The system improved 3-fold through structured reasoning, not random experimentation.

---

## What's left unverified

- **Other benchmarks.** Only Cranfield was tested. BEIR (multiple domains), MS MARCO (large scale) would tell us how SCBS generalizes.
- **Comparison vs learned models.** SBERT, BGE, DPR likely beat both BM25 and SCBS on standard benchmarks, but were out of scope here.
- **Larger corpora.** Cranfield is 1,400 docs. Behavior at 100K+ docs is unmeasured.

These are the natural next steps. Everything in this repo reflects what was actually run, with the failures preserved alongside the wins.
