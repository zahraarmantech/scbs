# Development Journey

This document records the path from the initial Approach 3 baseline to the final Approach 10 energy-based system. The honest version: what was tried, what worked, what failed, and what each failure taught.

---

## Starting Point: Approach 3

Approach 3 was the first version that produced strong-looking numbers on a 1,000-sentence test corpus:

- P@1 = 100%
- P@10 = 96%
- p50 latency = 0.4ms

The architecture: cluster words by co-occurrence, assign each document to exactly one cluster (its "bucket"), and at query time only compare the query against documents in matching buckets.

**Recall was 23%.** This was acknowledged but treated as an acceptable tradeoff for precision-focused search.

---

## First Reality Check: Scale Testing

The 1K corpus was small. We built a 5,000-document realistic test (templates across 6 domains: DevOps, Finance, HR, Security, Customer Support, Engineering) and ran Approach 3 on it.

**Results at 5K:**
- P@10 dropped from 96% to **58%**
- Recall collapsed from 23% to **0.1%**
- p50 went from 0.4ms to 3.7ms

The "good" 1K numbers did not survive standard-scale testing. The bucket filter — which had seemed clever — turned out to be the binding constraint. Each bucket at 5K contained ~350 documents, and the filter was correctly identifying buckets but missing relevant documents that had been assigned elsewhere.

This was the honest moment. The system as designed did not scale.

---

## Approach 5 and 6: Eliminating Hypotheses

**Approach 5** tried to refine bucket assignment quality. Marginal improvement. Did not fix the underlying issue.

**Approach 6** ran a threshold sweep — testing distance cutoffs from 50 to 9000. The result was striking: **identical metrics across all thresholds**. This proved the threshold parameter was a dead lever. The bucket filter was making the actual selection decisions; the distance threshold was just cosmetic.

Approach 6's contribution was negative evidence: it ruled out a class of fixes. The bottleneck was upstream of distance computation.

---

## Approach 9: Probabilistic Neighborhoods

The diagnosis from Approaches 3-6: hard bucket assignment forces a binary "in or out" decision per document, and the document's actual location in semantic space is more nuanced.

**The fix:** soft membership. Each document belongs to its top-3 most similar clusters with normalized membership scores. Queries identify their own top-3 clusters and search all documents with any membership in those neighborhoods.

The first version used membership directly in ranking (`distance / membership`). It broke precision — too many weak matches reached the top. The second version separated concerns: **membership for candidate generation, a 5-component scoring formula for ranking**:

```
score = 2.0 × semantic_alignment
      + 1.5 × rare_term_gain
      + 1.0 × interaction_strength
      + 0.5 × directional_activation
      - 1.0 × semantic_conflict
```

**Results at 1K:**
- Recall: 23% → **81%** (+58pp)
- P@10: 96% → 87% (-9pp)
- F1: 29% → 36%

Genuine improvement at small scale. The bucket filter ceiling was broken.

**Results at 5K:**
- P@10: **92%**
- Recall: **0.2%**
- p50: 60ms

At scale, neighborhoods collapsed. The Jaccard-similarity-based centroid computation that defined neighborhoods didn't generalize from 14 buckets of ~70 docs to 14 buckets of ~350 docs. Documents stopped reaching candidate pools.

We tried expanding neighborhoods (top-3 → top-5). No effect. The issue was architectural, not parametric.

---

## Approach 10: Energy-Based Retrieval

The conceptual shift came from rethinking what retrieval is. Vector similarity asks: *"how close is this document to my query in some learned space?"* But that framing presumes the right operation is distance computation. What if it isn't?

Cognitive science offers an older paradigm: **spreading activation**. The brain doesn't compute cosine similarity. Memory retrieval is associative — concepts activate related concepts, and what surfaces is what gets sufficiently activated.

**The architecture:**
1. Build a graph where nodes are concept clusters and edges are co-occurrence strength
2. Query injects energy at its concept nodes
3. Energy diffuses through the graph for N iterations with damping
4. Documents score by sum of energy at their concept nodes

No distance computation. No nearest-neighbor search. Just activation flow.

**Results at 1K:**
- F1: 29% → **53%** (+24pp vs Approach 3)
- Recall: 23% → 52% (+29pp)
- p50: 0.4ms → **0.1ms** (faster)

Faster *and* better recall. The first time in the project that the precision-recall-speed tradeoff appeared to break favorably.

**Results at 5K, 10K, 50K:** see RESULTS.md. Precision stayed at 97-98% across the entire scale range. NDCG@10 stayed above 0.969.

---

## A Failed Optimization (Documented Honestly)

At 50K, latency had grown from 0.1ms to 5.6ms. This was hypothesized to be a scoring bottleneck — at 50K with 14 clusters, each cluster has ~3,500 documents, and scoring touched all of them in activated clusters.

A "generic fix" was proposed: after diffusion, only score documents from concepts whose energy exceeds the mean activated energy. This was designed to have no magic numbers and adapt automatically to the data.

**It didn't work.** At 20K and 50K the "optimized" version was slower across most percentiles. Quality stayed identical, but p50 degraded.

**Why it failed:** the hypothesized bottleneck wasn't real. Energy decay during diffusion already produces a sparse distribution naturally. Adding an explicit threshold filter added overhead without meaningful savings. The original Approach 10 was already handling this efficiently.

The lesson: diagnose before optimizing. The optimization was discarded.

---

## What This Process Demonstrated

Across 10 numbered approaches (5 are documented here, the others were small variations or dead ends), some patterns held:

- **Strong 1K numbers do not survive scale testing.** Every approach that looked good at 1K had to prove itself at 5K and beyond. Most didn't.
- **Negative results are useful.** Approach 6's threshold sweep produced no improvement but eliminated a class of fixes. That made the path forward clearer.
- **Architectural shifts beat parametric tuning.** Approaches 5, 6, and the parameter tweaks within 9 produced marginal gains. The jumps came from rethinking what the system was actually doing — soft membership (Approach 9), then activation flow (Approach 10).
- **Failed optimizations should be documented.** The "generic threshold fix" didn't work. Recording it prevents someone else from trying the same thing and saves the next person time.

---

## What's Left Unverified

The biggest open question: **do these numbers hold on standard benchmarks?**

Our test corpus is synthetic — templates across 6 domains, ~600-word vocabulary, keyword-overlap relevance. BEIR and MS MARCO use real text, larger vocabularies, and human-judged relevance.

Until Approach 10 is run on those datasets and compared against BM25 (NDCG@10 ≈ 0.65-0.70) and dense retrievers, the synthetic-corpus numbers cannot be claimed as production-ready results. They establish that the architecture *works* on a controlled test. Whether it *generalizes* is the next question.

If the numbers hold, this is a real result worth publishing. If they don't, the synthetic-corpus permissive relevance was inflating things and the architecture needs further work. Either outcome is honest progress.
