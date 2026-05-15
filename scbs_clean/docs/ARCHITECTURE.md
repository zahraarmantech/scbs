# Architecture

This document explains how SCBS works and why each design choice was made.

---

## Overview

SCBS performs retrieval through three layers:

1. **Representation:** Documents and queries are encoded as sparse activations over a learned dictionary of semantic atoms.
2. **Comparison:** Documents and queries are compared via late interaction — for each strong query atom, find the best matching document atom and weight by their cross-atom affinity.
3. **Ranking:** Sum the weighted best matches, multiplied by IDF, to produce a document score.

The system is fully unsupervised — no labeled training data, no learned embeddings, no neural network. Just TF-IDF + dictionary learning + a specific scoring function.

---

## Index time

### Step 1: TF-IDF

A standard TF-IDF matrix is built from the corpus. Defaults:
- Lowercase
- `min_df=2`, `max_df=0.95` (filter very rare and very common terms)
- `sublinear_tf=True` (log-scaled term frequencies)
- Up to 10,000 features

Output: `X ∈ R^(n_docs × n_terms)`.

### Step 2: Sparse Dictionary Learning

We decompose:

```
X ≈ H · D
```

where:
- `D ∈ R^(n_atoms × n_terms)` is the dictionary (atoms are rows).
- `H ∈ R^(n_docs × n_atoms)` are the sparse activations.

Critical detail: we use `sklearn.decomposition.MiniBatchDictionaryLearning` with `positive_dict=True` and `positive_code=True`. This produces a non-negative decomposition similar to NMF but without the orthogonality constraint.

**Why this matters.** NMF minimizes reconstruction error subject to non-negativity, which incentivizes near-orthogonal basis (each atom captures a distinct subset of vocabulary). Dictionary learning doesn't have this incentive — atoms can share vocabulary, just like real semantic concepts do.

Measured atom affinity on Cranfield:
- NMF: mean off-diagonal cosine = 0.035 (nearly orthogonal)
- Dictionary: mean off-diagonal cosine = 0.129, max = 0.637, 62% above 0.1

This overlap is what enables the late interaction in step 5 to work.

### Step 3: Atom Affinity Matrix

Compute the cosine similarity between every pair of L2-normalized atoms:

```
A = D_norm · D_norm^T   ∈ R^(n_atoms × n_atoms)
```

This is constant-cost — computed once at index time, reused on every query. With 256 atoms it's a 256×256 matrix (~262KB of floats).

### Step 4: Sparse Document Representation

For each document `i`, store its top-M strongest atoms and their activations:

```
doc_top_atoms[i]   ∈ Z^M      # atom indices
doc_top_values[i]  ∈ R^M      # activation values
```

With `top_m=15` and 1,400 docs, this is about 84KB. Sparsification reduces memory and prevents weak signals from dominating.

### Step 5: Per-Atom IDF

For each atom `a`, count documents in which it appears in the top-M:

```
idf[a] = log((n_docs + 1) / (df[a] + 1)) + 1
```

Rare atoms (appearing in few docs) get high IDF; common atoms get low IDF. This is the same intuition as term-IDF in BM25, applied at the atom level.

---

## Query time

### Step 1: Project query into atom space

```
q_codes = dict.transform(tfidf(query))
```

This solves a Lasso problem (`transform_algorithm='lasso_cd'`) to find a sparse non-negative code that reconstructs the query's TF-IDF vector from the dictionary.

### Step 2: Top-K query gating

Keep only the strongest `top_k_query` atoms. All others are zeroed.

This is the "hard gate" — it forces the system to commit to a few strong concepts rather than averaging weak signals across all atoms. Empirically this gives a ~35% NDCG improvement over uniform scoring.

### Step 3: IDF-weighted query contributions

For each surviving query atom `i`:

```
weight(i) = q_codes[i] × idf[i]
```

Rare atoms get amplified; common ones suppressed.

### Step 4: Late interaction (MaxSim)

For each query atom `i` and each document `d`, look up the best matching atom in the document:

```
best_match(i, d) = max over j in doc_top_atoms[d]:
                       doc_top_values[d][j] × affinity[i][j]
```

This is the MaxSim operation from ColBERT, applied here to dictionary atoms instead of token embeddings. The `affinity[i][j]` term is what links query atoms to semantically related document atoms even when they're not the same atom — this is why atom overlap matters.

### Step 5: Final score

```
score(q, d) = Σ over query atoms i:  weight(i) × best_match(i, d)
```

Rank documents by `score`. Return top-K.

---

## Why this works

Three things have to be true simultaneously:

1. **The representation must be sparse.** Dense representations average signal across too many dimensions — discriminating information drowns. We saw this in our early experiments: dense diffusion-based scoring gave NDCG ~0.07 on Cranfield. Top-K + top-M sparsification was a 3× improvement.

2. **The atoms must overlap.** Orthogonal atoms make the affinity matrix near-identity, which makes late interaction equivalent to a simple dot product — losing most of its benefit. Dictionary learning (without orthogonality constraints) produces atoms that share vocabulary. The cross-atom affinity matrix then has real signal.

3. **Scoring must be decisive.** Soft / smooth scoring (sum over all atoms, or worse, diffusion-based propagation) lets generic signals dominate. Top-K query gating + MaxSim makes the system commit: "this is the most relevant evidence, pick the best match for it."

Remove any one of these and the system degrades significantly. We measured each ablation — see `docs/JOURNEY.md` and `research_history/` for the full record.

---

## Computational cost

### Index time

- TF-IDF: `O(n_docs × avg_doc_length)`
- Dictionary learning: dominant cost, `O(n_iter × n_docs × n_atoms × vocab)`. ~35s for 1,400 docs × 256 atoms × 4,500 vocab on CPU.
- Atom affinity: `O(n_atoms²)`. Negligible.
- Per-doc top-M extraction: `O(n_docs × n_atoms log M)`. Fast.

### Query time

- TF-IDF transform: `O(query_length)`
- Sparse coding (Lasso): `O(n_atoms × vocab × iterations)`. ~3-8ms.
- Top-K selection: `O(n_atoms log K)`. Negligible.
- Late interaction: `O(top_k × n_docs × top_m)`. The main cost.

For 1,400 docs with top_k=10 and top_m=15: 210,000 operations per query. ~12ms p50 on a single CPU core.

### Memory

- Vocabulary: ~50KB (10K terms × 5 bytes avg)
- Dictionary `D`: `n_atoms × vocab × 4 bytes` = 4.5MB for 256 atoms, 4,500 vocab
- Affinity matrix: `n_atoms² × 4 bytes` = 262KB for 256 atoms
- Per-doc sparse rep: `n_docs × top_m × 8 bytes` = 168KB for 1,400 docs

Total under 10MB for the Cranfield index. Scales linearly in corpus size.

---

## Comparison vs alternatives

| Property | BM25 | SCBS | Dense (SBERT/BGE) |
|----------|------|------|--------------------|
| Learned components | None | None | Trained embedding model |
| Representation | Bag of words | Sparse over learned atoms | Dense vector |
| Scoring | TF × IDF × length norm | MaxSim over atoms | Cosine over vectors |
| Index time complexity | Linear | NMF-dominated | Forward pass through model |
| Query time complexity | Linear in vocab | Linear in docs | Linear in docs |
| Quality on standard benchmarks | Strong | Matches BM25 (this work) | Typically beats BM25 |
| Resource cost | Trivial | Modest CPU | GPU or paid API |
| Interpretability | Full | Full | Limited |

---

## Hyperparameter notes

- **`n_atoms`**: Start with 256. If atoms collapse to too few docs each (check `get_atom_stats()`), increase. For corpora > 10K docs try 512.
- **`top_k_query`**: 5–15 works. Lower = more decisive but risks missing signal.
- **`top_m_doc`**: 10–20. Larger improves recall slightly at memory cost.
- **`sparsity_alpha`**: 0.05–0.2. Lower = more overlap (more late interaction signal) but slower coding and noisier representations.
- **`max_features`**: 10K is fine for most corpora up to ~100K docs.
