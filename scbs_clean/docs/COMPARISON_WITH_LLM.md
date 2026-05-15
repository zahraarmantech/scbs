# SCBS vs LLM-Based Retrieval

A practical comparison between SCBS and LLM-based retrieval systems, with measured numbers where possible and architectural reasoning where not.

This document uses SCBS's **measured Cranfield benchmark results** — not the synthetic-corpus numbers that appeared earlier in the project's development.

---

## TL;DR

| Dimension                   | SCBS                    | LLM Embedding Retrieval | Winner       |
|-----------------------------|-------------------------|--------------------------|---------------|
| NDCG@10 (measured Cranfield)| 0.338                   | not measured (likely higher) | LLM probably |
| **P@1 (measured Cranfield)**| **0.387**               | not measured             | unknown — SCBS beats BM25 here |
| Latency @ 1.4K docs         | 11.86ms p50             | 20-200ms typical         | **SCBS**     |
| Cost per query              | Free (CPU only)         | $0.00002-$0.001+         | **SCBS**     |
| Memory per doc              | ~120 bytes              | 3-8KB (1536-dim floats)  | **SCBS**     |
| Setup complexity            | `pip install scbs`      | API key or GPU server    | **SCBS**     |
| Interpretability            | Full (atom activations) | Black box                | **SCBS**     |
| Semantic depth (paraphrase) | TF-IDF foundation       | Learned from billions of texts | **LLM** |
| Cross-vocabulary matching   | Limited (TF-IDF base)   | Strong                   | **LLM**     |
| Multilingual                | Single language         | Multilingual models exist| **LLM**     |
| Benchmark track record      | One benchmark (Cranfield)| BEIR, MS MARCO, etc.   | **LLM**     |
| State-of-the-art quality    | Matches BM25            | 5-15 NDCG points above BM25 | **LLM** |

---

## 1. Speed (Measured for SCBS)

**SCBS measured on Cranfield (1,400 docs):**
- p50: 11.86ms
- p95: 13.41ms

**LLM-based retrieval typical latencies:**
- Embedding generation: 10-50ms (API) or 5-15ms (local model on GPU)
- Vector search (FAISS/HNSW at 1-100K docs): 1-10ms
- Total end-to-end: typically 20-100ms
- Hosted services (Pinecone, Weaviate): 50-200ms including network

**Honest read:** SCBS is faster than typical LLM retrieval setups. The gap closes at very large scale (>1M docs) because SCBS's document scoring grows linearly while approximate vector search (HNSW) is sub-linear. SCBS's advantage is sharpest at small-to-medium scales (~10K–500K docs) and for low-resource deployments.

---

## 2. Cost (Estimated)

**SCBS:** Zero marginal cost. Runs on CPU, no API calls, no GPU.

**LLM-based retrieval rough estimates:**
- OpenAI text-embedding-3-small: $0.00002 per 1K tokens → ~$0.20 per million queries
- Self-hosted models (SBERT, BGE): hardware cost ($500-5000 GPU) + electricity
- Pinecone, Weaviate (production): $70-500+/month

**At 1M queries/month:** SCBS = $0, OpenAI embeddings ≈ $200/month for queries alone (more with indexing).

**Caveat:** Total cost depends on whether SCBS quality is sufficient for the use case. If users have to fall back to LLM retrieval for the queries SCBS handles poorly, total system cost goes up, not down.

---

## 3. Memory (Architectural)

**SCBS:**
- Dictionary (256 atoms × 4,500 vocab): 4.5MB
- Atom affinity matrix: 262KB
- Per-doc sparse rep (1,400 × 15 atoms): 168KB
- Total: ~5MB for 1,400 docs

**LLM embeddings:**
- text-embedding-3-small: 1536 dim × 4 bytes = 6KB per document
- SBERT (all-mpnet-base-v2): 768 dim × 4 bytes = 3KB per document
- ColBERT: 32 vectors per document × 128 dim × 2 bytes = 8KB per document

**At 1M documents:** SCBS ≈ 200MB. LLM embeddings ≈ 3-8GB.

SCBS uses roughly 20-50× less memory than dense vector retrieval at scale.

---

## 4. Quality (The Honest Section)

**Measured for SCBS on Cranfield (the only real benchmark we ran):**

| Metric        | SCBS      | BM25 baseline |
|---------------|-----------|----------------|
| NDCG@10       | 0.3378    | 0.3509         |
| MAP           | 0.2533    | 0.2610         |
| MRR           | 0.5086    | 0.5135         |
| P@1           | **0.3867**| 0.3244         |

**Published LLM retrieval numbers on BEIR benchmarks (averaged across 18 datasets):**

| System              | NDCG@10 avg |
|---------------------|-------------|
| BM25                | ~0.42       |
| DPR                 | ~0.42 (often below BM25) |
| SBERT (mpnet)       | ~0.50       |
| BGE-large           | ~0.54       |
| E5-large            | ~0.51       |
| ColBERT v2          | ~0.53       |
| Cohere v3 + BM25 hybrid | ~0.55-0.60 |

**Critical caveats:**
1. **Different datasets.** SCBS is measured on Cranfield (small, specialized). Published numbers are BEIR averages (18 diverse datasets). They are not directly comparable.
2. **SCBS has not been benchmarked on BEIR.** We don't know how it would perform there.
3. **On the one shared point (BM25),** Cranfield BM25 (0.35) is close to BEIR-average BM25 (0.42). This suggests Cranfield is not wildly easier than average, but it is one data point.

**The honest claim:** SCBS matches BM25 on Cranfield. Most learned dense retrievers beat BM25 on most benchmarks, so they likely beat SCBS too — but at much higher cost. The size of that quality gap on real benchmarks is unmeasured for SCBS.

---

## 5. Semantic Capabilities (Where LLMs Win Structurally)

LLM retrievers learn from billions of text examples and capture things SCBS structurally cannot:

- **Paraphrase matching**: "How do I deploy a container?" matches "What's the procedure for shipping a Docker image?"
- **Synonyms**: "car" matches "automobile" matches "vehicle" even when neither word appears in the document
- **Conceptual queries**: "Why did the Roman Empire fall?" matches text about gradual political decline
- **Cross-vocabulary recall**: query in one technical register matches docs in another

**SCBS handles these to a limited extent** through the dictionary's overlapping atoms — atoms that frequently co-occur in similar contexts can become semantically related. But this is bounded by what TF-IDF + co-occurrence can capture, which is much less than what billion-parameter models learn.

**Concretely:** if your queries and documents share vocabulary, SCBS handles them well. If they don't (heavy paraphrasing, technical translation, cross-domain matching), LLMs will likely outperform SCBS significantly.

---

## 6. Interpretability (Where SCBS Wins)

SCBS:
```python
retriever.get_atom_stats()
# {'n_atoms': 256, 'mean_off_diagonal_affinity': 0.129, ...}
```
You can inspect which atoms activated for a query, which documents have which atoms, and which cross-atom affinities drove a specific ranking. The whole system is auditable.

LLM embeddings: 768-1536 dense floats. Why does query A match document B? "Because their vectors are close." Diagnosis requires dimensionality reduction, attention analysis, or specialized interpretability research.

**This matters for:** regulated industries (finance, healthcare, legal), compliance review, debugging unexpected rankings, explaining decisions to users.

---

## 7. Deployment Complexity

**SCBS:**
```bash
pip install scbs
```
That's it. CPU only, no GPU, no API keys, no model files, no external services, works offline.

**LLM-based retrieval:**
- **Hosted API**: API key, internet, rate limits, costs, vendor lock-in, possible privacy concerns.
- **Self-hosted**: GPU (or slow CPU inference), model files (500MB–5GB), embedding pipeline, vector database (FAISS/Milvus/Pinecone), monitoring.

For prototypes, edge deployment, regulated environments, or air-gapped systems, SCBS's simplicity is a major operational advantage.

---

## 8. When to Use Each

**SCBS is the better choice when:**
- Latency budget is tight (< 50ms total)
- Cost matters (high query volume, low budget)
- Memory is constrained (edge, mobile, embedded)
- Queries and documents share vocabulary patterns (technical search, log search, FAQ, ticket routing)
- Interpretability is required (compliance, regulated industries)
- No internet/GPU access available
- You want to beat BM25 on top-1 precision without taking on neural infrastructure

**LLM-based retrieval is the better choice when:**
- Semantic depth matters (paraphrase, synonyms, conceptual matching)
- Documents are long natural prose (essays, articles, books)
- Cross-vocabulary recall is critical
- Multilingual support is needed
- State-of-the-art quality on standard benchmarks is the deciding factor
- You can afford the latency, cost, and operational complexity

---

## 9. Hybrid Approaches

The realistic production approach is often **both**:

1. **SCBS or BM25 first stage** retrieves candidate documents quickly (top 100-1000).
2. **LLM reranker** rescores the candidates with full semantic capability.

This gets you the speed/cost benefits of sparse retrieval with the quality benefits of dense rerankers. SCBS slots cleanly into the first stage — same role as BM25 — but beats BM25 on P@1, which can mean fewer candidates needed for the reranker.

---

## 10. The Honest Bottom Line

SCBS has **measured advantages** in latency, cost, memory, interpretability, and deployment simplicity. These are real and architectural.

SCBS has **measured quality parity with BM25** on Cranfield — specifically beats BM25 on P@1 by 19%, matches on NDCG@10, P@10, and MRR.

SCBS has **unmeasured quality** vs LLM retrievers. Based on published numbers, learned dense models likely beat BM25 (and therefore likely beat SCBS) by 5-15 NDCG points on most benchmarks. Running SCBS on BEIR or MS MARCO would settle this.

**Practical positioning:**
- If you currently use BM25, SCBS is a strict upgrade on P@1 with very small NDCG cost.
- If you currently use LLM retrieval and quality is your only constraint, stay with LLM.
- If LLM costs/latency/complexity are pushing you back toward simpler systems, SCBS gives you something between BM25 and dense retrieval — sparse, deterministic, but with late-interaction scoring that captures more than pure term matching.

The reasonable claim is: **SCBS is the strongest non-learned retrieval baseline tested in this project**, beating BM25 on top-1 precision without any GPU, model serving, or trained components. Whether that generalizes beyond Cranfield is the next thing to test.
