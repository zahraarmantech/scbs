"""
Cranfield Benchmark — SCBS vs BM25

The Cranfield collection (1968) is the first standardized IR test
collection. We include it here for reproducible benchmarking:
- 1,400 aerospace paper abstracts
- 225 real queries
- 1,837 human-judged relevance assessments

Reproduces the headline result reported in README.md.

Usage:
    python benchmarks/run_cranfield.py
"""
from __future__ import annotations

import os
import re
import sys
import time
from collections import defaultdict

# Allow running from repo root without installing
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from scbs import Retriever, BM25, metrics


CRAN_DIR = os.path.join(_HERE, "cranfield")


def parse_cranfield_docs(xml_path: str) -> dict[str, str]:
    """Parse Cranfield documents from the XML file."""
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()
    docs = {}
    for block in re.findall(r"<doc>(.*?)</doc>", content, re.DOTALL):
        docno_m = re.search(r"<docno>(\d+)</docno>", block)
        title_m = re.search(r"<title>(.*?)</title>", block, re.DOTALL)
        text_m = re.search(r"<text>(.*?)</text>", block, re.DOTALL)
        if not docno_m:
            continue
        doc_id = docno_m.group(1).strip()
        title = title_m.group(1).strip() if title_m else ""
        text = text_m.group(1).strip() if text_m else ""
        full = re.sub(r"\s+", " ", f"{title} {text}").strip()
        docs[doc_id] = full
    return docs


def parse_cranfield_queries(xml_path: str) -> dict[str, str]:
    """
    Parse Cranfield queries.

    Note: Cranfield XML uses non-sequential <num> values but the
    TREC-format qrels file uses sequential IDs. The Nth query in the
    XML corresponds to qid=N in qrels.
    """
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()
    queries = {}
    pattern = re.compile(
        r"<top>\s*<num>\s*(\d+)\s*</num>\s*<title>\s*(.*?)\s*</title>",
        re.DOTALL,
    )
    for position, match in enumerate(pattern.finditer(content), start=1):
        text = re.sub(r"\s+", " ", match.group(2).strip())
        queries[str(position)] = text
    return queries


def parse_qrels(qrel_path: str) -> dict[str, dict[str, int]]:
    """Parse TREC-format qrels: qid 0 docid relevance"""
    qrels = defaultdict(dict)
    with open(qrel_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                qrels[qid][docid] = max(0, rel) if rel >= 0 else 0
    return dict(qrels)


def evaluate(name, search_fn, queries, qrels, top_k=100):
    """Run evaluation and collect metrics."""
    ndcg, ap, mrr_, p1, p10, r100, lats = [], [], [], [], [], [], []
    for qid, query_text in queries.items():
        if qid not in qrels or not any(r > 0 for r in qrels[qid].values()):
            continue
        q_qrels = qrels[qid]
        t0 = time.perf_counter()
        results = search_fn(query_text, top_k)
        lats.append((time.perf_counter() - t0) * 1000)
        ndcg.append(metrics.ndcg_at_k(results, q_qrels, k=10))
        ap.append(metrics.average_precision(results, q_qrels))
        mrr_.append(metrics.reciprocal_rank(results, q_qrels))
        p1.append(metrics.precision_at_k(results, q_qrels, 1))
        p10.append(metrics.precision_at_k(results, q_qrels, 10))
        r100.append(metrics.recall_at_k(results, q_qrels, 100))
    lats.sort()
    return {
        "name": name,
        "queries": len(ndcg),
        "ndcg@10": sum(ndcg) / len(ndcg),
        "map": sum(ap) / len(ap),
        "mrr": sum(mrr_) / len(mrr_),
        "p@1": sum(p1) / len(p1),
        "p@10": sum(p10) / len(p10),
        "recall@100": sum(r100) / len(r100),
        "p50_ms": lats[len(lats) // 2],
        "p95_ms": lats[int(len(lats) * 0.95)],
    }


def main():
    print("=" * 72)
    print("Cranfield Benchmark — SCBS vs BM25")
    print("=" * 72)

    print("\nLoading Cranfield collection...")
    docs = parse_cranfield_docs(os.path.join(CRAN_DIR, "cran.all.1400.xml"))
    queries = parse_cranfield_queries(os.path.join(CRAN_DIR, "cran.qry.xml"))
    qrels = parse_qrels(os.path.join(CRAN_DIR, "cranqrel.trec.txt"))
    print(f"  {len(docs)} documents, {len(queries)} queries, "
          f"{sum(len(v) for v in qrels.values())} relevance judgments")

    doc_ids = list(docs.keys())
    corpus_texts = [docs[d] for d in doc_ids]

    # BM25
    print("\nBuilding BM25 index...")
    bm25 = BM25()
    bm25.fit(corpus_texts, doc_ids)
    print("Evaluating BM25...")
    bm25_results = evaluate(
        "BM25",
        lambda q, k: [r.doc_id for r in bm25.search(q, k)],
        queries, qrels,
    )

    # SCBS
    print("\nBuilding SCBS index...")
    retriever = Retriever(
        n_atoms=256,
        top_k_query=10,
        top_m_doc=15,
        sparsity_alpha=0.1,
    )
    retriever.fit(corpus_texts, doc_ids, verbose=True)
    print("Evaluating SCBS...")
    scbs_results = evaluate(
        "SCBS",
        lambda q, k: [r.doc_id for r in retriever.search(q, k)],
        queries, qrels,
    )

    # Results
    print("\n" + "=" * 72)
    print("Results")
    print("=" * 72)
    print(f"\n  {'Metric':<14} {'BM25':>10} {'SCBS':>10} {'SCBS/BM25':>12}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*12}")
    for m in ["ndcg@10", "map", "mrr", "p@1", "p@10", "recall@100"]:
        b = bm25_results[m]
        s = scbs_results[m]
        pct = (s / b * 100) if b > 0 else 0
        mark = "✓" if s >= b else " "
        print(f"  {m:<14} {b:>10.4f} {s:>10.4f} {pct:>10.0f}% {mark}")

    print(f"\n  {'Speed':<14} {'BM25':>10} {'SCBS':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10}")
    print(f"  {'p50 (ms)':<14} {bm25_results['p50_ms']:>10.2f} {scbs_results['p50_ms']:>10.2f}")
    print(f"  {'p95 (ms)':<14} {bm25_results['p95_ms']:>10.2f} {scbs_results['p95_ms']:>10.2f}")
    print(f"\n  Queries evaluated: {bm25_results['queries']}")


if __name__ == "__main__":
    main()
