"""Debug a single query to understand what's failing."""
import sys, os, re
sys.path.insert(0, '/home/claude/scbs_release/src')
sys.path.insert(0, '/home/claude/scbs_release/experiments/approach_10_energy_based')

from scbs.clustering import build_cooccurrence, cluster_words
from energy_store import build_semantic_graph, inject_energy, diffuse_energy
from test_real_benchmark_direct import parse_trec_corpus, parse_trec_queries, parse_qrels, tokenize, text_to_clusters

DATA_DIR = "/home/claude/benchmark_data/pyterrier/tests/fixtures/vaswani_npl"

corpus = parse_trec_corpus(f"{DATA_DIR}/corpus/doc-text.trec")
queries = parse_trec_queries(f"{DATA_DIR}/query-text.trec")
qrels = parse_qrels(f"{DATA_DIR}/qrels")

print(f"Loaded {len(corpus)} docs, {len(queries)} queries")

# Pick query 1
qid = "1"
qtext = queries[qid]
relevant_docs = qrels[qid]
print(f"\nQuery {qid}: {qtext}")
print(f"Relevant docs: {list(relevant_docs.keys())[:5]}")

# Show a relevant document
sample_rel = list(relevant_docs.keys())[0]
print(f"\nSample relevant doc ({sample_rel}):")
print(f"  {corpus[sample_rel][:300]}")

# Build clustering
print("\nBuilding clustering...")
cooc = build_cooccurrence(list(corpus.values()))
print(f"  Co-occurrence has {len(cooc)} words")

word_clusters = cluster_words(cooc, n_clusters=50, top_words=10000)
print(f"  Clustered {len(word_clusters)} words")

# Check coverage
query_tokens = tokenize(qtext)
print(f"\nQuery tokens: {query_tokens}")
covered = [t for t in query_tokens if t in word_clusters]
not_covered = [t for t in query_tokens if t not in word_clusters]
print(f"  Covered ({len(covered)}): {covered}")
print(f"  NOT covered ({len(not_covered)}): {not_covered}")

# Check relevant doc coverage
rel_tokens = tokenize(corpus[sample_rel])
covered_in_rel = [t for t in rel_tokens if t in word_clusters]
not_covered_in_rel = [t for t in rel_tokens if t not in word_clusters]
print(f"\nRelevant doc tokens: {len(rel_tokens)} total")
print(f"  Covered: {len(covered_in_rel)} ({len(covered_in_rel)*100//len(rel_tokens)}%)")
print(f"  NOT covered: {len(not_covered_in_rel)}")
print(f"  Examples not covered: {not_covered_in_rel[:10]}")

# What clusters do query and doc share?
query_clusters = set(word_clusters[t] for t in covered)
rel_doc_clusters = set(word_clusters[t] for t in covered_in_rel)
print(f"\nQuery clusters: {query_clusters}")
print(f"Rel doc has {len(rel_doc_clusters)} unique clusters")
print(f"Shared clusters: {query_clusters & rel_doc_clusters}")

# Now do search
print("\nBuilding graph and indexing...")
graph = build_semantic_graph(list(corpus.values()), word_clusters, 50)
print(f"  Graph has {len(graph)} nodes")
print(f"  Edges from cluster 0: {dict(list(graph.get(0, {}).items())[:5])}")

# Inject and diffuse
initial_energy = {cid: 1.0 for cid in query_clusters}
print(f"\nInitial energy: {initial_energy}")

final_energy = diffuse_energy(initial_energy, graph, iterations=3, damping=0.7)
print(f"Final energy ({len(final_energy)} concepts):")
sorted_energy = sorted(final_energy.items(), key=lambda x: -x[1])
for cid, e in sorted_energy[:10]:
    print(f"  cluster {cid}: {e:.4f}")

# Now compute doc scores
print("\nScoring documents...")
doc_clusters = {did: set(text_to_clusters(t, word_clusters)) for did, t in corpus.items()}

# Score the relevant doc
rel_score = sum(final_energy[c] for c in doc_clusters[sample_rel] if c in final_energy)
print(f"  Relevant doc {sample_rel} score: {rel_score:.4f}")

# Score an irrelevant doc (random)
irrelevant_did = "100"
irrel_score = sum(final_energy[c] for c in doc_clusters[irrelevant_did] if c in final_energy)
print(f"  Irrelevant doc {irrelevant_did} score: {irrel_score:.4f}")

# Top 10 scoring docs overall
all_scores = []
for did, clusters in doc_clusters.items():
    score = sum(final_energy.get(c, 0) for c in clusters)
    all_scores.append((did, score))
all_scores.sort(key=lambda x: -x[1])

print(f"\nTop 10 scoring docs:")
for did, score in all_scores[:10]:
    is_rel = "★" if did in relevant_docs else " "
    print(f"  {is_rel} doc {did}: score={score:.4f}")

print(f"\nRelevant docs scores:")
for did in list(relevant_docs.keys())[:5]:
    score = sum(final_energy.get(c, 0) for c in doc_clusters[did])
    rank = next((i for i, (d, _) in enumerate(all_scores) if d == did), -1)
    print(f"  doc {did}: score={score:.4f}, rank={rank}")
