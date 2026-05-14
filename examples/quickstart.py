"""
Quickstart example for SCBS.

Run with:
    python examples/quickstart.py
"""

from scbs import Encoder, Store


def main():
    encoder = Encoder()
    store   = Store(n_clusters=8)

    corpus = [
        "kafka consumer lag increasing critical alert",
        "kubernetes pod eviction memory pressure",
        "deployment succeeded all health checks passing",
        "redis cache miss ratio threshold breached",
        "docker container restart loop health failing",
        "fraud transaction blocked suspicious activity",
        "payment processed successfully confirmation",
        "employee performance review scheduled Monday",
        "salary adjustment approved effective immediately",
        "customer complaint refund order delivery failed",
        "authentication token expired login failed",
        "good morning team standup in five minutes",
        "database migration completed without errors",
        "security vulnerability patched and deployed",
        "happy to announce new feature is live",
    ]

    print(f"Building index with {len(corpus)} sentences...")
    for text in corpus:
        store.add(encoder.encode(text), text)
    store.build(corpus_texts=corpus)
    print(f"Index built: {len(store)} records\n")

    queries = [
        "kafka deployment failure alert",
        "employee promotion review",
        "suspicious fraud transaction",
    ]

    for query_text in queries:
        print(f"Query: {query_text}")
        query_row    = encoder.encode(query_text)
        results, _   = store.search(
            query_row, query_text, top_k=3, threshold=200,
        )

        if not results:
            print("  No matches found within threshold.\n")
            continue

        for r in results:
            print(f"  dist={r['distance']:6.1f}  {r['text']}")
        print()


if __name__ == "__main__":
    main()
