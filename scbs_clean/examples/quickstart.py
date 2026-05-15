"""
SCBS quickstart example.

Builds an index on a small corpus, runs queries, prints results.
For a real benchmark see benchmarks/run_cranfield.py.
"""
from scbs import Retriever


corpus = [
    # DevOps
    "Kafka consumer group lag increased beyond limits in production cluster after deployment.",
    "Kubernetes pod evicted due to memory pressure on node, autoscaler triggered.",
    "Database migration completed successfully after rolling restart procedure.",
    "API response latency exceeded p99 threshold, on-call engineer paged.",
    "Container registry push failed due to authentication credentials expiring.",
    # Security
    "SSL certificate renewal scheduled for next maintenance window to prevent outage.",
    "Authentication service experiencing elevated error rates from token validation.",
    "Multiple failed login attempts detected, account temporarily locked for security.",
    "Security incident response team activated after firewall alert triggered.",
    "OAuth token refresh failing, users repeatedly logged out of applications.",
    # Finance
    "Wire transfer flagged for manual fraud review by compliance team.",
    "Customer reported duplicate charge on their monthly account statement.",
    "Market data feed latency spike caused pricing engine delays in trading.",
    "Regulatory reporting deadline approaching, data validation still in progress.",
    "Payment processor declining transactions due to elevated risk score.",
    # HR
    "Performance review cycle starting next month for engineering team members.",
    "New hire onboarding documents have been prepared and shared with HR.",
    "Annual benefits enrollment period closes Friday, employees must select coverage.",
    "Technical interview scheduled for senior software engineer position next week.",
    "Promotion approved for high-performing employee exceeding quarterly targets.",
    # Customer
    "Customer complaint about delayed shipping escalated to senior support manager.",
    "Refund request approved for damaged product, replacement shipped expedited.",
    "Service outage affecting multiple regions, communications sent to all customers.",
    "Bulk order processing stuck due to inventory shortage in warehouse system.",
    "Subscription renewal failed because payment method on file has expired.",
]

doc_ids = [f"doc-{i:03d}" for i in range(len(corpus))]


def main():
    print("Building index...")
    retriever = Retriever(n_atoms=32, top_k_query=5, top_m_doc=8)
    retriever.fit(corpus, doc_ids, verbose=True)

    queries = [
        "kafka consumer lag deployment",
        "authentication token failures",
        "fraud detection wire transfer",
        "hiring interview engineering",
        "customer refund damaged product",
    ]

    for q in queries:
        print(f"\nQuery: {q!r}")
        results = retriever.search(q, top_k=3)
        if not results:
            print("  (no matches)")
            continue
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.doc_id}] score={r.score:.3f}")
            print(f"     {r.text}")


if __name__ == "__main__":
    main()
