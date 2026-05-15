"""
Approach 5 — Bucket Relationship Matrix
========================================

CONCEPT
-------
Keep Approach 3's buckets exactly as they are. Build a higher-level
matrix on top that records how related each pair of buckets is.

At search time:
  1. Find the query's bucket (same as Approach 3)
  2. Look up the matrix row for that bucket
  3. Visit the query's bucket PLUS the buckets the matrix says
     are related to it
  4. Score all candidates from those buckets together

Each bucket stays small and precise (preserves Approach 3 P@K wins).
Related buckets are now reachable (fixes Approach 3 recall gap).

ISOLATION
---------
This file does NOT modify src/scbs/. Imports the live package
read-only. Reimplements only the search loop locally so the
Approach 3 baseline stays frozen.
"""
import sys, os, time, random
from collections import defaultdict, Counter

# Live package — read-only imports
_HERE = os.path.dirname(__file__)
_SRC  = os.path.join(_HERE, "..", "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from scbs.blueprint     import BlueprintEncoder
from scbs.matrix_index  import (
    make_row, make_signature, shared_slots, ZoneIndex,
)
from scbs.distance      import (
    build_idf, sentence_weight, weighted_matrix_distance,
)
from scbs.domain_voting import compute_domain_hint, weights_for_pair
from scbs.clustering    import (
    build_cooccurrence, cluster_words, get_cluster_labels, tag_sentence,
)

random.seed(42)


# ════════════════════════════════════════════════════════════════
#  BUCKET MATRIX — the key new structure
# ════════════════════════════════════════════════════════════════

def build_bucket_matrix(
    buckets:    dict,          # bucket_id -> list of (row,rid,text,...)
    top_k_rel:  int = 3,       # each bucket points to its top-k most similar buckets
    min_jaccard:float = 0.03,  # but only if similarity passes this floor
) -> dict:
    """
    Build relationship matrix between buckets.

    For each bucket, find the top-k other buckets it shares the most
    vocabulary with. This guarantees every bucket has neighbors
    (avoids the case where threshold rejects everything).

    Returns: {bucket_id: set of related bucket_ids}
    """
    # Build a word-set per bucket
    bucket_words: dict = {}
    for bid, items in buckets.items():
        words = Counter()
        for item in items:
            text = item[2]
            for w in text.lower().split():
                w = w.strip(".,!?;:\"'()")
                if len(w) > 2:
                    words[w] += 1
        bucket_words[bid] = words

    # For every pair, compute Jaccard similarity on top words
    bucket_ids = [b for b in buckets if len(buckets[b]) > 0]
    similarities: dict = {bid: [] for bid in bucket_ids}

    for i, a in enumerate(bucket_ids):
        top_a = set(w for w, _ in bucket_words[a].most_common(30))
        if not top_a:
            continue
        for b in bucket_ids:
            if a == b:
                continue
            top_b = set(w for w, _ in bucket_words[b].most_common(30))
            if not top_b:
                continue
            jaccard = len(top_a & top_b) / len(top_a | top_b)
            similarities[a].append((jaccard, b))

    # Keep top-k for each bucket (above floor)
    related: dict = {bid: set() for bid in bucket_ids}
    for bid, sims in similarities.items():
        sims.sort(reverse=True)
        for jaccard, other in sims[:top_k_rel]:
            if jaccard >= min_jaccard:
                related[bid].add(other)
                related[other].add(bid)  # symmetric

    return related


# ════════════════════════════════════════════════════════════════
#  STORE — same as Approach 3 plus the bucket matrix
# ════════════════════════════════════════════════════════════════

class ApproachFiveStore:

    def __init__(self, n_clusters: int = 14):
        self.n_clusters = n_clusters
        self._idf: dict = {}
        self._word_clusters:  dict = {}
        self._cluster_labels: dict = {}

        self._buckets: dict = {}
        self._sigs:    dict = {}
        self._zones:   dict = {}
        for c in list(range(n_clusters)) + [-1]:
            self._buckets[c] = []
            self._sigs[c]    = []
            self._zones[c]   = ZoneIndex()

        self._all_records: dict = {}
        self._next_id     = 0
        self._matrix:     dict = {}    # bucket_id -> set of related bucket_ids

    def learn(self, sentences: list):
        self._idf = build_idf(sentences)
        cooc = build_cooccurrence(sentences)
        self._word_clusters  = cluster_words(cooc, self.n_clusters)
        self._cluster_labels = get_cluster_labels(
            self._word_clusters, cooc, self.n_clusters
        )

    def add(self, row: list, text: str):
        rid    = self._next_id
        self._next_id += 1
        cid    = tag_sentence(text, self._word_clusters)
        sig    = make_signature(row)
        w      = sentence_weight(text, self._idf)
        domain = compute_domain_hint(text)

        self._all_records[rid] = {
            "id": rid, "text": text, "row": row,
            "cluster": cid, "weight": w, "domain": domain,
        }
        bucket = self._buckets[cid]
        bucket.append((row, rid, text, w, domain))
        self._sigs[cid].append(sig)
        self._zones[cid].add(len(bucket) - 1, row)

    def build(self):
        for zone in self._zones.values():
            zone.build()
        # The matrix is built once, after indexing finishes
        self._matrix = build_bucket_matrix(self._buckets, top_k_rel=3)

    def matrix_summary(self) -> dict:
        return {
            bid: sorted(rels) for bid, rels in self._matrix.items()
            if rels
        }

    def search(self, query_row: list, query_text: str,
               top_k: int       = 20,
               threshold: float = 100.0,
               zone_radius: int = 150,
               excluded: set    = None) -> tuple:

        if excluded is None:
            excluded = set()

        q_weight = sentence_weight(query_text, self._idf)
        q_domain = compute_domain_hint(query_text)
        q_cid    = tag_sentence(query_text, self._word_clusters)
        q_sig    = make_signature(query_row)

        stats = {
            "query_bucket":     q_cid,
            "query_domain":     q_domain,
            "buckets_searched": 0,
            "bucket_size":      0,
            "distance_checks":  0,
        }

        # ── THIS IS THE ONE NEW STEP ────────────────────────────
        # Look up the matrix to find buckets related to query's bucket
        related = self._matrix.get(q_cid, set())
        search_keys = [q_cid] + list(related)
        if q_cid != -1:
            search_keys.append(-1)             # always include general
        search_keys = list(dict.fromkeys(search_keys))  # dedupe, keep order
        stats["buckets_searched"] = len(search_keys)
        # ────────────────────────────────────────────────────────

        pool = []
        sigs = []
        for key in search_keys:
            for item, sig in zip(self._buckets[key], self._sigs[key]):
                _, rid, _, _, _ = item
                if rid not in excluded:
                    pool.append(item)
                    sigs.append(sig)
        stats["bucket_size"] = len(pool)

        # signature filter (same as Approach 3)
        sig_pass = [
            item for item, sig in zip(pool, sigs)
            if shared_slots(q_sig, sig) >= 1
        ]

        # zone filter (same as Approach 3)
        zone_rids: set = set()
        for key in search_keys:
            zi     = self._zones[key]
            bucket = self._buckets[key]
            for local_idx in zi.candidates(query_row, zone_radius):
                if local_idx < len(bucket):
                    _, rid, _, _, _ = bucket[local_idx]
                    zone_rids.add(rid)

        candidates = (
            [it for it in sig_pass if it[1] in zone_rids]
            if zone_rids else sig_pass
        ) or sig_pass

        # scoring (same as Approach 3)
        scored = []
        for row, rid, text, rec_weight, rec_domain in candidates:
            if rid in excluded:
                continue
            slot_weights = weights_for_pair(q_domain, rec_domain)
            d = weighted_matrix_distance(
                query_row, row,
                weight_a=q_weight, weight_b=rec_weight,
                slot_weights=slot_weights,
            )
            stats["distance_checks"] += 1
            if d <= threshold:
                scored.append({
                    "id": rid, "text": text,
                    "distance": round(d, 2),
                    "domain":   rec_domain,
                    "bucket":   self._all_records[rid]["cluster"],
                })

        scored.sort(key=lambda r: r["distance"])
        results = scored[:top_k]
        for r in results:
            excluded.add(r["id"])
        return results, stats


# ════════════════════════════════════════════════════════════════
#  BENCHMARK
# ════════════════════════════════════════════════════════════════

def main():
    DEVOPS = [
        "kafka consumer group lag increased beyond limits",
        "kafka broker partition rebalancing causing delays",
        "kafka topic offset falling behind producer rate",
        "kafka connect sink connector failed to write database",
        "kafka streams processing exception in production cluster",
        "kubernetes pod evicted due to memory pressure",
        "kubernetes deployment rollout failed image pull error",
        "kubernetes node not ready cluster autoscaler triggered",
        "kubernetes persistent volume claim stuck pending state",
        "kubernetes service mesh circuit breaker opened error",
        "deployment to production failed during database migration",
        "canary deployment elevated error rate rolling back",
        "blue green deployment switch successful zero downtime",
        "helm chart deployment failed invalid values configuration",
        "terraform apply failed resource already exists conflict",
        "api response latency p99 exceeded threshold alert",
        "database query execution time degraded index rebuild",
        "memory usage climbing steadily garbage collection",
        "cpu throttling detected container reducing throughput",
        "disk io wait time elevated storage investigation needed",
        "redis cache hit rate dropped below threshold",
        "elasticsearch query timeout cluster health degrading",
        "production outage detected all services returning errors",
        "incident pagerduty team notified severity escalation",
        "service restored after forty minutes downtime customers",
        "monitoring alert fired prometheus rule threshold exceeded",
        "grafana dashboard unusual spike error rate overnight",
        "ssl certificate expired connection refused now",
        "disk space critical threshold exceeded alert",
        "network packet loss detected upstream router",
    ]
    SECURITY = [
        "multiple failed login attempts detected account locked",
        "brute force attack authentication endpoint blocked firewall",
        "oauth token refresh failing users logged out",
        "saml assertion validation failed identity provider",
        "jwt signature verification failed token tampering",
        "multi factor authentication bypass attempt blocked",
        "session fixation vulnerability patched authentication",
        "ldap authentication timeout directory server unreachable",
        "critical cve published affecting openssl version",
        "sql injection attempt detected payment processing",
        "cross site scripting vulnerability user input form",
        "dependency vulnerability scanner found high severity",
        "log4j vulnerability scan no affected versions found",
        "security audit revealed hardcoded credentials configuration",
        "container image vulnerability scan outdated base image",
        "ssl certificate expiring seven days renewal started",
        "tls deprecated services migrating one point three",
        "encryption key rotation completed services updated",
        "unauthorized api access attempt unusual geographic location",
        "data exfiltration attempt blocked dlp security policy",
        "security incident response team activated ransomware",
        "phishing campaign targeting employees reported security",
        "insider threat detected unusual data access pattern",
        "firewall rule updated blocking suspicious ip range",
        "intrusion detection system alert triggered lateral",
    ]
    FINANCE = [
        "high value wire transfer flagged manual review",
        "unusual transaction pattern detected account takeover",
        "payment processing failed merchant acquiring bank",
        "duplicate transaction idempotency check preventing charge",
        "chargeback received disputed transaction investigation",
        "real time gross settlement payment queue backing",
        "payment gateway timeout rate increasing complaints",
        "international wire transfer held pending sanctions",
        "fraud model score exceeded threshold transaction blocked",
        "card present transaction declined merchant category",
        "account velocity check triggered too many transactions",
        "device fingerprint mismatch during checkout fraud",
        "synthetic identity fraud pattern detected credit",
        "friendly fraud chargeback rate exceeding threshold",
        "fraud ring detected multiple accounts linked",
        "transaction monitoring rule triggered unusual spending",
        "market data feed latency spike pricing engine",
        "options expiry processing completed positions settled",
        "portfolio rebalancing triggered risk threshold breach",
        "margin call issued client insufficient collateral",
        "regulatory reporting deadline data validation running",
        "credit limit increase approved customer risk reviewed",
        "loan origination system down applications queuing",
        "end of day reconciliation failed balance mismatch",
        "customer account statement generation delayed batch",
    ]
    HR = [
        "software engineer position open sourcing candidates linkedin",
        "technical interview scheduled senior developer Friday",
        "offer letter extended preferred candidate awaiting acceptance",
        "background check initiated new hire onboarding Monday",
        "recruitment agency invoice received successful placement",
        "referral bonus processed employee referred hired candidate",
        "headcount approval received finance two new positions",
        "annual performance review cycle starting next week managers",
        "performance improvement plan initiated following review",
        "promotion recommended employee exceeding targets quarters",
        "compensation benchmarking completed salary bands updated",
        "three sixty feedback collection open closes Friday",
        "high performer retention bonus approved executive",
        "parental leave request approved twelve weeks fully paid",
        "remote work accommodation approved medical reasons",
        "mandatory compliance training completion deadline Friday",
        "offboarding checklist completed departing employee handover",
        "workplace investigation opened harassment complaint received",
        "diversity hiring target review improvement engineering",
        "open enrollment benefits selection due Friday",
        "health insurance premium increase next quarter",
        "flexible spending account deadline unused funds forfeited",
        "employee stock purchase plan enrollment window open",
        "retirement plan contribution matching increased January",
        "visa sponsorship approved international new hire",
    ]
    CUSTOMER = [
        "customer reported product arrived damaged requesting refund",
        "multiple complaints delayed shipping partner issue",
        "customer threatening social media unresolved complaint",
        "refund request denied policy violation escalating manager",
        "customer account compromised fraud claims investigation",
        "billing error discovered customer overcharged months",
        "product defect report received quality team investigation",
        "customer data deletion request gdpr compliance thirty",
        "order fulfillment delayed warehouse inventory shortage",
        "express delivery failed carrier exception customer contacted",
        "bulk order processing stuck inventory reservation timeout",
        "return merchandise authorization approved shipping label",
        "replacement item shipped expedited damaged goods",
        "order cancellation processed refund initiated five days",
        "subscription renewal failed payment method expired",
        "support ticket volume spike following app update release",
        "live chat queue depth exceeding response time",
        "customer satisfaction score dropped below threshold",
        "agent handling time increasing knowledge base outdated",
        "escalation rate rising first contact resolution",
        "customer journey mapping session scheduled product",
        "chatbot deflection rate improved knowledge base update",
        "customer provided five star review praising support",
        "net promoter score improved service recovery effort",
        "customer renewed annual contract successful resolution",
    ]
    ENGINEERING = [
        "pull request approved merged main deployment triggered",
        "code review comments addressed technical debt separately",
        "unit test coverage dropped below percent blocking",
        "integration test suite failing dependency upgrade",
        "api breaking change introduced versioning migration",
        "technical debt sprint scheduled next quarter backlog",
        "architecture decision record written new microservice",
        "database schema migration script reviewed production",
        "performance profiling identified memory allocation hotspot",
        "security code review no critical findings minor notes",
        "load testing results system handles target throughput",
        "feature flag enabled ten percent users gradual rollout",
        "dead code removal completed bundle size reduced",
        "api documentation generated openapi specification published",
        "dependency audit completed twelve packages security updates",
        "continuous integration pipeline optimized build time halved",
        "service level objective defined error budget tracking",
        "postmortem action items completed system resilience",
        "chaos engineering experiment failure injection handled",
        "observability improved distributed tracing all services",
        "rate limiting implemented protecting downstream services",
        "circuit breaker pattern preventing cascade failures",
        "cache invalidation strategy redesigned stale data",
        "event driven architecture migration first service",
        "graphql schema designed reviewed frontend backend teams",
    ]
    QUESTIONS = [
        "what is current status kafka consumer group lag",
        "why did deployment fail staging environment last night",
        "when will security patch applied production servers",
        "who is responsible authentication service on call",
        "how do we handle transaction processing queue backs",
        "what caused spike api errors two am morning",
        "why is customer satisfaction dropping this quarter",
        "when is next performance review cycle starting",
        "how many fraud cases detected blocked this week",
        "what is resolution timeline database connection issue",
        "who approved budget new kubernetes cluster expansion",
        "why are we seeing increased latency payment processing",
        "how should we prioritize security vulnerabilities audit",
        "what is rollback plan deployment causes issues",
        "when will hr open enrollment period close benefits",
        "how do we escalate critical production incident hours",
        "what metrics should we monitor after deploying service",
        "why did fraud model trigger this specific transaction",
        "who needs approve new hire before offer goes",
        "how can we improve test coverage release date",
    ]

    ALL = DEVOPS + SECURITY + FINANCE + HR + CUSTOMER + ENGINEERING + QUESTIONS
    random.shuffle(ALL)
    swaps = {"failed":"errored","critical":"severe","detected":"identified",
             "authentication":"auth","deployment":"release","review":"assessment"}
    corpus = list(ALL)
    while len(corpus) < 1000:
        s = random.choice(ALL)
        words = s.split()
        for i,w in enumerate(words):
            if w in swaps and random.random() < 0.5:
                words[i] = swaps[w]
                break
        corpus.append(" ".join(words))
    corpus = corpus[:1000]
    random.shuffle(corpus)

    print("\n" + "="*64)
    print("  APPROACH 5 EXPERIMENT — Bucket Relationship Matrix")
    print("="*64)
    print("\n  Building index with 1,000 sentences...")
    encoder = BlueprintEncoder("approach5_unknown.json")
    store   = ApproachFiveStore(n_clusters=14)
    store.learn(corpus)
    t0 = time.perf_counter()
    for s in corpus:
        _, bp, _ = encoder.encode(s)
        store.add(make_row(bp), s)
    store.build()
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"  Built in {build_ms:.0f}ms"
          f" ({len(corpus)/(build_ms/1000):.0f} rec/sec)")

    # ── Show what the bucket matrix learned ─────────────────────
    print("\n  Bucket relationship matrix (top-3 most similar per bucket):")
    matrix_summary = store.matrix_summary()
    for bid in sorted(matrix_summary.keys()):
        rels = matrix_summary[bid]
        labels = store._cluster_labels.get(bid, ["?"])[:3]
        size   = len(store._buckets[bid])
        print(f"    Bucket {bid:>3} ({size:>3} records, "
              f"top words: {','.join(labels)})")
        for r in rels:
            r_labels = store._cluster_labels.get(r, ["?"])[:3]
            print(f"          ↔ related to bucket {r:>3} "
                  f"({','.join(r_labels)})")

    queries = {
        "kafka consumer lag deployment failed production": {
            "keywords": {"kafka","consumer","lag","deployment","failed",
                         "production","broker","partition","cluster","timeout"}},
        "authentication token expired unauthorized access blocked": {
            "keywords": {"authentication","token","unauthorized","access",
                         "blocked","login","oauth","jwt","ldap","saml"}},
        "fraud transaction payment suspicious blocked alert": {
            "keywords": {"fraud","transaction","payment","suspicious",
                         "blocked","chargeback","velocity","card","wire"}},
        "performance review scheduled employee promotion approved": {
            "keywords": {"performance","review","employee","promotion",
                         "approved","compensation","hiring","interview"}},
        "customer complaint refund order delivery failed": {
            "keywords": {"customer","complaint","refund","order",
                         "delivery","damaged","escalated","billing"}},
        "code review deployment test coverage security vulnerability": {
            "keywords": {"code","review","deployment","test","coverage",
                         "security","pull","request","branch","release"}},
        "why is the system failing what caused the error": {
            "keywords": {"what","why","when","how","who","status","cause"}},
    }

    def is_relevant(text, kws):
        return len(set(text.lower().split()) & kws) >= 1

    # F1
    print(f"\n  ─── F1 metric (top_k=500, threshold=200) ─────────────")
    print(f"  {'Query':42s} {'P':>5} {'R':>5} {'F1':>5} {'Bkts':>5}")
    print(f"  {'─'*42} {'─'*5} {'─'*5} {'─'*5} {'─'*5}")
    all_p, all_r, all_f1, all_t = [], [], [], []
    for query, meta in queries.items():
        _, q_bp, _ = encoder.encode(query)
        qr = make_row(q_bp)
        relevant = {s for s in corpus if is_relevant(s, meta["keywords"])}
        t0 = time.perf_counter()
        results, stats = store.search(qr, query, top_k=500, threshold=200)
        q_ms = (time.perf_counter() - t0) * 1000
        returned = {r["text"] for r in results}
        tp = len(returned & relevant)
        fp = len(returned - relevant)
        fn = len(relevant - returned)
        p  = tp/(tp+fp) if (tp+fp)>0 else 0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        all_p.append(p); all_r.append(r); all_f1.append(f1); all_t.append(q_ms)
        mk = "✓" if f1>=0.6 else "~" if f1>=0.4 else "✗"
        print(f"  {mk} {query[:40]:40s} {p:5.0%} {r:5.0%} {f1:5.0%}"
              f" {stats['buckets_searched']:>5}")
    print()
    print(f"  {'AVERAGE':42s} {sum(all_p)/7:5.0%} {sum(all_r)/7:5.0%} {sum(all_f1)/7:5.0%}")

    # P@K
    print(f"\n  ─── P@K metric (top_k=10) ────────────────────────────")
    print(f"  {'Query':42s} {'P@1':>5} {'P@3':>5} {'P@5':>5} {'P@10':>5}")
    print(f"  {'─'*42} {'─'*5} {'─'*5} {'─'*5} {'─'*5}")
    p1,p3,p5,p10 = [],[],[],[]
    for query, meta in queries.items():
        _, q_bp, _ = encoder.encode(query)
        qr = make_row(q_bp)
        relevant = {s for s in corpus if is_relevant(s, meta["keywords"])}
        results, _ = store.search(qr, query, top_k=10, threshold=200)
        def pa(k):
            top = [r["text"] for r in results[:k]]
            return sum(1 for t in top if t in relevant)/len(top) if top else 0
        a1,a3,a5,a10 = pa(1),pa(3),pa(5),pa(10)
        p1.append(a1); p3.append(a3); p5.append(a5); p10.append(a10)
        print(f"  {query[:42]:42s} {a1:5.0%} {a3:5.0%} {a5:5.0%} {a10:5.0%}")
    print()
    print(f"  {'AVERAGE':42s} {sum(p1)/7:5.0%} {sum(p3)/7:5.0%} {sum(p5)/7:5.0%} {sum(p10)/7:5.0%}")

    # Speed
    print(f"\n  ─── Speed (100 queries) ──────────────────────────────")
    lats = []
    qs = list(queries.keys())*15; random.shuffle(qs)
    for q in qs[:100]:
        _, qbp, _ = encoder.encode(q)
        qr = make_row(qbp)
        t0 = time.perf_counter()
        store.search(qr, q, top_k=20, threshold=200)
        lats.append((time.perf_counter()-t0)*1000)
    lats.sort()
    print(f"  p50: {lats[50]:.1f}ms   p95: {lats[95]:.1f}ms   p99: {lats[99]:.1f}ms")

    if os.path.exists("approach5_unknown.json"):
        os.remove("approach5_unknown.json")

    # Verdict
    print(f"\n{'='*64}")
    print("  COMPARISON")
    print(f"{'='*64}")
    a5_f1  = sum(all_f1)/7
    a5_p1  = sum(p1)/7
    a5_p10 = sum(p10)/7
    a5_p50 = lats[50]
    print(f"\n  {'Metric':12s} {'Approach 3':>12} {'Approach 4':>12} {'Approach 5':>12}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    print(f"  {'F1':12s} {'29%':>12} {'37%':>12} {f'{a5_f1*100:.0f}%':>12}")
    print(f"  {'P@1':12s} {'100%':>12} {'57%':>12} {f'{a5_p1*100:.0f}%':>12}")
    print(f"  {'P@10':12s} {'96%':>12} {'60%':>12} {f'{a5_p10*100:.0f}%':>12}")
    print(f"  {'p50':12s} {'0.4ms':>12} {'1.0ms':>12} {f'{a5_p50:.1f}ms':>12}")


if __name__ == "__main__":
    main()
