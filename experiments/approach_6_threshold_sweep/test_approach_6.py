"""
Approach 6 — Threshold Sweep
============================

GOAL
----
The Approach 3 baseline uses threshold=200. Question: what happens
to F1 / P@K / speed if we raise the threshold? Lower it?

Test a range of values and see the full curve. The best threshold
for our actual data may not be what we hand-picked.

ISOLATION
---------
Does NOT touch src/scbs/. Uses Approach 3 live code directly with
different threshold values passed at search time. No code change
in the package.
"""
import sys, os, time, random

# Live package — read-only imports
_HERE = os.path.dirname(__file__)
_SRC  = os.path.join(_HERE, "..", "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from scbs import Encoder, Store

random.seed(42)


def main():
    # Same corpus as every previous benchmark — apples to apples
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

    queries = {
        "kafka consumer lag deployment failed production":            {"keywords":{"kafka","consumer","lag","deployment","failed","production","broker","partition","cluster","timeout"}},
        "authentication token expired unauthorized access blocked":   {"keywords":{"authentication","token","unauthorized","access","blocked","login","oauth","jwt","ldap","saml"}},
        "fraud transaction payment suspicious blocked alert":          {"keywords":{"fraud","transaction","payment","suspicious","blocked","chargeback","velocity","card","wire"}},
        "performance review scheduled employee promotion approved":    {"keywords":{"performance","review","employee","promotion","approved","compensation","hiring","interview"}},
        "customer complaint refund order delivery failed":             {"keywords":{"customer","complaint","refund","order","delivery","damaged","escalated","billing"}},
        "code review deployment test coverage security vulnerability": {"keywords":{"code","review","deployment","test","coverage","security","pull","request","branch","release"}},
        "why is the system failing what caused the error":             {"keywords":{"what","why","when","how","who","status","cause"}},
    }

    def is_relevant(text, kws):
        return len(set(text.lower().split()) & kws) >= 1

    # Build index ONCE — threshold is a search-time parameter
    print("\n" + "="*68)
    print("  APPROACH 6 — THRESHOLD SWEEP")
    print("="*68)
    print("\n  Building Approach 3 index with 1000 sentences...")
    encoder = Encoder()
    store   = Store(n_clusters=14)
    for s in corpus:
        store.add(encoder.encode(s), s)
    store.build(corpus_texts=corpus)
    print(f"  Built. Now sweeping threshold values.")

    # Sweep across a wide threshold range
    thresholds = [50, 100, 200, 300, 500, 800, 1500, 3000, 9000]

    print(f"\n{'─'*68}")
    print(f"  {'thresh':>7}  {'F1':>6}  {'P':>6}  {'R':>6}  "
          f"{'P@1':>6}  {'P@5':>6}  {'P@10':>6}  {'p50':>7}")
    print(f"  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}  "
          f"{'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}")

    results_table = []
    for th in thresholds:
        all_p, all_r, all_f1 = [], [], []
        p1, p5, p10 = [], [], []
        lats = []

        for query, meta in queries.items():
            qr = encoder.encode(query)
            relevant = {s for s in corpus if is_relevant(s, meta["keywords"])}

            # F1 measurement (top_k=500, threshold=th)
            t0 = time.perf_counter()
            results, _ = store.search(qr, query, top_k=500, threshold=th)
            q_ms = (time.perf_counter() - t0) * 1000
            lats.append(q_ms)

            returned = {r["text"] for r in results}
            tp = len(returned & relevant)
            fp = len(returned - relevant)
            fn = len(relevant - returned)
            p  = tp/(tp+fp) if (tp+fp)>0 else 0
            r  = tp/(tp+fn) if (tp+fn)>0 else 0
            f1 = 2*p*r/(p+r) if (p+r)>0 else 0
            all_p.append(p); all_r.append(r); all_f1.append(f1)

            # P@K measurement
            top10 = results[:10]
            top10_texts = [r["text"] for r in top10]
            def pa(k):
                t = top10_texts[:k]
                return sum(1 for x in t if x in relevant)/len(t) if t else 0
            p1.append(pa(1)); p5.append(pa(5)); p10.append(pa(10))

        # Averages
        avg_f1  = sum(all_f1)/7
        avg_p   = sum(all_p)/7
        avg_r   = sum(all_r)/7
        avg_p1  = sum(p1)/7
        avg_p5  = sum(p5)/7
        avg_p10 = sum(p10)/7
        avg_lat = sum(lats)/7

        results_table.append({
            "threshold": th, "f1": avg_f1, "p": avg_p, "r": avg_r,
            "p1": avg_p1, "p5": avg_p5, "p10": avg_p10, "lat": avg_lat,
        })

        print(f"  {th:>7}  {avg_f1*100:5.0f}%  {avg_p*100:5.0f}%  "
              f"{avg_r*100:5.0f}%  {avg_p1*100:5.0f}%  "
              f"{avg_p5*100:5.0f}%  {avg_p10*100:5.0f}%  "
              f"{avg_lat:5.1f}ms")

    # Find best on each metric
    best_f1   = max(results_table, key=lambda x: x["f1"])
    best_p1   = max(results_table, key=lambda x: x["p1"])
    best_p10  = max(results_table, key=lambda x: x["p10"])
    best_bal  = max(results_table, key=lambda x: x["f1"] + x["p10"]/2)

    print(f"\n{'='*68}")
    print(f"  BEST THRESHOLD PER METRIC")
    print(f"{'='*68}")
    print(f"\n  Best F1 ({best_f1['f1']*100:.0f}%): threshold={best_f1['threshold']}")
    print(f"  Best P@1 ({best_p1['p1']*100:.0f}%): threshold={best_p1['threshold']}")
    print(f"  Best P@10 ({best_p10['p10']*100:.0f}%): threshold={best_p10['threshold']}")
    print(f"  Best balanced (F1 + P@10/2): threshold={best_bal['threshold']}")

    print(f"\n  Approach 3 baseline (threshold=200):")
    base = next(x for x in results_table if x["threshold"] == 200)
    print(f"    F1={base['f1']*100:.0f}%  P@1={base['p1']*100:.0f}%  P@10={base['p10']*100:.0f}%  p50={base['lat']:.1f}ms")


if __name__ == "__main__":
    main()
