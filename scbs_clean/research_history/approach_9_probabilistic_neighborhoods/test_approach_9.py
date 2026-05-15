"""
Approach 9 — Overlapping Probabilistic Neighborhoods
=====================================================

CONCEPT
-------
Replace hard bucket assignment (each doc in exactly 1 bucket) with
soft neighborhood membership (each doc belongs to 3-5 neighborhoods
with varying membership strengths).

At index time:
  - Compute document similarity to all 14 cluster centroids
  - Assign document to top-3 clusters with membership scores
  - Store: doc_id → [(cluster_id, membership_score), ...]

At search time:
  - Query computes its top-3 neighborhoods
  - Search ALL documents with membership in ANY of those neighborhoods
  - Rank by: base_distance × (1 / avg_membership_strength)

WHY THIS FIXES RECALL
----------------------
Hard buckets: document in bucket A is invisible to bucket B queries
Soft neighborhoods: document with 60% in A, 30% in B is reachable from both

A document at the boundary between DevOps and Security can now be
found by queries from either domain.

ISOLATION
---------
Does NOT modify src/scbs/. Uses Approach 3 read-only and adds the
soft membership layer on top.
"""
import sys, os, time, random, math
from collections import defaultdict, Counter

_HERE = os.path.dirname(__file__)
_SRC  = os.path.join(_HERE, "..", "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from scbs.blueprint import BlueprintEncoder
from scbs.matrix_index import make_row, make_signature, shared_slots, ZoneIndex
from scbs.distance import (
    build_idf, sentence_weight, weighted_matrix_distance,
)
from scbs.domain_voting import compute_domain_hint, weights_for_pair
from scbs.clustering import build_cooccurrence, cluster_words, get_cluster_labels

random.seed(42)


# ════════════════════════════════════════════════════════════════
#  PROBABILISTIC NEIGHBORHOOD ASSIGNMENT
# ════════════════════════════════════════════════════════════════

def compute_cluster_centroids(corpus, word_clusters, n_clusters):
    """
    Compute representative word sets for each cluster.
    Centroid = top 20 most common words in that cluster's documents.
    """
    cluster_words = defaultdict(Counter)
    
    for text in corpus:
        words = [w.lower().strip(".,!?;:\"'()") for w in text.split() if len(w) > 2]
        word_counter = Counter(words)
        
        # Find which cluster this document belongs to (hard assignment for centroid building)
        cluster_votes = Counter()
        for word in word_counter:
            if word in word_clusters:
                cluster_votes[word_clusters[word]] += word_counter[word]
        
        if cluster_votes:
            primary_cluster = cluster_votes.most_common(1)[0][0]
            for word, count in word_counter.items():
                cluster_words[primary_cluster][word] += count
    
    # Keep top 20 words per cluster as centroid
    centroids = {}
    for cid in range(n_clusters):
        if cid in cluster_words:
            centroids[cid] = set(w for w, _ in cluster_words[cid].most_common(20))
        else:
            centroids[cid] = set()
    centroids[-1] = set()  # general bucket has empty centroid
    
    return centroids


def compute_neighborhood_memberships(text, centroids, top_k=3):
    """
    Compute membership scores for text in all clusters.
    Returns top-K clusters with scores.
    
    Score = Jaccard similarity between text words and cluster centroid.
    """
    words = set(w.lower().strip(".,!?;:\"'()") for w in text.split() if len(w) > 2)
    
    scores = []
    for cid, centroid_words in centroids.items():
        if cid == -1:  # skip general bucket
            continue
        if not centroid_words or not words:
            jaccard = 0.0
        else:
            intersection = len(words & centroid_words)
            union = len(words | centroid_words)
            jaccard = intersection / union if union > 0 else 0.0
        scores.append((cid, jaccard))
    
    # Sort by score descending, keep top-K
    scores.sort(key=lambda x: -x[1])
    top = scores[:top_k]
    
    # Normalize scores to sum to 1 (softmax-like)
    total = sum(s for _, s in top)
    if total > 0:
        top = [(cid, s / total) for cid, s in top]
    
    # Always include general bucket with small weight
    top.append((-1, 0.05))
    
    return top


# ════════════════════════════════════════════════════════════════
#  STORE WITH PROBABILISTIC NEIGHBORHOODS
# ════════════════════════════════════════════════════════════════

class ProbabilisticStore:
    
    def __init__(self, n_clusters=14, top_k_neighborhoods=3):
        self.n_clusters = n_clusters
        self.top_k_neighborhoods = top_k_neighborhoods
        
        self._idf = {}
        self._word_clusters = {}
        self._cluster_labels = {}
        self._centroids = {}
        
        # Record storage: each record knows its neighborhoods
        self._all_records = {}
        self._next_id = 0
        
        # Inverted index: cluster_id → list of (record_id, membership_score)
        self._neighborhood_members = defaultdict(list)
        
        # Zone index per cluster (same as Approach 3)
        self._zones = {}
        self._sigs = {}
        for c in list(range(n_clusters)) + [-1]:
            self._zones[c] = ZoneIndex()
            self._sigs[c] = []
    
    def learn(self, sentences):
        self._idf = build_idf(sentences)
        cooc = build_cooccurrence(sentences)
        self._word_clusters = cluster_words(cooc, self.n_clusters)
        self._cluster_labels = get_cluster_labels(
            self._word_clusters, cooc, self.n_clusters
        )
        self._centroids = compute_cluster_centroids(
            sentences, self._word_clusters, self.n_clusters
        )
    
    def add(self, row, text):
        rid = self._next_id
        self._next_id += 1
        
        # Compute neighborhood memberships
        memberships = compute_neighborhood_memberships(
            text, self._centroids, top_k=self.top_k_neighborhoods
        )
        
        w = sentence_weight(text, self._idf)
        domain = compute_domain_hint(text)
        sig = make_signature(row)
        
        self._all_records[rid] = {
            "id": rid, "text": text, "row": row,
            "weight": w, "domain": domain,
            "memberships": memberships,  # list of (cluster_id, score)
        }
        
        # Add to inverted index for each neighborhood
        for cid, score in memberships:
            self._neighborhood_members[cid].append((rid, score))
            
            # Also add to zone index for that cluster
            local_idx = len(self._sigs[cid])
            self._sigs[cid].append(sig)
            self._zones[cid].add(local_idx, row)
    
    def build(self):
        for zone in self._zones.values():
            zone.build()
    
    def search(self, query_row, query_text, *,
               top_k=10, threshold=200.0, zone_radius=150,
               excluded=None):
        
        if excluded is None:
            excluded = set()
        
        # Compute query's neighborhoods
        q_memberships = compute_neighborhood_memberships(
            query_text, self._centroids, top_k=self.top_k_neighborhoods
        )
        q_clusters = set(cid for cid, _ in q_memberships)
        q_membership_map = dict(q_memberships)
        
        q_weight = sentence_weight(query_text, self._idf)
        q_domain = compute_domain_hint(query_text)
        q_sig = make_signature(query_row)
        
        stats = {
            "query_neighborhoods": sorted(q_clusters),
            "candidates_from_neighborhoods": 0,
            "distance_checks": 0,
        }
        
        # Collect ALL documents with membership in query's neighborhoods
        candidate_rids = set()
        for cid in q_clusters:
            for rid, member_score in self._neighborhood_members[cid]:
                if rid not in excluded:
                    candidate_rids.add(rid)
        
        stats["candidates_from_neighborhoods"] = len(candidate_rids)
        
        # Score candidates with NEW 5-component formula
        scored = []
        for rid in candidate_rids:
            rec = self._all_records[rid]
            row = rec["row"]
            text = rec["text"]
            rec_weight = rec["weight"]
            rec_domain = rec["domain"]
            
            # Convert rows to dicts for easier access
            q_dict = {slot: val for slot, val in query_row}
            r_dict = {slot: val for slot, val in row}
            
            q_slots = set(q_dict.keys())
            r_slots = set(r_dict.keys())
            shared = q_slots & r_slots
            
            if not shared:
                continue  # no alignment possible
            
            # ═══ COMPONENT 1: semantic_alignment ═══
            # Gaussian similarity on shared slots, weighted by domain
            slot_weights = weights_for_pair(q_domain, rec_domain)
            alignment = 0.0
            for slot in shared:
                distance = abs(q_dict[slot] - r_dict[slot])
                similarity = math.exp(-(distance * distance) / (15 * 15))  # sigma=15
                weight = slot_weights.get(slot, 1.0)
                alignment += similarity * weight
            alignment /= max(len(shared), 1)  # normalize by shared slots
            
            # ═══ COMPONENT 2: rare_term_gain ═══
            # Boost for matching rare terms (high IDF)
            q_words = set(query_text.lower().split())
            r_words = set(text.lower().split())
            matched_words = q_words & r_words
            rare_gain = sum(self._idf.get(w, 0.0) for w in matched_words)
            rare_gain = min(rare_gain / 10.0, 1.0)  # normalize to [0, 1]
            
            # ═══ COMPONENT 3: interaction_strength ═══
            # How strongly do the filled slots interact?
            # Strong interaction = many shared slots with close values
            interaction = len(shared) / max(len(q_slots | r_slots), 1)
            interaction *= (1.0 - sum(abs(q_dict[s] - r_dict[s]) 
                                       for s in shared) / (len(shared) * 200.0))
            interaction = max(0.0, min(interaction, 1.0))
            
            # ═══ COMPONENT 4: directional_activation ═══
            # Does query's primary concept activate document's primary?
            # Primary = slot with lowest cluster ID (most important in vocab order)
            if shared:
                q_primary_slot = min(shared, key=lambda s: q_dict[s])
                q_primary_val = q_dict[q_primary_slot]
                r_primary_val = r_dict[q_primary_slot]
                activation = math.exp(-abs(q_primary_val - r_primary_val) / 10.0)
            else:
                activation = 0.0
            
            # ═══ COMPONENT 5: semantic_conflict ═══
            # Penalty for contradictory signals in same slot
            # Large distance in important slot = conflict
            conflict = 0.0
            for slot in shared:
                distance = abs(q_dict[slot] - r_dict[slot])
                if distance > 100:  # far apart in cluster space
                    weight = slot_weights.get(slot, 1.0)
                    conflict += (distance / 200.0) * weight
            conflict = min(conflict / max(len(shared), 1), 1.0)
            
            # ═══ FINAL SCORE ═══
            # Higher is better
            final_score = (
                alignment * 2.0           # semantic alignment (most important)
                + rare_gain * 1.5         # rare term matching
                + interaction * 1.0       # slot interaction strength
                + activation * 0.5        # directional activation
                - conflict * 1.0          # semantic conflict penalty
            )
            
            stats["distance_checks"] += 1
            
            scored.append({
                "id": rid, "text": text,
                "score": round(final_score, 3),
                "alignment": round(alignment, 2),
                "rare_gain": round(rare_gain, 2),
                "interaction": round(interaction, 2),
                "activation": round(activation, 2),
                "conflict": round(conflict, 2),
                "domain": rec_domain,
            })
        
        # Sort by score DESCENDING (higher = better)
        scored.sort(key=lambda r: -r["score"])
        results = scored[:top_k]
        for r in results:
            excluded.add(r["id"])
        
        return results, stats


# ════════════════════════════════════════════════════════════════
#  BENCHMARK ON 1K CORPUS
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

    print("\n" + "="*68)
    print("  APPROACH 9 — PROBABILISTIC NEIGHBORHOODS")
    print("  Soft membership replaces hard buckets")
    print("="*68)
    print("\n  Building index with 1,000 sentences...")
    encoder = BlueprintEncoder("approach9_unknown.json")
    store = ProbabilisticStore(n_clusters=14, top_k_neighborhoods=3)
    store.learn(corpus)
    
    t0 = time.perf_counter()
    for s in corpus:
        _, bp, _ = encoder.encode(s)
        store.add(make_row(bp), s)
    store.build()
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"  Built in {build_ms:.0f}ms ({len(corpus)/(build_ms/1000):.0f} rec/sec)")

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

    # F1 at top_k=500
    print(f"\n  ─── F1 metric (top_k=500) ──────────────────────────")
    print(f"  {'Query':42s} {'P':>5} {'R':>5} {'F1':>5}")
    print(f"  {'─'*42} {'─'*5} {'─'*5} {'─'*5}")
    all_p, all_r, all_f1 = [], [], []
    for query, meta in queries.items():
        _, q_bp, _ = encoder.encode(query)
        qr = make_row(q_bp)
        relevant = {s for s in corpus if is_relevant(s, meta["keywords"])}
        results, _ = store.search(qr, query, top_k=500, threshold=9000)
        returned = {r["text"] for r in results}
        tp = len(returned & relevant)
        fp = len(returned - relevant)
        fn = len(relevant - returned)
        p  = tp/(tp+fp) if (tp+fp)>0 else 0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        all_p.append(p); all_r.append(r); all_f1.append(f1)
        mk = "✓" if f1>=0.6 else "~" if f1>=0.4 else "✗"
        print(f"  {mk} {query[:40]:40s} {p:5.0%} {r:5.0%} {f1:5.0%}")
    print(f"\n  {'AVERAGE':42s} {sum(all_p)/7:5.0%} {sum(all_r)/7:5.0%} {sum(all_f1)/7:5.0%}")

    # P@K
    print(f"\n  ─── P@K metric (top_k=10) ──────────────────────────")
    print(f"  {'Query':42s} {'P@1':>5} {'P@3':>5} {'P@5':>5} {'P@10':>5}")
    print(f"  {'─'*42} {'─'*5} {'─'*5} {'─'*5} {'─'*5}")
    p1, p3, p5, p10 = [], [], [], []
    for query, meta in queries.items():
        _, q_bp, _ = encoder.encode(query)
        qr = make_row(q_bp)
        relevant = {s for s in corpus if is_relevant(s, meta["keywords"])}
        results, _ = store.search(qr, query, top_k=10, threshold=9000)
        def pa(k):
            top = [r["text"] for r in results[:k]]
            return sum(1 for t in top if t in relevant)/len(top) if top else 0
        a1,a3,a5,a10 = pa(1),pa(3),pa(5),pa(10)
        p1.append(a1); p3.append(a3); p5.append(a5); p10.append(a10)
        print(f"  {query[:42]:42s} {a1:5.0%} {a3:5.0%} {a5:5.0%} {a10:5.0%}")
    print(f"\n  {'AVERAGE':42s} {sum(p1)/7:5.0%} {sum(p3)/7:5.0%} {sum(p5)/7:5.0%} {sum(p10)/7:5.0%}")

    # Speed
    print(f"\n  ─── Speed (100 queries) ──────────────────────────")
    lats = []
    qs = list(queries.keys())*15; random.shuffle(qs)
    for q in qs[:100]:
        _, qbp, _ = encoder.encode(q)
        qr = make_row(qbp)
        t0 = time.perf_counter()
        store.search(qr, q, top_k=20, threshold=9000)
        lats.append((time.perf_counter()-t0)*1000)
    lats.sort()
    print(f"  p50: {lats[50]:.1f}ms   p95: {lats[95]:.1f}ms   p99: {lats[99]:.1f}ms")

    if os.path.exists("approach9_unknown.json"):
        os.remove("approach9_unknown.json")

    # Comparison
    print(f"\n{'='*68}")
    print("  COMPARISON")
    print(f"{'='*68}")
    a9_f1  = sum(all_f1)/7
    a9_p1  = sum(p1)/7
    a9_p10 = sum(p10)/7
    a9_p50 = lats[50]
    print(f"\n  {'Metric':12s} {'Approach 3':>14} {'Approach 9':>14} {'Δ':>10}")
    print(f"  {'─'*12} {'─'*14} {'─'*14} {'─'*10}")
    print(f"  {'F1':12s} {'29%':>14} {f'{a9_f1*100:.0f}%':>14} {(a9_f1*100-29):+.0f}pp")
    print(f"  {'Recall':12s} {'23%':>14} {f'{sum(all_r)/7*100:.0f}%':>14} {(sum(all_r)/7*100-23):+.0f}pp")
    print(f"  {'P@1':12s} {'100%':>14} {f'{a9_p1*100:.0f}%':>14} {(a9_p1*100-100):+.0f}pp")
    print(f"  {'P@10':12s} {'96%':>14} {f'{a9_p10*100:.0f}%':>14} {(a9_p10*100-96):+.0f}pp")
    print(f"  {'p50':12s} {'0.4ms':>14} {f'{a9_p50:.1f}ms':>14} {(a9_p50-0.4):+.1f}ms")


if __name__ == "__main__":
    main()
