"""
Domain Voting (Approach 3)
===========================
Smart domain detection at encode time, not search time.

Each known word in the V2 vocabulary "votes" for the domain
its sub-cluster belongs to. A sentence's domain is the
majority winner among its content-word votes.

The voting infrastructure is precomputed once at module load.
Per-comparison cost is one hash lookup and one slot-weight
table swap — no inner-loop overhead.

This keeps Approach 3 within the 20% latency budget.
"""

from collections import Counter
from .vocabulary import SEMANTIC_VOCAB_V2, id_to_subcluster


# ──────────────────────────────────────────────────────────────
# 1. SUB-CLUSTER → DOMAIN MAPPING
# ──────────────────────────────────────────────────────────────

# Aggregate fine-grained sub-clusters into broad voting domains.
# Each domain has its own slot priority profile.
DOMAIN_BUCKETS = {
    "Finance":  ["Financial Operations"],
    "HR":       ["HR Operations"],
    "Customer": ["Customer Operations"],
    "Security": ["Security Events"],
    "Incident": ["Incident Management"],
    "DevOps":   ["Tech: Messaging & Streaming",
                 "Tech: Container & Orchestration",
                 "Tech: Monitoring",
                 "Tech: Cloud & Infra"],
    "Code":     ["Code: Languages", "Code: Constructs",
                 "Code: Execution", "Code: Algorithms",
                 "Code: Quality",   "Code: Version Control"],
    "Data":     ["Tech: Database & Storage",
                 "Data: File Ops",  "Data: Database Ops",
                 "Data: Processing","Data: Sync & Backup"],
    "AI":       ["Tech: AI & ML", "Tech: Networking & API"],
    "Question": ["Questions & Inquiry"],
}


# ──────────────────────────────────────────────────────────────
# 2. PRECOMPUTED VOTING TABLE (built once at import)
# ──────────────────────────────────────────────────────────────
# word → list of domains it votes for

WORD_VOTES: dict = {}

def _build_word_votes() -> None:
    for word, cid in SEMANTIC_VOCAB_V2.items():
        sc = id_to_subcluster(cid)
        for domain, subs in DOMAIN_BUCKETS.items():
            if sc in subs:
                WORD_VOTES.setdefault(word, []).append(domain)

_build_word_votes()


# ──────────────────────────────────────────────────────────────
# 3. DOMAIN-SPECIFIC SLOT WEIGHTS
# ──────────────────────────────────────────────────────────────
# Slot indices: 0:WHO 1:ACTION 2:TECH 3:EMOTION 4:WHEN
#               5:SOCIAL 6:INTENT 7:MODIFIER 8:WORLD 9:DOMAIN

DEFAULT_WEIGHTS = {i: 1.0 for i in range(10)}

DOMAIN_SLOT_WEIGHTS = {
    "Finance":  {0:1.0, 1:2.0, 2:1.5, 3:0.3, 4:1.0,
                 5:0.5, 6:1.0, 7:2.0, 8:1.0, 9:5.0},
    "HR":       {0:3.0, 1:2.0, 2:0.5, 3:1.0, 4:2.0,
                 5:1.0, 6:1.0, 7:1.0, 8:0.5, 9:5.0},
    "Customer": {0:3.0, 1:2.5, 2:0.5, 3:2.0, 4:1.0,
                 5:1.0, 6:1.0, 7:1.5, 8:0.5, 9:5.0},
    "Security": {0:1.0, 1:3.0, 2:4.0, 3:1.0, 4:1.0,
                 5:0.3, 6:1.0, 7:4.0, 8:0.5, 9:5.0},
    "Incident": {0:1.0, 1:3.0, 2:4.0, 3:2.0, 4:2.0,
                 5:0.3, 6:1.0, 7:3.0, 8:0.5, 9:5.0},
    "DevOps":   {0:0.5, 1:3.0, 2:5.0, 3:1.0, 4:1.5,
                 5:0.2, 6:0.5, 7:2.0, 8:1.0, 9:3.0},
    "Code":     {0:1.0, 1:2.5, 2:5.0, 3:0.5, 4:1.0,
                 5:0.2, 6:0.5, 7:2.5, 8:1.0, 9:1.0},
    "Data":     {0:0.5, 1:3.0, 2:5.0, 3:0.3, 4:1.0,
                 5:0.2, 6:0.5, 7:2.0, 8:1.5, 9:2.0},
    "AI":       {0:1.0, 1:2.0, 2:5.0, 3:0.5, 4:1.0,
                 5:0.3, 6:1.0, 7:2.5, 8:1.0, 9:1.5},
    "Question": {0:1.5, 1:1.5, 2:2.0, 3:0.5, 4:1.5,
                 5:0.5, 6:5.0, 7:1.5, 8:1.0, 9:1.5},
    "general":  DEFAULT_WEIGHTS,
}


# ──────────────────────────────────────────────────────────────
# 4. PUBLIC API
# ──────────────────────────────────────────────────────────────

def compute_domain_hint(text: str) -> str:
    """
    Return the dominant domain for a sentence based on
    majority vote among its content words.

    Called once per record at encode time and once per query
    at search time. Cost: linear scan of words, one dict
    lookup per word. Negligible overhead.

    Args:
        text: original sentence (already lowercased internally)

    Returns:
        Name of the winning domain, or "general" if no
        content words vote.
    """
    votes = Counter()
    for word in text.lower().split():
        word = word.strip(".,!?;:\"'()")
        if word in WORD_VOTES:
            for domain in WORD_VOTES[word]:
                votes[domain] += 1

    if not votes:
        return "general"
    return votes.most_common(1)[0][0]


def weights_for_pair(domain_a: str, domain_b: str) -> dict:
    """
    Pick slot weights for a comparison.

    Both same domain → use that domain's weights
    Different domains → use 'general' (neutral) weights
    Either is general → use 'general' weights

    Args:
        domain_a, domain_b: domain hints for the two records

    Returns:
        dict mapping slot index to weight multiplier
    """
    if domain_a == domain_b and domain_a != "general":
        return DOMAIN_SLOT_WEIGHTS[domain_a]
    return DEFAULT_WEIGHTS
