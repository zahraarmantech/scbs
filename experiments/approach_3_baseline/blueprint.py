"""
Blueprint Extractor — SCBS
===========================
Extracts a sparse slot record from any encoded sentence during
the encoding pass itself. Zero extra cost — no second pass.

The blueprint is a dict {slot_index: cluster_id} where:
  slot 0 — WHO        (People & Pronouns)
  slot 1 — ACTION     (Actions & Verbs)
  slot 2 — TECH       (Technology + Programming + Data & Storage)
  slot 3 — EMOTION    (Positive + Negative Emotions)
  slot 4 — WHEN       (Time & Dates)
  slot 5 — SOCIAL     (Greetings + Farewells)
  slot 6 — INTENT     (Questions & Inquiry)
  slot 7 — MODIFIER   (Adjectives & Descriptors)
  slot 8 — CONNECTOR  (Connectors + Prepositions)
  slot 9 — WORLD      (Nature + Colors + Numbers + Food)

Rules:
- Each slot holds the FIRST semantically significant token
  that maps to it (most specific wins if tie)
- Empty slots are absent from the dict (sparse — no padding)
- Blueprint is stored as "bp" field alongside "e" in NDJSON
- Comparison is always O(k) where k = filled slots (small, consistent)

Distance formula:
  For each slot present in either blueprint:
    if both have it  → |A[slot] - B[slot]|
    if only one has  → penalty (configurable, default 200)
  Average over all compared slots.
"""

import json
import math
import os
import hashlib
from datetime import datetime, timezone
from collections import defaultdict

from .encoder import (
    SemanticEncoder,
    id_to_cluster,
    CLUSTER_RANGES,
    semantic_distance as distance_1d,
    derive_sem_group,
    derive_sem_score,
)


# ==============================================================
# SECTION 1: SLOT MAP
# ==============================================================

# Cluster name → slot index
# Multiple clusters can map to the same slot (semantic family)
CLUSTER_TO_SLOT = {
    # WHO — slot 0
    "People & Pronouns":               0,
    "People: Pronouns":                0,
    "People: Tech Roles":              0,
    "People: Business Roles":          0,
    "People: General":                 0,
    # ACTION — slot 1
    "Actions & Verbs":                 1,
    "Actions: Communication":          1,
    "Actions: Creation":               1,
    "Actions: Control":                1,
    "Actions: Movement":               1,
    "Actions: Analysis":               1,
    "Actions: Change":                 1,
    "Actions: Access":                 1,
    "Actions: Management":             1,
    "Actions: Connection":             1,
    # TECH — slot 2
    "Technology":                      2,
    "Programming":                     2,
    "Data & Storage":                  2,
    "Tech: Messaging & Streaming":     2,
    "Tech: Container & Orchestration": 2,
    "Tech: Database & Storage":        2,
    "Tech: Networking & API":          2,
    "Tech: AI & ML":                   2,
    "Tech: Cloud & Infra":             2,
    "Tech: Monitoring":                2,
    "Code: Languages":                 2,
    "Code: Constructs":                2,
    "Code: Execution":                 2,
    "Code: Algorithms":                2,
    "Code: Quality":                   2,
    "Code: Version Control":           2,
    "Data: File Ops":                  2,
    "Data: Database Ops":              2,
    "Data: Processing":                2,
    "Data: Sync & Backup":             2,
    # EMOTION — slot 3
    "Positive Emotions":               3,
    "Negative Emotions":               3,
    # WHEN — slot 4
    "Time & Dates":                    4,
    "Time: Units":                     4,
    "Time: Relative":                  4,
    "Time: Direction":                 4,
    "Time: Scheduling":                4,
    # SOCIAL — slot 5
    "Greetings":                       5,
    "Farewells":                       5,
    # INTENT — slot 6
    "Questions & Inquiry":             6,
    # MODIFIER — slot 7
    "Adjectives & Descriptors":        7,
    "Adj: Size & Scale":               7,
    "Adj: Speed & Performance":        7,
    "Adj: Quality":                    7,
    "Adj: State":                      7,
    "Adj: Security & Risk":            7,
    "Adj: Technical":                  7,
    # WORLD — slot 8
    "Nature & World":                  8,
    "Colors":                          8,
    "Numbers & Math":                  8,
    "Food & Drink":                    8,
    # DOMAIN — slot 9 (new)
    "Security Events":                 9,
    "Incident Management":             9,
    "Financial Operations":            9,
    "HR Operations":                   9,
    "Customer Operations":             9,
}

SLOT_NAMES = {
    0: "WHO",
    1: "ACTION",
    2: "TECH",
    3: "EMOTION",
    4: "WHEN",
    5: "SOCIAL",
    6: "INTENT",
    7: "MODIFIER",
    8: "WORLD",
    9: "DOMAIN",
}

# Specificity rank per cluster — higher = more specific = wins tie
CLUSTER_SPECIFICITY_RANK = {
    # V1 names
    "Technology":               5,
    "Programming":              5,
    "Data & Storage":           4,
    "Negative Emotions":        4,
    "Positive Emotions":        3,
    "Questions & Inquiry":      3,
    # V2 sub-cluster names — domain-specific slots rank highest
    "Security Events":          9,
    "Incident Management":      9,
    "Financial Operations":     9,
    "HR Operations":            9,
    "Customer Operations":      9,
    "Tech: Messaging & Streaming":     7,
    "Tech: Container & Orchestration": 7,
    "Tech: Database & Storage":        7,
    "Tech: Networking & API":          7,
    "Tech: AI & ML":                   7,
    "Tech: Cloud & Infra":             6,
    "Tech: Monitoring":                6,
    "Code: Languages":                 6,
    "Code: Constructs":                5,
    "Code: Execution":                 5,
    "Code: Algorithms":                5,
    "Code: Quality":                   5,
    "Code: Version Control":           5,
    "Data: File Ops":                  4,
    "Data: Database Ops":              5,
    "Data: Processing":                4,
    "Data: Sync & Backup":             4,
    "People: Tech Roles":              4,
    "People: Business Roles":          3,
    "People: General":                 2,
    "People: Pronouns":                1,
    "Adj: State":                      4,
    "Adj: Security & Risk":            5,
    "Adj: Technical":                  4,
    "Adj: Speed & Performance":        3,
    "Adj: Quality":                    3,
    "Adj: Size & Scale":               2,
    "Time: Scheduling":                4,
    "Time: Units":                     2,
    "Time: Relative":                  2,
    "Time: Direction":                 2,
    "Actions: Access":                 5,
    "Actions: Analysis":               4,
    "Actions: Management":             3,
    "Actions: Communication":          3,
    "Actions: Change":                 3,
    "Actions: Control":                3,
    "Actions: Creation":               3,
    "Actions: Movement":               3,
    "Actions: Connection":             3,
    "Actions & Verbs":          3,
    "People & Pronouns":        2,
    "Time & Dates":             2,
    "Adjectives & Descriptors": 2,
    "Greetings":                2,
    "Farewells":                2,
    "Nature & World":           1,
    "Colors":                   1,
    "Numbers & Math":           1,
    "Food & Drink":             1,
    "Connectors & Conjunctions":0,
    "Prepositions & Articles":  0,
    "Miscellaneous":            0,
}

# Missing slot penalty — used when one blueprint has a slot
# and the other doesn't. Tunable.
MISSING_SLOT_PENALTY = 200


# ==============================================================
# SECTION 2: BLUEPRINT EXTRACTOR
# ==============================================================

class BlueprintExtractor:
    """
    Extracts a sparse slot record from an encoded token stream.
    Runs in a single pass — O(n) where n = number of tokens,
    which is the same pass as encoding itself.
    """

    def extract(self, trace):
        """
        Build blueprint from the encoder's trace output.

        trace  — list of step dicts from SemanticEncoder.encode()
        returns dict {slot_index: cluster_id}
        """
        blueprint = {}
        slot_rank  = {}

        for step in trace:
            cid     = step["id"]
            cluster = step["cluster"]

            # Skip low-tier, ASCII and miscellaneous
            if cluster in ("Low-Tier", "ASCII", "Miscellaneous"):
                continue

            # Skip any surface token shorter than 4 chars —
            # these are greedy split fragments (ging, ure, nt, me)
            # that carry no real slot meaning regardless of cluster
            if len(step["surface"].strip()) < 4:
                continue

            slot = CLUSTER_TO_SLOT.get(cluster)
            if slot is None:
                continue

            rank = CLUSTER_SPECIFICITY_RANK.get(cluster, 0)

            if slot not in blueprint or rank > slot_rank[slot]:
                blueprint[slot]   = cid
                slot_rank[slot]   = rank

        return blueprint

    def slot_names(self, blueprint):
        """Return blueprint with human-readable slot names."""
        return {SLOT_NAMES[k]: v for k, v in blueprint.items()}


# ==============================================================
# SECTION 3: BLUEPRINT-AWARE ENCODER
# ==============================================================

class BlueprintEncoder:
    """
    Wraps SemanticEncoder. Returns both the flat token stream
    AND the sparse blueprint in one pass.
    """

    def __init__(self, registry_path="unknown_words.json"):
        self._enc = SemanticEncoder(registry_path)
        self._bp  = BlueprintExtractor()

    def encode(self, text):
        """
        Returns (token_ids, blueprint, trace).
        token_ids  — flat list of ints (original SCBS output)
        blueprint  — sparse dict {slot: cluster_id}
        trace      — full step-by-step breakdown
        """
        ids, trace = self._enc.encode(text)
        bp = self._bp.extract(trace)
        return ids, bp, trace

    def decode(self, token_ids):
        return self._enc.decode(token_ids)


# ==============================================================
# SECTION 4: BLUEPRINT DISTANCE  O(k)
# ==============================================================

def blueprint_distance(A, B, penalty=MISSING_SLOT_PENALTY):
    """
    Compare two sparse blueprints.

    For each slot present in either blueprint:
      both present  → |A[slot] - B[slot]|
      only one has  → penalty

    Returns average over all compared slots.
    O(k) where k = union of filled slots (typically 3-6).
    """
    all_slots = set(A.keys()) | set(B.keys())
    if not all_slots:
        return 0.0

    total = 0.0
    for slot in all_slots:
        if slot in A and slot in B:
            total += abs(A[slot] - B[slot])
        else:
            total += penalty

    return round(total / len(all_slots), 4)


def interpret_blueprint(d):
    if d <= 10:   return "Nearly identical"
    if d <= 50:   return "Very similar"
    if d <= 150:  return "Related topic"
    if d <= 350:  return "Different topic"
    return "Completely unrelated"


# ==============================================================
# SECTION 5: BLUEPRINT-AWARE NDJSON STORE
# ==============================================================

class InvertedIndex:
    """
    Inverted index over blueprint slots, bucketed by semantic cluster.

    Structure:
      _index[slot][cluster_bucket] → set of record IDs

    cluster_bucket = which named cluster the ID falls in (0-18).
    So Technology IDs (3401-3500) and Programming IDs (3501-3600)
    are in different buckets — only records in the same cluster
    family are returned as candidates.

    Query: for each slot in query blueprint,
           find the cluster it belongs to,
           return only records whose same slot is in that cluster
           or an adjacent cluster (controlled by radius).

    At 1M records with 18 clusters:
      ~55K records per cluster per slot on average.
      With 2-3 slots in query, intersection drops to ~1-5K candidates.
      That's 0.1-0.5% of the corpus — 200-1000x speedup over linear.
    """

    # Ordered cluster list — index = cluster_bucket number
    CLUSTERS = list(CLUSTER_RANGES.keys())
    CLUSTER_BUCKET = {name: i for i, name in enumerate(CLUSTERS)}

    def __init__(self):
        # slot → cluster_bucket → set of record IDs
        self._index: dict[int, dict[int, set]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._blueprints: dict[int, dict] = {}

    def _bucket(self, cid: int) -> int:
        """Map a cluster ID value to its cluster bucket number."""
        for name, (lo, hi) in CLUSTER_RANGES.items():
            if lo <= cid <= hi:
                return self.CLUSTER_BUCKET.get(name, -1)
        return -1   # Low-tier / unknown

    def add(self, record_id: int, blueprint: dict):
        """Index one record. O(k) where k = filled slots."""
        self._blueprints[record_id] = blueprint
        for slot, cid in blueprint.items():
            bucket = self._bucket(cid)
            if bucket >= 0:
                self._index[slot][bucket].add(record_id)

    def candidates(self, query_bp: dict,
                   radius: int = 0) -> set:
        """
        Return candidate record IDs that share at least one
        slot-cluster with the query.

        Strategy: UNION across slots with exact cluster match (radius=0).
        - Each slot contributes ~1/18 of corpus (one cluster)
        - Union of 3 slots → ~15-20% of corpus as candidates
        - No recall loss — any record matching any slot is included
        - Distance check then filters to true matches

        radius=0  → exact cluster match only (recommended)
        radius=1  → include adjacent clusters for higher recall
        """
        found: set = set()
        for slot, cid in query_bp.items():
            bucket = self._bucket(cid)
            if bucket < 0:
                continue
            for b in range(max(0, bucket - radius),
                           min(len(self.CLUSTERS),
                               bucket + radius + 1)):
                found |= self._index[slot].get(b, set())
        return found

    def size(self) -> int:
        return len(self._blueprints)


class BlueprintStore:
    """
    NDJSON store with an in-memory inverted index for fast search.

    Write path  — O(k) index update alongside NDJSON append
    Search path — O(c·k) where c = candidates (~200-500)
                  instead of O(n·k) linear scan over all records

    Record format:
    {
      "id": 1,
      "ts": "2026-05-12T10:00:00",
      "e":  [3721, 32, 4297, ...],
      "bp": {"0": 3721, "1": 4297},
      "len": 7,
      "hash": "abc12345",
      "sem_group": "programming",
      "sem_score": "3501-3600"
    }
    """

    def __init__(self, filepath="blueprint_store.ndjson"):
        self.filepath  = filepath
        self._index    = InvertedIndex()
        self._records: dict[int, dict] = {}
        self._next_id  = 1
        self._build_index()   # warm up from existing NDJSON

    # ── Index warm-up ──────────────────────────────────────────

    def _build_index(self):
        """Load existing NDJSON and build index. One-time startup cost."""
        if not os.path.exists(self.filepath):
            return
        with open(self.filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                bp  = self._parse_bp(rec)
                self._index.add(rec["id"], bp)
                self._records[rec["id"]] = rec
                self._next_id = max(self._next_id, rec["id"] + 1)

    def _parse_bp(self, record) -> dict:
        """String keys → int keys."""
        return {int(k): v for k, v in record.get("bp", {}).items()}

    # ── Write ──────────────────────────────────────────────────

    def append(self, token_ids, blueprint, original_text=""):
        record = {
            "id":        self._next_id,
            "ts":        datetime.now(timezone.utc).strftime(
                             "%Y-%m-%dT%H:%M:%S"),
            "e":         token_ids,
            "bp":        {str(k): v for k, v in blueprint.items()},
            "len":       len(token_ids),
            "hash":      hashlib.md5(
                             original_text.encode()).hexdigest()[:8],
            "sem_group": derive_sem_group(token_ids),
            "sem_score": derive_sem_score(token_ids),
        }
        # Update index and in-memory store
        self._index.add(record["id"], blueprint)
        self._records[record["id"]] = record
        self._next_id += 1

        # Append to NDJSON
        with open(self.filepath, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    # ── Indexed search  O(c·k) ─────────────────────────────────

    def blueprint_search(self, query_bp: dict,
                         threshold: float = 150.0,
                         radius: int = 1) -> list:
        """
        Fast indexed semantic search.

        1. Inverted index lookup  → candidate record IDs  O(k)
        2. Blueprint distance     → score candidates only O(c·k)
        3. Threshold filter + sort

        c = candidates (typically 0.1-5% of corpus with cluster index)
        k = slots in query blueprint (2-5)
        """
        candidate_ids = self._index.candidates(query_bp, radius)

        results = []
        for rid in candidate_ids:
            rec = self._records.get(rid)
            if not rec:
                continue
            bp = self._parse_bp(rec)
            d  = blueprint_distance(query_bp, bp)
            if d <= threshold:
                results.append({**rec, "bp_distance": d})

        return sorted(results, key=lambda r: r["bp_distance"])

    # ── Full scan fallback (kept for completeness) ─────────────

    def full_scan_search(self, query_bp: dict,
                         threshold: float = 150.0) -> list:
        """Linear scan — O(n·k). Use only for correctness testing."""
        results = []
        for rec in self._records.values():
            bp = self._parse_bp(rec)
            d  = blueprint_distance(query_bp, bp)
            if d <= threshold:
                results.append({**rec, "bp_distance": d})
        return sorted(results, key=lambda r: r["bp_distance"])

    # ── Slot filter ────────────────────────────────────────────

    def slot_filter(self, slot_index: int,
                    cluster_name: str) -> list:
        """
        Return all records where a specific slot contains
        a token from a given cluster. Pure integer range check.
        Uses index for fast retrieval.
        """
        lo, hi  = CLUSTER_RANGES.get(cluster_name, (0, 0))
        lo_b    = lo // InvertedIndex.BUCKET_SIZE
        hi_b    = hi // InvertedIndex.BUCKET_SIZE
        found   = set()
        idx     = self._index._index.get(slot_index, {})
        for b in range(lo_b, hi_b + 1):
            found |= idx.get(b, set())

        results = []
        for rid in found:
            rec = self._records.get(rid)
            if rec:
                bp  = self._parse_bp(rec)
                cid = bp.get(slot_index, 0)
                if lo <= cid <= hi:
                    results.append(rec)
        return results

    def load_all(self) -> list:
        return list(self._records.values())

    def index_size(self) -> int:
        return self._index.size()




# ==============================================================
# SECTION 6: DEMO
# ==============================================================

def sep(title=""):
    print(f"\n{'─'*66}")
    if title:
        print(f"  {title}")
        print(f"{'─'*66}")


def demo():
    print("\n" + "="*66)
    print("  BLUEPRINT EXTRACTOR — Sparse Slot Record")
    print("  Variable bytes · O(k) compare · zero extra encode cost")
    print("="*66)

    enc   = BlueprintEncoder()
    store = BlueprintStore("demo_bp_store.ndjson")

    sentences = [
        "hello great morning team",
        "the developer is debugging python code",
        "unhappy with the deployment failure",
        "microservice authentication is misbehaving",
        "preprocessing the data successfully",
        "reconfiguring the kafka pipeline",
        "the optimizer ran efficiently today",
        "bye see you tomorrow",
        "why is the authentication failing",
        "great job on the kafka deployment",
    ]

    # ── Step 1: Encode + extract blueprint ────────────────────
    sep("Step 1 — Encode and extract blueprint in one pass")

    blueprints = []
    for sent in sentences:
        ids, bp, trace = enc.encode(sent)
        blueprints.append(bp)
        rec = store.append(ids, bp, sent)

        print(f"\n  '{sent}'")
        print(f"  Token stream : {ids}")
        print(f"  Blueprint    : ", end="")
        named = {SLOT_NAMES[k]: v for k, v in bp.items()}
        print(json.dumps(named))
        print(f"  Slots filled : {len(bp)} / 9  "
              f"({len(json.dumps({str(k):v for k,v in bp.items()}))} bytes "
              f"vs {len(ids)*4} bytes raw)")

    # ── Step 2: Blueprint distance vs token distance ──────────
    sep("Step 2 — Blueprint O(k) distance vs token O(n) distance")

    pairs = [
        (0, 7,  "hello team vs bye tomorrow"),
        (0, 1,  "hello team vs debugging code"),
        (1, 2,  "debugging code vs unhappy failure"),
        (1, 9,  "debugging code vs great kafka deployment"),
        (2, 3,  "unhappy failure vs auth misbehaving"),
        (4, 5,  "preprocessing vs kafka pipeline"),
        (8, 2,  "why auth failing vs unhappy failure"),
    ]

    print(f"\n  {'Pair':38s} {'Token dist':>11} {'BP dist':>8} "
          f"{'Slots':>6} {'Interpretation'}")
    print(f"  {'─'*38} {'─'*11} {'─'*8} {'─'*6} {'─'*18}")

    for i, j, label in pairs:
        ids_i, _, _ = enc.encode(sentences[i])
        ids_j, _, _ = enc.encode(sentences[j])
        td = distance_1d(ids_i, ids_j)
        bd = blueprint_distance(blueprints[i], blueprints[j])
        k  = len(set(blueprints[i]) | set(blueprints[j]))
        interp = interpret_blueprint(bd)
        print(f"  {label:38s} {td:11.1f} {bd:8.1f} "
              f"{k:6d}  {interp}")

    # ── Step 3: Blueprint search ──────────────────────────────
    sep("Step 3 — Semantic search using blueprint distance O(k)")

    queries = [
        "frustrated about the kafka error",
        "good morning everyone",
        "what is wrong with the system",
    ]

    for q in queries:
        _, q_bp, _ = enc.encode(q)
        print(f"\n  Query: '{q}'")
        print(f"  Blueprint: {json.dumps({SLOT_NAMES[k]:v for k,v in q_bp.items()})}")
        results = store.blueprint_search(q_bp, threshold=250)
        if results:
            for r in results[:3]:
                orig = sentences[r["id"] - 1]
                print(f"    dist={r['bp_distance']:6.1f}  '{orig}'")
        else:
            print("    (no matches within threshold)")

    # ── Step 4: Slot filter ───────────────────────────────────
    sep("Step 4 — Slot filter (pure integer range check)")

    filters = [
        (3, "Negative Emotions", "EMOTION slot contains negative word"),
        (2, "Programming",       "TECH slot contains programming word"),
        (5, "Greetings",         "SOCIAL slot is a greeting"),
    ]

    for slot, cluster, desc in filters:
        matches = store.slot_filter(slot, cluster)
        print(f"\n  Filter: {desc}")
        for m in matches:
            orig = sentences[m["id"] - 1]
            print(f"    '{orig}'")

    # ── Step 5: Storage size comparison ──────────────────────
    sep("Step 5 — Storage comparison")

    print(f"\n  {'Sentence':40s} {'Raw bytes':>10} {'BP bytes':>10} {'Saving':>8}")
    print(f"  {'─'*40} {'─'*10} {'─'*10} {'─'*8}")
    for sent, bp in zip(sentences, blueprints):
        ids, _, _ = enc.encode(sent)
        raw_bytes = len(ids) * 4
        bp_bytes  = len(json.dumps(
            {str(k): v for k, v in bp.items()}))
        saving = f"{100*(1-bp_bytes/raw_bytes):.0f}%" \
                 if raw_bytes > 0 else "—"
        print(f"  {sent[:40]:40s} {raw_bytes:10d} "
              f"{bp_bytes:10d} {saving:>8}")

    print("\n" + "="*66)
    print("  Blueprint demo complete.")
    print("="*66 + "\n")

    for f in ["demo_bp_store.ndjson"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    demo()
