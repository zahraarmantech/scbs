"""
Semantic Chunk Buffer System
============================
Full implementation with 4-Layer Intelligent Unknown Word Resolution.

Unknown Word Resolution Pipeline (runs in order, first confident match wins):
  Layer 1 — Prefix morphological analysis  (e.g. "un" -> Negative Emotions)
  Layer 2 — Suffix morphological analysis  (e.g. "-ing" -> Actions & Verbs)
  Layer 3 — Edit distance to nearest known semantic word
  Layer 4 — Character n-gram cluster voting
  Fallback — Dynamic ID assignment in Miscellaneous (4901-9999)

Key encoder improvement over the original spec:
  Word-boundary-aware hybrid greedy: when at an alphabetic position,
  the encoder checks the FULL word first. If the full word is unknown,
  it routes to the resolution pipeline instead of consuming it one
  character at a time (which would lose all semantic signal).
"""

import json
import hashlib
import os
from datetime import datetime, timezone
from collections import defaultdict


# ==============================================================
# SECTION 1: VOCABULARY DEFINITIONS
# ==============================================================

# Tier 1 - Single ASCII characters (IDs 32-126)
ASCII_VOCAB = {chr(i): i for i in range(32, 127)}

# Tier 2 - Common 2-character pairs (IDs 96-595)
TWO_CHAR_VOCAB = {
    "is": 110, "in": 111, "it": 112, "at": 113, "an": 114,
    "as": 115, "be": 116, "by": 117, "do": 118, "go": 119,
    "he": 120, "if": 121, "me": 122, "my": 123, "no": 124,
    "of": 125, "on": 126, "or": 127, "so": 128, "to": 129,
    "up": 130, "us": 131, "we": 132, "ok": 133,
    "db": 136, "id": 137, "ip": 138, "os": 139,
}

# Tier 3 - Common 3-character combinations (IDs 596-1595)
THREE_CHAR_VOCAB = {
    "the": 600, "and": 601, "for": 602, "are": 603, "but": 604,
    "not": 605, "you": 606, "all": 607, "can": 608, "her": 609,
    "was": 610, "one": 611, "our": 612, "out": 613, "day": 614,
    "get": 615, "has": 616, "him": 617, "his": 618, "how": 619,
    "let": 621, "man": 622, "new": 623, "now": 624,
    "old": 625, "see": 626, "two": 627, "way": 628, "did": 630,
    "she": 631, "use": 632, "may": 633, "say": 634, "any": 635,
    "try": 636, "run": 637, "set": 638, "var": 639, "int": 640,
    "str": 641, "def": 642, "yes": 643, "own": 644, "per": 645,
    "log": 646, "url": 647, "sql": 652, "csv": 653, "xml": 654,
}

# Semantic Vocabulary - Clusters starting at ID 3000
from .vocabulary import (
    SEMANTIC_VOCAB_V2 as SEMANTIC_VOCAB,
    CANONICAL_OVERRIDES_V2 as _OVERRIDE_V2,
    CLUSTER_RANGES_V2 as CLUSTER_RANGES,
    id_to_subcluster as _id_to_subcluster,
)
CANONICAL_OVERRIDES = _OVERRIDE_V2  # use V2 overrides


# Prefix hints -> cluster (longest first)
PREFIX_CLUSTER_HINTS = {
    "cyber":  "Security Events",
    "micro":  "Tech: Container & Orchestration",
    "macro":  "Tech: AI & ML",
    "meta":   "Tech: AI & ML",
    "inter":  "Tech: Networking & API",
    "intra":  "Tech: Networking & API",
    "auto":   "Adj: Technical",
    "multi":  "Numbers & Math",
    "anti":   "Negative Emotions",
    "mal":    "Negative Emotions",
    "mis":    "Negative Emotions",
    "dis":    "Negative Emotions",
    "non":    "Negative Emotions",
    "un":     "Negative Emotions",
    "im":     "Negative Emotions",
    "de":     "Actions: Change",
    "re":     "Actions: Change",
    "co":     "People: General",
    "pre":    "Time: Direction",
    "post":   "Time: Direction",
    "over":   "Adj: Size & Scale",
    "under":  "Adj: Size & Scale",
    "semi":   "Adj: Technical",
    "sub":    "Adj: Technical",
    "super":  "Positive Emotions",
    "hyper":  "Positive Emotions",
    "ultra":  "Positive Emotions",
}

# Suffix hints -> cluster (longest first)
SUFFIX_CLUSTER_HINTS = {
    "ology":  "Questions & Inquiry",
    "tion":   "Actions: Management",
    "sion":   "Actions: Management",
    "ment":   "Actions: Management",
    "ness":   "Adj: Quality",
    "able":   "Adj: Technical",
    "ible":   "Adj: Technical",
    "ical":   "Adj: Technical",
    "ware":   "Tech: Monitoring",
    "base":   "Data: Database Ops",
    "gram":   "Data: File Ops",
    "less":   "Negative Emotions",
    "ful":    "Positive Emotions",
    "ize":    "Actions: Change",
    "ise":    "Actions: Change",
    "ify":    "Actions: Change",
    "ing":    "Actions: Management",
    "ate":    "Actions: Management",
    "ive":    "Adj: Technical",
    "ous":    "Adj: Quality",
    "al":     "Adj: Quality",
    "ic":     "Adj: Technical",
    "ly":     "Adj: Quality",
    "ed":     "Actions: Management",
    "er":     "People: General",
    "or":     "People: General",
    "ist":    "People: General",
    "ian":    "People: General",
    "age":    "Data: File Ops",
    "log":    "Data: File Ops",
    "net":    "Tech: Networking & API",
    "tech":   "Tech: Monitoring",
    "time":   "Time: Units",
    "date":   "Time: Scheduling",
    "day":    "Time: Units",
}


# ==============================================================
# SECTION 2: HELPERS
# ==============================================================

def id_to_cluster(chunk_id):
    for cluster, (lo, hi) in CLUSTER_RANGES.items():
        if lo <= chunk_id <= hi:
            return cluster
    return None


def build_master_lookup():
    lookup = {}
    lookup.update(ASCII_VOCAB)
    lookup.update(TWO_CHAR_VOCAB)
    lookup.update(THREE_CHAR_VOCAB)
    lookup.update(SEMANTIC_VOCAB)
    lookup.update(CANONICAL_OVERRIDES)  # always wins
    return lookup


def build_reverse_lookup(lookup):
    reverse = {}
    for word, cid in lookup.items():
        if cid not in reverse:
            reverse[cid] = word
    return reverse


# ==============================================================
# SECTION 3: UNKNOWN WORD INTELLIGENCE ENGINE
# ==============================================================

class UnknownWordResolver:
    """
    4-Layer pipeline to assign any unknown word to the most
    semantically appropriate cluster, then give it a stable ID.

    Layer 1 - Prefix morphological analysis
    Layer 2 - Suffix morphological analysis
    Layer 3 - Edit distance to nearest known semantic word
    Layer 4 - Character n-gram cluster voting
    Fallback - Miscellaneous range
    """

    def __init__(self, registry_path="unknown_words.json"):
        self.registry_path = registry_path
        self.registry = self._load_registry()
        self._cluster_counters = {}
        self._init_counters()
        self._sorted_prefixes = sorted(
            PREFIX_CLUSTER_HINTS.items(), key=lambda x: -len(x[0])
        )
        self._sorted_suffixes = sorted(
            SUFFIX_CLUSTER_HINTS.items(), key=lambda x: -len(x[0])
        )

    def _load_registry(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path) as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def _init_counters(self):
        for cluster, (lo, hi) in CLUSTER_RANGES.items():
            used = {v for v in SEMANTIC_VOCAB.values() if lo <= v <= hi}
            used |= {v for v in CANONICAL_OVERRIDES.values() if lo <= v <= hi}
            for entry in self.registry.values():
                if entry.get("cluster") == cluster:
                    used.add(entry["id"])
            self._cluster_counters[cluster] = max(used, default=lo - 1) + 1

    def _next_id_in_cluster(self, cluster):
        lo, hi = CLUSTER_RANGES[cluster]
        nxt = self._cluster_counters.get(cluster, lo)
        if nxt > hi:
            return None
        self._cluster_counters[cluster] = nxt + 1
        return nxt

    # Layer 1: Prefix Analysis
    def _layer1_prefix(self, word):
        for prefix, cluster in self._sorted_prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                return cluster
        return None

    # Layer 2: Suffix Analysis
    def _layer2_suffix(self, word):
        for suffix, cluster in self._sorted_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return cluster
        return None

    # Layer 3: Levenshtein Edit Distance
    @staticmethod
    def _levenshtein(a, b):
        if len(a) < len(b):
            a, b = b, a
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for ca in a:
            curr = [prev[0] + 1]
            for j, cb in enumerate(b):
                curr.append(min(
                    prev[j + 1] + 1,
                    curr[j] + 1,
                    prev[j] + (ca != cb),
                ))
            prev = curr
        return prev[-1]

    def _layer3_edit_distance(self, word):
        best_dist = float("inf")
        best_cluster = None
        for known_word, known_id in SEMANTIC_VOCAB.items():
            if abs(len(known_word) - len(word)) > 5:
                continue
            d = self._levenshtein(word, known_word)
            if d < best_dist:
                best_dist = d
                best_cluster = id_to_cluster(known_id)
        threshold = max(2, len(word) // 3)
        if best_dist <= threshold and best_cluster:
            return best_cluster
        return None

    # Layer 4: Character N-gram Cluster Voting
    def _layer4_ngram_voting(self, word):
        votes = defaultdict(float)
        for n in (3, 4):
            grams = [word[i:i + n] for i in range(len(word) - n + 1)]
            for gram in grams:
                for known_word, known_id in SEMANTIC_VOCAB.items():
                    if gram in known_word:
                        cluster = id_to_cluster(known_id)
                        if cluster and cluster != "Miscellaneous":
                            votes[cluster] += 1.0 / n
        if not votes:
            return None
        winner = max(votes, key=lambda k: votes[k])
        return winner if votes[winner] >= 0.5 else None

    def resolve(self, word):
        """Resolve an unknown word. Result is cached persistently."""
        if word in self.registry:
            return self.registry[word]

        cluster, method = None, None

        cluster = self._layer1_prefix(word)
        if cluster:
            method = "prefix"

        if not cluster:
            cluster = self._layer2_suffix(word)
            if cluster:
                method = "suffix"

        if not cluster:
            cluster = self._layer3_edit_distance(word)
            if cluster:
                method = "edit_distance"

        if not cluster:
            cluster = self._layer4_ngram_voting(word)
            if cluster:
                method = "ngram_voting"

        if not cluster:
            cluster, method = "Miscellaneous", "fallback"

        assigned_id = self._next_id_in_cluster(cluster)
        if assigned_id is None:
            cluster, method = "Miscellaneous", "fallback_cluster_full"
            assigned_id = self._next_id_in_cluster("Miscellaneous")

        entry = {"id": assigned_id, "cluster": cluster,
                 "method": method, "word": word}
        self.registry[word] = entry
        self._save_registry()
        return entry


# ==============================================================
# SECTION 4: ENCODER
# ==============================================================

class SemanticEncoder:
    """
    Hybrid word-boundary-aware greedy encoder.

    For alphabetic words:
      1. If full word is in vocabulary -> use it directly
      2. If greedy finds a meaningful sub-word prefix (>1 char) -> use it
         and continue (handles "debugging" = debug + g + in + g)
      3. If no useful match -> route ENTIRE word to unknown resolver
         (prevents "unhappy" from being split into u + n + happy)

    For non-alpha characters: standard greedy then ASCII fallback.
    """

    def __init__(self, registry_path="unknown_words.json"):
        self.lookup = build_master_lookup()
        self.reverse = build_reverse_lookup(self.lookup)
        self.resolver = UnknownWordResolver(registry_path)
        self._sorted_vocab = sorted(self.lookup.keys(), key=len, reverse=True)

    def _word_end(self, text, pos):
        end = pos
        while end < len(text) and text[end].isalpha():
            end += 1
        return end

    def _greedy_match(self, text, pos):
        for word in self._sorted_vocab:
            wlen = len(word)
            if text[pos:pos + wlen] == word:
                return word, wlen
        return None

    def encode(self, text):
        """Returns (token_ids, trace)."""
        tokens = []
        trace = []
        lower = text.lower()
        pos = 0

        while pos < len(lower):

            if lower[pos].isalpha():
                wend = self._word_end(lower, pos)
                full_word = lower[pos:wend]

                # 1. Full word in vocabulary
                if full_word in self.lookup:
                    cid = self.lookup[full_word]
                    tokens.append(cid)
                    trace.append({
                        "surface": text[pos:wend], "matched": full_word,
                        "id": cid, "cluster": id_to_cluster(cid) or "Low-Tier",
                        "method": "vocab_exact", "unknown": False,
                    })
                    pos = wend
                    continue

                # 2. Greedy sub-word (must consume >1 char to be useful)
                m = self._greedy_match(lower, pos)
                if m and len(m[0]) > 1:
                    word_matched, wlen = m
                    cid = self.lookup[word_matched]
                    tokens.append(cid)
                    trace.append({
                        "surface": text[pos:pos + wlen], "matched": word_matched,
                        "id": cid, "cluster": id_to_cluster(cid) or "Low-Tier",
                        "method": "vocab_subword", "unknown": False,
                    })
                    pos += wlen
                    continue

                # 3. Unknown: resolve entire word as a unit
                resolution = self.resolver.resolve(full_word)
                cid = resolution["id"]
                tokens.append(cid)
                trace.append({
                    "surface": text[pos:wend], "matched": full_word,
                    "id": cid, "cluster": resolution["cluster"],
                    "method": resolution["method"], "unknown": True,
                })
                pos = wend

            else:
                m = self._greedy_match(lower, pos)
                if m:
                    word_matched, wlen = m
                    cid = self.lookup[word_matched]
                    tokens.append(cid)
                    trace.append({
                        "surface": text[pos:pos + wlen], "matched": word_matched,
                        "id": cid, "cluster": id_to_cluster(cid) or "Low-Tier",
                        "method": "vocab_exact", "unknown": False,
                    })
                    pos += wlen
                else:
                    cid = ord(lower[pos])
                    tokens.append(cid)
                    trace.append({
                        "surface": text[pos], "matched": lower[pos],
                        "id": cid, "cluster": "ASCII",
                        "method": "ascii_fallback", "unknown": False,
                    })
                    pos += 1

        return tokens, trace

    def decode(self, token_ids):
        id_to_unknown = {e["id"]: w for w, e in self.resolver.registry.items()}
        parts = []
        for tid in token_ids:
            if tid in id_to_unknown:
                parts.append(id_to_unknown[tid])
            elif tid in self.reverse:
                parts.append(self.reverse[tid])
            elif 32 <= tid <= 126:
                parts.append(chr(tid))
            else:
                parts.append(f"<UNK:{tid}>")
        return "".join(parts)


# ==============================================================
# SECTION 5: SEMANTIC DISTANCE
# ==============================================================

def semantic_distance(a, b):
    n = max(len(a), len(b))
    a_p = a + [32] * (n - len(a))
    b_p = b + [32] * (n - len(b))
    return round(sum(abs(x - y) for x, y in zip(a_p, b_p)) / n, 4)


def interpret_distance(d):
    if d <= 5:   return "Nearly identical"
    if d <= 20:  return "Very similar"
    if d <= 100: return "Related topic"
    if d <= 500: return "Different topic"
    return "Completely unrelated"


# ==============================================================
# SECTION 6: SEMANTIC TAGGING
# ==============================================================

def derive_sem_group(token_ids):
    votes = defaultdict(int)
    for tid in token_ids:
        c = id_to_cluster(tid)
        if c and c != "Miscellaneous":
            votes[c] += 1
    if not votes:
        return "miscellaneous"
    winner = max(votes, key=lambda k: votes[k])
    return winner.lower().replace(" ", "_").replace("&", "and")


def derive_sem_score(token_ids):
    group = derive_sem_group(token_ids)
    for cluster, (lo, hi) in CLUSTER_RANGES.items():
        if cluster.lower().replace(" ", "_").replace("&", "and") == group:
            return f"{lo}-{hi}"
    return "4901-9999"


# ==============================================================
# SECTION 7: NDJSON STORAGE
# ==============================================================

class NDJSONStore:
    def __init__(self, filepath="buffer_store.ndjson"):
        self.filepath = filepath
        self._next_id = self._count() + 1

    def _count(self):
        if not os.path.exists(self.filepath):
            return 0
        with open(self.filepath) as f:
            return sum(1 for ln in f if ln.strip())

    def append(self, token_ids, original_text=""):
        record = {
            "id": self._next_id,
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "e": token_ids, "len": len(token_ids),
            "hash": hashlib.md5(original_text.encode()).hexdigest()[:8],
            "chk": hashlib.md5(str(token_ids).encode()).hexdigest()[:8],
            "sem_group": derive_sem_group(token_ids),
            "sem_score": derive_sem_score(token_ids),
        }
        self._next_id += 1
        with open(self.filepath, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def load_all(self):
        if not os.path.exists(self.filepath):
            return []
        with open(self.filepath) as f:
            return [json.loads(ln) for ln in f if ln.strip()]

    def semantic_search(self, query_ids, threshold=50.0):
        results = []
        for rec in self.load_all():
            d = semantic_distance(query_ids, rec["e"])
            if d <= threshold:
                results.append({**rec, "distance": d})
        return sorted(results, key=lambda r: r["distance"])

    def range_query(self, cluster_name):
        lo, hi = CLUSTER_RANGES.get(cluster_name, (0, 0))
        return [r for r in self.load_all()
                if any(lo <= tid <= hi for tid in r["e"])]


# ==============================================================
# SECTION 8: DEMO
# ==============================================================

def sep(title=""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}")


def demo():
    print("\n" + "=" * 60)
    print("  SEMANTIC CHUNK BUFFER SYSTEM")
    print("  4-Layer Unknown Word Intelligence")
    print("=" * 60)

    enc = SemanticEncoder("demo_unknown.json")
    store = NDJSONStore("demo_store.ndjson")

    # 1. Standard encoding
    sep("Demo 1 — Standard Known Words (spec examples)")
    for sent in ["hello good", "hi great", "bye bad", "python code"]:
        ids, trace = enc.encode(sent)
        print(f"\n  '{sent}'  ->  {ids}")
        for s in trace:
            print(f"    '{s['surface']}' -> {s['id']} ({s['cluster']})")

    # 2. Unknown word intelligence - the KEY demo
    sep("Demo 2 — 4-Layer Unknown Word Resolution")
    unknowns = [
        ("unhappy with the result",
         "Layer 1: prefix 'un' -> Negative Emotions"),
        ("preprocessing the dataset",
         "Layer 1: prefix 'pre' -> Time & Dates"),
        ("the optimizer running efficiently",
         "Layer 2: suffix 'er' -> People; suffix 'ly' -> Adjectives"),
        ("microservice authentication failed",
         "Both in vocab: microservice(3443), authentication(3441)"),
        ("hopelessness and despair",
         "Layer 2: suffix 'ness' -> Adjectives; despair in vocab"),
        ("reconfiguring the scalable system",
         "Layer 1: 're' -> Actions; scalable in vocab"),
        ("zorbulating the frimblex",
         "Fully invented -> Layer 4 ngram voting -> fallback"),
        ("she is cybersecure and distributed",
         "Layer 1: 'cyber' -> Technology; distributed in vocab"),
    ]

    for sent, note in unknowns:
        ids, trace = enc.encode(sent)
        print(f"\n  Input : '{sent}'")
        print(f"  Reason: {note}")
        for s in trace:
            tag = "  <-- UNKNOWN RESOLVED" if s["unknown"] else ""
            print(f"    {s['surface']:22s} -> {s['id']:5d}"
                  f"  [{s['cluster']:26s}]  {s['method']}{tag}")

    # 3. Semantic distance
    sep("Demo 3 — Semantic Distance")
    pairs = [
        ("hello good",       "hi great"),
        ("unhappy failure",  "sad terrible"),
        ("python code",      "javascript function"),
        ("hello good",       "python code"),
    ]
    for s1, s2 in pairs:
        a, _ = enc.encode(s1)
        b, _ = enc.encode(s2)
        d = semantic_distance(a, b)
        print(f"\n  '{s1}'  vs  '{s2}'")
        print(f"    {a}  vs  {b}")
        print(f"    distance = {d:.2f}  ->  {interpret_distance(d)}")

    # 4. NDJSON storage and search
    sep("Demo 4 — NDJSON Storage + Semantic Search")
    sentences = [
        "hello good morning",
        "unhappy about the error",
        "python debugging function",
        "microservice authentication failed",
        "hi nice to meet you",
    ]
    print()
    for s in sentences:
        ids, _ = enc.encode(s)
        rec = store.append(ids, s)
        print(f"  Stored id={rec['id']}  "
              f"sem_group={rec['sem_group']:30s}  {ids}")

    query = "unhappy and frustrated"
    qids, _ = enc.encode(query)
    print(f"\n  Search: '{query}'  ->  {qids}")
    results = store.semantic_search(qids, threshold=150)
    if results:
        for r in results:
            print(f"    Match id={r['id']}  dist={r['distance']:.1f}"
                  f"  sem_group={r['sem_group']}")
    else:
        print("    No matches within threshold.")

    # 5. Unknown word registry summary
    sep("Demo 5 — Unknown Word Registry (persistent)")
    reg = enc.resolver.registry
    if reg:
        print(f"\n  {len(reg)} words learned and saved to disk:\n")
        print(f"  {'Word':25s} {'ID':6s} {'Cluster':28s} Method")
        print(f"  {'─'*25} {'─'*6} {'─'*28} {'─'*14}")
        for word, entry in sorted(reg.items()):
            print(f"  {word:25s} {entry['id']:6d} "
                  f"{entry['cluster']:28s} {entry['method']}")
    else:
        print("  No unknown words encountered.")

    # 6. Round-trip decode
    sep("Demo 6 — Encode then Decode (round trip)")
    for sent in ["hello good", "unhappy failure today", "python code is great"]:
        ids, _ = enc.encode(sent)
        decoded = enc.decode(ids)
        print(f"\n  Original : '{sent}'")
        print(f"  Encoded  : {ids}")
        print(f"  Decoded  : '{decoded}'")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60 + "\n")

    for f in ["demo_store.ndjson", "demo_unknown.json"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    demo()
