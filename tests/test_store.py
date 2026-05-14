"""Tests for the SCBS Store."""

import pytest
from scbs import Encoder, Store


@pytest.fixture
def encoder():
    return Encoder(registry_path="test_store_unknown.json")


@pytest.fixture
def store_with_corpus(encoder):
    store = Store(n_clusters=8)
    corpus = [
        "kafka consumer lag increasing critical alert",
        "kubernetes pod eviction memory pressure",
        "deployment succeeded all health checks passing",
        "fraud transaction blocked suspicious activity",
        "employee performance review scheduled Monday",
        "customer complaint refund order delivery failed",
        "authentication token expired login failed",
        "good morning team standup five minutes",
        "happy to announce new feature is live",
        "database connection pool exhausted retry",
    ]
    for text in corpus:
        store.add(encoder.encode(text), text)
    store.build(corpus_texts=corpus)
    return store, encoder, corpus


class TestStoreBasics:
    def test_search_before_build_raises(self, encoder):
        store = Store()
        store.add(encoder.encode("test"), "test")
        with pytest.raises(RuntimeError):
            store.search(encoder.encode("query"), "query")

    def test_store_length_before_build(self, encoder):
        store = Store()
        for text in ["a", "b", "c"]:
            store.add(encoder.encode(text), text)
        assert len(store) == 3

    def test_search_returns_results_list_and_stats(self, store_with_corpus):
        store, encoder, _ = store_with_corpus
        query = encoder.encode("kafka failed alert")
        results, stats = store.search(query, "kafka failed alert")
        assert isinstance(results, list)
        assert isinstance(stats, dict)

    def test_search_results_have_required_fields(self, store_with_corpus):
        store, encoder, _ = store_with_corpus
        query = encoder.encode("kafka lag alert")
        results, _ = store.search(query, "kafka lag alert", threshold=200)
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "distance" in r

    def test_search_results_sorted_by_distance(self, store_with_corpus):
        store, encoder, _ = store_with_corpus
        query = encoder.encode("kafka deployment alert")
        results, _ = store.search(query, "kafka deployment alert", threshold=200)
        if len(results) > 1:
            distances = [r["distance"] for r in results]
            assert distances == sorted(distances)


class TestStoreExclusion:
    def test_exclusion_prevents_duplicate_results(self, store_with_corpus):
        store, encoder, _ = store_with_corpus
        query = encoder.encode("kafka alert")

        excluded = set()
        first, _ = store.search(
            query, "kafka alert", top_k=1, threshold=200, excluded=excluded,
        )

        if first:
            second, _ = store.search(
                query, "kafka alert", top_k=1, threshold=200, excluded=excluded,
            )
            if second:
                assert first[0]["id"] != second[0]["id"]


class TestStoreLinearSearch:
    def test_linear_search_returns_list(self, store_with_corpus):
        store, encoder, _ = store_with_corpus
        query = encoder.encode("kafka alert")
        results = store.linear_search(query, "kafka alert", threshold=200)
        assert isinstance(results, list)
