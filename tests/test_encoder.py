"""Tests for the SCBS Encoder."""

import pytest
from scbs import Encoder


@pytest.fixture
def encoder():
    return Encoder(registry_path="test_unknown.json")


class TestEncoder:
    def test_encode_returns_list_of_tuples(self, encoder):
        row = encoder.encode("kafka deployment failed")
        assert isinstance(row, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in row)

    def test_encode_empty_string(self, encoder):
        row = encoder.encode("")
        assert row == []

    def test_encode_deterministic(self, encoder):
        row_1 = encoder.encode("kafka consumer lag")
        row_2 = encoder.encode("kafka consumer lag")
        assert row_1 == row_2

    def test_encode_different_sentences_different_rows(self, encoder):
        row_1 = encoder.encode("kafka deployment succeeded")
        row_2 = encoder.encode("welcome to the team onboarding")
        assert row_1 != row_2

    def test_encode_detailed_returns_dict(self, encoder):
        result = encoder.encode_detailed("kafka deployment")
        assert "row" in result
        assert "token_ids" in result
        assert "blueprint" in result
        assert "trace" in result

    def test_encode_handles_unknown_words(self, encoder):
        row = encoder.encode("xyzabc qwerty unknownword")
        assert isinstance(row, list)

    def test_encode_slots_are_within_range(self, encoder):
        row = encoder.encode("kafka authentication fraud incident")
        for slot, value in row:
            assert 0 <= slot < 10, f"slot {slot} out of range"
            assert value > 0, f"value {value} should be positive"


class TestEncoderClusters:
    def test_tech_words_in_tech_slot(self, encoder):
        row = encoder.encode("kafka")
        slots = {slot for slot, _ in row}
        assert 2 in slots, "kafka should fill TECH slot 2"

    def test_emotion_words_in_emotion_slot(self, encoder):
        row = encoder.encode("frustrated")
        slots = {slot for slot, _ in row}
        assert 3 in slots, "frustrated should fill EMOTION slot 3"

    def test_question_words_in_intent_slot(self, encoder):
        row = encoder.encode("what")
        slots = {slot for slot, _ in row}
        assert 6 in slots, "what should fill INTENT slot 6"

    def test_domain_words_in_domain_slot(self, encoder):
        row = encoder.encode("fraud")
        slots = {slot for slot, _ in row}
        assert 9 in slots, "fraud should fill DOMAIN slot 9"
