"""Tests for apc_cache.normalize — L1 input standardisation.

Covers: normalize(), task_sig(), qhash(), kw_lock_hash()
"""

import hashlib
import unicodedata

import pytest

from apc_cache.normalize import kw_lock_hash, normalize, qhash, task_sig


class TestNormalize:
    """Blueprint §4.1: Lossless surface normalisation."""

    def test_lowercase(self):
        assert normalize("Calculate Working Capital") == "calculate working capital"

    def test_merge_whitespace(self):
        assert normalize("  hello   world  ") == "hello world"

    def test_nfkc_normalization(self):
        """Full-width latin characters → ASCII equivalents."""
        fullwidth = "Ｈｅｌｌｏ"  # Fullwidth Latin
        result = normalize(fullwidth)
        # NFKC converts fullwidth to ASCII
        assert result == "hello"

    def test_nfkc_combined_chars(self):
        """Combining characters are normalised to single codepoints."""
        combined = "café"  # é as e + combining acute
        nfc_form = unicodedata.normalize("NFC", "café")
        assert normalize(combined) == nfc_form.lower()

    def test_empty_string(self):
        assert normalize("") == ""

    def test_only_whitespace(self):
        assert normalize("   \t\n  ") == ""

    def test_preserves_hyphens_and_underscores(self):
        """Deliberately not removing special chars — L1 is surface-only."""
        assert normalize("working_capital-ratio") == "working_capital-ratio"

    def test_idempotent(self):
        q = "  Calculate  Working  CAPITAL  "
        once = normalize(q)
        twice = normalize(once)
        assert once == twice

    def test_numbers_preserved(self):
        """Numbers are kept — L1 does not do placeholder substitution."""
        assert normalize("Q3 2024 Report") == "q3 2024 report"


class TestTaskSig:
    """task_sig = sha256(normalize(query) | agent_id | tools_hash)[:16]"""

    def test_deterministic(self):
        sig1 = task_sig("Hello World", "agent1", "hash123")
        sig2 = task_sig("Hello World", "agent1", "hash123")
        assert sig1 == sig2
        assert len(sig1) == 16

    def test_whitespace_insensitive(self):
        """Different spacing produces same sig due to normalize()."""
        sig1 = task_sig("hello   world", "agent1", "hash123")
        sig2 = task_sig("hello world", "agent1", "hash123")
        assert sig1 == sig2

    def test_different_agent_yields_different_sig(self):
        sig1 = task_sig("query", "agent_A", "hash")
        sig2 = task_sig("query", "agent_B", "hash")
        assert sig1 != sig2

    def test_different_tools_hash_yields_different_sig(self):
        sig1 = task_sig("query", "agent1", "hash_A")
        sig2 = task_sig("query", "agent1", "hash_B")
        assert sig1 != sig2

    def test_case_insensitive(self):
        sig1 = task_sig("HELLO WORLD", "agent1", "hash")
        sig2 = task_sig("hello world", "agent1", "hash")
        assert sig1 == sig2

    def test_unicode_normalization(self):
        """Fullwidth and standard ASCII produce the same sig."""
        sig1 = task_sig("hello", "agent1", "hash")
        sig2 = task_sig("ｈｅｌｌｏ", "agent1", "hash")  # fullwidth
        assert sig1 == sig2

    def test_hex_output(self):
        sig = task_sig("test", "a", "b")
        assert all(c in "0123456789abcdef" for c in sig)


class TestQHash:
    """qhash = sha256(normalize(query))[:16]"""

    def test_deterministic(self):
        assert qhash("Hello World") == qhash("Hello World")

    def test_only_depends_on_query(self):
        """qhash does NOT include agent_id or tools_hash."""
        h1 = qhash("query")
        h2 = qhash("query")
        assert h1 == h2

    def test_different_query_different_hash(self):
        assert qhash("Calculate ratio") != qhash("Analyze growth")


class TestKwLockHash:
    def test_deterministic(self):
        assert kw_lock_hash("my_keyword") == kw_lock_hash("my_keyword")

    def test_length(self):
        assert len(kw_lock_hash("test")) == 16
