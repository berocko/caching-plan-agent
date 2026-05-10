"""Tests for apc_cache.keyword.multilang — model registry and language detection.

Blueprint Phase 4: multi-language embedding model switching.
"""

import pytest

from apc_cache.keyword.multilang import (
    DEFAULT_MODEL_KEY,
    MODEL_REGISTRY,
    EmbedModelSpec,
    detect_query_lang,
    get_model_spec,
    is_multilingual_query,
)


class TestModelRegistry:
    def test_registry_contains_default(self):
        assert DEFAULT_MODEL_KEY in MODEL_REGISTRY

    def test_get_default_spec(self):
        spec = get_model_spec("miniLM-en")
        assert spec.name == "all-MiniLM-L6-v2"
        assert spec.ver == "v1"
        assert spec.dim == 384
        assert spec.multilingual is False

    def test_get_multilingual_spec(self):
        spec = get_model_spec("miniLM-multi")
        assert spec.multilingual is True
        assert spec.dim == 384

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown model key"):
            get_model_spec("nonexistent")

    def test_default_model_key(self):
        spec = get_model_spec()
        assert spec.name == "all-MiniLM-L6-v2"

    def test_spec_dataclass(self):
        spec = EmbedModelSpec(
            name="test-model", ver="v0", dim=768, multilingual=True, description="test"
        )
        assert spec.name == "test-model"
        assert spec.ver == "v0"
        assert spec.dim == 768
        assert spec.multilingual is True
        assert spec.description == "test"


class TestDetectQueryLang:
    def test_english_default(self):
        assert detect_query_lang("Calculate working capital ratio") == "en"

    def test_chinese(self):
        assert detect_query_lang("计算营运资金比率") == "zh"

    def test_japanese(self):
        assert detect_query_lang("運転資金比率を計算する") == "ja"

    def test_korean(self):
        assert detect_query_lang("운전 자본 비율 계산") == "ko"

    def test_arabic(self):
        assert detect_query_lang("حساب نسبة رأس المال العامل") == "ar"

    def test_russian(self):
        assert detect_query_lang("Рассчитать коэффициент оборотного капитала") == "ru"

    def test_mixed_script_detects_first_non_latin(self):
        # Chinese characters trigger zh
        assert detect_query_lang("Calculate 营运资本 ratio") == "zh"


class TestIsMultilingualQuery:
    def test_english_is_not_multilingual(self):
        assert is_multilingual_query("Calculate working capital ratio") is False

    def test_chinese_is_multilingual(self):
        assert is_multilingual_query("计算营运资金比率") is True
