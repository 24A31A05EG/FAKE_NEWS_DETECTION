"""Tests for configuration loading and news input validation."""

from __future__ import annotations

import pytest

from src.data.validator import validate_news_input
from src.utils.config_loader import get_project_root, load_config


def test_load_config_branding_and_sections() -> None:
    cfg = load_config()
    assert cfg["project"]["name"] == "FAKE_NEWS_DETECTION"
    assert cfg["input_validation"]["min_words"] == 1
    assert cfg["labels"]["uncertain"] == "UNVERIFIED"
    assert cfg["evidence_retrieval"]["never_fabricate_evidence"] is True


def test_get_project_root_points_to_workspace() -> None:
    root = get_project_root()
    assert root.name == "Fake_News_Detection_Advanced"
    assert (root / "config" / "config.yaml").is_file()


@pytest.mark.parametrize(
    "headline",
    [
        "Earthquake hits coast",
        "NASA launches new telescope",
        "Central bank cuts interest rates by 25 basis points",
        "Oil prices drop",
        "WHO updates vaccine guidance for seasonal flu",
    ],
)
def test_validator_accepts_short_headlines_without_10_word_minimum(headline: str) -> None:
    cfg = load_config()["input_validation"]
    result = validate_news_input(headline, cfg)
    assert result.is_valid is True
    assert result.error_code is None
    assert result.word_count < 10


def test_validator_accepts_long_article() -> None:
    cfg = load_config()["input_validation"]
    article = (
        "Federal officials announced today that the quarterly inflation report showed "
        "moderate cooling across housing and energy sectors. Economists noted that "
        "consumer spending remained resilient despite higher borrowing costs."
    )
    result = validate_news_input(article, cfg)
    assert result.is_valid is True
    assert result.detected_language == "en"


@pytest.mark.parametrize("bad_input", [None, "", "   \n\t  ", 12345])
def test_validator_rejects_empty_or_non_string(bad_input: object) -> None:
    cfg = load_config()["input_validation"]
    result = validate_news_input(bad_input, cfg)
    assert result.is_valid is False
    assert result.error_code in {"EMPTY_OR_INVALID_TYPE", "EMPTY_TEXT"}


def test_validator_rejects_too_short_and_too_long() -> None:
    cfg = load_config()["input_validation"]
    short_res = validate_news_input("hi", cfg)
    assert short_res.is_valid is False
    assert short_res.error_code == "TOO_SHORT"

    long_res = validate_news_input("word " * 5000, cfg)
    assert long_res.is_valid is False
    assert long_res.error_code == "TOO_LONG"


@pytest.mark.parametrize(
    "gibberish",
    [
        "asdfghjklzxcvbnm qwertyuiopasdfg",
        "aaaaaaa bbbbbbb ccccccc",
        "1234567890 !@#$%^&*()",
        "https://example.com/some-news-article",
        "bcdfghjklmnpqrstvwxyz",
    ],
)
def test_validator_rejects_gibberish_and_url_only(gibberish: str) -> None:
    cfg = load_config()["input_validation"]
    result = validate_news_input(gibberish, cfg)
    assert result.is_valid is False
    assert result.error_code == "GIBBERISH_INPUT"


@pytest.mark.parametrize(
    "foreign_text",
    [
        "El presidente anunció nuevas medidas económicas para el país hoy.",
        "Le gouvernement a publié un nouveau rapport sur la santé publique.",
        "Новости экономики сегодня выросли на два процента.",
        "आज सरकार ने नई आर्थिक नीति की घोषणा की है।",
    ],
)
def test_validator_rejects_unsupported_languages(foreign_text: str) -> None:
    cfg = load_config()["input_validation"]
    result = validate_news_input(foreign_text, cfg)
    assert result.is_valid is False
    assert result.error_code == "UNSUPPORTED_LANGUAGE"
