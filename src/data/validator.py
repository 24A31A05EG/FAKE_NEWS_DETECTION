"""Input validation for headlines, short claims, and long news articles."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 42
    _HAS_LANGDETECT = True
except ImportError:  # pragma: no cover - covered via fallback when langdetect is absent
    _HAS_LANGDETECT = False


_VOWELS = set("aeiouy")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_URL_ONLY_RE = re.compile(r"^(https?://\S+|www\.\S+)$", re.IGNORECASE)

# Common English function words used by the lightweight offline fallback when
# langdetect is unavailable or text is too short for statistical n-grams.
# Avoids single-letter words like "a" that also appear frequently in French/Spanish.
_ENGLISH_HINTS = frozenset(
    {
        "the", "is", "are", "was", "were", "be", "been", "an", "and", "or",
        "of", "to", "in", "on", "at", "for", "with", "by", "from", "as", "that",
        "this", "it", "not", "no", "new", "says", "said", "will", "has", "have",
        "had", "after", "over", "under", "about", "before", "report",
        "reports", "study", "government", "president", "official", "officials",
        "people", "market", "economy", "bill", "court", "police", "health",
    }
)

# High-signal non-English Latin-script stopwords to catch common unsupported
# languages even when langdetect is not installed.
_NON_ENGLISH_LATIN_HINTS = frozenset(
    {
        "el", "la", "los", "las", "una", "con", "por", "para", "como", "pero",
        "que", "del", "esta", "este", "son", "le", "les", "des", "une", "un",
        "dans", "pour", "avec", "est", "sont", "pas", "sur", "du", "au", "aux",
        "der", "die", "das", "und", "ist", "nicht", "ein", "eine", "mit", "auf",
        "aufgrund", "nach", "über", "não", "com", "uma", "dos", "che", "sono",
        "della", "nella",
    }
)


@dataclass(frozen=True)
class ValidationResult:
    """Structured result returned by :func:`validate_news_input`."""

    is_valid: bool
    error_code: str | None
    reason: str
    normalized_text: str
    char_count: int
    word_count: int
    detected_language: str | None


def _max_repeated_char_run(text: str) -> int:
    """Return the length of the longest consecutive identical-character run."""
    if not text:
        return 0
    max_run = 1
    current_run = 1
    prev_char = text[0].lower()
    for ch in text[1:]:
        low = ch.lower()
        if low == prev_char and low.isalpha():
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1
            prev_char = low
    return max_run


def _has_non_latin_letters(text: str) -> bool:
    """Return True if the text contains alphabetic characters from non-Latin scripts."""
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            if "LATIN" not in name:
                return True
    return False


def _detect_language(text: str, words: list[str]) -> tuple[str | None, float]:
    """Detect language code and confidence without falsely rejecting short English headlines."""
    if _has_non_latin_letters(text):
        return "non-latin", 0.99

    lowered_words = [w.strip(".,!?;:\"'()[]{}").lower() for w in words]
    lowered_words = [w for w in lowered_words if w.isalpha()]

    non_en_hits = sum(1 for w in lowered_words if w in _NON_ENGLISH_LATIN_HINTS)
    en_hits = sum(1 for w in lowered_words if w in _ENGLISH_HINTS)

    if len(lowered_words) >= 3 and non_en_hits >= 2 and non_en_hits > en_hits:
        return "non-en", 0.90

    if _HAS_LANGDETECT and len(text) >= 25 and len(lowered_words) >= 5:
        try:
            candidates = detect_langs(text)
            if candidates:
                top = candidates[0]
                return top.lang, float(top.prob)
        except LangDetectException:
            pass

    if en_hits > 0 or all(ord(c) < 128 for c in text):
        return "en", 0.70

    # Diacritic-heavy Latin words with no English hints
    non_ascii_alpha = sum(1 for ch in text if ch.isalpha() and ord(ch) >= 128)
    if non_ascii_alpha >= 2 and en_hits == 0:
        return "non-en", 0.85

    return "en", 0.55


def _looks_like_gibberish(
    text: str,
    words: list[str],
    min_alpha_ratio: float,
    max_repeated_char_run: int,
    min_vowel_ratio: float,
) -> tuple[bool, str]:
    """Check whether text looks like keyboard mash, symbol spam, or gibberish."""
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return True, "Text contains only whitespace."

    alpha_chars = [ch for ch in non_space if ch.isalpha()]
    alpha_ratio = len(alpha_chars) / len(non_space)
    if alpha_ratio < min_alpha_ratio:
        return True, "Text contains too few alphabetic characters (looks like symbols or numbers only)."

    if _URL_ONLY_RE.match(text):
        return True, "Please paste a headline, claim, or article text rather than only a URL."

    if _max_repeated_char_run(text) > max_repeated_char_run:
        return True, "Text contains excessive repeated characters (looks like gibberish)."

    alpha_tokens = [re.sub(r"[^a-zA-Z]", "", w) for w in words]
    alpha_tokens = [t for t in alpha_tokens if t]
    if not alpha_tokens:
        return True, "Text does not contain valid words."

    # Check vowel ratio across alphabetic characters (skip short all-caps acronyms <= 5 chars).
    non_acronym_chars = "".join(
        t.lower() for t in alpha_tokens if not (t.isupper() and len(t) <= 5)
    )
    if len(non_acronym_chars) >= 6:
        vowel_count = sum(1 for ch in non_acronym_chars if ch in _VOWELS)
        if (vowel_count / len(non_acronym_chars)) < min_vowel_ratio:
            return True, "Text has an abnormal consonant-to-vowel pattern (looks like gibberish)."

    # Catch very long unpronounceable tokens (e.g., "asdfghjklzxcvbnm")
    for token in alpha_tokens:
        if len(token) >= 10 and not (token.isupper() and len(token) <= 6):
            token_vowels = sum(1 for ch in token.lower() if ch in _VOWELS)
            if token_vowels == 0 or (token_vowels / len(token)) < 0.12:
                return True, "Text contains keyboard-mash or unpronounceable tokens."

    return False, ""


def validate_news_input(
    text: Any,
    validation_config: dict[str, Any] | None = None,
) -> ValidationResult:
    """Validate headlines, short claims, and long news articles.

    Accepts short headlines and claims without imposing an arbitrary 10-word
    minimum while rejecting empty inputs, gibberish, and unsupported languages.

    Args:
        text: Raw input text to validate.
        validation_config: Optional ``input_validation`` dictionary from
            ``config/config.yaml``.

    Returns:
        A :class:`ValidationResult` describing validity and diagnostics.
    """
    cfg = validation_config or {}
    min_chars = int(cfg.get("min_characters", 3))
    min_words = int(cfg.get("min_words", 1))
    max_chars = int(cfg.get("max_characters", 20000))
    supported_languages = set(cfg.get("supported_languages", ["en"]))

    gib_cfg = cfg.get("gibberish_checks", {})
    min_alpha_ratio = float(gib_cfg.get("min_alpha_ratio", 0.45))
    max_repeated_run = int(gib_cfg.get("max_repeated_char_run", 5))
    min_vowel_ratio = float(gib_cfg.get("min_vowel_ratio", 0.15))

    if text is None or not isinstance(text, str):
        return ValidationResult(
            is_valid=False,
            error_code="EMPTY_OR_INVALID_TYPE",
            reason="Input must be a non-empty text string.",
            normalized_text="",
            char_count=0,
            word_count=0,
            detected_language=None,
        )

    normalized = _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()
    char_count = len(normalized)
    words = normalized.split() if normalized else []
    word_count = len(words)

    if char_count == 0:
        return ValidationResult(
            is_valid=False,
            error_code="EMPTY_TEXT",
            reason="Input text is empty or contains only whitespace.",
            normalized_text="",
            char_count=0,
            word_count=0,
            detected_language=None,
        )

    if char_count < min_chars or word_count < min_words:
        return ValidationResult(
            is_valid=False,
            error_code="TOO_SHORT",
            reason=f"Input must have at least {min_chars} characters and {min_words} word(s).",
            normalized_text=normalized,
            char_count=char_count,
            word_count=word_count,
            detected_language=None,
        )

    if char_count > max_chars:
        return ValidationResult(
            is_valid=False,
            error_code="TOO_LONG",
            reason=f"Input exceeds the maximum allowed length of {max_chars} characters.",
            normalized_text=normalized,
            char_count=char_count,
            word_count=word_count,
            detected_language=None,
        )

    detected_lang, lang_confidence = _detect_language(normalized, words)
    if detected_lang is not None and detected_lang not in supported_languages and lang_confidence >= 0.80:
        return ValidationResult(
            is_valid=False,
            error_code="UNSUPPORTED_LANGUAGE",
            reason=f"Unsupported language detected ('{detected_lang}'). Supported languages: {sorted(supported_languages)}.",
            normalized_text=normalized,
            char_count=char_count,
            word_count=word_count,
            detected_language=detected_lang,
        )

    is_gibberish, gib_reason = _looks_like_gibberish(
        normalized,
        words=words,
        min_alpha_ratio=min_alpha_ratio,
        max_repeated_char_run=max_repeated_run,
        min_vowel_ratio=min_vowel_ratio,
    )
    if is_gibberish:
        return ValidationResult(
            is_valid=False,
            error_code="GIBBERISH_INPUT",
            reason=gib_reason,
            normalized_text=normalized,
            char_count=char_count,
            word_count=word_count,
            detected_language=detected_lang,
        )

    return ValidationResult(
        is_valid=True,
        error_code=None,
        reason="Input is valid.",
        normalized_text=normalized,
        char_count=char_count,
        word_count=word_count,
        detected_language=detected_lang,
    )
