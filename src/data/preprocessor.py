"""Robust text preprocessing for headlines, short claims, and full news articles."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from typing import Any

import pandas as pd


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_ONLY_RE = re.compile(r"[^\w\s]")

# Leading publisher datelines common in ISOT True.csv (e.g., "WASHINGTON (Reuters) - ")
# Stripping these prevents models from cheating on agency names instead of content.
_LEADING_DATELINE_RE = re.compile(
    r"^\s*(?:[A-Z][A-Za-z.\s,/&-]{1,40}\s*)?\(\s*(?:Reuters|AP|AFP|UPI|Bloomberg|CNN|BBC)\s*\)\s*[-—:]+\s*",
    re.IGNORECASE,
)

# Common trailing boilerplate tags found in scraped fake/real corpora
_TRAILING_BOILERPLATE_RE = re.compile(
    r"(?:featured image via[^\n]*|21st century wire says[^\n]*|read more:?\s*(?:https?://\S+)?)\s*$",
    re.IGNORECASE,
)


def clean_text(
    text: Any,
    *,
    lowercase: bool = True,
    strip_html: bool = True,
    strip_urls: bool = True,
    strip_publisher_datelines: bool = True,
) -> str:
    """Clean and normalize a single headline, claim, or article body.

    Args:
        text: Input value (handles ``None`` and ``NaN`` safely).
        lowercase: Whether to convert cleaned text to lowercase.
        strip_html: Whether to unescape HTML entities and strip HTML tags.
        strip_urls: Whether to remove URLs and email addresses.
        strip_publisher_datelines: Whether to strip leading wire-service
            datelines (e.g. ``WASHINGTON (Reuters) - ``) to reduce source leakage.

    Returns:
        Normalized plain-text string (empty string if input was missing/blank).
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    value = str(text)
    if not value.strip():
        return ""

    value = unicodedata.normalize("NFKC", value)

    if strip_html:
        value = html.unescape(value)
        value = _HTML_TAG_RE.sub(" ", value)

    if strip_publisher_datelines:
        value = _LEADING_DATELINE_RE.sub("", value)
        value = _TRAILING_BOILERPLATE_RE.sub("", value)

    if strip_urls:
        value = _URL_RE.sub(" ", value)
        value = _EMAIL_RE.sub(" ", value)

    value = _WHITESPACE_RE.sub(" ", value).strip()

    if lowercase:
        value = value.lower()

    return value


def combine_title_and_text(title: Any, text: Any, **clean_kwargs: Any) -> str:
    """Combine cleaned headline and article body without duplicating identical prefixes.

    Works seamlessly for headline-only inputs, body-only inputs, or full articles.
    """
    cleaned_title = clean_text(title, **clean_kwargs)
    cleaned_body = clean_text(text, **clean_kwargs)

    if cleaned_title and cleaned_body:
        if cleaned_body.startswith(cleaned_title):
            return cleaned_body
        return f"{cleaned_title}. {cleaned_body}" if not cleaned_title.endswith((".", "!", "?")) else f"{cleaned_title} {cleaned_body}"
    return cleaned_title or cleaned_body


def compute_text_fingerprint(text: str) -> str:
    """Generate a normalized SHA-256 fingerprint for leak-free deduplication.

    Lowercases, strips punctuation, and collapses whitespace so near-identical
    copies (differing only in casing, spacing, or punctuation) match reliably.
    """
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    normalized = _PUNCT_ONLY_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def preprocess_dataframe(
    df: pd.DataFrame,
    preprocessing_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply clean text preprocessing to a canonical dataset DataFrame.

    Preserves the input DataFrame by returning a new copy with ``clean_title``,
    ``clean_text``, ``combined_text``, ``char_count``, ``word_count``, and
    ``text_fingerprint`` columns.
    """
    cfg = preprocessing_config or {}
    clean_kwargs = {
        "lowercase": bool(cfg.get("lowercase", True)),
        "strip_html": bool(cfg.get("strip_html", True)),
        "strip_urls": bool(cfg.get("strip_urls", True)),
        "strip_publisher_datelines": bool(cfg.get("strip_publisher_datelines", True)),
    }

    result = df.copy()
    titles = result["title"] if "title" in result.columns else pd.Series([""] * len(result), index=result.index)
    bodies = result["text"] if "text" in result.columns else pd.Series([""] * len(result), index=result.index)

    result["clean_title"] = [clean_text(t, **clean_kwargs) for t in titles]
    result["clean_text"] = [clean_text(b, **clean_kwargs) for b in bodies]
    result["combined_text"] = [
        combine_title_and_text(t, b, **clean_kwargs)
        for t, b in zip(titles, bodies)
    ]
    result["char_count"] = result["combined_text"].str.len()
    result["word_count"] = result["combined_text"].apply(lambda s: len(s.split()) if s else 0)
    result["text_fingerprint"] = result["combined_text"].apply(compute_text_fingerprint)

    return result
