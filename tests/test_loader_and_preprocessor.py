"""Tests for Fake.csv/True.csv, WELFake, LIAR loaders and text preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import (
    CANONICAL_COLUMNS,
    auto_load_dataset,
    load_isot_pair,
    load_liar_tsv,
    load_welfake,
)
from src.data.preprocessor import (
    clean_text,
    combine_title_and_text,
    compute_text_fingerprint,
    preprocess_dataframe,
)


def test_load_isot_pair_canonical_and_raw_unchanged(tmp_path: Path) -> None:
    fake_path = tmp_path / "Fake.csv"
    true_path = tmp_path / "True.csv"

    fake_csv = (
        "title,text,subject,date\n"
        '"Shocking secret","Unverified claim body","News","Jan 1, 2017"\n'
    )
    true_csv = (
        "title,text,subject,date\n"
        '"Senate passes budget","WASHINGTON (Reuters) - The Senate voted today.","politicsNews","Jan 2, 2017"\n'
    )
    fake_path.write_text(fake_csv, encoding="utf-8")
    true_path.write_text(true_csv, encoding="utf-8")

    df = load_isot_pair(fake_path, true_path)
    assert list(df.columns) == CANONICAL_COLUMNS
    assert len(df) == 2
    assert df.loc[0, "label"] == 1
    assert df.loc[0, "label_text"] == "FAKE"
    assert df.loc[1, "label"] == 0
    assert df.loc[1, "label_text"] == "REAL"
    # Verify raw files remain byte-for-byte unchanged
    assert fake_path.read_text(encoding="utf-8") == fake_csv
    assert true_path.read_text(encoding="utf-8") == true_csv


def test_load_welfake_supports_zenodo_and_inverted_conventions(tmp_path: Path) -> None:
    csv_path = tmp_path / "WELFake_Dataset.csv"
    pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "title": ["Verified report", "Fabricated story"],
            "text": ["Official agency data released.", "Secret miracle cure discovered."],
            "label": [1, 0],
        }
    ).to_csv(csv_path, index=False)

    zenodo_df = load_welfake(csv_path, label_convention="zenodo")
    assert list(zenodo_df["label"]) == [0, 1]
    assert list(zenodo_df["label_text"]) == ["REAL", "FAKE"]

    inverted_df = load_welfake(csv_path, label_convention="inverted")
    assert list(inverted_df["label"]) == [1, 0]
    assert list(inverted_df["label_text"]) == ["FAKE", "REAL"]


def test_load_liar_tsv_maps_labels_and_filters_half_true(tmp_path: Path) -> None:
    tsv_path = tmp_path / "sample.tsv"
    rows = [
        ["1.json", "true", "Unemployment fell last year.", "jobs", "sp1", "gov", "TX", "R", 0, 0, 0, 1, 0, "speech"],
        ["2.json", "pants-fire", "The moon is made of cheese.", "science", "sp2", "none", "CA", "D", 0, 1, 0, 0, 1, "tv"],
        ["3.json", "half-true", "Taxes changed slightly.", "tax", "sp3", "sen", "NY", "I", 0, 0, 1, 0, 0, "interview"],
    ]
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", header=False, index=False)

    df = load_liar_tsv(tsv_path, include_half_true=False)
    assert len(df) == 2
    assert list(df["label"]) == [0, 1]
    assert list(df["source_dataset"]) == ["LIAR", "LIAR"]


def test_auto_load_dataset_detects_isot_directory(tmp_path: Path) -> None:
    pd.DataFrame({"title": ["Fake title"], "text": ["Fake body"]}).to_csv(tmp_path / "Fake.csv", index=False)
    pd.DataFrame({"title": ["Real title"], "text": ["Real body"]}).to_csv(tmp_path / "True.csv", index=False)

    df = auto_load_dataset(tmp_path)
    assert len(df) == 2
    assert set(df["source_dataset"]) == {"ISOT"}


def test_clean_text_strips_html_urls_and_reuters_dateline() -> None:
    raw = "<p>WASHINGTON (Reuters) - Central bank &amp; treasury met today. https://reuters.com/test</p>"
    cleaned = clean_text(raw)
    assert "reuters" not in cleaned
    assert "https://" not in cleaned
    assert "<p>" not in cleaned
    assert cleaned == "central bank & treasury met today."


def test_combine_title_and_text_avoids_duplicate_prefix_and_handles_headline_only() -> None:
    headline_only = combine_title_and_text("GDP rises 2 percent", None)
    assert headline_only == "gdp rises 2 percent"

    no_dup = combine_title_and_text("Market rallies", "Market rallies after rate pause.")
    assert no_dup == "market rallies after rate pause."

    combined = combine_title_and_text("Market rallies", "Stocks climbed sharply on Friday.")
    assert combined == "market rallies. stocks climbed sharply on friday."


def test_fingerprint_matches_across_casing_spacing_and_punctuation() -> None:
    fp1 = compute_text_fingerprint("Breaking: Central Bank Cuts Rates!")
    fp2 = compute_text_fingerprint("breaking   central bank cuts rates")
    assert fp1 == fp2


def test_preprocess_dataframe_adds_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "title": ["Headline one"],
            "text": ["WASHINGTON (AP) - Body text here."],
            "label": [0],
            "label_text": ["REAL"],
            "source_dataset": ["ISOT"],
            "subject": ["politics"],
            "date": ["2020-01-01"],
        }
    )
    out = preprocess_dataframe(df)
    for col in ("clean_title", "clean_text", "combined_text", "char_count", "word_count", "text_fingerprint"):
        assert col in out.columns
    assert "ap" not in out.loc[0, "clean_text"]
