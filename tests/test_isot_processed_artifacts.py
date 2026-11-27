"""Verification tests for the processed ISOT dataset artifacts and reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.utils.config_loader import get_project_root


ROOT = get_project_root()
PROCESSED_DIR = ROOT / "data" / "processed"
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"

HAS_ISOT_PROCESSED = (PROCESSED_DIR / "isot_train.csv").is_file()


@pytest.mark.skipif(not HAS_ISOT_PROCESSED, reason="Real ISOT processed splits not present in this environment.")
def test_isot_processed_splits_exist_non_empty_and_disjoint() -> None:
    cleaned = pd.read_csv(PROCESSED_DIR / "isot_cleaned.csv")
    train = pd.read_csv(PROCESSED_DIR / "isot_train.csv")
    val = pd.read_csv(PROCESSED_DIR / "isot_val.csv")
    test = pd.read_csv(PROCESSED_DIR / "isot_test.csv")

    assert len(train) + len(val) + len(test) == len(cleaned)
    assert len(cleaned) == 38827

    for split_df in (cleaned, train, val, test):
        assert split_df["combined_text"].isna().sum() == 0
        assert (split_df["combined_text"].astype(str).str.strip() == "").sum() == 0
        assert set(split_df["label"].unique()) == {0, 1}

    train_fps = set(train["text_fingerprint"])
    val_fps = set(val["text_fingerprint"])
    test_fps = set(test["text_fingerprint"])
    assert train_fps.isdisjoint(val_fps)
    assert train_fps.isdisjoint(test_fps)
    assert val_fps.isdisjoint(test_fps)


@pytest.mark.skipif(not HAS_ISOT_PROCESSED, reason="Real ISOT processed splits not present in this environment.")
def test_isot_inspection_reports_and_charts_exist() -> None:
    json_path = METRICS_DIR / "isot_raw_inspection.json"
    md_path = METRICS_DIR / "isot_raw_inspection.md"
    assert json_path.is_file()
    assert md_path.is_file()

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["total_rows"] == 44898
    assert report["data_quality"]["conflicting_label_fingerprints"] == 0

    expected_figs = [
        FIGURES_DIR / "isot_raw_class_distribution.png",
        FIGURES_DIR / "isot_raw_text_length_distribution.png",
        FIGURES_DIR / "isot_raw_data_quality.png",
    ]
    for fig in expected_figs:
        assert fig.is_file() and fig.stat().st_size > 0
