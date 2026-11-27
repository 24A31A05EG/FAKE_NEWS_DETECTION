"""Tests for dataset quality inspection, visualizations, and leak-free splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.data.inspector import (
    generate_inspection_plots,
    inspect_dataset,
    save_inspection_report,
)
from src.data.pipeline import run_data_pipeline
from src.data.splitter import save_processed_splits, split_dataset_leak_free


def _build_synthetic_corpus() -> pd.DataFrame:
    rows = []
    for i in range(12):
        rows.append(
            {
                "title": f"Economic report number {i}",
                "text": f"Official statistical release details growth metric {i} across regional sectors.",
                "label": 0,
                "label_text": "REAL",
                "source_dataset": "SYNTH",
                "subject": "economy",
                "date": "2024-01-01",
            }
        )
        rows.append(
            {
                "title": f"Shocking secret claim {i}",
                "text": f"Unverified viral post claims miracle cure {i} hidden by authorities.",
                "label": 1,
                "label_text": "FAKE",
                "source_dataset": "SYNTH",
                "subject": "viral",
                "date": "2024-01-02",
            }
        )

    # Add an empty row, a punctuation-variant duplicate, and a conflicting-label duplicate
    rows.append(
        {
            "title": "   ",
            "text": "",
            "label": 1,
            "label_text": "FAKE",
            "source_dataset": "SYNTH",
            "subject": "",
            "date": "",
        }
    )
    rows.append(
        {
            "title": "ECONOMIC REPORT NUMBER 0!!!",
            "text": "Official statistical release details growth metric 0 across regional sectors",
            "label": 0,
            "label_text": "REAL",
            "source_dataset": "SYNTH",
            "subject": "economy",
            "date": "2024-01-01",
        }
    )
    rows.append(
        {
            "title": "Ambiguous statement",
            "text": "Identical text assigned conflicting labels.",
            "label": 0,
            "label_text": "REAL",
            "source_dataset": "SYNTH",
            "subject": "misc",
            "date": "2024-01-03",
        }
    )
    rows.append(
        {
            "title": "Ambiguous statement",
            "text": "Identical text assigned conflicting labels.",
            "label": 1,
            "label_text": "FAKE",
            "source_dataset": "SYNTH",
            "subject": "misc",
            "date": "2024-01-03",
        }
    )
    return pd.DataFrame(rows)


def test_inspect_dataset_detects_duplicates_conflicts_and_empty_rows(tmp_path: Path) -> None:
    df = _build_synthetic_corpus()
    report = inspect_dataset(df, dataset_name="synth_test")

    assert report["total_rows"] == len(df)
    assert report["data_quality"]["empty_combined_text_rows"] == 1
    assert report["data_quality"]["normalized_fingerprint_duplicates"] == 2
    assert report["data_quality"]["conflicting_label_fingerprints"] == 1

    json_path, md_path = save_inspection_report(report, metrics_dir=tmp_path / "metrics")
    assert json_path.is_file()
    assert md_path.is_file()

    plots = generate_inspection_plots(df, report, figures_dir=tmp_path / "figures")
    assert len(plots) == 3
    assert all(p.is_file() and p.stat().st_size > 0 for p in plots)


def test_split_dataset_leak_free_is_reproducible_and_disjoint(tmp_path: Path) -> None:
    df = _build_synthetic_corpus()
    split_a = split_dataset_leak_free(df, val_size=0.15, test_size=0.2, random_seed=42)
    split_b = split_dataset_leak_free(df, val_size=0.15, test_size=0.2, random_seed=42)

    assert split_a.stats["dropped_empty"] == 1
    assert split_a.stats["dropped_conflicting_labels"] == 2
    assert split_a.stats["dropped_duplicates"] == 1
    assert split_a.stats["cleaned_rows"] == 24

    # Exact reproducibility
    assert list(split_a.train_df["text_fingerprint"]) == list(split_b.train_df["text_fingerprint"])
    assert list(split_a.val_df["text_fingerprint"]) == list(split_b.val_df["text_fingerprint"])
    assert list(split_a.test_df["text_fingerprint"]) == list(split_b.test_df["text_fingerprint"])

    # Zero leakage across splits
    train_fps = set(split_a.train_df["text_fingerprint"])
    val_fps = set(split_a.val_df["text_fingerprint"])
    test_fps = set(split_a.test_df["text_fingerprint"])
    assert train_fps.isdisjoint(val_fps)
    assert train_fps.isdisjoint(test_fps)
    assert val_fps.isdisjoint(test_fps)

    saved = save_processed_splits(split_a, processed_dir=tmp_path / "processed", dataset_prefix="synth")
    assert all(path.is_file() for path in saved.values())


def test_run_data_pipeline_keeps_raw_untouched(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df = _build_synthetic_corpus()[["title", "text", "label"]]
    raw_csv = raw_dir / "custom_news.csv"
    df.to_csv(raw_csv, index=False)
    raw_snapshot = raw_csv.read_bytes()

    cfg_path = tmp_path / "config.yaml"
    cfg_data = {
        "project": {"name": "FAKE_NEWS_DETECTION", "random_seed": 42},
        "paths": {
            "processed_data_dir": str(tmp_path / "processed"),
            "metrics_dir": str(tmp_path / "metrics"),
            "figures_dir": str(tmp_path / "figures"),
        },
        "evaluation": {"val_size": 0.15, "test_size": 0.2},
        "preprocessing": {"lowercase": True, "strip_html": True, "strip_urls": True, "strip_publisher_datelines": True},
    }
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    summary = run_data_pipeline(raw_csv, dataset_name="unit_pipeline", config_path=cfg_path)
    assert raw_csv.read_bytes() == raw_snapshot
    assert summary["report_paths"]["json"].is_file()
    assert summary["report_paths"]["markdown"].is_file()
    assert len(summary["figure_paths"]) == 3
    assert all(p.is_file() for p in summary["split_paths"].values())
