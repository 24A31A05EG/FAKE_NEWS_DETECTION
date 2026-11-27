"""Dataset quality inspection, reporting, and visualization utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.preprocessor import preprocess_dataframe


def inspect_dataset(
    df: pd.DataFrame,
    *,
    dataset_name: str = "dataset",
    preprocessing_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute missingness, duplicate/leakage risk, class balance, and length metrics.

    Expects a canonical DataFrame with at least ``title``, ``text``, and ``label``.
    Does not mutate the input DataFrame.
    """
    pre_df = (
        df
        if {"combined_text", "text_fingerprint", "char_count", "word_count"}.issubset(df.columns)
        else preprocess_dataframe(df, preprocessing_config=preprocessing_config)
    )

    total_rows = int(len(pre_df))
    missing_values: dict[str, dict[str, Any]] = {}
    for col in pre_df.columns:
        series = pre_df[col]
        na_count = int(series.isna().sum())
        blank_count = int((series.astype(str).str.strip() == "").sum()) if series.dtype == object or str(series.dtype) == "str" else 0
        missing_values[col] = {
            "na_count": na_count,
            "blank_count": blank_count,
            "na_pct": round((na_count / total_rows) * 100, 2) if total_rows else 0.0,
            "blank_pct": round((blank_count / total_rows) * 100, 2) if total_rows else 0.0,
        }

    empty_combined = int((pre_df["combined_text"].str.strip() == "").sum()) if total_rows else 0
    exact_text_dups = int(pre_df.duplicated(subset=["title", "text"], keep="first").sum()) if {"title", "text"}.issubset(pre_df.columns) else 0
    normalized_dups = int(pre_df.duplicated(subset=["text_fingerprint"], keep="first").sum()) if total_rows else 0

    # Conflicting-label duplicates: same normalized text appears with both REAL (0) and FAKE (1)
    if total_rows and "label" in pre_df.columns:
        label_counts_per_fp = pre_df.groupby("text_fingerprint")["label"].nunique()
        conflicting_fingerprints = int((label_counts_per_fp > 1).sum())
    else:
        conflicting_fingerprints = 0

    class_counts = pre_df["label"].value_counts().to_dict() if "label" in pre_df.columns else {}
    real_count = int(class_counts.get(0, 0))
    fake_count = int(class_counts.get(1, 0))
    total_labeled = real_count + fake_count

    class_distribution = {
        "REAL (0)": {
            "count": real_count,
            "percentage": round((real_count / total_labeled) * 100, 2) if total_labeled else 0.0,
        },
        "FAKE (1)": {
            "count": fake_count,
            "percentage": round((fake_count / total_labeled) * 100, 2) if total_labeled else 0.0,
        },
    }

    word_counts = pre_df["word_count"] if total_rows else pd.Series([], dtype="int64")
    char_counts = pre_df["char_count"] if total_rows else pd.Series([], dtype="int64")

    def _summary_stats(series: pd.Series) -> dict[str, float]:
        if len(series) == 0:
            return {"min": 0.0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "min": float(series.min()),
            "median": float(series.median()),
            "mean": round(float(series.mean()), 2),
            "p95": round(float(series.quantile(0.95)), 2),
            "max": float(series.max()),
        }

    short_inputs = int(((word_counts > 0) & (word_counts < 15)).sum()) if total_rows else 0
    long_articles = int((word_counts >= 15).sum()) if total_rows else 0

    return {
        "dataset_name": dataset_name,
        "total_rows": total_rows,
        "columns": list(pre_df.columns),
        "missing_values": missing_values,
        "data_quality": {
            "empty_combined_text_rows": empty_combined,
            "exact_title_text_duplicates": exact_text_dups,
            "normalized_fingerprint_duplicates": normalized_dups,
            "conflicting_label_fingerprints": conflicting_fingerprints,
            "short_inputs_under_15_words": short_inputs,
            "long_articles_15_plus_words": long_articles,
        },
        "class_distribution": class_distribution,
        "word_count_stats": _summary_stats(word_counts),
        "char_count_stats": _summary_stats(char_counts),
    }


def save_inspection_report(
    report: dict[str, Any],
    *,
    metrics_dir: str | Path,
    filename_prefix: str | None = None,
) -> tuple[Path, Path]:
    """Save the dataset inspection dictionary as JSON and Markdown files."""
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or report.get("dataset_name", "dataset")

    json_path = out_dir / f"{prefix}_inspection.json"
    md_path = out_dir / f"{prefix}_inspection.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    dq = report["data_quality"]
    cd = report["class_distribution"]
    wc = report["word_count_stats"]
    lines = [
        f"# Dataset Inspection Report: {report['dataset_name']}",
        "",
        f"- **Total rows:** {report['total_rows']}",
        f"- **REAL (0):** {cd['REAL (0)']['count']} ({cd['REAL (0)']['percentage']}%)",
        f"- **FAKE (1):** {cd['FAKE (1)']['count']} ({cd['FAKE (1)']['percentage']}%)",
        "",
        "## Data Quality & Leakage Checks",
        f"- **Empty combined text rows:** {dq['empty_combined_text_rows']}",
        f"- **Exact (title + text) duplicates:** {dq['exact_title_text_duplicates']}",
        f"- **Normalized fingerprint duplicates:** {dq['normalized_fingerprint_duplicates']}",
        f"- **Conflicting-label fingerprints:** {dq['conflicting_label_fingerprints']}",
        f"- **Short inputs (<15 words):** {dq['short_inputs_under_15_words']}",
        f"- **Longer articles (>=15 words):** {dq['long_articles_15_plus_words']}",
        "",
        "## Word Count Summary",
        f"- Min: {wc['min']} | Median: {wc['median']} | Mean: {wc['mean']} | P95: {wc['p95']} | Max: {wc['max']}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def generate_inspection_plots(
    df: pd.DataFrame,
    report: dict[str, Any],
    *,
    figures_dir: str | Path,
    filename_prefix: str | None = None,
) -> list[Path]:
    """Create and save inspection plots (class balance, word lengths, quality issues)."""
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or report.get("dataset_name", "dataset")

    saved_paths: list[Path] = []

    # 1. Class Distribution Bar Chart
    fig, ax = plt.subplots(figsize=(6, 4))
    cd = report["class_distribution"]
    labels = ["REAL (0)", "FAKE (1)"]
    counts = [cd["REAL (0)"]["count"], cd["FAKE (1)"]["count"]]
    bars = ax.bar(labels, counts, color=["#2E7D32", "#C62828"])
    ax.set_title(f"{report['dataset_name']} - Class Distribution")
    ax.set_ylabel("Number of Samples")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom")
    fig.tight_layout()
    class_path = out_dir / f"{prefix}_class_distribution.png"
    fig.savefig(class_path, dpi=150)
    plt.close(fig)
    saved_paths.append(class_path)

    # 2. Word Length Distribution Histogram
    pre_df = (
        df
        if "word_count" in df.columns
        else preprocess_dataframe(df)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(pre_df) > 0:
        clipped = pre_df["word_count"].clip(upper=pre_df["word_count"].quantile(0.98) if len(pre_df) > 5 else pre_df["word_count"].max())
        ax.hist(clipped, bins=min(30, max(5, len(pre_df))), color="#1F4E79", edgecolor="white")
    ax.set_title(f"{report['dataset_name']} - Word Count Distribution")
    ax.set_xlabel("Word Count (clipped at 98th percentile)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    len_path = out_dir / f"{prefix}_text_length_distribution.png"
    fig.savefig(len_path, dpi=150)
    plt.close(fig)
    saved_paths.append(len_path)

    # 3. Data Quality Summary Bar Chart
    fig, ax = plt.subplots(figsize=(7, 4))
    dq = report["data_quality"]
    issue_names = [
        "Empty Text",
        "Exact Duplicates",
        "Normalized Dups",
        "Conflicting Labels",
    ]
    issue_counts = [
        dq["empty_combined_text_rows"],
        dq["exact_title_text_duplicates"],
        dq["normalized_fingerprint_duplicates"],
        dq["conflicting_label_fingerprints"],
    ]
    bars = ax.bar(issue_names, issue_counts, color=["#D97706", "#2563EB", "#7C3AED", "#DC2626"])
    ax.set_title(f"{report['dataset_name']} - Data Quality & Leakage Checks")
    ax.set_ylabel("Row / Fingerprint Count")
    for bar, count in zip(bars, issue_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom")
    fig.tight_layout()
    dq_path = out_dir / f"{prefix}_data_quality.png"
    fig.savefig(dq_path, dpi=150)
    plt.close(fig)
    saved_paths.append(dq_path)

    return saved_paths
