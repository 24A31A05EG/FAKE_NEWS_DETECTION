"""Leak-free deduplication and reproducible train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessor import preprocess_dataframe


@dataclass(frozen=True)
class SplitResult:
    """Container holding cleaned dataset splits and deduplication stats."""

    cleaned_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    stats: dict[str, int]


def deduplicate_and_clean(
    df: pd.DataFrame,
    *,
    preprocessing_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Preprocess text, drop empty rows, and remove duplicate/conflicting fingerprints.

    Deduplicating *before* splitting prevents the same news article (or minor
    punctuation/whitespace variants) from leaking across train, validation, and test sets.
    """
    pre_df = preprocess_dataframe(df, preprocessing_config=preprocessing_config)
    initial_rows = len(pre_df)

    # 1. Remove rows with empty combined text after cleaning
    non_empty = pre_df[pre_df["combined_text"].str.strip() != ""].copy()
    dropped_empty = initial_rows - len(non_empty)

    # 2. Drop rows whose normalized fingerprint has conflicting labels (0 and 1)
    fp_label_nunique = non_empty.groupby("text_fingerprint")["label"].transform("nunique")
    conflict_mask = fp_label_nunique > 1
    dropped_conflicting = int(conflict_mask.sum())
    no_conflicts = non_empty[~conflict_mask].copy()

    # 3. Deduplicate by normalized text fingerprint, keeping first deterministic row
    deduped = no_conflicts.drop_duplicates(subset=["text_fingerprint"], keep="first").reset_index(drop=True)
    dropped_duplicates = len(no_conflicts) - len(deduped)

    stats = {
        "initial_rows": initial_rows,
        "dropped_empty": dropped_empty,
        "dropped_conflicting_labels": dropped_conflicting,
        "dropped_duplicates": dropped_duplicates,
        "cleaned_rows": len(deduped),
    }
    return deduped, stats


def split_dataset_leak_free(
    df: pd.DataFrame,
    *,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_seed: int = 42,
    preprocessing_config: dict[str, Any] | None = None,
) -> SplitResult:
    """Clean, deduplicate, and reproducibly split data into train, val, and test sets.

    Args:
        df: Canonical dataset DataFrame.
        val_size: Fraction of the cleaned dataset reserved for validation.
        test_size: Fraction of the cleaned dataset reserved for testing.
        random_seed: Random state for reproducible stratified splitting.
        preprocessing_config: Optional preprocessing configuration dictionary.

    Returns:
        :class:`SplitResult` with guaranteed zero fingerprint overlap across splits.
    """
    if not (0.0 < val_size < 0.5) or not (0.0 < test_size < 0.5) or (val_size + test_size >= 1.0):
        raise ValueError("val_size and test_size must be positive fractions summing to less than 1.0.")

    cleaned_df, stats = deduplicate_and_clean(df, preprocessing_config=preprocessing_config)
    if len(cleaned_df) < 10:
        raise ValueError(f"At least 10 unique cleaned rows are required for 3-way stratified splitting (got {len(cleaned_df)}).")

    stratify_labels = cleaned_df["label"] if cleaned_df["label"].nunique() > 1 else None
    train_val_df, test_df = train_test_split(
        cleaned_df,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_labels,
    )

    relative_val_size = val_size / (1.0 - test_size)
    stratify_tv = train_val_df["label"] if train_val_df["label"].nunique() > 1 else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_seed,
        stratify=stratify_tv,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Explicit leakage verification
    train_fps = set(train_df["text_fingerprint"])
    val_fps = set(val_df["text_fingerprint"])
    test_fps = set(test_df["text_fingerprint"])
    assert train_fps.isdisjoint(val_fps), "Data leakage detected between train and validation sets!"
    assert train_fps.isdisjoint(test_fps), "Data leakage detected between train and test sets!"
    assert val_fps.isdisjoint(test_fps), "Data leakage detected between validation and test sets!"

    full_stats = {
        **stats,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }

    return SplitResult(
        cleaned_df=cleaned_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        stats=full_stats,
    )


def save_processed_splits(
    split_result: SplitResult,
    *,
    processed_dir: str | Path,
    dataset_prefix: str = "dataset",
) -> dict[str, Path]:
    """Save cleaned dataset and train/val/test CSVs separately from raw datasets."""
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "cleaned": out_dir / f"{dataset_prefix}_cleaned.csv",
        "train": out_dir / f"{dataset_prefix}_train.csv",
        "val": out_dir / f"{dataset_prefix}_val.csv",
        "test": out_dir / f"{dataset_prefix}_test.csv",
    }

    split_result.cleaned_df.to_csv(outputs["cleaned"], index=False, encoding="utf-8")
    split_result.train_df.to_csv(outputs["train"], index=False, encoding="utf-8")
    split_result.val_df.to_csv(outputs["val"], index=False, encoding="utf-8")
    split_result.test_df.to_csv(outputs["test"], index=False, encoding="utf-8")
    return outputs
