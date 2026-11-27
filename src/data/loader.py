"""Multi-format dataset loaders for Fake.csv/True.csv, WELFake, LIAR, and standard CSVs.

All loaders map inputs into a single canonical schema without modifying raw files:
- ``title`` (str)
- ``text`` (str)
- ``label`` (int): ``1`` = FAKE, ``0`` = REAL
- ``label_text`` (str): ``"FAKE"`` or ``"REAL"``
- ``source_dataset`` (str)
- ``subject`` (str)
- ``date`` (str)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


CANONICAL_COLUMNS = [
    "title",
    "text",
    "label",
    "label_text",
    "source_dataset",
    "subject",
    "date",
]

LABEL_TO_TEXT = {0: "REAL", 1: "FAKE"}

_LIAR_COLUMNS = [
    "statement_id",
    "raw_label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

_LIAR_BINARY_MAP = {
    "true": 0,
    "mostly-true": 0,
    "false": 1,
    "barely-true": 1,
    "pants-fire": 1,
}


def _find_column(columns: pd.Index, candidates: list[str]) -> str | None:
    """Find the first matching column name case-insensitively."""
    lookup = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match is not None:
            return match
    return None


def _finalize_canonical_frame(
    *,
    title: pd.Series,
    text: pd.Series,
    label: pd.Series,
    source_dataset: str,
    subject: pd.Series | None = None,
    date: pd.Series | None = None,
) -> pd.DataFrame:
    """Assemble and validate a DataFrame matching ``CANONICAL_COLUMNS``."""
    n = len(label)
    frame = pd.DataFrame(
        {
            "title": title.fillna("").astype(str),
            "text": text.fillna("").astype(str),
            "label": label.astype("int64"),
            "label_text": label.map(LABEL_TO_TEXT),
            "source_dataset": [source_dataset] * n,
            "subject": subject.fillna("").astype(str) if subject is not None else [""] * n,
            "date": date.fillna("").astype(str) if date is not None else [""] * n,
        }
    )
    return frame[CANONICAL_COLUMNS]


def _normalize_binary_label(series: pd.Series, mapping: dict[Any, int] | None = None) -> pd.Series:
    """Convert raw labels (integers or strings) into canonical 0 (REAL) / 1 (FAKE)."""
    if mapping is not None:
        mapped = series.map(mapping)
        if mapped.isna().any():
            bad_vals = sorted(series[mapped.isna()].dropna().unique().tolist())
            raise ValueError(f"Unexpected label values encountered: {bad_vals}")
        return mapped.astype("int64")

    normalized: list[int] = []
    for val in series:
        if pd.isna(val):
            raise ValueError("Dataset contains missing (NaN) label values.")
        if isinstance(val, (int, float)) and val in (0, 1):
            normalized.append(int(val))
            continue
        text_val = str(val).strip().lower()
        if text_val in {"1", "fake", "likely fake", "false", "unreliable"}:
            normalized.append(1)
        elif text_val in {"0", "real", "likely real", "true", "reliable"}:
            normalized.append(0)
        else:
            raise ValueError(f"Unrecognized label value: {val!r}")
    return pd.Series(normalized, index=series.index, dtype="int64")


def load_isot_pair(fake_path: str | Path, true_path: str | Path) -> pd.DataFrame:
    """Load ISOT ``Fake.csv`` and ``True.csv`` into the canonical format.

    Args:
        fake_path: Path to ``Fake.csv`` (mapped to ``label = 1``).
        true_path: Path to ``True.csv`` (mapped to ``label = 0``).

    Returns:
        Combined canonical DataFrame without modifying the raw CSV files.
    """
    fake_file = Path(fake_path)
    true_file = Path(true_path)
    if not fake_file.is_file():
        raise FileNotFoundError(f"ISOT Fake.csv not found: {fake_file}")
    if not true_file.is_file():
        raise FileNotFoundError(f"ISOT True.csv not found: {true_file}")

    parts: list[pd.DataFrame] = []
    for file_path, label_value in ((fake_file, 1), (true_file, 0)):
        raw_df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
        title_col = _find_column(raw_df.columns, ["title", "headline"])
        text_col = _find_column(raw_df.columns, ["text", "content", "article"])
        if title_col is None and text_col is None:
            raise ValueError(f"Expected 'title' or 'text' column in {file_path}")

        subject_col = _find_column(raw_df.columns, ["subject", "category"])
        date_col = _find_column(raw_df.columns, ["date", "published"])

        titles = raw_df[title_col] if title_col else pd.Series([""] * len(raw_df))
        texts = raw_df[text_col] if text_col else pd.Series([""] * len(raw_df))
        labels = pd.Series([label_value] * len(raw_df), index=raw_df.index, dtype="int64")

        parts.append(
            _finalize_canonical_frame(
                title=titles,
                text=texts,
                label=labels,
                source_dataset="ISOT",
                subject=raw_df[subject_col] if subject_col else None,
                date=raw_df[date_col] if date_col else None,
            )
        )

    return pd.concat(parts, ignore_index=True)


def load_welfake(
    csv_path: str | Path,
    *,
    label_convention: str = "zenodo",
) -> pd.DataFrame:
    """Load a WELFake-format CSV into the canonical schema.

    Args:
        csv_path: Path to ``WELFake_Dataset.csv``.
        label_convention: ``"zenodo"`` (original release: ``1 = Real``, ``0 = Fake``)
            or ``"inverted"`` (some Kaggle forks: ``1 = Fake``, ``0 = Real``).

    Returns:
        Canonical DataFrame with ``1 = FAKE`` and ``0 = REAL``.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"WELFake CSV not found: {path}")

    raw_df = pd.read_csv(path, encoding="utf-8")
    title_col = _find_column(raw_df.columns, ["title", "headline"])
    text_col = _find_column(raw_df.columns, ["text", "content", "article"])
    label_col = _find_column(raw_df.columns, ["label", "target", "class"])

    if label_col is None or (title_col is None and text_col is None):
        raise ValueError(f"WELFake CSV missing required columns in {path}: found {list(raw_df.columns)}")

    raw_df = raw_df.dropna(subset=[label_col]).copy()

    if label_convention == "zenodo":
        # Original Zenodo WELFake: 1 = Real -> canonical 0, 0 = Fake -> canonical 1
        mapping = {1: 0, 0: 1, 1.0: 0, 0.0: 1, "1": 0, "0": 1}
    elif label_convention == "inverted":
        mapping = {1: 1, 0: 0, 1.0: 1, 0.0: 0, "1": 1, "0": 0}
    else:
        raise ValueError("label_convention must be 'zenodo' or 'inverted'")

    labels = _normalize_binary_label(raw_df[label_col], mapping=mapping)
    titles = raw_df[title_col] if title_col else pd.Series([""] * len(raw_df), index=raw_df.index)
    texts = raw_df[text_col] if text_col else pd.Series([""] * len(raw_df), index=raw_df.index)

    return _finalize_canonical_frame(
        title=titles,
        text=texts,
        label=labels,
        source_dataset="WELFake",
    )


def load_liar_tsv(tsv_path: str | Path, *, include_half_true: bool = False) -> pd.DataFrame:
    """Load a LIAR dataset TSV file (short claims/statements) into canonical format."""
    path = Path(tsv_path)
    if not path.is_file():
        raise FileNotFoundError(f"LIAR TSV not found: {path}")

    raw_df = pd.read_csv(path, sep="\t", header=None, names=_LIAR_COLUMNS, encoding="utf-8")
    raw_df["raw_label_clean"] = raw_df["raw_label"].astype(str).str.strip().str.lower()

    if not include_half_true:
        raw_df = raw_df[raw_df["raw_label_clean"].isin(_LIAR_BINARY_MAP.keys())].copy()
        mapping = _LIAR_BINARY_MAP
    else:
        mapping = {**_LIAR_BINARY_MAP, "half-true": 1}

    labels = raw_df["raw_label_clean"].map(mapping).astype("int64")
    return _finalize_canonical_frame(
        title=raw_df["statement"],
        text=pd.Series([""] * len(raw_df), index=raw_df.index),
        label=labels,
        source_dataset="LIAR",
        subject=raw_df["subject"],
    )


def load_standard_csv(csv_path: str | Path, *, source_name: str = "CUSTOM") -> pd.DataFrame:
    """Load a single generic CSV with ``title``/``text`` and ``label`` columns."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    raw_df = pd.read_csv(path, encoding="utf-8")
    title_col = _find_column(raw_df.columns, ["title", "headline", "claim"])
    text_col = _find_column(raw_df.columns, ["text", "content", "article", "body"])
    label_col = _find_column(raw_df.columns, ["label", "target", "class", "is_fake"])

    if label_col is None or (title_col is None and text_col is None):
        raise ValueError(f"CSV missing required text/label columns: {list(raw_df.columns)}")

    raw_df = raw_df.dropna(subset=[label_col]).copy()
    labels = _normalize_binary_label(raw_df[label_col])
    titles = raw_df[title_col] if title_col else pd.Series([""] * len(raw_df), index=raw_df.index)
    texts = raw_df[text_col] if text_col else pd.Series([""] * len(raw_df), index=raw_df.index)
    subject_col = _find_column(raw_df.columns, ["subject", "category"])
    date_col = _find_column(raw_df.columns, ["date", "published"])

    return _finalize_canonical_frame(
        title=titles,
        text=texts,
        label=labels,
        source_dataset=source_name,
        subject=raw_df[subject_col] if subject_col else None,
        date=raw_df[date_col] if date_col else None,
    )


def auto_load_dataset(
    input_path: str | Path,
    *,
    welfake_label_convention: str = "zenodo",
) -> pd.DataFrame:
    """Auto-detect and load ISOT (Fake.csv + True.csv), WELFake, LIAR, or standard CSV.

    If ``input_path`` is a directory containing ``Fake.csv`` and ``True.csv``,
    both are loaded as ISOT. Additional supported files in the directory
    (such as ``WELFake_Dataset.csv``) can also be combined when passed explicitly.
    """
    path = Path(input_path)
    if path.is_dir():
        fake_csv = path / "Fake.csv"
        true_csv = path / "True.csv"
        if fake_csv.is_file() and true_csv.is_file():
            return load_isot_pair(fake_csv, true_csv)

        welfake_candidates = sorted(
            p for p in path.glob("*.csv") if "welfake" in p.name.lower()
        )
        if welfake_candidates:
            return load_welfake(welfake_candidates[0], label_convention=welfake_label_convention)

        tsv_candidates = sorted(path.glob("*.tsv"))
        if tsv_candidates:
            return pd.concat([load_liar_tsv(p) for p in tsv_candidates], ignore_index=True)

        csv_candidates = sorted(path.glob("*.csv"))
        if csv_candidates:
            return load_standard_csv(csv_candidates[0], source_name=csv_candidates[0].stem)

        raise FileNotFoundError(f"No supported dataset files (.csv / .tsv) found in {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    lower_name = path.name.lower()
    if "welfake" in lower_name:
        return load_welfake(path, label_convention=welfake_label_convention)
    if path.suffix.lower() == ".tsv":
        return load_liar_tsv(path)
    return load_standard_csv(path, source_name=path.stem)
