"""Phase 2 CLI & programmatic pipeline for inspection, preprocessing, and splitting.

This pipeline never downloads datasets automatically and never trains models.
Raw input files remain completely untouched; cleaned outputs are written to
``data/processed/`` and reports/charts to ``reports/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.data.inspector import generate_inspection_plots, inspect_dataset, save_inspection_report
from src.data.loader import auto_load_dataset
from src.data.splitter import save_processed_splits, split_dataset_leak_free
from src.utils.config_loader import load_config, resolve_project_path


def run_data_pipeline(
    input_path: str | Path,
    *,
    dataset_name: str = "news_dataset",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load raw data, run inspection & plots, clean/deduplicate, and save splits."""
    cfg = load_config(config_path)
    pre_cfg = cfg.get("preprocessing", {})
    eval_cfg = cfg.get("evaluation", {})
    paths_cfg = cfg.get("paths", {})
    seed = int(cfg.get("project", {}).get("random_seed", 42))

    raw_df = auto_load_dataset(
        input_path,
        welfake_label_convention=str(pre_cfg.get("welfake_label_convention", "zenodo")),
    )

    raw_report = inspect_dataset(raw_df, dataset_name=f"{dataset_name}_raw", preprocessing_config=pre_cfg)
    metrics_dir = resolve_project_path(paths_cfg.get("metrics_dir", "reports/metrics"))
    figures_dir = resolve_project_path(paths_cfg.get("figures_dir", "reports/figures"))
    processed_dir = resolve_project_path(paths_cfg.get("processed_data_dir", "data/processed"))

    json_path, md_path = save_inspection_report(raw_report, metrics_dir=metrics_dir)
    figure_paths = generate_inspection_plots(raw_df, raw_report, figures_dir=figures_dir)

    split_res = split_dataset_leak_free(
        raw_df,
        val_size=float(eval_cfg.get("val_size", 0.1)),
        test_size=float(eval_cfg.get("test_size", 0.2)),
        random_seed=seed,
        preprocessing_config=pre_cfg,
    )
    split_paths = save_processed_splits(split_res, processed_dir=processed_dir, dataset_prefix=dataset_name)

    return {
        "raw_report": raw_report,
        "split_stats": split_res.stats,
        "report_paths": {"json": json_path, "markdown": md_path},
        "figure_paths": figure_paths,
        "split_paths": split_paths,
    }


def main() -> None:
    """CLI entrypoint for Phase 2 dataset inspection and preprocessing."""
    parser = argparse.ArgumentParser(description="FAKE_NEWS_DETECTION Phase 2 data inspection and preprocessing pipeline.")
    parser.add_argument("--input", required=True, help="Path to raw dataset file or directory (e.g., data/raw).")
    parser.add_argument("--name", default="news_dataset", help="Prefix name for reports and processed split files.")
    parser.add_argument("--config", default=None, help="Optional path to YAML config file.")
    args = parser.parse_args()

    summary = run_data_pipeline(args.input, dataset_name=args.name, config_path=args.config)
    print(f"Dataset inspected: {summary['raw_report']['total_rows']} raw rows")
    print(f"Cleaned & split stats: {summary['split_stats']}")


if __name__ == "__main__":
    main()
