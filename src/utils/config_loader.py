"""Configuration loading and path resolution helpers for FAKE_NEWS_DETECTION."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "config.yaml"


def get_project_root() -> Path:
    """Return the root directory of the Fake_News_Detection_Advanced workspace."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the central YAML configuration file.

    Args:
        config_path: Optional path to a YAML config file. Defaults to
            ``<project_root>/config/config.yaml``.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML file is empty or does not parse into a mapping.
    """
    resolved_path = (
        Path(config_path)
        if config_path is not None
        else get_project_root() / DEFAULT_CONFIG_RELATIVE_PATH
    )
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {resolved_path}")

    return config


def resolve_project_path(relative_or_abs_path: str | Path, project_root: Path | None = None) -> Path:
    """Resolve a path relative to the workspace root unless already absolute."""
    path = Path(relative_or_abs_path)
    if path.is_absolute():
        return path
    root = project_root if project_root is not None else get_project_root()
    return root / path
