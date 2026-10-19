"""Configurable data / script locations for reproducible experiments.

Hard-coded absolute paths from the original research machine are replaced by
environment variables (preferred) or a local ``config.toml`` next to the
repository root.

Environment variables
---------------------
EDVF_RSS_ROOT
    Repository root (defaults to the parent of the ``edvf_rss`` package).
EDVF_RSS_JIAXING_PARQUET
    Path to ``jiaxing_transactions_clean.parquet``.
EDVF_RSS_JIAXING_EVENT
    Path to the Jiaxing replay helper script that exposes
    ``build_replay_data`` and ``build_topology``.
EDVF_RSS_URBANEV_ROOT
    Path to the UrbanEV raw station folder containing
    ``station_information.csv``, ``station_distance.csv`` and ``charge_5min/``.
EDVF_RSS_OUTPUT_ROOT
    Where experiment outputs are written (default: ``<repo>/outputs``).
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_toml() -> dict:
    cfg = REPO_ROOT / "config.toml"
    if not cfg.exists() or tomllib is None:
        return {}
    # Accept UTF-8 with optional BOM (Windows editors often add one).
    text = cfg.read_text(encoding="utf-8-sig")
    return tomllib.loads(text)


def _cfg_get(section: str, key: str) -> str | None:
    data = _load_toml()
    value = data.get(section, {}).get(key)
    return str(value) if value else None


def _resolve(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def env_or_cfg(env_name: str, section: str, key: str, default: Path | None = None) -> Path:
    raw = os.environ.get(env_name) or _cfg_get(section, key)
    if raw:
        return _resolve(raw)
    if default is not None:
        return default if default.is_absolute() else (REPO_ROOT / default).resolve()
    raise FileNotFoundError(
        f"Set {env_name} or config.toml [{section}] {key}=... "
        f"(see data/README.md)."
    )


def repo_root() -> Path:
    return Path(os.environ.get("EDVF_RSS_ROOT", REPO_ROOT)).resolve()


def output_root() -> Path:
    return env_or_cfg(
        "EDVF_RSS_OUTPUT_ROOT",
        "paths",
        "output_root",
        default=repo_root() / "outputs",
    )


def jiaxing_parquet() -> Path:
    return env_or_cfg("EDVF_RSS_JIAXING_PARQUET", "paths", "jiaxing_parquet")


def jiaxing_event_script() -> Path:
    return env_or_cfg("EDVF_RSS_JIAXING_EVENT", "paths", "jiaxing_event")


def urbanev_root() -> Path:
    return env_or_cfg("EDVF_RSS_URBANEV_ROOT", "paths", "urbanev_root")
