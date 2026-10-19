"""Checkpoint / resume primitives for the Jiaxing batch runner.

Provides deterministic task IDs, atomic per-task shard persistence, structured
failure recording, resume-time validation, and a final merge step that only
produces {stage}_raw.csv when every task has passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── task identity ─────────────────────────────────────────────────────────

def make_task_id(task: dict[str, Any], stage: str, max_days: int | None) -> str:
    """Deterministic, stable task ID covering all semantics-changing parameters.

    The ID is a 16‑char hex SHA256 digest of a canonical JSON dump that
    includes *stage*, *max_days* and every key in *task* (sorted), except
    internal keys prefixed with ``__`` (e.g. ``__task_id``).  Skipping those
    makes the function idempotent: re-deriving the ID for a task that already
    carries its own ``__task_id`` yields the same digest, which is required by
    ``scan_shards``/``merge_shards`` (they re-compute IDs on the same task
    dicts the runner has already stamped).
    """
    canonical: dict[str, Any] = {"stage": stage, "max_days": max_days}
    for key in sorted(task.keys()):
        if key.startswith("__"):
            continue
        canonical[key] = task[key]
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _expected_days(task: dict[str, Any], stage: str) -> int:
    """Return the number of evaluation days this task is expected to produce."""
    md = task.get("max_days")
    if md is not None:
        return int(md)
    if stage == "sensitivity":
        return 30
    return 153


# ── atomic shard persistence ──────────────────────────────────────────────

_REQUIRED_COLUMNS = [
    "date", "method", "policy", "topology_index", "topology_seed",
    "total_rss_kwh", "recovered_rss_kwh", "residual_rss_kwh", "coverage",
    "movement_km", "runtime_seconds", "feasible_decisions", "optimal_decisions",
    "decisions", "served_age0_kwh", "served_backlog_kwh", "expired_kwh",
    "move_decisions", "stay_decisions", "intransit_unit_slots",
    "max_conservation_error",
    "horizon", "scenarios", "fleet", "capacity_kwh", "response_slots",
    "cvar_weight", "speed_km_per_slot", "rss_scale",
    "include_backlog", "include_future",
    "noise_percent", "repetition", "noise_seed",
]


def save_shard(
    shard_dir: Path,
    task_id: str,
    task: dict[str, Any],
    rows: list[dict[str, Any]],
) -> Path:
    """Atomically persist one task's result rows as {task_id}.csv.

    Writes to a unique temp file, flushes + fsyncs, then ``os.replace``.
    Also writes a companion ``{task_id}.json`` metadata file.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)

    # -- write CSV to temp --------------------------------------------------
    csv_path = shard_dir / f"{task_id}.csv"
    tmp_csv = shard_dir / f".{task_id}.{os.getpid()}.tmp"
    try:
        frame.to_csv(tmp_csv, index=False, encoding="utf-8-sig")
        # flush + fsync via file handle
        # On Windows, fsync on a read-only descriptor can raise EBADF.
        # Re-open read/write so the completed pandas output can be durably
        # flushed before the atomic rename.
        with open(tmp_csv, "r+b") as _f:
            _f.flush()
            os.fsync(_f.fileno())
    except BaseException:
        if tmp_csv.exists():
            tmp_csv.unlink()
        raise
    os.replace(tmp_csv, csv_path)

    # -- write companion metadata -------------------------------------------
    meta = {
        "task_id": task_id,
        "task": {k: (int(v) if isinstance(v, np.integer) else
                      float(v) if isinstance(v, np.floating) else
                      bool(v) if isinstance(v, (np.bool_,)) else v)
                 for k, v in task.items()},
        "rows": len(frame),
        "dates": sorted(frame["date"].unique().tolist()),
        "columns": list(frame.columns),
        "max_conservation_error": float(frame["max_conservation_error"].max())
        if "max_conservation_error" in frame.columns else None,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = shard_dir / f"{task_id}.json"
    tmp_meta = shard_dir / f".{task_id}.{os.getpid()}.meta.tmp"
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(tmp_meta, "r+b") as _f:
        _f.flush()
        os.fsync(_f.fileno())
    os.replace(tmp_meta, meta_path)

    # A task that previously failed leaves {task_id}.failed.json behind.  Once
    # it succeeds the failure record is stale; remove it so later scans/audits
    # do not mistake a completed task for a failed one.
    stale_fail = shard_dir / f"{task_id}.failed.json"
    if stale_fail.exists():
        stale_fail.unlink()

    return csv_path


def save_failure(
    shard_dir: Path,
    task_id: str,
    task: dict[str, Any],
    exc: BaseException,
) -> Path:
    """Record a failed task as structured JSON."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "task_id": task_id,
        "task": {k: (int(v) if isinstance(v, np.integer) else
                      float(v) if isinstance(v, np.floating) else
                      bool(v) if isinstance(v, (np.bool_,)) else v)
                 for k, v in task.items()},
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "failed_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = shard_dir / f"{task_id}.failed.json"
    tmp = shard_dir / f".{task_id}.{os.getpid()}.fail.tmp"
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(tmp, "r+b") as _f:
        _f.flush()
        os.fsync(_f.fileno())
    os.replace(tmp, path)
    return path


# ── resume-time validation ────────────────────────────────────────────────

def validate_shard(
    shard_dir: Path,
    task_id: str,
    expected_task: dict[str, Any],
    stage: str,
) -> tuple[bool, str]:
    """Validate a completed shard.  Returns (ok, reason)."""
    csv_path = shard_dir / f"{task_id}.csv"
    meta_path = shard_dir / f"{task_id}.json"

    if not csv_path.exists():
        return False, "CSV file missing"
    if not meta_path.exists():
        return False, "metadata JSON missing"

    # -- load metadata ------------------------------------------------------
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, f"metadata JSON corrupt: {e}"

    if meta.get("task_id") != task_id:
        return False, f"metadata task_id mismatch: {meta.get('task_id')} != {task_id}"

    # -- verify task parameters match ---------------------------------------
    stored_task = meta.get("task", {})
    for key in sorted(expected_task.keys()):
        expected = expected_task[key]
        stored = stored_task.get(key)
        # Normalise numpy types
        if isinstance(expected, np.integer):
            expected = int(expected)
        if isinstance(expected, np.floating):
            expected = float(expected)
        if isinstance(expected, (np.bool_,)):
            expected = bool(expected)
        if stored != expected:
            return False, f"task parameter '{key}' mismatch: stored={stored!r} expected={expected!r}"

    # -- load CSV -----------------------------------------------------------
    try:
        frame = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"CSV read error: {e}"

    # -- column check -------------------------------------------------------
    missing = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        return False, f"missing columns: {missing}"

    # -- duplicate key check ------------------------------------------------
    dup_mask = frame.duplicated(subset=["method", "topology_index", "date"], keep=False)
    if dup_mask.any():
        return False, f"{dup_mask.sum()} duplicate (method, topology_index, date) rows"

    expected_n = _expected_days(expected_task, stage)
    if len(frame) != expected_n:
        return False, f"row count {len(frame)} != expected {expected_n}"

    # -- conservation check -------------------------------------------------
    if "max_conservation_error" in frame.columns:
        max_err = frame["max_conservation_error"].max()
        if max_err > 1e-6:
            return False, f"max_conservation_error {max_err} > 1e-6"

    # -- date count check ---------------------------------------------------
    if len(frame["date"].unique()) != expected_n:
        return False, f"unique dates {len(frame['date'].unique())} != expected {expected_n}"

    return True, "ok"


def quarantine_shard(shard_dir: Path, task_id: str) -> None:
    """Move a shard's CSV + JSON into _quarantine/ so it won't be silently reused."""
    quarantine = shard_dir / "_quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    for suffix in (".csv", ".json", ".failed.json"):
        src = shard_dir / f"{task_id}{suffix}"
        if src.exists():
            dst = quarantine / f"{task_id}{suffix}"
            # If dst already exists, append a counter
            if dst.exists():
                counter = 1
                while (quarantine / f"{task_id}.{counter}{suffix}").exists():
                    counter += 1
                dst = quarantine / f"{task_id}.{counter}{suffix}"
            shutil.move(str(src), str(dst))


# Files written by the checkpoint system itself (not per-task shards) that sit
# inside the shard directory.  ``scan_shards`` must never treat these as orphan
# metadata.  Task metadata always has a 16-hex stem (see ``make_task_id``), so
# ``_is_task_id_stem`` gives a precise second guard: any ``*.json`` whose stem
# is not a well-formed task id is a control/state artifact, never orphan task
# metadata.  That guard is narrow by construction — it cannot hide a genuine
# unknown task metadata file, which is always 16 hex characters.
_CONTROL_STATE_FILES: frozenset[str] = frozenset({
    "checkpoint_state.json",
})

_TASK_ID_HEX_DIGITS: frozenset[str] = frozenset("0123456789abcdef")


def _is_task_id_stem(stem: str) -> bool:
    """Return True iff *stem* is shaped like a ``make_task_id`` digest (16 hex)."""
    return len(stem) == 16 and all(ch in _TASK_ID_HEX_DIGITS for ch in stem)


def scan_shards(
    shard_dir: Path,
    tasks: list[dict[str, Any]],
    stage: str,
    max_days: int | None,
) -> tuple[set[str], set[str], list[str]]:
    """Scan existing shards and return (valid_ids, invalid_ids, error_messages).

    *valid_ids*: task IDs whose shards passed all checks.
    *invalid_ids*: task IDs with shards that failed validation (quarantined).
    *error_messages*: human-readable descriptions of each failure.
    """
    valid: set[str] = set()
    invalid: set[str] = set()
    errors: list[str] = []

    expected_ids = {make_task_id(t, stage, max_days) for t in tasks}

    if not shard_dir.exists():
        return valid, invalid, errors

    for csv_file in sorted(shard_dir.glob("*.csv")):
        tid = csv_file.stem
        if tid.startswith("."):  # temp file
            continue
        # Find the matching task
        matching = [t for t in tasks if make_task_id(t, stage, max_days) == tid]
        if not matching:
            # Unknown shard — quarantine it
            errors.append(f"Shard {tid}: no matching task in current task list; quarantining")
            quarantine_shard(shard_dir, tid)
            invalid.add(tid)
            continue

        expected_task = matching[0]
        ok, reason = validate_shard(shard_dir, tid, expected_task, stage)
        if ok:
            valid.add(tid)
        else:
            errors.append(f"Shard {tid}: {reason}; quarantining")
            quarantine_shard(shard_dir, tid)
            invalid.add(tid)

    # Also quarantine any orphaned companion .json (metadata) without a
    # matching .csv.  ``*.failed.json`` failure records are not companion
    # metadata: they are overwritten on re-run and removed on success, so they
    # are skipped here rather than treated as orphan metadata.  Control/state
    # files written by the checkpoint system itself (e.g. checkpoint_state.json)
    # are likewise never task metadata and must not be quarantined.
    for json_file in sorted(shard_dir.glob("*.json")):
        name = json_file.name
        if name.startswith("."):
            continue
        if name in _CONTROL_STATE_FILES:
            continue
        if name.endswith(".failed.json"):
            continue
        tid = json_file.stem
        if not _is_task_id_stem(tid):
            # Not shaped like a make_task_id digest: a control/state artifact,
            # not orphan task metadata.  Genuine (unknown) task metadata always
            # has a 16-hex stem, so this guard is precise and never hides it.
            continue
        if tid in valid or tid in invalid:
            continue
        if tid in expected_ids:
            continue
        # Orphan
        errors.append(f"Orphan metadata {tid}: no matching CSV; quarantining")
        quarantine_shard(shard_dir, tid)
        invalid.add(tid)

    return valid, invalid, errors


# ── final merge ───────────────────────────────────────────────────────────

def merge_shards(
    shard_dir: Path,
    tasks: list[dict[str, Any]],
    stage: str,
    max_days: int | None,
    valid_ids: set[str],
) -> pd.DataFrame:
    """Read all valid shards and return a single sorted DataFrame."""
    frames: list[pd.DataFrame] = []
    for task in tasks:
        tid = make_task_id(task, stage, max_days)
        if tid not in valid_ids:
            continue
        csv_path = shard_dir / f"{tid}.csv"
        frames.append(pd.read_csv(csv_path))

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["method", "topology_index", "date"]).reset_index(drop=True)
    return result


def write_checkpoint_state(
    shard_dir: Path,
    tasks: list[dict[str, Any]],
    stage: str,
    max_days: int | None,
    valid_ids: set[str],
    failed_ids: set[str],
    jobs: int,
    started: float,
    protocol_path: Path,
    engine_path: Path,
    runner_path: Path,
    data_file: Path,
    event_script: Path,
) -> None:
    """Persist checkpoint state for inspection."""
    state = {
        "stage": stage,
        "max_days": max_days,
        "jobs": jobs,
        "total_tasks": len(tasks),
        "completed_valid": len(valid_ids),
        "failed": len(failed_ids),
        "pending": len(tasks) - len(valid_ids) - len(failed_ids),
        "valid_ids": sorted(valid_ids),
        "failed_ids": sorted(failed_ids),
        "elapsed_seconds": __import__("time").perf_counter() - started,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp = shard_dir / ".checkpoint_state.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(tmp, "r+b") as _f:
        _f.flush()
        os.fsync(_f.fileno())
    os.replace(tmp, shard_dir / "checkpoint_state.json")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finalize(
    out_dir: Path,
    shard_dir: Path,
    stage: str,
    tasks: list[dict[str, Any]],
    max_days: int | None,
    valid_ids: set[str],
    failed_ids: set[str],
    jobs: int,
    started: float,
    protocol_path: Path,
    engine_path: Path,
    runner_path: Path,
    data_file: Path,
    event_script: Path,
    reused_count: int = 0,
    newly_completed_count: int | None = None,
) -> Path:
    """Merge all valid shards, write {stage}_raw.csv and manifest atomically.

    Returns the path to the final CSV.  Raises RuntimeError if any tasks failed.
    """
    if failed_ids:
        raise RuntimeError(
            f"Cannot finalise: {len(failed_ids)} task(s) failed. "
            f"Checkpoint at {shard_dir}"
        )

    frame = merge_shards(shard_dir, tasks, stage, max_days, valid_ids)
    expected_total = sum(_expected_days(t, stage) for t in tasks)
    if len(frame) != expected_total:
        raise RuntimeError(
            f"Merge row count {len(frame)} != expected {expected_total}"
        )

    # -- write final CSV atomically -----------------------------------------
    final_csv = out_dir / f"{stage}_raw.csv"
    tmp_csv = out_dir / f".{stage}_raw.{os.getpid()}.tmp"
    frame.to_csv(tmp_csv, index=False, encoding="utf-8-sig")
    with open(tmp_csv, "r+b") as _f:
        _f.flush()
        os.fsync(_f.fileno())
    os.replace(tmp_csv, final_csv)

    # -- build manifest -----------------------------------------------------
    import platform
    import numpy as np
    import pandas as pd
    import scipy

    # Compute result hash
    result_hash = sha256_path(final_csv)

    elapsed = __import__("time").perf_counter() - started
    manifest = {
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "working_directory": str(out_dir.parent),
        "rows": len(frame),
        "tasks": len(tasks),
        "tasks_reused": reused_count,
        "tasks_newly_completed": (
            len(valid_ids) - reused_count
            if newly_completed_count is None else newly_completed_count
        ),
        "tasks_failed": len(failed_ids),
        "jobs": jobs,
        "max_days": max_days,
        "elapsed_seconds": elapsed,
        "valid_output_root": str(out_dir.parent),
        "protocol_sha256": sha256_path(protocol_path),
        "engine_sha256": sha256_path(engine_path),
        "runner_sha256": sha256_path(runner_path),
        "input_sha256": {
            "jiaxing_parquet": sha256_path(data_file),
            "event_reader": sha256_path(event_script),
        },
        "result_sha256": result_hash,
        "checkpoint_dir": str(shard_dir),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    manifest_path = out_dir / f"{stage}_manifest.json"
    tmp_manifest = out_dir / f".{stage}_manifest.{os.getpid()}.tmp"
    tmp_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(tmp_manifest, "r+b") as _f:
        _f.flush()
        os.fsync(_f.fileno())
    os.replace(tmp_manifest, manifest_path)

    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return final_csv
