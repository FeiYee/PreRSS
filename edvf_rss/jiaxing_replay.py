"""Build the reviewer-revision Jiaxing RSS replay from the raw Figshare CSV.

Main changes relative to the submitted construction:
1) target energy is estimated only from pre-July-2021 normal/full-charge data;
2) July 2021 is used only to select shrinkage hyperparameters;
3) anonymized user history is optionally blended with station/time history;
4) the primary task definition uses binary cause eligibility, not 0.7/0.9
   continuous energy shrinkage;
5) original and conservative weights are retained only as sensitivity variants.

Outputs are deterministic and intended for EDVF/EDF/oracle replay.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd

COLUMN_ALIASES = {
    "Order number": "order_id", "User ID": "user_id",
    "Charging Post ID": "station_id",
    "Charging station name": "station_name", "Station Name": "station_name",
    "District": "district", "District Name": "district",
    "Start charging time": "start_time", "Start Time": "start_time",
    "End charging time": "end_time", "End Time": "end_time",
    "Transaction power": "energy_kwh", "Transaction Power": "energy_kwh",
    "Transaction power/kwh": "energy_kwh", "Charging capacity": "energy_kwh",
    "is_full_charge": "is_full_charge", "Is_full_charge": "is_full_charge",
    "is_user_stop": "is_user_stop",
    "is_nonsystem_fault": "is_nonsystem_fault",
    "is_system_issue": "is_system_issue",
    "is_EV_fault": "is_ev_fault", "is_ev_fault": "is_ev_fault",
    "is_other_fault": "is_other_fault",
    "end_cause": "end_cause", "End_cause": "end_cause",
    "End cause": "end_cause",
}

FLAG_COLUMNS = [
    "is_full_charge", "is_user_stop", "is_nonsystem_fault",
    "is_system_issue", "is_ev_fault", "is_other_fault",
]

@dataclass(frozen=True)
class EstimatorConfig:
    k_station: float = 30.0
    k_local: float = 10.0
    k_user: float = 5.0
    use_user_history: bool = True


def _snake(x: str) -> str:
    return re.sub(r"\W+", "_", str(x).strip().lower()).strip("_")


def read_charging_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    errors = []
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            break
        except Exception as exc:
            errors.append(f"{enc}: {exc}")
    else:
        raise RuntimeError("Could not read charging CSV: " + " | ".join(errors))

    renamed = {}
    for c in df.columns:
        key = str(c).strip()
        renamed[c] = COLUMN_ALIASES.get(key, _snake(key))
    return df.rename(columns=renamed)

def normalize_charging(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    renamed = {}
    for c in out.columns:
        key = str(c).strip()
        renamed[c] = COLUMN_ALIASES.get(key, _snake(key))
    out = out.rename(columns=renamed)
    required = ["start_time", "end_time", "energy_kwh"]
    missing = [c for c in required if c not in out]
    if missing:
        raise ValueError(f"Missing required Jiaxing columns: {missing}; have {list(out.columns)}")

    if "station_name" not in out:
        if "station_id" in out:
            out["station_name"] = out["station_id"].astype(str)
        else:
            raise ValueError("Need station_name or station_id")
    if "user_id" not in out:
        out["user_id"] = "__missing_user__"
    if "district" not in out:
        out["district"] = "unknown"
    if "end_cause" not in out:
        out["end_cause"] = ""

    out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce")
    out["end_time"] = pd.to_datetime(out["end_time"], errors="coerce")
    out["energy_kwh"] = pd.to_numeric(out["energy_kwh"], errors="coerce")
    out = out.dropna(subset=["start_time", "end_time", "energy_kwh", "station_name"]).copy()
    out["energy_kwh"] = out["energy_kwh"].clip(lower=0)

    for c in FLAG_COLUMNS:
        if c not in out:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return out

def _cause_flags(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    cause = x["end_cause"].astype(str).str.lower()
    # Only fill from text when explicit indicator columns are all zero.
    no_flag = x[FLAG_COLUMNS].sum(axis=1).eq(0)
    text_system = cause.str.contains("system|系统", regex=True)
    text_non_system = cause.str.contains("charger|pile|桩|设备", regex=True)
    text_user = cause.str.contains("user|用户|主动", regex=True)
    text_ev = cause.str.contains("vehicle|ev fault|车辆", regex=True)
    text_full = cause.str.contains("full|充满|正常结束", regex=True)
    text_other = cause.str.contains("other|其他", regex=True)

    x.loc[no_flag & text_full, "is_full_charge"] = 1
    x.loc[no_flag & text_user, "is_user_stop"] = 1
    x.loc[no_flag & text_ev, "is_ev_fault"] = 1
    x.loc[no_flag & text_system, "is_system_issue"] = 1
    x.loc[no_flag & text_non_system & ~text_system, "is_nonsystem_fault"] = 1
    x.loc[no_flag & text_other, "is_other_fault"] = 1

    excluded = x["is_full_charge"].eq(1) | x["is_user_stop"].eq(1) | x["is_ev_fault"].eq(1)
    eligible = (
        x["is_nonsystem_fault"].eq(1)
        | x["is_system_issue"].eq(1)
        | x["is_other_fault"].eq(1)
    ) & ~excluded
    x["eligible_binary"] = eligible.astype(int)
    return x

def _build_stats(train: pd.DataFrame) -> dict:
    y = train["energy_kwh"].astype(float)
    global_med = float(y.median())
    station = train.groupby("station_name")["energy_kwh"].agg(["median", "count"])
    local = train.groupby(["station_name", "start_halfhour"])["energy_kwh"].agg(["median", "count"])
    user = train.groupby("user_id")["energy_kwh"].agg(["median", "count"])
    return {"global": global_med, "station": station, "local": local, "user": user}


def predict_reference(df: pd.DataFrame, stats: dict, cfg: EstimatorConfig) -> np.ndarray:
    preds = np.zeros(len(df), dtype=float)
    global_med = float(stats["global"])
    station, local, user = stats["station"], stats["local"], stats["user"]

    for k, r in enumerate(df.itertuples(index=False)):
        sid, hh, uid = str(r.station_name), int(r.start_halfhour), str(r.user_id)
        if sid in station.index:
            sm, sn = float(station.loc[sid, "median"]), float(station.loc[sid, "count"])
        else:
            sm, sn = global_med, 0.0
        station_sh = (sn * sm + cfg.k_station * global_med) / (sn + cfg.k_station)

        key = (sid, hh)
        if key in local.index:
            lm, ln = float(local.loc[key, "median"]), float(local.loc[key, "count"])
            base = (ln * lm + cfg.k_local * station_sh) / (ln + cfg.k_local)
        else:
            base = station_sh

        if cfg.use_user_history and uid in user.index and uid != "__missing_user__":
            um, un = float(user.loc[uid, "median"]), float(user.loc[uid, "count"])
            base = (un * um + cfg.k_user * base) / (un + cfg.k_user)
        preds[k] = max(base, 0.0)
    return preds


def _metric(y: np.ndarray, p: np.ndarray) -> dict:
    e = np.asarray(p) - np.asarray(y)
    ae = np.abs(e)
    return {
        "MAE_kWh": float(np.mean(ae)),
        "MdAE_kWh": float(np.median(ae)),
        "RMSE_kWh": float(np.sqrt(np.mean(e ** 2))),
        "Within5_pct": float(np.mean(ae <= 5) * 100),
        "Within10_pct": float(np.mean(ae <= 10) * 100),
        "Bias_kWh": float(np.mean(e)),
    }


def tune_estimator(train_normal: pd.DataFrame, val_normal: pd.DataFrame) -> tuple[EstimatorConfig, pd.DataFrame]:
    stats = _build_stats(train_normal)
    rows = []
    configs = [EstimatorConfig(use_user_history=False)]
    for kl in (5.0, 10.0, 20.0, 40.0):
        for ku in (2.0, 5.0, 10.0, 20.0):
            configs.append(EstimatorConfig(k_local=kl, k_user=ku, use_user_history=True))

    y = val_normal["energy_kwh"].to_numpy(float)
    for cfg in configs:
        p = predict_reference(val_normal, stats, cfg)
        rows.append({**cfg.__dict__, **_metric(y, p)})
    table = pd.DataFrame(rows).sort_values(["MAE_kWh", "RMSE_kWh"]).reset_index(drop=True)
    best = EstimatorConfig(
        k_station=float(table.iloc[0].k_station),
        k_local=float(table.iloc[0].k_local),
        k_user=float(table.iloc[0].k_user),
        use_user_history=bool(table.iloc[0].use_user_history),
    )
    return best, table

def apply_rss_construction(df: pd.DataFrame, stats: dict, cfg: EstimatorConfig) -> pd.DataFrame:
    x = _cause_flags(df)
    x["reference_kwh"] = predict_reference(x, stats, cfg)
    x["raw_shortfall_kwh"] = (x["reference_kwh"] - x["energy_kwh"]).clip(lower=0)

    x["rss_binary_kwh"] = x["eligible_binary"] * x["raw_shortfall_kwh"]
    old_weight = (
        0.9 * x["is_nonsystem_fault"].eq(1).astype(float)
        + 0.9 * x["is_system_issue"].eq(1).astype(float)
        + 0.7 * x["is_other_fault"].eq(1).astype(float)
    ).clip(upper=1.0)
    conservative = (
        0.7 * x["is_nonsystem_fault"].eq(1).astype(float)
        + 0.7 * x["is_system_issue"].eq(1).astype(float)
        + 0.5 * x["is_other_fault"].eq(1).astype(float)
    ).clip(upper=1.0)
    x["rss_legacy_weight_kwh"] = old_weight * x["raw_shortfall_kwh"]
    x["rss_conservative_kwh"] = conservative * x["raw_shortfall_kwh"]
    x["task_time"] = x["end_time"]
    x["task_date"] = x["task_time"].dt.normalize()
    x["task_slot"] = (x["task_time"].dt.hour * 2 + (x["task_time"].dt.minute >= 30).astype(int)).astype(int)
    return x

def _cube(df: pd.DataFrame, dates: pd.DatetimeIndex, stations: list[str], value: str) -> np.ndarray:
    idx_date = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    idx_station = {s: i for i, s in enumerate(stations)}
    cube = np.zeros((len(dates), 48, len(stations)), dtype=float)
    for r in df[["task_date", "task_slot", "station_name", value]].itertuples(index=False):
        di = idx_date.get(pd.Timestamp(r.task_date))
        si = idx_station.get(str(r.station_name))
        if di is None or si is None or not np.isfinite(r[3]):
            continue
        cube[di, int(r.task_slot), si] += float(r[3])
    return cube


def _demand_cube(df: pd.DataFrame, dates: pd.DatetimeIndex, stations: list[str]) -> np.ndarray:
    x = df.copy()
    x["task_time"] = x["end_time"]
    x["task_date"] = x["task_time"].dt.normalize()
    x["task_slot"] = (x["task_time"].dt.hour * 2 + (x["task_time"].dt.minute >= 30).astype(int)).astype(int)
    return _cube(x, dates, stations, "energy_kwh")


def _fault_cube(df: pd.DataFrame, dates: pd.DatetimeIndex, stations: list[str]) -> np.ndarray:
    x = _cause_flags(df)
    x["fault_unit"] = x["eligible_binary"].astype(float)
    x["task_time"] = x["end_time"]
    x["task_date"] = x["task_time"].dt.normalize()
    x["task_slot"] = (x["task_time"].dt.hour * 2 + (x["task_time"].dt.minute >= 30).astype(int)).astype(int)
    return _cube(x, dates, stations, "fault_unit")

def build_replay_data(df: pd.DataFrame, variant: str = "binary") -> dict:
    x = normalize_charging(df)
    x["start_halfhour"] = (x.start_time.dt.hour * 2 + (x.start_time.dt.minute >= 30).astype(int)).astype(int)

    train_mask = x.start_time < pd.Timestamp("2021-07-01")
    val_mask = (x.start_time >= pd.Timestamp("2021-07-01")) & (x.start_time < pd.Timestamp("2021-08-01"))
    test_mask = (x.start_time >= pd.Timestamp("2021-08-01")) & (x.start_time < pd.Timestamp("2022-01-01"))

    normal = _cause_flags(x)["is_full_charge"].eq(1)
    train_normal = x[train_mask & normal & x.energy_kwh.gt(0)].copy()
    val_normal = x[val_mask & normal & x.energy_kwh.gt(0)].copy()
    if len(train_normal) == 0 or len(val_normal) == 0:
        raise ValueError("Need pre-July training and July full-charge validation sessions")

    best, tuning = tune_estimator(train_normal, val_normal)
    stats = _build_stats(train_normal)
    constructed = apply_rss_construction(x, stats, best)

    rss_col = {
        "binary": "rss_binary_kwh",
        "legacy": "rss_legacy_weight_kwh",
        "conservative": "rss_conservative_kwh",
    }.get(variant)
    if rss_col is None:
        raise ValueError("variant must be binary, legacy, or conservative")

    stations = sorted(constructed.station_name.astype(str).unique())
    train_dates = pd.date_range(constructed.loc[train_mask, "task_date"].min(), "2021-06-30", freq="D")
    val_dates = pd.date_range("2021-07-01", "2021-07-31", freq="D")
    test_dates = pd.date_range("2021-08-01", "2021-12-31", freq="D")

    train_df = constructed[constructed.task_date.isin(train_dates)]
    val_df = constructed[constructed.task_date.isin(val_dates)]
    test_df = constructed[constructed.task_date.isin(test_dates)]

    train_rss = _cube(train_df, train_dates, stations, rss_col)
    val_rss = _cube(val_df, val_dates, stations, rss_col)
    test_rss = _cube(test_df, test_dates, stations, rss_col)
    train_demand = _demand_cube(train_df, train_dates, stations)
    test_demand = _demand_cube(test_df, test_dates, stations)
    train_fault = _fault_cube(train_df, train_dates, stations)
    test_fault = _fault_cube(test_df, test_dates, stations)

    station_district = []
    station_weights = []
    for s in stations:
        sx = constructed[constructed.station_name.astype(str).eq(s)]
        district = sx.district.astype(str).mode()
        station_district.append(district.iloc[0] if len(district) else "unknown")
        station_weights.append(float(train_df.station_name.astype(str).eq(s).sum()) + 1.0)
    station_weights = np.asarray(station_weights, float)
    station_weights /= station_weights.sum()

    demand_sum = float(train_demand.sum())
    fault_sum = float(train_fault.sum())
    rss_sum = float(train_rss.sum())

    return {
        "stations": stations,
        "station_district": np.asarray(station_district, dtype=object),
        "station_weights": station_weights,
        "train_dates": list(train_dates),
        "val_dates": list(val_dates),
        "test_dates": list(test_dates),
        "train_rss": train_rss,
        "val_rss": val_rss,
        "test_rss": test_rss,
        "train_demand": train_demand,
        "test_demand": test_demand,
        "train_fault": train_fault,
        "test_fault": test_fault,
        "global_rss_per_demand_energy": rss_sum / demand_sum if demand_sum else 0.0,
        "global_rss_per_fault_loss": rss_sum / fault_sum if fault_sum else 0.0,
        "best_estimator": best.__dict__,
        "tuning_table": tuning,
        "constructed_transactions": constructed,
        "variant": variant,
    }


def build_topology(station_district: np.ndarray, station_weights: np.ndarray, seed: int):
    """Deterministic stylized Jiaxing topology retained only for transaction replay."""
    rng = np.random.default_rng(seed)
    districts = list(dict.fromkeys(map(str, station_district)))
    # Broad district centers; absolute coordinates are deliberately stylized.
    centers = {}
    angles = np.linspace(0, 2 * np.pi, max(len(districts), 3), endpoint=False)
    for k, district in enumerate(districts):
        centers[district] = np.array([10.0 * np.cos(angles[k]), 10.0 * np.sin(angles[k])])
    coords = np.zeros((len(station_district), 2), float)
    for i, district in enumerate(map(str, station_district)):
        spread = 2.0 + 2.0 * (1.0 - float(station_weights[i]) / max(float(station_weights.max()), 1e-9))
        coords[i] = centers[district] + rng.normal(0.0, spread, size=2)
    delta = coords[:, None, :] - coords[None, :, :]
    distance = np.sqrt((delta ** 2).sum(axis=2))
    return coords[:, 0], coords[:, 1], distance

def save_replay(replay: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"jiaxing_replay_{replay['variant']}.npz",
        train_rss=replay["train_rss"], val_rss=replay["val_rss"], test_rss=replay["test_rss"],
        train_demand=replay["train_demand"], test_demand=replay["test_demand"],
        train_fault=replay["train_fault"], test_fault=replay["test_fault"],
        stations=np.asarray(replay["stations"], dtype=str),
        station_district=np.asarray(replay["station_district"], dtype=str),
        station_weights=np.asarray(replay["station_weights"], dtype=float),
        train_dates=np.asarray([str(pd.Timestamp(x).date()) for x in replay["train_dates"]]),
        test_dates=np.asarray([str(pd.Timestamp(x).date()) for x in replay["test_dates"]]),
    )
    replay["tuning_table"].to_csv(out / "jiaxing_target_estimator_tuning.csv", index=False)
    tx = replay["constructed_transactions"]
    keep = [
        c for c in ["order_id","user_id","station_name","district","start_time","end_time",
                    "energy_kwh","reference_kwh","raw_shortfall_kwh","eligible_binary",
                    "rss_binary_kwh","rss_legacy_weight_kwh","rss_conservative_kwh",
                    "is_full_charge","is_user_stop","is_nonsystem_fault","is_system_issue",
                    "is_ev_fault","is_other_fault","task_date","task_slot"] if c in tx.columns
    ]
    tx[keep].to_parquet(out / "jiaxing_revision_transactions.parquet", index=False)

    summary = {
        "variant": replay["variant"],
        "stations": len(replay["stations"]),
        "train_days": len(replay["train_dates"]),
        "test_days": len(replay["test_dates"]),
        "best_estimator": replay["best_estimator"],
        "train_total_rss_kWh": float(replay["train_rss"].sum()),
        "test_total_rss_kWh": float(replay["test_rss"].sum()),
        "positive_test_station_slots": int((replay["test_rss"] > 0).sum()),
        "global_rss_per_demand_energy": float(replay["global_rss_per_demand_energy"]),
        "global_rss_per_fault_loss": float(replay["global_rss_per_fault_loss"]),
    }
    (out / "jiaxing_replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--charging", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="binary", choices=["binary","legacy","conservative"])
    args = ap.parse_args()
    df = read_charging_csv(args.charging)
    replay = build_replay_data(df, args.variant)
    save_replay(replay, args.out)
    print(json.dumps({
        "rows": len(df), "variant": args.variant,
        "best_estimator": replay["best_estimator"],
        "test_total_rss_kWh": float(replay["test_rss"].sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
