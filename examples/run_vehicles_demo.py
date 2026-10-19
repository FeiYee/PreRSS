"""Vehicles companion demo — EDVF-only figures (no baseline suite).

1) Live Exact/Sparse smoke on synthetic + one Jiaxing day
2) Plots from Jiaxing / UrbanEV EDVF result summaries
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edvf_rss import DispatchConfig, simulate_independent_day  # noqa: E402
from edvf_rss.paths import jiaxing_event_script, jiaxing_parquet  # noqa: E402

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "outputs" / "demo"
OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results"


def load_event():
    path = jiaxing_event_script()
    spec = importlib.util.spec_from_file_location("jiaxing_event_vehicles", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def fig_viability(path: Path) -> None:
    d = np.arange(1, 5)
    demand = np.array([18.0, 32.0, 40.0, 44.0])
    good = np.array([20.0, 36.0, 48.0, 56.0])
    bad = np.array([10.0, 18.0, 24.0, 28.0])
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=160)
    ax.plot(d, demand, "o-", color="#1f4e79", lw=2.2, label=r"$D(d)$ deadline demand")
    ax.plot(d, good, "s--", color="#2e7d32", lw=2.0, label=r"$S(d;f^\star)$ under EDVF")
    ax.plot(d, bad, "^:", color="#c62828", lw=2.0, label=r"$S(d;f)$ poor flow")
    ax.fill_between(d, bad, demand, where=demand >= bad, color="#c62828", alpha=0.15)
    ax.set_xlabel("Deadline prefix $d$")
    ax.set_ylabel("Cumulative energy (kWh)")
    ax.set_xticks(d)
    ax.set_title("Deadline-viability envelope (Vehicles / EDVF)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_urbanev_energy(path: Path) -> None:
    p = RES / "urbanev" / "finite_energy" / "summary.csv"
    if not p.exists():
        p = RES / "urbanev" / "finite_energy_summary.csv"
    if not p.exists():
        print("[warn] no finite-energy summary")
        return
    df = pd.read_csv(p)
    labels = df["config"].astype(str).tolist() if "config" in df.columns else df.index.astype(str).tolist()
    cov = df["coverage_pct"].to_numpy(float)
    res = df["residual_mean_kWh"].to_numpy(float)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=160)
    axes[0].bar(x, cov, color="#215a9a", width=0.6)
    axes[0].set_ylabel("Coverage (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].set_title("UrbanEV finite onboard energy — EDVF")
    axes[1].bar(x, res, color="#c62828", width=0.6)
    axes[1].set_ylabel("Unrecovered RSS (kWh)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_travel_speed(path: Path) -> None:
    p = RES / "urbanev" / "travel_speed" / "summary_by_topology.csv"
    if not p.exists():
        print("[warn] no travel-speed summary")
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
    for topo, g in df.groupby("topology"):
        g = g.sort_values("speed_kmh")
        ax.plot(g["speed_kmh"], g["coverage_pct"], "o-", label=str(topo), lw=1.6)
    ax.legend(frameon=False, fontsize=8, title="Topology")
    ax.set_xlabel("Travel speed (km/h)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("UrbanEV travel-speed stress — EDVF")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def live_demo() -> dict:
    cfg = DispatchConfig(fleet=3, horizon=3, scenarios=1)
    distance = np.array(
        [
            [0.0, 8.0, 12.0, 18.0],
            [8.0, 0.0, 10.0, 15.0],
            [12.0, 10.0, 0.0, 9.0],
            [18.0, 15.0, 9.0, 0.0],
        ]
    )
    actual = np.zeros((48, 4))
    for t, j, e in [(5, 1, 25.0), (12, 2, 30.0), (20, 3, 18.0), (28, 1, 22.0)]:
        actual[t, j] = e
    history = np.stack([actual, actual], 0)
    rows = []
    for policy in ("EDVF-RSS-Exact", "EDVF-RSS-Sparse"):
        r = simulate_independent_day(
            actual, actual, history, np.array([0, 1]), distance, cfg, policy, include_future=False
        )
        rows.append(
            {
                "method": policy,
                "coverage": float(r.coverage),
                "unrecovered_kwh": float(r.residual_rss_kwh),
                "movement_km": float(r.movement_km),
                "runtime_s": float(r.runtime_seconds),
            }
        )
    # Jiaxing one day
    jx = None
    try:
        event = load_event()
        replay = event.build_replay_data(pd.read_parquet(jiaxing_parquet()), "base")
        dates = list(replay["test_dates"])
        selected = [
            i
            for i, d in enumerate(dates)
            if pd.Timestamp("2021-08-01") <= pd.Timestamp(d) <= pd.Timestamp("2021-12-31")
        ]
        day_idx = selected[len(selected) // 2]
        _s, _c, dist = event.build_topology(
            replay["station_district"], replay["station_weights"], 20260724 + 700_000
        )
        actual_day = np.asarray(replay["test_rss"][day_idx], float)
        jcfg = DispatchConfig(fleet=4, horizon=3, scenarios=1)
        r = simulate_independent_day(
            actual_day,
            actual_day,
            np.asarray(replay["train_rss"], float)[:2],
            np.array([0, 1]),
            dist,
            jcfg,
            "EDVF-RSS-Exact",
            include_future=False,
        )
        jx = {
            "date": str(pd.Timestamp(dates[day_idx]).date()),
            "coverage": float(r.coverage),
            "unrecovered_kwh": float(r.residual_rss_kwh),
            "movement_km": float(r.movement_km),
            "runtime_s": float(r.runtime_seconds),
        }
    except Exception as exc:
        print(f"[warn] Jiaxing live day skipped: {type(exc).__name__}: {exc}")
    return {"synthetic": rows, "jiaxing_day": jx}


def main() -> None:
    print("=== Vehicles EDVF open-source demo ===")
    fig_viability(FIG / "fig_viability_envelope.png")
    fig_urbanev_energy(FIG / "fig_urbanev_finite_energy.png")
    fig_travel_speed(FIG / "fig_urbanev_travel_speed.png")
    live = live_demo()
    print("Live synthetic:", live["synthetic"])
    print("Live Jiaxing:", live["jiaxing_day"])
    summary = {
        "live_demo": live,
        "figures": sorted(p.name for p in FIG.glob("*.png")),
    }
    (OUT / "demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Figures ->", FIG)
    print("Summary ->", OUT / "demo_summary.json")


if __name__ == "__main__":
    main()
