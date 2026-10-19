"""Replay event-triggered mobile rescue on real Jiaxing charging interruptions.

The transaction stream and protocol-derived causes are real.  The depot,
station coordinates, mobile fleet, travel, service capacity, and recoverability
are explicit simulation assumptions because the public dataset anonymizes exact
station positions and contains no mobile-charger deployment logs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CORE_PATH = ROOT / "run_charging_core_benchmark.py"
if not CORE_PATH.exists():
    CORE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_charging_core_benchmark.py"
CORE_SPEC = importlib.util.spec_from_file_location("charging_core_event", CORE_PATH)
assert CORE_SPEC is not None and CORE_SPEC.loader is not None
core = importlib.util.module_from_spec(CORE_SPEC)
sys.modules[CORE_SPEC.name] = core
CORE_SPEC.loader.exec_module(core)


DEFAULT_DATA = (
    ROOT
    / "outputs"
    / "charging_paper_jiaxing_audit"
    / "data"
    / "processed"
    / "jiaxing_transactions_clean.parquet"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "charging_paper_jiaxing_event_dispatch"
POLICIES = (
    "fixed_only",
    "demand_only",
    "aggregate_event",
    "ecap_rss_event",
    "oracle",
)
POLICY_LABELS = {
    "fixed_only": "Fixed-only",
    "demand_only": "Demand-only",
    "aggregate_event": "Aggregate event",
    "ecap_rss_event": "Cause-gated ECap/RSS",
    "oracle": "Oracle capacity upper bound",
}
RECOVERABILITY_SCENARIOS = {
    "low": {
        "nonsystem_pile_fault": 0.6,
        "system_issue": 0.6,
        "other_pile_fault": 0.4,
    },
    "base": {
        "nonsystem_pile_fault": 0.9,
        "system_issue": 0.9,
        "other_pile_fault": 0.7,
    },
    "high": {
        "nonsystem_pile_fault": 1.0,
        "system_issue": 1.0,
        "other_pile_fault": 0.9,
    },
    "binary": {
        "nonsystem_pile_fault": 1.0,
        "system_issue": 1.0,
        "other_pile_fault": 1.0,
    },
}
DISTRICT_CENTERS_KM = {
    "Nanhu": np.array([7.0, 5.0]),
    "Xiuzhou": np.array([0.0, 8.0]),
    "Tongxiang": np.array([-24.0, -8.0]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fleet-units", type=int, default=4)
    parser.add_argument("--unit-capacity-kwh", type=float, default=20.0)
    parser.add_argument("--response-slots", type=int, default=4)
    parser.add_argument(
        "--relocation-cost-per-km",
        type=float,
        default=0.003,
    )
    parser.add_argument("--rolling-horizon", type=int, default=2)
    parser.add_argument("--forecast-scenarios", type=int, default=3)
    parser.add_argument("--cvar-weight", type=float, default=0.35)
    parser.add_argument(
        "--ignore-travel-capacity-loss",
        action="store_true",
    )
    parser.add_argument(
        "--ignore-backlog-in-forecast",
        action="store_true",
        help=(
            "Keep the physical response window unchanged but remove prior-slot "
            "backlog from the rolling optimization signal."
        ),
    )
    parser.add_argument("--topology-seeds", type=int, default=5)
    parser.add_argument(
        "--topology-seed-offset",
        type=int,
        default=0,
        help="Offset into the frozen deterministic topology-seed sequence.",
    )
    parser.add_argument(
        "--recoverability",
        choices=tuple(RECOVERABILITY_SCENARIOS),
        default="base",
    )
    parser.add_argument("--max-days", type=int)
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
    )
    parser.add_argument("--variant-label", type=str, default="primary")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Seven days and one topology seed for implementation checks.",
    )
    return parser.parse_args()


def euclidean_distance(coordinates: np.ndarray) -> np.ndarray:
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt((delta**2).sum(axis=2))


def build_topology(
    station_district: pd.Series,
    station_weights: pd.Series,
    seed: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    stations = station_district.index.tolist()
    rng = np.random.default_rng(seed)
    station_coordinates: list[np.ndarray] = []
    for station in stations:
        district = station_district.loc[station]
        if district not in DISTRICT_CENTERS_KM:
            raise ValueError(f"Unknown district for topology: {district}")
        center = DISTRICT_CENTERS_KM[district]
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(1.5, 5.0)
        offset = radius * np.array([np.cos(angle), np.sin(angle)])
        station_coordinates.append(center + offset)
    station_array = np.vstack(station_coordinates)
    weights = station_weights.reindex(stations).to_numpy(dtype=float)
    depot = np.average(station_array, axis=0, weights=weights)
    coordinates = np.vstack([depot, station_array])
    return ["DEPOT_A"] + stations, coordinates, euclidean_distance(coordinates)


def make_daily_cube(
    frame: pd.DataFrame,
    dates: list[pd.Timestamp],
    stations: list[str],
    value: pd.Series,
) -> np.ndarray:
    working = frame[["date", "half_hour", "station"]].copy()
    working["value"] = value.to_numpy(dtype=float)
    grouped = working.groupby(
        ["date", "half_hour", "station"],
        observed=True,
    )["value"].sum()
    index = pd.MultiIndex.from_product(
        [dates, range(48), stations],
        names=["date", "half_hour", "station"],
    )
    station_cube = (
        grouped.reindex(index, fill_value=0.0)
        .to_numpy(dtype=float)
        .reshape(len(dates), 48, len(stations))
    )
    return np.concatenate(
        [np.zeros((len(dates), 48, 1)), station_cube],
        axis=2,
    )


def build_replay_data(
    frame: pd.DataFrame,
    recoverability_name: str,
) -> dict[str, object]:
    weights = RECOVERABILITY_SCENARIOS[recoverability_name]
    working = frame.copy()
    working["recoverability_replay"] = (
        working["cause"].map(weights).fillna(0.0).astype(float)
    )
    working["rss_replay_kwh"] = (
        working["recoverability_replay"]
        * working["counterfactual_loss_kwh"]
    )
    working["fault_loss_kwh"] = np.where(
        working["protocol_fault"],
        working["counterfactual_loss_kwh"],
        0.0,
    )
    working["demand_energy_kwh"] = working[
        "target_completed_energy_kwh"
    ].clip(lower=0.0)
    train = working.loc[working["start_time"] < "2021-07-01"].copy()
    test = working.loc[working["start_time"] >= "2021-07-01"].copy()
    stations = sorted(working["station"].unique().tolist())
    station_district_counts = (
        working.groupby(["station", "district"]).size().unstack(fill_value=0)
    )
    station_district = station_district_counts.idxmax(axis=1).reindex(stations)
    if (station_district_counts.gt(0).sum(axis=1) != 1).any():
        raise ValueError("A station appears in more than one district.")
    station_weights = (
        train.groupby("station")["demand_energy_kwh"].sum().reindex(stations)
    )
    train_dates = sorted(train["date"].drop_duplicates().tolist())
    test_dates = sorted(test["date"].drop_duplicates().tolist())
    result: dict[str, object] = {
        "stations": stations,
        "station_district": station_district,
        "station_weights": station_weights,
        "train_dates": train_dates,
        "test_dates": test_dates,
    }
    for split_name, split, dates in [
        ("train", train, train_dates),
        ("test", test, test_dates),
    ]:
        result[f"{split_name}_rss"] = make_daily_cube(
            split,
            dates,
            stations,
            split["rss_replay_kwh"],
        )
        result[f"{split_name}_fault"] = make_daily_cube(
            split,
            dates,
            stations,
            split["fault_loss_kwh"],
        )
        result[f"{split_name}_demand"] = make_daily_cube(
            split,
            dates,
            stations,
            split["demand_energy_kwh"],
        )
    train_fault = np.asarray(result["train_fault"], dtype=float)
    train_rss = np.asarray(result["train_rss"], dtype=float)
    train_demand = np.asarray(result["train_demand"], dtype=float)
    result["global_rss_per_fault_loss"] = float(
        train_rss.sum() / max(train_fault.sum(), 1e-9)
    )
    result["global_rss_per_demand_energy"] = float(
        train_rss.sum() / max(train_demand.sum(), 1e-9)
    )
    return result


def shift_and_add(
    age: np.ndarray,
    new_work: np.ndarray,
) -> float:
    expired = float(age[-1].sum())
    age[1:] = age[:-1]
    age[0] = new_work
    return expired


def serve_cause_known(
    actual_age: np.ndarray,
    available: np.ndarray,
) -> tuple[float, float]:
    useful = 0.0
    attempted = 0.0
    for node in range(actual_age.shape[1]):
        capacity = float(available[node])
        for age_index in range(actual_age.shape[0] - 1, -1, -1):
            if capacity <= 1e-12:
                break
            served = min(capacity, float(actual_age[age_index, node]))
            actual_age[age_index, node] -= served
            capacity -= served
            useful += served
            attempted += served
    return useful, attempted


def serve_cause_blind(
    actual_age: np.ndarray,
    fault_age: np.ndarray,
    available: np.ndarray,
) -> tuple[float, float]:
    useful = 0.0
    attempted_total = 0.0
    for node in range(actual_age.shape[1]):
        capacity = float(available[node])
        for age_index in range(actual_age.shape[0] - 1, -1, -1):
            if capacity <= 1e-12:
                break
            fault = float(fault_age[age_index, node])
            if fault <= 1e-12:
                continue
            attempted = min(capacity, fault)
            recoverable_fraction = min(
                1.0,
                float(actual_age[age_index, node]) / fault,
            )
            useful_service = attempted * recoverable_fraction
            fault_age[age_index, node] -= attempted
            actual_age[age_index, node] = max(
                0.0,
                actual_age[age_index, node] - useful_service,
            )
            capacity -= attempted
            attempted_total += attempted
            useful += useful_service
    return useful, attempted_total


def serve_oracle_upper_bound(
    actual_age: np.ndarray,
    total_capacity: float,
) -> tuple[float, float]:
    """Serve the oldest RSS system-wide, ignoring travel and positioning.

    This deliberately relaxes the physical routing constraints.  It is an
    unattainable capacity upper bound, not another dispatch policy, and
    therefore cannot be outperformed by a feasible routed policy.
    """
    remaining = float(total_capacity)
    useful = 0.0
    for age_index in range(actual_age.shape[0] - 1, -1, -1):
        if remaining <= 1e-12:
            break
        age_total = float(actual_age[age_index].sum())
        if age_total <= 1e-12:
            continue
        served = min(remaining, age_total)
        fraction = served / age_total
        actual_age[age_index] *= 1.0 - fraction
        remaining -= served
        useful += served
    return useful, useful


def forecast_indices_for_day(
    train_dates: list[pd.Timestamp],
    test_date: pd.Timestamp,
    scenarios: int,
    rng: np.random.Generator,
) -> np.ndarray:
    target_weekend = pd.Timestamp(test_date).dayofweek >= 5
    candidates = np.array(
        [
            index
            for index, date in enumerate(train_dates)
            if (pd.Timestamp(date).dayofweek >= 5) == target_weekend
        ],
        dtype=int,
    )
    if len(candidates) == 0:
        candidates = np.arange(len(train_dates), dtype=int)
    return rng.choice(candidates, size=scenarios, replace=True)


def build_gap_scenarios(
    policy: str,
    slot: int,
    actual_age: np.ndarray,
    fault_age: np.ndarray,
    test_rss_day: np.ndarray,
    train_rss: np.ndarray,
    train_fault: np.ndarray,
    train_demand: np.ndarray,
    forecast_indices: np.ndarray,
    rss_per_fault: float,
    rss_per_demand: float,
    horizon: int,
    include_backlog: bool = True,
) -> np.ndarray:
    scenarios = len(forecast_indices)
    n_nodes = actual_age.shape[1]
    gap = np.zeros((scenarios, horizon, n_nodes), dtype=float)
    current_actual = (
        actual_age.sum(axis=0) if include_backlog else actual_age[0].copy()
    )
    current_fault = (
        fault_age.sum(axis=0) if include_backlog else fault_age[0].copy()
    )
    for scenario, train_index in enumerate(forecast_indices):
        for h in range(horizon):
            source_slot = min(slot + h, 47)
            if policy == "demand_only":
                gap[scenario, h] = (
                    train_demand[train_index, source_slot] * rss_per_demand
                )
            elif policy == "aggregate_event":
                future = train_fault[train_index, source_slot] * rss_per_fault
                gap[scenario, h] = current_fault * rss_per_fault + (
                    future if h > 0 else 0.0
                )
            elif policy == "ecap_rss_event":
                future = train_rss[train_index, source_slot]
                gap[scenario, h] = current_actual + (
                    future if h > 0 else 0.0
                )
            else:
                raise ValueError(f"Unsupported policy: {policy}")
    return gap


def simulate_day(
    policy: str,
    date: pd.Timestamp,
    test_rss_day: np.ndarray,
    test_fault_day: np.ndarray,
    train_rss: np.ndarray,
    train_fault: np.ndarray,
    train_demand: np.ndarray,
    forecast_indices: np.ndarray,
    distance_km: np.ndarray,
    fleet_units: int,
    unit_capacity_kwh: float,
    response_slots: int,
    rss_per_fault: float,
    rss_per_demand: float,
    config: core.BenchmarkConfig,
    include_backlog: bool = True,
) -> dict[str, object]:
    n_nodes = test_rss_day.shape[1]
    actual_age = np.zeros((response_slots, n_nodes), dtype=float)
    fault_age = np.zeros_like(actual_age)
    current = np.zeros(n_nodes, dtype=int)
    current[0] = fleet_units
    total_rss = 0.0
    residual_rss = 0.0
    useful_service = 0.0
    attempted_service = 0.0
    movement_km = 0.0
    solver_runtime = 0.0
    solver_failures = 0

    for slot in range(48):
        residual_rss += shift_and_add(actual_age, test_rss_day[slot])
        shift_and_add(fault_age, test_fault_day[slot])
        total_rss += float(test_rss_day[slot].sum())
        if policy == "fixed_only":
            continue
        if policy == "oracle":
            useful, attempted = serve_oracle_upper_bound(
                actual_age,
                fleet_units * unit_capacity_kwh,
            )
            useful_service += useful
            attempted_service += attempted
            continue

        gap = build_gap_scenarios(
            policy,
            slot,
            actual_age,
            fault_age,
            test_rss_day,
            train_rss,
            train_fault,
            train_demand,
            forecast_indices,
            rss_per_fault,
            rss_per_demand,
            config.rolling_horizon,
            include_backlog,
        )
        horizon = gap.shape[1]
        solved = core.solve_rolling_milp(
            gap,
            current,
            distance_km,
            np.full(horizon, unit_capacity_kwh),
            np.full(horizon, config.max_relocation_km),
            config,
        )
        solver_runtime += solved.runtime_seconds
        solver_failures += int(not solved.success)
        flow = solved.first_flow
        current = solved.first_allocation
        if config.travel_capacity_loss:
            travel_factor = np.clip(
                1.0 - distance_km / config.max_relocation_km,
                0.0,
                1.0,
            )
        else:
            travel_factor = np.ones_like(distance_km, dtype=float)
        np.fill_diagonal(travel_factor, 1.0)
        available = (flow * travel_factor * unit_capacity_kwh).sum(axis=0)
        movement_km += float(
            (
                flow
                * distance_km
                * (~np.eye(n_nodes, dtype=bool))
            ).sum()
        )
        if policy == "ecap_rss_event":
            useful, attempted = serve_cause_known(actual_age, available)
        else:
            useful, attempted = serve_cause_blind(
                actual_age,
                fault_age,
                available,
            )
        useful_service += useful
        attempted_service += attempted

    residual_rss += float(actual_age.sum())
    if not np.isclose(
        total_rss,
        useful_service + residual_rss,
        atol=1e-5,
        rtol=1e-6,
    ):
        raise AssertionError(
            "Recoverable-energy conservation failed: "
            f"total={total_rss}, served={useful_service}, residual={residual_rss}"
        )
    return {
        "date": pd.Timestamp(date).date().isoformat(),
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "total_rss_kwh": total_rss,
        "recovered_rss_kwh": useful_service,
        "residual_rss_kwh": residual_rss,
        "rss_coverage_rate": (
            useful_service / total_rss if total_rss > 0 else 1.0
        ),
        "attempted_mobile_service_kwh": attempted_service,
        "wasted_mobile_service_kwh": max(
            0.0,
            attempted_service - useful_service,
        ),
        "movement_km": movement_km,
        "recovered_kwh_per_movement_km": (
            useful_service / movement_km if movement_km > 0 else math.nan
        ),
        "solver_runtime_seconds": solver_runtime,
        "solver_failures": solver_failures,
    }


def run_replay(
    replay: dict[str, object],
    config: core.BenchmarkConfig,
    fleet_units: int,
    unit_capacity_kwh: float,
    response_slots: int,
    topology_seeds: int,
    topology_seed_offset: int,
    policies: list[str],
    start_date: str | None,
    end_date: str | None,
    max_days: int | None,
    include_backlog: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_test_dates = list(replay["test_dates"])
    selected_indices = [
        index
        for index, date in enumerate(all_test_dates)
        if (start_date is None or pd.Timestamp(date) >= pd.Timestamp(start_date))
        and (end_date is None or pd.Timestamp(date) <= pd.Timestamp(end_date))
    ]
    if max_days is not None:
        selected_indices = selected_indices[:max_days]
    if not selected_indices:
        raise ValueError("No test dates remain after applying date filters.")
    test_dates = [all_test_dates[index] for index in selected_indices]
    test_rss = np.asarray(replay["test_rss"], dtype=float)[selected_indices]
    test_fault = np.asarray(replay["test_fault"], dtype=float)[selected_indices]
    train_rss = np.asarray(replay["train_rss"], dtype=float)
    train_fault = np.asarray(replay["train_fault"], dtype=float)
    train_demand = np.asarray(replay["train_demand"], dtype=float)
    train_dates = list(replay["train_dates"])
    rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    for topology_index in range(topology_seeds):
        seed = (
            config.random_seed
            + 700_000
            + topology_seed_offset
            + topology_index
        )
        node_ids, coordinates, distance = build_topology(
            replay["station_district"],
            replay["station_weights"],
            seed,
        )
        topology_rows.extend(
            {
                "topology_seed": seed,
                "node_id": node_id,
                "x_km": coordinates[node, 0],
                "y_km": coordinates[node, 1],
            }
            for node, node_id in enumerate(node_ids)
        )
        rng = np.random.default_rng(seed + 50_000)
        for day_index, date in enumerate(test_dates):
            forecast_indices = forecast_indices_for_day(
                train_dates,
                date,
                config.forecast_scenarios,
                rng,
            )
            for policy in policies:
                result = simulate_day(
                    policy,
                    date,
                    test_rss[day_index],
                    test_fault[day_index],
                    train_rss,
                    train_fault,
                    train_demand,
                    forecast_indices,
                    distance,
                    fleet_units,
                    unit_capacity_kwh,
                    response_slots,
                    float(replay["global_rss_per_fault_loss"]),
                    float(replay["global_rss_per_demand_energy"]),
                    config,
                    include_backlog,
                )
                result.update(
                    {
                        "topology_seed": seed,
                        "fleet_units": fleet_units,
                        "unit_capacity_kwh": unit_capacity_kwh,
                        "response_slots": response_slots,
                    }
                )
                rows.append(result)
            print(
                f"topology={topology_index + 1}/{topology_seeds} "
                f"day={day_index + 1}/{len(test_dates)} "
                f"date={pd.Timestamp(date).date()}",
                flush=True,
            )
    return pd.DataFrame(rows), pd.DataFrame(topology_rows)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    raw_dir = output / "raw"
    metadata_dir = output / "metadata"
    for directory in [raw_dir, metadata_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.data.resolve())
    replay = build_replay_data(frame, args.recoverability)
    topology_seeds = args.topology_seeds
    max_days = args.max_days
    if args.quick:
        topology_seeds = 1
        max_days = 7

    config = replace(
        core.BenchmarkConfig(),
        service_nodes=13,
        fixed_nodes=13,
        slots_per_day=48,
        rolling_horizon=args.rolling_horizon,
        forecast_scenarios=args.forecast_scenarios,
        mobile_unit_capacity=args.unit_capacity_kwh,
        max_relocation_fraction=0.75,
        max_relocation_km=30.0,
        cvar_alpha=0.90,
        cvar_weight=args.cvar_weight,
        relocation_cost_per_km=args.relocation_cost_per_km,
        travel_capacity_loss=not args.ignore_travel_capacity_loss,
        solver_time_limit_seconds=5.0,
        solver_mip_gap=0.005,
        random_seed=20260724,
    )
    start = time.perf_counter()
    runs, topology = run_replay(
        replay,
        config,
        args.fleet_units,
        args.unit_capacity_kwh,
        args.response_slots,
        topology_seeds,
        args.topology_seed_offset,
        list(args.policies),
        args.start_date,
        args.end_date,
        max_days,
        not args.ignore_backlog_in_forecast,
    )
    elapsed = time.perf_counter() - start
    runs["recoverability_scenario"] = args.recoverability
    runs["variant_label"] = args.variant_label
    runs.to_csv(
        raw_dir / "jiaxing_event_dispatch_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    topology.to_csv(
        metadata_dir / "simulated_topologies.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "evidence_boundary": (
            "Real Jiaxing transactions and protocol-derived causes; simulated "
            "mobile fleet, topology, travel, response, and recoverability."
        ),
        "config": asdict(config),
        "fleet_units": args.fleet_units,
        "unit_capacity_kwh": args.unit_capacity_kwh,
        "response_slots": args.response_slots,
        "variant_label": args.variant_label,
        "recoverability_scenario": args.recoverability,
        "include_backlog_in_forecast": not args.ignore_backlog_in_forecast,
        "topology_seeds": topology_seeds,
        "topology_seed_offset": args.topology_seed_offset,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "test_days": int(runs["date"].nunique()),
        "elapsed_seconds": elapsed,
        "global_rss_per_fault_loss": replay["global_rss_per_fault_loss"],
        "global_rss_per_demand_energy": replay[
            "global_rss_per_demand_energy"
        ],
        "policies": list(args.policies),
    }
    (metadata_dir / "event_dispatch_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(runs)} policy-day-topology runs to {output} "
        f"in {elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
