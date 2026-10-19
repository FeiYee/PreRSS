"""Run the mechanism-closure benchmark for the charging-paper study.

Evidence boundary
-----------------
Observed LaDe-P demand is used to calibrate the spatiotemporal request process.
Cause-specific interruption and recoverability are scenario-defined because the
workspace does not contain real charging-session cause labels.  The benchmark
therefore validates the optimization mechanism and its stress response; it is
not a substitute for real charging-field validation.

The deployable policies share exactly the same rolling-horizon MILP.  They
differ only in the information used to estimate mobile-recoverable workload:

* demand_only: overload and mobile-only demand;
* aggregate_risk: total interruption risk with one global recovery rate;
* ecap_rss: cause-specific interruption probabilities and recovery rates;
* oracle: realized recoverable workload (non-deployable upper bound).

The fixed_only policy is reported as an infrastructure baseline and does not
use mobile resources.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = (
    ROOT
    / "outputs"
    / "charging_paper_lade_validation"
    / "data"
    / "processed"
    / "lade_p_task_events_30min.parquet"
)
DEFAULT_PROFILES = (
    ROOT
    / "outputs"
    / "charging_paper_lade_validation"
    / "data"
    / "processed"
    / "lade_p_region_profiles.parquet"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "charging_paper_core_benchmark"

POLICIES = (
    "fixed_only",
    "demand_only",
    "aggregate_risk",
    "ecap_rss",
    "oracle",
)
POLICY_LABELS = {
    "fixed_only": "Fixed-only",
    "demand_only": "Demand-only MILP",
    "aggregate_risk": "Aggregate-risk MILP",
    "ecap_rss": "Cause-specific ECap/RSS MILP",
    "oracle": "Oracle recovery MILP",
}
STRESSES = (
    "normal",
    "demand_surge",
    "equipment_fault",
    "unrecoverable_shift",
    "weather",
    "recharge_limited",
)
STRESS_LABELS = {
    "normal": "Normal",
    "demand_surge": "Demand surge",
    "equipment_fault": "Equipment fault",
    "unrecoverable_shift": "Unrecoverable-cause shift",
    "weather": "Adverse weather",
    "recharge_limited": "Recharge limited",
}
CAUSES = ("system_or_site", "user_stop", "vehicle_fault")


@dataclass(frozen=True)
class BenchmarkConfig:
    city: str = "Hangzhou"
    service_nodes: int = 12
    fixed_nodes: int = 11
    slots_per_day: int = 24
    rolling_horizon: int = 3
    forecast_scenarios: int = 6
    mobile_unit_capacity: float = 20.0
    fleet_ratio: float = 0.90
    fixed_capacity_quantile: float = 0.95
    fleet_reference_quantile: float = 0.975
    max_relocation_fraction: float = 0.55
    max_relocation_km: float = 50.0
    cvar_alpha: float = 0.90
    cvar_weight: float = 0.35
    relocation_cost_per_km: float = 0.004
    aggregate_recovery_rate: float = 0.4345
    cause_heterogeneity: float = 1.0
    travel_capacity_loss: bool = True
    solver_time_limit_seconds: float = 15.0
    solver_mip_gap: float = 0.001
    seeds: int = 20
    scalability_repetitions: int = 3
    random_seed: int = 20260724


@dataclass
class CaseData:
    region_ids: list[str]
    node_types: np.ndarray
    coordinates: np.ndarray
    distance_km: np.ndarray
    risk_proxy: np.ndarray
    fixed_capacity: np.ndarray
    initial_allocation: np.ndarray
    fleet_units: int
    mobile_unit_capacity: float
    test_dates: list[pd.Timestamp]
    test_demand: dict[pd.Timestamp, np.ndarray]
    train_demand: np.ndarray


@dataclass
class Realization:
    demand: np.ndarray
    mobile_target: np.ndarray
    reliable_fixed: np.ndarray
    unavoidable: np.ndarray
    recoverable: np.ndarray
    overload_or_mobile_only: np.ndarray
    cause_counts: np.ndarray
    recovered_counts: np.ndarray
    total_interruption_probability: np.ndarray
    cause_mix: np.ndarray
    recovery_probability: np.ndarray
    mobile_capacity: np.ndarray
    max_relocation_km: np.ndarray


@dataclass
class ForecastBundle:
    demand: np.ndarray
    total_interruption_probability: np.ndarray
    cause_mix: np.ndarray
    recovery_probability: np.ndarray
    mobile_capacity: np.ndarray
    max_relocation_km: np.ndarray


@dataclass
class SolveResult:
    first_flow: np.ndarray
    first_allocation: np.ndarray
    objective: float
    runtime_seconds: float
    status: int
    success: bool
    mip_gap: float
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, default=BenchmarkConfig.seeds)
    parser.add_argument(
        "--stresses",
        nargs="+",
        choices=STRESSES,
        default=list(STRESSES),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
    )
    parser.add_argument("--scalability-repetitions", type=int, default=3)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Two seeds, two stresses, and a shorter day for implementation checks.",
    )
    parser.add_argument(
        "--scalability-only",
        action="store_true",
        help="Run only the synthetic MILP scaling instances and preserve main runs.",
    )
    return parser.parse_args()


def haversine_matrix(coordinates: np.ndarray) -> np.ndarray:
    lon = np.radians(coordinates[:, 0])
    lat = np.radians(coordinates[:, 1])
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def largest_remainder_allocation(total: int, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if total <= 0:
        return np.zeros_like(weights, dtype=int)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones_like(weights, dtype=float)
    raw = total * weights / weights.sum()
    allocation = np.floor(raw).astype(int)
    remainder = total - int(allocation.sum())
    if remainder > 0:
        order = np.argsort(-(raw - allocation))
        allocation[order[:remainder]] += 1
    return allocation


def _dense_hourly_cube(
    frame: pd.DataFrame,
    region_ids: list[str],
    dates: list[pd.Timestamp],
    slots_per_day: int,
) -> np.ndarray:
    if not dates:
        return np.empty((0, slots_per_day, len(region_ids)), dtype=float)
    source_slots_per_day = 48
    if source_slots_per_day % slots_per_day != 0:
        raise ValueError(
            "slots_per_day must divide the 48 half-hour LaDe source slots."
        )
    bin_width = source_slots_per_day // slots_per_day
    grouped = (
        frame.assign(hour=frame["release_slot"].astype(int) // bin_width)
        .groupby(["service_date", "hour", "region_id"], observed=True)["task_count"]
        .sum()
    )
    index = pd.MultiIndex.from_product(
        [dates, range(slots_per_day), region_ids],
        names=["service_date", "hour", "region_id"],
    )
    return (
        grouped.reindex(index, fill_value=0)
        .to_numpy(dtype=float)
        .reshape(len(dates), slots_per_day, len(region_ids))
    )


def build_case(
    events_path: Path,
    profiles_path: Path,
    config: BenchmarkConfig,
) -> CaseData:
    profile_columns = [
        "city",
        "region_id",
        "centroid_lng",
        "centroid_lat",
        "train_tasks",
        "smoothed_late_risk",
    ]
    profiles = pd.read_parquet(profiles_path, columns=profile_columns)
    profiles["region_id"] = profiles["region_id"].astype(str)
    selected = (
        profiles.loc[profiles["city"].eq(config.city)]
        .nlargest(config.service_nodes, "train_tasks")
        .reset_index(drop=True)
    )
    if len(selected) != config.service_nodes:
        raise ValueError(
            f"Expected {config.service_nodes} service nodes, found {len(selected)}."
        )
    region_ids = selected["region_id"].tolist()

    event_columns = [
        "city",
        "region_id",
        "service_date",
        "release_slot",
        "task_count",
        "split",
    ]
    events = pd.read_parquet(events_path, columns=event_columns)
    events["region_id"] = events["region_id"].astype(str)
    events = events.loc[
        events["city"].eq(config.city) & events["region_id"].isin(region_ids)
    ].copy()

    train_dates = sorted(
        pd.to_datetime(events.loc[events["split"].eq("train"), "service_date"])
        .drop_duplicates()
        .tolist()
    )
    test_dates = sorted(
        pd.to_datetime(events.loc[events["split"].eq("test"), "service_date"])
        .drop_duplicates()
        .tolist()
    )
    train_cube = _dense_hourly_cube(
        events.loc[events["split"].eq("train")],
        region_ids,
        train_dates,
        config.slots_per_day,
    )
    test_cube = _dense_hourly_cube(
        events.loc[events["split"].eq("test")],
        region_ids,
        test_dates,
        config.slots_per_day,
    )
    if not test_dates:
        raise ValueError("No test dates were found for the selected city.")

    node_types = np.array(
        ["A"] + ["B"] * config.fixed_nodes + ["C"] * (config.service_nodes - config.fixed_nodes),
        dtype=object,
    )
    selected_coords = selected[["centroid_lng", "centroid_lat"]].to_numpy(float)
    demand_weights = selected["train_tasks"].to_numpy(float)
    depot = np.average(selected_coords, axis=0, weights=demand_weights)
    coordinates = np.vstack([depot, selected_coords])
    distance_km = haversine_matrix(coordinates)

    risk = selected["smoothed_late_risk"].to_numpy(float)
    if np.ptp(risk) > 0:
        risk = (risk - risk.min()) / np.ptp(risk)
    else:
        risk = np.zeros_like(risk)
    risk_proxy = np.concatenate([[0.0], risk])

    fixed_capacity = np.zeros(config.service_nodes + 1, dtype=float)
    for node in range(config.fixed_nodes):
        values = train_cube[:, :, node].reshape(-1)
        fixed_capacity[node + 1] = max(
            1.0, float(np.quantile(values, config.fixed_capacity_quantile))
        )

    # Calibrate the fleet to the high end of observed mobile-relevant workload.
    expected_target = np.zeros_like(train_cube, dtype=float)
    for node in range(config.service_nodes):
        demand = train_cube[:, :, node]
        if node < config.fixed_nodes:
            cap = fixed_capacity[node + 1]
            overload = np.maximum(demand - cap, 0.0)
            failure_rescue = np.minimum(demand, cap) * 0.12
            expected_target[:, :, node] = overload + failure_rescue
        else:
            expected_target[:, :, node] = demand
    system_target = expected_target.sum(axis=2).reshape(-1)
    reference_units = int(
        math.ceil(
            np.quantile(system_target, config.fleet_reference_quantile)
            / config.mobile_unit_capacity
        )
    )
    fleet_units = max(3, int(math.ceil(reference_units * config.fleet_ratio)))

    allocation_weights = np.concatenate(
        [[0.08], expected_target.mean(axis=(0, 1))]
    )
    initial_allocation = largest_remainder_allocation(
        fleet_units, allocation_weights
    )
    test_demand = {
        date: np.column_stack(
            [np.zeros(config.slots_per_day), test_cube[idx]]
        )
        for idx, date in enumerate(test_dates)
    }
    train_demand = np.concatenate(
        [
            np.zeros((train_cube.shape[0], config.slots_per_day, 1)),
            train_cube,
        ],
        axis=2,
    )
    return CaseData(
        region_ids=["DEPOT_A"] + region_ids,
        node_types=node_types,
        coordinates=coordinates,
        distance_km=distance_km,
        risk_proxy=risk_proxy,
        fixed_capacity=fixed_capacity,
        initial_allocation=initial_allocation,
        fleet_units=fleet_units,
        mobile_unit_capacity=config.mobile_unit_capacity,
        test_dates=test_dates,
        test_demand=test_demand,
        train_demand=train_demand,
    )


def stress_adjusted_demand(
    base_demand: np.ndarray,
    stress: str,
    slots_per_day: int = 24,
) -> np.ndarray:
    demand = np.asarray(base_demand, dtype=float).copy()
    hours = np.arange(demand.shape[0]) * 24.0 / slots_per_day
    peak = (hours >= 7) & (hours <= 12)
    if stress == "demand_surge":
        demand[peak, 1:] *= 1.45
    elif stress == "weather":
        demand[:, 1:] *= 1.12
        demand[peak, 1:] *= 1.10
    return np.rint(demand).astype(int)


def cause_parameters(
    case: CaseData,
    demand: np.ndarray,
    hour: int,
    stress: str,
    config: BenchmarkConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if config is None:
        config = BenchmarkConfig()
    n_nodes = len(case.region_ids)
    p_total = np.zeros(n_nodes, dtype=float)
    mix = np.zeros((n_nodes, len(CAUSES)), dtype=float)
    q = np.tile(np.array([0.92, 0.06, 0.16], dtype=float), (n_nodes, 1))

    b_mask = case.node_types == "B"
    capacity = np.maximum(case.fixed_capacity, 1.0)
    load_ratio = demand / capacity
    sigmoid = 1.0 / (1.0 + np.exp(-2.2 * (load_ratio - 0.85)))
    p_total[b_mask] = np.clip(
        0.035
        + 0.105 * sigmoid[b_mask]
        + 0.035 * case.risk_proxy[b_mask],
        0.03,
        0.24,
    )
    # Cause composition is spatially heterogeneous even when the aggregate
    # interruption rate is similar.  A higher LaDe lateness proxy represents
    # infrastructure-sensitive sites; lower-risk sites have a larger
    # user-stop share.  This is the information discarded by the
    # aggregate-risk ablation.
    heterogeneity = float(np.clip(config.cause_heterogeneity, 0.0, 1.0))
    global_mix = np.array(
        [0.4122093023255814, 0.3877906976744186, 0.20],
        dtype=float,
    )
    risk = case.risk_proxy[b_mask]
    heterogeneous_mix = np.column_stack(
        [
            0.25 + 0.45 * risk,
            0.55 - 0.35 * risk,
            0.20 - 0.10 * risk,
        ]
    )
    mix[b_mask] = (
        global_mix
        + heterogeneity * (heterogeneous_mix - global_mix)
    )

    hour_of_day = hour * 24.0 / config.slots_per_day
    active_peak = 7 <= hour_of_day <= 14
    b_indices = np.flatnonzero(b_mask)
    risk_order = b_indices[np.argsort(-case.risk_proxy[b_indices])]
    cluster_size = min(
        max(3, int(math.ceil(0.25 * len(b_indices)))),
        max(1, len(b_indices) // 2),
    )
    infrastructure_cluster = risk_order[:cluster_size]
    user_stop_cluster = risk_order[-cluster_size:]
    if stress == "equipment_fault" and active_peak:
        # Two clusters have the same aggregate interruption probability but
        # sharply different recoverability.  Aggregate-risk therefore cannot
        # identify where mobile backup has the greater expected benefit.
        p_total[infrastructure_cluster] = 0.38
        p_total[user_stop_cluster] = 0.38
        infrastructure_mix = np.array([0.86, 0.08, 0.06])
        user_stop_mix = np.array([0.08, 0.82, 0.10])
        mix[infrastructure_cluster] = global_mix + heterogeneity * (
            infrastructure_mix - global_mix
        )
        mix[user_stop_cluster] = global_mix + heterogeneity * (
            user_stop_mix - global_mix
        )
    elif stress == "unrecoverable_shift" and active_peak:
        # Aggregate risk is deliberately highest at user-stop dominated sites,
        # while the lower-rate infrastructure cluster remains more recoverable.
        # This reverses the priority ordering for a cause-aware policy.
        p_total[infrastructure_cluster] = 0.28
        p_total[user_stop_cluster] = 0.46
        infrastructure_mix = np.array([0.86, 0.08, 0.06])
        user_stop_mix = np.array([0.05, 0.88, 0.07])
        mix[infrastructure_cluster] = global_mix + heterogeneity * (
            infrastructure_mix - global_mix
        )
        mix[user_stop_cluster] = global_mix + heterogeneity * (
            user_stop_mix - global_mix
        )
        remaining = np.setdiff1d(b_indices, np.concatenate(
            [infrastructure_cluster, user_stop_cluster]
        ))
        p_total[remaining] = np.clip(p_total[remaining] * 1.35, 0.0, 0.38)
        remaining_mix = np.array([0.18, 0.65, 0.17])
        mix[remaining] = global_mix + heterogeneity * (
            remaining_mix - global_mix
        )
    elif stress == "weather":
        p_total[b_mask] = np.clip(p_total[b_mask] * 1.25, 0.0, 0.42)
        weather_mix = np.array([0.48, 0.30, 0.22])
        mix[b_mask] = global_mix + heterogeneity * (
            weather_mix - global_mix
        )
        q[b_mask] *= 0.82

    row_sums = mix.sum(axis=1, keepdims=True)
    mix = np.divide(mix, row_sums, out=np.zeros_like(mix), where=row_sums > 0)
    return p_total, mix, q


def stress_mobile_parameters(
    config: BenchmarkConfig,
    stress: str,
) -> tuple[float, float]:
    capacity = config.mobile_unit_capacity
    max_km = config.max_relocation_km
    if stress == "weather":
        capacity *= 0.74
        max_km *= 0.70
    elif stress == "recharge_limited":
        capacity *= 0.68
    return capacity, max_km


def build_realization(
    case: CaseData,
    base_demand: np.ndarray,
    stress: str,
    seed: int,
    config: BenchmarkConfig,
) -> Realization:
    rng = np.random.default_rng(seed)
    demand = stress_adjusted_demand(
        base_demand,
        stress,
        config.slots_per_day,
    )
    t_steps, n_nodes = demand.shape
    mobile_target = np.zeros_like(demand, dtype=float)
    reliable_fixed = np.zeros_like(demand, dtype=float)
    unavoidable = np.zeros_like(demand, dtype=float)
    recoverable = np.zeros_like(demand, dtype=float)
    base_mobile = np.zeros_like(demand, dtype=float)
    cause_counts = np.zeros((t_steps, n_nodes, len(CAUSES)), dtype=int)
    recovered_counts = np.zeros_like(cause_counts)
    p_all = np.zeros((t_steps, n_nodes), dtype=float)
    mix_all = np.zeros((t_steps, n_nodes, len(CAUSES)), dtype=float)
    q_all = np.zeros_like(mix_all)
    mobile_capacity = np.zeros(t_steps, dtype=float)
    max_relocation_km = np.zeros(t_steps, dtype=float)

    for t in range(t_steps):
        p_total, mix, q = cause_parameters(
            case,
            demand[t],
            t,
            stress,
            config,
        )
        p_all[t] = p_total
        mix_all[t] = mix
        q_all[t] = q
        mobile_capacity[t], max_relocation_km[t] = stress_mobile_parameters(
            config, stress
        )
        for node in range(n_nodes):
            d = int(demand[t, node])
            if d <= 0 or case.node_types[node] == "A":
                continue
            if case.node_types[node] == "C":
                mobile_target[t, node] = d
                base_mobile[t, node] = d
                continue

            at_fixed = min(d, int(round(case.fixed_capacity[node])))
            overload = max(d - at_fixed, 0)
            probabilities = np.concatenate(
                [[1.0 - p_total[node]], p_total[node] * mix[node]]
            )
            probabilities /= probabilities.sum()
            sampled = rng.multinomial(at_fixed, probabilities)
            reliable_fixed[t, node] = sampled[0]
            cause_counts[t, node] = sampled[1:]
            for cause in range(len(CAUSES)):
                recovered_counts[t, node, cause] = rng.binomial(
                    cause_counts[t, node, cause], q[node, cause]
                )
            recoverable[t, node] = recovered_counts[t, node].sum()
            unavoidable[t, node] = (
                cause_counts[t, node] - recovered_counts[t, node]
            ).sum()
            base_mobile[t, node] = overload
            mobile_target[t, node] = overload + recoverable[t, node]

    return Realization(
        demand=demand,
        mobile_target=mobile_target,
        reliable_fixed=reliable_fixed,
        unavoidable=unavoidable,
        recoverable=recoverable,
        overload_or_mobile_only=base_mobile,
        cause_counts=cause_counts,
        recovered_counts=recovered_counts,
        total_interruption_probability=p_all,
        cause_mix=mix_all,
        recovery_probability=q_all,
        mobile_capacity=mobile_capacity,
        max_relocation_km=max_relocation_km,
    )


def build_forecasts(
    case: CaseData,
    realization: Realization,
    stress: str,
    seed: int,
    config: BenchmarkConfig,
) -> list[ForecastBundle]:
    rng = np.random.default_rng(seed)
    bundles: list[ForecastBundle] = []
    t_steps, n_nodes = realization.demand.shape
    s_count = config.forecast_scenarios
    horizon = config.rolling_horizon
    for t in range(t_steps):
        demand_s = np.zeros((s_count, horizon, n_nodes), dtype=float)
        p_s = np.zeros_like(demand_s)
        mix_s = np.zeros((s_count, horizon, n_nodes, len(CAUSES)), dtype=float)
        q_s = np.zeros_like(mix_s)
        capacity_h = np.zeros(horizon, dtype=float)
        max_km_h = np.zeros(horizon, dtype=float)
        for h in range(horizon):
            source_t = min(t + h, t_steps - 1)
            mean_demand = realization.demand[source_t].astype(float)
            p_base, mix_base, q_base = cause_parameters(
                case,
                mean_demand,
                source_t,
                stress,
                config,
            )
            capacity_h[h] = realization.mobile_capacity[source_t]
            max_km_h[h] = realization.max_relocation_km[source_t]
            for s in range(s_count):
                demand_s[s, h] = rng.poisson(np.maximum(mean_demand, 0.0))
                p_noise = rng.lognormal(mean=0.0, sigma=0.10, size=n_nodes)
                p_s[s, h] = np.clip(p_base * p_noise, 0.0, 0.70)
                for node in range(n_nodes):
                    if mix_base[node].sum() <= 0:
                        continue
                    concentration = np.maximum(mix_base[node] * 80.0, 0.5)
                    mix_s[s, h, node] = rng.dirichlet(concentration)
                q_s[s, h] = np.clip(
                    q_base + rng.normal(0.0, 0.025, size=q_base.shape),
                    0.0,
                    1.0,
                )
        bundles.append(
            ForecastBundle(
                demand=demand_s,
                total_interruption_probability=p_s,
                cause_mix=mix_s,
                recovery_probability=q_s,
                mobile_capacity=capacity_h,
                max_relocation_km=max_km_h,
            )
        )
    return bundles


def policy_gap_scenarios(
    policy: str,
    case: CaseData,
    bundle: ForecastBundle,
    oracle_target: np.ndarray,
    config: BenchmarkConfig,
) -> np.ndarray:
    s_count, horizon, n_nodes = bundle.demand.shape
    if policy == "oracle":
        target = np.asarray(oracle_target, dtype=float)
        if target.shape[0] < horizon:
            target = np.vstack(
                [target, np.repeat(target[-1:, :], horizon - target.shape[0], axis=0)]
            )
        return np.repeat(target[None, :horizon, :], s_count, axis=0)

    gap = np.zeros_like(bundle.demand, dtype=float)
    for node in range(n_nodes):
        node_type = case.node_types[node]
        demand = bundle.demand[:, :, node]
        if node_type == "A":
            continue
        if node_type == "C":
            gap[:, :, node] = demand
            continue
        fixed = case.fixed_capacity[node]
        at_fixed = np.minimum(demand, fixed)
        overload = np.maximum(demand - fixed, 0.0)
        if policy == "demand_only":
            expected_recovery = 0.0
        elif policy == "aggregate_risk":
            expected_recovery = (
                bundle.total_interruption_probability[:, :, node]
                * config.aggregate_recovery_rate
            )
        elif policy == "ecap_rss":
            expected_recovery = (
                bundle.total_interruption_probability[:, :, node, None]
                * bundle.cause_mix[:, :, node]
                * bundle.recovery_probability[:, :, node]
            ).sum(axis=2)
        else:
            raise ValueError(f"Unsupported deployable policy: {policy}")
        gap[:, :, node] = overload + at_fixed * expected_recovery
    return gap


def solve_rolling_milp(
    gap_scenarios: np.ndarray,
    current_allocation: np.ndarray,
    distance_km: np.ndarray,
    unit_capacity: np.ndarray,
    max_relocation_km: np.ndarray,
    config: BenchmarkConfig,
) -> SolveResult:
    gap = np.asarray(gap_scenarios, dtype=float)
    s_count, horizon, n_nodes = gap.shape
    fleet = int(np.asarray(current_allocation, dtype=int).sum())
    current = np.asarray(current_allocation, dtype=int)
    if current.shape != (n_nodes,):
        raise ValueError("current_allocation has the wrong shape")

    nx = horizon * n_nodes
    nf = horizon * n_nodes * n_nodes
    ny = s_count * horizon * n_nodes
    eta_index = nx + nf + ny
    z_offset = eta_index + 1
    n_variables = z_offset + s_count

    def ix(h: int, n: int) -> int:
        return h * n_nodes + n

    def iff(h: int, i: int, j: int) -> int:
        return nx + (h * n_nodes + i) * n_nodes + j

    def iy(s: int, h: int, n: int) -> int:
        return nx + nf + (s * horizon + h) * n_nodes + n

    objective = np.zeros(n_variables, dtype=float)
    for s in range(s_count):
        for h in range(horizon):
            for node in range(n_nodes):
                objective[iy(s, h, node)] = 1.0 / s_count
    objective[eta_index] = config.cvar_weight
    objective[z_offset : z_offset + s_count] = (
        config.cvar_weight / ((1.0 - config.cvar_alpha) * s_count)
    )
    for h in range(horizon):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    objective[iff(h, i, j)] = (
                        config.relocation_cost_per_km * distance_km[i, j]
                    )

    lower = np.zeros(n_variables, dtype=float)
    upper = np.full(n_variables, np.inf, dtype=float)
    upper[:nx] = fleet
    upper[nx : nx + nf] = fleet
    for h in range(horizon):
        max_km = float(max_relocation_km[h])
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and distance_km[i, j] > max_km:
                    upper[iff(h, i, j)] = 0.0
    integrality = np.zeros(n_variables, dtype=int)
    integrality[: nx + nf] = 1

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add_constraint(
        entries: Iterable[tuple[int, float]],
        lb: float,
        ub: float,
    ) -> None:
        row = len(row_lower)
        for column, value in entries:
            if value:
                rows.append(row)
                columns.append(column)
                values.append(float(value))
        row_lower.append(float(lb))
        row_upper.append(float(ub))

    for h in range(horizon):
        add_constraint(((ix(h, n), 1.0) for n in range(n_nodes)), fleet, fleet)
        for i in range(n_nodes):
            entries = [(iff(h, i, j), 1.0) for j in range(n_nodes)]
            if h == 0:
                add_constraint(entries, current[i], current[i])
            else:
                entries.append((ix(h - 1, i), -1.0))
                add_constraint(entries, 0.0, 0.0)
        for j in range(n_nodes):
            entries = [(iff(h, i, j), 1.0) for i in range(n_nodes)]
            entries.append((ix(h, j), -1.0))
            add_constraint(entries, 0.0, 0.0)
        max_moves = max(1, int(math.floor(config.max_relocation_fraction * fleet)))
        add_constraint(
            (
                (iff(h, i, j), 1.0)
                for i in range(n_nodes)
                for j in range(n_nodes)
                if i != j
            ),
            -np.inf,
            max_moves,
        )

    for s in range(s_count):
        for h in range(horizon):
            max_km = float(max_relocation_km[h])
            if config.travel_capacity_loss:
                travel_factor = np.clip(
                    1.0 - distance_km / max(max_km, 1e-6),
                    0.0,
                    1.0,
                )
            else:
                travel_factor = np.ones_like(distance_km, dtype=float)
            np.fill_diagonal(travel_factor, 1.0)
            for node in range(n_nodes):
                entries = [(iy(s, h, node), 1.0)]
                entries.extend(
                    (
                        iff(h, origin, node),
                        unit_capacity[h] * travel_factor[origin, node],
                    )
                    for origin in range(n_nodes)
                )
                add_constraint(entries, gap[s, h, node], np.inf)

        entries = [(z_offset + s, 1.0), (eta_index, 1.0)]
        entries.extend(
            (iy(s, h, node), -1.0)
            for h in range(horizon)
            for node in range(n_nodes)
        )
        add_constraint(entries, 0.0, np.inf)

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(len(row_lower), n_variables),
        dtype=float,
    ).tocsr()
    constraint = LinearConstraint(
        matrix, np.asarray(row_lower), np.asarray(row_upper)
    )
    start = time.perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraint,
        options={
            "time_limit": config.solver_time_limit_seconds,
            "mip_rel_gap": config.solver_mip_gap,
            "presolve": True,
        },
    )
    runtime = time.perf_counter() - start
    if result.x is None:
        fallback_flow = np.zeros((n_nodes, n_nodes), dtype=int)
        np.fill_diagonal(fallback_flow, current)
        return SolveResult(
            first_flow=fallback_flow,
            first_allocation=current.copy(),
            objective=float("nan"),
            runtime_seconds=runtime,
            status=int(result.status),
            success=False,
            mip_gap=float("nan"),
            message=str(result.message),
        )

    first_flow = np.rint(
        result.x[nx : nx + n_nodes * n_nodes].reshape(n_nodes, n_nodes)
    ).astype(int)
    first_allocation = np.rint(result.x[:n_nodes]).astype(int)
    return SolveResult(
        first_flow=first_flow,
        first_allocation=first_allocation,
        objective=float(result.fun),
        runtime_seconds=runtime,
        status=int(result.status),
        success=bool(result.success),
        mip_gap=float(getattr(result, "mip_gap", np.nan)),
        message=str(result.message),
    )


def cvar(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    threshold = float(np.quantile(values, alpha))
    tail = values[values >= threshold]
    return float(tail.mean()) if tail.size else threshold


def execute_policy(
    policy: str,
    case: CaseData,
    realization: Realization,
    forecasts: list[ForecastBundle],
    config: BenchmarkConfig,
    hours_to_run: int | None = None,
) -> dict[str, float | int | str]:
    t_steps, n_nodes = realization.demand.shape
    if hours_to_run is not None:
        t_steps = min(t_steps, hours_to_run)
    current = case.initial_allocation.copy()
    total_demand = 0.0
    total_mobile_target = 0.0
    mobile_served = 0.0
    total_unavoidable = 0.0
    total_unserved = 0.0
    movement_km = 0.0
    available_mobile_capacity = 0.0
    solver_runtime = 0.0
    solver_failures = 0
    solver_gaps: list[float] = []
    slot_loss_rate: list[float] = []

    for t in range(t_steps):
        demand = realization.demand[t]
        target = realization.mobile_target[t]
        total_demand += demand.sum()
        total_mobile_target += target.sum()
        total_unavoidable += realization.unavoidable[t].sum()

        if policy == "fixed_only":
            available = np.zeros(n_nodes, dtype=float)
        else:
            oracle_horizon = realization.mobile_target[
                t : min(t + config.rolling_horizon, t_steps)
            ]
            gap = policy_gap_scenarios(
                policy, case, forecasts[t], oracle_horizon, config
            )
            solved = solve_rolling_milp(
                gap,
                current,
                case.distance_km,
                forecasts[t].mobile_capacity,
                forecasts[t].max_relocation_km,
                config,
            )
            solver_runtime += solved.runtime_seconds
            solver_failures += int(not solved.success)
            if np.isfinite(solved.mip_gap):
                solver_gaps.append(solved.mip_gap)
            flow = solved.first_flow
            current = solved.first_allocation
            max_km = realization.max_relocation_km[t]
            if config.travel_capacity_loss:
                travel_factor = np.clip(
                    1.0 - case.distance_km / max(max_km, 1e-6),
                    0.0,
                    1.0,
                )
            else:
                travel_factor = np.ones_like(case.distance_km, dtype=float)
            np.fill_diagonal(travel_factor, 1.0)
            available = (
                flow
                * travel_factor
                * realization.mobile_capacity[t]
            ).sum(axis=0)
            movement_km += float(
                (flow * case.distance_km * (~np.eye(n_nodes, dtype=bool))).sum()
            )

        served = np.minimum(available, target)
        mobile_served += served.sum()
        available_mobile_capacity += available.sum()
        slot_unserved = realization.unavoidable[t].sum() + (
            target - served
        ).sum()
        total_unserved += slot_unserved
        slot_loss_rate.append(slot_unserved / max(demand.sum(), 1.0))

    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "total_demand": total_demand,
        "mobile_target": total_mobile_target,
        "mobile_served": mobile_served,
        "unavoidable_unserved": total_unavoidable,
        "total_unserved": total_unserved,
        "service_rate": 1.0 - total_unserved / max(total_demand, 1.0),
        "recoverable_service_rate": mobile_served / max(total_mobile_target, 1.0),
        "mobile_target_rate": total_mobile_target / max(total_demand, 1.0),
        "unavoidable_rate": total_unavoidable / max(total_demand, 1.0),
        "cvar90_slot_loss_rate": cvar(
            np.asarray(slot_loss_rate), config.cvar_alpha
        ),
        "movement_km": movement_km,
        "movement_km_per_1000_requests": 1000.0
        * movement_km
        / max(total_demand, 1.0),
        "mobile_utilization": mobile_served
        / max(available_mobile_capacity, 1.0),
        "solver_runtime_seconds": solver_runtime,
        "mean_solver_mip_gap": float(np.mean(solver_gaps))
        if solver_gaps
        else float("nan"),
        "solver_failures": solver_failures,
    }


def run_benchmark(
    case: CaseData,
    config: BenchmarkConfig,
    stresses: list[str],
    policies: list[str],
    seeds: int,
    hours_to_run: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for stress_index, stress in enumerate(stresses):
        for seed_index in range(seeds):
            date = case.test_dates[seed_index % len(case.test_dates)]
            realization_seed = (
                config.random_seed + stress_index * 100_000 + seed_index
            )
            forecast_seed = realization_seed + 50_000
            realization = build_realization(
                case,
                case.test_demand[date],
                stress,
                realization_seed,
                config,
            )
            forecasts = build_forecasts(
                case, realization, stress, forecast_seed, config
            )
            for policy in policies:
                result = execute_policy(
                    policy,
                    case,
                    realization,
                    forecasts,
                    config,
                    hours_to_run=hours_to_run,
                )
                result.update(
                    {
                        "stress": stress,
                        "stress_label": STRESS_LABELS[stress],
                        "seed": seed_index,
                        "service_date": date.date().isoformat(),
                        "realization_seed": realization_seed,
                        "fleet_units": case.fleet_units,
                        "service_nodes": len(case.region_ids) - 1,
                        "fixed_nodes": int((case.node_types == "B").sum()),
                        "mobile_only_nodes": int((case.node_types == "C").sum()),
                    }
                )
                rows.append(result)
                print(
                    f"[{stress:20s}] seed={seed_index:02d} "
                    f"policy={policy:14s} service={result['service_rate']:.4f} "
                    f"cvar={result['cvar90_slot_loss_rate']:.4f} "
                    f"runtime={result['solver_runtime_seconds']:.2f}s",
                    flush=True,
                )
    return pd.DataFrame(rows)


def run_scalability(
    config: BenchmarkConfig,
    repetitions: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(config.random_seed + 900_000)
    settings = [
        (6, 3, 6),
        (13, 3, 6),
        (20, 3, 6),
        (30, 3, 6),
    ]
    for n_nodes, horizon, scenarios in settings:
        for repetition in range(repetitions):
            coordinates = rng.uniform(0.0, 25.0, size=(n_nodes, 2))
            distance = np.sqrt(
                ((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(
                    axis=2
                )
            )
            fleet = max(3, int(math.ceil(n_nodes * 1.7)))
            current = largest_remainder_allocation(
                fleet, rng.uniform(0.2, 1.0, size=n_nodes)
            )
            gap = rng.gamma(
                shape=2.0,
                scale=12.0,
                size=(scenarios, horizon, n_nodes),
            )
            scale_config = BenchmarkConfig(
                **{
                    **asdict(config),
                    "rolling_horizon": horizon,
                    "forecast_scenarios": scenarios,
                }
            )
            solved = solve_rolling_milp(
                gap,
                current,
                distance,
                np.full(horizon, config.mobile_unit_capacity),
                np.full(horizon, config.max_relocation_km),
                scale_config,
            )
            rows.append(
                {
                    "nodes": n_nodes,
                    "horizon": horizon,
                    "scenarios": scenarios,
                    "variables": (
                        horizon * n_nodes
                        + horizon * n_nodes * n_nodes
                        + scenarios * horizon * n_nodes
                        + 1
                        + scenarios
                    ),
                    "constraints": (
                        horizon * (2 * n_nodes + 2)
                        + scenarios * horizon * n_nodes
                        + scenarios
                    ),
                    "fleet_units": fleet,
                    "repetition": repetition,
                    "runtime_seconds": solved.runtime_seconds,
                    "success": solved.success,
                    "status": solved.status,
                    "mip_gap": solved.mip_gap,
                    "objective": solved.objective,
                    "message": solved.message,
                }
            )
            print(
                f"[scalability] nodes={n_nodes:02d} H={horizon} "
                f"S={scenarios} rep={repetition} "
                f"runtime={solved.runtime_seconds:.3f}s "
                f"success={solved.success}",
                flush=True,
            )
    return pd.DataFrame(rows)


def write_case_metadata(
    output_dir: Path,
    case: CaseData,
    config: BenchmarkConfig,
    stresses: list[str],
    policies: list[str],
) -> None:
    metadata = output_dir / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    payload = {
        "evidence_boundary": (
            "Observed LaDe-P demand calibration plus scenario-defined charging "
            "interruption and recoverability. Not real charging-field validation."
        ),
        "config": asdict(config),
        "stresses": stresses,
        "policies": policies,
        "policy_labels": POLICY_LABELS,
        "stress_labels": STRESS_LABELS,
        "causes": CAUSES,
        "case": {
            "region_ids": case.region_ids,
            "node_types": case.node_types.tolist(),
            "coordinates": case.coordinates.tolist(),
            "fixed_capacity": case.fixed_capacity.tolist(),
            "initial_allocation": case.initial_allocation.tolist(),
            "fleet_units": case.fleet_units,
            "test_dates": [d.date().isoformat() for d in case.test_dates],
        },
    }
    (metadata / "benchmark_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    node_table = pd.DataFrame(
        {
            "node_id": case.region_ids,
            "node_type": case.node_types,
            "longitude": case.coordinates[:, 0],
            "latitude": case.coordinates[:, 1],
            "risk_proxy": case.risk_proxy,
            "fixed_capacity_per_hour": case.fixed_capacity,
            "initial_mobile_units": case.initial_allocation,
        }
    )
    node_table.to_csv(metadata / "case_nodes.csv", index=False)


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        seeds=args.seeds,
        scalability_repetitions=args.scalability_repetitions,
    )
    stresses = list(args.stresses)
    policies = list(args.policies)
    hours_to_run: int | None = None
    seeds = args.seeds
    repetitions = args.scalability_repetitions
    if args.quick:
        stresses = ["normal", "equipment_fault"]
        policies = list(POLICIES)
        seeds = 2
        repetitions = 0

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    if args.scalability_only:
        scalability = run_scalability(config, repetitions)
        scalability.to_csv(
            output_dir / "raw" / "solver_scalability_runs.csv",
            index=False,
        )
        print(f"Wrote scalability outputs to {output_dir}", flush=True)
        return

    case = build_case(args.events.resolve(), args.profiles.resolve(), config)
    write_case_metadata(output_dir, case, config, stresses, policies)
    print(
        f"Case built: {len(case.region_ids)} total nodes, "
        f"{case.fleet_units} mobile units, {len(case.test_dates)} test dates.",
        flush=True,
    )

    benchmark = run_benchmark(
        case,
        config,
        stresses,
        policies,
        seeds,
        hours_to_run=hours_to_run,
    )
    benchmark.to_csv(output_dir / "raw" / "core_benchmark_runs.csv", index=False)

    scalability = run_scalability(config, repetitions)
    scalability.to_csv(
        output_dir / "raw" / "solver_scalability_runs.csv", index=False
    )
    print(f"Wrote benchmark outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
