"""Unified dispatch engine for the rigorous RSS revision.

This module is intentionally self-contained.  It never reads historical result
tables.  All policies share the same explicit in-transit executor and task-age
accounting; only their information and decision rules differ.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import coo_matrix

from .names import canonicalize_policy


@dataclass(frozen=True)
class DispatchConfig:
    slots_per_day: int = 48
    response_slots: int = 4
    horizon: int = 3
    scenarios: int = 3
    fleet: int = 4
    capacity_kwh: float = 20.0
    speed_km_per_slot: float = 15.0
    max_relocation_km: float = 30.0
    max_relocation_fraction: float = 0.75
    cvar_alpha: float = 0.90
    cvar_weight: float = 0.35
    relocation_cost_per_km: float = 0.03
    solver_time_limit_seconds: float = 5.0
    solver_mip_gap: float = 0.001
    # EDVF-RSS-Sparse keeps at most K destination candidates for each occupied
    # origin.  Candidates are ranked only by the current deadline-viability
    # envelope; no forecast or scenario statistic enters this policy.
    edvf_candidate_retention: int = 20
    # Quantization of the sparse viability field.  Gains within one cell are
    # treated as lexicographic ties, allowing the next EDVF stage (and then
    # movement) to resolve them.  Exact EDVF remains unquantized.
    edvf_field_resolution_kwh: float = 5.0
    # Number of cumulative deadline checkpoints retained in the EDVF field.
    # The default keeps every prefix (exact field); smaller values are used
    # only in the field-resolution sensitivity experiment.
    edvf_deadline_levels: int | None = None
    # Multiplier of the physical response-window time constant in the
    # secondary exponential recovery integral.  theta=1 is the EDVF default.
    edvf_urgency_scale: float = 1.0
    # Number of strict lexicographic stages executed by EDVF Exact.  The
    # production method uses all three; 1/2 are mechanism ablations.
    edvf_lexicographic_stages: int = 3
    # P0-1: age-weighted remaining-slack terminal value.  At the plan end the
    # leftover RSS is penalised in proportion to how close it is to its true
    # expiry (age w-1 = about to expire = 1.0; age 0 = just arrived = 0.25).
    # Length must equal response_slots; if None a linear schedule is used.
    terminal_weights: tuple[float, ...] | None = None


def _terminal_schedule(w: int) -> tuple[float, ...]:
    """Default monotone remaining-slack weights for age 0..w-1."""
    return tuple(float(x) for x in np.linspace(0.25, 1.0, w))


@dataclass
class SolveResult:
    first_flow: np.ndarray
    runtime_seconds: float
    feasible: bool
    optimal: bool
    status: int
    mip_gap: float
    objective: float
    message: str


def transit_slots(distance_km: np.ndarray, speed_km_per_slot: float) -> np.ndarray:
    tau = np.ceil(np.asarray(distance_km, dtype=float) / speed_km_per_slot - 1e-12).astype(int)
    np.fill_diagonal(tau, 0)
    return tau


def shift_age(age: np.ndarray, arrivals: np.ndarray) -> float:
    """Advance one slot; return work that expired before this slot's service."""
    expired = float(age[-1].sum())
    age[1:] = age[:-1].copy()
    age[0] = np.asarray(arrivals, dtype=float)
    return expired


def serve_oldest(age: np.ndarray, capacity: np.ndarray) -> tuple[float, np.ndarray]:
    """Serve oldest cohorts first and return total and service by age index."""
    remaining = np.asarray(capacity, dtype=float).copy()
    by_age = np.zeros(age.shape[0], dtype=float)
    for a in range(age.shape[0] - 1, -1, -1):
        served = np.minimum(age[a], remaining)
        age[a] -= served
        remaining -= served
        by_age[a] = float(served.sum())
    return float(by_age.sum()), by_age


def _cvar_upper(values: np.ndarray, alpha: float) -> np.ndarray:
    """Column-wise empirical upper-tail mean."""
    values = np.asarray(values, dtype=float)
    count = max(1, int(math.ceil((1.0 - alpha) * values.shape[0])))
    return np.sort(values, axis=0)[-count:].mean(axis=0)


class ExactRollingMILP:
    """Time-expanded integer vehicle flow with scenario task-age accounting."""

    def __init__(self, distance_km: np.ndarray, config: DispatchConfig):
        self.distance = np.asarray(distance_km, dtype=float)
        self.cfg = config
        self.n = self.distance.shape[0]
        self.h = config.horizon
        self.s = config.scenarios
        self.w = config.response_slots
        self.tau = transit_slots(self.distance, config.speed_km_per_slot)
        self._build()

    def _build(self) -> None:
        n, h, s, w, cfg = self.n, self.h, self.s, self.w, self.cfg
        nf = h * n * n
        ni = s * h * w * n
        ns = ni
        loss0 = nf + ni + ns
        eta = loss0 + s
        z0 = eta + 1
        nv = z0 + s
        self.nv, self.loss0, self.eta, self.z0 = nv, loss0, eta, z0

        def f(t: int, i: int, j: int) -> int:
            return (t * n + i) * n + j

        def inv(k: int, t: int, a: int, i: int) -> int:
            return nf + (((k * h + t) * w + a) * n + i)

        def srv(k: int, t: int, a: int, i: int) -> int:
            return nf + ni + (((k * h + t) * w + a) * n + i)

        self.f, self.inv, self.srv = f, inv, srv
        # P0-1: remaining-slack terminal weights, aligned to response_slots.
        tw = getattr(cfg, "terminal_weights", None)
        if tw is None:
            tw = _terminal_schedule(w)
        else:
            tw = tuple(float(x) for x in tw)
            if len(tw) != w:
                raise ValueError(f"terminal_weights must have length {w}, got {len(tw)}")
        self.terminal_weights = tw
        objective = np.zeros(nv, dtype=float)
        for k in range(s):
            objective[loss0 + k] = 1.0 / s
        if cfg.cvar_weight > 0:
            objective[eta] = cfg.cvar_weight
            objective[z0 : z0 + s] = cfg.cvar_weight / ((1.0 - cfg.cvar_alpha) * s)
        for t in range(h):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        objective[f(t, i, j)] = cfg.relocation_cost_per_km * self.distance[i, j]

        lower = np.zeros(nv, dtype=float)
        upper = np.full(nv, np.inf, dtype=float)
        upper[:nf] = cfg.fleet
        for t in range(h):
            for i in range(n):
                for j in range(n):
                    if i != j and self.distance[i, j] > cfg.max_relocation_km + 1e-12:
                        upper[f(t, i, j)] = 0.0
        integrality = np.zeros(nv, dtype=int)
        integrality[:nf] = 1

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        lb: list[float] = []
        ub: list[float] = []

        def add(entries: Iterable[tuple[int, float]], lo: float, hi: float) -> int:
            r = len(lb)
            for c, v in entries:
                if v:
                    rows.append(r); cols.append(c); vals.append(float(v))
            lb.append(float(lo)); ub.append(float(hi))
            return r

        flow_rows = np.empty((h, n), dtype=int)
        for t in range(h):
            for i in range(n):
                entries = [(f(t, i, j), 1.0) for j in range(n)]
                if t > 0:
                    entries.append((f(t - 1, i, i), -1.0))
                    for origin in range(n):
                        if origin == i:
                            continue
                        depart = t - int(self.tau[origin, i])
                        if depart >= 0:
                            entries.append((f(depart, origin, i), -1.0))
                flow_rows[t, i] = add(entries, 0.0, 0.0)
            max_moves = int(math.floor(cfg.max_relocation_fraction * cfg.fleet))
            add(
                ((f(t, i, j), 1.0) for i in range(n) for j in range(n) if i != j),
                -np.inf,
                max_moves,
            )

        inv_rows = np.full((s, h, w, n), -1, dtype=int)
        for k in range(s):
            for t in range(h):
                for a in range(w):
                    for i in range(n):
                        if t == 0 or a == 0:
                            inv_rows[k, t, a, i] = add([(inv(k, t, a, i), 1.0)], 0.0, 0.0)
                        else:
                            add(
                                [
                                    (inv(k, t, a, i), 1.0),
                                    (inv(k, t - 1, a - 1, i), -1.0),
                                    (srv(k, t - 1, a - 1, i), 1.0),
                                ],
                                0.0,
                                0.0,
                            )
                        add([(srv(k, t, a, i), 1.0), (inv(k, t, a, i), -1.0)], -np.inf, 0.0)
                for i in range(n):
                    add(
                        [(srv(k, t, a, i), 1.0) for a in range(w)]
                        + [(f(t, i, i), -cfg.capacity_kwh)],
                        -np.inf,
                        0.0,
                    )
            loss_entries = [(loss0 + k, 1.0)]
            for t in range(h - 1):
                for i in range(n):
                    loss_entries.extend([(inv(k, t, w - 1, i), -1.0), (srv(k, t, w - 1, i), 1.0)])
            for a in range(w):
                weight = tw[a]
                for i in range(n):
                    loss_entries.extend([(inv(k, h - 1, a, i), -weight), (srv(k, h - 1, a, i), weight)])
            add(loss_entries, 0.0, 0.0)
            add([(z0 + k, 1.0), (loss0 + k, -1.0), (eta, 1.0)], 0.0, np.inf)

        self.objective = objective
        self.bounds = Bounds(lower, upper)
        self.integrality = integrality
        self.matrix = coo_matrix((vals, (rows, cols)), shape=(len(lb), nv), dtype=float).tocsr()
        self.base_lb = np.asarray(lb, dtype=float)
        self.base_ub = np.asarray(ub, dtype=float)
        self.flow_rows = flow_rows
        self.inv_rows = inv_rows

    def solve(
        self,
        visible_age: np.ndarray,
        future_arrivals: np.ndarray,
        current: np.ndarray,
        scheduled_arrivals: np.ndarray,
        fixed_first_flow: np.ndarray | None = None,
    ) -> SolveResult:
        cfg, n, h, s, w = self.cfg, self.n, self.h, self.s, self.w
        age = np.asarray(visible_age, dtype=float)
        future = np.asarray(future_arrivals, dtype=float)
        if age.shape != (w, n) or future.shape != (s, h, n):
            raise ValueError(f"Bad planner shapes: age={age.shape}, future={future.shape}")
        lb = self.base_lb.copy(); ub = self.base_ub.copy()
        for t in range(h):
            rhs = np.asarray(current if t == 0 else scheduled_arrivals[t], dtype=float)
            lb[self.flow_rows[t]] = rhs; ub[self.flow_rows[t]] = rhs
        for k in range(s):
            for a in range(w):
                r = self.inv_rows[k, 0, a]
                lb[r] = age[a]; ub[r] = age[a]
            for t in range(1, h):
                r = self.inv_rows[k, t, 0]
                lb[r] = future[k, t]; ub[r] = future[k, t]
        bounds = self.bounds
        if fixed_first_flow is not None:
            fixed = np.asarray(fixed_first_flow, dtype=float)
            if fixed.shape != (n, n):
                raise ValueError(f"Bad fixed first-flow shape: {fixed.shape}")
            if not np.isfinite(fixed).all():
                raise ValueError("Fixed first flow contains non-finite values")
            if (fixed < -1e-9).any() or not np.allclose(fixed, np.rint(fixed), atol=1e-9):
                raise ValueError("Fixed first flow must be a non-negative integer matrix")
            fixed = np.rint(fixed)
            if not np.allclose(fixed.sum(axis=1), np.asarray(current, dtype=float), atol=1e-9):
                raise ValueError("Fixed first flow violates current-vehicle row conservation")
            lower = np.asarray(self.bounds.lb, dtype=float).copy()
            upper = np.asarray(self.bounds.ub, dtype=float).copy()
            first_upper = upper[: n * n].reshape(n, n)
            if (fixed > first_upper + 1e-9).any():
                raise ValueError("Fixed first flow uses a forbidden or over-capacity arc")
            lower[: n * n] = fixed.reshape(-1); upper[: n * n] = fixed.reshape(-1)
            bounds = Bounds(lower, upper)
        started = time.perf_counter()
        result = milp(
            c=self.objective,
            integrality=self.integrality,
            bounds=bounds,
            constraints=LinearConstraint(self.matrix, lb, ub),
            options={
                "time_limit": cfg.solver_time_limit_seconds,
                "mip_rel_gap": cfg.solver_mip_gap,
                "presolve": True,
            },
        )
        elapsed = time.perf_counter() - started
        if result.x is None:
            flow = np.zeros((n, n), dtype=int)
            np.fill_diagonal(flow, np.asarray(current, dtype=int))
            return SolveResult(flow, elapsed, False, False, int(result.status), math.nan, math.nan, str(result.message))
        first = np.rint(result.x[: n * n].reshape(n, n)).astype(int)
        return SolveResult(
            first,
            elapsed,
            True,
            bool(result.success),
            int(result.status),
            float(getattr(result, "mip_gap", math.nan)),
            float(result.fun),
            str(result.message),
        )


def stay_flow(current: np.ndarray) -> np.ndarray:
    flow = np.zeros((len(current), len(current)), dtype=int)
    np.fill_diagonal(flow, np.asarray(current, dtype=int))
    return flow


def _unit_origins(current: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(current), dtype=int), np.asarray(current, dtype=int))


def _oldest(age: np.ndarray) -> np.ndarray:
    value = np.zeros(age.shape[1], dtype=int)
    for a in range(age.shape[0]):
        value[age[a] > 1e-12] = a
    return value


def njn_flow(current: np.ndarray, visible_age: np.ndarray, distance: np.ndarray, cfg: DispatchConfig) -> np.ndarray:
    origins = _unit_origins(current)
    destinations = origins.copy()
    remaining = visible_age.sum(axis=0).astype(float)
    urgency = _oldest(visible_age)
    move_budget = int(math.floor(cfg.max_relocation_fraction * cfg.fleet))
    moves = 0
    for u, origin in enumerate(origins):
        candidates = np.flatnonzero(remaining > 1e-12)
        candidates = candidates[distance[origin, candidates] <= cfg.max_relocation_km + 1e-12]
        if len(candidates) == 0:
            continue
        destination = min(candidates.tolist(), key=lambda j: (distance[origin, j], -urgency[j], -remaining[j], j))
        if destination != origin and moves >= move_budget:
            continue
        destinations[u] = destination
        moves += int(destination != origin)
        remaining[destination] = max(0.0, remaining[destination] - cfg.capacity_kwh)
    flow = np.zeros((len(current), len(current)), dtype=int)
    for i, j in zip(origins, destinations):
        flow[i, j] += 1
    return flow


def edf_reactive_flow(
    current: np.ndarray,
    visible_age: np.ndarray,
    distance: np.ndarray,
    cfg: DispatchConfig,
) -> np.ndarray:
    """Reactive earliest-deadline-first baseline with no future information."""
    current = np.asarray(current, dtype=int)
    age = np.asarray(visible_age, dtype=float)
    origins = _unit_origins(current)
    n = len(current)
    if len(origins) == 0:
        return np.zeros((n, n), dtype=int)

    tau = transit_slots(distance, cfg.speed_km_per_slot)
    remaining = age.copy()
    destinations = origins.copy()
    move_budget = int(math.floor(cfg.max_relocation_fraction * cfg.fleet))
    moves = 0

    for u, origin_raw in enumerate(origins):
        origin = int(origin_raw)
        candidates = []
        for j in range(n):
            if distance[origin, j] > cfg.max_relocation_km + 1e-12:
                continue
            lag = int(tau[origin, j])
            reachable = [
                a for a in range(cfg.response_slots)
                if remaining[a, j] > 1e-12 and a + lag <= cfg.response_slots - 1
            ]
            if not reachable:
                continue
            urgent_age = max(reachable)
            backlog = float(sum(remaining[a, j] for a in reachable))
            candidates.append((-urgent_age, lag, float(distance[origin, j]), -backlog, j))

        if not candidates:
            continue
        _, _, _, _, destination = min(candidates)
        if destination != origin and moves >= move_budget:
            local = [a for a in range(cfg.response_slots) if remaining[a, origin] > 1e-12]
            if not local:
                continue
            destination = origin

        destinations[u] = destination
        moves += int(destination != origin)
        lag = int(tau[origin, destination])
        cap = float(cfg.capacity_kwh)
        for a in range(cfg.response_slots - 1, -1, -1):
            if a + lag > cfg.response_slots - 1:
                continue
            take = min(cap, float(remaining[a, destination]))
            remaining[a, destination] -= take
            cap -= take
            if cap <= 1e-12:
                break

    flow = np.zeros((n, n), dtype=int)
    for i, j in zip(origins, destinations):
        flow[int(i), int(j)] += 1
    return flow


def _route_feasible(route: list[int], origin: int, visible_age: np.ndarray, distance: np.ndarray, cfg: DispatchConfig) -> bool:
    lag = 0
    here = origin
    urgency = _oldest(visible_age)
    for node in route:
        if distance[here, node] > cfg.max_relocation_km + 1e-12:
            return False
        lag += int(math.ceil(distance[here, node] / cfg.speed_km_per_slot - 1e-12))
        if lag > cfg.response_slots - 1 - int(urgency[node]):
            return False
        here = node
        lag += max(0, int(math.ceil(visible_age[:, node].sum() / cfg.capacity_kwh)) - 1)
    return True


def modified_savings_flow(current: np.ndarray, visible_age: np.ndarray, distance: np.ndarray, cfg: DispatchConfig) -> np.ndarray:
    origins = _unit_origins(current)
    nodes = [int(i) for i in np.flatnonzero(visible_age.sum(axis=0) > 1e-12)]
    if not nodes:
        return stay_flow(current)
    routes = [[i] for i in nodes]
    loads = [float(visible_age[:, i].sum()) for i in nodes]
    d0 = {i: float(distance[origins, i].min()) for i in nodes}
    savings = sorted(((d0[i] + d0[j] - distance[i, j], i, j) for x, i in enumerate(nodes) for j in nodes[x + 1 :]), reverse=True)
    for _, i, j in savings:
        ri = next((r for r, route in enumerate(routes) if i in route), None)
        rj = next((r for r, route in enumerate(routes) if j in route), None)
        if ri is None or rj is None or ri == rj:
            continue
        left, right = routes[ri], routes[rj]
        if i not in (left[0], left[-1]) or j not in (right[0], right[-1]):
            continue
        if left[-1] != i: left = left[::-1]
        if right[0] != j: right = right[::-1]
        merged = left + right
        if loads[ri] + loads[rj] > cfg.capacity_kwh * cfg.response_slots + 1e-12:
            continue
        if not any(_route_feasible(merged, int(o), visible_age, distance, cfg) for o in origins):
            continue
        keep, drop = min(ri, rj), max(ri, rj)
        routes[keep] = merged; loads[keep] = loads[ri] + loads[rj]
        routes.pop(drop); loads.pop(drop)
    destinations = origins.copy()
    free = set(range(len(origins)))
    move_budget = int(math.floor(cfg.max_relocation_fraction * cfg.fleet)); moves = 0
    for route in sorted(routes, key=lambda r: -max(_oldest(visible_age)[r])):
        choices = []
        for u in free:
            for candidate_route in (route, route[::-1]):
                if _route_feasible(candidate_route, int(origins[u]), visible_age, distance, cfg):
                    choices.append((distance[int(origins[u]), candidate_route[0]], u, candidate_route[0]))
        if not choices:
            continue
        _, u, destination = min(choices)
        if destination != origins[u] and moves >= move_budget:
            continue
        destinations[u] = destination; moves += int(destination != origins[u]); free.remove(u)
    flow = np.zeros((len(current), len(current)), dtype=int)
    for i, j in zip(origins, destinations): flow[i, j] += 1
    return flow


def ps_imcs_flow(
    current: np.ndarray,
    visible_age: np.ndarray,
    future: np.ndarray,
    distance: np.ndarray,
    cfg: DispatchConfig,
) -> np.ndarray:
    origins = _unit_origins(current)
    destinations = origins.copy()
    remaining = visible_age.sum(axis=0).astype(float)
    urgency = _oldest(visible_age)
    mean_future = future.mean(axis=(0, 1))
    attraction_all = (1.0 / (1.0 + distance) ** 2) @ mean_future
    move_budget = int(math.floor(cfg.max_relocation_fraction * cfg.fleet)); moves = 0
    assigned: list[int] = []
    idle: list[int] = []
    for u, origin in enumerate(origins):
        candidates = np.flatnonzero(remaining > 1e-12)
        candidates = candidates[distance[origin, candidates] <= cfg.max_relocation_km + 1e-12]
        if len(candidates):
            j = min(candidates.tolist(), key=lambda x: (distance[origin, x], -urgency[x], -remaining[x]))
            if j != origin and moves >= move_budget:
                idle.append(u); continue
            destinations[u] = j; moves += int(j != origin); assigned.append(j)
            remaining[j] = max(0.0, remaining[j] - cfg.capacity_kwh)
        else:
            idle.append(u)
    if assigned:
        repulsion_all = np.sum(cfg.capacity_kwh / (1.0 + distance[:, np.asarray(assigned, dtype=int)]) ** 2, axis=1)
    else:
        repulsion_all = np.zeros(len(current), dtype=float)
    for u in idle:
        origin = int(origins[u])
        feasible = np.flatnonzero(distance[origin] <= cfg.max_relocation_km + 1e-12)
        score = attraction_all[feasible] - 0.35 * repulsion_all[feasible] - cfg.relocation_cost_per_km * distance[origin, feasible]
        choice = int(np.argmax(score)); best = int(feasible[choice]); best_score = float(score[choice])
        if best_score <= 0.0:
            best = origin
        if best != origin and moves < move_budget:
            destinations[u] = best; moves += 1; assigned.append(best)
            repulsion_all += cfg.capacity_kwh / (1.0 + distance[:, best]) ** 2
    flow = np.zeros((len(current), len(current)), dtype=int)
    for i, j in zip(origins, destinations): flow[i, j] += 1
    return flow


def sparse_flow(
    current: np.ndarray,
    visible_age: np.ndarray,
    future: np.ndarray,
    distance: np.ndarray,
    cfg: DispatchConfig,
    risk_weight: float,
) -> np.ndarray:
    """Sparse marginal-task assignment used only as a scalable approximation."""
    origins = _unit_origins(current)
    m, n = len(origins), len(current)
    if m == 0:
        return np.zeros((n, n), dtype=int)
    tau = transit_slots(distance, cfg.speed_km_per_slot)
    # Candidate columns represent marginal service units at a node.  Global
    # high-value columns are augmented with local reachable columns for every
    # occupied origin, preventing a distant hotspot from crowding all feasible
    # alternatives out of the sparse assignment matrix.
    marginal_all: list[tuple[float, int, int]] = []
    effective_by_lag = np.zeros((cfg.horizon, n), dtype=float)
    max_lag = min(cfg.horizon - 1, cfg.response_slots - 1)
    for lag in range(max_lag + 1):
        surviving = visible_age[: max(0, cfg.response_slots - lag)].sum(axis=0)
        predicted = future[:, 1 : lag + 1].sum(axis=1) if lag else np.zeros((cfg.scenarios, n))
        scenario_value = predicted + surviving[None, :]
        effective_by_lag[lag] = scenario_value.mean(axis=0) + risk_weight * _cvar_upper(scenario_value, cfg.cvar_alpha)
    for node in range(n):
        effective = float(effective_by_lag[:, node].max())
        units = min(m, int(math.ceil(effective / cfg.capacity_kwh))) if effective > 1e-12 else 0
        for k in range(units):
            gain = min(cfg.capacity_kwh, max(0.0, effective - k * cfg.capacity_kwh))
            marginal_all.append((gain, node, k))
    if not marginal_all:
        return stay_flow(current)
    global_keep = sorted(marginal_all, key=lambda item: item[0], reverse=True)[: min(len(marginal_all), 4 * m)]
    selected = {(node, k): (gain, node, k) for gain, node, k in global_keep}
    for origin in np.unique(origins):
        feasible = np.flatnonzero(
            (distance[origin] <= cfg.max_relocation_km + 1e-12)
            & (tau[origin] <= cfg.horizon - 1)
        )
        local = sorted(
            ((effective_by_lag[int(tau[origin, node]), node] - cfg.relocation_cost_per_km * distance[origin, node], int(node)) for node in feasible),
            reverse=True,
        )[:3]
        for value, node in local:
            if value > 1e-12:
                selected.setdefault((node, 0), (float(effective_by_lag[int(tau[origin, node]), node]), node, 0))
    marginal = list(selected.values())
    score = np.full((m, len(marginal) + m), -1e12, dtype=float)
    marginal_nodes = np.asarray([item[1] for item in marginal], dtype=int)
    marginal_units = np.asarray([item[2] for item in marginal], dtype=float)
    lags = tau[np.ix_(origins, marginal_nodes)]
    distances = distance[np.ix_(origins, marginal_nodes)]
    feasible = (distances <= cfg.max_relocation_km + 1e-12) & (lags <= cfg.horizon - 1)
    safe_lags = np.minimum(lags, cfg.horizon - 1)
    effective = effective_by_lag[safe_lags, marginal_nodes[None, :]]
    gains = np.minimum(cfg.capacity_kwh, np.maximum(0.0, effective - marginal_units[None, :] * cfg.capacity_kwh))
    score[:, : len(marginal)] = np.where(feasible, gains - cfg.relocation_cost_per_km * distances, -1e12)
    score[np.arange(m), len(marginal) + np.arange(m)] = 0.0
    rows, columns = linear_sum_assignment(score, maximize=True)
    destinations = origins.copy(); gains = np.zeros(m, dtype=float)
    for u, c in zip(rows, columns):
        if c < len(marginal) and score[u, c] > 0:
            destinations[u] = marginal[c][1]; gains[u] = score[u, c]
    movers = np.flatnonzero(destinations != origins)
    move_budget = int(math.floor(cfg.max_relocation_fraction * cfg.fleet))
    if len(movers) > move_budget:
        cancel = movers[np.argsort(-gains[movers])[move_budget:]]
        destinations[cancel] = origins[cancel]
    flow = np.zeros((n, n), dtype=int)
    for i, j in zip(origins, destinations): flow[i, j] += 1
    return flow


def edvf_sparse_flow(
    current: np.ndarray,
    visible_age: np.ndarray,
    transit: list[np.ndarray],
    distance: np.ndarray,
    cfg: DispatchConfig,
    candidate_retention: int | None = None,
) -> np.ndarray:
    """Candidate-pruned deterministic EDVF dispatch for large networks.

    This is the scalable counterpart of :func:`edvf_flow_adapter`.  It uses
    exactly the same information set and the same three lexicographic criteria:

      1. reduce irreversible deadline loss
         ``sum_j [max_d(D_j(d)-S_j(d))]_+``;
      2. among equal primary gains, increase exponentially discounted useful
         service;
      3. among equal first/second-stage gains, reduce relocation distance.

    The exact time-expanded MILP retains all feasible arcs.  Here, each
    occupied origin keeps only K destinations with the largest attainable
    current-backlog viability gain (plus its stay arc), after which vehicles
    are allocated by a global marginal-flow sweep with the same lexicographic
    comparison.  The deadline-capacity envelope is updated after every
    accepted vehicle, so both diminishing capacity and origin competition are
    represented explicitly.  Future scenarios and CVaR are deliberately
    absent.
    """
    current = np.asarray(current, dtype=int)
    age = np.asarray(visible_age, dtype=float)
    distance = np.asarray(distance, dtype=float)
    origins = _unit_origins(current)
    fleet, n = len(origins), len(current)
    if fleet == 0:
        return np.zeros((n, n), dtype=int)
    if age.shape != (cfg.response_slots, n):
        raise ValueError(f"Bad visible_age shape: {age.shape}")

    W = int(cfg.response_slots)
    C = float(cfg.capacity_kwh)
    K = max(1, int(cfg.edvf_candidate_retention if candidate_retention is None else candidate_retention))
    resolution = max(0.0, float(cfg.edvf_field_resolution_kwh))
    tau = transit_slots(distance, cfg.speed_km_per_slot)

    # D[j,d]: current RSS that must be completed by prefix d+1.
    D = np.zeros((n, W), dtype=float)
    for a in range(W):
        deadline = W - a
        D[:, deadline - 1 :] += age[a, :, None]

    # S[j,d]: one service slot already committed by vehicles currently in
    # transit.  Sparse EDVF is a receding-horizon one-relocation approximation:
    # after that first service slot the next rolling decision reallocates the
    # vehicle, so later capacity is not counted in the present sparse arc.
    S = np.zeros((n, W), dtype=float)
    for q, arrivals in enumerate(transit):
        arrival_prefix = q + 1
        if arrival_prefix >= W:
            break
        counts = np.asarray(arrivals, dtype=float)
        if not np.any(counts):
            continue
        for d in range(arrival_prefix + 1, W + 1):
            S[:, d - 1] += C * counts

    weights = np.exp(-np.arange(W, dtype=float) / W)

    prefixes = np.arange(1, W + 1, dtype=float)

    def marginal_many(nodes: np.ndarray, travel_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized (primary loss reduction, secondary service gain)."""
        nodes = np.asarray(nodes, dtype=int)
        travel_times = np.asarray(travel_times, dtype=int)
        if len(nodes) == 0:
            return np.empty(0), np.empty(0)
        profile = C * (prefixes[None, :] > travel_times[:, None])
        before = D[nodes] - S[nodes]
        loss_before = np.maximum(0.0, before.max(axis=1))
        loss_after = np.maximum(0.0, (before - profile).max(axis=1))
        primary = np.maximum(0.0, loss_before - loss_after)
        useful_before = np.minimum(D[nodes], S[nodes])
        useful_after = np.minimum(D[nodes], S[nodes] + profile)
        secondary = np.sum(
            weights[None, :] * np.maximum(0.0, useful_after - useful_before),
            axis=1,
        )
        return primary, secondary

    def rank_gain(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if resolution <= 1e-12:
            return values
        return np.floor(values / resolution + 1e-10) * resolution

    # Candidate pruning is performed once from the current state.  A compact
    # global urgency pool is complemented by the nearest active nodes of each
    # occupied origin.  This avoids scanning all n destinations for every
    # vehicle while retaining both severe hotspots and spatially local tasks.
    active = np.flatnonzero(D[:, -1] > 1e-12)
    base_short = D - S
    base_loss = np.maximum(0.0, base_short.max(axis=1))
    base_profile = np.full(W, C, dtype=float)
    base_secondary = np.sum(
        weights[None, :]
        * np.maximum(
            0.0,
            np.minimum(D, S + base_profile[None, :]) - np.minimum(D, S),
        ),
        axis=1,
    )
    rank_base_loss = rank_gain(base_loss)
    rank_base_secondary = rank_gain(base_secondary)
    global_count = min(len(active), max(4 * K, 24))
    if global_count:
        order = np.lexsort((-D[active, -1], -rank_base_secondary[active], -rank_base_loss[active]))
        global_nodes = active[order[:global_count]]
    else:
        global_nodes = np.empty(0, dtype=int)

    # Keeping the stay arc guarantees feasibility even when no reachable task
    # has value.
    candidates: dict[int, np.ndarray] = {}
    for origin in np.unique(origins):
        if len(active) > K:
            local_pos = np.argpartition(distance[origin, active], K - 1)[:K]
            local_nodes = active[local_pos]
        else:
            local_nodes = active
        pool = np.unique(np.concatenate((global_nodes, local_nodes, np.asarray([origin], dtype=int))))
        feasible = pool[
            (distance[origin, pool] <= cfg.max_relocation_km + 1e-12)
            & (tau[origin, pool] <= W - 1)
        ]
        p_gain, s_gain = marginal_many(feasible, tau[origin, feasible])
        p_rank, s_rank = rank_gain(p_gain), rank_gain(s_gain)
        scored = [
            (float(p), float(s), -float(distance[origin, node]), int(node))
            for node, p, s in zip(feasible, p_rank, s_rank)
        ]
        scored.sort(reverse=True)
        kept = [node for p, s, _, node in scored if p > 1e-12 or s > 1e-12][:K]
        if int(origin) not in kept:
            kept.append(int(origin))
        candidates[int(origin)] = np.asarray(kept, dtype=int)

    # Build at most K sparse arcs for every occupied origin.  Source counts
    # stay aggregated, so the working set is O(number of occupied origins*K),
    # rather than a fleet-by-fleet assignment matrix.
    source_nodes = np.asarray(sorted(candidates), dtype=int)
    source_remaining = current[source_nodes].astype(int).copy()
    arc_source: list[int] = []
    arc_destination: list[int] = []
    arc_travel: list[int] = []
    arc_distance: list[float] = []
    for source_index, origin in enumerate(source_nodes):
        for destination in candidates[int(origin)]:
            arc_source.append(source_index)
            arc_destination.append(int(destination))
            arc_travel.append(int(tau[origin, destination]))
            arc_distance.append(float(distance[origin, destination]))
    if not arc_source:
        return stay_flow(current)

    arc_source_a = np.asarray(arc_source, dtype=int)
    arc_destination_a = np.asarray(arc_destination, dtype=int)
    arc_travel_a = np.asarray(arc_travel, dtype=int)
    arc_distance_a = np.asarray(arc_distance, dtype=float)
    arc_profiles = C * (prefixes[None, :] > arc_travel_a[:, None])

    # The allocated envelope is updated after every choice.  A lazy heap keeps
    # the best feasible tuple (primary, secondary, -distance) over all sources.
    # Only arcs incident to a destination whose envelope changed are rescored;
    # stale heap entries are detected by the destination version counter.
    allocated = np.zeros((n, W), dtype=float)
    flow = np.zeros((n, n), dtype=int)
    mover_budget = int(math.floor(cfg.max_relocation_fraction * cfg.fleet))
    movers = 0
    destination_version = np.zeros(n, dtype=int)

    def arc_gain(arc_id: int) -> tuple[float, float]:
        destination = int(arc_destination_a[arc_id])
        prior = S[destination] + allocated[destination]
        before = D[destination] - prior
        profile = arc_profiles[arc_id]
        loss_before = max(0.0, float(before.max()))
        loss_after = max(0.0, float((before - profile).max()))
        primary = float(rank_gain(np.asarray([max(0.0, loss_before - loss_after)]))[0])
        useful_before = np.minimum(D[destination], prior)
        useful_after = np.minimum(D[destination], prior + profile)
        secondary = float(rank_gain(np.asarray([
            np.dot(weights, np.maximum(0.0, useful_after - useful_before))
        ]))[0])
        return primary, secondary

    heap: list[tuple[float, float, float, int, int]] = []
    for arc_id in range(len(arc_source_a)):
        primary, secondary = arc_gain(arc_id)
        if primary > 1e-12 or secondary > 1e-12:
            destination = int(arc_destination_a[arc_id])
            heapq.heappush(heap, (
                -primary, -secondary, float(arc_distance_a[arc_id]),
                int(arc_id), int(destination_version[destination]),
            ))

    for _ in range(fleet):
        chosen = -1
        while heap:
            _, _, _, arc_id, version = heapq.heappop(heap)
            source_index = int(arc_source_a[arc_id])
            if source_remaining[source_index] <= 0:
                continue
            origin = int(source_nodes[source_index])
            destination = int(arc_destination_a[arc_id])
            if movers >= mover_budget and origin != destination:
                continue
            current_version = int(destination_version[destination])
            if version != current_version:
                primary, secondary = arc_gain(arc_id)
                if primary > 1e-12 or secondary > 1e-12:
                    heapq.heappush(heap, (
                        -primary, -secondary, float(arc_distance_a[arc_id]),
                        int(arc_id), current_version,
                    ))
                continue
            chosen = int(arc_id)
            break
        if chosen < 0:
            break

        source_index = int(arc_source_a[chosen])
        origin = int(source_nodes[source_index])
        destination = int(arc_destination_a[chosen])
        flow[origin, destination] += 1
        source_remaining[source_index] -= 1
        allocated[destination] += arc_profiles[chosen]
        destination_version[destination] += 1
        movers += int(origin != destination)

        # The same source-destination arc may remain useful for another unit.
        # Push its current marginal value immediately; all other arcs to this
        # destination are lazily refreshed when they reach the heap top.
        if source_remaining[source_index] > 0:
            primary, secondary = arc_gain(chosen)
            if primary > 1e-12 or secondary > 1e-12:
                heapq.heappush(heap, (
                    -primary, -secondary, float(arc_distance_a[chosen]),
                    chosen, int(destination_version[destination]),
                ))

    # Zero-gain vehicles stay at their origin, which is the third-stage
    # movement minimum for a primary/secondary tie.
    for source_index, origin in enumerate(source_nodes):
        if source_remaining[source_index] > 0:
            flow[int(origin), int(origin)] += int(source_remaining[source_index])
    return flow


def softdbf_flow(
    current: np.ndarray,
    visible_age: np.ndarray,
    distance: np.ndarray,
    cfg: DispatchConfig,
) -> np.ndarray:
    """SoftDBF joint fleet-placement policy (backup 2026-08-16).

    At each decision epoch, solve the joint fleet-placement problem by
    minimizing the normalized soft shortfall free energy over deadline demand
    bounds, with lexicographic tie-breaking on immediate movement.  Distance
    enters only through travel time.  This adapter converts the engine's
    age-structured visible RSS into RSSTask / VehicleState and maps the joint
    destinations back to a flow matrix, executed by the shared executor.
    """
    try:
        import softdbf_dispatch as sdb
    except ImportError:
        raise ValueError("softdbf_dispatch.py is required for policy SoftDBF-RSS")
    n = len(current)
    tasks: list[sdb.RSSTask] = []
    # Age index a => deadline = response_slots - a (a=0 newest, a=w-1 about to expire).
    for a in range(cfg.response_slots):
        for j in range(n):
            rem = float(visible_age[a, j])
            if rem > 1e-12:
                tasks.append(sdb.RSSTask(
                    station=j, remaining_kwh=rem,
                    deadline_slots=cfg.response_slots - a,
                ))
    vehicles: list[sdb.VehicleState] = []
    for j in range(n):
        for _ in range(int(current[j])):
            vehicles.append(sdb.VehicleState(node=j))
    if not vehicles:
        return np.zeros((n, n), dtype=int)
    res = sdb.solve_softdbf_joint(
        tasks=tasks,
        vehicles=vehicles,
        distance_km=np.asarray(distance, dtype=float),
        capacity_kwh_per_slot=cfg.capacity_kwh,
        response_slots=cfg.response_slots,
        speed_km_per_slot=cfg.speed_km_per_slot,
        max_move_km=cfg.max_relocation_km,
        max_relocated_share=cfg.max_relocation_fraction,
        beta=5.0,
    )
    flow = np.zeros((n, n), dtype=int)
    k = 0
    for j in range(n):
        for _ in range(int(current[j])):
            flow[j, int(res.destinations[k])] += 1
            k += 1
    return flow


def softdbf_greedy_flow(
    current: np.ndarray,
    visible_age: np.ndarray,
    distance: np.ndarray,
    cfg: DispatchConfig,
    forecast_future: np.ndarray | None = None,
    committed_arrivals: np.ndarray | None = None,
) -> np.ndarray:
    """SoftDBF-G: closed-form greedy potential balance (zero hand-tuned params).

    Deadline-aware version.  For each station j the deadline demand bound
        D_j(d) = cumulative RSS due by prefix d   (d = 1..W)
    is built from the visible age structure (a task at age a is due by prefix
    d = W-a) plus the climatology forecast (an arrival at slot t+h is due by
    prefix W-h).  Service already committed by in-transit vehicles digests D
    first (arrival after tau slots provides max(0, d-tau) slots at each prefix).

    The greedy repeatedly picks the (idle vehicle i, station j) pair with the
    largest *shortfall reduction*
        gain_ij = sum_d [ max(0, D_j(d)-S_j(d)) - max(0, D_j(d)-S_j(d)-C*max(0,d-tau_ij)) ]
                - gamma * d_ij
    assigns it, and updates S_j.  This is the greedy analogue of SoftDBF's
    exact joint enumeration; the only free coefficient is the movement cost
    gamma (fitted one-shot on 1-2 training days).  All structural quantities
    (age weights, deadline structure, forecast decay) are derived, not tuned.
    """
    n = len(current)
    W = cfg.response_slots
    C = cfg.capacity_kwh
    v = cfg.speed_km_per_slot
    gamma = cfg.relocation_cost_per_km

    # ---- deadline demand bound per station: D[j, d] (cumulative kWh due by prefix d)
    D = np.zeros((n, W), dtype=float)
    for j in range(n):
        for a in range(W):
            q = float(visible_age[a, j])
            if q > 1e-12:
                d = W - a  # due by prefix d
                D[j, d - 1:] += q
    if forecast_future is not None:
        for r in range(min(len(forecast_future), W - 1)):
            d = W - 1 - r  # forecast arrival at t+r+1 due by prefix W-(r+1)
            if d >= 1:
                D[:, d - 1:] += np.asarray(forecast_future[r], dtype=float)[:, None]

    # ---- service already committed in transit: S[j, d]
    S = np.zeros((n, W), dtype=float)
    if committed_arrivals is not None:
        for r in range(min(len(committed_arrivals), W - 1)):
            count = np.asarray(committed_arrivals[r], dtype=float)
            for d in range(1, W + 1):
                avail = C * max(0, d - (r + 1))
                if avail > 0:
                    S[:, d - 1] += avail * count

    shortfall = np.maximum(D - S, 0.0)  # [n, W]
    total_short = float(shortfall.sum())

    origins = _unit_origins(current)
    F = len(origins)
    if F == 0:
        return np.zeros((n, n), dtype=int)
    max_moves = max(1, int(math.floor(cfg.max_relocation_fraction * F)))
    tau = transit_slots(np.asarray(distance, dtype=float), v)

    # Deadline-urgency weights for prefix d: (W-d+1)/W (d=1 most urgent -> 1.0).
    # Structural constant, same schedule the MILP terminal loss uses.
    urgency = np.array([(W - d + 1.0) / W for d in range(1, W + 1)])

    dest = origins.copy()
    assigned = np.zeros(F, dtype=bool)
    moved = 0

    while total_short > 1e-9 and moved < max_moves and not assigned.all():
        best_i = -1
        best_j = -1
        best_gain = -np.inf
        for i in range(F):
            if assigned[i]:
                continue
            o = int(origins[i])
            for j in range(n):
                t_ij = int(tau[o, j])
                if t_ij > W - 1:
                    continue
                if float(distance[o, j]) > cfg.max_relocation_km + 1e-12:
                    continue
                # shortfall reduction at each prefix if this vehicle goes to j
                red = 0.0
                for d in range(1, W + 1):
                    avail = C * max(0, d - t_ij)
                    if avail <= 0:
                        continue
                    s_cur = float(shortfall[j, d - 1])
                    if s_cur > 0:
                        red += min(s_cur, avail) * float(urgency[d - 1])
                g = red - gamma * float(distance[o, j])
                if g > best_gain:
                    best_gain = g
                    best_i = i
                    best_j = j
        if best_i < 0 or best_gain <= 0.0:
            break
        o = int(origins[best_i])
        assigned[best_i] = True
        dest[best_i] = best_j
        if o != best_j:
            moved += 1
        # digest shortfall at best_j by this vehicle's service profile
        t_ij = int(tau[o, best_j])
        for d in range(1, W + 1):
            avail = C * max(0, d - t_ij)
            if avail > 0:
                shortfall[best_j, d - 1] = max(0.0, shortfall[best_j, d - 1] - avail)
        total_short = float(shortfall.sum())

    flow = np.zeros((n, n), dtype=int)
    for i, j in zip(origins, dest):
        flow[i, j] += 1
    return flow


def edvf_flow_adapter(
    current: np.ndarray,
    visible_age: np.ndarray,
    transit: list[np.ndarray],
    distance: np.ndarray,
    cfg: DispatchConfig,
) -> np.ndarray:
    """EDVF-RSS: Exponential Deadline-Viability Flow (backup 2026-08-16).

    Deterministic no-future lexicographic solver:
        1) min irreversible deadline loss L(f) = [max_d (D_u(d)-S_u(d))]_+
        2) max exponentially discounted recovery integral A(f), time constant W
        3) min total movement M(f)
    Returns the decoded first-step flow for the shared executor.
    """
    try:
        from . import edvf_dispatch as edvf
    except ImportError:
        raise ValueError("edvf_dispatch.py is required for policy EDVF-RSS")
    result = edvf.edvf_flow(
        current=current,
        visible_age=visible_age,
        transit=transit,
        distance_km=np.asarray(distance, dtype=float),
        cfg=cfg,
    )
    if not result.feasible:
        # fall back to stay-in-place if the solver fails
        flow = np.zeros((len(current), len(current)), dtype=int)
        np.fill_diagonal(flow, current)
        return flow
    return np.asarray(result.first_flow, dtype=int)


def scheduled_arrivals_from_queue(transit: list[np.ndarray], horizon: int, n: int) -> np.ndarray:
    result = np.zeros((horizon, n), dtype=int)
    for h in range(1, horizon):
        if h - 1 < len(transit):
            result[h] = transit[h - 1]
    return result


def execute_flow(
    flow: np.ndarray,
    current: np.ndarray,
    transit: list[np.ndarray],
    distance: np.ndarray,
    cfg: DispatchConfig,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    # Defensive snapshot: decouple the flow buffer from any caller-held array so
    # np.nonzero below scans a private copy (guards against a spurious
    # "number of non-zero array elements changed during function execution"
    # race under heavy multi-process + multi-threaded BLAS oversubscription).
    flow = np.array(flow, dtype=int, copy=True)
    if not np.array_equal(flow.sum(axis=1), current):
        raise AssertionError("Flow does not conserve stationary vehicles")
    tau = transit_slots(distance, cfg.speed_km_per_slot)
    stationary = np.zeros_like(current)
    available = np.zeros(len(current), dtype=float)
    movement = 0.0; movers = 0
    for i, j in zip(*np.nonzero(flow)):
        count = int(flow[i, j])
        if i == j:
            stationary[j] += count; available[j] += count * cfg.capacity_kwh
        else:
            if distance[i, j] > cfg.max_relocation_km + 1e-12:
                raise AssertionError("Infeasible relocation distance")
            transit[int(tau[i, j]) - 1][j] += count
            movement += count * float(distance[i, j]); movers += count
    if movers > int(math.floor(cfg.max_relocation_fraction * cfg.fleet)):
        raise AssertionError("Move budget violated")
    return stationary, available, movement, movers


@dataclass
class DayResult:
    total_rss_kwh: float
    recovered_rss_kwh: float
    residual_rss_kwh: float
    coverage: float
    movement_km: float
    runtime_seconds: float
    feasible_decisions: int
    solver_success: int
    optimal_within_tolerance: int
    decisions: int
    served_age0_kwh: float
    served_backlog_kwh: float
    expired_kwh: float
    move_decisions: int
    stay_decisions: int
    intransit_unit_slots: int
    max_conservation_error: float
    move_1slot_units: int
    move_2slot_units: int
    full_service_units: int
    partial_service_units: int
    idle_units: int
    service_energy_kwh: float


def simulate_independent_day(
    actual_rss: np.ndarray,
    signal_cube: np.ndarray,
    history_signal: np.ndarray,
    forecast_indices: np.ndarray,
    distance: np.ndarray,
    cfg: DispatchConfig,
    policy: str,
    include_backlog: bool = True,
    include_future: bool = True,
    exact_model: ExactRollingMILP | None = None,
    perfect_future: bool = False,
    forecast_cube: np.ndarray | None = None,
) -> DayResult:
    """Replay one natural day plus W-1 tail slots from a common depot (node 0).

    ``perfect_future`` (P0-3 diagnostic): feed the true next-slot RSS to the
    planner instead of any sampled forecast.  ``forecast_cube`` (P0-2): an
    optional pre-rendered scenario cube of shape (scenarios, slots_per_day, n)
    used instead of ``history_signal[forecast_indices]``; for a natural day the
    full 48-slot profile lets the planner see every forecast-slot arrival.
    """
    policy = canonicalize_policy(policy)
    n = actual_rss.shape[1]
    actual_age = np.zeros((cfg.response_slots, n), dtype=float)
    signal_age = np.zeros_like(actual_age)
    current = np.zeros(n, dtype=int); current[0] = cfg.fleet
    max_tau = int(transit_slots(distance, cfg.speed_km_per_slot).max())
    transit = [np.zeros(n, dtype=int) for _ in range(max(1, max_tau))]
    exact = exact_model
    if policy in {"Ours-Exact", "Myopic", "Demand-only", "Aggregate-event", "Scenario-Sampling-RSS"}:
        local_cfg = cfg if policy != "Scenario-Sampling-RSS" else DispatchConfig(**{**cfg.__dict__, "cvar_weight": 0.0})
        if exact is None:
            exact = ExactRollingMILP(distance, local_cfg)

    total = recovered = expired = movement = runtime = 0.0
    served_age0 = served_backlog = 0.0
    feasible = solver_success = optimal_within_tol = decisions = moves = stays = intransit_slots = 0
    move_1slot = move_2slot = full_service = partial_service = idle = 0
    service_energy = 0.0
    tau_matrix = transit_slots(distance, cfg.speed_km_per_slot)
    tail_end = cfg.slots_per_day + cfg.response_slots - 1
    for slot in range(tail_end):
        current = current + transit[0]
        transit = transit[1:] + [np.zeros(n, dtype=int)]
        arrivals = actual_rss[slot] if slot < cfg.slots_per_day else np.zeros(n)
        signal_arrivals = signal_cube[slot] if slot < cfg.slots_per_day else np.zeros(n)
        expired += shift_age(actual_age, arrivals)
        shift_age(signal_age, signal_arrivals)
        total += float(arrivals.sum())
        visible = signal_age.copy() if include_backlog else np.vstack([signal_age[0], np.zeros_like(signal_age[1:])])
        future = np.zeros((cfg.scenarios, cfg.horizon, n), dtype=float)
        if include_future:
            for h in range(1, cfg.horizon):
                source = slot + h
                if source < cfg.slots_per_day:
                    if perfect_future:
                        future[:, h] = actual_rss[source]
                    elif forecast_cube is not None:
                        future[:, h] = forecast_cube[:, source]
                    else:
                        future[:, h] = history_signal[forecast_indices, source]
        scheduled = scheduled_arrivals_from_queue(transit, cfg.horizon, n)
        started = time.perf_counter()
        solved: SolveResult | None = None
        if exact is not None:
            solved = exact.solve(visible, future, current, scheduled)
            flow = solved.first_flow
        elif policy == "EDF-Reactive":
            flow = edf_reactive_flow(current, visible, distance, cfg)
        elif policy == "NJN-RSS":
            flow = njn_flow(current, visible, distance, cfg)
        elif policy == "Modified-Savings-RSS":
            flow = modified_savings_flow(current, visible, distance, cfg)
        elif policy == "PS-IMCS-RSS":
            flow = ps_imcs_flow(current, visible, future, distance, cfg)
        elif policy == "SoftDBF-RSS":
            flow = softdbf_flow(current, visible, distance, cfg)
        elif policy == "SoftDBF-G":
            ff = None
            if forecast_cube is not None:
                rows = []
                for h in range(1, cfg.response_slots):
                    src = slot + h
                    if src < cfg.slots_per_day:
                        rows.append(forecast_cube[:, src].mean(axis=0))
                    else:
                        rows.append(np.zeros(n))
                ff = np.stack(rows) if rows else np.zeros((0, n))
            # committed arrivals: vehicles already in transit, arriving after
            # r+1 slots (transit[r] = arrivals in r+1 slots)
            committed = None
            if len(transit):
                rows = []
                for r in range(min(len(transit), cfg.response_slots - 1)):
                    rows.append(np.asarray(transit[r], dtype=float))
                committed = np.stack(rows) if rows else np.zeros((0, n))
            flow = softdbf_greedy_flow(current, visible, distance, cfg, ff, committed)
        elif policy in {"EDVF-RSS", "EDVF-LossOnly", "EDVF-LossEarly"}:
            flow = edvf_flow_adapter(current, visible, transit, distance, cfg)
        elif policy == "EDVF-RSS-Sparse":
            flow = edvf_sparse_flow(current, visible, transit, distance, cfg)
        elif policy == "Ours-Sparse":
            flow = sparse_flow(current, visible, future, distance, cfg, cfg.cvar_weight)
        elif policy == "Scenario-Sampling-Sparse":
            flow = sparse_flow(current, visible, future, distance, cfg, 0.0)
        else:
            raise ValueError(policy)
        elapsed = solved.runtime_seconds if solved is not None else time.perf_counter() - started
        runtime += elapsed; decisions += 1
        if solved is not None:
            feasible += int(solved.feasible)
            solver_success += int(solved.optimal)
            gap = solved.mip_gap
            gap_ok = (not math.isnan(gap)) and gap <= cfg.solver_mip_gap + 1e-9
            optimal_within_tol += int(solved.optimal and gap_ok)
        else:
            # Heuristic policies: always execute (feasible by construction), but
            # "solver success / optimality" is not meaningful for them.
            feasible += 1; solver_success += 0; optimal_within_tol += 0
        off_diagonal = flow.copy()
        np.fill_diagonal(off_diagonal, 0)
        move_1slot += int(off_diagonal[tau_matrix == 1].sum())
        move_2slot += int(off_diagonal[tau_matrix >= 2].sum())
        current, available, moved_km, moved_units = execute_flow(flow, current, transit, distance, cfg)
        movement += moved_km; moves += moved_units; stays += int(np.trace(flow))
        intransit_slots += int(sum(x.sum() for x in transit))
        served_station = np.minimum(actual_age.sum(axis=0), available)
        full_now = np.floor(served_station / cfg.capacity_kwh + 1e-12).astype(int)
        partial_now = ((served_station - full_now * cfg.capacity_kwh) > 1e-9).astype(int)
        full_service += int(full_now.sum())
        partial_service += int(partial_now.sum())
        idle += int(np.maximum(0, current - full_now - partial_now).sum())
        service_energy += float(served_station.sum())
        useful, by_age = serve_oldest(actual_age, available)
        serve_oldest(signal_age, available)
        recovered += useful; served_age0 += by_age[0]; served_backlog += float(by_age[1:].sum())
        if int(current.sum() + sum(x.sum() for x in transit)) != cfg.fleet:
            raise AssertionError("Fleet conservation failed")
    residual = expired + float(actual_age.sum())
    error = abs(total - recovered - residual)
    if error > max(1e-6, 1e-8 * max(total, 1.0)):
        raise AssertionError(f"RSS conservation failed: {total=} {recovered=} {residual=}")
    return DayResult(
        total, recovered, residual, recovered / total if total > 0 else 1.0,
        movement, runtime, feasible, solver_success, optimal_within_tol, decisions,
        served_age0, served_backlog, expired, moves, stays, intransit_slots, error,
        move_1slot, move_2slot, full_service, partial_service, idle, service_energy,
    )


def permute_station_slot_cells(
    train_rss: np.ndarray,
    proportion: float,
    seed: int,
    station_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Permute selected station-slot cells within each scenario day.

    ``station_indices`` allows structural non-task nodes (for example a depot)
    to remain untouched.  The default includes every final-axis position.
    """
    source = np.asarray(train_rss, dtype=float)
    if proportion <= 0:
        return source.copy()
    rng = np.random.default_rng(seed)
    result = source.copy()
    stations = np.arange(source.shape[2], dtype=int) if station_indices is None else np.asarray(station_indices, dtype=int)
    if stations.ndim != 1 or len(stations) == 0 or np.any(stations < 0) or np.any(stations >= source.shape[2]):
        raise ValueError("station_indices must contain valid final-axis indices")
    cells = source.shape[1] * len(stations)
    selected_count = int(round(proportion * cells))
    for day in range(source.shape[0]):
        eligible = result[day][:, stations].reshape(-1).copy()
        selected = rng.choice(cells, size=selected_count, replace=False)
        eligible[selected] = eligible[selected][rng.permutation(selected_count)]
        result[day][:, stations] = eligible.reshape(source.shape[1], len(stations))
    return result
