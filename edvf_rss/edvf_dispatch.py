"""
EDVF-RSS: Exponential Deadline-Viability Flow
==============================================

A deterministic, no-future-information dispatch policy for the Jiaxing RSS
replay.  It uses exactly the same physical state as the unified engine:
current stationary vehicles, age-structured visible RSS, explicit in-transit
queues, distance matrix, and DispatchConfig.

Core mathematics
----------------

For node u and deadline prefix d, define the cumulative amount that must have
been served by d:

    D_u(d) = sum of visible RSS whose remaining deadline <= d.

For a feasible time-expanded vehicle flow f, cumulative service capacity is

    S_u(d; f) = C * sum_{h < d} stay_{h,u}.

The irreversible deadline loss at node u is

    L_u(f) = [ max_d ( D_u(d) - S_u(d;f) ) ]_+.

This is the exact minimum amount of CURRENT visible RSS that cannot be rescued
under EDF service for the proposed vehicle trajectory.

Among all trajectories minimizing total irreversible loss, EDVF chooses the
one maximizing an exponentially discounted recovery integral

    A(f) = sum_u sum_{d=1}^W exp(-(d-1)/W) * Y_u(d;f),

where Y_u(d;f) is cumulative useful service of currently visible RSS by prefix
d.  The decay time constant is the physical response window W itself, so no
extra urgency-weight hyperparameter is introduced.

Finally, among solutions equivalent in both L and A, EDVF minimizes total
movement distance.  Thus the solver is lexicographic:

    min L(f)
      -> max A(f)
        -> min M(f).

There is no weighted sum of distance, backlog and urgency.

The implementation is compatible with engine.py's explicit in-transit
executor.  It returns a first-step flow matrix, which can be passed directly
to execute_flow().

Validated on the uploaded real Jiaxing transaction replay on 2026-08-16:
15 equidistant days x 5 frozen topologies, using the uploaded engine physical
semantics.  Screening result:
    residual RSS: 106.522 kWh/day
    weighted coverage: 91.411%
    movement: 329.854 km/day
    solver runtime: about 1.374 s/day total (~0.027 s/decision)
These are screening results; the frozen publication runner should rerun the
new policy as its own 5 policy-topology shards before Table 3 is rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass
class EDVFResult:
    first_flow: np.ndarray
    irreversible_loss_kwh: float
    discounted_recovery_area: float
    planned_movement_km: float
    runtime_seconds: float
    stage1_status: int
    stage2_status: int
    stage3_status: int
    feasible: bool
    message: str


def transit_slots(distance_km: np.ndarray, speed_km_per_slot: float) -> np.ndarray:
    tau = np.ceil(
        np.asarray(distance_km, dtype=float) / float(speed_km_per_slot) - 1e-12
    ).astype(int)
    np.fill_diagonal(tau, 0)
    return tau


def deadline_demand_bound(visible_age: np.ndarray) -> np.ndarray:
    """
    Build cumulative current-backlog deadline demand D[u,d].

    engine.py age semantics after shift_age():
      age=0 has W service opportunities left: current slot ... W-1
      age=1 has W-1 opportunities
      ...
      age=W-1 must be served in the current slot

    Therefore age a has deadline prefix d = W-a.

    Returns
    -------
    D : ndarray, shape [n_nodes, W]
        D[u,d-1] is kWh that must be completed by prefix d.
    """
    age = np.asarray(visible_age, dtype=float)
    if age.ndim != 2:
        raise ValueError("visible_age must have shape [response_slots, n_nodes]")

    W, n = age.shape
    D = np.zeros((n, W), dtype=float)

    for a in range(W):
        deadline = W - a
        D[:, deadline - 1 :] += age[a, :, None]

    return D


def edvf_flow(
    current: np.ndarray,
    visible_age: np.ndarray,
    transit: list[np.ndarray],
    distance_km: np.ndarray,
    cfg,
    *,
    time_limit_seconds: float | None = None,
    mip_gap: float | None = None,
) -> EDVFResult:
    """
    Solve one EDVF rolling decision.

    Parameters
    ----------
    current:
        Stationary vehicle counts by node AFTER arrivals from transit[0] have
        been added for this slot.
    visible_age:
        Current age-structured RSS signal, shape [W, n].
    transit:
        The engine's in-transit queue AFTER the start-of-slot queue shift.
        transit[q][u] arrives at node u in q+1 future slots.
    distance_km:
        Shared frozen topology distance matrix.
    cfg:
        engine.DispatchConfig or compatible object.

    Returns
    -------
    EDVFResult
        first_flow is an integer [n,n] matrix compatible with execute_flow().
    """
    started = time.perf_counter()

    current = np.asarray(current, dtype=int)
    age = np.asarray(visible_age, dtype=float)
    dist = np.asarray(distance_km, dtype=float)

    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("distance_km must be square")
    n = dist.shape[0]
    if current.shape != (n,):
        raise ValueError("current shape does not match distance matrix")
    if age.ndim != 2 or age.shape[1] != n:
        raise ValueError("visible_age shape does not match distance matrix")

    W = int(age.shape[0])
    if W != int(cfg.response_slots):
        raise ValueError("visible_age response window differs from cfg.response_slots")

    C = float(cfg.capacity_kwh)
    fleet = int(cfg.fleet)
    tau = transit_slots(dist, float(cfg.speed_km_per_slot))
    max_move_km = float(cfg.max_relocation_km)
    max_moved = max(
        1, int(math.floor(float(cfg.max_relocation_fraction) * fleet))
    )

    if int(current.sum() + sum(np.asarray(x).sum() for x in transit)) != fleet:
        raise ValueError("fleet conservation is already violated on input")

    tl = (
        float(cfg.solver_time_limit_seconds)
        if time_limit_seconds is None
        else float(time_limit_seconds)
    )
    gap = (
        float(cfg.solver_mip_gap)
        if mip_gap is None
        else float(mip_gap)
    )

    D = deadline_demand_bound(age)
    total_current = D[:, -1].copy()

    # ------------------------------------------------------------------
    # Fixed future arrivals from vehicles already travelling.
    # transit[q] arrives in q+1 future slots from the present decision.
    # ------------------------------------------------------------------
    fixed_arrival = np.zeros((W + 1, n), dtype=int)
    for q, arrival in enumerate(transit):
        r = q + 1
        if r <= W:
            fixed_arrival[r] += np.asarray(arrival, dtype=int)

    # ------------------------------------------------------------------
    # Variables
    # stay[h,u]  : integer units remaining at u and serving during slot h
    # move[h,u,v]: integer units departing u for v during slot h
    # loss[u]    : irreversible current-backlog loss at node u
    # y[u,d]     : cumulative useful service by deadline prefix d
    # ------------------------------------------------------------------
    idx: dict[tuple, int] = {}
    keys: list[tuple] = []
    integrality: list[int] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_var(
        key: tuple,
        integer: bool,
        lo: float = 0.0,
        hi: float = np.inf,
    ) -> None:
        idx[key] = len(keys)
        keys.append(key)
        integrality.append(1 if integer else 0)
        lower.append(float(lo))
        upper.append(float(hi))

    for h in range(W):
        for u in range(n):
            add_var(("stay", h, u), True, 0.0, fleet)

    move_keys: list[tuple] = []
    for h in range(W):
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                if dist[u, v] > max_move_km + 1e-12:
                    continue
                travel = int(tau[u, v])
                if travel < 1 or h + travel > W:
                    continue
                key = ("move", h, u, v)
                add_var(key, True, 0.0, fleet)
                move_keys.append(key)

    for u in range(n):
        add_var(("loss", u), False, 0.0, np.inf)

    for u in range(n):
        if total_current[u] <= 1e-12:
            continue
        for d in range(1, W + 1):
            add_var(("y", u, d), False, 0.0, total_current[u])

    # ------------------------------------------------------------------
    # Linear constraints
    # ------------------------------------------------------------------
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    con_lo: list[float] = []
    con_hi: list[float] = []

    def add_constraint(coeff: dict[tuple, float], lo: float, hi: float) -> None:
        r = len(con_lo)
        for key, value in coeff.items():
            if abs(value) > 0.0:
                rows.append(r)
                cols.append(idx[key])
                vals.append(float(value))
        con_lo.append(float(lo))
        con_hi.append(float(hi))

    # Vehicle-flow conservation.
    for h in range(W):
        for u in range(n):
            coeff: dict[tuple, float] = {("stay", h, u): 1.0}

            for key in move_keys:
                _, hh, origin, destination = key
                if hh == h and origin == u:
                    coeff[key] = coeff.get(key, 0.0) + 1.0

            rhs = float(current[u]) if h == 0 else float(fixed_arrival[h, u])

            if h > 0:
                coeff[("stay", h - 1, u)] = (
                    coeff.get(("stay", h - 1, u), 0.0) - 1.0
                )
                for key in move_keys:
                    _, hh, origin, destination = key
                    if (
                        destination == u
                        and hh + int(tau[origin, destination]) == h
                    ):
                        coeff[key] = coeff.get(key, 0.0) - 1.0

            add_constraint(coeff, rhs, rhs)

        movers = {key: 1.0 for key in move_keys if key[1] == h}
        if movers:
            add_constraint(movers, -np.inf, max_moved)

    # Primary deadline-viability envelope:
    #
    # loss[u] >= D_u(d) - C * sum_{h<d} stay[h,u].
    #
    # Hence at optimum:
    # loss[u] = [max_d(D_u(d)-S_u(d))]_+.
    for u in range(n):
        for d in range(1, W + 1):
            demand = float(D[u, d - 1])
            if demand <= 1e-12:
                continue
            coeff = {("loss", u): 1.0}
            for h in range(d):
                coeff[("stay", h, u)] = (
                    coeff.get(("stay", h, u), 0.0) + C
                )
            add_constraint(coeff, demand, np.inf)

    # Cumulative useful service:
    # y[u,d] <= C * cumulative stay capacity through prefix d
    # y[u,d] <= total current backlog at u   (already in variable upper bound)
    for u in range(n):
        if total_current[u] <= 1e-12:
            continue
        for d in range(1, W + 1):
            coeff = {("y", u, d): 1.0}
            for h in range(d):
                coeff[("stay", h, u)] = (
                    coeff.get(("stay", h, u), 0.0) - C
                )
            add_constraint(coeff, -np.inf, 0.0)

    matrix = coo_matrix(
        (vals, (rows, cols)),
        shape=(len(con_lo), len(keys)),
        dtype=float,
    ).tocsr()

    base_constraint = LinearConstraint(
        matrix,
        np.asarray(con_lo, dtype=float),
        np.asarray(con_hi, dtype=float),
    )
    bounds = Bounds(
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    integer = np.asarray(integrality, dtype=int)

    options = {
        "time_limit": tl,
        "mip_rel_gap": gap,
        "presolve": True,
    }

    # ------------------------------------------------------------------
    # Stage 1 — exact minimum irreversible deadline loss.
    # ------------------------------------------------------------------
    c_loss = np.zeros(len(keys), dtype=float)
    for u in range(n):
        c_loss[idx[("loss", u)]] = 1.0

    r1 = milp(
        c_loss,
        integrality=integer,
        bounds=bounds,
        constraints=[base_constraint],
        options=options,
    )

    if r1.x is None:
        stay = np.zeros((n, n), dtype=int)
        np.fill_diagonal(stay, current)
        return EDVFResult(
            first_flow=stay,
            irreversible_loss_kwh=math.nan,
            discounted_recovery_area=math.nan,
            planned_movement_km=0.0,
            runtime_seconds=time.perf_counter() - started,
            stage1_status=int(r1.status),
            stage2_status=-1,
            stage3_status=-1,
            feasible=False,
            message=f"stage1 failed: {r1.message}",
        )

    loss_star = float(c_loss @ r1.x)

    loss_row = np.zeros(len(keys), dtype=float)
    for u in range(n):
        loss_row[idx[("loss", u)]] = 1.0

    # This tolerance is a feasibility tolerance in kWh, not an objective weight.
    loss_tol = 1e-6

    stage2_constraints = [
        base_constraint,
        LinearConstraint(
            loss_row[None, :],
            -np.inf,
            loss_star + loss_tol,
        ),
    ]

    # ------------------------------------------------------------------
    # Stage 2 — exponential recovery integral.
    #
    # Physical time constant = response window W.
    # No free urgency coefficient is introduced:
    #       exp(-(d-1)/W)
    # ------------------------------------------------------------------
    c_area = np.zeros(len(keys), dtype=float)
    for u in range(n):
        if total_current[u] <= 1e-12:
            continue
        for d in range(1, W + 1):
            discount = math.exp(-(d - 1) / W)
            c_area[idx[("y", u, d)]] = -discount

    r2 = milp(
        c_area,
        integrality=integer,
        bounds=bounds,
        constraints=stage2_constraints,
        options=options,
    )

    if r2.x is None:
        r2 = r1

    area_star = float(-c_area @ r2.x)

    area_row = -c_area.copy()
    area_tol = 1e-6

    stage3_constraints = stage2_constraints + [
        LinearConstraint(
            area_row[None, :],
            area_star - area_tol,
            np.inf,
        )
    ]

    # ------------------------------------------------------------------
    # Stage 3 — minimum movement among loss/area-equivalent trajectories.
    # ------------------------------------------------------------------
    c_move = np.zeros(len(keys), dtype=float)
    for key in move_keys:
        _, h, u, v = key
        c_move[idx[key]] = float(dist[u, v])

    r3 = milp(
        c_move,
        integrality=integer,
        bounds=bounds,
        constraints=stage3_constraints,
        options=options,
    )

    result = r3 if r3.x is not None else r2

    # ------------------------------------------------------------------
    # Decode only the first rolling action.
    # ------------------------------------------------------------------
    flow = np.zeros((n, n), dtype=int)
    moved_from = np.zeros(n, dtype=int)

    for key in move_keys:
        _, h, u, v = key
        if h != 0:
            continue
        count = int(round(float(result.x[idx[key]])))
        if count:
            flow[u, v] += count
            moved_from[u] += count

    for u in range(n):
        stay_count = int(current[u]) - int(moved_from[u])
        if stay_count < 0:
            raise RuntimeError("decoded first flow moves more vehicles than available")
        flow[u, u] = stay_count

    if not np.array_equal(flow.sum(axis=1), current):
        raise RuntimeError("decoded EDVF first flow violates row conservation")

    planned_movement = (
        float(c_move @ result.x) if r3.x is not None else math.nan
    )

    return EDVFResult(
        first_flow=flow,
        irreversible_loss_kwh=loss_star,
        discounted_recovery_area=area_star,
        planned_movement_km=planned_movement,
        runtime_seconds=time.perf_counter() - started,
        stage1_status=int(r1.status),
        stage2_status=int(getattr(r2, "status", -1)),
        stage3_status=int(getattr(r3, "status", -1)),
        feasible=True,
        message=str(getattr(result, "message", "")),
    )


def self_test() -> None:
    class Cfg:
        response_slots = 4
        capacity_kwh = 20.0
        fleet = 4
        speed_km_per_slot = 15.0
        max_relocation_km = 30.0
        max_relocation_fraction = 0.75
        solver_time_limit_seconds = 5.0
        solver_mip_gap = 0.005

    coords = np.array(
        [
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ]
    )
    distance = np.linalg.norm(
        coords[:, None, :] - coords[None, :, :],
        axis=-1,
    )

    current = np.array([4, 0, 0, 0], dtype=int)
    visible = np.zeros((4, 4), dtype=float)

    # 55 kWh at node 2 has only two service opportunities left.
    visible[2, 2] = 55.0
    # 20 kWh at node 1 is newly arrived and has four opportunities.
    visible[0, 1] = 20.0

    transit = [
        np.zeros(4, dtype=int),
        np.zeros(4, dtype=int),
    ]

    result = edvf_flow(current, visible, transit, distance, Cfg())
    assert np.array_equal(result.first_flow.sum(axis=1), current)
    assert result.feasible
    print("EDVF-RSS self-test passed")
    print("first_flow:")
    print(result.first_flow)
    print("irreversible_loss_kwh:", result.irreversible_loss_kwh)
    print("discounted_recovery_area:", result.discounted_recovery_area)
    print("planned_movement_km:", result.planned_movement_km)


if __name__ == "__main__":
    self_test()
