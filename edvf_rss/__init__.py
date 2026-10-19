"""RSS-driven Exponential Deadline-Viability Flow dispatch.

This package accompanies the *Vehicles* article by Zhi, Chen and Li (2026),
"Recoverable Service Shortfall-Driven Dispatch for Fixed-Mobile Charging
Coordination: Exponential Deadline-Viability Flow on Real Transaction Event
Streams."

The public package provides the proposed EDVF Exact and Sparse policies plus
the shared age-structured RSS executor.
"""

from .engine import (
    DayResult,
    DispatchConfig,
    execute_flow,
    edvf_sparse_flow,
    serve_oldest,
    shift_age,
    simulate_independent_day,
    transit_slots,
)
from .edvf_dispatch import EDVFResult, edvf_flow
from .names import canonicalize_policy, paper_name

__all__ = [
    "DayResult",
    "DispatchConfig",
    "EDVFResult",
    "canonicalize_policy",
    "edvf_flow",
    "edvf_sparse_flow",
    "execute_flow",
    "paper_name",
    "serve_oldest",
    "shift_age",
    "simulate_independent_day",
    "transit_slots",
]

__version__ = "0.2.0"
