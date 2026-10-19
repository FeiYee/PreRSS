"""Paper-facing names for the proposed EDVF solvers."""

from __future__ import annotations

_ALIASES = {
    "EDVF-RSS-Exact": "EDVF-RSS",
    "Ours-Exact": "EDVF-RSS",  # legacy experiment key
    "Ours-Sparse": "EDVF-RSS-Sparse",
}

_PAPER = {
    "EDVF-RSS": "EDVF-RSS-Exact",
    "EDVF-RSS-Sparse": "EDVF-RSS-Sparse",
}


def canonicalize_policy(name: str) -> str:
    return _ALIASES.get(name, name)


def paper_name(name: str) -> str:
    return _PAPER.get(name, name)
