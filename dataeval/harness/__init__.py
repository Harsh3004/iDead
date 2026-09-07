"""Outage simulation, trajectory replay harness, and metric scoring against ground truth."""

from dataeval.harness.outage import (
    DEFAULT_EDGE_MARGIN_S,
    DEFAULT_OUTAGE_DURATIONS,
    OutageSkip,
    OutageWindow,
    OutageWindowList,
    apply_outage,
    find_real_gaps,
    ground_truth_trajectory,
    select_outage_windows,
)

__all__ = [
    "DEFAULT_EDGE_MARGIN_S",
    "DEFAULT_OUTAGE_DURATIONS",
    "OutageSkip",
    "OutageWindow",
    "OutageWindowList",
    "apply_outage",
    "find_real_gaps",
    "ground_truth_trajectory",
    "select_outage_windows",
]
