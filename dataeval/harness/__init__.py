"""Outage simulation, trajectory replay harness, and metric scoring against ground truth."""

from dataeval.harness.baseline import (
    DEFAULT_PRE_OUTAGE_WINDOW_S,
    EARTH_RADIUS_M,
    PreOutageState,
    extract_pre_outage_state,
    propagate_constant_velocity_heading,
)
from dataeval.harness.metrics import (
    MIN_TRAVEL_DISTANCE_M,
    OutageMetrics,
    compute_outage_metrics,
    decompose_along_cross_track,
    haversine_distance,
    wrap_heading_error,
)
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
    "DEFAULT_PRE_OUTAGE_WINDOW_S",
    "EARTH_RADIUS_M",
    "MIN_TRAVEL_DISTANCE_M",
    "OutageMetrics",
    "OutageSkip",
    "OutageWindow",
    "OutageWindowList",
    "PreOutageState",
    "apply_outage",
    "compute_outage_metrics",
    "decompose_along_cross_track",
    "extract_pre_outage_state",
    "find_real_gaps",
    "ground_truth_trajectory",
    "haversine_distance",
    "propagate_constant_velocity_heading",
    "select_outage_windows",
    "wrap_heading_error",
]
