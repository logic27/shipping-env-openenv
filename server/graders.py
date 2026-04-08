"""Deterministic graders for shipping_env task submissions."""

from __future__ import annotations

from typing import Any, Dict, Iterable

try:
    from ..models import ShippingAction
except ImportError:
    from models import ShippingAction


MIN_SCORE = 0.01
MAX_TASK_SCORE = 0.95
INTERACTION_REWARD = 0.001


def _clamp_open_unit(value: float) -> float:
    return max(MIN_SCORE, min(MAX_TASK_SCORE, round(value, 3)))


def interaction_reward() -> float:
    """Small positive shaping reward that keeps cumulative task reward sub-unitary."""

    return INTERACTION_REWARD


def evidence_score(evidence_types: Iterable[str]) -> float:
    required = {
        "inspect_vessel",
        "inspect_congestion_history",
        "inspect_forecast",
        "inspect_route_options",
    }
    seen = set(evidence_types)
    if required.issubset(seen):
        return 0.10
    if len(required.intersection(seen)) >= 2:
        return 0.05
    return MIN_SCORE


def forecast_model_score(action: ShippingAction, optimal_plan: Dict[str, Any]) -> float:
    return 0.22 if action.forecast_model == optimal_plan["forecast_model"] else MIN_SCORE


def target_port_score(
    action: ShippingAction,
    optimal_plan: Dict[str, Any],
    candidate_ports: Iterable[str],
) -> float:
    if action.target_port_id == optimal_plan["target_port_id"]:
        return 0.43
    if action.target_port_id in set(candidate_ports):
        return 0.15
    return MIN_SCORE


def service_speed_score(action: ShippingAction, optimal_plan: Dict[str, Any]) -> float:
    return 0.20 if action.service_speed_knots == optimal_plan["service_speed_knots"] else MIN_SCORE


def task_grader(
    action: ShippingAction,
    optimal_plan: Dict[str, Any],
    candidate_ports: Iterable[str],
    evidence_types: Iterable[str],
) -> Dict[str, float]:
    """Return final task score plus its normalized deterministic breakdown."""

    breakdown = {
        "forecast_model_score": forecast_model_score(action, optimal_plan),
        "target_port_score": target_port_score(action, optimal_plan, candidate_ports),
        "service_speed_score": service_speed_score(action, optimal_plan),
        "evidence_score": evidence_score(evidence_types),
    }
    breakdown["task_score"] = _clamp_open_unit(sum(breakdown.values()))
    breakdown["score"] = breakdown["task_score"]
    return breakdown
