# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Seeded maritime planning scenarios for the shipping environment."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


PORTS: Dict[str, Dict[str, Any]] = {
    "rotterdam": {
        "port_id": "rotterdam",
        "name": "Port of Rotterdam",
        "country": "Netherlands",
        "region": "North Europe",
        "congestion_index": 0.61,
        "available_berths": 4,
        "storage_pressure": "medium",
    },
    "antwerp": {
        "port_id": "antwerp",
        "name": "Port of Antwerp-Bruges",
        "country": "Belgium",
        "region": "North Europe",
        "congestion_index": 0.53,
        "available_berths": 5,
        "storage_pressure": "medium",
    },
    "shanghai": {
        "port_id": "shanghai",
        "name": "Port of Shanghai",
        "country": "China",
        "region": "East Asia",
        "congestion_index": 0.92,
        "available_berths": 1,
        "storage_pressure": "high",
    },
    "ningbo": {
        "port_id": "ningbo",
        "name": "Port of Ningbo-Zhoushan",
        "country": "China",
        "region": "East Asia",
        "congestion_index": 0.48,
        "available_berths": 6,
        "storage_pressure": "low",
    },
    "busan": {
        "port_id": "busan",
        "name": "Port of Busan",
        "country": "South Korea",
        "region": "East Asia",
        "congestion_index": 0.37,
        "available_berths": 7,
        "storage_pressure": "low",
    },
    "wilhelmshaven": {
        "port_id": "wilhelmshaven",
        "name": "JadeWeserPort Wilhelmshaven",
        "country": "Germany",
        "region": "North Europe",
        "congestion_index": 0.34,
        "available_berths": 6,
        "storage_pressure": "low",
    },
}


VESSELS: Dict[str, Dict[str, Any]] = {
    "maersk_bruges": {
        "vessel_id": "maersk_bruges",
        "name": "MAERSK BRUGES",
        "mmsi": 219021624,
        "vessel_type": "container",
        "cargo": "temperature-controlled pharma",
        "origin_region": "North Atlantic",
        "current_lat": 50.71,
        "current_lon": -4.86,
        "heading_deg": 78,
        "current_speed_knots": 13,
        "destination_port_id": "rotterdam",
    },
    "pacific_resolve": {
        "vessel_id": "pacific_resolve",
        "name": "PACIFIC RESOLVE",
        "mmsi": 563118420,
        "vessel_type": "container",
        "cargo": "consumer electronics",
        "origin_region": "South China Sea",
        "current_lat": 24.28,
        "current_lon": 121.17,
        "heading_deg": 316,
        "current_speed_knots": 14,
        "destination_port_id": "shanghai",
    },
    "atlas_sentinel": {
        "vessel_id": "atlas_sentinel",
        "name": "ATLAS SENTINEL",
        "mmsi": 636092014,
        "vessel_type": "tanker",
        "cargo": "refined products",
        "origin_region": "Arabian Sea",
        "current_lat": 16.49,
        "current_lon": 64.11,
        "heading_deg": 302,
        "current_speed_knots": 12,
        "destination_port_id": "rotterdam",
    },
}


TASKS: Dict[str, Dict[str, Any]] = {
    "easy_rotterdam_watch": {
        "task_id": "easy_rotterdam_watch",
        "difficulty": "easy",
        "title": "Protect a cold-chain arrival into Rotterdam",
        "objective": (
            "Choose the best congestion forecast, destination port, and service speed "
            "for a pharma container vessel approaching North Europe."
        ),
        "briefing": (
            "MAERSK BRUGES is carrying temperature-controlled pharma cargo. Rotterdam is "
            "the preferred discharge port, but Antwerp is a viable backup. Pick the better "
            "forecast family and submit a low-risk arrival plan."
        ),
        "vessel_id": "maersk_bruges",
        "candidate_ports": ["rotterdam", "antwerp"],
        "allowed_speeds": [12, 14],
        "deadline_hours": 175,
        "fuel_weight": 1.0,
        "lateness_multiplier": 2.0,
        "route_options": [
            {
                "port_id": "rotterdam",
                "distance_nm": 1515,
                "eta_hours": {"12": 126, "14": 108},
                "fuel_index": {"12": 58, "14": 94},
                "weather_penalty_hours": 0,
                "narrative": "Direct North Sea approach with no disruption flags.",
            },
            {
                "port_id": "antwerp",
                "distance_nm": 1556,
                "eta_hours": {"12": 130, "14": 112},
                "fuel_index": {"12": 61, "14": 97},
                "weather_penalty_hours": 0,
                "narrative": "Slightly longer inland leg and slower berth turnover.",
            },
        ],
        "congestion_history": {
            "rotterdam": [
                {"day_offset": -6, "wait_hours": 7, "berth_outage_flag": 0},
                {"day_offset": -5, "wait_hours": 8, "berth_outage_flag": 0},
                {"day_offset": -4, "wait_hours": 8, "berth_outage_flag": 1},
                {"day_offset": -3, "wait_hours": 9, "berth_outage_flag": 1},
                {"day_offset": -2, "wait_hours": 9, "berth_outage_flag": 0},
                {"day_offset": -1, "wait_hours": 8, "berth_outage_flag": 0},
            ],
            "antwerp": [
                {"day_offset": -6, "wait_hours": 13, "berth_outage_flag": 0},
                {"day_offset": -5, "wait_hours": 14, "berth_outage_flag": 0},
                {"day_offset": -4, "wait_hours": 15, "berth_outage_flag": 0},
                {"day_offset": -3, "wait_hours": 15, "berth_outage_flag": 0},
                {"day_offset": -2, "wait_hours": 16, "berth_outage_flag": 0},
                {"day_offset": -1, "wait_hours": 15, "berth_outage_flag": 0},
            ],
        },
        "forecasts": {
            "rotterdam": {
                "sarimax": {
                    "predicted_wait_hours": 8,
                    "confidence": 0.87,
                    "rationale": "Uses berth outage flags, so it captures the short disruption cleanly.",
                },
                "ets": {
                    "predicted_wait_hours": 14,
                    "confidence": 0.52,
                    "rationale": "Over-smooths the outage and leaves too much residual waiting time.",
                },
            },
            "antwerp": {
                "sarimax": {
                    "predicted_wait_hours": 15,
                    "confidence": 0.76,
                    "rationale": "Stable estimate for Antwerp's slower but steady berth pattern.",
                },
                "ets": {
                    "predicted_wait_hours": 13,
                    "confidence": 0.62,
                    "rationale": "Slightly optimistic because it ignores slower inland turn times.",
                },
            },
        },
        "actual_wait_hours": {
            "rotterdam": 9,
            "antwerp": 15,
        },
        "optimal_plan": {
            "forecast_model": "sarimax",
            "target_port_id": "rotterdam",
            "service_speed_knots": 12,
        },
    },
    "medium_asia_reroute": {
        "task_id": "medium_asia_reroute",
        "difficulty": "medium",
        "title": "Reroute around a Shanghai congestion spike",
        "objective": (
            "Find the best alternate port for a consumer electronics vessel facing a sudden "
            "Shanghai queue while still meeting a retailer replenishment window."
        ),
        "briefing": (
            "PACIFIC RESOLVE is expected to discharge electronics within 108 hours. Shanghai "
            "is heavily congested due to yard closures. Consider Ningbo or Busan if the total "
            "cost-to-serve improves."
        ),
        "vessel_id": "pacific_resolve",
        "candidate_ports": ["shanghai", "ningbo", "busan"],
        "allowed_speeds": [12, 14],
        "deadline_hours": 108,
        "fuel_weight": 0.55,
        "lateness_multiplier": 2.0,
        "route_options": [
            {
                "port_id": "shanghai",
                "distance_nm": 1128,
                "eta_hours": {"12": 94, "14": 81},
                "fuel_index": {"12": 52, "14": 80},
                "weather_penalty_hours": 6,
                "narrative": "Typhoon swell still affects pilot windows into Shanghai.",
            },
            {
                "port_id": "ningbo",
                "distance_nm": 1224,
                "eta_hours": {"12": 102, "14": 87},
                "fuel_index": {"12": 54, "14": 82},
                "weather_penalty_hours": 2,
                "narrative": "Ningbo diversion is longer but berth access is open.",
            },
            {
                "port_id": "busan",
                "distance_nm": 1680,
                "eta_hours": {"12": 140, "14": 120},
                "fuel_index": {"12": 60, "14": 92},
                "weather_penalty_hours": 0,
                "narrative": "Busan is reliable but materially farther away.",
            },
        ],
        "congestion_history": {
            "shanghai": [
                {"day_offset": -6, "wait_hours": 16, "yard_closure_flag": 0},
                {"day_offset": -5, "wait_hours": 18, "yard_closure_flag": 0},
                {"day_offset": -4, "wait_hours": 25, "yard_closure_flag": 1},
                {"day_offset": -3, "wait_hours": 33, "yard_closure_flag": 1},
                {"day_offset": -2, "wait_hours": 39, "yard_closure_flag": 1},
                {"day_offset": -1, "wait_hours": 41, "yard_closure_flag": 1},
            ],
            "ningbo": [
                {"day_offset": -6, "wait_hours": 8, "yard_closure_flag": 0},
                {"day_offset": -5, "wait_hours": 9, "yard_closure_flag": 0},
                {"day_offset": -4, "wait_hours": 10, "yard_closure_flag": 0},
                {"day_offset": -3, "wait_hours": 10, "yard_closure_flag": 0},
                {"day_offset": -2, "wait_hours": 11, "yard_closure_flag": 0},
                {"day_offset": -1, "wait_hours": 12, "yard_closure_flag": 0},
            ],
            "busan": [
                {"day_offset": -6, "wait_hours": 7, "yard_closure_flag": 0},
                {"day_offset": -5, "wait_hours": 7, "yard_closure_flag": 0},
                {"day_offset": -4, "wait_hours": 8, "yard_closure_flag": 0},
                {"day_offset": -3, "wait_hours": 8, "yard_closure_flag": 0},
                {"day_offset": -2, "wait_hours": 9, "yard_closure_flag": 0},
                {"day_offset": -1, "wait_hours": 9, "yard_closure_flag": 0},
            ],
        },
        "forecasts": {
            "shanghai": {
                "sarimax": {
                    "predicted_wait_hours": 42,
                    "confidence": 0.84,
                    "rationale": "Uses yard-closure flags and correctly anticipates the backlog spillover.",
                },
                "ets": {
                    "predicted_wait_hours": 31,
                    "confidence": 0.48,
                    "rationale": "Underestimates the disruption because it assumes rapid mean reversion.",
                },
            },
            "ningbo": {
                "sarimax": {
                    "predicted_wait_hours": 19,
                    "confidence": 0.59,
                    "rationale": "Overreacts to nearby Shanghai disruptions despite spare berth capacity.",
                },
                "ets": {
                    "predicted_wait_hours": 12,
                    "confidence": 0.79,
                    "rationale": "Tracks Ningbo's stable berth pattern and diversion capacity well.",
                },
            },
            "busan": {
                "sarimax": {
                    "predicted_wait_hours": 9,
                    "confidence": 0.71,
                    "rationale": "Reliable but route length still makes the plan expensive.",
                },
                "ets": {
                    "predicted_wait_hours": 8,
                    "confidence": 0.68,
                    "rationale": "Very close to SARIMAX because Busan is stable and uncongested.",
                },
            },
        },
        "actual_wait_hours": {
            "shanghai": 44,
            "ningbo": 12,
            "busan": 8,
        },
        "optimal_plan": {
            "forecast_model": "ets",
            "target_port_id": "ningbo",
            "service_speed_knots": 14,
        },
    },
    "hard_north_sea_allocation": {
        "task_id": "hard_north_sea_allocation",
        "difficulty": "hard",
        "title": "Replan a tanker arrival during a storm-driven North Sea disruption",
        "objective": (
            "Evaluate competing North Europe discharge options, use the better forecast family, "
            "and choose a service speed that balances arrival risk against fuel burn."
        ),
        "briefing": (
            "ATLAS SENTINEL is carrying refined products into North Europe while a North Sea storm "
            "is disrupting pilot boarding windows. Rotterdam is the commercial default, but Antwerp "
            "and Wilhelmshaven are open alternatives."
        ),
        "vessel_id": "atlas_sentinel",
        "candidate_ports": ["rotterdam", "antwerp", "wilhelmshaven"],
        "allowed_speeds": [12, 14],
        "deadline_hours": 290,
        "fuel_weight": 1.1,
        "lateness_multiplier": 1.4,
        "route_options": [
            {
                "port_id": "rotterdam",
                "distance_nm": 3024,
                "eta_hours": {"12": 252, "14": 216},
                "fuel_index": {"12": 70, "14": 106},
                "weather_penalty_hours": 18,
                "narrative": "Direct route, but the storm heavily impacts pilot windows and anchorage time.",
            },
            {
                "port_id": "antwerp",
                "distance_nm": 3120,
                "eta_hours": {"12": 260, "14": 223},
                "fuel_index": {"12": 72, "14": 109},
                "weather_penalty_hours": 10,
                "narrative": "Moderate storm exposure with extra inland transit once berthed.",
            },
            {
                "port_id": "wilhelmshaven",
                "distance_nm": 2976,
                "eta_hours": {"12": 248, "14": 213},
                "fuel_index": {"12": 68, "14": 103},
                "weather_penalty_hours": 6,
                "narrative": "Best weather shelter and strongest spare berth position.",
            },
        ],
        "congestion_history": {
            "rotterdam": [
                {"day_offset": -6, "wait_hours": 18, "storm_flag": 0},
                {"day_offset": -5, "wait_hours": 21, "storm_flag": 0},
                {"day_offset": -4, "wait_hours": 26, "storm_flag": 1},
                {"day_offset": -3, "wait_hours": 31, "storm_flag": 1},
                {"day_offset": -2, "wait_hours": 34, "storm_flag": 1},
                {"day_offset": -1, "wait_hours": 35, "storm_flag": 1},
            ],
            "antwerp": [
                {"day_offset": -6, "wait_hours": 15, "storm_flag": 0},
                {"day_offset": -5, "wait_hours": 16, "storm_flag": 0},
                {"day_offset": -4, "wait_hours": 17, "storm_flag": 1},
                {"day_offset": -3, "wait_hours": 18, "storm_flag": 1},
                {"day_offset": -2, "wait_hours": 18, "storm_flag": 1},
                {"day_offset": -1, "wait_hours": 18, "storm_flag": 1},
            ],
            "wilhelmshaven": [
                {"day_offset": -6, "wait_hours": 7, "storm_flag": 0},
                {"day_offset": -5, "wait_hours": 8, "storm_flag": 0},
                {"day_offset": -4, "wait_hours": 8, "storm_flag": 1},
                {"day_offset": -3, "wait_hours": 9, "storm_flag": 1},
                {"day_offset": -2, "wait_hours": 10, "storm_flag": 1},
                {"day_offset": -1, "wait_hours": 10, "storm_flag": 1},
            ],
        },
        "forecasts": {
            "rotterdam": {
                "sarimax": {
                    "predicted_wait_hours": 34,
                    "confidence": 0.83,
                    "rationale": "Incorporates storm flags and anchorage spillover from the prior two days.",
                },
                "ets": {
                    "predicted_wait_hours": 24,
                    "confidence": 0.47,
                    "rationale": "Too optimistic because it smooths over storm-driven queue growth.",
                },
            },
            "antwerp": {
                "sarimax": {
                    "predicted_wait_hours": 18,
                    "confidence": 0.79,
                    "rationale": "Captures the moderate weather impact without overstating congestion.",
                },
                "ets": {
                    "predicted_wait_hours": 24,
                    "confidence": 0.43,
                    "rationale": "Pulls the forecast upward as if Antwerp were moving with Rotterdam.",
                },
            },
            "wilhelmshaven": {
                "sarimax": {
                    "predicted_wait_hours": 10,
                    "confidence": 0.86,
                    "rationale": "Best fit because Wilhelmshaven keeps spare berths and lower storm exposure.",
                },
                "ets": {
                    "predicted_wait_hours": 17,
                    "confidence": 0.44,
                    "rationale": "Overstates queue growth after the regional storm signal.",
                },
            },
        },
        "actual_wait_hours": {
            "rotterdam": 36,
            "antwerp": 18,
            "wilhelmshaven": 11,
        },
        "optimal_plan": {
            "forecast_model": "sarimax",
            "target_port_id": "wilhelmshaven",
            "service_speed_knots": 12,
        },
    },
}


SCENARIO_CONFIG: List[Dict[str, Any]] = [
    {
        "task_id": task["task_id"],
        "difficulty": task["difficulty"],
        "title": task["title"],
        "objective": task["objective"],
        "expected_commands": [
            "inspect_vessel",
            "inspect_congestion_history",
            "inspect_forecast",
            "inspect_route_options",
            "submit_plan",
        ],
    }
    for task in TASKS.values()
]


def get_task_catalog() -> List[Dict[str, Any]]:
    """Return a lightweight catalog for task discovery."""

    return [
        {
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "title": task["title"],
            "objective": task["objective"],
            "candidate_ports": task["candidate_ports"],
            "allowed_speeds": task["allowed_speeds"],
        }
        for task in TASKS.values()
    ]


def get_task(task_id: str) -> Dict[str, Any]:
    """Return a deep copy of a seeded task by identifier."""

    if task_id not in TASKS:
        raise KeyError(task_id)
    return deepcopy(TASKS[task_id])


def get_port(port_id: str) -> Dict[str, Any]:
    """Return seeded port metadata."""

    if port_id not in PORTS:
        raise KeyError(port_id)
    return deepcopy(PORTS[port_id])


def get_vessel(vessel_id: str) -> Dict[str, Any]:
    """Return seeded vessel metadata."""

    if vessel_id not in VESSELS:
        raise KeyError(vessel_id)
    return deepcopy(VESSELS[vessel_id])
