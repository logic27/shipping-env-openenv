# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed action, observation, and state models for the shipping environment."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


ShippingCommand = Literal[
    "list_tasks",
    "load_task",
    "inspect_vessel",
    "inspect_port",
    "inspect_congestion_history",
    "inspect_route_options",
    "inspect_forecast",
    "submit_plan",
]

ForecastModel = Literal["sarimax", "ets"]


class ShippingAction(Action):
    """Agent action for the maritime disruption planning environment."""

    command: ShippingCommand = Field(
        ...,
        description="High-level environment command to execute.",
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Task identifier used with load_task.",
    )
    vessel_id: Optional[str] = Field(
        default=None,
        description="Vessel identifier for vessel or route lookups.",
    )
    port_id: Optional[str] = Field(
        default=None,
        description="Port identifier for port or forecast lookups.",
    )
    forecast_model: Optional[ForecastModel] = Field(
        default=None,
        description="Forecast family to inspect or commit to in the final plan.",
    )
    target_port_id: Optional[str] = Field(
        default=None,
        description="Chosen destination port for the final submission.",
    )
    service_speed_knots: Optional[int] = Field(
        default=None,
        description="Chosen sailing speed in knots for the final submission.",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Optional explanation for the final plan.",
    )


class ShippingObservation(Observation):
    """Observation returned after each environment interaction."""

    # Use string markers here so raw observation payloads avoid numeric booleans.
    done: str = Field(
        default="false",
        description="Whether the episode has terminated, serialized as 'true' or 'false'.",
    )

    summary: str = Field(default="", description="Human-readable step result.")
    active_task_id: Optional[str] = Field(
        default=None,
        description="Currently loaded task identifier.",
    )
    phase: str = Field(
        default="task_selection",
        description="Current interaction phase inside the environment.",
    )
    available_commands: List[str] = Field(
        default_factory=list,
        description="Commands that make sense in the current phase.",
    )
    artifacts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured records returned by the latest command.",
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Numeric metrics for the latest step or final grade.",
    )


class ShippingState(State):
    """Sanitized state model for raw transport payloads."""

    step_count: str = Field(
        default="active",
        description="Session progression marker, serialized as text for validator safety.",
    )


# Backward-compatible aliases so existing imports do not break.
MyAction = ShippingAction
MyObservation = ShippingObservation
