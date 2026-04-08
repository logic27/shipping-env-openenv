# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core environment logic for the shipping environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ShippingAction, ShippingObservation
    from ..scenario_data import get_port, get_task, get_task_catalog, get_vessel
except ImportError:
    from models import ShippingAction, ShippingObservation
    from scenario_data import get_port, get_task, get_task_catalog, get_vessel


class ShippingEnvironment(Environment):
    """Offline maritime disruption planning environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MIN_SCORE: float = 0.01

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._active_task_id: Optional[str] = None
        self._active_task: Optional[Dict[str, Any]] = None
        self._seen_commands: Set[str] = set()
        self._evidence_types: Set[str] = set()

    def reset(self) -> ShippingObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._active_task_id = None
        self._active_task = None
        self._seen_commands.clear()
        self._evidence_types.clear()

        return self._observation(
            summary=(
                "shipping_env is ready. Start with `list_tasks`, then `load_task` to begin "
                "a seeded maritime planning scenario."
            ),
            phase="task_selection",
            reward=self.MIN_SCORE,
            done=False,
            metadata={
                "environment": "shipping_env",
                "theme": "forecast-assisted maritime disruption planning",
                "task_count": len(get_task_catalog()),
            },
        )

    def step(self, action: ShippingAction) -> ShippingObservation:  # type: ignore[override]
        self._state.step_count += 1

        command = action.command
        if command == "list_tasks":
            return self._handle_list_tasks()
        if command == "load_task":
            return self._handle_load_task(action)

        if self._active_task is None:
            return self._invalid(
                "No task is active. Use `load_task` with one of the catalog ids first."
            )

        if command == "inspect_vessel":
            return self._handle_inspect_vessel(action)
        if command == "inspect_port":
            return self._handle_inspect_port(action)
        if command == "inspect_congestion_history":
            return self._handle_inspect_history(action)
        if command == "inspect_route_options":
            return self._handle_inspect_routes(action)
        if command == "inspect_forecast":
            return self._handle_inspect_forecast(action)
        if command == "submit_plan":
            return self._handle_submit_plan(action)

        return self._invalid(f"Unsupported command `{command}`.")

    @property
    def state(self) -> State:
        return self._state

    def _handle_list_tasks(self) -> ShippingObservation:
        return self._observation(
            summary=(
                "Three deterministic maritime tasks are available. Load one task to inspect "
                "vessels, ports, congestion history, route options, and model forecasts."
            ),
            phase="task_selection",
            reward=self._shape_reward("list_tasks"),
            done=False,
            artifacts=get_task_catalog(),
            metadata={"task_catalog": get_task_catalog()},
        )

    def _handle_load_task(self, action: ShippingAction) -> ShippingObservation:
        if not action.task_id:
            return self._invalid("`load_task` requires `task_id`.")

        try:
            task = get_task(action.task_id)
        except KeyError:
            return self._invalid(f"Unknown task `{action.task_id}`.")

        self._active_task_id = action.task_id
        self._active_task = task
        self._seen_commands = {f"load_task:{action.task_id}"}
        self._evidence_types.clear()

        vessel = get_vessel(task["vessel_id"])
        briefing_artifact = {
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "title": task["title"],
            "objective": task["objective"],
            "briefing": task["briefing"],
            "primary_vessel": vessel["name"],
            "candidate_ports": task["candidate_ports"],
            "allowed_speeds": task["allowed_speeds"],
        }

        return self._observation(
            summary=task["briefing"],
            phase="analysis",
            reward=0.10,
            done=False,
            artifacts=[briefing_artifact],
            metadata={
                "candidate_ports": task["candidate_ports"],
                "allowed_speeds": task["allowed_speeds"],
                "deadline_hours": task["deadline_hours"],
            },
        )

    def _handle_inspect_vessel(self, action: ShippingAction) -> ShippingObservation:
        assert self._active_task is not None
        vessel_id = action.vessel_id or self._active_task["vessel_id"]
        if vessel_id != self._active_task["vessel_id"]:
            return self._invalid(
                f"Vessel `{vessel_id}` is not part of active task `{self._active_task_id}`."
            )

        vessel = get_vessel(vessel_id)
        self._evidence_types.add("inspect_vessel")
        return self._observation(
            summary=(
                f"{vessel['name']} is the active vessel. It currently points toward "
                f"{self._active_task['candidate_ports'][0]} but can be replanned."
            ),
            phase="analysis",
            reward=self._shape_reward(f"inspect_vessel:{vessel_id}"),
            done=False,
            artifacts=[vessel],
        )

    def _handle_inspect_port(self, action: ShippingAction) -> ShippingObservation:
        assert self._active_task is not None
        port_id = action.port_id
        if not port_id:
            return self._invalid("`inspect_port` requires `port_id`.")
        if port_id not in self._active_task["candidate_ports"]:
            return self._invalid(
                f"Port `{port_id}` is not in the candidate set for `{self._active_task_id}`."
            )

        port = get_port(port_id)
        self._evidence_types.add("inspect_port")
        return self._observation(
            summary=(
                f"{port['name']} currently has congestion index {port['congestion_index']:.2f} "
                f"with {port['available_berths']} berth slots available."
            ),
            phase="analysis",
            reward=self._shape_reward(f"inspect_port:{port_id}"),
            done=False,
            artifacts=[port],
        )

    def _handle_inspect_history(self, action: ShippingAction) -> ShippingObservation:
        assert self._active_task is not None
        port_id = action.port_id
        if not port_id:
            return self._invalid("`inspect_congestion_history` requires `port_id`.")
        history = self._active_task["congestion_history"].get(port_id)
        if history is None:
            return self._invalid(
                f"No congestion history is available for port `{port_id}` in this task."
            )

        self._evidence_types.add("inspect_congestion_history")
        return self._observation(
            summary=(
                f"Loaded the last {len(history)} congestion snapshots for `{port_id}`. "
                "Use them to judge whether SARIMAX or ETS is more credible."
            ),
            phase="analysis",
            reward=self._shape_reward(f"inspect_congestion_history:{port_id}"),
            done=False,
            artifacts=history,
            metadata={"port_id": port_id},
        )

    def _handle_inspect_routes(self, action: ShippingAction) -> ShippingObservation:
        assert self._active_task is not None
        vessel_id = action.vessel_id or self._active_task["vessel_id"]
        if vessel_id != self._active_task["vessel_id"]:
            return self._invalid(
                f"Route options are only seeded for vessel `{self._active_task['vessel_id']}`."
            )

        self._evidence_types.add("inspect_route_options")
        return self._observation(
            summary=(
                "Route options include ETA, fuel index, and weather penalty for each "
                "candidate port at both service speeds."
            ),
            phase="analysis",
            reward=self._shape_reward("inspect_route_options"),
            done=False,
            artifacts=self._active_task["route_options"],
            metadata={
                "fuel_weight": self._active_task["fuel_weight"],
                "deadline_hours": self._active_task["deadline_hours"],
                "lateness_multiplier": self._active_task["lateness_multiplier"],
            },
        )

    def _handle_inspect_forecast(self, action: ShippingAction) -> ShippingObservation:
        assert self._active_task is not None
        if not action.port_id or not action.forecast_model:
            return self._invalid(
                "`inspect_forecast` requires both `port_id` and `forecast_model`."
            )

        port_forecasts = self._active_task["forecasts"].get(action.port_id)
        if port_forecasts is None:
            return self._invalid(f"No forecast bundle found for port `{action.port_id}`.")
        forecast = port_forecasts.get(action.forecast_model)
        if forecast is None:
            return self._invalid(
                f"Model `{action.forecast_model}` is unavailable for port `{action.port_id}`."
            )

        self._evidence_types.add("inspect_forecast")
        return self._observation(
            summary=(
                f"{action.forecast_model.upper()} predicts {forecast['predicted_wait_hours']} "
                f"hours of waiting at `{action.port_id}`."
            ),
            phase="analysis",
            reward=self._shape_reward(
                f"inspect_forecast:{action.port_id}:{action.forecast_model}"
            ),
            done=False,
            artifacts=[
                {
                    "port_id": action.port_id,
                    "forecast_model": action.forecast_model,
                    **forecast,
                }
            ],
        )

    def _handle_submit_plan(self, action: ShippingAction) -> ShippingObservation:
        assert self._active_task is not None
        missing = [
            name
            for name, value in (
                ("forecast_model", action.forecast_model),
                ("target_port_id", action.target_port_id),
                ("service_speed_knots", action.service_speed_knots),
            )
            if value is None
        ]
        if missing:
            return self._invalid(
                f"`submit_plan` is missing required fields: {', '.join(missing)}."
            )

        target_port = action.target_port_id
        speed = str(action.service_speed_knots)
        route_option = self._route_option_for(target_port)
        if route_option is None:
            return self._invalid(
                f"Target port `{target_port}` is not valid for task `{self._active_task_id}`."
            )
        if speed not in route_option["eta_hours"]:
            return self._invalid(
                f"Speed `{action.service_speed_knots}` knots is not seeded for port `{target_port}`."
            )

        predicted_wait = self._active_task["forecasts"][target_port][action.forecast_model][
            "predicted_wait_hours"
        ]
        actual_wait = self._active_task["actual_wait_hours"][target_port]
        eta_hours = route_option["eta_hours"][speed]
        weather_penalty = route_option["weather_penalty_hours"]
        fuel_index = route_option["fuel_index"][speed]

        business_cost_pred = self._business_cost(
            eta_hours=eta_hours,
            wait_hours=predicted_wait,
            weather_penalty=weather_penalty,
            fuel_index=fuel_index,
        )
        business_cost_actual = self._business_cost(
            eta_hours=eta_hours,
            wait_hours=actual_wait,
            weather_penalty=weather_penalty,
            fuel_index=fuel_index,
        )

        optimal = self._active_task["optimal_plan"]
        evidence_score = self._evidence_score()
        score_breakdown = {
            "forecast_model": 0.25
            if action.forecast_model == optimal["forecast_model"]
            else self.MIN_SCORE,
            "target_port": 0.45
            if target_port == optimal["target_port_id"]
            else 0.15 if target_port in self._active_task["candidate_ports"] else self.MIN_SCORE,
            "service_speed": 0.20
            if action.service_speed_knots == optimal["service_speed_knots"]
            else self.MIN_SCORE,
            "evidence": evidence_score,
        }
        raw_total_score = round(sum(score_breakdown.values()), 2)
        total_score = max(0.01, min(0.99, raw_total_score))

        return self._observation(
            summary=(
                f"Plan submitted for `{self._active_task_id}` with score {total_score:.2f}. "
                f"Predicted business cost index: {business_cost_pred:.1f}. "
                f"Realized business cost index: {business_cost_actual:.1f}."
            ),
            phase="submitted",
            reward=total_score,
            done=True,
            artifacts=[
                {
                    "task_id": self._active_task_id,
                    "chosen_forecast_model": action.forecast_model,
                    "chosen_port_id": target_port,
                    "chosen_speed_knots": action.service_speed_knots,
                    "predicted_wait_hours": predicted_wait,
                    "actual_wait_hours": actual_wait,
                    "predicted_business_cost": round(business_cost_pred, 2),
                    "actual_business_cost": round(business_cost_actual, 2),
                    "rationale": action.rationale or "",
                }
            ],
            metrics={
                "score": total_score,
                "forecast_model_score": score_breakdown["forecast_model"],
                "target_port_score": score_breakdown["target_port"],
                "service_speed_score": score_breakdown["service_speed"],
                "evidence_score": score_breakdown["evidence"],
            },
            metadata={
                "optimal_plan": optimal,
                "score_breakdown": score_breakdown,
                "predicted_business_cost": round(business_cost_pred, 2),
                "actual_business_cost": round(business_cost_actual, 2),
            },
        )

    def _business_cost(
        self,
        eta_hours: int,
        wait_hours: int,
        weather_penalty: int,
        fuel_index: int,
    ) -> float:
        assert self._active_task is not None
        total_hours = eta_hours + wait_hours + weather_penalty
        lateness = max(0, total_hours - self._active_task["deadline_hours"])
        return (
            total_hours
            + self._active_task["fuel_weight"] * fuel_index
            + self._active_task["lateness_multiplier"] * lateness
        )

    def _evidence_score(self) -> float:
        required = {
            "inspect_vessel",
            "inspect_congestion_history",
            "inspect_forecast",
            "inspect_route_options",
        }
        if required.issubset(self._evidence_types):
            return 0.10
        if len(required.intersection(self._evidence_types)) >= 2:
            return 0.05
        return self.MIN_SCORE

    def _route_option_for(self, port_id: str) -> Optional[Dict[str, Any]]:
        assert self._active_task is not None
        for route in self._active_task["route_options"]:
            if route["port_id"] == port_id:
                return route
        return None

    def _shape_reward(self, command_key: str) -> float:
        if command_key in self._seen_commands:
            return self.MIN_SCORE
        self._seen_commands.add(command_key)
        return 0.05

    def _invalid(self, message: str) -> ShippingObservation:
        phase = "analysis" if self._active_task_id else "task_selection"
        return self._observation(
            summary=message,
            phase=phase,
            reward=self.MIN_SCORE,
            done=False,
        )

    def _available_commands(self) -> List[str]:
        base = ["list_tasks", "load_task"]
        if self._active_task_id is None:
            return base
        return base + [
            "inspect_vessel",
            "inspect_port",
            "inspect_congestion_history",
            "inspect_route_options",
            "inspect_forecast",
            "submit_plan",
        ]

    def _observation(
        self,
        summary: str,
        phase: str,
        reward: float,
        done: bool,
        artifacts: Optional[List[Dict[str, Any]]] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ShippingObservation:
        return ShippingObservation(
            summary=summary,
            active_task_id=self._active_task_id,
            phase=phase,
            available_commands=self._available_commands(),
            artifacts=artifacts or [],
            metrics=metrics or {},
            metadata=metadata or {},
            reward=reward,
            done=done,
        )


# Backward-compatible alias.
MyEnvironment = ShippingEnvironment
