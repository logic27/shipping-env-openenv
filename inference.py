# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Heuristic inference baseline for shipping_env."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from my_env import ShippingAction, ShippingEnv
from my_env.scenario_data import get_task_catalog
from my_env.server.my_env_environment import ShippingEnvironment


def _local_step(env: ShippingEnvironment, action: ShippingAction) -> Dict[str, Any]:
    observation = env.step(action)
    return {
        "reward": observation.reward,
        "done": observation.done,
        "observation": observation,
    }


def _route_cost(
    route: Dict[str, Any],
    predicted_wait_hours: int,
    deadline_hours: int,
    fuel_weight: float,
    lateness_multiplier: float,
    speed: int,
) -> float:
    eta_hours = route["eta_hours"][str(speed)]
    weather_penalty = route["weather_penalty_hours"]
    fuel_index = route["fuel_index"][str(speed)]
    total_hours = eta_hours + predicted_wait_hours + weather_penalty
    lateness = max(0, total_hours - deadline_hours)
    return total_hours + fuel_weight * fuel_index + lateness_multiplier * lateness


def solve_task_http(base_url: str, task_id: str) -> Dict[str, Any]:
    with ShippingEnv(base_url=base_url) as env:
        env.reset()
        env.step(ShippingAction(command="load_task", task_id=task_id))
        env.step(ShippingAction(command="inspect_vessel"))
        task_info = env.step(ShippingAction(command="inspect_route_options")).observation
        route_options = task_info.artifacts
        task_meta = task_info.metadata

        best_plan: Optional[Dict[str, Any]] = None
        for route in route_options:
            port_id = route["port_id"]
            env.step(ShippingAction(command="inspect_congestion_history", port_id=port_id))
            for model_name in ("sarimax", "ets"):
                forecast_obs = env.step(
                    ShippingAction(
                        command="inspect_forecast",
                        port_id=port_id,
                        forecast_model=model_name,
                    )
                ).observation
                predicted_wait = forecast_obs.artifacts[0]["predicted_wait_hours"]
                for speed in (12, 14):
                    cost = _route_cost(
                        route=route,
                        predicted_wait_hours=predicted_wait,
                        deadline_hours=task_meta["deadline_hours"],
                        fuel_weight=task_meta["fuel_weight"],
                        lateness_multiplier=task_meta["lateness_multiplier"],
                        speed=speed,
                    )
                    candidate = {
                        "forecast_model": model_name,
                        "target_port_id": port_id,
                        "service_speed_knots": speed,
                        "cost": cost,
                    }
                    if best_plan is None or candidate["cost"] < best_plan["cost"] or (
                        candidate["cost"] == best_plan["cost"]
                        and candidate["service_speed_knots"] < best_plan["service_speed_knots"]
                    ):
                        best_plan = candidate

        assert best_plan is not None
        final = env.step(
            ShippingAction(
                command="submit_plan",
                forecast_model=best_plan["forecast_model"],
                target_port_id=best_plan["target_port_id"],
                service_speed_knots=best_plan["service_speed_knots"],
                rationale="Heuristic baseline minimizing seeded business cost.",
            )
        )
        return {
            "task_id": task_id,
            "plan": best_plan,
            "reward": final.reward,
            "metrics": final.observation.metrics,
        }


def solve_task_local(task_id: str) -> Dict[str, Any]:
    env = ShippingEnvironment()
    env.reset()
    _local_step(env, ShippingAction(command="load_task", task_id=task_id))
    _local_step(env, ShippingAction(command="inspect_vessel"))
    route_result = _local_step(env, ShippingAction(command="inspect_route_options"))
    route_options = route_result["observation"].artifacts
    task = env._active_task  # type: ignore[attr-defined]
    assert task is not None

    best_plan: Optional[Dict[str, Any]] = None
    for route in route_options:
        port_id = route["port_id"]
        _local_step(env, ShippingAction(command="inspect_congestion_history", port_id=port_id))
        for model_name in ("sarimax", "ets"):
            forecast_result = _local_step(
                env,
                ShippingAction(
                    command="inspect_forecast",
                    port_id=port_id,
                    forecast_model=model_name,
                ),
            )
            predicted_wait = forecast_result["observation"].artifacts[0]["predicted_wait_hours"]
            for speed in (12, 14):
                cost = _route_cost(
                    route=route,
                    predicted_wait_hours=predicted_wait,
                    deadline_hours=task["deadline_hours"],
                    fuel_weight=task["fuel_weight"],
                    lateness_multiplier=task["lateness_multiplier"],
                    speed=speed,
                )
                candidate = {
                    "forecast_model": model_name,
                    "target_port_id": port_id,
                    "service_speed_knots": speed,
                    "cost": round(cost, 2),
                }
                if best_plan is None or candidate["cost"] < best_plan["cost"] or (
                    candidate["cost"] == best_plan["cost"]
                    and candidate["service_speed_knots"] < best_plan["service_speed_knots"]
                ):
                    best_plan = candidate

    assert best_plan is not None
    final = _local_step(
        env,
        ShippingAction(
            command="submit_plan",
            forecast_model=best_plan["forecast_model"],
            target_port_id=best_plan["target_port_id"],
            service_speed_knots=best_plan["service_speed_knots"],
            rationale="Local heuristic baseline minimizing seeded business cost.",
        ),
    )
    return {
        "task_id": task_id,
        "plan": best_plan,
        "reward": final["reward"],
        "metrics": final["observation"].metrics,
    }


def run_all_tasks() -> List[Dict[str, Any]]:
    base_url = os.getenv("API_BASE_URL")
    results: List[Dict[str, Any]] = []
    for task in get_task_catalog():
        if base_url:
            results.append(solve_task_http(base_url, task["task_id"]))
        else:
            results.append(solve_task_local(task["task_id"]))
    return results


if __name__ == "__main__":
    print(json.dumps(run_all_tasks(), indent=2))
