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

from openai import OpenAI

from my_env import ShippingAction, ShippingEnv
from my_env.scenario_data import get_task_catalog
from my_env.server.my_env_environment import ShippingEnvironment

# Required environment variables for the hackathon validator.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
BENCHMARK = "shipping_env"


def emit_log(stage: str, payload: Dict[str, Any]) -> None:
    """Print validator-friendly structured logs."""

    print(f"[{stage}] {json.dumps(payload)}", flush=True)


def log_start(task: str, env: str, model: str) -> None:
    emit_log(
        "START",
        {
            "task": task,
            "env": env,
            "model": model,
        },
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    emit_log(
        "STEP",
        {
            "step": step,
            "action": action,
            "reward": reward,
            "done": done,
            "error": error,
        },
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    emit_log(
        "END",
        {
            "success": success,
            "steps": steps,
            "score": score,
            "rewards": rewards,
        },
    )


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


def build_openai_client() -> Optional[OpenAI]:
    """Return an OpenAI-compatible client for HF router calls when configured."""

    if not HF_TOKEN:
        return None
    return OpenAI(
        base_url=HF_ROUTER_BASE_URL,
        api_key=HF_TOKEN,
    )


def generate_llm_rationale(
    task_id: str,
    candidate_plan: Dict[str, Any],
    route: Dict[str, Any],
    predicted_wait_hours: int,
) -> str:
    """
    Generate a short rationale using the OpenAI client when HF credentials exist.

    Any failure falls back to a deterministic explanation so inference never crashes.
    """

    client = build_openai_client()
    if client is None:
        return (
            f"Selected {candidate_plan['target_port_id']} at "
            f"{candidate_plan['service_speed_knots']} knots because it minimized the "
            f"seeded business cost proxy with {candidate_plan['forecast_model']}."
        )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise maritime operations planner. "
                        "Return one sentence under 35 words."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {task_id}. Chosen port: {candidate_plan['target_port_id']}. "
                        f"Speed: {candidate_plan['service_speed_knots']} knots. "
                        f"Forecast model: {candidate_plan['forecast_model']}. "
                        f"Predicted wait: {predicted_wait_hours} hours. "
                        f"Weather penalty: {route['weather_penalty_hours']} hours. "
                        "Explain why this is a reasonable maritime plan."
                    ),
                },
            ],
            max_completion_tokens=80,
        )
        content = response.choices[0].message.content
        if content:
            return content.strip()
    except Exception:
        pass

    return (
        f"Selected {candidate_plan['target_port_id']} at "
        f"{candidate_plan['service_speed_knots']} knots because it minimized the "
        f"seeded business cost proxy with {candidate_plan['forecast_model']}."
    )


def build_candidate_plans_http(
    env: Any,
) -> Dict[str, Any]:
    """Inspect a task over HTTP and return all candidate plans plus task metadata."""

    env.step(ShippingAction(command="inspect_vessel"))
    task_info = env.step(ShippingAction(command="inspect_route_options")).observation
    route_options = task_info.artifacts
    task_meta = task_info.metadata

    candidates: List[Dict[str, Any]] = []
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
                candidates.append(
                    {
                        "forecast_model": model_name,
                        "target_port_id": port_id,
                        "service_speed_knots": speed,
                        "cost": round(cost, 2),
                        "predicted_wait_hours": predicted_wait,
                        "route": route,
                    }
                )

    return {
        "task_meta": task_meta,
        "candidates": candidates,
    }


def build_candidate_plans_local(env: ShippingEnvironment) -> Dict[str, Any]:
    """Inspect an already-loaded local task and return all candidate plans."""

    _local_step(env, ShippingAction(command="inspect_vessel"))
    route_result = _local_step(env, ShippingAction(command="inspect_route_options"))
    route_options = route_result["observation"].artifacts
    task = env._active_task  # type: ignore[attr-defined]
    assert task is not None

    candidates: List[Dict[str, Any]] = []
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
                candidates.append(
                    {
                        "forecast_model": model_name,
                        "target_port_id": port_id,
                        "service_speed_knots": speed,
                        "cost": round(cost, 2),
                        "predicted_wait_hours": predicted_wait,
                        "route": route,
                    }
                )

    return {
        "task_meta": task,
        "candidates": candidates,
    }


def choose_best_plan(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the cheapest deterministic plan, preferring slower speed on ties."""

    best_plan: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        if best_plan is None or candidate["cost"] < best_plan["cost"] or (
            candidate["cost"] == best_plan["cost"]
            and candidate["service_speed_knots"] < best_plan["service_speed_knots"]
        ):
            best_plan = candidate
    assert best_plan is not None
    return best_plan


def solve_task_http(base_url: str, task_id: str) -> Dict[str, Any]:
    with ShippingEnv(base_url=base_url).sync() as env:
        env.reset()
        env.step(ShippingAction(command="load_task", task_id=task_id))
        candidate_bundle = build_candidate_plans_http(env)
        best_plan = choose_best_plan(candidate_bundle["candidates"])
        rationale = generate_llm_rationale(
            task_id=task_id,
            candidate_plan=best_plan,
            route=best_plan["route"],
            predicted_wait_hours=best_plan["predicted_wait_hours"],
        )

        final = env.step(
            ShippingAction(
                command="submit_plan",
                forecast_model=best_plan["forecast_model"],
                target_port_id=best_plan["target_port_id"],
                service_speed_knots=best_plan["service_speed_knots"],
                rationale=rationale,
            )
        )
        return {
            "task_id": task_id,
            "plan": {
                "forecast_model": best_plan["forecast_model"],
                "target_port_id": best_plan["target_port_id"],
                "service_speed_knots": best_plan["service_speed_knots"],
                "cost": best_plan["cost"],
                "rationale": rationale,
            },
            "reward": final.reward,
            "metrics": final.observation.metrics,
            "execution_mode": "http",
            "base_url": base_url,
        }


def solve_task_local(task_id: str) -> Dict[str, Any]:
    env = ShippingEnvironment()
    env.reset()
    _local_step(env, ShippingAction(command="load_task", task_id=task_id))
    candidate_bundle = build_candidate_plans_local(env)
    best_plan = choose_best_plan(candidate_bundle["candidates"])
    rationale = generate_llm_rationale(
        task_id=task_id,
        candidate_plan=best_plan,
        route=best_plan["route"],
        predicted_wait_hours=best_plan["predicted_wait_hours"],
    )

    final = _local_step(
        env,
        ShippingAction(
            command="submit_plan",
            forecast_model=best_plan["forecast_model"],
            target_port_id=best_plan["target_port_id"],
            service_speed_knots=best_plan["service_speed_knots"],
            rationale=rationale,
        ),
    )
    return {
        "task_id": task_id,
        "plan": {
            "forecast_model": best_plan["forecast_model"],
            "target_port_id": best_plan["target_port_id"],
            "service_speed_knots": best_plan["service_speed_knots"],
            "cost": best_plan["cost"],
            "rationale": rationale,
        },
        "reward": final["reward"],
        "metrics": final["observation"].metrics,
        "execution_mode": "local",
    }


def solve_task(task_id: str) -> Dict[str, Any]:
    """Run one task, preferring HTTP if the validator provides an env URL."""

    if not API_BASE_URL:
        return solve_task_local(task_id)

    try:
        return solve_task_http(API_BASE_URL, task_id)
    except Exception as exc:
        fallback_result = solve_task_local(task_id)
        fallback_result["execution_mode"] = "local_fallback"
        fallback_result["fallback_reason"] = (
            f"HTTP execution failed for {API_BASE_URL}: {exc.__class__.__name__}: {exc}"
        )
        return fallback_result


def run_all_tasks() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for task in get_task_catalog():
        task_id = task["task_id"]
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        result = solve_task(task_id)
        reward = float(result["reward"] or 0.0)
        action = (
            "submit_plan:"
            f"{result['plan']['forecast_model']}/"
            f"{result['plan']['target_port_id']}/"
            f"{result['plan']['service_speed_knots']}"
        )
        error = result.get("fallback_reason")
        log_step(
            step=1,
            action=action,
            reward=reward,
            done=True,
            error=error,
        )
        score = float(result["metrics"].get("score", reward))
        score = min(max(score, 0.0), 1.0)
        log_end(
            success=score >= 0.8,
            steps=1,
            score=score,
            rewards=[reward],
        )
        results.append(result)
    return results


if __name__ == "__main__":
    try:
        run_all_tasks()
    except Exception as exc:
        log_end(
            success=False,
            steps=0,
            score=0.0,
            rewards=[],
        )
        print(
            f"[DEBUG] inference.py failed with {exc.__class__.__name__}: {exc}",
            flush=True,
        )
