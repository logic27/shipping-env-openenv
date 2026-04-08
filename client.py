# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client helpers for the shipping environment."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ShippingAction, ShippingObservation


class ShippingEnv(
    EnvClient[ShippingAction, ShippingObservation, State]
):
    """
    Client for the shipping environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with ShippingEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.summary)
        ...
        ...     result = client.step(ShippingAction(command="list_tasks"))
        ...     print(result.observation.artifacts)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ShippingEnv.from_docker_image("shipping-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(ShippingAction(command="list_tasks"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ShippingAction) -> Dict[str, Any]:
        """
        Convert ShippingAction to JSON payload for step message.

        Args:
            action: ShippingAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        if hasattr(action, "model_dump"):
            return action.model_dump(exclude_none=True)
        return action.dict(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ShippingObservation]:
        """
        Parse server response into StepResult[ShippingObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ShippingObservation
        """
        obs_data = payload.get("observation", {})
        raw_done = payload.get("done", False)
        if isinstance(raw_done, str):
            done_str = raw_done.lower()
            done_value = done_str == "true"
        else:
            done_value = bool(raw_done)
            done_str = "true" if done_value else "false"

        observation = ShippingObservation(
            summary=obs_data.get("summary", ""),
            active_task_id=obs_data.get("active_task_id"),
            phase=obs_data.get("phase", "task_selection"),
            available_commands=obs_data.get("available_commands", []),
            artifacts=obs_data.get("artifacts", []),
            metrics=obs_data.get("metrics", {}),
            done=done_str,
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=done_value,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        step_count = payload.get("step_count", 0)
        if not isinstance(step_count, int):
            step_count = 0

        return State(
            episode_id=payload.get("episode_id"),
            step_count=step_count,
        )


# Backward-compatible alias.
MyEnv = ShippingEnv
