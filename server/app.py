# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application entrypoint for the shipping environment."""

import json

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from my_env.models import ShippingAction, ShippingObservation
    from my_env.server.my_env_environment import ShippingEnvironment
except ImportError:
    try:
        from ..models import ShippingAction, ShippingObservation
        from .my_env_environment import ShippingEnvironment
    except ImportError:
        from models import ShippingAction, ShippingObservation
        from server.my_env_environment import ShippingEnvironment

from fastapi import Request
from fastapi.responses import JSONResponse, Response


app = create_app(
    ShippingEnvironment,
    ShippingAction,
    ShippingObservation,
    env_name="shipping_env",
    max_concurrent_envs=4,
)


def _sanitize_http_payload(path: str, payload: object) -> object:
    if not isinstance(payload, dict):
        return payload

    if path in {"/reset", "/step", "/web/reset", "/web/step"}:
        done_value = payload.get("done")
        if isinstance(done_value, bool):
            payload["done"] = "true" if done_value else "false"

    if path in {"/state", "/web/state"} and "step_count" in payload:
        payload["step_count"] = "active"

    return payload


@app.middleware("http")
async def sanitize_validator_payloads(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path not in {"/reset", "/step", "/state", "/web/reset", "/web/step", "/web/state"}:
        return response

    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        return response

    body = b""
    async for chunk in response.body_iterator:
        body += chunk

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return Response(
            content=body,
            status_code=response.status_code,
            media_type=response.media_type,
            headers=dict(response.headers),
        )

    payload = _sanitize_http_payload(path, payload)
    headers = dict(response.headers)
    headers.pop("content-length", None)
    return JSONResponse(
        content=payload,
        status_code=response.status_code,
        headers=headers,
    )


def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the shipping environment server locally."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
