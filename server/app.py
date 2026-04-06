# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application entrypoint for the shipping environment."""

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


app = create_app(
    ShippingEnvironment,
    ShippingAction,
    ShippingObservation,
    env_name="shipping_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the shipping environment server locally."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
