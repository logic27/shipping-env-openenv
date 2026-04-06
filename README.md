---
title: Shipping Env
emoji: 🚢
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - logistics
  - maritime
---

# shipping_env

`shipping_env` is a compact OpenEnv environment for **forecast-assisted maritime disruption planning**.
It is built for Round 1 hackathon evaluation: deterministic, offline, Docker-friendly, and easy to grade.

The environment simulates a shipping control tower workflow:

- inspect a vessel
- inspect candidate ports
- inspect congestion history
- compare `SARIMAX` and `ETS`-style forecasts
- review route options
- submit a final arrival plan

The current MVP uses seeded data inspired by AIS and port operations rather than live external APIs. That keeps evaluation reproducible and avoids dependency failures during judging.

## What Makes This Judge-Friendly

- Real-world domain: maritime logistics and port disruption planning
- Typed `Action` and `Observation` models
- Deterministic scoring from `0.0` to `1.0`
- Three seeded tasks: `easy`, `medium`, `hard`
- Partial reward shaping for evidence gathering
- No live API keys or network access needed at runtime

## Task Set

### `easy_rotterdam_watch`

Protect a pharma vessel arrival into Rotterdam.

- choose the better congestion model
- compare Rotterdam vs Antwerp
- pick a safe service speed

### `medium_asia_reroute`

Reroute a container vessel around a Shanghai congestion spike.

- compare Shanghai, Ningbo, and Busan
- balance service deadline vs fuel burn
- use the more credible forecast family

### `hard_north_sea_allocation`

Replan a tanker arrival during a North Sea storm disruption.

- compare Rotterdam, Antwerp, and Wilhelmshaven
- use forecast quality plus weather penalty
- pick a speed that balances risk against fuel cost

## Action Schema

The environment exposes one typed action model: `ShippingAction`.

Core commands:

- `list_tasks`
- `load_task`
- `inspect_vessel`
- `inspect_port`
- `inspect_congestion_history`
- `inspect_route_options`
- `inspect_forecast`
- `submit_plan`

`submit_plan` expects:

- `forecast_model`
- `target_port_id`
- `service_speed_knots`

## Observation Schema

Each step returns `ShippingObservation` with:

- `summary`
- `active_task_id`
- `phase`
- `available_commands`
- `artifacts`
- `metrics`

The final submission step also returns:

- score
- predicted business cost index
- realized business cost index
- score breakdown in metadata

## Quick Start

### 1. Install dependencies

From the environment directory:

```bash
cd /Users/anmolkoul07/hail-mary/my_env
source ../.venv/bin/activate
pip install -e .
```

If you prefer `uv`:

```bash
uv sync
```

### 2. Run locally

```bash
python -m my_env.server.app
```

Then open:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

### 3. Smoke-test with the Python client

```python
from my_env import ShippingAction, ShippingEnv

with ShippingEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(result.observation.summary)

    result = env.step(ShippingAction(command="list_tasks"))
    print(result.observation.artifacts)

    env.step(ShippingAction(command="load_task", task_id="easy_rotterdam_watch"))
    env.step(ShippingAction(command="inspect_vessel"))
    env.step(ShippingAction(command="inspect_congestion_history", port_id="rotterdam"))
    env.step(
        ShippingAction(
            command="inspect_forecast",
            port_id="rotterdam",
            forecast_model="sarimax",
        )
    )
    env.step(ShippingAction(command="inspect_route_options"))

    final = env.step(
        ShippingAction(
            command="submit_plan",
            forecast_model="sarimax",
            target_port_id="rotterdam",
            service_speed_knots=12,
            rationale="Stable outage-aware forecast and low-risk cold-chain arrival.",
        )
    )
    print(final.reward, final.observation.metrics)
```

## Inference Baseline

A root-level [`inference.py`](/Users/anmolkoul07/hail-mary/my_env/inference.py) script is included.

It can:

- connect to a running API via `API_BASE_URL`
- or solve tasks directly in-process as a local heuristic baseline

The baseline:

- inspects each task
- queries vessel and route data
- compares both forecast families for each candidate port
- selects the plan with the lowest seeded business cost proxy

## Docker

Build the image from the environment directory:

```bash
docker build -t shipping-env:latest -f server/Dockerfile .
```

Run it:

```bash
docker run --rm -p 8000:8000 shipping-env:latest
```

## Project Structure

```text
my_env/
├── README.md
├── openenv.yaml
├── pyproject.toml
├── client.py
├── inference.py
├── models.py
├── scenario_config.json
├── scenario_data.py
└── server/
    ├── app.py
    ├── my_env_environment.py
    └── Dockerfile
```

## Notes

- The Python package path is still `my_env` to keep the scaffold stable.
- User-facing naming is `shipping_env`.
- The seeded forecasts are intentionally lightweight proxies so the environment stays deterministic and fast.
