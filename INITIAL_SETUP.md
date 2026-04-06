# Initial Setup for `shipping_env`

This folder now contains a working **maritime OpenEnv MVP** under the stable Python package path `my_env/`.
The user-facing environment name is `shipping_env`.

## 1. Local prerequisites

- Python `3.11+`
- Docker Desktop
- Git
- Optional: `uv`

Check them:

```bash
python3.11 --version
git --version
docker --version
```

## 2. Create and activate a virtual environment

From the repo root:

```bash
cd /path/to/martime-supply-chain
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

## 3. Install dependencies

### Option A: `uv`

```bash
cd my_env
uv sync
```

### Option B: `pip`

```bash
cd my_env
pip install --upgrade pip
pip install -e .
```

## 4. Run the environment locally

From `my_env/`:

```bash
python -m my_env.server.app
```

Server endpoints:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`
- `POST /reset`
- `POST /step`
- `GET /state`

## 5. Quick validation

In another terminal:

```bash
curl http://localhost:8000/health
```

If the OpenEnv CLI is available:

```bash
openenv validate --directory .
```

## 6. Build and run the Docker image

From `my_env/`:

```bash
docker build -t shipping-env:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 shipping-env:latest
```

## 7. Included task set

The current MVP already includes three seeded tasks:

1. `easy_rotterdam_watch`
2. `medium_asia_reroute`
3. `hard_north_sea_allocation`

Each task supports:

- vessel inspection
- port inspection
- congestion history inspection
- `SARIMAX` / `ETS` forecast comparison
- route option review
- deterministic final grading

## 8. Included files

- `README.md`
- `openenv.yaml`
- `pyproject.toml`
- `models.py`
- `client.py`
- `inference.py`
- `scenario_config.json`
- `scenario_data.py`
- `server/app.py`
- `server/my_env_environment.py`
- `server/Dockerfile`

## 9. Recommended next improvements

If you keep iterating before submission, the highest-value next steps are:

1. Add a simple web demo screenshot or GIF
2. Add lightweight tests for `submit_plan`
3. Add Hugging Face Space deployment metadata
4. Replace heuristic inference with an OpenAI-client planner if required by your submission flow
