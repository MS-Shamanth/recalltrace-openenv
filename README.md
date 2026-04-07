---
title: RecallTrace OpenEnv
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# RecallTrace OpenEnv

RecallTrace is a submission-grade OpenEnv environment for **product recall traceability and precision containment**. It models a real workflow that quality, operations, and supply-chain teams actually perform during a food or pharmaceutical recall:

- map the affected batch across a distribution network,
- follow repacked or relabeled inventory back to the source lot,
- inspect local evidence at each node,
- quarantine exactly the unsafe quantity,
- notify the right downstream stakeholders,
- close the incident with minimal business disruption.

This is not a toy. The hard task models a realistic and expensive failure mode: contaminated stock gets mixed with safe stock during cross-docking, so the agent must contain only the unsafe portion instead of blanket-blocking everything.

## Why This Environment Matters

Recall-response benchmarks are rare, but they are a strong fit for agent evaluation because they combine:

- graph reasoning,
- partial observability,
- evidence gathering,
- precision actions under cost pressure,
- deterministic grading.

That makes RecallTrace useful both for RL-style environments and for benchmarking frontier LLM agents on operational decision-making.

## Tasks

### 1. `phase1_direct_recall` - Easy

- One contaminated lot (`LotA`)
- Straightforward warehouse -> store distribution
- Goal: find every holder of the original lot and quarantine all unsafe units

### 2. `phase2_relabel_recall` - Medium

- The original contaminated lot is repacked and relabeled (`LotA_R1`, `LotA_R2`)
- Goal: trace the lineage from the source lot to all derived labels and quarantine every affected label precisely

### 3. `phase3_mixed_shipments` - Hard

- Contaminated inventory is mixed with safe stock into `LotBlend`
- Inspection reveals only part of the lot is unsafe at each node
- Goal: quarantine the unsafe quantity only, avoid over-blocking safe stock, and still notify all affected nodes

## Action Space

| Action | Parameters | Meaning |
| --- | --- | --- |
| `inspect_node` | `node_id` | Inspect a warehouse/store/cross-dock node and reveal local evidence |
| `trace_lot` | `lot_id` | Trace the root lot across relabels and downstream movement |
| `quarantine` | `node_id`, `lot_id`, `quantity` | Move inventory from active stock to quarantine |
| `notify` | `node_id` or `all` | Send recall notifications to one or all nodes |
| `finalize` | none | End the episode and compute the deterministic score |

## Observation Space

Each `reset()` and `step()` returns a typed `RecallObservation` model with:

- `task_id`
- `phase`
- `recall_notice`
- `available_actions`
- `inventory`
- `discovered_shipments`
- `inspected_nodes`
- `inspection_results`
- `trace_results`
- `notified_nodes`
- `quarantined_inventory`
- `history`
- `steps_taken`
- `remaining_step_budget`

## Reward Design

Trajectory rewards are shaped so the agent gets useful signal throughout the episode:

- positive reward for tracing the correct recall lineage,
- positive reward for inspecting nodes and gathering evidence,
- strong positive reward for exact quarantine,
- smaller positive reward for notifying affected stakeholders,
- negative reward for repeated low-value actions,
- negative reward for quarantining safe stock,
- timeout penalty if the agent exhausts the step budget.

Final episode score is deterministic in `[0.0, 1.0]` and combines:

- quarantine precision,
- notification coverage,
- investigation coverage,
- efficiency.

## Project Structure

```text
recalltrace-openenv/
|-- env/
|   |-- env.py
|   |-- models.py
|   `-- __init__.py
|-- grader/
|   `-- grader.py
|-- inference/
|   |-- inference.py
|   `-- policy.py
|-- scenario/
|   `-- scenario.py
|-- tests/
|   `-- test_env.py
|-- config/
|   `-- openenv.yaml
|-- Dockerfile
|-- inference.py
|-- openenv.yaml
|-- requirements.txt
|-- server.py
`-- README.md
```

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the API server locally

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Run the baseline inference script

```bash
python inference.py
```

### Run unit tests

```bash
python -m unittest discover -s tests -v
```

## OpenEnv Interface

The environment implements:

- typed Pydantic `RecallAction`, `RecallObservation`, and `RewardSignal` models,
- `reset()` -> initial typed observation,
- `step(action)` -> `(observation, reward, done, info)`,
- `state()` -> full typed internal snapshot,
- `openenv.yaml` metadata,
- HTTP endpoints at `/reset`, `/step`, `/state`, `/tasks`, `/health`.

## Baseline Inference

The root `inference.py` file is the submission entrypoint.

It:

- uses the OpenAI client when `OPENAI_API_KEY` or `HF_TOKEN` is configured,
- falls back to a deterministic heuristic policy when no model credentials are present,
- emits structured stdout logs in `[START]`, `[STEP]`, `[END]` format,
- evaluates all 3 tasks,
- reports reproducible scores.

Environment variables:

- `API_BASE_URL` - API endpoint for model inference
- `MODEL_NAME` - model identifier
- `OPENAI_API_KEY` or `HF_TOKEN` - credential for the OpenAI-compatible endpoint

## Baseline Scores

Current deterministic heuristic baseline:

- `phase1_direct_recall`: ~0.9700
- `phase2_relabel_recall`: ~0.9643
- `phase3_mixed_shipments`: ~0.9688
- average: ~0.9677

These scores are reproducible because the environment is deterministic and the fallback heuristic is deterministic.

## Docker

Build and run locally:

```bash
docker build -t recalltrace .
docker run -p 7860:7860 recalltrace
```

## Hugging Face Spaces

This repo is container-ready for a Docker-based HF Space.

Recommended launch command is already encoded in the root `Dockerfile`:

```text
uvicorn server:app --host 0.0.0.0 --port 7860
```

## Deterministic Grading

Programmatic graders live in `grader/grader.py` and can:

- replay a full action plan against any task,
- compute a deterministic `TaskGrade`,
- validate that final scores remain within `[0.0, 1.0]`.

## What Makes It Novel

- uncommon real-world domain for OpenEnv,
- graph tracing plus evidence collection,
- relabel lineage reasoning,
- mixed-lot precision containment,
- explicit tradeoff between safety and operational disruption.
