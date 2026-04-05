# RecallTrace OpenEnv Environment

A fully offline OpenEnv environment simulating **product recall traceability and containment** across a supply-chain network.

---

## Problem Statement

In real-world product recalls, organizations must:

* Identify **affected lot codes**
* Trace product movement across **warehouses and stores**
* Handle **relabeling/repacking cases**
* Quarantine only **unsafe inventory**
* Avoid blocking **safe stock**

Mistakes can lead to:

* Financial loss (over-quarantine)
* Public safety risks (missed contamination)

---

## Objective

Build an AI-compatible environment where an agent:

* Traces contaminated inventory
* Identifies affected nodes
* Takes precise containment actions
* Minimizes unnecessary disruption

---

## Environment Overview

### Core Components

* **Env Core**

  * `reset()` → initializes recall scenario
  * `step(action)` → processes actions
  * `state()` → returns full internal state

* **Scenario Generator**

  * Shipment graph
  * Inventory distribution
  * Lot mappings
  * Contamination source

* **Action Handler**

  * inspect_node
  * trace_lot
  * quarantine
  * notify
  * finalize

* **Ground Truth (Hidden Oracle)**

  * True contaminated lots
  * Affected nodes
  * Correct quantities

* **Grader + Reward System**

  * Deterministic scoring (0.0–1.0)
  * Partial rewards and penalties

---

## Action Space

| Action                           | Description                  |
| -------------------------------- | ---------------------------- |
| inspect_node(node_id)            | View inventory and shipments |
| trace_lot(lot_id)                | Trace lot across network     |
| quarantine(node_id, lot_id, qty) | Isolate affected inventory   |
| notify(node_id)                  | Send alert                   |
| finalize()                       | Submit containment plan      |

---

## Observation Space

* Recall notice
* Inventory snapshot
* Shipment graph (partial)
* Action history

---

## Tasks

### 🔹 Task 1 — Direct Recall (Easy)

* Single contaminated lot
* Simple distribution

### 🔹 Task 2 — Relabeled Inventory (Medium)

* Lot transformed or relabeled

### 🔹 Task 3 — Mixed Shipments (Hard)

* Safe + unsafe inventory mixed

---

## Reward Design

### Positive Rewards

* Correct tracing
* Accurate quarantine
* Proper notifications

### Negative Rewards

* Over-quarantine
* Missed contamination
* Unnecessary actions

### Final Bonus

* Complete and efficient containment

---

## Project Structure

```
recalltrace-openenv/
│
├── env/                # Environment core
├── scenario/           # Scenario generation
├── grader/             # Grading + reward logic
├── inference/          # Baseline agent
├── config/             # OpenEnv config
├── docker/             # Docker setup
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone Repository

```
git clone <your-repo-link>
cd recalltrace-openenv
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run Environment

```
uv run server
```

---

## Run Inference

```
python inference/inference.py
```

---

## Docker Setup

```
docker build -t recalltrace .
docker run recalltrace
```

---

## OpenEnv Compliance

* Implements `reset()`, `step()`, `state()`
* Uses typed models (Pydantic)
* Includes `openenv.yaml`
* Passes `openenv validate`

---

## Evaluation

* Deterministic grading
* Score range: **0.0 → 1.0**
* Based on:

  * Accuracy
  * Completeness
  * Efficiency

---

## Key Features

* Real-world industrial problem
* Multi-step reasoning (graph + logic)
* Offline and reproducible
* Easy to evaluate and benchmark

---

## Goal

Build a **real-world, deterministic, OpenEnv-compliant environment** that enables AI agents to solve complex supply-chain recall problems efficiently.

---

## Team

* Shamanth MS
* P G Ayush Rai
* Shreya B J

---

## Submission

* Hugging Face Space: *(Add link here)*

---
