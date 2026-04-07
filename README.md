---
title: RecallTrace OpenEnv
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
---

# 🚀 RecallTrace OpenEnv 

RecallTrace is a **real-world AI environment** designed for **product recall tracing and precision containment**.

It simulates how companies handle:
- contaminated product recalls
- supply chain tracing
- selective quarantine decisions

This environment evaluates **agent reasoning + decision-making**, not just correctness.

---

# 🧠 What This Environment Does

Given a recall notice (e.g., *"Lot A is contaminated"*), the agent must:

1. Trace where the product went  
2. Identify affected nodes (warehouses, stores)  
3. Handle relabeling / transformations  
4. Quarantine **only unsafe inventory**  
5. Avoid blocking safe stock  
6. Notify affected entities  
7. Finalize with correct containment  

---

# 🎯 Why This Is Important

This is a **real industry problem** seen in:
- food recalls  
- pharma defects  
- logistics failures  

Challenges include:
- Graph traversal  
- Partial observability  
- Lot transformations  
- Mixed inventory reasoning  
- Precision decision-making  

---

# 🧩 Tasks (Scenarios)

## 🔹 Easy — Direct Recall
- Single contaminated lot  
- Straight supply chain  
- Goal: trace and quarantine correctly  

---

## 🔹 Medium — Relabeled Inventory
- Lot gets renamed (LotA → LotA1)  
- Goal: track transformations and quarantine  

---

## 🔹 Hard — Mixed Inventory
- Contaminated + safe stock mixed  
- Goal: isolate unsafe quantity **without over-blocking**  

---

# ⚙️ Action Space

| Action | Description |
|------|------------|
| inspect_node | View inventory at a node |
| trace_lot | Follow product lineage |
| quarantine | Block unsafe stock |
| notify | Inform affected nodes |
| finalize | End task |

---

# 📦 Observation Structure

Each step returns:

- recall_notice  
- inventory  
- action history  
- trace results  
- inspection data  

---

# 🏆 Reward & Grading

### Reward System
- + Correct tracing  
- + Correct quarantine  
- + Correct notification  
- − Wrong node  
- − Over-quarantine  
- − Missed unsafe stock  

---

### Final Score
Range: **0.0 → 1.0**

Based on:
- accuracy  
- precision  
- efficiency  

---

# 🧱 Project Structure

```bash
recalltrace-openenv/
│
├── env/                # Environment logic
│   ├── env.py
│   └── __init__.py
│
├── scenario/           # Scenario generation
│   └── scenario.py
│
├── grader/             # Evaluation + reward
│   └── grader.py
│
├── inference/          # Agent simulation
│   └── inference.py
│
├── config/
│   └── openenv.yaml
│
├── Dockerfile
├── requirements.txt
├── README.md
