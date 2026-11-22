# EAT-Sched: Dynamic QoS-Aware Task Scheduling in Cloud-Edge Environments

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Final-Year B.Tech Capstone Project**  
> **Institution:** Mohan Babu University (MBU), Tirupati  
> **Department:** Computer Science and Engineering  
> **Base Paper:** Z. Xu et al., *"EAT: QoS-Aware Edge-Collaborative AIGC Task Scheduling via Attention-Guided Diffusion RL"*, IEEE Transactions on Mobile Computing (TMC), vol. 25, no. 7, pp. 9980-9996, July 2026. DOI: [10.1109/TMC.2026.3656318](https://doi.org/10.1109/TMC.2026.3656318).

---

## 📌 Overview

**EAT-Sched** is an end-to-end, high-performance reinforcement learning scheduler for heterogeneous cloud-edge environments. It resolves the fundamental conflict between generation quality, inference delay, and initialization overhead by coupling:
1. **Asymmetric Cross-Attention Feature Extraction**: Encodes dynamic task queues as queries and heterogeneous server clusters as keys/values, providing permutation-invariant representations of server states.
2. **DDIM Diffusion Policy Network with Learnable Exploration**: Employs a multi-modal continuous diffusion actor where the DDIM stochasticity parameter $\eta$ is explicitly trained via policy gradients, eliminating the intractability of diffusion policy entropy computation.
3. **Model & Process Group Decoupling**: Minimizes cold starts (30s $\to$ 0.5–2s) and eliminates overhead via instant gang reuse (0s).
4. **Dual Operational Modes**:
   - `paper_mode`: Faithful reproduction of the IEEE TMC 2026 EAT paper MDP and benchmarks.
   - `extended_mode`: Realizes the Mohan Babu University Capstone extensions, integrating multi-priority QoS, hard deadlines, minimum quality thresholds, multi-tier cloud-edge topologies, energy-aware rewards, and dynamic arrival/capacity drift adaptation.

---

## 🏗️ Repository Layout

```text
eat-sched/
├── AGENTS.md                   # Multi-agent specifications & coding rules
├── README.md                   # Project overview, setup, and viva guide
├── Makefile                    # Single-command build & test workflows
├── docker-compose.yml          # Full-stack container orchestration
├── pyproject.toml              # Build metadata, linter & test configuration
├── requirements.txt            # Pinned dependencies
├── docs/
│   ├── SPEC.md                 # Formal mathematical & architectural specification
│   ├── ASSUMPTIONS.md          # Documented assumptions with justifications
│   ├── RESULTS.md              # Empirical experiment records (mean ± std)
│   └── AUDIT.md                # Traceability matrix & compliance audit
├── configs/
│   ├── clusters/               # edge32, edge48, edge64, cloud_edge
│   ├── workloads/              # AIGC, ML inference, video analytics, CPU batch
│   ├── experiments/            # Hyperparameters and experiment configurations
│   └── agents/                 # Model configs for EAT, D2SAC, DPPO, SAC, PPO, etc.
├── src/eatsched/
│   ├── env/                    # Gymnasium discrete-event simulator & metrics
│   ├── agents/                 # EAT (attention + DDIM) and baselines
│   ├── experiments/            # Multi-seed evaluation & ablation runners
│   ├── train.py                # Actor-critic training pipeline
│   ├── evaluate.py             # Evaluation harness
│   └── utils/                  # Logging, math utilities, and config loaders
├── api/                        # FastAPI REST & WebSocket streaming server
├── web/                        # Next.js 15 interactive scheduling dashboard
├── tests/                      # Comprehensive unit and integration test suite
├── results/                    # Validated experimental logs and JSON metrics
└── checkpoints/                # Model weights with configuration hashes
```

---

## ⚡ Quickstart

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/eat-sched/eat-sched.git
cd eat-sched

# Install python dependencies in editable mode
pip install -e .
```

### 2. Run Tests & Validation
```bash
make test
```

### 3. Quick Smoke Training (CPU)
```bash
make train-smoke
```

### 4. Evaluate Schedulers
```bash
make eval
```

### 5. Launch Full Stack
```bash
# Terminal 1: Backend API
make api

# Terminal 2: Web Dashboard
make web
```
Visit `http://localhost:3000` to interact with the scheduling Gantt visualization and comparative analytics.
