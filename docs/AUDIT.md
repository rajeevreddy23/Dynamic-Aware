# EAT-Sched Self-Audit & Traceability Matrix (AUDIT.md)

**System Name:** EAT-Sched: Dynamic QoS-Aware Task Scheduling in Cloud-Edge Environments  
**Status:** Initialized (Phase 0 Scaffold)  
**Authors:** Senior ML-Systems Engineer & B.Tech Capstone Project Team (Mohan Babu University)  

---

## 1. Requirements Compliance Matrix (Sections C–G)

| Requirement | Section | Target Module / File | Verification Test | Status |
|---|---|---|---|---|
| Task Model & Sampling ($c_k \in \{1,2,4,8\}$, arrivals) | C1 | `src/eatsched/env/tasks.py` | `tests/test_env_invariants.py` | Scaffolded (P0) |
| Parallel Execution Time ($t^e$ & gang constraint) | C1 | `src/eatsched/env/tasks.py` | `tests/test_env_invariants.py` | Scaffolded (P0) |
| Three-tier Init Delay (Cold 30s, Reconfig 1.76s, Reuse 0s) | C1 | `src/eatsched/env/cluster.py` | `tests/test_env_invariants.py` | Scaffolded (P0) |
| Saturating Quality & `QualityBackend` Interface | C1 | `src/eatsched/env/quality.py` | `tests/test_math.py` | Scaffolded (P0) |
| Server Cluster State & Presets (edge32/48/64) | C2 | `src/eatsched/env/cluster.py` | `tests/test_env_invariants.py` | Scaffolded (P0) |
| Safe No-Op Invariant on Invalid Action | C3 | `src/eatsched/env/sched_env.py` | `tests/test_env_invariants.py` | Scaffolded (P0) |
| Deterministic Greedy Server Selector (Alg. 1) | C4 | `src/eatsched/env/selector.py` | `tests/test_env_invariants.py` | Scaffolded (P0) |
| Metrics (Response Time, Quality, Utility $U$) | C5 | `src/eatsched/env/metrics.py` | `tests/test_metrics.py` | Scaffolded (P0) |
| MDP State Representation & Key-Padding Masking | D1 | `src/eatsched/env/sched_env.py` | `tests/test_env_invariants.py` | Scaffolded (P0) |
| Continuous Action Mapping ($a_c > 0$, steps, task) | D1 | `src/eatsched/agents/eat.py` | `tests/test_math.py` | Scaffolded (P0) |
| Normalized Eq. 9 Immediate Reward with $b$ bias | D1 | `src/eatsched/env/rewards.py` | `tests/test_env_invariants.py` | Scaffolded (P0) |
| Cross-Attention Layer (Permutation invariant over servers) | D2 | `src/eatsched/agents/cross_attention.py` | `tests/test_attention.py` | Scaffolded (P0) |
| DDIM Diffusion Policy with Learnable $\eta$ | D3 | `src/eatsched/agents/ddim_policy.py` | `tests/test_ddim.py` | Scaffolded (P0) |
| Double Critic Networks & Target Soft Updates | D4 | `src/eatsched/agents/critics.py` | `tests/test_rl_plumbing.py` | Scaffolded (P0) |
| Terminal Done Masking in Target $y$ | D4 | `src/eatsched/agents/eat.py` | `tests/test_rl_plumbing.py` | Scaffolded (P0) |
| Extended Mode: Priority, Deadlines, $q_{min}$, Workloads | E1-E4 | `src/eatsched/env/sched_env.py` | `tests/test_extended_mode.py` | Scaffolded (P0) |
| Extended Reward: Priority Deadline Miss & Energy | E5 | `src/eatsched/env/rewards.py` | `tests/test_extended_mode.py` | Scaffolded (P0) |
| Dynamic Adaptation (Drift, Faults, Bursts) | E6 | `src/eatsched/experiments/adaptation.py` | `tests/test_extended_mode.py` | Scaffolded (P0) |
| Non-Learned Baselines (Random, FIFO, Greedy, EDF, P-FIFO) | F | `src/eatsched/agents/heuristics.py` | `tests/test_baselines.py` | Scaffolded (P0) |
| Learned Baselines (DPPO, D2SAC, SAC, PPO, TD3, DDPG) | F | `src/eatsched/agents/baselines/` | `tests/test_baselines.py` | Scaffolded (P0) |
| Ablation Variants (EAT-D, EAT-CA, D2SAC+SA) | F | `src/eatsched/agents/ablations/` | `tests/test_baselines.py` | Scaffolded (P0) |
| FastAPI Server with Health, Schedulers, Simulate, WS | G | `api/main.py` | `tests/test_api.py` | Scaffolded (P0) |
| Next.js 15 Interactive Dashboard | G | `web/` | Web lint & build test | Scaffolded (P0) |

---

## 2. Known Limitations & Technical Constraints

1. **Simulated Diffusion Quality**: In the CPU environment, quality is evaluated through the calibrated saturating function $q(s_k)$ (`ASM-02`). Workstations equipped with CUDA GPUs can seamlessly switch to the live `DistriSDPipeline` and real CLIP/BRISQUE evaluators.
2. **Simulated WAN Latency**: Cloud latency in the cloud-edge tier is simulated using deterministic packet transmission and jitter models (`ASM-11`) rather than cross-continental physical networks.
3. **Hardware Acceleration**: All components are validated to run natively on standard x86 CPU architectures without requiring proprietary GPU drivers.
