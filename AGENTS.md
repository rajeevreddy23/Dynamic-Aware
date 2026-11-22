# AGENTS.md

## Project Mission
EAT-Sched: Dynamic QoS-Aware Task Scheduling in Cloud-Edge Environments using Attention-Guided Diffusion Reinforcement Learning.

This project implements an end-to-end discrete-event reinforcement learning scheduling system based on:
1. **[PAPER]** Z. Xu et al., "EAT: QoS-Aware Edge-Collaborative AIGC Task Scheduling via Attention-Guided Diffusion RL", *IEEE Transactions on Mobile Computing*, vol. 25, no. 7, pp. 9980-9996, July 2026.
2. **[ABSTRACT]** Mohan Babu University B.Tech Capstone Abstract, "Dynamic-Aware Task Scheduling in Cloud-Edge Environments" (2026-2027), adding QoS parameters (priority, deadlines, quality requirements), heterogeneous workloads, cloud-edge tiers, energy modeling, and dynamic adaptation.

## Operating Principles for Agents and Engineers
1. **Correctness over Speed**: Every implementation must be backed by unit tests verifying mathematical and structural invariants.
2. **Empirical Honesty**: Results must never be simulated or fabricated. All numbers reported in documentation, web dashboards, or viva presentations must originate from executable scripts saved in `results/` with deterministic seeds and recorded configs.
3. **Traceability**: Every design choice must be tagged in `docs/SPEC.md` and `docs/ASSUMPTIONS.md` as `[PAPER]`, `[ABSTRACT]`, or `[ASSUMPTION]`.
4. **Zero Mocks / No Stubs**: Production-grade code only. All components must run on CPU natively, with GPU acceleration enabled via `--device cuda`.
5. **Phase-Gated Development**:
   - P0: Scaffold, Specifications (`SPEC.md`, `ASSUMPTIONS.md`), and CI verification.
   - P1: Discrete-event simulation environment (`CloudEdgeSchedEnv`), metrics, and energy models.
   - P2: Non-learned baselines (FIFO, Greedy, Random, EDF, Priority-FIFO) and evaluation harness.
   - P3: Attention-guided DDIM diffusion RL agent (`EAT`) and 7 learned baselines (DPPO, D2SAC, SAC, PPO, TD3, DDPG, D2SAC+SA).
   - P4: Multi-seed experiments across cluster scales, QoS sensitivities, ablations, and dynamic drift.
   - P5: Production FastAPI backend with deterministic replay and streaming simulation.
   - P6: Interactive Next.js 15 dashboard (Gantt chart, Pareto analysis, adaptive workloads).
   - P7: Containerization, deployment verification, comprehensive report, and viva demo script.

## Coding and Contribution Standards
- All code must conform to PEP 8 enforced via `ruff` and be strictly typed with `mypy`.
- Tests must be executed with `pytest` and pass 100% without skipping or weakening assertions.
- Modules must be structured cleanly under `src/eatsched/`.
