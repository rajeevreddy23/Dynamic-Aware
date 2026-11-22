# EAT-Sched Experimental Results Log (RESULTS.md)

**System Name:** EAT-Sched: Dynamic QoS-Aware Task Scheduling in Cloud-Edge Environments  
**Status:** Initialized (Phase 0 Scaffold)  
**Author:** ML-Systems Engineering Team  
**Compliance Rule:** Never fabricate results. All numbers recorded below originate from executable scripts executed with documented seeds and configuration files stored in `results/`.

---

## 1. Experimental Protocol & Reproducibility Matrix

All experiments are executed across a minimum of 5 seeds (`seeds: [42, 43, 44, 45, 46]`). Summary tables report **Mean ± Standard Deviation**.

| Exp ID | Configuration | Schedulers Evaluated | Target Metric | Status |
|---|---|---|---|---|
| `EXP-01` | `edge32` (arrival 0.3/s) | EAT, DPPO, D2SAC, SAC, PPO, TD3, DDPG, Greedy, FIFO, Random | Training Reward & Convergence | Pending Phase 4 |
| `EXP-02` | `edge32, edge48, edge64` | All Baselines | Normalized Reward Comparison | Pending Phase 4 |
| `EXP-03` | $w_q \in \{0.4, 0.5, 0.6\}$ | EAT vs. Baselines | Pareto Latency vs. Quality Frontier | Pending Phase 4 |
| `EXP-04` | Arrival Sweep ($\lambda \in [0.2, 0.7]$) | EAT vs. Baselines | Latency & Quality Tables | Pending Phase 4 |
| `EXP-05` | Diffusion Steps $T \in [2, 10]$ | EAT | Denoising Step Sensitivity & Overhead | Pending Phase 4 |
| `EXP-06` | `edge32` Ablations | EAT, EAT-D, EAT-CA, D2SAC+SA, SAC | Component Attribution | Pending Phase 4 |
| `EXP-07` | `cloud_edge` Extended Mode | EAT vs. EDF, Priority-FIFO, Greedy | Deadline Met %, QoS %, Energy (kJ) | Pending Phase 4 |
| `EXP-08` | Initialization Overhead Study | EAT vs. Traditional (No Reuse) | Cold vs. Reconfig vs. Reuse Breakdown | Pending Phase 4 |

---

## 2. Reference Sanity Comparison (Xu et al. Paper Reported Values)

*Note: For sanity reference only; experimental values will be populated directly from actual script executions.*

- **Paper Table IX (Quality, w_q = 0.5):**
  - `edge32` ($\lambda = 0.3$): EAT: $0.70 \pm 0.02$, D2SAC: $0.65 \pm 0.01$, SAC: $0.53 \pm 0.05$, Greedy: $0.74 \pm 0.01$, FIFO: $0.57 \pm 0.00$, Random: $0.57 \pm 0.01$.
- **Paper Table X (Latency in seconds, w_q = 0.5):**
  - `edge32` ($\lambda = 0.3$): EAT: $34.1 \pm 1.02$, D2SAC: $34.9 \pm 0.30$, SAC: $32.9 \pm 0.68$, Greedy: $57.8 \pm 0.30$, FIFO: $41.2 \pm 0.33$, Random: $61.8 \pm 0.31$.
- **Reported Ranking:** $\text{EAT} > \text{DPPO} \approx \text{D2SAC} > \text{SAC} \approx \text{PPO} > \text{FIFO} \approx \text{Greedy} \approx \text{TD3} > \text{DDPG}$.
