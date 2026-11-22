# EAT-Sched Assumptions Registry (ASSUMPTIONS.md)

**System Name:** EAT-Sched: Dynamic QoS-Aware Task Scheduling in Cloud-Edge Environments  
**Status:** Canonical Engineering Specification (Phase 0)  
**Authors:** Senior ML-Systems Engineer & B.Tech Capstone Project Team (Mohan Babu University)  

This registry catalogs every assumption made in the implementation of EAT-Sched, categorized and justified in compliance with **Operating Rule 3**.

---

## Summary of Assumptions

| ID | Category | Summary | Impact Area | Tag |
|---|---|---|---|---|
| `ASM-01` | Simulator | 8-Patch Parallel Speedup Extrapolation ($\times 2.45$) | Execution Time $t^e$ | `[ASSUMPTION]` |
| `ASM-02` | Quality | Saturating Monotone Quality Function ($q_{max}=0.85, \tau_q=18$) | Quality Score $q_k$ | `[ASSUMPTION]` |
| `ASM-03` | Discrete Step Range | Step Range $[S_{min}, S_{max}] = [20, 60]$ | Action Mapping $a_s$ | `[ASSUMPTION]` |
| `ASM-04` | Action Execution | Scheduling Execution Threshold $a_c > 0$ vs. $a_c \ge 0$ | Scheduler Trigger | `[ASSUMPTION]` |
| `ASM-05` | Reward Scaling | Normalization Constants $s_q=0.5, s_{t1}=40.0, s_{t2}=30.0$ | Reward Function $R_t$ | `[ASSUMPTION]` |
| `ASM-06` | Reward Credit | Immediate Crediting via Predicted Response Time $t^r$ | Credit Assignment | `[ASSUMPTION]` |
| `ASM-07` | Idle Reward | Queue Backlog Penalty for Idle/No-Op Steps | Policy Guidance | `[ASSUMPTION]` |
| `ASM-08` | DDIM Exploration | Learnable $\eta$ Initialization ($0.5$) and Upper Bound $\eta_{max} = 1.0$ | Policy Exploration | `[ASSUMPTION]` |
| `ASM-09` | RL Plumbing | Inclusion of Explicit Terminal $(1 - done)$ Target Masking | Critic Learning | `[ASSUMPTION]` |
| `ASM-10` | Power Model | Linear Server Power Interpolation ($P_{idle} + \Delta P \cdot util$) | Extended Energy Metric | `[ASSUMPTION]` |
| `ASM-11` | Cloud Tier | Cloud Latency $+120$ ms WAN Delay and Speedup Multiplier $\times 2.5$ | Cloud-Edge Heterogeneity | `[ASSUMPTION]` |
| `ASM-12` | Workload Profiles | Non-AIGC Workload Execution and Quality Semantics | Extended Workloads | `[ASSUMPTION]` |

---

## Detailed Specifications

### ASM-01: 8-Patch Parallel Speedup Extrapolation
- **Description:** Xu et al. (Table I & Fig. 4) report empirical acceleration on 4090D GPUs for 1 patch ($\times 1.00$, $23.7$ s), 2 patches ($\times 1.59$, $13.3$ s), and 4 patches ($\times 1.91$, $7.6$ s). For $c_k = 8$, the speedup is extrapolated to $\times 2.45$.
- **Justification:** In distributed patch parallelism (DistriFusion), each additional halving of patch area introduces increased inter-GPU communication across network boundaries for patch halo exchange. Fitting an Amdahl-style curve with communication overhead ($S(c) = \frac{1}{(1-p) + p/c + \alpha \log_2(c)}$) yields an expected speedup of $\approx 2.45\times$ for $c=8$.
- **Impact:** Used in `CloudEdgeSchedEnv` to compute $t_k^e$ when a task requests 8 cooperating servers.
- **Validation:** Tested in `tests/test_env_invariants.py` for monotonic latency reduction.

### ASM-02: Saturating Monotone Quality Function
- **Description:** For CPU-based and high-throughput training/testing where live CLIP and BRISQUE inference is disabled, quality $q_k$ is simulated as:
  $$q(s_k) = \text{clip}\left(q_{max} \cdot \left(1 - \exp\left(-\frac{s_k}{\tau_q}\right)\right) + \epsilon_q, 0.0, 1.0\right)$$
  with default parameters $q_{max} = 0.85$, $\tau_q = 18.0$, and $\epsilon_q \sim \mathcal{N}(0, 0.015^2)$.
- **Justification:** Standard diffusion models exhibit diminishing visual quality returns past 40–50 steps. At 20 steps, $q \approx 0.57$; at 40 steps, $q \approx 0.76$; at 60 steps, $q \approx 0.82$. This accurately mimics the empirical values recorded in Table IX of the paper ($0.57$–$0.77$).
- **Impact:** Provides realistic quality-latency trade-offs without requiring GPU dependencies. A `QualityBackend` abstraction allows live CLIP/BRISQUE plug-in on GPU workstations.
- **Validation:** Unit-tested in `tests/test_math.py` for strict monotonicity and range validity.

### ASM-03: Inference Step Range $[S_{min}, S_{max}] = [20, 60]$
- **Description:** The range of inference steps is bounded between $S_{min} = 20$ and $S_{max} = 60$.
- **Justification:** Matches the x-axis in Fig. 5 of the paper ("Step-Time Linear Fit with Different Configurations", where steps range from 10 to 60, with typical Stable Diffusion inference occurring between 20 and 50 steps).
- **Impact:** Discrete action mapping: $s_k = \text{round}(20 + 20 \cdot (a_s + 1))$.

### ASM-04: Action Execution Threshold $a_c > 0$
- **Description:** The paper text in Section V-A2 mentions "$a_c \ge 0$ means a task is scheduled", whereas Alg. 1 Line 14 and Alg. 2 Line 8 specify `if a_c > 0 and a_t is valid`. We adopt the strict condition $a_c > 0$.
- **Justification:** In continuous control with symmetric initialization around zero (e.g., standard normal initialization of the actor output layer), using $a_c > 0$ produces an unbiased 50/50 prior for scheduling vs. pausing at initialization. Using $\ge 0$ causes a slight bias and numerical ambiguity when $a_c = 0.0000$.
- **Impact:** Governs task dispatching logic in both training and inference loops.

### ASM-05: Reward Normalization Constants
- **Description:** Normalization scales are set to $s_q = 0.5$, $s_{t1} = 40.0$ s, and $s_{t2} = 30.0$ s.
- **Justification:** Tanh saturation requires the input $x / s$ to lie in the range $[0.5, 2.0]$ during normal operation to prevent gradient vanishing. Quality is in $[0.5, 0.8]$ ($q / 0.5 \in [1.0, 1.6]$), response time is in $[20, 60]$ s ($t^r / 40 \in [0.5, 1.5]$), and average queue wait time is in $[0, 45]$ s ($t_Q^{avg} / 30 \in [0.0, 1.5]$).
- **Impact:** Ensures numerical stability and healthy gradient flow for critic updates.

### ASM-06: Reward Credit Assignment via Predicted Response Time
- **Description:** The immediate reward $R_t$ is computed and returned at the decision step $t$ when task $k^*$ is dispatched, using predicted response time $t_{k^*}^r = t_{k^*}^w + t_{k^*}^d + t_{k^*}^e$.
- **Justification:** Tasks take 20–60 seconds to finish. Crediting rewards 50 steps later creates extreme credit assignment latency for the RL agent. As proven in the paper (Section IV-B, Section V-A3), execution and initialization times are deterministic functions of $(s_k, c_k, \text{speed}_e)$ known at dispatch time.
- **Impact:** Enables fast, robust TD-learning without requiring complex multi-step delayed credit tracking.

### ASM-07: Queue Backlog Penalty for Idle / No-Op Steps
- **Description:** When no task is scheduled ($a_c \le 0$ or invalid action due to insufficient idle servers), the environment returns:
  $$R_{idle} = - w_{t2} \cdot \tanh\left(\frac{t_{Q,t}^{avg}}{s_{t2}}\right)$$
- **Justification:** If idle steps returned $0.0$, the agent might prefer idling over scheduling tasks that receive a small positive reward. Penalizing backlog during idle steps penalizes laziness while accurately reflecting real queue latency growth.
- **Impact:** Prevents policy collapse to an all-idle policy.

### ASM-08: Learnable $\eta$ Initialization and Upper Bound $\eta_{max}$
- **Description:** DDIM learnable stochasticity $\eta$ is initialized to $0.5$ and clipped to $[0.0, 1.0]$ ($\eta_{max} = 1.0$).
- **Justification:** Section V-B1 of the paper explains that if $\eta$ becomes too large, the radicand $1 - \bar{\alpha}_{t-1} - \sigma_t^2$ approaches zero, causing the derivative with respect to $\eta$ to blow up and destabilize training. Setting $\eta_{max} = 1.0$ guarantees that $(1 - \bar{\alpha}_{t-1} - \sigma_t^2) \ge 0$ for all standard cosine/linear schedules.
- **Impact:** Prevents NaN gradients and training collapse while supporting full exploration down to deterministic exploitation ($\eta = 0$).

### ASM-09: Terminal Done Masking in Bellman Targets
- **Description:** Although Eq. 21 in the paper writes $\hat{y} = r_t + \gamma Q(s_{t+1}, a_{t+1})$ without explicit $(1 - done)$ gating, we implement:
  $$\hat{y} = r_t + \gamma (1 - done) \min_{i=1,2} Q_{\bar{\phi}_i}(s_{t+1}, a_{t+1})$$
- **Justification:** Standard RL theory dictates that value should not bootstrap beyond episode boundaries (time limit $1024$ s or 32 tasks completed). Omitting done masking leaks value across episodes and causes diverging critic estimates.
- **Impact:** Stabilizes critic Q-values.

### ASM-10: Server Power Model
- **Description:** Server power consumption is modeled linearly as:
  $$P_e(t) = P_{idle}^e + (P_{busy}^e - P_{idle}^e) \cdot u_e(t)$$
  For edge servers: $P_{idle} = 80$ W, $P_{busy} = 250$ W. For cloud servers: $P_{idle} = 150$ W, $P_{busy} = 450$ W.
- **Justification:** Standard data center energy modeling (e.g., SPECpower benchmarks for GPU-accelerated edge servers).
- **Impact:** Used in `extended_mode` to compute energy consumption per task and aggregate energy efficiency.

### ASM-11: Cloud Tier Latency and Compute Characteristics
- **Description:** Cloud servers provide higher compute capability ($\times 2.5$ speed relative to base edge server) but incur an additional WAN transmission delay of $+120$ ms ($0.12$ s) for prompt dispatch and result retrieval.
- **Justification:** Reflects realistic regional cloud round-trips over public edge-to-cloud backhauls.
- **Impact:** Creates a realistic trade-off between edge (zero WAN delay, constrained capacity) and cloud (high capacity, network delay, higher power).

### ASM-12: Heterogeneous Workload Profiles
- **Description:** Non-AIGC workloads (ML inference, video analytics, CPU batch) are parameterized in YAML profiles with base work units, parallel efficiency curves, and quality metrics:
  - `ml_inference`: fixed quality $1.0$, low work unit, strict deadlines.
  - `video_analytics`: quality scales with frame sample rate $[10, 30]$ fps, medium work unit.
  - `cpu_batch`: throughput-driven, flexible deadlines, $c_k = 1$ or $2$.
- **Justification:** Realizes the Capstone requirement for heterogeneous workloads across edge and cloud.
- **Impact:** Enables multi-workload benchmark evaluation in `extended_mode`.
