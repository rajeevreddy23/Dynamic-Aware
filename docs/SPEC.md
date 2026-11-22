# EAT-Sched Technical Specification (SPEC.md)

**System Name:** EAT-Sched: Dynamic QoS-Aware Task Scheduling in Cloud-Edge Environments  
**Status:** Canonical Engineering Specification (Phase 0)  
**Authors:** Senior ML-Systems Engineer & B.Tech Capstone Project Team (Mohan Babu University)  
**Target Publication Reference:** IEEE TMC 2026 (Xu et al., DOI: 10.1109/TMC.2026.3656318) `[PAPER]`  
**Capstone Reference:** Mohan Babu University Abstract (2026–2027) `[ABSTRACT]`  

---

## 1. System Overview & Operational Modes

EAT-Sched provides an intelligent, reinforcement learning-driven scheduler for distributed AIGC and heterogeneous computational tasks across edge and cloud servers.

To reconcile the base paper's formulation with the capstone project abstract requirements, the system supports two first-class operational modes:

1. **`paper_mode`** `[PAPER]`:
   - Exact reproduction of the MDP, state/action spaces, server selector, and reward function from Xu et al. (IEEE TMC 2026).
   - Uniform task priority, no explicit deadlines, homogeneous text-to-image AIGC tasks (Stable Diffusion v1.4 with DistriFusion patch parallelism).
   - Serves as the scientific benchmark against reported paper results.

2. **`extended_mode`** `[ABSTRACT]`:
   - Extends the base MDP with QoS parameters (multi-level priority $\in \{1, 2, 3\}$, relative deadlines, minimum quality thresholds $q_{min}$, heterogeneous workload types).
   - Multi-tier topology: Low-latency local edge clusters coupled with high-capacity remote cloud nodes.
   - Extended reward penalizing deadline misses weighted by priority, incorporating server energy consumption, and rewarding QoS compliance.
   - Dynamic workload adaptation handling bursty/diurnal arrivals and transient server faults/slowdowns.
   - This mode represents the primary capstone innovation and is the default for dashboard demonstration.

---

## 2. Discrete-Event Environment Specification (`CloudEdgeSchedEnv`)

The scheduling environment is formulated as a discrete-event Gymnasium environment running in continuous time with discrete decision steps.

### 2.1 Task Model

- **Task Tuple** `[PAPER]`:
  Each incoming task $k \in K$ is defined by:
  $$k = (g_k, c_k, t_k^a)$$
  where:
  - $g_k$: Prompt text or task descriptor `[PAPER]`.
  - $c_k \in \{1, 2, 4, 8\}$: Number of cooperating parallel servers (sampled from distribution $D_c$) `[PAPER]`.
  - $t_k^a$: Arrival timestamp, where inter-arrival time $t_k^g = t_{k+1}^a - t_k^a \sim D_g$ (default: Exponential distribution with arrival rate $\lambda$) `[PAPER]`.

- **Extended Task Parameters** `[ABSTRACT]`:
  In `extended_mode`, task $k$ is augmented to:
  $$k = (g_k, c_k, t_k^a, p_k, d_k^{rel}, q_k^{min}, w_k^{type})$$
  - Priority $p_k \in \{1, 2, 3\}$: 1 (Low / Best Effort), 2 (Standard), 3 (High / Critical) `[ABSTRACT]`.
  - Relative deadline $d_k^{rel} > 0$: Absolute deadline $d_k^{abs} = t_k^a + d_k^{rel}$ `[ABSTRACT]`.
  - Quality requirement $q_k^{min} \in [0.0, 1.0]$: Minimum acceptable generation or service quality `[ABSTRACT]`.
  - Workload type $w_k^{type} \in \{\text{AIGC}, \text{ML\_Inference}, \text{Video\_Analytics}, \text{CPU\_Batch}\}$ `[ABSTRACT]`.

- **Waiting Queue $Q(t)$** `[PAPER]`:
  - Arrived tasks are enqueued in $Q(t)$, strictly ordered by arrival time $t_k^a$ (FIFO base ordering) `[PAPER]`.
  - The scheduler has visibility into the top-$l$ tasks in the queue (default $l = 8$) `[PAPER]`.
  - If $|Q(t)| < l$, key-padding masks are applied so that empty queue slots are never fed as valid data to neural attention layers `[PAPER]`.

### 2.2 Task Execution Time & Quality Calibration

- **Execution Time $t_k^e$** `[PAPER]`:
  Modeled as a function of inference denoising steps $s_k \in [S_{min}, S_{max}]$ (with $S_{min} = 20, S_{max} = 60$ `[ASSUMPTION]`) and parallelism $c_k$:
  $$t_k^e = f(s_k, c_k) = \frac{s_k \cdot \text{step\_time\_base}}{\text{speedup}(c_k) \cdot \min_{e \in E_k} (\text{speed}_e)}$$
  - Measured speedups from paper Table I & Fig. 4 `[PAPER]`:
    - 1 patch ($c=1$): $\times 1.00$ (baseline, 23.7 s for 50 steps $\implies 0.474$ s/step) `[PAPER]`.
    - 2 patches ($c=2$): $\times 1.59$ (13.3 s for 50 steps) `[PAPER]`.
    - 4 patches ($c=4$): $\times 1.91$ (7.6 s for 50 steps) `[PAPER]`.
    - 8 patches ($c=8$): Extrapolated to $\times 2.45$ `[ASSUMPTION]` (justification: sublinear scaling due to distributed patch boundary synchronization overhead in DistriFusion).
  - Gang execution property: Per-step execution time is throttled by the slowest allocated server in the gang $E_k$: $\min_{e \in E_k} (\text{speed}_e)$ `[PAPER]`.

- **Initialization Delay $t_k^d$** `[PAPER]`:
  Represents model loading and distributed process group establishment:
  1. **Cold Start** ($t_k^d \approx 30.0$ s): Requires loading weights from disk into GPU VRAM and building NCCL process groups `[PAPER]`.
  2. **In-Memory Reconfiguration** ($t_k^d \in [0.5, 2.0]$ s, calibrated default $1.76$ s per Table VI): Decoupled model weights remain resident in GPU memory; only process group communication buffers and CUDA Graphs are rebuilt `[PAPER]`.
  3. **Process Group Reuse** ($t_k^d = 0.0$ s): An idle server group $G_m^t$ with matching group ID $m_k$ and exactly $c_k$ servers already exists; zero initialization overhead is incurred `[PAPER]`.

- **Quality Score $q_k$** `[PAPER]`:
  In the real system, $q_k = w_c \cdot \text{CLIP}(i_k, g_k) + w_b \cdot \text{BRISQUE}(i_k)$ with $w_c > 0, w_b < 0$ `[PAPER]`.
  In simulation, we implement a saturating monotone function of inference steps $s_k$ with bounded zero-mean Gaussian noise `[ASSUMPTION]`:
  $$q_k(s_k) = \text{clip}\left( q_{max} \cdot \left(1 - \exp\left(-\frac{s_k}{\tau_q}\right)\right) + \epsilon_q, 0.0, 1.0 \right)$$
  - Justification: Matches empirical diffusion behavior (Fig. 4 & Fig. 8 in paper) where quality rises rapidly from 20 to 40 steps and plateaus past 50 steps. Calibrated default parameters: $q_{max} = 0.85$, $\tau_q = 18.0$, $\sigma_q = 0.015$ `[ASSUMPTION]`.
  - Architecture provides an extensible `QualityBackend` abstract base class to permit direct injection of live CLIP/BRISQUE inference if GPU resources are enabled `[PAPER]`.

### 2.3 Server Cluster Topologies & Presets

Each server $e \in E$ is characterized by the tuple $\{a_e(t), t_e^r(t), d_e(t), \text{speed}_e, \text{tier}_e\}$:
- $a_e(t) \in \{0, 1\}$: Server availability status ($1 = \text{idle}, 0 = \text{busy}$) `[PAPER]`.
- $t_e^r(t) \ge 0$: Estimated remaining time until task completion `[PAPER]`.
- $d_e(t) \in \mathbb{N}_0$: Current process-group ID ($0 = \text{no loaded model}$) `[PAPER]`.
- $\text{speed}_e$: Normalized compute throughput multiplier `[PAPER]`.
- $\text{tier}_e \in \{\text{edge}, \text{cloud}\}$: Infrastructure hierarchy level `[ABSTRACT]`.

#### Cluster Presets:
1. **`edge32`** `[PAPER]`:
   - 32 homogeneous edge servers ($\text{speed}_e = 1.0$).
   - Task arrival rate $\lambda = 0.3$ tasks/s.
2. **`edge48`** `[PAPER]`:
   - 48 edge servers divided into 3 equal partitions of 16 servers:
     - Group 1: 16 servers at $\times 1.5$ speed.
     - Group 2: 16 servers at $\times 1.0$ speed.
     - Group 3: 16 servers at $\times 0.5$ speed.
   - Task arrival rate $\lambda = 0.45$ tasks/s.
3. **`edge64`** `[PAPER]`:
   - 64 edge servers divided into 4 equal partitions of 16 servers:
     - Group 1: 16 servers at $\times 1.6$ speed.
     - Group 2: 16 servers at $\times 1.2$ speed.
     - Group 3: 16 servers at $\times 0.8$ speed.
     - Group 4: 16 servers at $\times 0.4$ speed.
   - Task arrival rate $\lambda = 0.6$ tasks/s.
4. **`cloud_edge` (Extended Preset)** `[ABSTRACT]`:
   - 32 edge servers (local low-latency, moderate speed $\times 1.0$, idle power 80W, busy power 250W).
   - 16 cloud servers (high-throughput $\times 2.5$, WAN round-trip latency $+120$ ms `[ASSUMPTION]`, idle power 150W, busy power 450W `[ASSUMPTION]`).
   - Task arrival rate $\lambda = 0.7$ tasks/s with diurnal/burst patterns.

---

## 3. Decision Process, State, and Action Spaces

### 3.1 State Representation $s_t$

The state $s_t = (s_t^{Edge}, s_t^{Queue})$ is observed at each scheduling step:
1. **Edge State $s_t^{Edge}$** ($3 \times |E|$) `[PAPER]`:
   $$s_t^{Edge} = \begin{bmatrix} a_{e_1}(t) & \dots & a_{e_{|E|}}(t) \\ t_{e_1}^r(t) & \dots & t_{e_{|E|}}^r(t) \\ d_{e_1}(t) & \dots & d_{e_{|E|}}(t) \end{bmatrix}$$
   - Normalized: $t_e^r$ scaled by time limit $T_{max} = 1024$ s `[ASSUMPTION]`, $d_e$ encoded or scaled `[ASSUMPTION]`.
   - In `extended_mode`, server tier and power states are appended, yielding a $(5 \times |E|)$ matrix `[ABSTRACT]`.

2. **Queue State $s_t^{Queue}$** ($2 \times l$) `[PAPER]`:
   For the top-$l$ visible tasks in queue $Q(t)$:
   $$s_t^{Queue} = \begin{bmatrix} t_{k_1}^a & \dots & t_{k_l}^a \\ c_{k_1} & \dots & c_{k_l} \end{bmatrix}$$
   - Normalized: Arrival time represented relative to current time $(t - t_k^a) / T_{max}$ (waiting time), $c_k / 8.0$ `[ASSUMPTION]`.
   - In `extended_mode`, queue state expands to $(6 \times l)$ including priority $p_k / 3.0$, slack $(d_k^{abs} - t) / T_{max}$, quality target $q_k^{min}$, and workload type embedding `[ABSTRACT]`.
   - Accompanied by a boolean key-padding mask $M_t^{Queue} \in \{0, 1\}^l$ where $M_i = 1$ if slot $i$ is empty `[PAPER]`.

### 3.2 Action Representation $a_t$

The agent produces a continuous 3-vector $a_t = [a_c, a_s, a_k] \in [-1, 1]^3$ `[PAPER]`:
1. **Execution Decision $a_c \in [-1, 1]$**:
   - Condition: A task is scheduled if and only if $a_c > 0$ and valid resources exist; if $a_c \le 0$, the scheduler skips this step and advances simulation time by $\Delta t$ `[PAPER Alg. 1-2]`.
   - *Discrepancy Note*: The text in Section V-A2 states "$a_c \ge 0$ means a task is scheduled", whereas Alg. 1 Line 14 and Alg. 2 Line 8 specify `if a_c > 0 and a_t is valid`. We strictly adhere to $a_c > 0$ to prevent ambiguous zero-boundary scheduling `[PAPER]`.
2. **Inference Steps $a_s \in [-1, 1]$**:
   - Linearly mapped to discrete diffusion steps:
     $$s_k = \text{round}\left( S_{min} + \frac{a_s + 1}{2} (S_{max} - S_{min}) \right) \in [S_{min}, S_{max}]$$
3. **Task Selection $a_k \in [-1, 1]$**:
   - Linearly mapped to index in the top-$l$ visible queue:
     $$\text{idx} = \text{clamp}\left( \left\lfloor \frac{a_k + 1}{2} \cdot \min(l, |Q(t)|) \right\rfloor, 0, \min(l, |Q(t)|) - 1 \right)$$
   - Direct integer indexing guarantees valid task extraction among available queue tasks `[PAPER]`.

---

## 4. Deterministic Greedy Server Selector (Alg. 1) `[PAPER]`

When task $k^*$ with requirement $c_{k^*}$ is triggered for execution, servers are chosen deterministically according to Algorithm 1:

1. **Step 1: Process Group Reuse Check** `[PAPER]`:
   - Inspect all idle servers $E_{idle} = \{e \in E \mid a_e(t) = 1\}$.
   - Find existing idle process groups: $G_m^t = \{e \in E_{idle} \mid d_e(t) = m\}$.
   - If any group has $|G_m^t| = c_{k^*}$, select group $G_m^t$. If multiple such groups exist, select the group with the smallest group ID $m$.
   - Selected servers: $E_{k^*} = G_m^t$. Initialization cost $t_d = 0.0$ s (Reuse) `[PAPER]`.

2. **Step 2: Server Aggregation & Group Eviction** `[PAPER]`:
   - If no matching idle group of size $c_{k^*}$ exists:
     a. Collect all idle servers having no loaded process ($d_e(t) = 0$).
     b. If more servers are needed to reach $c_{k^*}$, iteratively destroy existing idle process groups in ascending order of their size (primary key) and group ID (secondary key), releasing their constituent servers until exactly $c_{k^*}$ idle servers are gathered.
     c. Evict old process group assignments from the selected servers.
     d. Allocate a new unique process group ID $m_{new}$.
     e. Initialization cost:
        - If all selected servers had models resident in memory $\implies t_d = 1.76$ s (In-Memory Reconfiguration) `[PAPER Sec. VI-D, Table VI]`.
        - If any selected server required loading weights from scratch ($d_e = 0$) $\implies t_d = 30.0$ s (Cold Start) `[PAPER Sec. VI-D]`.

3. **Step 3: State Transition & Safe No-Op Invariant** `[PAPER]`:
   - If fewer than $c_{k^*}$ idle servers exist in total across the cluster, the action is marked **invalid**.
   - An invalid action executes a **safe no-op**: No task is removed from $Q(t)$, no server states are corrupted, and simulation time advances by $\Delta t$ ($1.0$ s) `[PAPER]`.
   - Running servers decrement remaining time: $t_e^r(t + \Delta t) = \max(0, t_e^r(t) - \Delta t)$. Servers reaching $t_e^r = 0$ transition to $a_e = 1$ (idle) while retaining their process group ID $d_e$ `[PAPER]`.

---

## 5. Reward Formulation

### 5.1 Paper Mode Immediate Reward (Eq. 9) `[PAPER]`

For scheduled task $k^*$:
$$R_t = b + w_q \cdot \tanh\left(\frac{q_{k^*}}{s_q}\right) - w_{t1} \cdot \tanh\left(\frac{t_{k^*}^r}{s_{t1}}\right) - w_{t2} \cdot \tanh\left(\frac{t_{Q,t}^{avg}}{s_{t2}}\right)$$
where:
- $t_{k^*}^r = t_{k^*}^w + t_{k^*}^d + t_{k^*}^e$: Predicted total response time computed at scheduling time `[PAPER]`.
- $t_{k^*}^w = t - t_{k^*}^a$: Queue waiting duration `[PAPER]`.
- $t_{Q,t}^{avg} = \frac{\sum_{k \in Q(t)} (t - t_k^a)}{|Q(t)|}$: Average waiting time across all tasks in queue at decision step $t$ (penalizes queue starvation; defined as $0$ if queue is empty) `[PAPER]`.
- Scaling constants `[ASSUMPTION]`:
  - $s_q = 0.5$ (normalizes quality score $\approx 0.5-0.8$ into dynamic tanh range).
  - $s_{t1} = 40.0$ s (calibrated against typical response latency $25-60$ s).
  - $s_{t2} = 30.0$ s (calibrated against queue waiting time).
- Hyperparameters:
  - Default weights: $w_q = 0.5$, $w_{t1} = 0.3$, $w_{t2} = 0.2$ `[PAPER]`.
  - Sensitivity sweeps: $w_q \in \{0.4, 0.5, 0.6\}$ `[PAPER Sec. VII-B6]`.
  - Positive bias offset: $b = w_{t1} + w_{t2} = 0.5$ (prevents degeneration into a passive no-scheduling policy) `[PAPER Eq. 9]`.
- **Idle / No-Op Step Reward** `[ASSUMPTION]`:
  If no task is scheduled (either $a_c \le 0$ or invalid action), no task completion reward is credited, but queue backlog is penalized:
  $$R_{idle} = - w_{t2} \cdot \tanh\left(\frac{t_{Q,t}^{avg}}{s_{t2}}\right)$$
  Justification: Provides continuous gradient feedback encouraging the scheduler to dispatch tasks as soon as valid resources become available.

### 5.2 Extended Mode Reward `[ABSTRACT]`

In `extended_mode`, the reward incorporates priority-weighted deadline penalties, energy consumption, and quality threshold satisfaction:
$$R_t^{ext} = R_t - w_{dl} \cdot p_{k^*} \cdot \mathbf{1}[t_{k^*}^r > d_{k^*}^{abs} - t_{k^*}^a] - w_e \cdot \tanh\left(\frac{E_{k^*}}{s_e}\right) + R_{bonus}$$
where:
- $\mathbf{1}[\cdot]$: Indicator function for deadline violation `[ABSTRACT]`.
- $p_{k^*} \in \{1, 2, 3\}$: Priority weighting (critical tasks incur $3\times$ miss penalty) `[ABSTRACT]`.
- $E_{k^*}$: Estimated energy consumed by allocated servers over $t_k^e + t_k^d$ `[ABSTRACT]`:
  $$E_{k^*} = \sum_{e \in E_{k^*}} P_{busy}^e \cdot (t_k^e + t_k^d)$$
- $R_{bonus} = +0.1$ if $q_{k^*} \ge q_{k^*}^{min}$, else $-0.1$ `[ASSUMPTION]`.
- Default weights: $w_{dl} = 0.5$, $w_e = 0.1$, $s_e = 5000.0$ Joules `[ASSUMPTION]`.

---

## 6. EAT Agent Architecture & Neural Modules (`src/eatsched/agents/`)

### 6.1 Cross-Attention State Encoder (`cross_attention.py`) `[PAPER Eq. 13-14]`

Given raw inputs $s_t^{Queue} \in \mathbb{R}^{l \times d_q}$ and $s_t^{Edge} \in \mathbb{R}^{|E| \times d_e}$:
1. **Feature Projection** `[PAPER Eq. 13]`:
   $$\tilde{s}_t^{Queue} = \phi_Q(s_t^{Queue}) \in \mathbb{R}^{l \times d_{emb}}, \quad \tilde{s}_t^{Edge} = \phi_E(s_t^{Edge}) \in \mathbb{R}^{|E| \times d_{emb}}$$
   where embedding dimension $d_{emb} = 8$ `[PAPER Table VII]`.
2. **Asymmetric Multi-Head / Cross-Attention** `[PAPER Eq. 14]`:
   Task queue serves as queries; server cluster serves as keys and values:
   $$Q = \tilde{s}_t^{Queue} W_Q, \quad K = \tilde{s}_t^{Edge} W_K, \quad V = \tilde{s}_t^{Edge} W_V$$
   $$A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_{emb}}} + M^{mask}\right) \in \mathbb{R}^{l \times |E|}$$
   $$f_s = A V \in \mathbb{R}^{l \times d_{emb}} \implies \text{flattened to } \mathbb{R}^{l \cdot d_{emb}} = \mathbb{R}^{64}$$
3. **Invariance Properties**:
   - **Permutation Invariance over Servers** `[PAPER Sec. IV-B]`: Any permutation of server columns in $s_t^{Edge}$ yields the identical attention output $f_s$ (mathematically guaranteed as $A V = \sum_j A_{ij} V_j$ is a permutation-invariant set reduction).
   - **Padding Masking**: Empty queue positions are masked with $-\infty$ prior to softmax, ensuring zero attention leakage `[PAPER]`.

### 6.2 DDIM Diffusion Policy Network (`ddim_policy.py`) `[PAPER Sec. V-B1]`

1. **Denoiser Network $\epsilon_\theta(x_t, t_{diff}, f_s)$** `[PAPER]`:
   - Denoising steps $T = 4$ `[PAPER Table VIII]`.
   - Continuous action dimension: $d_a = 3$ ($a_c, a_s, a_k$).
   - Timestep embedding: 8-dimensional sinusoidal / MLP embedding `[PAPER Table VII]`.
   - Trunk: Fully-connected MLP with layers $[(64 + 8 + 3) \to 256 \to 256 \to 3]$, Mish activation `[PAPER Table VII]`, final Tanh layer bounding predictions to $[-1, 1]$.
2. **DDIM Reverse Sampling Chain** `[PAPER Eq. 10-12]`:
   Given noise schedule $\bar{\alpha}_t \in (0, 1)$ ($t = T, \dots, 1$):
   - Predicted clean action:
     $$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t, f_s)}{\sqrt{\bar{\alpha}_t}}$$
   - Stochastic noise variance:
     $$\sigma_t = \eta \sqrt{\left(\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\right) \left(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}\right)}$$
   - Next sample:
     $$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{\max\left(0, 1 - \bar{\alpha}_{t-1} - \sigma_t^2\right)} \epsilon_\theta + \sigma_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
   - *Numerical Safeguard*: The term $(1 - \bar{\alpha}_{t-1} - \sigma_t^2)$ is strictly clamped to $\ge 0.0$ before `torch.sqrt` to prevent NaN gradients `[PAPER Sec. V-B1]`.

3. **Learnable Stochasticity Parameter $\eta$** `[PAPER Sec. V-B1]`:
   - Parameterized as an unconstrained tensor transformed via $\eta = \eta_{max} \cdot \text{sigmoid}(w_\eta)$ or clamped directly to $[0, \eta_{max}]$ with $\eta_{max} = 1.0$ `[ASSUMPTION]`, initial $\eta = 0.5$ `[ASSUMPTION]`.
   - Gradients $\nabla_\eta Q(s, \pi_\Theta(s))$ flow back through the entire DDIM sampling chain (tested via `assert eta.grad is not None and eta.grad != 0`).
   - When $\eta = 0$, policy sampling is completely deterministic for a given initial noise $x_T$ `[PAPER]`.

### 6.3 Double Critic Networks (`critics.py`) `[PAPER Sec. V-B2]`

- Independent twin critics $Q_{\phi_1}(s, a)$ and $Q_{\phi_2}(s, a)$.
- Architecture: 2-layer MLP (256 units per layer), Mish activations `[PAPER Table VII]`.
- Target networks $Q_{\bar{\phi}_1}, Q_{\bar{\phi}_2}$ updated via Polyak soft-updates:
  $$\bar{\phi}_i \leftarrow \tau \phi_i + (1 - \tau) \bar{\phi}_i, \quad \tau = 0.005 \text{ `[PAPER Table VIII]`}$$

### 6.4 Actor & Critic Losses `[PAPER Eq. 15-23]`

- Critic Target:
  $$\hat{y} = r_t + \gamma (1 - \text{done}) \min_{i=1,2} Q_{\bar{\phi}_i}(s_{t+1}, a_{t+1})$$
  Discount factor $\gamma = 0.95$ `[PAPER Table VIII]`. Correct terminal masking is enforced.
- Critic Loss:
  $$L(\phi_i) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}} \left[ \left( Q_{\phi_i}(s, a) - \hat{y} \right)^2 \right]$$
- Actor Loss `[PAPER Eq. 16-17]`:
  $$L(\Theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ - \min_{i=1,2} Q_{\phi_i}(s, \pi_\Theta(s)) - \alpha \cdot \eta \right]$$
  with entropy temperature $\alpha = 0.05$ `[PAPER Table VIII]`.

---

## 7. Baselines and Ablations Specifications

### 7.1 Non-Learned Baselines
- **Random**: Selects valid action uniformly at random `[PAPER]`.
- **FIFO**: Schedules queue head ($k_0$) as soon as $c_{k_0}$ servers are available; fixed steps $s = 40$ `[PAPER]`.
- **Greedy**: Enumerates candidate actions $(k \in \{0..l-1\}, s \in \{20, 40, 60\})$, predicts immediate reward $R_t$, dispatches action with maximum $R_t$ `[PAPER]`.
- **EDF (Earliest Deadline First)**: Selects task with lowest absolute deadline $d_k^{abs}$ `[ABSTRACT]`.
- **Priority-FIFO**: Prioritizes higher priority tiers ($p=3 > p=2 > p=1$), scheduling FIFO within tiers `[ABSTRACT]`.

### 7.2 Learned Baselines
- **SAC**: Standard Soft Actor-Critic (Gaussian actor, no diffusion, no cross-attention) `[PAPER]`.
- **PPO**: Proximal Policy Optimization with GAE `[PAPER]`.
- **TD3**: Twin Delayed DDPG with deterministic policy and action noise `[PAPER]`.
- **DDPG**: Continuous actor-critic DDPG `[PAPER]`.
- **DPPO**: Diffusion Policy Optimization with two-layer MDP `[PAPER]`.
- **D2SAC**: Diffusion Soft Actor-Critic (DDPM based, Gaussian exploration) `[PAPER]`.

### 7.3 Ablation Models `[PAPER Sec. VII-B7]`
- **EAT-D**: EAT with DDIM removed (cross-attention actor directly outputs action) `[PAPER]`.
- **EAT-CA**: EAT with cross-attention replaced by standard concatenation + MLP `[PAPER]`.
- **D2SAC+SA**: D2SAC augmented with Self-Attention `[PAPER]`.

---

## 8. Training Budgets & Wall-Clock Protocols

1. **`smoke`**:
   - 20,000 environment steps ($\approx 30$ episodes).
   - Used for CI, unit testing, and fast sanity validation. Must complete within 2–4 minutes on modern CPU.
2. **`medium`**:
   - 200,000 environment steps ($\approx 300$ episodes).
   - Used for preliminary benchmarking and hyperparameter tuning.
3. **`full`**:
   - 1,000,000 environment steps ($1,500$ episodes per paper) `[PAPER Table VIII]`.
   - Used for final camera-ready figures and viva presentation tables.

---

## 9. FastAPI & Web Dashboard Specifications

### 9.1 FastAPI Backend (`api/`)
- `GET /health`: System status, device, checkpoint availability.
- `GET /clusters`: Metadata for `edge32`, `edge48`, `edge64`, `cloud_edge`.
- `GET /schedulers`: List of supported schedulers and ablation models.
- `POST /simulate`: Executes full episode simulation given `{scheduler, cluster, arrival_rate, w_q, mode, seed}`. Returns JSON timeline events (Gantt task allocations, reuse tags, init delays) and summary metrics.
- `GET /results/{experiment}`: Loads precomputed experimental results.
- `WebSocket /stream`: Streams step-by-step simulation events in real time.

### 9.2 Next.js 15 Web Dashboard (`web/`)
- Responsive TypeScript + Tailwind CSS application using Recharts.
- 5 core views:
  1. **Live Simulator**: Gantt timeline highlighting cold start (orange), reconfiguration (blue), and reuse (green), with interactive play/pause and queue inspection.
  2. **Scheduler Comparison**: Side-by-side run of two schedulers on identical seeds.
  3. **Learning & Ablation Curves**: Real TensorBoard/CSV curves showing reward trajectory.
  4. **Pareto & QoS Analytics**: Latency vs. Quality Pareto frontier and deadline satisfaction.
  5. **About & Viva Guide**: Formal mathematical formulation, architectural diagrams, and honest audit disclosures.
