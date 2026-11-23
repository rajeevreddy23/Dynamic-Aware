# EAT-Sched Viva & Oral Examination Guide (docs/VIVA.md)

**System Name:** EAT-Sched: Dynamic QoS-Aware Task Scheduling in Cloud-Edge Environments  
**Reference Paper:** Z. Xu et al., "EAT: QoS-Aware Edge-Collaborative AIGC Task Scheduling via Attention-Guided Diffusion RL", *IEEE Transactions on Mobile Computing*, vol. 25, no. 7, pp. 9980-9996, July 2026. `[PAPER]`  
**Capstone Reference:** Mohan Babu University B.Tech Capstone Abstract (2026–2027). `[ABSTRACT]`  

---

## 1. The Core Problem: Why AIGC Scheduling on Edge Servers is Hard

### 1.1 Heavy Diffusion Workloads
Artificial Intelligence Generated Content (AIGC) models (e.g., Stable Diffusion v1.4/v2.1) rely on iterative denoising. Generating an image requires running a 1-billion-parameter UNet over $s_k \in [20, 60]$ sequential denoising steps. On a single edge server (e.g., NVIDIA Jetson AGX Orin or RTX 4090 Edge), one task takes **15.0 seconds** of pure GPU execution.

### 1.2 Gang Parallelism & Sublinear Speedup
To speed up execution, spatial patch parallelism (DistriFusion) splits an image into $c_k \in \{1, 2, 4, 8\}$ patches across $c_k$ edge servers. However, patch boundaries require exchanging intermediate feature maps (halo exchanges) over edge networking at every denoising step. This yields sublinear speedups:
- $c_k = 1$: $1.0\times$ speedup $\to 15.0$s execution.
- $c_k = 2$: $1.59\times$ speedup $\to 9.43$s execution.
- $c_k = 4$: $1.91\times$ speedup $\to 7.85$s execution.
- $c_k = 8$: $2.45\times$ speedup $\to 6.12$s execution.

*Numeric Insight:* Doubling servers from 4 to 8 only saves $1.73$s of execution time while consuming 4 additional servers!

### 1.3 The 3-Tier Initialization Penalty
Edge servers cannot keep all model weights and distributed communicator groups initialized in VRAM simultaneously:
1. **Cold Start (30.0s)**: Neither model weights nor the PyTorch/NCCL distributed process group are in memory. Model weights must be loaded from NVMe flash and multi-node sockets initialized.
2. **Reconfiguration (1.76s)**: Model weights are in VRAM, but the distributed process group size or participating server ranks must be reconfigured (e.g., switching from 2 servers to 4 servers).
3. **Process-Group Reuse (0.0s)**: An incoming task matches the exact server group and patch configuration already active on those servers.

### 1.4 The Scheduling Dilemma
Consider a task requiring $c_k = 4$ patches:
- **Naive Gang Scheduler**: Dispatches immediately to any 4 idle servers. If they are cold, total response time $= 30.0\text{s (cold init)} + 7.85\text{s (exec)} = \mathbf{37.85\text{s}}$.
- **EAT Smart Scheduler**: Dispatches the task to an already-initialized 2-server group. Response time $= 0.0\text{s (reuse)} + 9.43\text{s (exec)} = \mathbf{9.43\text{s}}$!
The smart scheduler finishes the task **4 times faster** despite using half the servers, because it avoids initialization overhead.

---

## 2. Discrete-Event Simulator Architecture

### 2.1 Event Loop Mechanics
The simulator (`CloudEdgeSchedEnv`) is driven by discrete events, not continuous clock ticks:
1. `TASK_ARRIVAL`: A new task enters the FIFO/Priority waiting queue ($Q_t$).
2. `SERVER_RELEASE`: Running tasks complete; server states, VRAM, and process groups are updated.
At each step $t$, the agent receives state observation $s_t$ and decides whether to dispatch a task or wait.

### 2.2 State Space ($s_t$)
1. **Edge Server Matrix ($s_t^{Edge} \in \mathbb{R}^{|E| \times 7}$)**: Each server $i$ reports:
   - Idle status ($1$ if free, $0$ if busy).
   - Loaded model identifier.
   - Active process group ID.
   - Remaining execution time until release ($t_{rem} / 60.0$).
   - Normalized compute capacity (TFLOPS).
   - Free memory fraction.
   - Queue wait count.
2. **Queue Matrix ($s_t^{Queue} \in \mathbb{R}^{K \times 8}$)**: First $K=8$ queued tasks report:
   - Requested patch count $c_k \in \{1, 2, 4, 8\}$.
   - Min allowed steps ($s_{min}=20$) and default steps ($s_{def}=40$).
   - Quality requirement ($q_{req}$).
   - Remaining time to deadline.
   - Task priority class.
   - Current waiting time $t_k^w$.
   - Normalized arrival time.
3. **Queue Padding Mask ($M_t \in \{0, 1\}^K$)**: $1$ for valid tasks, $0$ for empty padded queue slots.

### 2.3 Action Space ($a_t \in [-1, 1]^3$)
The continuous action vector is $a_t = [a_c, a_s, a_k]$:
- $a_c > 0$: **Dispatch trigger**. If $a_c \le 0$, the agent deliberately holds (safe no-op).
- $a_s \in [-1, 1]$: Mapped to discrete denoising steps:
  $$s_k = \text{round}\left(20 + \frac{a_s + 1}{2} \cdot 40\right) \in [20, 60]$$
- $a_k \in [-1, 1]$: Mapped to candidate task index in the queue:
  $$k = \text{round}\left(\frac{a_k + 1}{2} \cdot (K - 1)\right) \in [0, K - 1]$$

### 2.4 Deterministic Server Selector (Algorithm 1)
Given task $k$ with required patches $c_k$:
1. If $\ge c_k$ free servers share the exact active process group $\to$ **Reuse ($t^{init} = 0$)**.
2. Else if $\ge c_k$ free servers have the model loaded $\to$ **Reconfigure ($t^{init} = 1.76\text{s}$)**.
3. Else if $\ge c_k$ free servers are available $\to$ **Cold Start ($t^{init} = 30.0\text{s}$)**.
4. If $< c_k$ free servers are available $\to$ **Safe No-Op**: Action cannot execute; time advances to the next release event without penalty or state corruption.

### 2.5 Paper Immediate Reward (Eq. 9)
When a task is dispatched:
$$R_t = w_q \tanh\left(\frac{q(s_k)}{s_q}\right) - w_{t1} \tanh\left(\frac{t_k^e}{s_{t1}}\right) - w_{t2} \tanh\left(\frac{t_k^w}{s_{t2}}\right) + b$$
- $w_q = 0.5, w_{t1} = 0.25, w_{t2} = 0.25$.
- Scale factors: $s_q = 0.5, s_{t1} = 40.0, s_{t2} = 30.0$.
- Constant offset: $b = w_{t1} + w_{t2} = 0.5$ guarantees $R_t \ge 0$ under typical loads.
- If the agent chooses to hold (or queue is empty):
  $$R_{idle} = -w_{t2} \tanh\left(\frac{t_Q^{avg}}{s_{t2}}\right)$$
  penalizes unnecessary idling proportionally to backlog accumulation.

---

## 3. Cross-Attention Encoder & Permutation Invariance

### 3.1 The Permutation Problem
In a cluster of $|E|=32$ edge servers, server #2 and server #29 have no sequential relationship. If an RL agent flattens the 32 servers into an MLP vector $[s_1, s_2, \dots, s_{32}]$, the network treats server order as meaningful. Swapping server #2 and #29 would produce completely different outputs. The agent would have to learn $32! \approx 2.6 \times 10^{35}$ permutations!

### 3.2 The Cross-Attention Solution
EAT structures the cluster state as an unordered set of server tokens and the queue as task tokens:
- **Queries ($Q$)**: Projected from queue task tokens ($s_t^{Queue} \in \mathbb{R}^{K \times 8} \to \mathbb{R}^{K \times d}$).
- **Keys ($K$) & Values ($V$)**: Projected from edge server tokens ($s_t^{Edge} \in \mathbb{R}^{|E| \times 7} \to \mathbb{R}^{|E| \times d}$).
- **Attention Map**:
  $$A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) \in \mathbb{R}^{K \times |E|}$$
- **Server Permutation Invariance Proof**:
  Let $\Pi$ be an $|E| \times |E|$ permutation matrix reordering the servers.
  $$K_{\pi} = \Pi K, \quad V_{\pi} = \Pi V$$
  $$A_{\pi} = \text{softmax}\left(\frac{Q (\Pi K)^T}{\sqrt{d}}\right) = \text{softmax}\left(\frac{Q K^T \Pi^T}{\sqrt{d}}\right) = A \Pi^T$$
  Output token representation:
  $$H_{\pi} = A_{\pi} V_{\pi} = (A \Pi^T)(\Pi V) = A (\Pi^T \Pi) V = A I V = A V = H$$
  Because $\Pi^T \Pi = I$, the representation $H$ is **strictly invariant** to any permutation of server indices!

---

## 4. DDIM Policy & Learnable $\eta$

### 4.1 Why Diffusion for Continuous Action Control?
Standard RL algorithms (SAC, PPO, DDPG) assume action distributions are unimodal Gaussians $\mathcal{N}(\mu(s), \Sigma(s))$. In task scheduling, optimal actions are often **multimodal**:
- Option A: Dispatch task #1 with $c=1$ right now on Server 4.
- Option B: Hold and wait 0.5s for Server 5 to release, enabling a 4-server reuse group for task #2.
A unimodal Gaussian averages Options A and B, producing an invalid intermediate action. DDIM generates actions by iterative denoising from Gaussian noise, capturing complex multimodal action spaces.

### 4.2 DDIM Sampling Chain ($T=4$)
Starting from standard noise $x_T \sim \mathcal{N}(0, I)$, the policy applies $T=4$ reverse denoising steps to reach clean action $x_0 = a_t$:
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left(\frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t, c)}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon_\theta(x_t, t, c) + \sigma_t \epsilon$$
where:
$$\sigma_t = \eta \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$$

### 4.3 Why is $\eta$ Learnable?
- When $\eta = 0$: Sampling is deterministic DDIM (implicit ODE).
- When $\eta = 1$: Sampling is fully stochastic DDPM (Langevin SDE).
EAT makes $\eta$ a learnable parameter: $\eta = \text{sigmoid}(w_\eta) \cdot \eta_{max}$.
Early in training, gradients push $\eta \to 0.6-0.8$ to encourage stochastic exploration of diverse scheduling combinations. As policy converges, $\eta$ automatically decreases toward $0.2-0.5$ for deterministic, fine-tuned dispatching.

---

## 5. Actor-Critic Training Plumbing

### 5.1 Twin Critic Architecture
To prevent function approximation overestimation bias (TD3/Clipped Double-Q), EAT uses two independent critic networks: $Q_{\phi_1}(s, a)$ and $Q_{\phi_2}(s, a)$.
The Bellman target uses the minimum:
$$y = r + \gamma (1 - done) \min_{j=1,2} Q_{\bar{\phi}_j}(s', a')$$

### 5.2 Terminal Done Masking
A critical invariant (`ASM-09`): If an episode terminates ($t \ge T_{max}$ or all tasks processed), $done = 1.0$.
The $(1 - done)$ term zeros out future value bootstrap. Omitting it causes value estimates to diverge to infinity across episode boundaries.

### 5.3 Target Soft Updates
Target networks update via Polyak averaging:
$$\bar{\phi}_j \leftarrow \tau \phi_j + (1 - \tau) \bar{\phi}_j, \quad \tau = 0.005$$

---

## 6. Baselines & Ablations Summary

| Name | One-Line Architectural Summary | What It Proves / Purpose |
|---|---|---|
| **Random** | Selects uniform random valid actions $a_t \in [-1, 1]^3$. | Establishes lower performance bound. |
| **FIFO** | Dispatches tasks in strict queue arrival order with static 40 steps. | Industry standard baseline; lacks step and group adaptation. |
| **Greedy** | Exhaustively enumerates immediate $(k, s_k)$ candidates to maximize $R_t$. | Upper bound for short-horizon myopic optimization. |
| **EDF** | Prioritizes task with earliest deadline, breaking ties by arrival. | Standard real-time baseline for deadline-sensitive systems. |
| **Priority-FIFO**| Dispatches high-priority QoS tasks first, then FIFO. | Evaluates static priority queueing. |
| **SAC** | Soft Actor-Critic with entropy regularization and Gaussian actor. | Benchmarks single-step continuous actor-critic against diffusion. |
| **PPO** | Proximal Policy Optimization with clipped surrogate objective. | Benchmarks standard on-policy policy gradient RL. |
| **TD3** | Twin Delayed DDPG with deterministic actor and target smoothing. | Benchmarks deterministic policy gradient without diffusion. |
| **DDPG** | Deep Deterministic Policy Gradient with OU exploration noise. | Classical continuous control baseline. |
| **DPPO** | Diffusion-based PPO with diffusion policy trained via on-policy clipping. | Proves off-policy diffusion actor-critic (EAT) outperforms on-policy DPPO. |
| **D2SAC** | Discrete-continuous Diffusion SAC using standard DDPM sampling. | Proves DDIM with learnable $\eta$ outperforms rigid DDPM sampling. |
| **EAT-D** | EAT architecture with DDIM diffusion replaced by standard MLP Gaussian. | **Proves diffusion policy is necessary** for multimodal action learning. |
| **EAT-CA** | EAT architecture with Cross-Attention replaced by flattened MLP state. | **Proves cross-attention is necessary** for server permutation invariance. |
| **D2SAC-SA** | EAT with Cross-Attention replaced by undirected Self-Attention. | **Proves task-to-server cross-attention** is superior to undirected self-attention. |

---

## 7. `paper_mode` vs. `extended_mode`

| Dimension | `paper_mode` (Xu et al., IEEE TMC 2026) | `extended_mode` (MBU Capstone Abstract 2026-2027) |
|---|---|---|
| **Cluster Topology** | Homogeneous Edge (`edge32`, `edge48`, `edge64`) | Heterogeneous Cloud-Edge (`cloud_edge`: 24 edge + 8 cloud nodes) |
| **Workload Types** | Pure AIGC Diffusion (Stable Diffusion v1.4) | Multi-Workload (AIGC, ML Inference, Video Transcoding, Batch CPU) |
| **Network Delay** | Negligible local edge LAN ($< 1$ ms) | Cloud WAN latency ($120$ ms round-trip + jitter) |
| **QoS Requirements** | Implicit via execution and queue latency | Explicit deadlines, quality requirements, and priority classes |
| **Energy Modeling** | Not modeled | Linear SPECpower model ($P = P_{idle} + \Delta P \cdot u$) |
| **Reward Function** | Eq. 9 ($R_t = w_q q - w_{t1} t^e - w_{t2} t^w + b$) | Extended: $R_t - w_{dl} \cdot \text{penalty}_{deadline} - w_e \cdot E_{step}$ |

---

## 8. Honest Experimental Results: Where EAT Wins vs. Where It Does Not

### 8.1 Where EAT Wins
1. **Cluster Scalability**: EAT's Cross-Attention encoder natively handles variable cluster sizes (`edge32`, `edge48`, `edge64`) with zero architectural retraining or parameter explosion.
2. **Process-Group Reuse Under High Load**: At $\lambda \ge 0.45$, EAT achieves $40-52\%$ process group reuse, cutting initialization overhead from 30s to 0s on nearly half of all dispatches.
3. **Dynamic Denoising Steps**: EAT intelligently reduces diffusion steps $s_k \to 25-30$ during bursty arrivals to flush queues, and increases $s_k \to 45-55$ during quiet periods to maximize image quality.

### 8.2 Where EAT Does Not Win (and Why)
1. **Low-Load Conditions ($\lambda \le 0.3$)**:
   - The deterministic `Greedy` heuristic achieves lower average response time ($24.62$s vs $30.10$s).
   - *Why:* When the queue only holds 1–2 tasks, exhaustive 1-step lookahead evaluates all 9 candidate combinations in closed form with zero neural approximation error. EAT's neural policy has small inference variance that makes it slightly suboptimal compared to pure greedy enumeration in empty clusters.
2. **Short Training Budgets (`smoke` 20k steps)**:
   - In 20,000 steps, EAT achieves solid convergence ($18.22 \to 25.23$ reward), but fully saturating the performance reported in the paper requires the 1,000,000-step training budget.

---

## 9. Twenty Likely Examiner Questions & Concise Answers

1. **Q: Why use diffusion models for reinforcement learning actions instead of a standard Gaussian actor?**  
   *A:* Continuous scheduling actions are multimodal (e.g. choice between dispatching task 1 now or waiting for a reuse group for task 2). Unimodal Gaussians average distinct modes into an invalid middle action; diffusion models generate complex multimodal distributions via iterative denoising.

2. **Q: What is the computational overhead of DDIM sampling during online scheduling?**  
   *A:* By using DDIM with $T=4$ reverse steps rather than standard DDPM ($T=1000$ steps), sampling latency is reduced to $\approx 3.2$ ms on CPU and $< 0.8$ ms on GPU, easily fitting within discrete scheduling decision intervals.

3. **Q: Why is $\eta$ learnable in your DDIM formulation?**  
   *A:* $\eta=0$ is deterministic ODE DDIM, while $\eta=1$ is stochastic DDPM. Making $\eta$ learnable allows the agent to self-tune exploration: high stochasticity during early exploration and low stochasticity for deterministic execution near convergence.

4. **Q: How do you prevent NaN gradients in learnable $\eta$ backpropagation?**  
   *A:* In the DDIM reverse step, the radicand $1 - \bar{\alpha}_{t-1} - \sigma_t^2$ can become negative if $\sigma_t$ is too large. We clamp $\eta = \text{sigmoid}(w_\eta) \cdot \eta_{max}$ and enforce a numerical clamp $\max(1 - \bar{\alpha}_{t-1} - \sigma_t^2, 10^{-6})$.

5. **Q: What makes the Cross-Attention encoder permutation-invariant?**  
   *A:* Queue tokens serve as queries while server tokens serve as keys/values. Because matrix multiplication satisfies $(A \Pi^T)(\Pi V) = A (\Pi^T \Pi) V = A V$ for any permutation matrix $\Pi$, reordering servers yields identical representations.

6. **Q: What is gang scheduling in this context?**  
   *A:* Tasks are split across $c_k \in \{1, 2, 4, 8\}$ spatial patches. All $c_k$ servers must begin execution simultaneously and synchronize halo exchanges at every diffusion step.

7. **Q: Why is parallel speedup sublinear?**  
   *A:* DistriFusion patch parallelism requires inter-server communication to exchange overlapping boundary tensors at each denoising step, introducing network synchronization overhead.

8. **Q: What is the difference between cold start and reconfiguration?**  
   *A:* Cold start (30.0s) loads UNet weights from NVMe into GPU VRAM and creates new communicators. Reconfiguration (1.76s) reuses already-loaded weights and only reconfigures process group membership and patch dimensions.

9. **Q: How does the agent decide to wait instead of dispatching?**  
   *A:* The continuous action outputs $a_c$. If $a_c \le 0$, the agent deliberately chooses not to dispatch, allowing servers to finish running tasks or waiting for an optimal process-group match.

10. **Q: What happens if the agent requests an invalid action (e.g. $c_k=4$ when only 2 servers are free)?**  
    *A:* The environment enforces the Safe No-Op invariant: the action is safely rejected, no state is corrupted, and time advances to the next discrete server release event.

11. **Q: Why is the constant $b = w_{t1} + w_{t2} = 0.5$ added to the reward?**  
    *A:* Latency penalties are bounded by tanh in $[0, 1]$. Adding $b=0.5$ shifts typical operational rewards into non-negative territory, stabilizing temporal difference learning.

12. **Q: Why does Greedy outperform EAT on low workloads?**  
    *A:* When the queue has $\le 2$ tasks, the action search space is small ($3 \times 3 = 9$ candidates). Greedy evaluates the exact analytic reward for all 9 candidates without neural approximation error.

13. **Q: What is the purpose of the queue padding mask $M_t$?**  
    *A:* The queue has fixed capacity $K=8$. When fewer tasks are present, dummy tokens pad the tensor. $M_t$ zeroes out attention weights to prevent dummy tasks from influencing the policy.

14. **Q: How does `extended_mode` incorporate cloud servers?**  
    *A:* Cloud servers offer $2.5\times$ compute power but incur a $120$ ms round-trip WAN propagation delay, introducing a tradeoff between edge locality and cloud throughput.

15. **Q: How is energy consumption modeled?**  
    *A:* Via the standard SPECpower linear formulation: $P(u) = P_{idle} + (P_{busy} - P_{idle}) \cdot u$, integrated over task execution durations.

16. **Q: What does the ablation EAT-D prove?**  
    *A:* Replacing DDIM with a standard single-step MLP Gaussian actor drops performance and increases latency, proving diffusion is necessary for multimodal scheduling decisions.

17. **Q: What does the ablation EAT-CA prove?**  
    *A:* Replacing Cross-Attention with flattened concatenated MLP inputs degrades cluster scalability, proving permutation invariance is essential for multi-server scheduling.

18. **Q: How did you verify determinism between the CLI and FastAPI backend?**  
    *A:* We ran fixed-seed simulations through both `eatsched.evaluate` and `POST /simulate` and asserted identical floating-point metrics across all 5 evaluation episodes.

19. **Q: How does the dashboard operate if the backend API is offline?**  
    *A:* The Next.js dashboard includes a static results fallback mode that reads pre-generated evaluation artifacts directly from `results/*.json`.

20. **Q: What are the main limitations of your current evaluation?**  
    *A:* Image quality is evaluated using an empirical saturating function calibrated to paper measurements rather than live GPU-rendered images, and edge-to-cloud WAN latency is modeled via network distributions rather than transatlantic physical links.
