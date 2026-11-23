# Capstone Project Report Outline (docs/REPORT_OUTLINE.md)

**Title:** Dynamic QoS-Aware Task Scheduling in Cloud-Edge Environments using Attention-Guided Diffusion Reinforcement Learning  
**Degree:** Bachelor of Technology in Computer Science and Engineering  
**Institution:** Mohan Babu University (MBU), Sree Sainath Nagar, Tirupati  
**Academic Year:** 2026–2027  
**Base Reference:** Z. Xu et al., *"EAT: QoS-Aware Edge-Collaborative AIGC Task Scheduling via Attention-Guided Diffusion RL"*, IEEE Transactions on Mobile Computing (TMC), July 2026. `[PAPER]`  

---

## Preliminary Pages

1. **Title Page**: Formal university formatting with institutional crest, candidate names, roll numbers, and supervisor details.
2. **Certificate of Authenticity**: Signed approval of the supervisor and Head of Department.
3. **Declaration**: Plagiarism-free and empirical honesty statement by the project candidates.
4. **Abstract**: 300-word structured abstract summarizing context, challenges, proposed EAT-Sched architecture, and key empirical results.
5. **Acknowledgements**: Acknowledging supervisor guidance, department compute resources, and family support.
6. **Table of Contents**: Structured listing with section-level numbering.
7. **List of Figures**: Automatic catalog of all architectural schematics, Gantt charts, and benchmark plots.
8. **List of Tables**: Catalog of specification matrices, hyperparameters, and experimental benchmark tables.
9. **List of Abbreviations & Symbols**: Definitions of mathematical variables ($s_k, c_k, a_t, \eta, R_t, P_{idle}$, etc.).

---

## Chapter 1: Introduction

### 1.1 Background & Context
- Rise of Artificial Intelligence Generated Content (AIGC): Diffusion models (Stable Diffusion v1.4/v2.1) transforming creative workflows.
- Computational demands of multi-step iterative denoising on resource-constrained edge servers.
- The shift from centralized mega-clouds to distributed cloud-edge collaborative architectures.

### 1.2 Problem Statement & Motivation
- The latency vs. quality conflict: more denoising steps improve perceptual quality but increase service response times.
- Gang scheduling and spatial patch parallelism (DistriFusion) across heterogeneous nodes.
- Heavy initialization penalties: cold starts (30s) vs. reconfigurations (1.76s) vs. instant group reuse (0s).

### 1.3 Project Objectives
1. Formalize the discrete-event cloud-edge scheduling problem as a continuous Markov Decision Process (MDP).
2. Design an asymmetric Cross-Attention encoder providing strict mathematical permutation invariance over unordered server clusters.
3. Develop a fast Denoising Diffusion Implicit Model (DDIM) actor with learnable stochastic exploration parameter $\eta$.
4. Eliminate initialization overhead via a 3-tier decoupled model and process-group reuse policy.
5. Extend the academic baseline to support multi-priority QoS, hard deadlines, SPECpower energy tracking, and dynamic arrival bursts.
6. Build a production-grade FastAPI replay backend and an interactive Next.js 15 visualization dashboard.

### 1.4 Major Contributions
- Summary of novel algorithmic contributions, software engineering standards, and empirical findings.

### 1.5 Report Organization
- Summary roadmap of subsequent chapters.

---

## Chapter 2: Literature Survey

### 2.1 Classical & Heuristic Task Scheduling in Cloud-Edge
- First-In-First-Out (FIFO), Earliest Deadline First (EDF), and Greedy heuristics.
- Limitations of static heuristics in dynamic, non-stationary arrival environments.

### 2.2 Deep Reinforcement Learning for Resource Allocation
- Overview of Deep Q-Networks (DQN), Policy Gradients (PPO), and Actor-Critic algorithms (DDPG, TD3, SAC).
- Limitations of unimodal Gaussian policy assumptions in multimodal discrete-continuous scheduling decisions.

### 2.3 Diffusion Models in Reinforcement Learning
- Diffusion probabilistic models for offline RL and continuous control (Diffusion-QL, DPPO, D2SAC).
- Comparison of DDPM ($T=1000$ steps) vs. accelerated DDIM ($T=4$ steps).

### 2.4 Distributed AIGC Inference & Gang Parallelism
- Spatial patch splitting (DistriFusion) and inter-node halo exchanges.
- Process group initialization delays and communication scaling bottlenecks.

### 2.5 Comparative Analysis & Research Gap
- **Table 2.1**: Comprehensive feature comparison matrix:
  - *Columns:* System / Paper, Scheduling Model, Action Space, Permutation Invariance, Init Overhead Modeling, Multi-QoS Support, Energy Modeling.
  - *Rows:* Classical Schedulers, Vanilla SAC/PPO, D2SAC, Xu et al. (IEEE TMC 2026), EAT-Sched (Proposed).

---

## Chapter 3: System Modeling & Problem Formulation

### 3.1 Cloud-Edge Infrastructure Model
- Multi-tier topology: Local edge nodes (Orin/RTX 4090) vs. Centralized cloud accelerators (A100/H100).
- Backhaul network model: Edge LAN (<1ms) vs. Cloud WAN (120ms round trip + jitter).
- **Figure 3.1**: Cloud-Edge distributed system topology and data flow diagram.

### 3.2 Task & Workload Characterization
- Heterogeneous task classes: AIGC Diffusion, ML Inference, Video Transcoding, CPU Batch.
- Spatial patch decomposition ($c_k \in \{1, 2, 4, 8\}$) and sublinear speedup formulation.
- Poisson arrival process with non-stationary rate drift ($\lambda_t$).
- **Table 3.1**: Task profile specifications, compute loads, memory requirements, and deadlines.

### 3.3 Three-Tier Initialization Cost Model
- Mathematical formulation: Cold start ($t^{init} = 30.0$s), Reconfiguration ($t^{init} = 1.76$s), Instant Reuse ($t^{init} = 0.0$s).
- Active process-group lifecycle and memory residency constraints.

### 3.4 Quality & Energy Consumption Models
- Saturating exponential perceptual quality model: $q(s_k) = q_{max}(1 - e^{-s_k/\tau_q})$.
- SPECpower linear energy formulation: $P(u) = P_{idle} + (P_{busy} - P_{idle}) \cdot u$.

### 3.5 Discrete-Event MDP Formulation
- **State Space**: Edge cluster state matrix $s_t^{Edge}$, Queue state matrix $s_t^{Queue}$, and padding mask $M_t$.
- **Action Space**: Continuous action vector $a_t = [a_c, a_s, a_k] \in [-1, 1]^3$.
- **Safe No-Op Invariant**: Preservation of state consistency during resource starvation.
- **Multi-Objective Reward Function**: Paper reward (Eq. 9) and Extended QoS reward ($R_t - \text{penalties}$).

---

## Chapter 4: Attention-Guided Diffusion RL Architecture

### 4.1 Permutation-Invariant Cross-Attention Encoder
- The $|E|!$ state explosion problem in flattened multi-server representations.
- Mathematical formulation: Queue tasks as Queries ($Q$), Edge servers as Keys ($K$) and Values ($V$).
- Mathematical proof of strict server permutation invariance: $H_{\pi} = H$.
- **Figure 4.1**: Detailed tensor architecture of the Cross-Attention Encoder.

### 4.2 DDIM Diffusion Policy Network
- Reverse denoising chain ($T=4$) starting from Gaussian noise $x_T \sim \mathcal{N}(0, I)$.
- Mish-activated denoising neural network $\epsilon_\theta(x_t, t, c)$.
- Learnable stochasticity parameter $\eta = \text{sigmoid}(w_\eta) \cdot \eta_{max}$.
- Numerical clamping and gradient flow preservation.
- **Figure 4.2**: DDIM 4-step reverse sampling process with learnable $\eta$.

### 4.3 Twin-Critic Actor-Critic Optimization
- Clipped Double-Q critics to eliminate value overestimation bias.
- Terminal done masking invariant in Bellman bootstrapping: $y = r + \gamma(1 - done)\min Q$.
- Polyak target network soft updates ($\tau = 0.005$).

### 4.4 Scheduling Algorithms
- **Algorithm 4.1**: Deterministic Server Selector (Alg. 1 from paper).
- **Algorithm 4.2**: End-to-End EAT Training and Online Inference Pipeline.

---

## Chapter 5: System Implementation & Engineering Architecture

### 5.1 Simulation Environment (`eatsched.env`)
- Gymnasium-compliant discrete-event simulation engine (`CloudEdgeSchedEnv`).
- Event queue management: `TASK_ARRIVAL` and `SERVER_RELEASE`.
- Deterministic random number generation and seed control.

### 5.2 Agent Modules & Baseline Implementations
- PyTorch implementation of EAT, Cross-Attention, and DDIM policy.
- Non-learned heuristics: Random, FIFO, Greedy, EDF, Priority-FIFO.
- Learned baselines: SAC, PPO, TD3, DDPG, DPPO, D2SAC.
- Architectural ablations: EAT-D (No DDIM), EAT-CA (No Attn), D2SAC-SA (Self-Attn).

### 5.3 FastAPI Production Backend (`api/`)
- Endpoints: `/health`, `/clusters`, `/schedulers`, `/simulate`, `/results/{exp}`, `/stream`.
- Deterministic seed alignment between CLI and Web engine.

### 5.4 Next.js 15 Interactive Dashboard (`web/`)
- Responsive UI components: Multi-server Gantt timeline chart, radar comparison, learning curves.
- Dual-mode architecture: Live API streaming + Offline static results fallback.
- **Figure 5.1**: Screenshot / layout of the interactive Web dashboard.

### 5.5 Software Quality & CI Verification
- Strict typing with `mypy` and PEP 8 linting with `ruff`.
- 22 automated unit and integration tests covering math, plumbing, and invariants.

---

## Chapter 6: Experimental Evaluation & Results

### 6.1 Experimental Setup & Evaluation Methodology
- Hardware and software testbed specifications (x86_64 CPU benchmark environment).
- Cluster scales: `edge32` (32 nodes), `edge48` (48 nodes), `edge64` (64 nodes), `cloud_edge` (24 edge + 8 cloud).
- Benchmark metrics: Response Time, Queue Waiting Time, Init Overhead, Reuse Rate, Quality, Utility.

### 6.2 Training Dynamics & Convergence
- Smoke training curve: 20,000 steps on CPU (~5.4 minutes).
- Reward progression: Climbing from initial $18.22$ to converged $25.23$.
- Evolution of learnable exploration parameter $\eta$.
- **Figure 6.1**: Training reward convergence and $\eta$ trajectory over 20k steps.

### 6.3 Benchmark Comparison across 13 Schedulers
- **Table 6.1**: Benchmark comparison table (Mean ± Std over 5 deterministic seeds) on `edge32` (`paper_mode`).
- Detailed comparison against non-learned heuristics (FIFO, Greedy, Random, EDF, P-FIFO).
- Detailed comparison against learned RL baselines (SAC, PPO, TD3, DDPG, DPPO, D2SAC).
- **Figure 6.2**: Response time vs. Utility Pareto frontier across all 13 schedulers.

### 6.4 Cluster Scalability Analysis
- Performance scaling across `edge32`, `edge48`, and `edge64`.
- Verification of $O(|E|)$ linear computational scaling of Cross-Attention vs. quadratic self-attention.
- **Figure 6.3**: Execution time and reward scaling across cluster sizes.

### 6.5 Dynamic Workload Adaptation & Resilience
- Performance under sudden arrival traffic bursts ($\lambda: 0.3 \to 0.8$).
- Resilience under sudden node dropouts (unannounced server failures).
- **Figure 6.4**: Real-time queue backlog and recovery response under dynamic drift.

### 6.6 Ablation Studies & Architectural Justifications
- Impact of Diffusion Policy (EAT vs. EAT-D): Multimodal vs. Unimodal action selection.
- Impact of Cross-Attention (EAT vs. EAT-CA): Permutation invariance vs. order sensitivity.
- Impact of Cross-Attention vs. Self-Attention (EAT vs. D2SAC-SA).
- **Table 6.2**: Summary of ablation experiment metrics.

### 6.7 Honest Discussion & Paper Discrepancy Analysis
- Honest analysis of conditions where Greedy outperforms EAT (low-load myopic regimes).
- Justification of differences between reported paper values and our implementation (`ASM-01` to `ASM-12`).

---

## Chapter 7: Conclusion & Future Work

### 7.1 Summary of Findings & Achievements
- Synthesis of how EAT-Sched successfully resolves the AIGC edge scheduling trade-off.
- Reflection on engineering principles: empirical honesty, zero mocks, and reproducible pipelines.

### 7.2 Limitations of the Current Study
- Simulated diffusion quality model vs. live neural image rendering.
- Simulated WAN propagation delay vs. physical cross-continental networking.
- Single-machine CPU benchmark execution vs. physical 32-node edge rack.

### 7.3 Future Research Directions
- Integration of live physical GPU clusters with NVIDIA Triton and vLLM backends.
- Extension to multi-modal text-to-video (Sora/VideoCrafter) multi-frame generation.
- Hierarchical multi-agent federated scheduling across multi-tier regional edge hubs.

---

## References & Appendices
- **References**: 40+ IEEE/ACM citations formatted in standard IEEE Transactions style.
- **Appendix A**: Specification Compliance Matrix (`docs/AUDIT.md`).
- **Appendix B**: Traceable Engineering Assumptions (`docs/ASSUMPTIONS.md`).
- **Appendix C**: Viva & Oral Examination Guide (`docs/VIVA.md`).
- **Appendix D**: Complete Project Execution & Reproduction Commands.
