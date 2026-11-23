# EAT-Sched 5-Minute Viva Demo Walkthrough Script (docs/DEMO_SCRIPT.md)

**System Name:** EAT-Sched: Dynamic QoS-Aware Task Scheduling in Cloud-Edge Environments  
**Target Duration:** Exactly 5 Minutes  
**Audience:** External University Examiners, Project Evaluation Committee, and Department Faculty  
**Presenter Role:** Lead Student / Developer  

---

## ⏱️ Timeline Overview

| Timestamp | Phase | Primary Action | Core Message to Examiners |
|---|---|---|---|
| **0:00 – 1:00** | **Motivation & Problem** | Slide / Diagram | Why AIGC scheduling on edge is hard; the 30s cold start bottleneck. |
| **1:00 – 2:00** | **Architecture & Novelty** | Architecture Diagram | Cross-attention permutation invariance & DDIM actor with learnable $\eta$. |
| **2:00 – 2:45** | **Test & Rigor** | Terminal (`make test`) | 100% production rigor: 0 mocks, strict mypy types, 22 passing tests. |
| **2:45 – 4:00** | **Live Interactive UI** | Web Dashboard (`:3000`) | Live Gantt chart, process group reuse (0s vs 30s), static fallback mode. |
| **4:00 – 5:00** | **Honest Results & Q&A** | Results Table | Real measured numbers across 13 schedulers; where EAT wins vs. Greedy. |

---

## 🎬 Minute-by-Minute Script & Actions

### 0:00 – 1:00: The Problem Statement (The "Why")

**[ACTION: Open README.md or Slide showing Cloud-Edge Cluster & DistriFusion]**

**[SPOKEN SCRIPT]:**
> "Good morning, respected examiners and faculty members.
> 
> Our capstone project addresses a critical real-world problem: **Deploying generative diffusion models like Stable Diffusion onto distributed edge servers**.
> 
> Generating an image requires running a billion-parameter neural network over 40 to 50 denoising steps, taking 15 seconds on a single edge device. While we can split an image across 2, 4, or 8 edge servers using patch parallelism, this introduces two major bottlenecks:
> 1. **Diminishing speedups** due to inter-server boundary communication.
> 2. **Devastating initialization delays**: Loading model weights from scratch causes a **30-second cold start**! Even reconfiguring an existing server group costs nearly 2 seconds.
> 
> If a scheduler naively assigns 4 cold servers to run a task faster, total response time explodes to **37 seconds**. But if a smart scheduler dispatches that task to an already-active server group, initialization overhead is **0 seconds**, finishing in under **10 seconds**.
> 
> To solve this multi-objective trade-off between image quality, response time, and initialization delays, we developed **EAT-Sched**."

---

### 1:00 – 2:00: Architecture & Core Innovations (The "How")

**[ACTION: Point to Architecture Diagram in README.md or Web Dashboard]**

**[SPOKEN SCRIPT]:**
> "EAT-Sched is built on three novel reinforcement learning contributions based on the IEEE Transactions on Mobile Computing (2026) paper with capstone extensions:
> 
> 1. **Permutation-Invariant Cross-Attention Encoder**: In a 32-node edge cluster, server order is completely arbitrary. Standard neural networks flatten server states, suffering from a combinatorial explosion of 32 factorial states. Our cross-attention module treats tasks as queries and servers as keys and values. We have mathematically proven that this representation is 100% invariant to any reordering of servers.
> 
> 2. **Accelerated DDIM Policy with Learnable Exploration ($\eta$)**: Unlike traditional RL algorithms like SAC or PPO that assume simple Gaussian actions, scheduling decisions are multimodal. We use a 4-step Denoising Diffusion Implicit Model. Crucially, the stochasticity parameter $\eta$ is made learnable via policy gradients, allowing the agent to explore widely early in training and transition smoothly into deterministic dispatching as it converges.
> 
> 3. **Decoupled 3-Tier Initialization Model**: We explicitly track server process groups, achieving up to **50% process-group reuse**, effectively eliminating cold starts during peak traffic."

---

### 2:00 – 2:45: Production Rigor & Automated Verification

**[ACTION: Switch to Terminal and execute `make test`]**

```powershell
make test
```

**[SPOKEN SCRIPT]:**
> "Before presenting our results, I want to emphasize our commitment to **empirical honesty and production-grade code**. Our repository contains zero stubs, zero mocks, and zero hardcoded metrics.
> 
> In this single command `make test`:
> 1. `ruff` verifies strict PEP 8 compliance across all modules.
> 2. `mypy` performs strict static type checking across 40 files with zero errors.
> 3. `pytest` executes 22 comprehensive unit and integration tests covering mathematical invariants, Bellman done masking, and simulation determinism.
> 
> As you can see on screen: all 22 tests pass in under 15 seconds."

---

### 2:45 – 4:00: Live Interactive UI & Simulation Replay

**[ACTION: Open Web Browser at `http://localhost:3000`]**

**[SPOKEN SCRIPT]:**
> "Now let us examine the system running live on our Next.js interactive scheduling dashboard.
> 
> **[Point to the Cluster Selector and Scheduler dropdown]**  
> We can select our cluster topology—such as `edge32` or our heterogeneous `cloud_edge`—and choose from 13 implemented schedulers.
> 
> **[Click 'Run Simulation' or inspect the live Gantt Timeline]**  
> Look at the multi-server Gantt chart:
> - Notice the color coding: **Red bars** represent 30-second cold starts.
> - **Yellow bars** represent 1.76-second reconfigurations.
> - **Green bars** represent instant **0-second process-group reuses**.
> 
> Under EAT-Sched, observe how consecutive tasks requiring 2 or 4 patches are intelligently held and scheduled on the same active server group. This yields a **40% to 50% reuse rate**, keeping average queue waiting time below 11 seconds.
> 
> **[Show Static Fallback Mode]**  
> Notice that if our FastAPI backend is offline, the web dashboard automatically shifts to static results mode, reading verified benchmark artifacts directly from disk, ensuring high availability during evaluations."

---

### 4:00 – 5:00: Benchmark Results & Honest Conclusion

**[ACTION: Navigate to the 'Evaluation Comparison' view or show Table 5.1 in AUDIT.md]**

**[SPOKEN SCRIPT]:**
> "Finally, let us look at the empirical benchmark results evaluated over 5 deterministic seeds:
> 
> **[Point to Table]**
> - **FIFO** incurs severe queue delays ($12.17$s wait time) because it rigidly runs tasks at 40 steps without adapting to load.
> - **SAC and PPO** struggle with multimodal actions, resulting in higher response times.
> - **Our EAT agent** achieves an average response time of **30.10 seconds** and high utility ($0.402$).
> 
> **[Deliver the Honest Empirical Statement]**  
> In the spirit of scientific honesty: Does EAT win in every scenario?
> **No.** Under extremely light queue loads ($\lambda \le 0.3$), the deterministic `Greedy` heuristic achieves lower latency ($24.62$s). This occurs because when the queue has only 1 or 2 tasks, Greedy can evaluate all 9 candidate actions in closed form with zero neural approximation error.
> 
> However, as cluster size scales to 48 and 64 nodes and queue congestion increases, EAT's permutation-invariant cross-attention and dynamic step adaptation significantly outperform both static heuristics and traditional RL baselines.
> 
> The codebase, documentation, and oral defense guides are complete and reproducible with a single `make all` command.
> 
> Thank you, and I welcome your questions."

---

## 💡 Quick Tips for the Presenter

1. **If the examiner asks: 'Is this running on physical GPUs or a simulator?'**
   - *Answer:* "The discrete-event simulation runs natively on CPU to ensure 100% reproducible testing. However, the models, speedups, and 3-tier initialization timings are strictly calibrated to real empirical measurements from NVIDIA Jetson AGX Orin clusters and DistriFusion benchmarks as published in IEEE TMC 2026."

2. **If the examiner asks: 'Why is your smoke training reward lower than the paper's final figure?'**
   - *Answer:* "Our smoke training runs 20,000 steps (~5.4 minutes on CPU) to prove end-to-end convergence ($18.22 \to 25.23$). The paper uses 1,000,000 steps (~15 GPU hours). The exact command to train the full 1M budget is `python -m eatsched.train --budget full`."

3. **If the examiner asks: 'What happens if a server fails mid-execution?'**
   - *Answer:* "Under our `extended_mode` adaptation tests, the simulator triggers an unannounced node failure event. The safe no-op invariant rejects dispatches to dead nodes and gracefully reconfigures the surviving nodes on the next discrete event."
