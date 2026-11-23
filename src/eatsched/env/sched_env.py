"""Gymnasium-compatible discrete-event Cloud-Edge Scheduling Environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from eatsched.env.cluster import ServerCluster
from eatsched.env.metrics import MetricTracker
from eatsched.env.quality import QualityBackend, SimulatedQualityBackend
from eatsched.env.rewards import (
    compute_extended_reward,
    compute_idle_reward,
    compute_paper_reward,
)
from eatsched.env.selector import GreedyServerSelector
from eatsched.env.tasks import TaskQueue, sample_task
from eatsched.utils.config import get_project_root, load_yaml


class CloudEdgeSchedEnv(gym.Env):
    """Discrete-Event Reinforcement Learning Environment for Cloud-Edge Scheduling."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        cluster_name: str = "edge32",
        mode: str = "paper_mode",
        time_limit: float = 1024.0,
        max_tasks: int = 32,
        top_l: int = 8,
        w_q: float = 0.5,
        w_t1: float = 0.3,
        w_t2: float = 0.2,
        w_dl: float = 0.5,
        w_e: float = 0.1,
        seed: Optional[int] = None,
        custom_cluster_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.cluster_name = cluster_name
        self.time_limit = time_limit
        self.max_tasks = max_tasks
        self.top_l = top_l
        self.w_q = w_q
        self.w_t1 = w_t1
        self.w_t2 = w_t2
        self.w_dl = w_dl
        self.w_e = w_e
        self.dt = 1.0  # Time step delta in seconds [PAPER Sec. C3]

        self.rng = np.random.default_rng(seed)

        # Load cluster configuration
        if custom_cluster_cfg is not None:
            self.cluster_cfg = custom_cluster_cfg
        else:
            root = get_project_root()
            cfg_path = root / "configs" / "clusters" / f"{cluster_name}.yaml"
            self.cluster_cfg = load_yaml(cfg_path)

        self.cluster = ServerCluster.from_config(self.cluster_cfg)
        self.num_servers = self.cluster.num_servers
        self.arrival_rate = float(self.cluster_cfg.get("arrival_rate", 0.3))
        self.c_dist = self.cluster_cfg.get("queue", {}).get(
            "c_distribution", [0.25, 0.25, 0.25, 0.25]
        )

        init_delays = self.cluster_cfg.get("init_delay", {})
        self.selector = GreedyServerSelector(
            cold_start_s=float(init_delays.get("cold_start_s", 30.0)),
            reconfig_s=float(init_delays.get("reconfig_s", 1.76)),
            reuse_s=float(init_delays.get("reuse_s", 0.0)),
        )

        self.quality_backend: QualityBackend = SimulatedQualityBackend()
        self.task_queue = TaskQueue(top_l=self.top_l)
        self.metrics = MetricTracker()

        # Action Space: continuous 3-vector [a_c, a_s, a_k] in [-1, 1]^3 [PAPER Sec. D1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation Space [PAPER Sec. D1, ABSTRACT]
        # In paper mode: edge state (num_servers, 3), queue state (top_l, 2), mask (top_l,)
        # In extended mode: edge state (num_servers, 5), queue state (top_l, 6), mask (top_l,)
        edge_dim = 3 if self.mode == "paper_mode" else 5
        queue_dim = 2 if self.mode == "paper_mode" else 6

        self.observation_space = spaces.Dict(
            {
                "edge": spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_servers, edge_dim), dtype=np.float32
                ),
                "queue": spaces.Box(
                    low=0.0, high=1.0, shape=(self.top_l, queue_dim), dtype=np.float32
                ),
                "mask": spaces.Box(low=0, high=1, shape=(self.top_l,), dtype=np.int8),
            }
        )

        # State tracking
        self.current_time: float = 0.0
        self.next_arrival_time: float = 0.0
        self.next_task_id: int = 0
        self.decision_steps: int = 0
        self.tasks_completed_count: int = 0
        self.timeline_events: List[Dict[str, Any]] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.cluster.reset()
        self.task_queue = TaskQueue(top_l=self.top_l)
        self.metrics.reset()

        self.current_time = 0.0
        self.decision_steps = 0
        self.tasks_completed_count = 0
        self.next_task_id = 0
        self.timeline_events.clear()

        # Schedule first arrival
        self.next_arrival_time = float(self.rng.exponential(1.0 / self.arrival_rate))
        self._check_arrivals()

        # Pre-populate queue with initial tasks so agent has initial choices
        while len(self.task_queue) < min(3, self.top_l):
            self.current_time = self.next_arrival_time
            self._check_arrivals()

        obs = self._get_obs()
        info = {"time": self.current_time, "queue_len": len(self.task_queue)}
        return obs, info

    def _check_arrivals(self) -> None:
        """Process Poisson arrivals up to current simulation time."""
        while self.next_arrival_time <= self.current_time:
            task = sample_task(
                task_id=self.next_task_id,
                arrival_time=self.next_arrival_time,
                rng=self.rng,
                mode=self.mode,
                c_dist=self.c_dist,
            )
            self.task_queue.push(task)
            self.next_task_id += 1
            inter_arrival = float(self.rng.exponential(1.0 / self.arrival_rate))
            self.next_arrival_time += inter_arrival

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Construct normalized observation dictionary."""
        # 1. Edge state matrix
        edge_dim = 3 if self.mode == "paper_mode" else 5
        edge_mat = np.zeros((self.num_servers, edge_dim), dtype=np.float32)

        for i, srv in enumerate(self.cluster.servers):
            avail = 1.0 if srv.available else 0.0
            rem_norm = np.clip(srv.remaining_time / self.time_limit, 0.0, 1.0)
            group_norm = np.clip(srv.group_id / 20.0, 0.0, 1.0)

            if self.mode == "paper_mode":
                edge_mat[i] = [avail, rem_norm, group_norm]
            else:
                tier_val = 1.0 if srv.tier == "cloud" else 0.0
                power_val = (srv.power() - 50.0) / 450.0
                edge_mat[i] = [avail, rem_norm, group_norm, tier_val, power_val]

        # 2. Queue state matrix & padding mask
        queue_dim = 2 if self.mode == "paper_mode" else 6
        queue_mat = np.zeros((self.top_l, queue_dim), dtype=np.float32)
        mask = np.ones(self.top_l, dtype=np.int8)  # 1 = masked/empty, 0 = valid

        visible_tasks = self.task_queue.peek_top()
        for i, task in enumerate(visible_tasks):
            wait_norm = np.clip((self.current_time - task.arrival_time) / self.time_limit, 0.0, 1.0)
            c_norm = task.parallelism / 8.0
            mask[i] = 0  # Valid task

            if self.mode == "paper_mode":
                queue_mat[i] = [wait_norm, c_norm]
            else:
                p_norm = task.priority / 3.0
                slack_norm = np.clip(
                    (task.absolute_deadline - self.current_time) / self.time_limit,
                    0.0,
                    1.0,
                )
                q_req = task.quality_threshold
                workload_map = {
                    "aigc_sd": 0.25,
                    "ml_inference": 0.50,
                    "video_analytics": 0.75,
                    "cpu_batch": 1.00,
                }
                w_type = workload_map.get(task.workload_type, 0.25)
                queue_mat[i] = [wait_norm, c_norm, p_norm, slack_norm, q_req, w_type]

        return {"edge": edge_mat, "queue": queue_mat, "mask": mask}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Advance simulation by one RL decision step."""
        self.decision_steps += 1
        a_c, a_s, a_k = float(action[0]), float(action[1]), float(action[2])

        # Map continuous actions [PAPER Sec. D1]
        # s_k in [20, 60] (ASM-03)
        steps = int(np.round(20 + ((a_s + 1.0) / 2.0) * 40))
        steps = max(20, min(60, steps))

        # Check if queue has tasks
        visible_tasks = self.task_queue.peek_top()
        num_visible = len(visible_tasks)

        reward = 0.0
        scheduled = False
        reuse_type = "none"

        # Action execution condition: a_c > 0 and queue non-empty [PAPER Alg. 1 Line 14, ASM-04]
        if a_c > 0 and num_visible > 0:
            task_idx = int(np.floor(((a_k + 1.0) / 2.0) * num_visible))
            task_idx = max(0, min(num_visible - 1, task_idx))
            task = visible_tasks[task_idx]

            # Query server allocation
            selected_servers, init_delay, r_type = self.selector.select_servers(
                self.cluster, task.parallelism
            )

            if selected_servers is not None:
                # Valid allocation! Pop task from queue
                self.task_queue.pop(task_idx)
                scheduled = True
                reuse_type = r_type

                task.steps = steps
                task.dispatched_time = self.current_time
                task.start_execution_time = self.current_time + init_delay
                task.init_delay = init_delay
                task.allocated_servers = [s.server_id for s in selected_servers]

                # Compute execution time & quality
                min_speed = min(s.speed for s in selected_servers)
                wan_latency = max(s.wan_latency_s for s in selected_servers)
                exec_time = task.compute_execution_time(min_speed, wan_latency)
                task.execution_time = exec_time
                task.completion_time = self.current_time + init_delay + exec_time
                task.quality = self.quality_backend.compute_quality(task.steps, self.rng)

                # Set servers busy
                total_duration = init_delay + exec_time
                for s in selected_servers:
                    s.available = False
                    s.remaining_time = total_duration
                    s.current_task_id = task.task_id

                # Calculate response time and rewards [PAPER Eq. 9, ABSTRACT]
                wait_time = self.current_time - task.arrival_time
                resp_time = wait_time + total_duration
                avg_q_wait = self.task_queue.average_waiting_time(self.current_time)

                r_paper = compute_paper_reward(
                    quality=task.quality,
                    response_time=resp_time,
                    avg_queue_wait=avg_q_wait,
                    w_q=self.w_q,
                    w_t1=self.w_t1,
                    w_t2=self.w_t2,
                )

                if self.mode == "paper_mode":
                    reward = r_paper
                else:
                    energy_task = sum(s.power_busy_watts * total_duration for s in selected_servers)
                    reward = compute_extended_reward(
                        paper_reward=r_paper,
                        response_time=resp_time,
                        relative_deadline=task.relative_deadline,
                        priority=task.priority,
                        energy_joules=energy_task,
                        quality=task.quality,
                        quality_threshold=task.quality_threshold,
                        w_dl=self.w_dl,
                        w_e=self.w_e,
                    )

                self.metrics.record_task_completion(task, reuse_type)
                self.tasks_completed_count += 1

                # Record timeline event for visualization & replay
                self.timeline_events.append(
                    {
                        "task_id": task.task_id,
                        "prompt": task.prompt,
                        "parallelism": task.parallelism,
                        "servers": task.allocated_servers,
                        "start_time": round(self.current_time, 2),
                        "init_delay": round(init_delay, 2),
                        "exec_time": round(exec_time, 2),
                        "completion_time": round(task.completion_time, 2),
                        "reuse_type": reuse_type,
                        "quality": round(task.quality, 3),
                        "priority": task.priority,
                        "workload_type": task.workload_type,
                    }
                )

        if not scheduled:
            # Safe No-Op: Backlog penalty applied, cluster time advances [ASM-07]
            avg_q_wait = self.task_queue.average_waiting_time(self.current_time)
            reward = compute_idle_reward(avg_q_wait, w_t2=self.w_t2)

        # Advance continuous simulation time by dt [PAPER Sec. C3]
        self.current_time += self.dt
        self.cluster.advance_time(self.dt)
        self._check_arrivals()

        # Termination conditions
        terminated = (
            self.tasks_completed_count >= self.max_tasks or self.current_time >= self.time_limit
        )
        truncated = self.decision_steps >= 1024

        obs = self._get_obs()
        info = {
            "scheduled": scheduled,
            "reuse_type": reuse_type,
            "tasks_completed": self.tasks_completed_count,
            "current_time": self.current_time,
            "queue_len": len(self.task_queue),
            "metrics": self.metrics.summary(),
        }

        return obs, float(reward), terminated, truncated, info
