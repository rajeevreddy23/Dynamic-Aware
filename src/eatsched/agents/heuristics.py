"""Non-learned heuristic schedulers for EAT-Sched [PAPER Sec. VII-A, ABSTRACT]."""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import numpy as np

from eatsched.env.rewards import compute_extended_reward, compute_paper_reward
from eatsched.env.tasks import PARALLEL_SPEEDUP


class BaseScheduler(abc.ABC):
    """Abstract base class for all scheduling agents."""

    @abc.abstractmethod
    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        env: Optional[Any] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select action [a_c, a_s, a_k] in [-1, 1]^3 given current observation."""
        raise NotImplementedError


class RandomScheduler(BaseScheduler):
    """Uniform random scheduler baseline [PAPER]."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        env: Optional[Any] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        # Uniform sampling in [-1, 1]^3
        return self.rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)


class FIFOScheduler(BaseScheduler):
    """First-In-First-Out scheduler baseline [PAPER].

    Schedules the head of the queue (index 0) with fixed 40 inference steps.
    """

    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        env: Optional[Any] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        mask = obs["mask"]
        if mask[0] == 1:  # Queue is empty
            return np.array([-1.0, 0.0, -1.0], dtype=np.float32)

        # a_c = 1.0 (schedule), a_s = 0.0 (40 steps), a_k = -1.0 (index 0)
        return np.array([1.0, 0.0, -1.0], dtype=np.float32)


class GreedyScheduler(BaseScheduler):
    """Greedy immediate-reward maximizing scheduler baseline [PAPER].

    Evaluates candidate tasks in top-l and candidate step choices {20, 40, 60}.
    Dispatches the task and step combination yielding the highest immediate reward.
    """

    def __init__(self, mode: str = "paper_mode") -> None:
        self.mode = mode

    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        env: Optional[Any] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        if env is None or len(env.task_queue) == 0:
            return np.array([-1.0, 0.0, -1.0], dtype=np.float32)

        visible_tasks = env.task_queue.peek_top()
        best_reward = -float("inf")
        best_k_idx = 0
        best_a_s = 0.0

        step_candidates = [(20, -1.0), (40, 0.0), (60, 1.0)]

        for k_idx, task in enumerate(visible_tasks):
            # Check if cluster can allocate
            idle_servers = env.cluster.get_idle_servers()
            if len(idle_servers) < task.parallelism:
                continue

            # Simulate greedy selection
            idle_groups = env.cluster.get_idle_groups()
            matching = [g for g in idle_groups.values() if len(g) == task.parallelism]
            if matching:
                init_delay = env.selector.reuse_s
                allocated_speeds = [s.speed for s in matching[0]]
                allocated_powers = [s.power_busy_watts for s in matching[0]]
            else:
                zero_srvs = [s for s in idle_servers if s.group_id == 0]
                if len(zero_srvs) >= task.parallelism:
                    init_delay = env.selector.cold_start_s
                    allocated_speeds = [s.speed for s in zero_srvs[: task.parallelism]]
                    allocated_powers = [s.power_busy_watts for s in zero_srvs[: task.parallelism]]
                else:
                    init_delay = env.selector.reconfig_s
                    allocated_speeds = [s.speed for s in idle_servers[: task.parallelism]]
                    allocated_powers = [
                        s.power_busy_watts for s in idle_servers[: task.parallelism]
                    ]

            min_speed = min(allocated_speeds) if allocated_speeds else 1.0
            speedup = PARALLEL_SPEEDUP.get(task.parallelism, 1.0)

            for steps, a_s in step_candidates:
                exec_time = (steps * 0.474) / (speedup * max(min_speed, 1e-6))
                quality = env.quality_backend.compute_quality(steps, None)

                wait_time = env.current_time - task.arrival_time
                resp_time = wait_time + init_delay + exec_time
                avg_q_wait = env.task_queue.average_waiting_time(env.current_time)

                r = compute_paper_reward(
                    quality=quality,
                    response_time=resp_time,
                    avg_queue_wait=avg_q_wait,
                    w_q=env.w_q,
                    w_t1=env.w_t1,
                    w_t2=env.w_t2,
                )

                if self.mode != "paper_mode":
                    total_dur = init_delay + exec_time
                    energy = sum(p * total_dur for p in allocated_powers)
                    r = compute_extended_reward(
                        paper_reward=r,
                        response_time=resp_time,
                        relative_deadline=task.relative_deadline,
                        priority=task.priority,
                        energy_joules=energy,
                        quality=quality,
                        quality_threshold=task.quality_threshold,
                        w_dl=env.w_dl,
                        w_e=env.w_e,
                    )

                if r > best_reward:
                    best_reward = r
                    best_k_idx = k_idx
                    best_a_s = a_s

        if best_reward == -float("inf"):
            # No valid servers available
            return np.array([-1.0, 0.0, -1.0], dtype=np.float32)

        # Map k_idx back to [-1, 1]
        num_v = len(visible_tasks)
        a_k = (float(best_k_idx) / max(1, num_v)) * 2.0 - 1.0 + (1.0 / num_v)
        a_k = max(-1.0, min(1.0, a_k))

        return np.array([1.0, best_a_s, a_k], dtype=np.float32)


class EDFScheduler(BaseScheduler):
    """Earliest Deadline First scheduler baseline [ABSTRACT]."""

    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        env: Optional[Any] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        if env is None or len(env.task_queue) == 0:
            return np.array([-1.0, 0.0, -1.0], dtype=np.float32)

        visible_tasks = env.task_queue.peek_top()
        best_k_idx = -1
        min_deadline = float("inf")

        for i, task in enumerate(visible_tasks):
            if task.absolute_deadline < min_deadline:
                min_deadline = task.absolute_deadline
                best_k_idx = i

        if best_k_idx == -1:
            return np.array([-1.0, 0.0, -1.0], dtype=np.float32)

        num_v = len(visible_tasks)
        a_k = (float(best_k_idx) / max(1, num_v)) * 2.0 - 1.0 + (1.0 / num_v)
        return np.array([1.0, 0.0, float(np.clip(a_k, -1.0, 1.0))], dtype=np.float32)


class PriorityFIFOScheduler(BaseScheduler):
    """Priority-aware FIFO scheduler baseline [ABSTRACT]."""

    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        env: Optional[Any] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        if env is None or len(env.task_queue) == 0:
            return np.array([-1.0, 0.0, -1.0], dtype=np.float32)

        visible_tasks = env.task_queue.peek_top()
        # Sort indices primarily by priority descending, secondarily by arrival time ascending
        indexed_tasks = list(enumerate(visible_tasks))
        indexed_tasks.sort(key=lambda item: (-item[1].priority, item[1].arrival_time))

        best_k_idx = indexed_tasks[0][0]
        num_v = len(visible_tasks)
        a_k = (float(best_k_idx) / max(1, num_v)) * 2.0 - 1.0 + (1.0 / num_v)
        return np.array([1.0, 0.0, float(np.clip(a_k, -1.0, 1.0))], dtype=np.float32)
