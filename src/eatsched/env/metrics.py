"""Evaluation metrics and tracking for EAT-Sched."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from eatsched.env.tasks import Task


@dataclass
class MetricTracker:
    """Tracks and aggregates scheduling performance metrics across episodes [PAPER Sec. C5, ABSTRACT]."""

    completed_tasks: List[Task] = field(default_factory=list)
    total_energy_joules: float = 0.0
    episodes_count: int = 0
    total_reward: float = 0.0
    total_steps: int = 0
    reuse_count: int = 0
    reconfig_count: int = 0
    cold_start_count: int = 0
    deadline_misses: int = 0
    high_priority_misses: int = 0

    def record_task_completion(self, task: Task, reuse_type: str) -> None:
        """Record completed task statistics."""
        self.completed_tasks.append(task)
        if reuse_type == "reuse":
            self.reuse_count += 1
        elif reuse_type == "reconfig":
            self.reconfig_count += 1
        elif reuse_type == "cold_start":
            self.cold_start_count += 1

        # Check deadline in extended mode
        resp_time = (
            (task.completion_time - task.arrival_time) if task.completion_time is not None else 0.0
        )
        if resp_time > task.relative_deadline:
            self.deadline_misses += 1
            if task.priority == 3:
                self.high_priority_misses += 1

    def summary(self) -> Dict[str, Any]:
        """Compute aggregated statistical summary."""
        n = len(self.completed_tasks)
        if n == 0:
            return {
                "num_completed_tasks": 0,
                "avg_response_time_s": 0.0,
                "p95_response_time_s": 0.0,
                "avg_waiting_time_s": 0.0,
                "avg_init_delay_s": 0.0,
                "reuse_rate": 0.0,
                "avg_quality": 0.0,
                "utility": 0.0,
                "total_energy_kj": self.total_energy_joules / 1000.0,
                "deadline_miss_rate": 0.0,
            }

        response_times = [
            (t.completion_time - t.arrival_time)
            for t in self.completed_tasks
            if t.completion_time is not None
        ]
        waiting_times = [
            (t.dispatched_time - t.arrival_time)
            for t in self.completed_tasks
            if t.dispatched_time is not None
        ]
        init_delays = [t.init_delay for t in self.completed_tasks]
        qualities = [t.quality for t in self.completed_tasks]

        avg_resp = float(np.mean(response_times)) if response_times else 0.0
        p95_resp = float(np.percentile(response_times, 95)) if response_times else 0.0
        avg_wait = float(np.mean(waiting_times)) if waiting_times else 0.0
        avg_init = float(np.mean(init_delays)) if init_delays else 0.0
        avg_q = float(np.mean(qualities)) if qualities else 0.0
        reuse_rate = float(self.reuse_count / n) if n > 0 else 0.0
        miss_rate = float(self.deadline_misses / n) if n > 0 else 0.0

        # System Utility U [PAPER Sec. C5]: balances quality against latency
        # U = quality - (response_time / 100.0)
        utility = avg_q - (avg_resp / 100.0)

        return {
            "num_completed_tasks": n,
            "avg_response_time_s": round(avg_resp, 3),
            "p95_response_time_s": round(p95_resp, 3),
            "avg_waiting_time_s": round(avg_wait, 3),
            "avg_init_delay_s": round(avg_init, 3),
            "reuse_rate": round(reuse_rate, 4),
            "avg_quality": round(avg_q, 4),
            "utility": round(utility, 4),
            "total_energy_kj": round(self.total_energy_joules / 1000.0, 2),
            "deadline_miss_rate": round(miss_rate, 4),
            "high_priority_misses": self.high_priority_misses,
            "cold_starts": self.cold_start_count,
            "reconfigs": self.reconfig_count,
            "reuses": self.reuse_count,
        }

    def reset(self) -> None:
        """Reset all tracking counters."""
        self.completed_tasks.clear()
        self.total_energy_joules = 0.0
        self.episodes_count = 0
        self.total_reward = 0.0
        self.total_steps = 0
        self.reuse_count = 0
        self.reconfig_count = 0
        self.cold_start_count = 0
        self.deadline_misses = 0
        self.high_priority_misses = 0
