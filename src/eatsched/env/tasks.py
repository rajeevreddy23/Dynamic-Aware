"""Task model and queue structures for EAT-Sched."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

# Parallelism speedup curve from IEEE TMC 2026 Table I & Fig. 4 [PAPER] + ASM-01
PARALLEL_SPEEDUP = {
    1: 1.00,
    2: 1.59,
    4: 1.91,
    8: 2.45,  # ASM-01
}
BASE_STEP_TIME_S = 0.474  # 23.7s for 50 steps at c=1 [PAPER]


@dataclass
class Task:
    """Represents a computational task arriving at the scheduling system."""

    task_id: int
    prompt: str
    parallelism: int  # c_k in {1, 2, 4, 8}
    arrival_time: float  # t_k^a
    steps: int = 40  # s_k in [20, 60]

    # Extended QoS parameters [ABSTRACT]
    priority: int = 2  # p_k in {1, 2, 3} (1=Low, 2=Standard, 3=High)
    relative_deadline: float = 60.0  # d_k^{rel}
    quality_threshold: float = 0.60  # q_k^{min}
    workload_type: str = "aigc_sd"  # aigc_sd, ml_inference, video_analytics, cpu_batch

    # Lifecycle tracking
    dispatched_time: Optional[float] = None
    start_execution_time: Optional[float] = None
    completion_time: Optional[float] = None
    init_delay: float = 0.0
    execution_time: float = 0.0
    allocated_servers: List[int] = field(default_factory=list)
    quality: float = 0.0

    @property
    def absolute_deadline(self) -> float:
        """Absolute deadline timestamp d_k^{abs}."""
        return self.arrival_time + self.relative_deadline

    def compute_execution_time(self, min_server_speed: float, wan_latency: float = 0.0) -> float:
        """Compute execution time t_k^e based on gang parallelism and server speed."""
        speedup = PARALLEL_SPEEDUP.get(self.parallelism, 1.0)
        t_exec = (self.steps * BASE_STEP_TIME_S) / (speedup * max(min_server_speed, 1e-6))
        return t_exec + wan_latency


class TaskQueue:
    """FIFO Waiting Queue Q(t) ordered strictly by arrival time."""

    def __init__(self, top_l: int = 8) -> None:
        self.top_l = top_l
        self.tasks: List[Task] = []

    def push(self, task: Task) -> None:
        """Enqueue task maintaining arrival time ordering."""
        self.tasks.append(task)
        self.tasks.sort(key=lambda t: t.arrival_time)

    def pop(self, index: int) -> Task:
        """Pop task at specific queue index."""
        return self.tasks.pop(index)

    def peek_top(self) -> List[Task]:
        """View the top-l tasks currently in queue."""
        return self.tasks[: self.top_l]

    def remove_task(self, task_id: int) -> Optional[Task]:
        """Remove and return task by task_id."""
        for i, task in enumerate(self.tasks):
            if task.task_id == task_id:
                return self.tasks.pop(i)
        return None

    def average_waiting_time(self, current_time: float) -> float:
        """Calculate average waiting time across all queued tasks [PAPER Eq. 9]."""
        if not self.tasks:
            return 0.0
        total_wait = sum(max(0.0, current_time - t.arrival_time) for t in self.tasks)
        return total_wait / len(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)


def sample_task(
    task_id: int,
    arrival_time: float,
    rng: np.random.Generator,
    mode: str = "paper_mode",
    c_dist: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    workload_mix: Optional[dict[str, float]] = None,
) -> Task:
    """Sample a new task according to distribution specified in config."""
    c_choices = [1, 2, 4, 8]
    c_k = int(rng.choice(c_choices, p=list(c_dist)))
    prompt = f"task_{task_id}_c{c_k}"

    if mode == "paper_mode":
        return Task(
            task_id=task_id,
            prompt=prompt,
            parallelism=c_k,
            arrival_time=arrival_time,
            steps=40,
            priority=2,
            relative_deadline=120.0,
            quality_threshold=0.6,
            workload_type="aigc_sd",
        )

    # Extended mode [ABSTRACT]
    priority = int(rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2]))
    # Relative deadlines: tight for high priority, relaxed for low
    deadline_base = {1: 90.0, 2: 60.0, 3: 40.0}[priority]
    relative_deadline = float(rng.uniform(deadline_base * 0.8, deadline_base * 1.4))
    quality_threshold = float(rng.uniform(0.55, 0.75))

    workload_type = "aigc_sd"
    if workload_mix:
        types = list(workload_mix.keys())
        probs = list(workload_mix.values())
        workload_type = str(rng.choice(types, p=probs))

    return Task(
        task_id=task_id,
        prompt=prompt,
        parallelism=c_k,
        arrival_time=arrival_time,
        steps=40,
        priority=priority,
        relative_deadline=relative_deadline,
        quality_threshold=quality_threshold,
        workload_type=workload_type,
    )
