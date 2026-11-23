"""Dynamic adaptation experiment harness (Workload bursts, Server failures) [ABSTRACT Sec. E6]."""

from __future__ import annotations

from typing import Any, Dict

from eatsched.agents.eat import EATAgent
from eatsched.agents.heuristics import EDFScheduler, FIFOScheduler, GreedyScheduler
from eatsched.env.sched_env import CloudEdgeSchedEnv


def run_adaptation_experiment(
    cluster: str = "cloud_edge",
    burst_factor: float = 2.0,
    burst_start: float = 200.0,
    burst_duration: float = 150.0,
    failure_count: int = 8,
    failure_start: float = 400.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulate cluster under dynamic workload surge and transient server failure."""
    env = CloudEdgeSchedEnv(cluster_name=cluster, mode="extended_mode", seed=seed)
    base_arrival_rate = env.arrival_rate

    schedulers = {
        "EAT": EATAgent(queue_dim=6, edge_dim=5, device="cpu"),
        "Greedy": GreedyScheduler(mode="extended_mode"),
        "EDF": EDFScheduler(),
        "FIFO": FIFOScheduler(),
    }

    results = {}

    for name, sched in schedulers.items():
        obs, _ = env.reset(seed=seed)
        done = False
        step = 0

        while not done:
            t = env.current_time

            # 1. Bursty arrival rate injection
            if burst_start <= t <= burst_start + burst_duration:
                env.arrival_rate = base_arrival_rate * burst_factor
            else:
                env.arrival_rate = base_arrival_rate

            # 2. Transient server failure injection
            if failure_start <= t <= failure_start + 100.0:
                for i in range(min(failure_count, len(env.cluster.servers))):
                    env.cluster.servers[i].available = False
                    env.cluster.servers[i].remaining_time = max(
                        env.cluster.servers[i].remaining_time, 10.0
                    )

            if name == "EAT":
                action = sched.select_action(obs, deterministic=True)
            elif name == "Greedy":
                action = sched.select_action(obs, env=env)
            else:
                action = sched.select_action(obs)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1

        results[name] = env.metrics.summary()

    return results
