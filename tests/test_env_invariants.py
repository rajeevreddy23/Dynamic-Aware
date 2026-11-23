"""Tests verifying simulation environment and mathematical invariants."""

import numpy as np

from eatsched.env.cluster import Server, ServerCluster
from eatsched.env.sched_env import CloudEdgeSchedEnv
from eatsched.env.selector import GreedyServerSelector
from eatsched.env.tasks import PARALLEL_SPEEDUP, Task, sample_task


def test_task_sampling_and_parallelism():
    """Verify task sampling respects parallelism distribution and speedups."""
    rng = np.random.default_rng(42)
    tasks = [sample_task(i, arrival_time=float(i), rng=rng, mode="paper_mode") for i in range(100)]
    for t in tasks:
        assert t.parallelism in {1, 2, 4, 8}
        assert t.steps == 40
        assert t.priority == 2

    # Monotonic latency reduction with parallelism [ASM-01]
    t1 = Task(0, "p", parallelism=1, arrival_time=0.0).compute_execution_time(1.0)
    t2 = Task(0, "p", parallelism=2, arrival_time=0.0).compute_execution_time(1.0)
    t4 = Task(0, "p", parallelism=4, arrival_time=0.0).compute_execution_time(1.0)
    t8 = Task(0, "p", parallelism=8, arrival_time=0.0).compute_execution_time(1.0)

    assert t1 > t2 > t4 > t8
    assert PARALLEL_SPEEDUP[8] == 2.45


def test_greedy_selector_algorithm_1():
    """Verify Algorithm 1 for reuse, cold start, and in-memory reconfiguration."""
    servers = [Server(server_id=i, speed=1.0) for i in range(8)]
    cluster = ServerCluster(servers)
    selector = GreedyServerSelector()

    # Initially all cold (group_id = 0)
    selected, delay, r_type = selector.select_servers(cluster, c_k=2)
    assert selected is not None and len(selected) == 2
    assert delay == 30.0
    assert r_type == "cold_start"
    gid1 = selected[0].group_id
    assert gid1 > 0

    # Mark them busy and then idle to preserve group_id
    for s in selected:
        s.available = True

    # Now request matching c_k = 2 -> Should hit Step 1 (Reuse) with 0.0s delay [PAPER]
    selected2, delay2, r_type2 = selector.select_servers(cluster, c_k=2)
    assert selected2 is not None and len(selected2) == 2
    assert delay2 == 0.0
    assert r_type2 == "reuse"
    assert selected2[0].group_id == gid1


def test_safe_noop_invariant():
    """Verify safe no-op on invalid action: time advances, no state corruption."""
    env = CloudEdgeSchedEnv(cluster_name="edge32", mode="paper_mode", seed=42)
    obs, _ = env.reset(seed=42)
    initial_queue_len = len(env.task_queue)
    initial_time = env.current_time

    # Send idle action (a_c <= 0)
    action = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert not info["scheduled"]
    assert env.current_time == initial_time + env.dt
    # No tasks removed incorrectly
    assert len(env.task_queue) >= initial_queue_len
