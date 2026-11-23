"""Tests for Extended Mode QoS, Cloud-Edge heterogeneity, and Dynamic Adaptation."""

from eatsched.env.cluster import Server
from eatsched.env.rewards import compute_extended_reward
from eatsched.experiments.adaptation import run_adaptation_experiment


def test_cloud_edge_heterogeneity():
    """Verify cloud tier has WAN latency and higher compute throughput [ASM-11]."""
    edge_srv = Server(
        server_id=0,
        speed=1.0,
        tier="edge",
        power_idle_watts=80.0,
        power_busy_watts=250.0,
        wan_latency_s=0.0,
    )
    cloud_srv = Server(
        server_id=1,
        speed=2.5,
        tier="cloud",
        power_idle_watts=150.0,
        power_busy_watts=450.0,
        wan_latency_s=0.120,
    )

    assert cloud_srv.speed > edge_srv.speed
    assert cloud_srv.wan_latency_s == 0.120
    assert cloud_srv.power_busy_watts > edge_srv.power_busy_watts


def test_extended_reward_penalties():
    """Verify deadline miss penalty scales with priority."""
    base_r = 1.0
    # Priority 1 miss vs Priority 3 miss
    r_p1 = compute_extended_reward(
        paper_reward=base_r,
        response_time=50.0,
        relative_deadline=40.0,
        priority=1,
        energy_joules=100.0,
        quality=0.8,
        quality_threshold=0.6,
        w_dl=0.5,
    )
    r_p3 = compute_extended_reward(
        paper_reward=base_r,
        response_time=50.0,
        relative_deadline=40.0,
        priority=3,
        energy_joules=100.0,
        quality=0.8,
        quality_threshold=0.6,
        w_dl=0.5,
    )

    assert r_p1 > r_p3, "Higher priority deadline miss must incur larger penalty!"


def test_adaptation_experiment():
    """Verify burst arrival and server failure adaptation test runs without error."""
    res = run_adaptation_experiment(
        cluster="cloud_edge",
        burst_factor=2.0,
        burst_start=50.0,
        burst_duration=30.0,
        failure_count=4,
        failure_start=80.0,
        seed=42,
    )
    assert "EAT" in res
    assert "Greedy" in res
    assert res["EAT"]["num_completed_tasks"] > 0
