"""Tests for FastAPI backend and simulation determinism."""

from fastapi.testclient import TestClient

from api.main import app
from eatsched.env.sched_env import CloudEdgeSchedEnv
from eatsched.evaluate import get_scheduler

client = TestClient(app)


def test_api_health_and_metadata():
    """Verify health and metadata endpoints."""
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    r_clusters = client.get("/clusters")
    assert r_clusters.status_code == 200
    assert len(r_clusters.json()) >= 4

    r_sched = client.get("/schedulers")
    assert r_sched.status_code == 200
    assert "EAT" in r_sched.json()
    assert "Greedy" in r_sched.json()


def test_simulation_determinism():
    """Verify that the same seed gives identical results in API and CLI simulation."""
    req_payload = {
        "scheduler": "FIFO",
        "cluster": "edge32",
        "mode": "paper_mode",
        "seed": 42,
    }

    # API Call 1
    res1 = client.post("/simulate", json=req_payload).json()
    # API Call 2
    res2 = client.post("/simulate", json=req_payload).json()

    assert res1["metrics"] == res2["metrics"]
    assert len(res1["timeline"]) == len(res2["timeline"])

    # CLI / direct env simulation with same seed
    env = CloudEdgeSchedEnv(cluster_name="edge32", mode="paper_mode", seed=42)
    scheduler = get_scheduler("FIFO", "paper_mode", "edge32")
    obs, _ = env.reset(seed=42)
    done = False
    while not done:
        action = scheduler.select_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    cli_metrics = env.metrics.summary()
    assert res1["metrics"] == cli_metrics, "API simulation must match direct CLI simulation!"
