"""Production FastAPI backend for EAT-Sched simulation and replay [SPEC Sec. 9.1]."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from eatsched.env.sched_env import CloudEdgeSchedEnv
from eatsched.evaluate import get_scheduler
from eatsched.utils.config import get_project_root, load_yaml

app = FastAPI(
    title="EAT-Sched API",
    description="Deterministic Discrete-Event Simulation & Replay Engine for Edge-Collaborative AIGC Scheduling",
    version="1.0.0",
)

# Enable CORS for local web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationRequest(BaseModel):
    scheduler: str = "EAT"
    cluster: str = "edge32"
    mode: str = "paper_mode"
    seed: int = 42
    arrival_rate: Optional[float] = None
    w_q: float = 0.5


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Check system health, torch device, and available checkpoints."""
    root = get_project_root()
    checkpoints_dir = root / "checkpoints"
    available_ckpts = (
        [p.name for p in checkpoints_dir.glob("*.pt")] if checkpoints_dir.exists() else []
    )
    return {
        "status": "healthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "checkpoints_available": available_ckpts,
    }


@app.get("/clusters")
def list_clusters() -> List[Dict[str, Any]]:
    """List all supported cluster topologies and their specifications."""
    root = get_project_root()
    clusters_dir = root / "configs" / "clusters"
    cluster_list = []
    for cfg_file in sorted(clusters_dir.glob("*.yaml")):
        cfg = load_yaml(cfg_file)
        cluster_list.append(
            {
                "name": cfg.get("name"),
                "num_servers": cfg.get("num_servers"),
                "arrival_rate": cfg.get("arrival_rate"),
                "init_delay": cfg.get("init_delay"),
            }
        )
    return cluster_list


@app.get("/schedulers")
def list_schedulers() -> List[str]:
    """List all supported scheduling algorithms."""
    return [
        "EAT",
        "Greedy",
        "FIFO",
        "Random",
        "EDF",
        "Priority_FIFO",
        "SAC",
        "PPO",
        "TD3",
        "DDPG",
        "DPPO",
        "D2SAC",
        "EAT_D",
        "EAT_CA",
        "D2SAC_SA",
    ]


@app.post("/simulate")
def run_simulation(req: SimulationRequest) -> Dict[str, Any]:
    """Execute a deterministic simulation episode and return Gantt timeline & metrics."""
    try:
        env = CloudEdgeSchedEnv(
            cluster_name=req.cluster,
            mode=req.mode,
            w_q=req.w_q,
            seed=req.seed,
        )
        if req.arrival_rate is not None:
            env.arrival_rate = req.arrival_rate

        scheduler = get_scheduler(
            name=req.scheduler,
            mode=req.mode,
            cluster=req.cluster,
            device="cpu",
        )

        obs, _ = env.reset(seed=req.seed)
        done = False

        while not done:
            if req.scheduler == "EAT":
                action = scheduler.select_action(obs, deterministic=True)
            elif req.scheduler in ["Greedy", "EDF", "Priority_FIFO"]:
                action = scheduler.select_action(obs, env=env)
            else:
                action = scheduler.select_action(obs)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        metrics = env.metrics.summary()
        timeline = env.timeline_events

        return {
            "scheduler": req.scheduler,
            "cluster": req.cluster,
            "mode": req.mode,
            "seed": req.seed,
            "metrics": metrics,
            "timeline": timeline,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/results/{experiment}")
def get_precomputed_results(experiment: str) -> Dict[str, Any]:
    """Load precomputed benchmark results from results directory."""
    root = get_project_root()
    res_path = root / "results" / f"{experiment}.json"
    if not res_path.exists():
        raise HTTPException(status_code=404, detail=f"Result file {experiment}.json not found")
    with open(res_path, "r") as f:
        data = json.load(f)
    return data


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """Stream live simulation steps to client via WebSocket."""
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        req_json = json.loads(data)
        cluster = req_json.get("cluster", "edge32")
        mode = req_json.get("mode", "paper_mode")
        sched_name = req_json.get("scheduler", "EAT")
        seed = req_json.get("seed", 42)

        env = CloudEdgeSchedEnv(cluster_name=cluster, mode=mode, seed=seed)
        scheduler = get_scheduler(sched_name, mode, cluster, device="cpu")
        obs, _ = env.reset(seed=seed)
        done = False

        while not done:
            if sched_name == "EAT":
                action = scheduler.select_action(obs, deterministic=True)
            elif sched_name == "Greedy":
                action = scheduler.select_action(obs, env=env)
            else:
                action = scheduler.select_action(obs)

            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            step_data = {
                "current_time": env.current_time,
                "reward": rew,
                "queue_len": len(env.task_queue),
                "tasks_completed": env.tasks_completed_count,
                "latest_event": env.timeline_events[-1] if env.timeline_events else None,
                "done": done,
            }
            await websocket.send_text(json.dumps(step_data))
    except WebSocketDisconnect:
        pass
