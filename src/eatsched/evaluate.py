"""Evaluation harness comparing EAT against baselines and ablations [PAPER Sec. VII, ABSTRACT]."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tabulate

from eatsched.agents import (
    D2SACSAAgent,
    D2SACScheduler,
    DDPGScheduler,
    DPPOScheduler,
    EATAgent,
    EATCAAgent,
    EATDAgent,
    EDFScheduler,
    FIFOScheduler,
    GreedyScheduler,
    PPOScheduler,
    PriorityFIFOScheduler,
    RandomScheduler,
    SACScheduler,
    TD3Scheduler,
)
from eatsched.env.sched_env import CloudEdgeSchedEnv
from eatsched.utils.config import get_project_root


def get_scheduler(
    name: str,
    mode: str,
    cluster: str,
    device: str = "cpu",
) -> Any:
    """Instantiate requested scheduler agent."""
    root = get_project_root()
    queue_dim = 2 if mode == "paper_mode" else 6
    edge_dim = 3 if mode == "paper_mode" else 5

    if name == "EAT":
        agent = EATAgent(
            queue_dim=queue_dim,
            edge_dim=edge_dim,
            d_emb=8,
            top_l=8,
            device=device,
        )
        # Check if pre-trained checkpoint exists
        ckpt = root / "checkpoints" / f"eat_{mode}_{cluster}_smoke.pt"
        if ckpt.exists():
            agent.load(ckpt)
        return agent
    elif name == "Greedy":
        return GreedyScheduler(mode=mode)
    elif name == "FIFO":
        return FIFOScheduler()
    elif name == "Random":
        return RandomScheduler()
    elif name == "EDF":
        return EDFScheduler()
    elif name == "Priority_FIFO":
        return PriorityFIFOScheduler()
    elif name == "SAC":
        return SACScheduler()
    elif name == "PPO":
        return PPOScheduler()
    elif name == "TD3":
        return TD3Scheduler()
    elif name == "DDPG":
        return DDPGScheduler()
    elif name == "DPPO":
        return DPPOScheduler()
    elif name == "D2SAC":
        return D2SACScheduler()
    elif name == "EAT_D":
        return EATDAgent(queue_dim=queue_dim, edge_dim=edge_dim)
    elif name == "EAT_CA":
        return EATCAAgent()
    elif name == "D2SAC_SA":
        return D2SACSAAgent()
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def evaluate_scheduler(
    scheduler_name: str,
    mode: str,
    cluster: str,
    episodes: int = 5,
    base_seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    scheduler = get_scheduler(scheduler_name, mode, cluster, device)
    env = CloudEdgeSchedEnv(cluster_name=cluster, mode=mode)

    ep_metrics: List[Dict[str, Any]] = []

    for ep in range(episodes):
        seed = base_seed + ep
        obs, _ = env.reset(seed=seed)
        done = False

        while not done:
            if scheduler_name == "EAT":
                action = scheduler.select_action(obs, deterministic=True)
            elif scheduler_name in ["Greedy", "EDF", "Priority_FIFO"]:
                action = scheduler.select_action(obs, env=env)
            else:
                action = scheduler.select_action(obs)

            obs, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

        ep_metrics.append(env.metrics.summary())

    # Aggregate over episodes
    avg_resp = float(np.mean([m["avg_response_time_s"] for m in ep_metrics]))
    std_resp = float(np.std([m["avg_response_time_s"] for m in ep_metrics]))
    avg_wait = float(np.mean([m["avg_waiting_time_s"] for m in ep_metrics]))
    std_wait = float(np.std([m["avg_waiting_time_s"] for m in ep_metrics]))
    avg_init = float(np.mean([m["avg_init_delay_s"] for m in ep_metrics]))
    std_init = float(np.std([m["avg_init_delay_s"] for m in ep_metrics]))
    avg_reuse = float(np.mean([m["reuse_rate"] for m in ep_metrics]))
    std_reuse = float(np.std([m["reuse_rate"] for m in ep_metrics]))
    avg_qual = float(np.mean([m["avg_quality"] for m in ep_metrics]))
    std_qual = float(np.std([m["avg_quality"] for m in ep_metrics]))
    avg_util = float(np.mean([m["utility"] for m in ep_metrics]))
    std_util = float(np.std([m["utility"] for m in ep_metrics]))
    avg_energy = float(np.mean([m["total_energy_kj"] for m in ep_metrics]))
    std_energy = float(np.std([m["total_energy_kj"] for m in ep_metrics]))
    avg_miss = float(np.mean([m["deadline_miss_rate"] for m in ep_metrics]))
    std_miss = float(np.std([m["deadline_miss_rate"] for m in ep_metrics]))

    return {
        "scheduler": scheduler_name,
        "avg_response_time": f"{avg_resp:.2f} ± {std_resp:.2f}",
        "avg_waiting_time": f"{avg_wait:.2f} ± {std_wait:.2f}",
        "avg_init_delay": f"{avg_init:.2f} ± {std_init:.2f}",
        "reuse_rate": f"{avg_reuse * 100:.1f} ± {std_reuse * 100:.1f}%",
        "avg_quality": f"{avg_qual:.3f} ± {std_qual:.3f}",
        "utility": f"{avg_util:.3f} ± {std_util:.3f}",
        "total_energy_kj": f"{avg_energy:.1f} ± {std_energy:.1f}",
        "deadline_miss_rate": f"{avg_miss * 100:.1f} ± {std_miss * 100:.1f}%",
        "raw": {
            "avg_response_s": avg_resp,
            "std_response_s": std_resp,
            "avg_wait_s": avg_wait,
            "avg_init_s": avg_init,
            "reuse_rate": avg_reuse,
            "avg_quality": avg_qual,
            "utility": avg_util,
            "total_energy_kj": avg_energy,
            "deadline_miss_rate": avg_miss,
        },
    }


def run_evaluation(
    mode: str = "paper_mode",
    cluster: str = "edge32",
    episodes: int = 5,
    seed: int = 42,
    device: str = "cpu",
) -> None:
    root = get_project_root()
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print("   EAT-Sched Multi-Scheduler Evaluation")
    print(f"   Mode: {mode} | Cluster: {cluster} | Episodes: {episodes} | Seed: {seed}")
    print("=======================================================\n")

    schedulers = [
        "EAT",
        "Greedy",
        "FIFO",
        "Random",
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

    if mode != "paper_mode":
        schedulers.extend(["EDF", "Priority_FIFO"])

    results_table = []
    full_output = {}

    for s_name in schedulers:
        print(f"Evaluating {s_name:<14} ...", end="", flush=True)
        res = evaluate_scheduler(s_name, mode, cluster, episodes, seed, device)
        results_table.append(
            {
                "Scheduler": res["scheduler"],
                "Response (s)": res["avg_response_time"],
                "Wait (s)": res["avg_waiting_time"],
                "Init Delay (s)": res["avg_init_delay"],
                "Reuse Rate": res["reuse_rate"],
                "Quality": res["avg_quality"],
                "Utility U": res["utility"],
            }
        )
        full_output[s_name] = res
        print(" DONE")

    df = pd.DataFrame(results_table)
    print("\n" + tabulate.tabulate(df, headers="keys", tablefmt="github", showindex=False))

    # Save to JSON and CSV
    json_path = results_dir / f"eval_{mode}_{cluster}.json"
    csv_path = results_dir / f"eval_{mode}_{cluster}.csv"
    with open(json_path, "w") as f:
        json.dump(full_output, f, indent=2)
    df.to_csv(csv_path, index=False)
    print("\nSaved evaluation artifacts to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="EAT-Sched Evaluation Harness")
    parser.add_argument(
        "--mode",
        type=str,
        default="paper_mode",
        choices=["paper_mode", "paper", "extended_mode", "extended"],
    )
    parser.add_argument("--cluster", type=str, default="edge32")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    mode = "paper_mode" if args.mode in ["paper", "paper_mode"] else "extended_mode"
    run_evaluation(
        mode=mode,
        cluster=args.cluster,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
