"""Training script for EAT-Sched."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from eatsched.agents.eat import EATAgent
from eatsched.env.sched_env import CloudEdgeSchedEnv
from eatsched.utils.config import get_project_root

BUDGETS = {
    "smoke": 20000,
    "medium": 200000,
    "full": 1000000,
}


def train(
    mode: str = "paper_mode",
    budget: str = "smoke",
    cluster: str = "edge32",
    seed: int = 42,
    device: str = "cpu",
) -> None:
    root = get_project_root()
    results_dir = root / "results"
    checkpoints_dir = root / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    max_steps = BUDGETS.get(budget, 20000)
    print("=== EAT-Sched Training ===", flush=True)
    print(
        f"Mode: {mode} | Cluster: {cluster} | Budget: {budget} ({max_steps} steps) | Seed: {seed} | Device: {device}",
        flush=True,
    )

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = CloudEdgeSchedEnv(
        cluster_name=cluster,
        mode=mode,
        seed=seed,
    )

    queue_dim = 2 if mode == "paper_mode" else 6
    edge_dim = 3 if mode == "paper_mode" else 5

    agent = EATAgent(
        queue_dim=queue_dim,
        edge_dim=edge_dim,
        d_emb=8,
        top_l=8,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.95,
        tau=0.005,
        batch_size=256 if budget == "smoke" else 512,
        device=device,
    )

    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_idx = 0
    ep_rewards: list[float] = []
    log_records: list[dict] = []

    start_time = time.time()

    for global_step in range(1, max_steps + 1):
        # Action selection with exploration
        action = agent.select_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(
            edge=obs["edge"],
            queue=obs["queue"],
            mask=obs["mask"],
            action=action,
            reward=reward,
            next_edge=next_obs["edge"],
            next_queue=next_obs["queue"],
            next_mask=next_obs["mask"],
            done=done,
        )

        obs = next_obs
        episode_reward += reward

        # Training update
        metrics = None
        if global_step % 4 == 0:
            metrics = agent.update()

        if done:
            episode_idx += 1
            ep_rewards.append(episode_reward)
            avg_rew_10 = float(np.mean(ep_rewards[-10:]))
            eta_val = float(agent.policy.eta.item())

            log_records.append(
                {
                    "step": global_step,
                    "episode": episode_idx,
                    "reward": round(episode_reward, 3),
                    "avg_reward_10": round(avg_rew_10, 3),
                    "eta": round(eta_val, 4),
                    "critic_loss": metrics["critic_loss"] if metrics else 0.0,
                    "actor_loss": metrics["actor_loss"] if metrics else 0.0,
                }
            )

            if episode_idx % 5 == 0 or global_step >= max_steps:
                elapsed = time.time() - start_time
                print(
                    f"Step {global_step:>6}/{max_steps} | Ep {episode_idx:>3} | "
                    f"Reward: {episode_reward:>7.2f} | Avg10: {avg_rew_10:>7.2f} | "
                    f"eta: {eta_val:.3f} | Elapsed: {elapsed:.1f}s",
                    flush=True,
                )

            episode_reward = 0.0
            obs, _ = env.reset(seed=seed + episode_idx)

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s across {episode_idx} episodes.", flush=True)

    # Save checkpoint
    checkpoint_file = checkpoints_dir / f"eat_{mode}_{cluster}_{budget}.pt"
    agent.save(checkpoint_file)
    print(f"Checkpoint saved to: {checkpoint_file}", flush=True)

    # Save curves
    df = pd.DataFrame(log_records)
    csv_file = results_dir / f"train_eat_{mode}_{cluster}_{budget}.csv"
    json_file = results_dir / f"train_eat_{mode}_{cluster}_{budget}.json"
    df.to_csv(csv_file, index=False)
    with open(json_file, "w") as f:
        json.dump(
            {
                "config": {"mode": mode, "cluster": cluster, "budget": budget, "seed": seed},
                "total_time_s": total_time,
                "records": log_records,
            },
            f,
            indent=2,
        )
    print(f"Training curves saved to: {csv_file} and {json_file}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="EAT-Sched Training")
    parser.add_argument(
        "--mode",
        type=str,
        default="paper_mode",
        choices=["paper_mode", "paper", "extended_mode", "extended"],
    )
    parser.add_argument("--budget", type=str, default="smoke", choices=["smoke", "medium", "full"])
    parser.add_argument("--cluster", type=str, default="edge32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    mode = "paper_mode" if args.mode in ["paper", "paper_mode"] else "extended_mode"
    train(
        mode=mode,
        budget=args.budget,
        cluster=args.cluster,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
