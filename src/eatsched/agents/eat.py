"""End-to-end Attention-Guided DDIM Reinforcement Learning Agent (EAT) [PAPER]."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from eatsched.agents.critics import TwinCritic
from eatsched.agents.cross_attention import CrossAttentionEncoder
from eatsched.agents.ddim_policy import DDIMPolicy
from eatsched.agents.heuristics import BaseScheduler


class ReplayBuffer:
    """Experience replay buffer for off-policy diffusion RL."""

    def __init__(self, capacity: int = 1000000) -> None:
        self.capacity = capacity
        self.buffer: collections.deque = collections.deque(maxlen=capacity)

    def push(
        self,
        edge: np.ndarray,
        queue: np.ndarray,
        mask: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_edge: np.ndarray,
        next_queue: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            (
                edge,
                queue,
                mask,
                action,
                reward,
                next_edge,
                next_queue,
                next_mask,
                done,
            )
        )

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        edges, queues, masks, actions, rewards, next_edges, next_queues, next_masks, dones = zip(
            *batch, strict=False
        )

        return (
            torch.tensor(np.array(edges), dtype=torch.float32, device=device),
            torch.tensor(np.array(queues), dtype=torch.float32, device=device),
            torch.tensor(np.array(masks), dtype=torch.float32, device=device),
            torch.tensor(np.array(actions), dtype=torch.float32, device=device),
            torch.tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.array(next_edges), dtype=torch.float32, device=device),
            torch.tensor(np.array(next_queues), dtype=torch.float32, device=device),
            torch.tensor(np.array(next_masks), dtype=torch.float32, device=device),
            torch.tensor(np.array(dones), dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class EATAgent(BaseScheduler):
    """EAT: Attention-Guided Diffusion RL Scheduler [PAPER Sec. V]."""

    def __init__(
        self,
        queue_dim: int = 2,
        edge_dim: int = 3,
        d_emb: int = 8,
        top_l: int = 8,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.95,
        tau: float = 0.005,
        alpha_entropy: float = 0.05,
        batch_size: int = 512,
        replay_capacity: int = 1000000,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.alpha_entropy = alpha_entropy
        self.batch_size = batch_size

        # 1. Cross-Attention State Encoder
        self.encoder = CrossAttentionEncoder(
            queue_dim=queue_dim,
            edge_dim=edge_dim,
            d_emb=d_emb,
            top_l=top_l,
        ).to(self.device)

        state_dim = top_l * d_emb  # 8 * 8 = 64

        # 2. DDIM Diffusion Policy
        self.policy = DDIMPolicy(
            state_dim=state_dim,
            action_dim=3,
            num_steps=4,
            eta_init=0.5,
            eta_max=1.0,
        ).to(self.device)

        # 3. Double Critics
        self.critics = TwinCritic(
            state_dim=state_dim,
            action_dim=3,
            hidden_dim=256,
            tau=tau,
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.policy.parameters()),
            lr=actor_lr,
            weight_decay=1e-4,
        )
        self.critic_optimizer = optim.Adam(
            self.critics.parameters(),
            lr=critic_lr,
            weight_decay=1e-4,
        )

        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        env: Optional[Any] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select action [a_c, a_s, a_k] given raw environment observation."""
        self.encoder.eval()
        self.policy.eval()

        with torch.no_grad():
            edge = torch.tensor(obs["edge"], dtype=torch.float32, device=self.device).unsqueeze(0)
            queue = torch.tensor(obs["queue"], dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = torch.tensor(obs["mask"], dtype=torch.float32, device=self.device).unsqueeze(0)

            f_s = self.encoder(queue, edge, mask)
            action, _ = self.policy.sample_action(f_s, deterministic=deterministic)

        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self) -> Optional[Dict[str, float]]:
        """Perform one training update step using transitions sampled from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.encoder.train()
        self.policy.train()
        self.critics.train()

        edges, queues, masks, actions, rewards, n_edges, n_queues, n_masks, dones = (
            self.replay_buffer.sample(self.batch_size, self.device)
        )

        # 1. Encode states
        f_s = self.encoder(queues, edges, masks)
        with torch.no_grad():
            f_s_next = self.encoder(n_queues, n_edges, n_masks)
            # Sample next actions from target policy
            next_actions, _ = self.policy.sample_action(f_s_next, deterministic=True)
            # Target Q-value with explicit (1 - done) terminal masking [ASM-09, PAPER Eq. 21]
            q_target_min = self.critics.target_min(f_s_next, next_actions)
            y_target = rewards + (1.0 - dones) * self.gamma * q_target_min

        # 2. Update Critic Networks
        q1, q2 = self.critics(f_s, actions)
        critic_loss = nn.functional.mse_loss(q1, y_target) + nn.functional.mse_loss(q2, y_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 3. Update Policy & Encoder via Actor Loss [PAPER Eq. 16-17]
        # Re-encode to allow gradients into encoder
        f_s_actor = self.encoder(queues, edges, masks)
        sampled_actions, _ = self.policy.sample_action(f_s_actor, deterministic=False)
        q1_actor, q2_actor = self.critics(f_s_actor, sampled_actions)
        q_actor_min = torch.min(q1_actor, q2_actor)

        # Actor loss maximizes Q and encourages stochasticity: L(Theta) = -Q - alpha * eta
        eta = self.policy.eta
        actor_loss = -torch.mean(q_actor_min) - self.alpha_entropy * eta

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 4. Soft update critic targets [PAPER Table VIII]
        self.critics.soft_update()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "eta": float(eta.item()),
            "q_mean": float(q_actor_min.mean().item()),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save weights and hyperparameters."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "policy": self.policy.state_dict(),
                "critics": self.critics.state_dict(),
            },
            path,
        )

    def load(self, filepath: Union[str, Path]) -> None:
        """Load weights from checkpoint."""
        path = Path(filepath)
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.critics.load_state_dict(checkpoint["critics"])
