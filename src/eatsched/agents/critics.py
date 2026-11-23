"""Double Critic networks and target update mechanics [PAPER Sec. V-B2, Eq. 18-23]."""

from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """2-layer MLP Critic with Mish activations [PAPER Table VII]."""

    def __init__(self, state_dim: int, action_dim: int = 3, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class TwinCritic(nn.Module):
    """Twin Q-networks with Polyak target network management."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        tau: float = 0.005,
    ) -> None:
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)

        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)

        # Freeze target networks
        for p in self.target_q1.parameters():
            p.requires_grad = False
        for p in self.target_q2.parameters():
            p.requires_grad = False

        self.tau = tau

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def target_min(self, next_state: torch.Tensor, next_action: torch.Tensor) -> torch.Tensor:
        """Compute min(Q1_target, Q2_target) for double Q-learning."""
        target_q1 = self.target_q1(next_state, next_action)
        target_q2 = self.target_q2(next_state, next_action)
        return torch.min(target_q1, target_q2)

    def soft_update(self) -> None:
        """Polyak soft update: target = tau * net + (1 - tau) * target [PAPER Table VIII]."""
        for target_p, p in zip(self.target_q1.parameters(), self.q1.parameters(), strict=False):
            target_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * target_p.data)
        for target_p, p in zip(self.target_q2.parameters(), self.q2.parameters(), strict=False):
            target_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * target_p.data)
