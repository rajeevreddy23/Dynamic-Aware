"""Soft Actor-Critic (SAC) Baseline [PAPER Sec. VII-A]."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from eatsched.agents.heuristics import BaseScheduler


def flatten_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([obs["edge"].flatten(), obs["queue"].flatten(), obs["mask"].flatten()])


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 3, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        feat = self.net(state)
        mean = self.mean_layer(feat)
        log_std = torch.clamp(self.log_std_layer(feat), -20, 2)
        std = torch.exp(log_std)
        normal = torch.randn_like(mean)
        action = torch.tanh(mean + std * normal)
        return action


class SACScheduler(BaseScheduler):
    """SAC scheduler baseline without diffusion or cross-attention."""

    def __init__(self, state_dim: int = 120, action_dim: int = 3, lr: float = 3e-4) -> None:
        self.actor = GaussianActor(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)

    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        env: Optional[Any] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        s = flatten_obs(obs)
        if s.shape[0] != self.actor.net[0].in_features:  # type: ignore[index]
            # Adaptively resize if state dim changes between clusters
            self.actor = GaussianActor(s.shape[0], 3)
            self.optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        with torch.no_grad():
            t_s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            action = self.actor(t_s).squeeze(0).numpy()
        return action.astype(np.float32)
