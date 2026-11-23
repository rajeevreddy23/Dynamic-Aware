"""Deep Deterministic Policy Gradient (DDPG) Baseline [PAPER Sec. VII-A]."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from eatsched.agents.heuristics import BaseScheduler


def flatten_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([obs["edge"].flatten(), obs["queue"].flatten(), obs["mask"].flatten()])


class DDPGActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 3, hidden_dim: int = 256) -> None:
        super().__init__()
        last_linear = nn.Linear(hidden_dim, action_dim)
        with torch.no_grad():
            last_linear.bias[0].fill_(0.5)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            last_linear,
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DDPGScheduler(BaseScheduler):
    """DDPG scheduler baseline."""

    def __init__(self, state_dim: int = 120, action_dim: int = 3) -> None:
        self.actor = DDPGActor(state_dim, action_dim)

    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        env: Optional[Any] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        s = flatten_obs(obs)
        if s.shape[0] != self.actor.net[0].in_features:  # type: ignore[index]
            self.actor = DDPGActor(s.shape[0], 3)

        with torch.no_grad():
            t_s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            action = self.actor(t_s).squeeze(0).numpy()
        return action.astype(np.float32)
