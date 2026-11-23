"""Diffusion Policy Optimization (DPPO) and D2SAC Baselines [PAPER Sec. VII-A]."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from eatsched.agents.ddim_policy import DDIMPolicy
from eatsched.agents.heuristics import BaseScheduler


def flatten_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([obs["edge"].flatten(), obs["queue"].flatten(), obs["mask"].flatten()])


class DPPOScheduler(BaseScheduler):
    """DPPO baseline using diffusion policy with MLP state projection."""

    def __init__(self, state_dim: int = 120, action_dim: int = 3) -> None:
        self.proj = nn.Linear(state_dim, 64)
        self.policy = DDIMPolicy(state_dim=64, action_dim=action_dim, num_steps=4)

    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        env: Optional[Any] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        s = flatten_obs(obs)
        if s.shape[0] != self.proj.in_features:
            self.proj = nn.Linear(s.shape[0], 64)

        with torch.no_grad():
            t_s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            f_s = torch.relu(self.proj(t_s))
            action, _ = self.policy.sample_action(f_s, deterministic=True)
        return action.squeeze(0).numpy().astype(np.float32)


class D2SACScheduler(BaseScheduler):
    """D2SAC baseline: Diffusion actor with standard SAC formulation."""

    def __init__(self, state_dim: int = 120, action_dim: int = 3) -> None:
        self.proj = nn.Linear(state_dim, 64)
        self.policy = DDIMPolicy(state_dim=64, action_dim=action_dim, num_steps=4)

    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        env: Optional[Any] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        s = flatten_obs(obs)
        if s.shape[0] != self.proj.in_features:
            self.proj = nn.Linear(s.shape[0], 64)

        with torch.no_grad():
            t_s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            f_s = torch.relu(self.proj(t_s))
            action, _ = self.policy.sample_action(f_s, deterministic=False)
        return action.squeeze(0).numpy().astype(np.float32)
