"""Ablation variants of EAT [PAPER Sec. VII-B7]."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eatsched.agents.cross_attention import CrossAttentionEncoder
from eatsched.agents.ddim_policy import DDIMPolicy
from eatsched.agents.heuristics import BaseScheduler


class EATDAgent(BaseScheduler):
    """EAT-D Ablation: EAT with DDIM removed.

    Cross-Attention encoder feeds directly into an MLP policy without diffusion sampling.
    """

    def __init__(
        self,
        queue_dim: int = 2,
        edge_dim: int = 3,
        d_emb: int = 8,
        top_l: int = 8,
        hidden_dim: int = 256,
    ) -> None:
        self.encoder = CrossAttentionEncoder(queue_dim, edge_dim, d_emb, top_l)
        state_dim = top_l * d_emb
        last_linear = nn.Linear(hidden_dim, 3)
        with torch.no_grad():
            last_linear.bias[0].fill_(0.5)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            last_linear,
            nn.Tanh(),
        )

    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        env: Optional[Any] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        with torch.no_grad():
            q = torch.tensor(obs["queue"], dtype=torch.float32).unsqueeze(0)
            e = torch.tensor(obs["edge"], dtype=torch.float32).unsqueeze(0)
            m = torch.tensor(obs["mask"], dtype=torch.float32).unsqueeze(0)
            f_s = self.encoder(q, e, m)
            action = self.actor(f_s).squeeze(0).numpy()
        return action.astype(np.float32)


class EATCAAgent(BaseScheduler):
    """EAT-CA Ablation: EAT with Cross-Attention removed.

    Replaces Cross-Attention with a standard concatenated MLP projection into DDIM policy.
    """

    def __init__(self, state_dim: int = 120, action_dim: int = 3) -> None:
        self.proj = nn.Linear(state_dim, 64)
        self.policy = DDIMPolicy(state_dim=64, action_dim=action_dim, num_steps=4)

    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        env: Optional[Any] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        s = np.concatenate([obs["edge"].flatten(), obs["queue"].flatten(), obs["mask"].flatten()])
        if s.shape[0] != self.proj.in_features:
            self.proj = nn.Linear(s.shape[0], 64)

        with torch.no_grad():
            t_s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            f_s = F.mish(self.proj(t_s))
            action, _ = self.policy.sample_action(f_s, deterministic=True)
        return action.squeeze(0).numpy().astype(np.float32)


class D2SACSAAgent(BaseScheduler):
    """D2SAC+SA Ablation: D2SAC augmented with Self-Attention."""

    def __init__(self, state_dim: int = 120, action_dim: int = 3) -> None:
        self.proj = nn.Linear(state_dim, 64)
        self.sa = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
        self.policy = DDIMPolicy(state_dim=64, action_dim=action_dim, num_steps=4)

    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        env: Optional[Any] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        s = np.concatenate([obs["edge"].flatten(), obs["queue"].flatten(), obs["mask"].flatten()])
        if s.shape[0] != self.proj.in_features:
            self.proj = nn.Linear(s.shape[0], 64)

        with torch.no_grad():
            t_s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            h = F.mish(self.proj(t_s)).unsqueeze(1)
            sa_out, _ = self.sa(h, h, h)
            f_s = sa_out.squeeze(1)
            action, _ = self.policy.sample_action(f_s, deterministic=True)
        return action.squeeze(0).numpy().astype(np.float32)
