"""DDIM Diffusion Policy Network with Learnable Stochasticity [PAPER Sec. V-B1, Eq. 10-12]."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion denoising timesteps."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Args: timesteps: (B,) tensor of integers in [1, T]."""
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1) if half_dim > 1 else 1.0
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb_scale
        )
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class DenoiserMLP(nn.Module):
    """Trunk MLP for diffusion noise prediction epsilon_theta(x_t, t_diff, f_s)."""

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 3,
        time_emb_dim: int = 8,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(dim=time_emb_dim)
        in_dim = state_dim + action_dim + time_emb_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Bounds predicted noise to [-1, 1] [PAPER Table VII]
        )

    def forward(self, x_t: torch.Tensor, t_diff: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
        """Args:

        x_t: (B, action_dim)
        t_diff: (B,)
        f_s: (B, state_dim)
        """
        t_emb = self.time_embed(t_diff)
        inp = torch.cat([f_s, x_t, t_emb], dim=-1)
        return self.net(inp)


class DDIMPolicy(nn.Module):
    """DDIM Reverse Diffusion Sampling Policy with Learnable Stochasticity eta [PAPER Sec. V-B1]."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_bar: torch.Tensor

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 3,
        num_steps: int = 4,
        eta_init: float = 0.5,
        eta_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_steps = num_steps
        self.eta_max = eta_max

        self.denoiser = DenoiserMLP(
            state_dim=state_dim,
            action_dim=action_dim,
            time_emb_dim=8,
            hidden_dim=256,
        )

        # Learnable eta parameter [PAPER Sec. V-B1, ASM-08]
        # Parameterized as w_eta such that eta = eta_max * sigmoid(w_eta)
        # For eta_init = 0.5 and eta_max = 1.0, w_eta starts at 0.0
        init_w = math.log(eta_init / (eta_max - eta_init + 1e-7))
        self.w_eta = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))

        # Linear noise schedule beta_1 ... beta_T [PAPER Table VIII]
        betas = torch.linspace(1e-4, 0.02, steps=num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)

    @property
    def eta(self) -> torch.Tensor:
        """Bounded learnable stochasticity parameter eta in [0, eta_max]."""
        return self.eta_max * torch.sigmoid(self.w_eta)

    def sample_action(
        self,
        f_s: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate action candidate via reverse DDIM sampling chain.

        Returns:
            (action, clean_x0)
        """
        batch_size = f_s.shape[0]
        device = f_s.device

        # Start from standard Gaussian noise x_T [PAPER Eq. 10]
        x_t = torch.randn(batch_size, self.action_dim, device=device)

        curr_eta = torch.tensor(0.0, device=device) if deterministic else self.eta

        # Reverse diffusion steps: t = T, ..., 1
        for step in reversed(range(1, self.num_steps + 1)):
            t_tensor = torch.full((batch_size,), step, device=device, dtype=torch.long)
            eps = self.denoiser(x_t, t_tensor, f_s)

            alpha_bar_t = self.alphas_bar[step - 1]
            alpha_bar_prev = (
                self.alphas_bar[step - 2] if step > 1 else torch.tensor(1.0, device=device)
            )

            # 1. Predicted clean action [PAPER Eq. 11]
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
            x_0_hat = (x_t - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t

            if step == 1:
                # Terminal step produces clean action directly
                x_t = x_0_hat
                break

            # 2. Stochastic noise variance sigma_t [PAPER Eq. 12]
            var_term = (
                (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_prev)
            )
            sigma_t = curr_eta * torch.sqrt(torch.clamp(var_term, min=0.0))

            # Direction pointing to x_t with numerical safety clamp [PAPER Sec. V-B1]
            dir_term = torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0)
            dir_xt = torch.sqrt(dir_term) * eps

            # Random noise perturbation
            noise = torch.randn_like(x_t) if not deterministic else torch.zeros_like(x_t)

            # Sample x_{t-1}
            x_t = torch.sqrt(alpha_bar_prev) * x_0_hat + dir_xt + sigma_t * noise

        action = torch.clamp(x_t, -1.0, 1.0)
        return action, x_0_hat
