"""Tests for DDIM Diffusion Policy and learnable stochasticity eta [PAPER Sec. V-B1]."""

import torch

from eatsched.agents.ddim_policy import DDIMPolicy


def test_ddim_sampling_and_range():
    """Verify reverse DDIM sampling outputs bounded continuous actions in [-1, 1]^3."""
    torch.manual_seed(42)
    policy = DDIMPolicy(state_dim=64, action_dim=3, num_steps=4)

    f_s = torch.randn(2, 64)
    action, x0 = policy.sample_action(f_s, deterministic=False)

    assert action.shape == (2, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_learnable_eta_gradient_flow():
    """Verify gradients propagate back through the entire DDIM sampling chain into eta."""
    torch.manual_seed(42)
    policy = DDIMPolicy(state_dim=64, action_dim=3, num_steps=4)

    f_s = torch.randn(2, 64)
    action, _ = policy.sample_action(f_s, deterministic=False)

    # Compute a dummy loss dependent on action
    loss = torch.sum(action**2)
    loss.backward()

    assert policy.w_eta.grad is not None
    assert torch.abs(policy.w_eta.grad) > 0.0, "Gradient did not flow to learnable eta!"
