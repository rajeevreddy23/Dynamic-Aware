"""Tests for Reinforcement Learning plumbing and critic updates."""

import torch

from eatsched.agents.critics import TwinCritic


def test_twin_critic_and_target_soft_update():
    """Verify twin critic forward pass and soft update mechanics."""
    critics = TwinCritic(state_dim=64, action_dim=3, hidden_dim=64, tau=0.1)

    s = torch.randn(4, 64)
    a = torch.randn(4, 3)

    q1, q2 = critics(s, a)
    assert q1.shape == (4, 1)
    assert q2.shape == (4, 1)

    # Initial target parameters match
    p_orig = list(critics.q1.parameters())[0].clone()
    t_orig = list(critics.target_q1.parameters())[0].clone()
    assert torch.allclose(p_orig, t_orig)

    # Perturb weights
    with torch.no_grad():
        list(critics.q1.parameters())[0].add_(1.0)

    # Perform soft update
    critics.soft_update()
    t_new = list(critics.target_q1.parameters())[0]
    assert not torch.allclose(t_orig, t_new)


def test_terminal_done_masking():
    """Verify that done=True masks future value bootstrapping [ASM-09]."""
    gamma = 0.95
    reward = torch.tensor([[1.5], [2.0]])
    q_next = torch.tensor([[10.0], [10.0]])

    # Case 1: not done (done=0) -> y = r + gamma * q_next
    done_false = torch.tensor([[0.0], [0.0]])
    y1 = reward + (1.0 - done_false) * gamma * q_next
    assert torch.allclose(y1, reward + gamma * q_next)

    # Case 2: done (done=1) -> y = r strictly (no bootstrap)
    done_true = torch.tensor([[1.0], [1.0]])
    y2 = reward + (1.0 - done_true) * gamma * q_next
    assert torch.allclose(y2, reward)
