"""Tests for Cross-Attention state encoder invariants."""

import torch

from eatsched.agents.cross_attention import CrossAttentionEncoder


def test_permutation_invariance_over_servers():
    """Verify that any permutation of servers yields mathematically identical attention output [PAPER Sec. IV-B]."""
    torch.manual_seed(42)
    encoder = CrossAttentionEncoder(queue_dim=2, edge_dim=3, d_emb=8, top_l=4)
    encoder.eval()

    # Create dummy queue and edge tensors
    queue = torch.randn(1, 4, 2)
    edge = torch.randn(1, 8, 3)
    mask = torch.zeros(1, 4)

    with torch.no_grad():
        out1 = encoder(queue, edge, mask)

        # Randomly permute server order
        perm = torch.randperm(8)
        edge_perm = edge[:, perm, :]
        out2 = encoder(queue, edge_perm, mask)

    # Invariance check: maximum absolute difference should be < 1e-5
    diff = torch.max(torch.abs(out1 - out2)).item()
    assert diff < 1e-5, f"Permutation invariance violated! Max diff: {diff}"


def test_empty_queue_padding_mask():
    """Verify that empty queue slots are masked and produce zero feature output."""
    encoder = CrossAttentionEncoder(queue_dim=2, edge_dim=3, d_emb=8, top_l=4)
    encoder.eval()

    queue = torch.randn(1, 4, 2)
    edge = torch.randn(1, 8, 3)
    # Mask last 2 slots
    mask = torch.tensor([[0, 0, 1, 1]])

    with torch.no_grad():
        out = encoder(queue, edge, mask)  # (1, 32)
        reshaped = out.reshape(1, 4, 8)
        # Slots 2 and 3 must be exactly zero
        assert torch.all(reshaped[:, 2:, :] == 0.0)
