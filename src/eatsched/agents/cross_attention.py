"""Cross-Attention State Encoder for EAT-Sched [PAPER Sec. IV-B, Eq. 13-14]."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionEncoder(nn.Module):
    """Cross-Attention encoder matching task queue queries to server cluster keys/values.

    Guarantees permutation invariance over server cluster dimension [PAPER Sec. IV-B].
    """

    def __init__(
        self,
        queue_dim: int = 2,
        edge_dim: int = 3,
        d_emb: int = 8,
        top_l: int = 8,
    ) -> None:
        super().__init__()
        self.queue_dim = queue_dim
        self.edge_dim = edge_dim
        self.d_emb = d_emb
        self.top_l = top_l

        # Feature projections phi_Q and phi_E [PAPER Eq. 13]
        self.phi_q = nn.Linear(queue_dim, d_emb)
        self.phi_e = nn.Linear(edge_dim, d_emb)

        # Cross-attention linear projections
        self.w_q = nn.Linear(d_emb, d_emb, bias=False)
        self.w_k = nn.Linear(d_emb, d_emb, bias=False)
        self.w_v = nn.Linear(d_emb, d_emb, bias=False)

        self.scale = 1.0 / math.sqrt(d_emb)

    def forward(
        self,
        queue_state: torch.Tensor,
        edge_state: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of cross-attention state encoder.

        Args:
            queue_state: (B, top_l, queue_dim)
            edge_state: (B, num_servers, edge_dim)
            mask: (B, top_l), 1 where slot is empty, 0 where valid

        Returns:
            f_s: Flattened feature vector of shape (B, top_l * d_emb)
        """
        # 1. Project features
        q_feat = F.relu(self.phi_q(queue_state))  # (B, top_l, d_emb)
        e_feat = F.relu(self.phi_e(edge_state))  # (B, num_servers, d_emb)

        # 2. Linear projections for Q, K, V
        queries = self.w_q(q_feat)  # (B, top_l, d_emb)
        keys = self.w_k(e_feat)  # (B, num_servers, d_emb)
        values = self.w_v(e_feat)  # (B, num_servers, d_emb)

        # 3. Scaled dot-product attention
        # scores: (B, top_l, num_servers)
        scores = torch.bmm(queries, keys.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)

        # 4. Attention weighted reduction over servers
        # out: (B, top_l, d_emb)
        out = torch.bmm(attn_weights, values)

        # 5. Apply task queue padding mask
        # If mask is 1 (empty slot), zero out corresponding query output
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).bool()  # (B, top_l, 1)
            out = out.masked_fill(mask_expanded, 0.0)

        # 6. Flatten to (B, top_l * d_emb) [PAPER Eq. 14]
        b = queue_state.shape[0]
        return out.reshape(b, self.top_l * self.d_emb)
