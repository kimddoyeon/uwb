"""Edge-aware GAT and the edge-only MLP ablation baseline.

Copied verbatim from the training project (``gat/model.py``) so the Raspberry Pi
deployment does not need the training tree.  ``uwb_probability_pipeline.py``
picks the class that matches the checkpoint's ``model_state_dict`` keys.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareGATLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        num_heads: int = 2,
        dropout: float = 0.1,
        leaky_slope: float = 0.2,
    ):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, out_dim, bias=False)

        self.attn = nn.Parameter(
            torch.empty(num_heads, 3 * self.head_dim)
        )
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))

        self.leaky_slope = leaky_slope
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_feat: torch.Tensor,
        adj_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = h.shape

        Wh = self.W(h)
        We = self.W_edge(edge_feat)

        Wh_heads = Wh.view(
            B, N, self.num_heads, self.head_dim
        )
        We_heads = We.view(
            B, N, N, self.num_heads, self.head_dim
        )

        Wh_i = Wh_heads.unsqueeze(2).expand(
            B, N, N, self.num_heads, self.head_dim
        )
        Wh_j = Wh_heads.unsqueeze(1).expand(
            B, N, N, self.num_heads, self.head_dim
        )

        cat = torch.cat([Wh_i, Wh_j, We_heads], dim=-1)

        e = torch.einsum(
            "bnmhd,hd->bnmh",
            cat,
            self.attn,
        )

        e = F.leaky_relu(
            e,
            negative_slope=self.leaky_slope,
        )

        e = e.masked_fill(
            ~adj_mask.unsqueeze(-1),
            float("-inf"),
        )

        alpha = F.softmax(e, dim=2)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        out = torch.einsum(
            "bnmh,bnmhd->bnhd",
            alpha,
            Wh_j,
        )

        out = out.reshape(B, N, self.out_dim)
        out = out.masked_fill(
            ~node_mask.unsqueeze(-1),
            0.0,
        )

        return out


class LightweightGAT(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 3,
        dropout: float = 0.1,
        leaky_slope: float = 0.2,
    ):
        super().__init__()

        self.input_proj = nn.Linear(
            node_feat_dim,
            hidden_dim,
        )

        self.layers = nn.ModuleList(
            [
                EdgeAwareGATLayer(
                    hidden_dim,
                    hidden_dim,
                    edge_feat_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    leaky_slope=leaky_slope,
                )
                for _ in range(num_layers)
            ]
        )

        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(hidden_dim)
                for _ in range(num_layers)
            ]
        )

        self.edge_scorer = nn.Sequential(
            nn.Linear(
                hidden_dim * 2 + edge_feat_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        adj_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(node_feat)

        for layer, norm in zip(
            self.layers,
            self.norms,
        ):
            h_new = layer(
                h,
                edge_feat,
                adj_mask,
                node_mask,
            )

            h = norm(h + h_new)
            h = F.elu(h)

        B, N, H = h.shape

        h_i = h.unsqueeze(2).expand(
            B, N, N, H
        )
        h_j = h.unsqueeze(1).expand(
            B, N, N, H
        )

        edge_input = torch.cat(
            [h_i, h_j, edge_feat],
            dim=-1,
        )

        logits = self.edge_scorer(
            edge_input
        ).squeeze(-1)

        logits = logits.masked_fill(
            ~adj_mask,
            float("-inf"),
        )

        return logits

    def count_parameters(self) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

class EdgeOnlyMLP(nn.Module):
    """그래프 구조(attention) 없이 edge feature만으로 판별하는 baseline."""

    def __init__(self, edge_feat_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feat, edge_feat, adj_mask, node_mask):
        logits = self.net(edge_feat).squeeze(-1)
        logits = logits.masked_fill(~adj_mask, float("-inf"))
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)