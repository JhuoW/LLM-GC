"""
Topology-Aware Residual Attention (TRA) — flex_attention backend.

Uses ``torch.nn.attention.flex_attention`` to inject graph-structure bias
into attention scores *without* materializing a dense T×T matrix.  The
bias is computed pointwise inside a compiled triton kernel, maintaining
Flash Attention-level O(T) memory.

Per-layer learned parameters:
    dist_embed        : nn.Embedding(d_max+2, n_heads)  — distance → bias
    alpha             : nn.Parameter(n_heads)            — per-head scaling
    same_entity_bias  : nn.Parameter(n_heads)            — same-node bonus

Requires PyTorch ≥ 2.5 with triton ≥ 2.1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torch.nn.attention.flex_attention import flex_attention


# ═════════════════════════════════════════════════════════════════════════════
# Per-batch graph state (set by training loop before each forward)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TRAState:
    """Holds graph tensors for the current micro-batch."""
    entity_map: torch.Tensor | None = None   # (B, T)  long, -1 = non-entity
    graph_dist: torch.Tensor | None = None   # (B, N, N) long, distances


# ═════════════════════════════════════════════════════════════════════════════
# score_mod factory (captured by flex_attention kernel)
# ═════════════════════════════════════════════════════════════════════════════

def create_gsb_score_mod(
    entity_map: torch.Tensor,     # (B, T)
    graph_dist: torch.Tensor,     # (B, N, N)
    dist_embed: torch.Tensor,     # (d_max+2, n_heads)
    same_bias: torch.Tensor,      # (n_heads,)
    alpha: torch.Tensor,          # (n_heads,)
    d_max: int = 1,
):
    """Create a ``score_mod`` closure for ``flex_attention``.

    All tensors are captured by the closure and accessed pointwise —
    no T×T matrix is ever materialized.
    """

    def score_mod(score, b, h, q_idx, kv_idx):
        q_entity = entity_map[b, q_idx]
        k_entity = entity_map[b, kv_idx]

        q_is_entity = q_entity >= 0
        k_is_entity = k_entity >= 0
        both_entities = q_is_entity & k_is_entity

        # Safe indexing (clamp non-entity to 0; masked out later)
        safe_q = torch.where(q_is_entity, q_entity, 0)
        safe_k = torch.where(k_is_entity, k_entity, 0)
        dist = graph_dist[b, safe_q, safe_k]
        clamped_dist = torch.clamp(dist, max=d_max + 1)

        # Learned bias for this distance
        dist_bias = dist_embed[clamped_dist.long(), h]

        # Same-entity override
        is_same = both_entities & (q_entity == k_entity)
        bias = torch.where(is_same, same_bias[h], dist_bias)

        # Zero out non-entity pairs
        bias = torch.where(both_entities, bias, 0.0)

        # Per-head gated scaling: sigmoid(α) * 4 keeps initial scale ≈ 2
        scaled_bias = alpha[h].sigmoid() * 4.0 * bias

        return score + scaled_bias

    return score_mod


# ═════════════════════════════════════════════════════════════════════════════
# Per-layer parameters
# ═════════════════════════════════════════════════════════════════════════════

class TRALayerParams(nn.Module):
    """Learned TRA parameters for one transformer layer."""

    def __init__(self, n_heads: int, d_max: int = 1):
        super().__init__()
        self.n_heads = n_heads
        self.d_max = d_max
        self.dist_embed = nn.Embedding(d_max + 2, n_heads)
        nn.init.zeros_(self.dist_embed.weight)
        self.alpha = nn.Parameter(torch.zeros(n_heads))
        self.same_entity_bias = nn.Parameter(torch.zeros(n_heads))


# ═════════════════════════════════════════════════════════════════════════════
# Top-level TRA module
# ═════════════════════════════════════════════════════════════════════════════

class TopologyAwareAttention(nn.Module):
    """TRA module for all layers.

    Holds per-layer ``TRALayerParams`` and the current-batch ``TRAState``.

    Args:
        num_layers: Number of transformer layers.
        num_heads:  Number of **query** heads (H_Q).  For GQA in LLaMA-3-8B
                    this is 32, not the 8 KV heads.
        d_max:      Maximum hop distance to encode.
    """

    def __init__(self, num_layers: int, num_heads: int, d_max: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_max = d_max

        self.layers = nn.ModuleList(
            [TRALayerParams(num_heads, d_max) for _ in range(num_layers)]
        )
        self.state = TRAState()

    # ── graph data lifecycle ─────────────────────────────────────────────

    def set_graph_data(
        self,
        entity_map: torch.Tensor,
        graph_dist: torch.Tensor,
    ) -> None:
        """Set graph tensors for the current micro-batch.

        Args:
            entity_map: ``(B, T)`` long — node ID per token, -1 for non-entity.
            graph_dist: ``(B, N, N)`` long — pairwise hop distances,
                        ``d_max+1`` for disconnected.
        """
        self.state.entity_map = entity_map
        self.state.graph_dist = graph_dist

    def clear_graph_data(self) -> None:
        self.state.entity_map = None
        self.state.graph_dist = None

    # ── convenience ──────────────────────────────────────────────────────

    @property
    def num_extra_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"layers={self.num_layers}, heads={self.num_heads}, "
            f"d_max={self.d_max}, params={self.num_extra_parameters:,}"
        )
