"""
Patch a HuggingFace LLaMA model to use Topology-Aware Residual Attention.

Uses **flash_attn** (Dao-AILab) for the base attention and injects TRA
graph-structure bias via a manual scaled-dot-product computation on layers
that have entity tokens.

  * Layers/batches WITHOUT entity tokens → original Flash Attention forward
    (maximum speed, O(T) memory).
  * Layers/batches WITH entity tokens → manual attention with dense bias
    (O(T²) memory, but only ~7% of tokens are entities so the bias is sparse).

Two entry points:
  * ``patch_model_with_tra``  — patches each attention layer.
  * ``prepare_tra_inputs``    — batch-level preprocessing (CPU).
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from functools import wraps

from .graph_parser import GraphParser
from .attention_bias import TopologyAwareAttention, create_gsb_score_mod


# ═════════════════════════════════════════════════════════════════════════════
# Model patching
# ═════════════════════════════════════════════════════════════════════════════

def patch_model_with_tra(
    model,
    tra: TopologyAwareAttention,
) -> TopologyAwareAttention:
    """Patch a LLaMA model to inject TRA graph-structure bias.

    The model is loaded with ``attn_implementation="flash_attention_2"``
    for maximum speed.  When graph data is present AND has entity tokens,
    the patched forward computes attention manually (QK^T + bias + softmax)
    instead of calling flash_attn.  When no entities are present, the
    original Flash Attention forward runs untouched.
    """
    model.tra = tra

    # Navigate through PEFT wrapper(s) to the transformer layers
    base = model
    while hasattr(base, "base_model") and base.base_model is not base:
        base = base.base_model
    while hasattr(base, "model") and not hasattr(base, "layers"):
        base = base.model
    layers = base.layers

    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        _orig_forward = attn.forward

        def _make_forward(orig_fwd, lidx, tra_mod, attn_module):
            tra_layer = tra_mod.layers[lidx]

            @wraps(orig_fwd)
            def _forward(hidden_states, attention_mask=None,
                         position_ids=None, **kwargs):
                state = tra_mod.state

                # ── Fast path: no graph data → original Flash forward ────
                if state.entity_map is None:
                    return orig_fwd(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **kwargs,
                    )

                device = hidden_states.device
                emap = state.entity_map.to(device)

                # ── Fast path: no entity tokens → original Flash forward ─
                if not (emap >= 0).any():
                    return orig_fwd(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **kwargs,
                    )

                # ── TRA path: manual attention with graph bias ───────────
                if tra_layer.dist_embed.weight.device != device:
                    tra_mod.to(device)

                bsz, q_len, _ = hidden_states.size()

                # Q / K / V projections (LoRA-wrapped if PEFT)
                n_heads = attn_module.config.num_attention_heads
                n_kv_heads = attn_module.config.num_key_value_heads
                head_dim = attn_module.head_dim

                query_states = attn_module.q_proj(hidden_states).view(
                    bsz, q_len, n_heads, head_dim
                ).transpose(1, 2)
                key_states = attn_module.k_proj(hidden_states).view(
                    bsz, q_len, n_kv_heads, head_dim
                ).transpose(1, 2)
                value_states = attn_module.v_proj(hidden_states).view(
                    bsz, q_len, n_kv_heads, head_dim
                ).transpose(1, 2)

                # RoPE
                position_embeddings = kwargs.get("position_embeddings", None)
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                else:
                    cos, sin = attn_module.rotary_emb(value_states, position_ids)
                from transformers.models.llama.modeling_llama import (
                    apply_rotary_pos_emb,
                )
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

                # GQA: expand KV heads to match query heads
                n_rep = n_heads // n_kv_heads
                if n_rep > 1:
                    key_states = key_states.repeat_interleave(n_rep, dim=1)
                    value_states = value_states.repeat_interleave(
                        n_rep, dim=1
                    )

                # Attention scores + TRA bias
                # (B, H, T, T) = (B, H, T, d) @ (B, H, d, T)
                scale = 1.0 / math.sqrt(head_dim)
                attn_weights = torch.matmul(
                    query_states, key_states.transpose(2, 3)
                ) * scale

                # Causal mask
                causal_mask = torch.triu(
                    torch.full((q_len, q_len), float("-inf"), device=device),
                    diagonal=1,
                )
                attn_weights = attn_weights + causal_mask

                # Padding mask (from attention_mask)
                if attention_mask is not None:
                    if attention_mask.dim() == 2:
                        # Flash-style (B, T) bool → expand to (B, 1, 1, T)
                        pad_mask = attention_mask[:, None, None, :].bool()
                        attn_weights = attn_weights.masked_fill(
                            ~pad_mask, float("-inf")
                        )
                    elif attention_mask.dim() == 4:
                        # SDPA-style (B, 1, T, T) bool or float
                        if attention_mask.dtype == torch.bool:
                            attn_weights = attn_weights.masked_fill(
                                ~attention_mask, float("-inf")
                            )
                        else:
                            attn_weights = attn_weights + attention_mask

                # TRA graph bias
                bias = _compute_dense_bias(
                    emap,
                    state.graph_dist.to(device),
                    tra_layer,
                    tra_mod.d_max,
                    q_len,
                )
                attn_weights = attn_weights + bias

                # Softmax + value projection
                attn_weights = F.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)

                # Output projection
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, -1)
                attn_output = attn_module.o_proj(attn_output)

                return attn_output, None

            return _forward

        attn.forward = _make_forward(_orig_forward, layer_idx, tra, attn)

    return tra


# ═════════════════════════════════════════════════════════════════════════════
# Dense bias computation
# ═════════════════════════════════════════════════════════════════════════════

def _compute_dense_bias(
    entity_map: torch.Tensor,    # (B, T)
    graph_dist: torch.Tensor,    # (B, N, N)
    tra_layer,                   # TRALayerParams
    d_max: int,
    seq_len: int,
) -> torch.Tensor:
    """Build ``(B, H, T, T)`` additive bias from sparse graph data."""
    B = entity_map.shape[0]
    device = entity_map.device
    n_heads = tra_layer.n_heads

    q_entity = entity_map.unsqueeze(2)   # (B, T, 1)
    k_entity = entity_map.unsqueeze(1)   # (B, 1, T)
    q_valid = q_entity >= 0
    k_valid = k_entity >= 0
    both_valid = q_valid & k_valid       # (B, T, T)

    safe_q = q_entity.clamp(min=0)
    safe_k = k_entity.clamp(min=0)
    safe_q_exp = safe_q.expand(B, seq_len, seq_len)
    safe_k_exp = safe_k.expand(B, seq_len, seq_len)

    N = graph_dist.shape[1]
    flat_idx = (safe_q_exp * N + safe_k_exp).reshape(B, -1)
    flat_idx = flat_idx.clamp(max=N * N - 1)
    flat_dist = graph_dist.reshape(B, -1)
    dists = torch.gather(flat_dist, 1, flat_idx).reshape(B, seq_len, seq_len)
    dists = dists.clamp(max=d_max + 1)

    is_same = both_valid & (q_entity == k_entity)

    dist_bias = F.embedding(dists, tra_layer.dist_embed.weight)  # (B, T, T, H)
    same_bias = tra_layer.same_entity_bias.view(1, 1, 1, n_heads)
    bias = torch.where(is_same.unsqueeze(-1), same_bias.expand_as(dist_bias), dist_bias)
    bias = bias * both_valid.unsqueeze(-1).float()
    alpha = tra_layer.alpha.sigmoid() * 4.0
    bias = bias * alpha.view(1, 1, 1, n_heads)

    return bias.permute(0, 3, 1, 2)  # (B, H, T, T)


# ═════════════════════════════════════════════════════════════════════════════
# Batch preprocessing (CPU)
# ═════════════════════════════════════════════════════════════════════════════

def prepare_tra_inputs(
    texts: list[str],
    tasks: list[str],
    tokenizer,
    parser: GraphParser | None = None,
    d_max: int = 1,
    max_length: int = 2048,
) -> dict[str, torch.Tensor]:
    """Tokenize a batch and build TRA graph tensors."""
    if parser is None:
        parser = GraphParser()

    graphs = [parser.parse(t, task) for t, task in zip(texts, tasks)]
    sp_list = [parser.compute_shortest_paths(g, d_max) for g in graphs]

    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt", return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    B, T = input_ids.shape

    entity_maps = torch.full((B, T), -1, dtype=torch.long)
    max_nodes = 0
    for i in range(B):
        offsets = enc["offset_mapping"][i].tolist()
        emap = parser.build_entity_map_from_offsets(texts[i], offsets, graphs[i])
        for pos, nid in emap.items():
            if pos < T:
                entity_maps[i, pos] = nid
        if graphs[i]["nodes"]:
            max_nodes = max(max_nodes, max(graphs[i]["nodes"]) + 1)

    max_nodes = max(max_nodes, 1)
    graph_dists = torch.full((B, max_nodes, max_nodes), d_max + 1, dtype=torch.long)
    for i in range(B):
        for (u, v), d in sp_list[i].items():
            if u < max_nodes and v < max_nodes:
                graph_dists[i, u, v] = d

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "entity_maps": entity_maps,
        "graph_dists": graph_dists,
    }
