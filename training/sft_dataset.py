"""
SFT dataset, grouped batch sampler, and data collator for GraphInstruct-Aug.

Key design:
  * Each augmentation group (1 anchor + k permuted) is kept together in a
    micro-batch so that L_repr can compare [GRAPH_REPR] hidden states.
  * The collator tokenizes prompt+response, masks prompt tokens in labels,
    builds TRA distance matrices, and locates [GRAPH_REPR] positions.
"""

from __future__ import annotations

import json
import torch
from torch.utils.data import Dataset, Sampler
from model.tra.graph_parser import GraphParser

# ─── prompt template (Alpaca-style, matching GraphWiz) ───────────────────────

ALPACA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request step by step.\n\n"
    "### Instruction:\n{query}\n\n### Response:"
)


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

class GraphInstructSFTDataset(Dataset):
    """Loads GraphInstruct-Aug and organises samples into augmentation groups.

    Filters to keep ``"none"`` (anchor) and ``"perm_*"`` (permuted) entries,
    dropping the ``"original"`` duplicates.  After filtering, each
    ``original_index`` has exactly ``k + 1`` samples.
    """

    def __init__(self, data_path: str, k: int = 4):
        raw: list[dict] = []
        with open(data_path) as f:
            for line in f:
                raw.append(json.loads(line))

        # Keep anchor ("none") + permuted copies only
        self.samples = [
            s for s in raw
            if s["augmentation"] == "none" or s["augmentation"].startswith("perm_")
        ]
        # Sort so groups are contiguous: (original_index, augmentation)
        self.samples.sort(
            key=lambda s: (s["original_index"], s["augmentation"])
        )

        self.k = k
        self.group_size = k + 1

        n_groups = len(self.samples) // self.group_size
        if len(self.samples) != n_groups * self.group_size:
            # Trim incomplete tail group (should not happen with clean data)
            self.samples = self.samples[: n_groups * self.group_size]

        self.num_groups = n_groups

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "prompt": ALPACA_PROMPT.format(query=s["query"]),
            "response": s["answer"],
            "query": s["query"],  # raw query for TRA graph parsing
            "task": s["task"],
            "original_index": s["original_index"],
            "augmentation": s["augmentation"],
        }


# ═════════════════════════════════════════════════════════════════════════════
# Grouped batch sampler
# ═════════════════════════════════════════════════════════════════════════════

class GroupedBatchSampler(Sampler[list[int]]):
    """Yields index-lists where each list contains complete augmentation groups.

    With ``groups_per_batch = 1`` and ``group_size = 5``, every micro-batch
    has exactly 5 samples that share the same ``original_index``.
    """

    def __init__(
        self,
        dataset: GraphInstructSFTDataset,
        groups_per_batch: int = 1,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.group_size = dataset.group_size
        self.num_groups = dataset.num_groups
        self.groups_per_batch = groups_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        # Each group occupies indices [g*gs .. (g+1)*gs)
        gs = self.group_size
        group_ids = list(range(self.num_groups))

        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(group_ids), generator=g).tolist()
            group_ids = [group_ids[i] for i in perm]

        batch: list[int] = []
        for gid in group_ids:
            batch.extend(range(gid * gs, (gid + 1) * gs))
            if len(batch) >= self.groups_per_batch * gs:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        return (self.num_groups + self.groups_per_batch - 1) // self.groups_per_batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


# ═════════════════════════════════════════════════════════════════════════════
# Data collator
# ═════════════════════════════════════════════════════════════════════════════

class SFTCollator:
    """Tokenizes a micro-batch and prepares everything the training loop needs.

    Returns a dict with:
        input_ids, attention_mask, labels  — standard causal LM
        entity_maps                        — (B, T) node ID per token (-1 = non-entity)
        graph_dists                        — (B, N, N) hop distances for flex_attention
        graph_repr_positions               — (B,) token position of [GRAPH_REPR]
        group_indices                      — list[list[int]] for L_repr
    """

    def __init__(
        self,
        tokenizer,
        parser: GraphParser | None = None,
        d_max: int = 1,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.parser = parser or GraphParser()
        self.d_max = d_max
        self.max_length = max_length

        self.graph_repr_id: int = tokenizer.convert_tokens_to_ids("[GRAPH_REPR]")
        assert self.graph_repr_id != tokenizer.unk_token_id, \
            "[GRAPH_REPR] not in tokenizer — call tokenizer.add_special_tokens first"

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor | list]:
        prompts = [b["prompt"] for b in batch]
        responses = [b["response"] for b in batch]
        tasks = [b["task"] for b in batch]
        eos = self.tokenizer.eos_token or ""
        full_texts = [p + r + eos for p, r in zip(prompts, responses)]

        # ── Tokenize ─────────────────────────────────────────────────────
        enc = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"]          # (B, T)
        attention_mask = enc["attention_mask"]  # (B, T)
        B, T = input_ids.shape

        # ── Labels: mask prompt tokens with -100 ─────────────────────────
        labels = input_ids.clone()
        for i, prompt in enumerate(prompts):
            # Tokenize prompt alone to find its length
            prompt_len = len(
                self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
            )
            labels[i, :prompt_len] = -100
        labels[attention_mask == 0] = -100  # mask padding

        # ── [GRAPH_REPR] positions ───────────────────────────────────────
        graph_repr_positions = torch.full((B,), -1, dtype=torch.long)
        for i in range(B):
            hits = (input_ids[i] == self.graph_repr_id).nonzero(as_tuple=True)[0]
            if len(hits) > 0:
                graph_repr_positions[i] = hits[0].item()

        # ── Group indices for L_repr ─────────────────────────────────────
        oi_to_batch: dict[int, list[int]] = {}
        for bi, b in enumerate(batch):
            oi = b["original_index"]
            oi_to_batch.setdefault(oi, []).append(bi)
        group_indices = list(oi_to_batch.values())

        # ── TRA: entity maps (B,T) + graph distances (B,N,N) ────────────
        entity_maps = torch.full((B, T), -1, dtype=torch.long)
        max_nodes = 0
        graphs = []
        sp_list = []
        for i in range(B):
            graph = self.parser.parse(full_texts[i], tasks[i])
            graphs.append(graph)
            sp = self.parser.compute_shortest_paths(graph, self.d_max)
            sp_list.append(sp)
            if graph["nodes"]:
                max_nodes = max(max_nodes, max(graph["nodes"]) + 1)
            offsets = enc["offset_mapping"][i].tolist()
            emap = self.parser.build_entity_map_from_offsets(
                full_texts[i], offsets, graph
            )
            for pos, nid in emap.items():
                if pos < T:
                    entity_maps[i, pos] = nid

        max_nodes = max(max_nodes, 1)
        graph_dists = torch.full(
            (B, max_nodes, max_nodes), self.d_max + 1, dtype=torch.long
        )
        for i in range(B):
            for (u, v), d in sp_list[i].items():
                if u < max_nodes and v < max_nodes:
                    graph_dists[i, u, v] = d

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "entity_maps": entity_maps,
            "graph_dists": graph_dists,
            "graph_repr_positions": graph_repr_positions,
            "group_indices": group_indices,
        }
