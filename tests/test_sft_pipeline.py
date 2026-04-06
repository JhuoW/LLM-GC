#!/usr/bin/env python3
"""
Stress-test the SFT pipeline end-to-end.

Tests every component with real data from GraphInstruct-Aug across all 9 tasks,
edge cases, and a mini forward pass with the actual model.
"""

import sys, os, json, re, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from model.tra import GraphParser, TopologyAwareAttention, patch_model_with_tra
from training.sft_dataset import (
    GraphInstructSFTDataset, GroupedBatchSampler, SFTCollator, ALPACA_PROMPT,
)
from training.sft_losses import compute_repr_loss

DATA_PATH = "dataset/GraphInstruct-Aug/train.jsonl"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
GRAPH_REPR_TOKEN = "[GRAPH_REPR]"

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} — {detail}")


# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. DATASET LOADING & FILTERING")
print("=" * 60)

ds = GraphInstructSFTDataset(DATA_PATH, k=4)
check("Dataset loads", len(ds) > 0, f"len={len(ds)}")
check("Group size = 5", ds.group_size == 5)
check("Sample count divisible by group_size", len(ds) % ds.group_size == 0)
check("Num groups", ds.num_groups == len(ds) // 5)

# Verify no "original" augmentation survived filtering
aug_types = set(ds.samples[i]["augmentation"] for i in range(min(1000, len(ds))))
check("No 'original' augmentation", "original" not in aug_types, f"found: {aug_types}")

# Verify group contiguity: every 5 consecutive samples share original_index
for g in range(min(100, ds.num_groups)):
    base = g * ds.group_size
    ois = [ds.samples[base + i]["original_index"] for i in range(ds.group_size)]
    if len(set(ois)) != 1:
        check(f"Group {g} contiguous", False, f"ois={ois}")
        break
else:
    check("First 100 groups contiguous", True)

# Verify each group has exactly [none, perm_0, perm_1, perm_2, perm_3]
for g in range(min(50, ds.num_groups)):
    base = g * ds.group_size
    augs = sorted(ds.samples[base + i]["augmentation"] for i in range(ds.group_size))
    expected = sorted(["none", "perm_0", "perm_1", "perm_2", "perm_3"])
    if augs != expected:
        check(f"Group {g} augmentations", False, f"got {augs}")
        break
else:
    check("First 50 groups have correct augmentation set", True)


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. GROUPED BATCH SAMPLER")
print("=" * 60)

sampler = GroupedBatchSampler(ds, groups_per_batch=1, shuffle=True, seed=42)
batches = list(sampler)
check("Sampler produces batches", len(batches) > 0)
check("All batches size = group_size", all(len(b) == ds.group_size for b in batches))

# Verify each batch is a complete group (same original_index)
bad_batches = 0
for b in batches[:200]:
    ois = set(ds.samples[i]["original_index"] for i in b)
    if len(ois) != 1:
        bad_batches += 1
check("First 200 batches are complete groups", bad_batches == 0, f"{bad_batches} bad")

# Verify all samples appear exactly once
all_indices = sorted(idx for b in batches for idx in b)
check("All indices covered", all_indices == list(range(len(ds))))

# Epoch shuffling changes order
sampler.set_epoch(1)
batches_ep1 = list(sampler)
check("Epoch shuffle changes batch order", batches[0] != batches_ep1[0])


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. TOKENIZER + [GRAPH_REPR] TOKEN")
print("=" * 60)

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tok.pad_token = tok.eos_token
num_added = tok.add_special_tokens({"additional_special_tokens": [GRAPH_REPR_TOKEN]})
repr_id = tok.convert_tokens_to_ids(GRAPH_REPR_TOKEN)

check("[GRAPH_REPR] added", num_added >= 0)
check("[GRAPH_REPR] is not UNK", repr_id != tok.unk_token_id, f"id={repr_id}")
check("[GRAPH_REPR] round-trips", tok.decode([repr_id]) == GRAPH_REPR_TOKEN)

# Verify it tokenizes as a single token in context
text = "edges are: (0,1). [GRAPH_REPR] Is there a path?"
ids = tok(text, add_special_tokens=False)["input_ids"]
check("[GRAPH_REPR] is single token in context", ids.count(repr_id) == 1)


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. GRAPH PARSER — ALL 9 TASKS")
print("=" * 60)

parser = GraphParser()

# Collect one sample per task from the dataset
task_samples = {}
for i in range(len(ds)):
    s = ds[i]
    t = s["task"]
    if t not in task_samples and s["augmentation"] == "none":
        task_samples[t] = s
    if len(task_samples) >= 9:
        break

for task, s in sorted(task_samples.items()):
    query = s["query"]
    graph = parser.parse(query, task)
    n_nodes = len(graph["nodes"])
    n_edges = len(graph["edges"])

    check(f"parse({task}): nodes>0", n_nodes > 0, f"nodes={n_nodes}")
    check(f"parse({task}): edges>0", n_edges > 0, f"edges={n_edges}")

    # Shortest paths
    sp = parser.compute_shortest_paths(graph, d_max=1)
    check(f"sp({task}): all pairs computed", len(sp) == n_nodes * n_nodes)
    # Self-distance = 0
    check(f"sp({task}): self-dist=0", all(sp[(u, u)] == 0 for u in graph["nodes"]))

    # Entity map
    emap = parser.build_entity_map(query, tok, graph)
    mapped_nodes = set(emap.values())
    check(f"emap({task}): nodes mapped", len(mapped_nodes) > 0, f"mapped={len(mapped_nodes)}")
    check(f"emap({task}): valid node ids", mapped_nodes.issubset(set(graph["nodes"])))


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. COLLATOR — ALL 9 TASKS")
print("=" * 60)

collator = SFTCollator(tokenizer=tok, d_max=1, max_length=2048)

for task, s in sorted(task_samples.items()):
    # Build a fake group: same sample repeated 5 times (for shape testing)
    group = [ds[i] for i in range(ds.group_size)]  # first group in dataset
    # Find a real group for this task
    for g in range(ds.num_groups):
        base = g * ds.group_size
        if ds.samples[base]["task"] == task:
            group = [ds[base + j] for j in range(ds.group_size)]
            break

    out = collator(group)
    B, T = out["input_ids"].shape

    check(f"collate({task}): shape B={B}", B == ds.group_size)
    check(f"collate({task}): labels has -100", (out["labels"] == -100).any().item())
    check(f"collate({task}): labels has real tokens", (out["labels"] != -100).any().item())

    # [GRAPH_REPR] found
    repr_pos = out["graph_repr_positions"]
    all_found = (repr_pos >= 0).all().item()
    check(f"collate({task}): [GRAPH_REPR] found", all_found, f"positions={repr_pos.tolist()}")

    # Distance matrix
    dm = out["distance_matrices"]
    check(f"collate({task}): dm shape", dm.shape == (B, T, T))
    has_entities = (dm >= 0).any().item()
    check(f"collate({task}): dm has entity pairs", has_entities)

    # Group indices
    gi = out["group_indices"]
    check(f"collate({task}): 1 group", len(gi) == 1)
    check(f"collate({task}): group size", len(gi[0]) == ds.group_size)


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. L_REPR LOSS — EDGE CASES")
print("=" * 60)

d = 128  # small dim for testing

# Case 1: identical hidden states → L_repr = 0
h_same = torch.ones(5, 50, d, requires_grad=True)
pos = torch.tensor([10, 10, 10, 10, 10])
gi = [[0, 1, 2, 3, 4]]
loss = compute_repr_loss(h_same, pos, gi)
check("L_repr(identical) ≈ 0", loss.item() < 1e-6, f"got {loss.item()}")

# Case 2: orthogonal hidden states → L_repr = 1
h_orth = torch.zeros(2, 50, d, requires_grad=True)
with torch.no_grad():
    h_orth[0, 10, 0] = 1.0
    h_orth[1, 10, 1] = 1.0
loss_orth = compute_repr_loss(h_orth, torch.tensor([10, 10]), [[0, 1]])
check("L_repr(orthogonal) = 1.0", abs(loss_orth.item() - 1.0) < 1e-5, f"got {loss_orth.item()}")

# Case 3: gradient flows
h_grad = torch.randn(5, 50, d, requires_grad=True)
loss_g = compute_repr_loss(h_grad, pos, gi)
loss_g.backward()
check("L_repr gradient flows", h_grad.grad is not None)
check("L_repr gradient nonzero", h_grad.grad.abs().sum() > 0)

# Case 4: missing [GRAPH_REPR] (position = -1) → graceful
pos_miss = torch.tensor([-1, -1, -1, -1, -1])
loss_miss = compute_repr_loss(torch.randn(5, 50, d), pos_miss, gi)
check("L_repr(all missing) = 0", loss_miss.item() == 0.0)

# Case 5: single sample group → no pairs → 0
loss_single = compute_repr_loss(torch.randn(1, 50, d), torch.tensor([10]), [[0]])
check("L_repr(single sample) = 0", loss_single.item() == 0.0)

# Case 6: multiple groups in one batch
h_multi = torch.randn(10, 50, d, requires_grad=True)
pos_multi = torch.tensor([10] * 10)
gi_multi = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
loss_multi = compute_repr_loss(h_multi, pos_multi, gi_multi)
check("L_repr(2 groups) computes", loss_multi.item() > 0)
loss_multi.backward()
check("L_repr(2 groups) gradient flows", h_multi.grad is not None)


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. TRA MODULE — BIAS COMPUTATION")
print("=" * 60)

tra = TopologyAwareAttention(num_layers=4, num_heads=8, d_max=1)

# Zero-init check
dm_test = torch.randint(-1, 3, (2, 20, 20))
bias = tra.get_bias(0, dm_test)
check("TRA zero-init: all bias = 0", (bias == 0).all().item())
check("TRA bias shape", bias.shape == (2, 8, 20, 20))

# After param update
with torch.no_grad():
    tra.distance_embeddings.fill_(1.0)
    tra.alpha.fill_(0.5)
bias2 = tra.get_bias(0, dm_test)
# Non-entity pairs (dm=-1) should still be 0
mask = (dm_test >= 0).unsqueeze(1).expand_as(bias2)
check("TRA non-entity pairs = 0", (bias2[~mask] == 0).all().item())
check("TRA entity pairs nonzero", bias2[mask].abs().sum() > 0)

# Different layers use different params
with torch.no_grad():
    tra.distance_embeddings[0, 0, 0] = 10.0
    tra.distance_embeddings[1, 0, 0] = 20.0
b0 = tra.get_bias(0, dm_test)
b1 = tra.get_bias(1, dm_test)
check("TRA layer 0 ≠ layer 1", not torch.equal(b0, b1))


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. MINI FORWARD PASS (actual model, 1 batch)")
print("=" * 60)

print("  Loading model (this may take a minute)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
).to("cuda:0")
old_size = model.get_input_embeddings().weight.shape[0]
model.resize_token_embeddings(len(tok))
new_size = model.get_input_embeddings().weight.shape[0]
with torch.no_grad():
    if new_size > old_size:
        mean_emb = model.get_input_embeddings().weight[:old_size].mean(dim=0)
        model.get_input_embeddings().weight[old_size:] = mean_emb

# LoRA
lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
model = get_peft_model(model, lora_cfg)
check("LoRA applied", any("lora" in n for n, _ in model.named_parameters()))

# TRA
cfg = model.config
while hasattr(cfg, "base_model_name_or_path") and not hasattr(cfg, "num_hidden_layers"):
    cfg = model.base_model.config if hasattr(model, "base_model") else cfg
    break
num_layers = getattr(cfg, "num_hidden_layers", 32)
num_heads = getattr(cfg, "num_attention_heads", 32)

tra_mod = TopologyAwareAttention(num_layers=num_layers, num_heads=num_heads, d_max=1)
patch_model_with_tra(model, tra_mod)
for p in tra_mod.parameters():
    p.requires_grad = True
check("TRA patched", hasattr(model, "tra"))

# Hook for last hidden state
last_hidden = {}
def _capture(mod, inp, out):
    last_hidden["h"] = out

base = model
while hasattr(base, "base_model") and base.base_model is not base:
    base = base.base_model
while hasattr(base, "model") and not hasattr(base, "norm"):
    base = base.model
hook_handle = base.norm.register_forward_hook(_capture)
check("Hook registered on final norm", True)

# Prepare a real batch
# Find a short group
for g in range(ds.num_groups):
    base_idx = g * ds.group_size
    items = [ds[base_idx + j] for j in range(ds.group_size)]
    max_chars = max(len(it["prompt"]) + len(it["response"]) for it in items)
    if max_chars < 1200:
        break

batch_data = [ds[base_idx + j] for j in range(ds.group_size)]
collator_test = SFTCollator(tokenizer=tok, d_max=1, max_length=512)
batch = collator_test(batch_data)

print(f"  Batch: task={batch_data[0]['task']}, shape={tuple(batch['input_ids'].shape)}")

# Forward pass
dm = batch.pop("distance_matrices")
repr_pos = batch.pop("graph_repr_positions")
group_idx = batch.pop("group_indices")

tra_mod.set_distance_matrix(dm.to(model.device))
with torch.no_grad():
    outputs = model(
        input_ids=batch["input_ids"].to(model.device),
        attention_mask=batch["attention_mask"].to(model.device),
        labels=batch["labels"].to(model.device),
    )
tra_mod.clear_distance_matrix()

check("Forward pass succeeds", True)
check("Loss is finite", torch.isfinite(outputs.loss).item(), f"loss={outputs.loss.item()}")
check("Loss > 0", outputs.loss.item() > 0)

# Check hidden state captured
check("Last hidden captured by hook", "h" in last_hidden)
h = last_hidden["h"]
check("Hidden shape matches batch", h.shape[0] == batch["input_ids"].shape[0])
check("Hidden has model dim", h.shape[2] == getattr(cfg, "hidden_size", 4096))

# L_repr from real hidden states
l_repr = compute_repr_loss(h, repr_pos.to(h.device), group_idx)
check("L_repr from real model", torch.isfinite(l_repr).item(), f"l_repr={l_repr.item()}")
check("L_repr value reasonable (0-2)", 0 <= l_repr.item() <= 2.0, f"got {l_repr.item()}")

# Compound loss
l_total = outputs.loss + 0.1 * l_repr
check("Compound loss finite", torch.isfinite(l_total).item())

hook_handle.remove()

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("9. GRADIENT FLOW (backward through compound loss)")
print("=" * 60)

# Enable grad for a proper backward test
tra_mod2 = TopologyAwareAttention(num_layers=num_layers, num_heads=num_heads, d_max=1)
patch_model_with_tra(model, tra_mod2)
for p in tra_mod2.parameters():
    p.requires_grad = True

last_hidden2 = {}
base2 = model
while hasattr(base2, "base_model") and base2.base_model is not base2:
    base2 = base2.base_model
while hasattr(base2, "model") and not hasattr(base2, "norm"):
    base2 = base2.model
hook2 = base2.norm.register_forward_hook(lambda m, i, o: last_hidden2.update({"h": o}))

tra_mod2.set_distance_matrix(dm.to(model.device))
outputs2 = model(
    input_ids=batch["input_ids"].to(model.device),
    attention_mask=batch["attention_mask"].to(model.device),
    labels=batch["labels"].to(model.device),
)
h2 = last_hidden2["h"]
l_repr2 = compute_repr_loss(h2, repr_pos.to(h2.device), group_idx)
l_total2 = outputs2.loss + 0.1 * l_repr2

l_total2.backward()
tra_mod2.clear_distance_matrix()
hook2.remove()

# Check LoRA grads
lora_has_grad = any(
    p.grad is not None and p.grad.abs().sum() > 0
    for n, p in model.named_parameters() if "lora" in n and p.requires_grad
)
check("LoRA parameters have gradients", lora_has_grad)

# Check TRA grads
tra_has_grad = any(
    p.grad is not None and p.grad.abs().sum() > 0
    for p in tra_mod2.parameters()
)
check("TRA parameters have gradients", tra_has_grad)

model.zero_grad()


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 60)
sys.exit(1 if failed > 0 else 0)
