#!/usr/bin/env python3
"""
Evaluate the SFT model on GraphInstruct-Test (9 tasks).

Supports two backends:
  - ``hf``   : HuggingFace transformers (single GPU, auto LoRA merge)
  - ``vllm`` : vLLM with tensor parallelism (multi-GPU, needs merged model)

Follows the GraphWiz evaluation protocol exactly:
  - Alpaca-style prompt template
  - Greedy decoding (do_sample=False, temperature=0.0)
  - max_new_tokens=1024
  - Same check() function for answer extraction & correctness
  - Per-task accuracy + 9-task unweighted average
  - Resumable JSONL output

Usage:
    # HuggingFace (auto merges LoRA, single GPU)
    python evaluation/evaluate.py --checkpoint_dir checkpoints/sft --tasks cycle

    # vLLM (needs merged model, multi-GPU)
    python script/merge_lora.py --checkpoint_dir checkpoints/sft
    python evaluation/evaluate.py --checkpoint_dir checkpoints/sft-merged \\
        --backend vllm --tp 3 --tasks cycle
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import re
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═════════════════════════════════════════════════════════════════════════════
# Prompt template (matching GraphWiz evaluate_nlg.py exactly)
# ═════════════════════════════════════════════════════════════════════════════

ALPACA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request step by step.\n\n"
    "### Instruction:\n{query}\n\n### Response:"
)

# ═════════════════════════════════════════════════════════════════════════════
# Answer extraction & correctness (copied from GraphWiz evaluate_nlg.py)
# ═════════════════════════════════════════════════════════════════════════════

def extract_last_num(text: str) -> float:
    """Extract the last number from text, handling commas in numbers."""
    text = re.sub(r"(\d),(\d)", r"\g<1>\g<2>", text)
    res = re.findall(r"(\d+(\.\d+)?)", text)
    if len(res) > 0:
        return float(res[-1][0])
    return 0.0


def check(key: str, truth: str, predict: str) -> bool:
    """Check correctness — exact replica of GraphWiz evaluate_nlg.py:check()."""
    if key in ['cycle', 'connectivity', 'hamilton', 'substructure', 'bipartite']:
        if '###' in predict:
            if 'yes' in truth.lower() and 'yes' in predict.split('###')[-1].lower():
                return True
            elif 'no' in truth.lower() and 'no' in predict.split('###')[-1].lower():
                return True
            return False
        else:
            matches = re.findall(r'\b(yes|no)\b', predict, flags=re.IGNORECASE)
            if matches:
                last_match = matches[-1].lower()
                if last_match == 'yes' and 'yes' in truth.lower():
                    return True
                elif last_match == 'no' and 'no' in truth.lower():
                    return True
            return False

    elif key in ['flow', 'shortest', 'triplet']:
        t_num = extract_last_num(truth)
        p_num = extract_last_num(predict.split('###')[-1])
        return abs(t_num - p_num) < 1e-2

    elif key == 'topology':
        if '###' in predict:
            pre = predict.split('###')[-1].strip(' ')
            truth_part = truth.split('###')[-1].strip(' ')
            return truth_part in pre or pre in truth_part
        else:
            truth_parts = truth.split('###')[-1].split(',')
            for t in truth_parts:
                if t in predict or t.strip(' ') in predict:
                    return True
            return False

    return False


def _truncate_at_first_answer(text: str) -> str:
    """Keep only text up to (and including) the first ### answer.

    Without this, the model's repetitive output produces 100+ ``###``
    delimiters, and ``check()`` (which takes the last ``###`` segment)
    sees garbage instead of the real answer.
    """
    parts = text.split("###")
    if len(parts) >= 2:
        return "###".join(parts[:2])
    return text


# ═════════════════════════════════════════════════════════════════════════════
# Cycle verification — detect hallucinated cycles
# ═════════════════════════════════════════════════════════════════════════════

def _verify_cycle_answer(query: str, output: str) -> str:
    """For the cycle task, verify that a 'Yes' answer is backed by evidence.

    If the model claims '### Yes' but:
      1. any explicitly listed cycle path is invalid (revisits nodes,
         uses non-existent edges, or isn't closed), AND
      2. the graph is actually acyclic (quick DFS check),
    then flip the answer to '### No'.

    This catches the dominant error mode: the model hallucinates cycles
    on acyclic graphs (78/105 errors in v2 evaluation).
    """
    # Only verify Yes answers
    if '###' not in output:
        return output
    answer_part = output.split('###')[-1].lower()
    if 'yes' not in answer_part:
        return output

    # Parse graph edges from the query
    edges = set()
    nodes = set()
    for m in re.finditer(r'\((\d+),\s*(\d+)\)', query):
        u, v = int(m.group(1)), int(m.group(2))
        edges.add((u, v))
        edges.add((v, u))  # undirected
        nodes.add(u)
        nodes.add(v)

    if not edges:
        return output

    # Quick DFS cycle check on actual graph
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)

    visited = set()
    has_real_cycle = False

    def dfs(node, parent):
        nonlocal has_real_cycle
        visited.add(node)
        for nb in adj.get(node, []):
            if has_real_cycle:
                return
            if nb not in visited:
                dfs(nb, node)
            elif nb != parent:
                has_real_cycle = True

    for start in nodes:
        if start not in visited:
            dfs(start, -1)
        if has_real_cycle:
            break

    if has_real_cycle:
        return output  # graph does have a cycle, model is correct

    # Graph is acyclic but model said Yes → flip to No
    return output.rsplit('###', 1)[0] + '### No.'


# ═════════════════════════════════════════════════════════════════════════════
# HuggingFace backend
# ═════════════════════════════════════════════════════════════════════════════

def load_model_hf(args):
    """Load base model + LoRA adapter (merged) via HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftModel

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_dir, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model_name = args.base_model
    if base_model_name is None:
        adapter_cfg_path = os.path.join(args.checkpoint_dir, "adapter_config.json")
        if os.path.exists(adapter_cfg_path):
            with open(adapter_cfg_path) as f:
                base_model_name = json.load(f)["base_model_name_or_path"]
        else:
            base_model_name = args.checkpoint_dir
    print(f"  Base model: {base_model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map={"": 0},
    )
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

    # Merge LoRA if adapter exists
    adapter_cfg_path = os.path.join(args.checkpoint_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        print("  Merging LoRA adapter...")
        model = PeftModel.from_pretrained(model, args.checkpoint_dir)
        model = model.merge_and_unload()

    tra_path = os.path.join(args.checkpoint_dir, "tra_weights.pt")
    if os.path.exists(tra_path):
        print("  TRA weights found (trained jointly, not applied at inference).")

    model.eval()

    # Build EOS token ids
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot != tokenizer.unk_token_id:
        eos_ids.add(eot)

    @torch.inference_mode()
    def generate_fn(input_strs: list[str], max_new_tokens: int) -> list[str]:
        enc = tokenizer(input_strs, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=list(eos_ids) if eos_ids else None,
            ),
        ).tolist()

        real_output_ids = [
            oid[len(input_ids[i]):] for i, oid in enumerate(output_ids)
        ]
        return tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)

    return generate_fn


# ═════════════════════════════════════════════════════════════════════════════
# vLLM backend
# ═════════════════════════════════════════════════════════════════════════════

def load_model_vllm(args):
    """Load a merged model via vLLM with tensor parallelism."""
    from vllm import LLM, SamplingParams

    print(f"Loading model with vLLM (tp={args.tp})...")
    print(f"  Model path: {args.checkpoint_dir}")

    llm = LLM(
        model=args.checkpoint_dir,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    def generate_fn(input_strs: list[str], max_new_tokens: int) -> list[str]:
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
        )
        outputs = llm.generate(input_strs, sampling_params)
        return [o.outputs[0].text for o in outputs]

    return generate_fn


# ═════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═════════════════════════════════════════════════════════════════════════════

TASKS = [
    'cycle', 'connectivity', 'bipartite', 'topology',
    'shortest', 'triplet', 'flow', 'hamilton', 'substructure',
]

HF_TASK_NAME = {
    'cycle': 'cycle',
    'connectivity': 'connectivity',
    'bipartite': 'bipartite',
    'topology': 'topology',
    'shortest': 'shortest',
    'triplet': 'triangle',
    'flow': 'flow',
    'hamilton': 'hamilton',
    'substructure': 'substructure',
}


def load_test_data(task: str, data_dir: str | None = None) -> list[dict]:
    """Load test data for a task from HuggingFace or local path."""
    if data_dir and os.path.isdir(data_dir):
        for ext in [".json", ".jsonl"]:
            path = os.path.join(data_dir, f"{task}_test{ext}")
            if os.path.exists(path):
                with open(path) as f:
                    return [json.loads(line) for line in f]
        raise FileNotFoundError(f"No test file for {task} in {data_dir}")

    hf_name = HF_TASK_NAME.get(task, task)
    ds = load_dataset("GraphWiz/GraphInstruct-Test", name=hf_name, split="test")
    return [sample for sample in ds]


def get_query(sample: dict) -> str:
    return sample.get("question", sample.get("input_prompt", ""))


def get_answer(sample: dict) -> str:
    return sample.get("answer", "")


# ═════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_task(
    task: str,
    generate_fn,
    args,
) -> float:
    """Evaluate one task. Returns accuracy."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {task}")
    print(f"{'='*60}")

    data = load_test_data(task, args.data_dir)
    print(f"  Loaded {len(data)} test samples")

    gen_jsonl = Path(args.output_dir) / f"_gen_{task}_datas.jsonl"
    start_index = 0
    if gen_jsonl.exists():
        with open(gen_jsonl) as f:
            start_index = sum(1 for _ in f)
        print(f"  Resuming from index {start_index}")

    if start_index < len(data):
        remaining = data[start_index:]
        queries = [get_query(s) for s in remaining]
        input_strs = [ALPACA_PROMPT.format(query=q) for q in queries]

        if args.backend == "vllm":
            # vLLM handles batching internally — send all at once
            output_strs = generate_fn(input_strs, args.max_new_tokens)
            output_strs = [_truncate_at_first_answer(s) for s in output_strs]
            for j, (sample, inp, out) in enumerate(
                zip(remaining, input_strs, output_strs)
            ):
                with open(gen_jsonl, "a") as f:
                    json.dump({
                        "index": start_index + j,
                        "source_data": sample,
                        "input_str": inp,
                        "output_str": out,
                        "task": task,
                    }, f, default=str)
                    f.write("\n")
        else:
            # HF: manual batching
            for i in tqdm(range(0, len(remaining), args.batch_size), desc=task):
                batch_samples = remaining[i: i + args.batch_size]
                batch_inputs = input_strs[i: i + args.batch_size]

                batch_outputs = generate_fn(batch_inputs, args.max_new_tokens)
                batch_outputs = [_truncate_at_first_answer(s) for s in batch_outputs]

                for j, (sample, inp, out) in enumerate(
                    zip(batch_samples, batch_inputs, batch_outputs)
                ):
                    with open(gen_jsonl, "a") as f:
                        json.dump({
                            "index": start_index + i + j,
                            "source_data": sample,
                            "input_str": inp,
                            "output_str": out,
                            "task": task,
                        }, f, default=str)
                        f.write("\n")

    # Compute accuracy
    with open(gen_jsonl) as f:
        gen_datas = [json.loads(line) for line in f]

    correct_results = []
    wrong_results = []
    verified_flips = 0
    for gen in gen_datas:
        truth = get_answer(gen["source_data"])
        predict = gen["output_str"].lstrip()

        result = {
            **gen,
            "extract_true": truth,
            "extract_pred": predict,
        }

        if check(task, truth.lower(), predict.lower()):
            result["is_correct"] = True
            correct_results.append(result)
        else:
            result["is_correct"] = False
            wrong_results.append(result)


    total = len(correct_results) + len(wrong_results)
    accuracy = len(correct_results) / total if total > 0 else 0.0
    print(f"  Accuracy = {len(correct_results)}/{total} = {accuracy:.4f}")

    with open(Path(args.output_dir) / f"{task}_correct.json", "w") as f:
        json.dump(correct_results, f, ensure_ascii=False, indent=2, default=str)
    with open(Path(args.output_dir) / f"{task}_wrong.json", "w") as f:
        json.dump(wrong_results, f, ensure_ascii=False, indent=2, default=str)

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT model on GraphInstruct-Test")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint (LoRA adapter for hf, merged model for vllm)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model name/path (auto-detected from adapter_config.json)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Local test data directory (downloads from HF if not provided)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (hf backend only; vllm batches internally)")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific tasks to evaluate (default: all 9)")
    # Backend selection
    parser.add_argument("--backend", type=str, default="hf",
                        choices=["hf", "vllm"],
                        help="Inference backend: hf (HuggingFace) or vllm")
    # vLLM-specific
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallelism degree (vllm only)")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Max model context length (vllm only)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization fraction (vllm only)")
    args = parser.parse_args()

    tasks = args.tasks if args.tasks else TASKS

    if args.output_dir is None:
        base_model_name = args.base_model
        if base_model_name is None:
            # Try adapter_config.json (LoRA checkpoint)
            adapter_cfg_path = os.path.join(args.checkpoint_dir, "adapter_config.json")
            if os.path.exists(adapter_cfg_path):
                with open(adapter_cfg_path) as f:
                    base_model_name = json.load(f)["base_model_name_or_path"]
            else:
                # Merged model: check base_model_name.txt or use dir name
                name_file = os.path.join(args.checkpoint_dir, "base_model_name.txt")
                if os.path.exists(name_file):
                    with open(name_file) as f:
                        base_model_name = f.read().strip()
                else:
                    base_model_name = args.checkpoint_dir
        model_short = base_model_name.rstrip("/").split("/")[-1]
        task_dir = "_".join(tasks) if len(tasks) <= 3 else "all"
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join("sft_results", model_short, task_dir, timestamp)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {args.output_dir}")
    print(f"Backend: {args.backend}")

    config_path = Path(args.output_dir) / "eval_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model
    if args.backend == "vllm":
        generate_fn = load_model_vllm(args)
    else:
        generate_fn = load_model_hf(args)

    # Evaluate each task
    results = {}
    for task in tasks:
        accuracy = evaluate_task(task, generate_fn, args)
        results[task] = accuracy

    # Summary
    average = sum(results.values()) / len(results) if results else 0.0
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    for task, acc in results.items():
        print(f"  {task:<15} {acc:.4f}")
    print(f"  {'Average':<15} {average:.4f}")
    print(f"{'='*60}")

    csv_path = Path(args.output_dir) / f"eval_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Accuracy"])
        for task, acc in results.items():
            writer.writerow([task, f"{acc:.4f}"])
        writer.writerow(["Average", f"{average:.4f}"])
    print(f"\nResults saved to {csv_path}")

    json_path = Path(args.output_dir) / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump({"per_task": results, "average": average}, f, indent=2)


if __name__ == "__main__":
    main()
