#!/usr/bin/env python3
"""
Evaluate the SFT model on GraphInstruct-Test (9 tasks).

Follows the GraphWiz evaluation protocol exactly:
  - Alpaca-style prompt template
  - Greedy decoding (do_sample=False, temperature=0.0)
  - max_new_tokens=1024
  - Same check() function for answer extraction & correctness
  - Per-task accuracy + 9-task unweighted average
  - Resumable JSONL output

Usage:
    python evaluation/evaluate.py \
        --checkpoint_dir checkpoints/sft \
        --output_dir results/sft_epoch2 \
        --batch_size 4
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import PeftModel

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


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════

def load_model(args):
    """Load base model + LoRA adapter (merged).

    Note: TRA is NOT applied during inference.  The TRA path in patch.py
    uses manual attention without KV-cache support, which is incompatible
    with ``model.generate()``.  Since LoRA weights are merged into the
    base model, the learned graph-structure knowledge is already baked
    into the attention weights.  TRA served as a training-time auxiliary
    that guided LoRA to learn better representations.
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_dir, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model_name = args.base_model
    if base_model_name is None:
        # Read from adapter_config.json
        adapter_cfg_path = os.path.join(args.checkpoint_dir, "adapter_config.json")
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg["base_model_name_or_path"]
    print(f"  Base model: {base_model_name}")

    # Use single GPU: 8B model in bf16 (~16GB) fits easily on one GPU.
    # device_map="auto" with pipeline parallelism breaks Flash Attention 2,
    # producing garbage output when the model is split across GPUs.
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map={"": 0},
    )

    # Resize embeddings to match tokenizer (same logic as training)
    model.resize_token_embeddings(
        int(8 * math.ceil(len(tokenizer) / 8.0))
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.checkpoint_dir)
    model = model.merge_and_unload()
    print("  LoRA merged into base model.")

    # TRA info (not applied during generation — see docstring)
    tra_path = os.path.join(args.checkpoint_dir, "tra_weights.pt")
    if os.path.exists(tra_path):
        print("  TRA weights found (trained jointly, not applied at inference).")

    model.eval()
    return model, tokenizer


# ═════════════════════════════════════════════════════════════════════════════
# Batch inference
# ═════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def batch_generate(
    model,
    tokenizer,
    input_strs: list[str],
    max_new_tokens: int = 1024,
) -> list[str]:
    """Run batched greedy generation (GraphWiz protocol).

    Uses EOS token to stop generation.  The model should emit EOS after
    ``### <answer>``.  As a safety net we also set max_new_tokens.
    """
    enc = tokenizer(
        input_strs,
        padding=True,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    # Build list of EOS / stop token ids
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    # LLaMA-3 has multiple stop tokens
    for name in ("eos_token_id",):
        val = getattr(tokenizer, name, None)
        if isinstance(val, list):
            eos_ids.update(val)
        elif isinstance(val, int):
            eos_ids.add(val)
    # Also stop on <|eot_id|> if present
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot != tokenizer.unk_token_id:
        eos_ids.add(eot)

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

    # Decode only the newly generated tokens
    real_output_ids = [
        output_id[len(input_ids[i]):] for i, output_id in enumerate(output_ids)
    ]
    output_strs = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)

    # Truncate after the first ### answer to avoid repetition.
    # The model may repeat "### Yes ### reasoning ### Yes ..." 100+ times.
    # GraphWiz's check() takes the LAST ### segment, so we keep only the
    # first two segments (reasoning + answer) to make the last == first.
    output_strs = [_truncate_at_first_answer(s) for s in output_strs]

    return output_strs


def _truncate_at_first_answer(text: str) -> str:
    """Keep only text up to (and including) the first ### answer.

    Without this, the model's repetitive output produces 100+ ``###``
    delimiters, and ``check()`` (which takes the last ``###`` segment)
    sees garbage instead of the real answer.
    """
    parts = text.split("###")
    if len(parts) >= 2:
        # parts[0] = reasoning, parts[1] = " Yes." or " 42" etc.
        return "###".join(parts[:2])
    return text


# ═════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═════════════════════════════════════════════════════════════════════════════

TASKS = [
    'cycle', 'connectivity', 'bipartite', 'topology',
    'shortest', 'triplet', 'flow', 'hamilton', 'substructure',
]

# HuggingFace config name mapping (test set uses "triangle" for triplet)
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
        # Local JSONL files: {task}_test.json or {task}_test.jsonl
        for ext in [".json", ".jsonl"]:
            path = os.path.join(data_dir, f"{task}_test{ext}")
            if os.path.exists(path):
                with open(path) as f:
                    return [json.loads(line) for line in f]
        raise FileNotFoundError(f"No test file for {task} in {data_dir}")

    # Download from HuggingFace
    hf_name = HF_TASK_NAME.get(task, task)
    ds = load_dataset("GraphWiz/GraphInstruct-Test", name=hf_name, split="test")
    return [sample for sample in ds]


def get_query(sample: dict) -> str:
    """Extract the query text from a test sample."""
    return sample.get("question", sample.get("input_prompt", ""))


def get_answer(sample: dict) -> str:
    """Extract the ground truth answer from a test sample."""
    return sample.get("answer", "")


# ═════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_task(
    task: str,
    model,
    tokenizer,
    args,
) -> float:
    """Evaluate one task. Returns accuracy."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {task}")
    print(f"{'='*60}")

    data = load_test_data(task, args.data_dir)
    print(f"  Loaded {len(data)} test samples")

    # Resumable output
    gen_jsonl = Path(args.output_dir) / f"_gen_{task}_datas.jsonl"
    start_index = 0
    if gen_jsonl.exists():
        with open(gen_jsonl) as f:
            start_index = sum(1 for _ in f)
        print(f"  Resuming from index {start_index}")

    # Run inference
    for i in tqdm(range(start_index, len(data), args.batch_size), desc=task):
        batch_samples = data[i: i + args.batch_size]

        queries = [get_query(s) for s in batch_samples]
        input_strs = [ALPACA_PROMPT.format(query=q) for q in queries]

        output_strs = batch_generate(
            model, tokenizer, input_strs, args.max_new_tokens,
        )

        for j, (sample, inp, out) in enumerate(
            zip(batch_samples, input_strs, output_strs)
        ):
            with open(gen_jsonl, "a") as f:
                json.dump({
                    "index": i + j,
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

    # Save correct/wrong for analysis
    with open(Path(args.output_dir) / f"{task}_correct.json", "w") as f:
        json.dump(correct_results, f, ensure_ascii=False, indent=2, default=str)
    with open(Path(args.output_dir) / f"{task}_wrong.json", "w") as f:
        json.dump(wrong_results, f, ensure_ascii=False, indent=2, default=str)

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT model on GraphInstruct-Test")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to SFT checkpoint (LoRA + TRA + tokenizer)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model name/path (auto-detected from adapter_config.json)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (default: {checkpoint_dir}/eval_results)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Local test data directory (downloads from HF if not provided)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific tasks to evaluate (default: all 9)")
    args = parser.parse_args()

    tasks = args.tasks if args.tasks else TASKS

    if args.output_dir is None:
        # Auto-build: sft_results/<base_model>/<datetime>/
        base_model_name = args.base_model
        if base_model_name is None:
            adapter_cfg_path = os.path.join(args.checkpoint_dir, "adapter_config.json")
            with open(adapter_cfg_path) as f:
                base_model_name = json.load(f)["base_model_name_or_path"]
        # Extract short model name (e.g. "Llama-3.1-8B-Instruct")
        model_short = base_model_name.rstrip("/").split("/")[-1]
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join("sft_results", model_short, timestamp)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {args.output_dir}")

    # Save eval config for reproducibility
    config_path = Path(args.output_dir) / "eval_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model (LoRA merged, no TRA at inference)
    model, tokenizer = load_model(args)

    # Evaluate each task
    results = {}
    for task in tasks:
        accuracy = evaluate_task(task, model, tokenizer, args)
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

    # Save CSV (matching GraphWiz format)
    csv_path = Path(args.output_dir) / f"eval_results_bs{args.batch_size}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Accuracy"])
        for task, acc in results.items():
            writer.writerow([task, f"{acc:.4f}"])
        writer.writerow(["Average", f"{average:.4f}"])
    print(f"\nResults saved to {csv_path}")

    # Also save as JSON for easier programmatic access
    json_path = Path(args.output_dir) / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump({"per_task": results, "average": average}, f, indent=2)


if __name__ == "__main__":
    main()
