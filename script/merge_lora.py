#!/usr/bin/env python3
"""Merge LoRA adapter into the base model and save for vLLM inference.

Usage:
    python script/merge_lora.py [--checkpoint_dir checkpoints/sft] [--output_dir checkpoints/sft-merged]
"""

import argparse
import json
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/sft_v2")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.checkpoint_dir.rstrip("/") + "-merged"

    # Load tokenizer (from checkpoint, already has [GRAPH_REPR])
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    # Load base model
    adapter_cfg_path = f"{args.checkpoint_dir}/adapter_config.json"
    with open(adapter_cfg_path) as f:
        base_model_name = json.load(f)["base_model_name_or_path"]
    print(f"Loading base model: {base_model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

    # Merge LoRA
    print("Merging LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.checkpoint_dir)
    model = model.merge_and_unload()

    # Save
    print(f"Saving merged model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Store base model name for eval script to pick up
    import os
    with open(os.path.join(args.output_dir, "base_model_name.txt"), "w") as f:
        f.write(base_model_name)
    print("Done.")


if __name__ == "__main__":
    main()
