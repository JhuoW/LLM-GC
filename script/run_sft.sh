#!/bin/bash
# Phase 1 SFT training — compound loss (L_task + λ_repr · L_repr)
# Raw DeepSpeed loop, following GraphWiz convention.
#
# Usage:
#   bash script/run_sft.sh [OUTPUT_DIR] [GPUS]
# python script/merge_lora.py --checkpoint_dir checkpoints/sft_v2
#  bash script/run_eval.sh checkpoints/sft_v2-merged cycle vllm 1  
# bash script/run_eval.sh checkpoints/sft_v2-merged all vllm 1
set -e

OUTPUT_DIR=${1:-"checkpoints/sft_v2"}
GPUS=${2:-"0,1,2"}

# Kill stale processes on the master port
kill $(lsof -ti:29501) 2>/dev/null || true
sleep 1

deepspeed --include "localhost:${GPUS}" \
    --master_port 29501 \
    training/sft_train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --data_path dataset/GraphInstruct-Aug/train.jsonl \
    --attn_implementation flash_attention_2 \
    --d_max 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lambda_repr 0.1 \
    --k 4 \
    --max_seq_len 2048 \
    --per_device_train_batch_size 5 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_steps 500 \
    --weight_decay 0.0 \
    --gradient_checkpointing \
    --zero_stage 2 \
    --seed 1234 \
    --logging_steps 10 \
    --save_per_epoch \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project llm-gc-sft \
    --wandb_run_name "sft-llama3-8b-tra"
