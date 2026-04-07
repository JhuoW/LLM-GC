#!/bin/bash
# Evaluate SFT model on GraphInstruct-Test.
# Follows the GraphWiz evaluation protocol for fair comparison.
#
# Results saved to: sft_results/<base_model>/<datetime>/
# Model runs on a single GPU (8B bf16 ~ 16GB).
#
# Usage:
#   bash script/run_eval.sh [CHECKPOINT_DIR] [TASKS] [GPU]
#
# Examples:
#   bash script/run_eval.sh                                 # all 9 tasks, GPU 0
#   bash script/run_eval.sh checkpoints/sft cycle           # cycle only
#   bash script/run_eval.sh checkpoints/sft "cycle connectivity" 1  # 2 tasks, GPU 1
#   bash script/run_eval.sh checkpoints/sft all 2           # all tasks, GPU 2

set -e

CHECKPOINT_DIR=${1:-"checkpoints/sft"}
TASKS=${2:-"all"}
GPU=${3:-"0"}

echo "=========================================="
echo "  LLM-GC Evaluation"
echo "=========================================="
echo "  Checkpoint : ${CHECKPOINT_DIR}"
echo "  Tasks      : ${TASKS}"
echo "  GPU        : ${GPU}"
echo "=========================================="

TASK_ARGS=""
if [ "${TASKS}" != "all" ]; then
    TASK_ARGS="--tasks ${TASKS}"
fi

CUDA_VISIBLE_DEVICES=${GPU} python evaluation/evaluate.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    ${TASK_ARGS} \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --attn_implementation flash_attention_2
