#!/bin/bash
# Evaluate SFT model on GraphInstruct-Test.
# Follows the GraphWiz evaluation protocol for fair comparison.
#
# Results saved to: sft_results/<base_model>/<tasks>/<datetime>/
#
# Three modes:
#   hf       — HuggingFace single GPU
#   hf-para  — HuggingFace parallel: one task per GPU (fastest for HF)
#   vllm     — vLLM with tensor parallelism
#
# Usage:
#   bash script/run_eval.sh [CHECKPOINT_DIR] [TASKS] [BACKEND] [GPU/TP]
#
# Examples:
#   bash script/run_eval.sh checkpoints/sft cycle                         # HF, 1 GPU
#   bash script/run_eval.sh checkpoints/sft all hf-para "0,1,2"          # HF parallel, 3 GPUs
#   bash script/run_eval.sh checkpoints/sft-merged all vllm 1            # vLLM TP=1

set -e

CHECKPOINT_DIR=${1:-"checkpoints/sft_v2"}
TASKS=${2:-"all"}
BACKEND=${3:-"hf"}
GPU_OR_TP=${4:-"0"}

ALL_TASKS="cycle connectivity bipartite topology shortest triplet flow hamilton substructure"

echo "=========================================="
echo "  LLM-GC Evaluation"
echo "=========================================="
echo "  Checkpoint : ${CHECKPOINT_DIR}"
echo "  Tasks      : ${TASKS}"
echo "  Backend    : ${BACKEND}"

TASK_LIST=""
if [ "${TASKS}" = "all" ]; then
    TASK_LIST="${ALL_TASKS}"
else
    TASK_LIST="${TASKS}"
fi

# ── vLLM backend ────────────────────────────────────────────
if [ "${BACKEND}" = "vllm" ]; then
    echo "  TP         : ${GPU_OR_TP}"
    echo "=========================================="
    export FLASHINFER_DISABLE_VERSION_CHECK=1

    TASK_ARGS=""
    if [ "${TASKS}" != "all" ]; then
        TASK_ARGS="--tasks ${TASK_LIST}"
    fi

    python evaluation/evaluate.py \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        ${TASK_ARGS} \
        --backend vllm \
        --tp "${GPU_OR_TP}" \
        --max_new_tokens 1024 \

# ── HF parallel backend ────────────────────────────────────
elif [ "${BACKEND}" = "hf-para" ]; then
    IFS=',' read -ra GPUS <<< "${GPU_OR_TP}"
    NUM_GPUS=${#GPUS[@]}
    echo "  GPUs       : ${GPU_OR_TP} (${NUM_GPUS} GPUs)"
    echo "=========================================="

    # Build output dir (same timestamp for all tasks)
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    # Detect base model name
    if [ -f "${CHECKPOINT_DIR}/adapter_config.json" ]; then
        MODEL_SHORT=$(python3 -c "import json; print(json.load(open('${CHECKPOINT_DIR}/adapter_config.json'))['base_model_name_or_path'].rstrip('/').split('/')[-1])")
    elif [ -f "${CHECKPOINT_DIR}/base_model_name.txt" ]; then
        MODEL_SHORT=$(python3 -c "print(open('${CHECKPOINT_DIR}/base_model_name.txt').read().strip().rstrip('/').split('/')[-1])")
    else
        MODEL_SHORT=$(basename "${CHECKPOINT_DIR}")
    fi

    if [ "${TASKS}" = "all" ]; then
        TASK_DIR="all"
    else
        TASK_DIR=$(echo "${TASK_LIST}" | tr ' ' '_')
    fi
    OUTPUT_DIR="sft_results/${MODEL_SHORT}/${TASK_DIR}/${TIMESTAMP}"
    mkdir -p "${OUTPUT_DIR}"
    echo "Output: ${OUTPUT_DIR}"

    # Distribute tasks round-robin across GPUs
    TASK_ARRAY=(${TASK_LIST})
    PIDS=()
    for i in "${!TASK_ARRAY[@]}"; do
        GPU_IDX=$((i % NUM_GPUS))
        GPU=${GPUS[$GPU_IDX]}
        TASK=${TASK_ARRAY[$i]}
        echo "  Launching ${TASK} on GPU ${GPU}..."
        CUDA_VISIBLE_DEVICES=${GPU} python evaluation/evaluate.py \
            --checkpoint_dir "${CHECKPOINT_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --tasks ${TASK} \
            --backend hf \
            --batch_size 4 \
            --max_new_tokens 1024 \
            --attn_implementation flash_attention_2 \
            > "${OUTPUT_DIR}/${TASK}.log" 2>&1 &
        PIDS+=($!)
    done

    echo ""
    echo "  ${#TASK_ARRAY[@]} tasks launched across ${NUM_GPUS} GPUs. Waiting..."
    echo ""

    # Wait and report
    FAILED=0
    for i in "${!PIDS[@]}"; do
        PID=${PIDS[$i]}
        TASK=${TASK_ARRAY[$i]}
        if wait ${PID}; then
            ACC=$(grep "Accuracy = " "${OUTPUT_DIR}/${TASK}.log" | tail -1 | sed 's/.*= //')
            echo "  [DONE] ${TASK}: ${ACC}"
        else
            echo "  [FAIL] ${TASK} (see ${OUTPUT_DIR}/${TASK}.log)"
            FAILED=$((FAILED + 1))
        fi
    done

    if [ ${FAILED} -gt 0 ]; then
        echo ""
        echo "WARNING: ${FAILED} task(s) failed."
        exit 1
    fi

    # Aggregate results
    echo ""
    python3 -c "
import json, csv, os
output_dir = '${OUTPUT_DIR}'
tasks = '${TASK_LIST}'.split()
results = {}
for task in tasks:
    correct_f = os.path.join(output_dir, f'{task}_correct.json')
    wrong_f = os.path.join(output_dir, f'{task}_wrong.json')
    if os.path.exists(correct_f) and os.path.exists(wrong_f):
        nc = len(json.load(open(correct_f)))
        nw = len(json.load(open(wrong_f)))
        results[task] = nc / (nc + nw) if (nc + nw) > 0 else 0.0

avg = sum(results.values()) / len(results) if results else 0
print('=' * 60)
print('RESULTS SUMMARY')
print('=' * 60)
for t, a in results.items():
    print(f'  {t:<15} {a:.4f}')
print(f\"  {'Average':<15} {avg:.4f}\")
print('=' * 60)

with open(os.path.join(output_dir, 'eval_results.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Task', 'Accuracy'])
    for t, a in results.items():
        w.writerow([t, f'{a:.4f}'])
    w.writerow(['Average', f'{avg:.4f}'])

with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
    json.dump({'per_task': results, 'average': avg}, f, indent=2)

print(f'\nResults saved to {output_dir}/eval_results.csv')
"

# ── HF single GPU backend ──────────────────────────────────
else
    echo "  GPU        : ${GPU_OR_TP}"
    echo "=========================================="

    TASK_ARGS=""
    if [ "${TASKS}" != "all" ]; then
        TASK_ARGS="--tasks ${TASK_LIST}"
    fi

    CUDA_VISIBLE_DEVICES=${GPU_OR_TP} python evaluation/evaluate.py \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        ${TASK_ARGS} \
        --backend hf \
        --batch_size 4 \
        --max_new_tokens 1024 \
        --attn_implementation flash_attention_2
fi
