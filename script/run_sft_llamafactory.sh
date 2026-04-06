#!/bin/bash
# Phase 1 SFT via LLaMA-Factory
#
# Usage:
#   bash script/run_sft_llamafactory.sh [CONFIG] [GPUS]

set -e

CONFIG=${1:-"config/sft_llamafactory.yaml"}
GPUS=${2:-"0,1,2"}

CUDA_VISIBLE_DEVICES=$GPUS llamafactory-cli train "$CONFIG"
