#!/bin/bash
# Zero-Shot SFT for LLaVA-OneVision-7B-SI — 2 GPU, ZeRO-3 WITHOUT CPU offload
# Uses CUDA 0 and 3
# ZeRO-3 no-offload ensures vision tower weights are properly gathered on save
# Effective batch = 2 GPUs × per_device_batch 1 × grad_accum 4 = 8 (matches IAD-R1's 4×1×2=8)
set -e

export CUDA_VISIBLE_DEVICES=0,3
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Paths. WORK_DIR is the workspace that holds LlamaFactory/, outputs/, hf_cache/ and the
# IAD-R1 checkout. It defaults to the parent of this repository (the cluster layout).
# On another machine:  export WORK_DIR=/path/to/workspace
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"

export HF_HOME=$WORK_DIR/hf_cache
export TRANSFORMERS_CACHE=$WORK_DIR/hf_cache
export TRITON_CACHE_DIR=/tmp/triton_cache_aacudad
export DISABLE_VERSION_CHECK=1

mkdir -p /tmp/triton_cache_aacudad

IAD_R1_SFT="${IAD_R1_SFT:-$WORK_DIR/iad-r1/IAD-R1/train/stage_sft}"
OUTPUT_DIR="$WORK_DIR/outputs/sft_llava_ov_7b_2gpu"
mkdir -p $OUTPUT_DIR

export PYTHONPATH=$IAD_R1_SFT

cd $IAD_R1_SFT
conda run -n iad_r1_sft --no-capture-output \
    torchrun --nproc_per_node=2 --nnodes=1 --master_port=12412 \
    train.py \
    --deepspeed $REPO_ROOT/configs/deepspeed/ds_z3_optim_offload.json \
    --stage sft \
    --do_train \
    --model_name_or_path llava-hf/llava-onevision-qwen2-7b-si-hf \
    --dataset iad_sft_small_train \
    --dataset_dir $WORK_DIR/LlamaFactory/data \
    --template llava_onevision_qwen \
    --finetuning_type full \
    --freeze_vision_tower false \
    --freeze_multi_modal_projector false \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 180000000 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --cutoff_len 8192 \
    --save_steps 100 \
    --save_total_limit 20 \
    --num_train_epochs 2 \
    --bf16 \
    --gradient_checkpointing \
    --trust_remote_code \
    2>&1 | tee $OUTPUT_DIR/train.log

echo "Done."
