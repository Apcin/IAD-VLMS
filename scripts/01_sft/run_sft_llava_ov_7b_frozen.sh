#!/bin/bash
# LLaVA-OneVision-7B-SI SFT on AnomalyThink with FROZEN VISION TOWER — the backbone-comparison run.
# Mirrors the Qwen2.5-VL-7B frozen-encoder headline recipe: frozen tower, trainable projector+LM,
# LR 1e-5, effective batch 32, 4 epochs with per-epoch checkpoints (best-epoch selection like Qwen).
# GPUs: cuda 1,2. All HF downloads/cache stay under $WORK_DIR (see the path block below).
#
# Registry: <dataset> must be a key in your LlamaFactory data/dataset_info.json. The keys
# this line used (iad_sft_6k_train, iad_sft_iter2, llava_iter1_C) are in
# configs/dataset_info.json and configs/dataset_info_additions.json; merge the additions in
# and expand their ${WORK_DIR} placeholder first. The trace JSONs themselves point at raw
# Real-IAD images, which this repository does not redistribute (README section 3).
#
# Usage: run_sft_llava_ov_7b_frozen.sh <dataset> <cuda_devices> [master_port]
#   e.g. run_sft_llava_ov_7b_frozen.sh iad_sft_6k_train 1,2 12455
set -e
DATASET="${1:?dataset name from dataset_info.json, e.g. iad_sft_6k_train}"
GPUS="${2:?cuda devices, e.g. 1,2}"
PORT="${3:-12455}"

export CUDA_VISIBLE_DEVICES=${GPUS}
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Paths. WORK_DIR is the workspace that holds LlamaFactory/, outputs/, hf_cache/ and the
# IAD-R1 checkout. It defaults to the parent of this repository (the cluster layout).
# On another machine:  export WORK_DIR=/path/to/workspace
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"

export HF_HOME=$WORK_DIR/hf_cache
export TRITON_CACHE_DIR=$WORK_DIR/tmp_cache/triton_llava_frozen_${DATASET}
export DISABLE_VERSION_CHECK=1
mkdir -p "$TRITON_CACHE_DIR"

# The IAD-R1 stage_sft trainer (LlamaFactory fork). Clone it next to this repo, or
# point IAD_R1_SFT at your own checkout.
IAD_R1_SFT="${IAD_R1_SFT:-$WORK_DIR/iad-r1/IAD-R1/train/stage_sft}"
OUTPUT_DIR="$WORK_DIR/outputs/sft_llava_ov_7b_frozen_${DATASET}"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH=$IAD_R1_SFT

cd $IAD_R1_SFT
conda run -n iad_r1_sft --no-capture-output \
    torchrun --nproc_per_node=2 --nnodes=1 --master_port=${PORT} \
    train.py \
    --deepspeed $REPO_ROOT/configs/deepspeed/ds_z3_optim_offload.json \
    --stage sft \
    --do_train \
    --model_name_or_path llava-hf/llava-onevision-qwen2-7b-si-hf \
    --dataset ${DATASET} \
    --dataset_dir $WORK_DIR/LlamaFactory/data \
    --template llava_onevision_qwen \
    --finetuning_type full \
    --freeze_vision_tower true \
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
    --save_steps 188 \
    --save_total_limit 6 \
    --num_train_epochs 4 \
    --save_only_model true \
    --bf16 \
    --gradient_checkpointing \
    --trust_remote_code \
    2>&1 | tee $OUTPUT_DIR/train.log

echo "[$(date)] LLaVA-OV FROZEN SFT (${DATASET}) DONE."
touch $OUTPUT_DIR/train.done
