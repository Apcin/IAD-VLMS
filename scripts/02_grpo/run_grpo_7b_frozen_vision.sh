#!/bin/bash
# Paths below assume the TU Delft workspace /bulk/aacudad/reasoning_traces (the parent of this repo). Replace with your WORK_DIR.
# GRPO from SFT ckpt-564 with the vision tower frozen (merger + LM trained). Identical to run 2 otherwise.
# SMOKE=1 -> 2 steps into a scratch dir.
set -e
export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache; export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONPATH=/bulk/aacudad/reasoning_traces/Training/iad_r1_grpo_custom/stage_rl
export MODEL_NAME_OR_PATH=/bulk/aacudad/reasoning_traces/outputs/_hf_pull/AnomalyThink-Qwen2.5-VL-7B-SFT   # = SFT ckpt-564 (epoch 3), the original dir has no weights any more
export DATASET_NAME=/bulk/aacudad/reasoning_traces/Training/datasets_small_15k_c1_only/grpo_train.json
export WANDB_MODE=disabled DEBUG_MODE=True CUDA_VISIBLE_DEVICES=1,2 FREEZE_VISION_TOWER=1
export TRITON_CACHE_DIR=/bulk/aacudad/reasoning_traces/tmp_cache/triton PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "${SMOKE:-0}" = "1" ]; then OUTPUT_DIR=/bulk/aacudad/reasoning_traces/outputs/_smoke_frozenvision; EXTRA="--max_steps 2 --save_steps 1"; rm -rf $OUTPUT_DIR; else OUTPUT_DIR=/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_6k_frozenvision_run1; EXTRA="--num_train_epochs 2 --save_steps 265"; fi
export OUTPUT_DIR LOG_PATH=$OUTPUT_DIR/reward.log; mkdir -p $OUTPUT_DIR
source ~/miniconda3/etc/profile.d/conda.sh; conda activate llama_sft
export GEMINI_JUDGE_URL=http://127.0.0.1:5200
python3 -c "import urllib.request,json; r=urllib.request.Request('http://127.0.0.1:5200/embed_similarity',data=json.dumps({'text1':'scratch','text2':'scratch'}).encode(),headers={'Content-Type':'application/json'}); s=json.loads(urllib.request.urlopen(r,timeout=30).read())['similarity']; assert s>0.99, s; print('[preflight] embed server 5200 OK, similarity(scratch,scratch)=%.3f'%s)" || { echo "[preflight] embed server :5200 NOT responding, aborting"; exit 1; }
torchrun --nproc_per_node=2 --nnodes=1 --master_port=29511 \
  /bulk/aacudad/reasoning_traces/Training/iad_r1_grpo_custom/stage_rl/grpo_ad.py \
  --deepspeed /bulk/aacudad/reasoning_traces/Training/zero3_offload.json \
  --output_dir $OUTPUT_DIR --model_name_or_path $MODEL_NAME_OR_PATH --dataset_name $DATASET_NAME --image_path / \
  --reward_funcs accuracy format \
  --use_vllm_for_gen false --use_system_prompt false --max_prompt_length 4096 --max_completion_length 512 \
  --num_generations 4 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --logging_steps 1 \
  --bf16 true --report_to none --gradient_checkpointing true --attn_implementation sdpa --max_pixels 480000 \
  --save_only_model true --single_img 1 $EXTRA 2>&1 | tee $OUTPUT_DIR/train.log
