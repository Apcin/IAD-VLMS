# Real-IAD v4 local Qwen3-VL reasoning-trace pipeline

> **Current implementation:** Real-IAD supervision construction is unchanged, but the
> remote Gemini teacher has been replaced by local `Qwen/Qwen3-VL-8B-Instruct` inference.
> The canonical generator is `generate_realiad_traces_qwen3vl.py`;
> `generate_realiad_traces_v4.py` remains only as a backward-compatible launcher.

## Current pipeline

```text
download_realiad.py
  -> data/Real-IAD/images/                         images + NG masks
sample_realiad_v4.py
  -> 3k_realiad_v4/{sft_train,grpo_train}.json   fixed 3K/3K indexes
generate_realiad_traces_qwen3vl.py
  -> Qwen3-VL JSON-schema constrained decoding
  -> validate_structured_trace.py                executable semantic/format QC
  -> data/output/reasoning_traces_realiad_qwen3vl8b/{sft,grpo}/trace_*.json
compile_traces_v4.py
  -> realiad_qwen3vl8b_compiled/{sft_train,grpo_train}.json
```

### Inputs and outputs

- An anomalous sample sends three images in one independent request: original query,
  red Mask overlay, and a deterministic normal reference from the same product and,
  where available, the same camera.
- A normal sample sends only its original query image.
- The outer response is constrained to JSON with Pydantic fields `image_id` and
  `reasoning`. The inner reasoning must be exactly
  `<think>...</think><location>...</location><type>...</type><answer>Yes</answer>` for
  anomalies, or `<think>...</think><answer>No</answer>` for normal samples.
- Each accepted `<think>` contains 120--200 words. The validator also checks the answer
  against the source label, the type against the Real-IAD defect code, the location
  against the Mask-derived 3x3 cells, and rejects supervision/overlay/reference leakage.
- A failed item is regenerated with explicit validation feedback up to three times.
  Exhausted failures are written under `data/output/reasoning_traces_realiad_qwen3vl8b/failed/`.
- Compilation rejects missing or unvalidated traces by default; use `--allow-missing`
  only when intentionally creating a partial dataset.

## Environment

Run all commands below from this directory:

```bash
cd scripts/00_generate/realiad_v4
conda create -n realiad-qwen3vl python=3.11 -y
conda activate realiad-qwen3vl
pip install -r requirements-qwen3vl.txt
```

Install a PyTorch build compatible with the machine's CUDA driver before the requirements
file if the default wheel is not suitable. `flash-attn` is optional for this vLLM path.

## Model weights

The default model directory is:

```text
scripts/00_generate/realiad_v4/models/Qwen3-VL-8B-Instruct/
```

Choose either download method from the repository root:

```bash
# Hugging Face
hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir scripts/00_generate/realiad_v4/models/Qwen3-VL-8B-Instruct

# ModelScope mirror
modelscope download --model Qwen/Qwen3-VL-8B-Instruct \
  --local_dir scripts/00_generate/realiad_v4/models/Qwen3-VL-8B-Instruct
```

To store the weights elsewhere, set `QWEN3VL_MODEL_PATH=/absolute/model/path` or pass
`--model /absolute/model/path`.

## End-to-end commands

```bash
cd scripts/00_generate/realiad_v4

# 1. Download Real-IAD to data/Real-IAD/. Public files usually need no token;
#    export HF_TOKEN first if Hugging Face requires authentication in your environment.
python download_realiad.py

# 2. Rebuild deterministic, disjoint 3K SFT + 3K GRPO source indexes.
python sample_realiad_v4.py

# 3. Test strict QC without loading the model.
python -m unittest test_validate_structured_trace.py

# 4. Inspect constructed prompts and images; no vLLM engine is started.
python generate_realiad_traces_qwen3vl.py --debug --max-samples 2

# 5a. Full generation on one GPU.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  python generate_realiad_traces_qwen3vl.py

# 5b. Or data-parallel generation on four GPUs, one 8B replica per GPU.
GPU_LIST=0,1,2,3 bash run_qwen3vl_generate.sh

# 6. Compile only after all 6,000 traces pass validation.
python compile_traces_v4.py
```

Generation is resumable: rerunning the same command skips accepted trace files and retries
items that never produced an accepted trace. Adjust throughput using `--batch-size`; reduce
it first if vLLM reports out-of-memory. The single-GPU default is `16`.

## Active files

- `generate_realiad_traces_qwen3vl.py`: local vLLM inference, sharding, retries, resume.
- `validate_structured_trace.py`: executable XML, length, vocabulary, label/Mask, and
  leakage validation.
- `config_qwen3vl.py`: paths and environment-overridable inference defaults.
- `utils_realiad.py`: Mask overlay, 3x3 location calculation, defect-code mapping.
- `sample_realiad_v4.py`: deterministic SFT/GRPO source sampling.
- `compile_traces_v4.py`: validated-trace-only final assembly.
- `run_qwen3vl_generate.sh`: one data shard and one model replica per selected GPU.
- `requirements-qwen3vl.txt`: local generator dependencies.

## Official references

- [Qwen3-VL-8B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Qwen3-VL official repository and vLLM deployment guidance](https://github.com/QwenLM/Qwen3-VL)
- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [Hugging Face `hf download` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
