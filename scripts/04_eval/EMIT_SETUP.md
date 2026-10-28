# Running the EMIT baseline (InternVL3-8B) — setup notes

EMIT is InternVL3-8B with custom modeling code (`CustomizedInternVLChatModel`) and does **not** run in
our Qwen `llama_sft` env (transformers 5.0). It needs a dedicated env matching EMIT's own stack.

## 1. Weights + repo
- Weights: `huggingface.co/gw49/EMIT-8B` (~15 GB, 4 shards) -> `/bulk/aacudad/reasoning_traces/EMIT-8B`
- Code:    `github.com/guanwei49/EMIT` -> `/bulk/aacudad/reasoning_traces/EMIT`

## 2. Dedicated conda env (on /bulk, NOT home)
```bash
export CONDA_PKGS_DIRS=/bulk/aacudad/conda_pkgs PIP_CACHE_DIR=/bulk/aacudad/.pip_cache TMPDIR=/bulk/aacudad/tmp
conda create -p /bulk/aacudad/conda_envs/emit python=3.10 -y
PIP=/bulk/aacudad/conda_envs/emit/bin/pip
$PIP install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
$PIP install "transformers==4.50.0" "tokenizers==0.21.0" "accelerate==1.2.1" "timm==1.0.15" \
             peft kornia opencv-python-headless imageio sentencepiece einops pillow numpy
```
transformers **4.50.0** is EMIT's own version and is required: transformers 5.0 breaks the load
(`all_tied_weights_keys` API change, and meta-init incompatibility).

## 3. Two code changes needed to load under this env
1. **swift stub** — the modeling file does `from swift.utils import get_env_args, is_deepspeed_enabled`.
   We do not install full ms-swift; a 2-function stub is enough (shipped here: `emit_swift_stub/swift/`).
   The eval script adds it to `sys.path`.
2. **meta-init patch** — `EMIT/model/internvl_chat/modeling_intern_vit.py` line ~314 does
   `dpr = [x.item() for x in torch.linspace(...)]`, which fails under meta-device init. Replace with the
   pure-Python equivalent:
   ```python
   _n = config.num_hidden_layers
   dpr = [config.drop_path_rate * _i / (_n - 1) for _i in range(_n)] if _n > 1 else [0.0]
   ```

Also: flash_attn is **not** installed -> the eval forces `attn_implementation="eager"` on every
sub-config, and loads with `low_cpu_mem_usage=False`.

## 4. Run
```bash
CUDA_VISIBLE_DEVICES=1 /bulk/aacudad/conda_envs/emit/bin/python \
    scripts/04_eval/eval_emit_binary.py --subdataset DS-MVTec \
    --out /bulk/aacudad/reasoning_traces/outputs/emit_eval/emit_dsmvtec_binary.json
# then --subdataset VisA
```
The script: single image, `rag=""` (no RAG), MMAD "Anomaly Detection" questions only, maps the model's
A/B letter to yes/no via the option text, and writes our balanced accuracy. See
[`results/emit_eval/NOTE.md`](../../results/emit_eval/NOTE.md) for the numbers and the "what this is /
isn't" caveats. Not in the thesis yet.
