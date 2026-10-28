# Stage 0 — Trace generation

How the **AnomalyThink** reasoning-trace corpora are produced. The current Real-IAD
pipeline uses a local Qwen3-VL teacher; other folders retain historical experiments.

## `realiad_v4/` — canonical local Real-IAD pipeline
The authoritative generator for the 3K SFT + 3K GRPO source pools. Flow:
```
download_realiad.py        -> Real-IAD images + masks (HF: Real-IAD/Real-IAD, 30 products)
sample_realiad_v4.py       -> C1-only, mask-filtered, 50/50, stratified, seed 42
generate_realiad_traces_qwen3vl.py <- prompts/inspector_prompt_test_v2.txt
                                    <- local Qwen/Qwen3-VL-8B-Instruct via vLLM
validate_structured_trace.py        -> executable XML/label/type/location/leakage QC
compile_traces_v4.py       -> final ShareGPT-format SFT/GRPO JSONs
```
Helpers: `config_qwen3vl.py` (paths/inference defaults) and `utils_realiad.py`
(overlay + Mask location + `ANOMALY_MAP`). Full setup, model downloads, and commands are in
[`realiad_v4/README.md`](realiad_v4/README.md).

## Real-IAD-**Variety** extension (this folder, `*variety*`)
The 160-product Real-IAD-Variety extension (Variety-STaR). Scripts:
`download_variety.py`, `sample_variety.py`, `generate_variety_traces.py`,
`validate_variety_traces.py`, `assemble_variety_sft_pool.py`, `run_variety_download.sh`,
`run_variety_generate.sh`. Same generator design as v4 (Vertex AI, **`google/gemini-3-flash-preview`**,
`inspector_prompt_test_v2.txt`). Output corpus shipped at [`traces/variety_star_6k/`](../../traces/variety_star_6k/).

## Older / auxiliary (top level)
`generate_realiad_traces.py`, `generate_mmad_traces.py`, `download_mmad.py`, `parallel_realiad_gen.py`,
`trace_loader.py`, `utils_mmad.py` — earlier/auxiliary versions. For the headline Real-IAD corpus,
**`realiad_v4/` supersedes `generate_realiad_traces.py`.**

## Credentials
The local Qwen3-VL generator requires no Gemini or Vertex credentials. Set `HF_TOKEN`
only if your environment requires authentication while downloading Real-IAD or model weights.
