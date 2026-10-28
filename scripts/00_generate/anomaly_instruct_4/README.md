# Anomaly-Instruct four-dataset CoT pipeline

This pipeline adapts IAD-VLMs to the four Anomaly-Instruct-125K sources that have
pixel-level localization supervision:

- MVTec AD
- Anomaly-ShapeNet
- MVTec3D
- Real3D

BMAD and WebAD are intentionally excluded from this first localization-oriented build.
The Prism source directory is treated as read-only. All manifests, generated MVTec3D
bbox prompts, traces, logs, and compiled outputs are written below this directory's
ignored `data/` directory (or `ANOMALY_INSTRUCT_WORK_DIR`).

## Supervision policy

For anomaly trace generation, Qwen3-VL receives the original query view(s), the
corresponding image(s) with a red bbox outline, a deterministic same-category normal
reference, the binary label, product category, and canonical anomaly type. It does
receive the exact ground-truth bbox coordinates as internal text guidance. The model
uses those coordinates as a teacher hint while using the original query, bbox outline,
and normal reference to produce visually grounded inspection reasoning and `<location>`. This
is teacher-data generation rather than a zero-shot localization evaluation.

Source mask paths are retained in the manifest only for provenance, bbox regeneration,
and annotation auditing. Mask images are never included in Qwen3-VL's image inputs.

For normal generation, Qwen3-VL receives the query view(s) and a deterministic
same-category normal reference. The compiled SFT files retain only the original query
view(s); all masks, bbox prompts, references, labels, and GT boxes remain teacher-only.

MVTec AD types come from the defect directory (`path_gt`). The three 3D datasets have
full anomaly-description coverage, but their canonical types are normalized from the
released conversations and are explicitly marked `conversation_derived`.

## Output location format

2D:

```xml
<location>[x1,y1,x2,y2]</location>
```

Multiview 3D (only views whose masks have foreground and whose prompt image has a box):

```xml
<location>[{"view_id":"the_view_id","bbox":[x1,y1,x2,y2]}]</location>
```

Coordinates are inclusive pixel `xyxy` in the original query frame. Following MMR-AD's
CoT verification policy, every predicted anomaly bbox must have IoU strictly greater
than 0.9 with its ground-truth bbox. Small coordinate deviations are therefore allowed.

## Run

From this directory:

```bash
cd scripts/00_generate/anomaly_instruct_4

# 1. Generate pure bbox-outline prompt images for MVTec3D only.
python build_mvtec3d_bboxes.py

# 2. Build four deduplicated manifests.
python prepare_manifests.py

# Before step 1 is complete, manifest logic can be inspected with:
python prepare_manifests.py --allow-missing-mvtec3d-bboxes --max-samples 10

# 3. Run unit tests.
python -m unittest test_pipeline.py
python audit_manifests.py

# 4. Inspect two complete model requests without inference.
python generate_traces_qwen3vl.py --debug --max-samples 2

# 5. Generate all traces, or select/shard datasets.
python generate_traces_qwen3vl.py
python generate_traces_qwen3vl.py --datasets mvtec_ad --shard-id 0 --total-shards 4

# 6. Compile validated query-only SFT data and separate evaluation sidecars.
python compile_traces.py
```

Useful environment overrides:

```bash
export ANOMALY_INSTRUCT_SOURCE_ROOT=/path/to/Anomaly-Instruct-125K
export ANOMALY_INSTRUCT_WORK_DIR=/path/to/writable/workdir
export QWEN3VL_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
```

`prepare_manifests.py` defaults to requiring every bbox prompt. Use
`--allow-missing-mvtec3d-bboxes` only for preflight inspection. Compilation rejects
missing or invalid traces unless `--allow-missing` is explicitly supplied.

## Evidence-first two-stage generation

The baseline command above remains available. The new entry point defaults to
`mvtec_ad anomaly_shapenet mvtec3d real3d` in that order and runs both stages with
one frozen Qwen3-VL/vLLM engine. It reads the existing manifests unchanged.

Stage 1 is type-blind and region-guided: it receives object category, neutral view
IDs, image roles/dimensions, original queries, bbox cue images when available, and
normal references. It receives no sample IDs, paths, binary answer, defect labels,
source descriptions, masks, or textual GT coordinates. Bbox cues still originate
from annotation, so this is not fully GT-blind. Stage 1 records concrete visual
observations and uncertainty rather than a final diagnosis.

Stage 2 rereads the persisted evidence, receives the same visual images plus the
existing GT supervision, and reinspects the images before synthesizing the original
80–220-word structured CoT. Images take priority over evidence; supported evidence
organizes reasoning; GT constrains the answer/type/location without establishing
unseen visual details. Uncertain/not-visible evidence is retained, not filtered.
The original `validate_reasoning()` checks are reused without additions.

```text
data/manifests/<dataset>.json
    -> Stage 1: VisualEvidence structured output
data/evidence_qwen3vl8b/evidence_<dataset>.json
    -> Stage 2: images + evidence + GT -> existing ReasoningTrace + validation
data/traces_qwen3vl8b_evidence_first/traces_<dataset>.json
    -> compile_traces.py --trace-source evidence-first
data/compiled_evidence_first/<dataset>_cot_sft.json
```

Each evidence/trace file contains one dataset. Records are appended to a JSON array;
loading builds an in-memory index by the original sample ID (last record wins).
The ID is added by Python after Stage 1 generation, not sent to Stage 1:

```json
{
  "schema_version": "anomaly-instruct-dataset-records-v2",
  "dataset": "mvtec_ad",
  "records": [
    {
      "image_id": "original_manifest_id",
      "dataset": "mvtec_ad",
      "evidence": {
        "object_observation": "...",
        "regional_observations": [{
          "view_id": "query", "region": "...", "query_evidence": ["..."],
          "normal_comparison": "...", "visibility": "partial", "uncertainty": "..."
        }]
      }
    }
  ]
}
```

Trace records use `reasoning` and `validation` instead of `evidence`, with generation
metadata in both collections. Only the `evidence` payload is included in Stage 2's
evidence text. Visibility is `clear`, `partial`, or `not_visible`; uncertainty may be
null. Regional observations always identify their query `view_id`.

Each accepted inference batch is appended and synced immediately, before retries
or Stage 2. Records are compact JSON, one per physical line inside a standard JSON
array. A normal save serializes only the new batch outside the lock, then replaces
the closing delimiter with that batch and a new delimiter. It does not reread or
rewrite old records. Repeated image IDs append new versions; loading uses the last
one. This reduces normal cumulative write volume from quadratic to linear in the
amount of generated data; reads still scan the dataset once to build its index.

A persistent `.json.lock` file coordinates exclusive writes/recovery and shared
reads. All shards must share an output directory on a filesystem supporting POSIX
file locks. Do not delete lock files while generators are running. Every completed
save is standard JSON. Programmatic readers must use `read_records()` while writers
are active, since an in-place append briefly removes the closing delimiter.

On an interrupted append, the next read or write repairs a missing/partial closing
delimiter or discards an incomplete final record. Complete earlier records are
preserved; a malformed complete line or wrong record identity raises without
modifying the file. Tail recovery is not a guarantee against arbitrary corruption
or power loss, and appends do not have the old full-snapshot atomicity. Legacy v1
dictionary files remain readable and are atomically converted once on their next
write; initialization/conversion use a temporary file that is then removed. No
extra per-sample files or journal/database are introduced. Generation and compile
commands are unchanged.

```bash
cd scripts/00_generate/anomaly_instruct_4

# Default: all four datasets, Stage 1 then Stage 2, automatic resume.
python generate_traces_qwen3vl_evidence_first.py

# Small run on one dataset (eight remaining samples).
python generate_traces_qwen3vl_evidence_first.py --datasets mvtec_ad --max-samples 8

# Select datasets; optionally restrict to normal or anomaly samples.
python generate_traces_qwen3vl_evidence_first.py --datasets anomaly_shapenet mvtec3d real3d
python generate_traces_qwen3vl_evidence_first.py --datasets mvtec_ad --normal-only --max-samples 4
python generate_traces_qwen3vl_evidence_first.py --datasets real3d --anomaly-only --max-samples 4

# Independent stages; cot-only uses existing evidence and never runs Stage 1.
python generate_traces_qwen3vl_evidence_first.py --stage evidence
python generate_traces_qwen3vl_evidence_first.py --stage cot

# Two GPU shards, one model replica per GPU; run in separate terminals.
CUDA_VISIBLE_DEVICES=0 python generate_traces_qwen3vl_evidence_first.py --shard-id 0 --total-shards 2
CUDA_VISIBLE_DEVICES=1 python generate_traces_qwen3vl_evidence_first.py --shard-id 1 --total-shards 2

# Stage 1 request preview without inference. Stage 2 preview needs cached evidence.
python generate_traces_qwen3vl_evidence_first.py --stage evidence --debug --datasets mvtec_ad --max-samples 2
python generate_traces_qwen3vl_evidence_first.py --stage cot --debug --datasets mvtec_ad --max-samples 2

# Optional generation settings; Stage 2 keeps the baseline sampling defaults.
python generate_traces_qwen3vl_evidence_first.py --batch-size 2 --evidence-temperature 0.2 --evidence-max-new-tokens 2048

# Compile query-only SFT files and separate existing evaluation sidecars.
python compile_traces.py --trace-source evidence-first
# Explicitly allow a partial compile after a small run.
python compile_traces.py --trace-source evidence-first --datasets mvtec_ad --allow-missing

# Offline tests, with no model loading or vLLM dependency.
python -m unittest test_pipeline.py test_evidence_first_pipeline.py
```

Use the same compatible inference environment as the baseline
(`../realiad_v4/requirements-qwen3vl.txt`, including vLLM >= 0.11.1 and a
Qwen3-VL-capable Transformers installation). Generation defaults to the model path
in `config.py`; `--model` or `QWEN3VL_MODEL_PATH` overrides it. New output directories
also honor `ANOMALY_INSTRUCT_WORK_DIR`.

Rerun the same command after interruption: valid final traces are skipped, cached
evidence is reused, and only missing/invalid results are generated. A successful
Stage 1 followed by a failed Stage 2 never triggers Stage 1 again. `--stage cot`
reports missing/invalid evidence and exits nonzero rather than generating it.
`--max-samples` limits remaining samples across the selected datasets in this shard,
after filtering/sharding/resume; use identical selections, ordering, and shard counts
for a parallel run. Failures are stored in `traces_qwen3vl8b_evidence_first/failed/`
as per-stage/per-shard JSONL logs; debug requests are aggregated by dataset and shard
under its `debug/` directory. A malformed dataset collection raises an error instead
of being overwritten. No automatic quality scoring, filtering, training, or judge
model is added.
