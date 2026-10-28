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
