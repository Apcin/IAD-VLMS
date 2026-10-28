"""Configuration for the four-dataset Anomaly-Instruct CoT pipeline."""

import os
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parents[2]

SOURCE_ROOT = Path(
    os.environ.get(
        "ANOMALY_INSTRUCT_SOURCE_ROOT",
        "/mnt/data/wangzijian/second_stage_work/code/Prism/data/Anomaly-Instruct-125K",
    )
)
DATA_DIR = Path(os.environ.get("ANOMALY_INSTRUCT_WORK_DIR", PIPELINE_ROOT / "data"))
MANIFEST_DIR = DATA_DIR / "manifests"
DERIVED_BBOX_DIR = DATA_DIR / "derived_bbox"
TRACE_DIR = DATA_DIR / "traces_qwen3vl8b"
COMPILED_DIR = DATA_DIR / "compiled"
LOG_DIR = PIPELINE_ROOT / "logs"
PROMPT_PATH = PIPELINE_ROOT / "prompt_anomaly_instruct_4.txt"

MODEL_PATH = os.environ.get(
    "QWEN3VL_MODEL_PATH",
    str(REPO_ROOT / "scripts" / "00_generate" / "realiad_v4" / "models" / "Qwen3-VL-8B-Instruct"),
)

SEED = int(os.environ.get("ANOMALY_INSTRUCT_SEED", "42"))
BATCH_SIZE = int(os.environ.get("QWEN3VL_BATCH_SIZE", "4"))
MAX_RETRIES = int(os.environ.get("QWEN3VL_MAX_RETRIES", "3"))
MAX_MODEL_LEN = int(os.environ.get("QWEN3VL_MAX_MODEL_LEN", "24576"))
MAX_NEW_TOKENS = int(os.environ.get("QWEN3VL_MAX_NEW_TOKENS", "768"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("QWEN3VL_GPU_MEMORY_UTILIZATION", "0.85"))
MIN_PIXELS = int(os.environ.get("QWEN3VL_MIN_PIXELS", str(128 * 28 * 28)))
MAX_PIXELS = int(os.environ.get("QWEN3VL_MAX_PIXELS", str(768 * 768)))
TEMPERATURE = float(os.environ.get("QWEN3VL_TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("QWEN3VL_TOP_P", "0.8"))
TOP_K = int(os.environ.get("QWEN3VL_TOP_K", "20"))
PRESENCE_PENALTY = float(os.environ.get("QWEN3VL_PRESENCE_PENALTY", "1.0"))
REPETITION_PENALTY = float(os.environ.get("QWEN3VL_REPETITION_PENALTY", "1.0"))
MIN_LOCATION_IOU = float(os.environ.get("ANOMALY_INSTRUCT_MIN_LOCATION_IOU", "0.9"))

DATASETS = {
    "mvtec_ad": {
        "directory": "mvtec_anomaly_detection",
        "json": "mvtec_zero_shot.json",
        "target_modality": "2d",
    },
    "anomaly_shapenet": {
        "directory": "anomaly_shapenet",
        "json": "anomaly_shapenet_zero_shot2.json",
        "target_modality": "3d_multiview",
    },
    "mvtec3d": {
        "directory": "mvtec3d",
        "json": "mvtec3d_zero_shot2.json",
        "target_modality": "3d_multiview",
    },
    "real3d": {
        "directory": "real3d",
        "json": "real3d_zero_shot2.json",
        "target_modality": "3d_multiview",
    },
}

for directory in (MANIFEST_DIR, DERIVED_BBOX_DIR, TRACE_DIR, COMPILED_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)
