"""Configuration for local Qwen3-VL Real-IAD trace generation."""

import os
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parents[2]

DATA_DIR = Path(os.environ.get("REALIAD_DATA_DIR", PIPELINE_ROOT / "data"))
OUTPUT_DIR = DATA_DIR / "output"
LOGS_DIR = PIPELINE_ROOT / "logs"
PROMPTS_DIR = REPO_ROOT / "prompts"

MODEL_PATH = os.environ.get(
    "QWEN3VL_MODEL_PATH",
    str(PIPELINE_ROOT / "models" / "Qwen3-VL-8B-Instruct"),
)

BATCH_SIZE = int(os.environ.get("QWEN3VL_BATCH_SIZE", "16"))
MAX_RETRIES = int(os.environ.get("QWEN3VL_MAX_RETRIES", "3"))
MAX_MODEL_LEN = int(os.environ.get("QWEN3VL_MAX_MODEL_LEN", "12288"))
MAX_NEW_TOKENS = int(os.environ.get("QWEN3VL_MAX_NEW_TOKENS", "768"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("QWEN3VL_GPU_MEMORY_UTILIZATION", "0.85"))
MIN_PIXELS = int(os.environ.get("QWEN3VL_MIN_PIXELS", str(256 * 28 * 28)))
MAX_PIXELS = int(os.environ.get("QWEN3VL_MAX_PIXELS", str(1024 * 1024)))
TEMPERATURE = float(os.environ.get("QWEN3VL_TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("QWEN3VL_TOP_P", "0.8"))
TOP_K = int(os.environ.get("QWEN3VL_TOP_K", "20"))
PRESENCE_PENALTY = float(os.environ.get("QWEN3VL_PRESENCE_PENALTY", "1.5"))
REPETITION_PENALTY = float(os.environ.get("QWEN3VL_REPETITION_PENALTY", "1.0"))
SEED = int(os.environ.get("QWEN3VL_SEED", "42"))

for directory in (OUTPUT_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
