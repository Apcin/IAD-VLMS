"""
Configuration for MMAD reasoning trace generation
"""
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MMAD_DIR = DATA_DIR / "MMAD"
OUTPUT_DIR = DATA_DIR / "output"
REASONING_TRACES_DIR = OUTPUT_DIR / "reasoning_traces"
LOGS_DIR = PROJECT_ROOT / "logs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Create directories
for dir_path in [REASONING_TRACES_DIR, LOGS_DIR, PROMPTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Configuration
GEMINI_API_KEYS_STR = os.getenv('GEMINI_API_KEYS', '')
GEMINI_API_KEYS = [key.strip() for key in GEMINI_API_KEYS_STR.split(',') if key.strip()]

if not GEMINI_API_KEYS:
    raise ValueError("No Gemini API keys found in .env file!")

# Model Configuration
GEMINI_MODELS_STR = os.getenv("GEMINI_MODELS", "gemini-2.5-flash")
GEMINI_MODELS = [m.strip() for m in GEMINI_MODELS_STR.split(",") if m.strip()]
GEMINI_MODEL = GEMINI_MODELS[0]  # Default to first model


# Batch Processing
BATCH_SIZE = 10  # Process 10 images at once
MAX_RETRIES = 3
RATE_LIMIT_WAIT = 4  # seconds

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / "mmad_generation.log"