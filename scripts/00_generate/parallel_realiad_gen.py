"""
Launcher script to run concurrent instances of generate_realiad_traces.py
in separate shells (Windows).
"""
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load keys
script_dir = Path(__file__).parent
load_dotenv(script_dir / ".env")
keys_str = os.getenv("GEMINI_API_KEYS", "")
all_keys = [k.strip() for k in keys_str.split(",") if k.strip()]

TOTAL_WORKERS = 1
KEYS_PER_WORKER = 16

if len(all_keys) < TOTAL_WORKERS * KEYS_PER_WORKER:
    print(f"Warning: You have {len(all_keys)} keys, but requested {TOTAL_WORKERS} workers with {KEYS_PER_WORKER} keys each.")
    print("Some workers might share keys or have fewer keys.")

# Distribute keys
worker_keys = []
for i in range(TOTAL_WORKERS):
    start = i * KEYS_PER_WORKER
    end = start + KEYS_PER_WORKER
    # Handle wrap around or just slice? 
    # Let's just slice safely
    subset = all_keys[start:end]
    if not subset:
        # Reuse keys if we run out? Or cycle?
        subset = all_keys[i % len(all_keys) : (i % len(all_keys)) + 1]
        
    worker_keys.append(",".join(subset))

print(f"Launching {TOTAL_WORKERS} workers...")

for i in range(TOTAL_WORKERS):
    keys = worker_keys[i]
    cmd = [
        "python",
        "reasoning_traces_gen/generate_realiad_traces.py",
        "--shard_id", str(i),
        "--total_shards", str(TOTAL_WORKERS),
        "--keys", keys
    ]
    
    # On Windows, use 'start' to open new cmd window
    # properly quoted command for start
    cmd_str = " ".join(cmd)
    print(f"Worker {i}: keys={[k[:5] + '...' for k in keys.split(',')]}")
    
    # subprocess.Popen with creationflags for new console
    if sys.platform == "win32":
        subprocess.Popen(
            f'start "Worker {i}" cmd /k {cmd_str}', 
            shell=True
        )
    else:
        # Linux/Mac (optional support)
        subprocess.Popen(cmd)

print("All workers launched.")



