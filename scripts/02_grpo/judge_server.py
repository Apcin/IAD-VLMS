"""
Reasoning Judge HTTP Server
============================
Loads Qwen2.5-VL-7B-Instruct on a single GPU and serves scoring requests.
Run with: CUDA_VISIBLE_DEVICES=3 python judge_server.py

Listens on localhost:5100.
POST /score  — JSON body with: think_text, gt_label, gt_type, gt_location, product, model_answer
POST /batch  — JSON body with: items (list of the above dicts)
GET  /health — health check
"""

import json
import os
import re
import sys
import logging
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ.setdefault("HF_HOME", "/bulk/aacudad/reasoning_traces/hf_cache")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [JUDGE] %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("JUDGE_PORT", "5100"))
MODEL_ID = os.environ.get("JUDGE_MODEL_ID", "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit")

JUDGE_SYSTEM = """You are an expert evaluator of industrial anomaly detection reasoning. You will be given a model's reasoning trace (from a <think> block) along with ground truth information about the sample. Your job is to evaluate the QUALITY of the reasoning, not just whether the final answer is correct.

Score the reasoning on these criteria:
1. **Observation Quality** (0-2): Does the reasoning describe specific visual features (texture, color, shape, surface details)? Generic/vague = 0, some details = 1, rich specific observations = 2.
2. **Logical Flow** (0-2): Does the reasoning follow a logical inspection process? Random/incoherent = 0, somewhat structured = 1, systematic inspection = 2.
3. **Consistency** (0-2): Is the reasoning consistent with the final conclusion? Contradicts itself = 0, partially consistent = 1, fully consistent = 2.
4. **Specificity** (0-2): For anomalies — does it identify the defect type and location with detail? For normals — does it explain why the item appears defect-free? Vague = 0, some specificity = 1, precise = 2.

Return ONLY a JSON object with this exact format:
{"observation": <0-2>, "logic": <0-2>, "consistency": <0-2>, "specificity": <0-2>, "total": <0-8>, "normalized": <0.0-1.0>}"""

# Globals
model = None
processor = None


def load_model():
    global model, processor
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    device_map = os.environ.get("JUDGE_DEVICE_MAP", "cuda:0")
    
    logger.info(f"Loading {MODEL_ID} with device_map={device_map}...")
    
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": device_map}
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    # Load processor from base model (pre-quantized repos may not have processor files)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model.eval()
    
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    logger.info(f"Model loaded successfully, GPU memory: {mem_gb:.2f} GB")


def score_one(item):
    """Score a single reasoning trace. Returns 0.0-1.0."""
    think_text = item.get("think_text", "")
    gt_label = item.get("gt_label", "no")
    gt_type = item.get("gt_type", "")
    gt_location = item.get("gt_location", "")
    product = item.get("product", "unknown")
    model_answer = item.get("model_answer", "")

    if not think_text or len(think_text.strip()) < 20:
        return 0.0

    parts = [f"Product: {product}", f"Ground truth: {'anomalous' if gt_label == 'yes' else 'normal'}"]
    if gt_label == "yes":
        if gt_type:
            parts.append(f"Ground truth defect type: {gt_type}")
        if gt_location:
            parts.append(f"Ground truth location: {gt_location}")
    parts.append(f"Model's final answer: {model_answer}")
    parts.append(f"\nModel's reasoning:\n{think_text}")
    user_text = "\n".join(parts)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": JUDGE_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Use first device of model (may be split across GPUs)
    first_device = next(model.parameters()).device
    inputs = processor(text=[text], images=None, padding=True, return_tensors="pt").to(first_device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    response = processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            score_data = json.loads(json_match.group())
            return float(score_data.get("normalized", 0.0))
    except (json.JSONDecodeError, ValueError):
        pass
    return 0.5  # neutral fallback


class JudgeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except Exception as e:
            self._respond(400, {"error": str(e)})
            return

        if self.path == "/score":
            score = score_one(body)
            self._respond(200, {"score": score})
        elif self.path == "/batch":
            items = body.get("items", [])
            scores = [score_one(item) for item in items]
            self._respond(200, {"scores": scores})
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        # Suppress per-request logs, use our logger instead
        pass


if __name__ == "__main__":
    load_model()
    server = HTTPServer(("127.0.0.1", PORT), JudgeHandler)
    logger.info(f"Judge server listening on http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.server_close()
