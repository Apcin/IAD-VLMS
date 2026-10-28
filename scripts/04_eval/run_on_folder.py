import os
"""Run a Qwen2.5-VL checkpoint over a folder of (unlabeled) images and dump verdict+reasoning per image.
Replicates the thesis eval prompt/decoding exactly (system 'Please answer by yes or no' + train-prompt,
greedy, max_new_tokens=1024, image_max_pixels=262144). Pin to a GPU via CUDA_VISIBLE_DEVICES."""
import argparse, json, os, re, time
os.environ.setdefault("HF_HOME", (os.environ.get('WORK_DIR','/bulk/aacudad/reasoning_traces')+"/hf_cache"))
from PIL import Image
import torch
from transformers import AutoProcessor

SYS_PROMPT = "Please answer by yes or no"


def make_train_prompt(pn):
    return (f"Analyze the provided image of the {pn}. Determine if there are any anomalies present. "
            "If an anomaly is detected, specify its type and location, and provide a detailed reasoning "
            "for your conclusion.")


def extract_tags(text):
    r = {}
    for t in ("think", "reasoning"):
        m = re.search(f"<{t}>(.*?)</{t}>", text, re.I | re.S)
        if m:
            r["reasoning"] = m.group(1).strip(); break
    for t in ("type", "location", "answer"):
        m = re.search(f"<{t}>(.*?)</{t}>", text, re.I | re.S)
        if m:
            v = m.group(1).strip().lower()
            if t == "answer":
                v = "yes" if v in ("a", "yes", "y", "true") else "no" if v in ("b", "no", "n", "false") else v
            r[t] = v
    return r


def fuzzy_yes_no(t):
    t = t.strip().lower()
    if t.startswith("yes"): return "yes"
    if t.startswith("no"): return "no"
    if "yes" in t and "no" not in t: return "yes"
    if "no" in t and "yes" not in t: return "no"
    return ""


def load_model(checkpoint, base_model):
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    is_peft = checkpoint and os.path.exists(os.path.join(checkpoint, "adapter_config.json"))
    def _load(path):
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(path, **kw)
        except Exception:
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(path, **kw)
    if is_peft:
        from peft import PeftModel
        model = PeftModel.from_pretrained(_load(base_model), checkpoint).merge_and_unload()
    else:
        model = _load(checkpoint)
    proc = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"
    proc.image_processor.min_pixels = 256 * 28 * 28
    proc.image_processor.max_pixels = 262144
    model.eval()
    return model, proc


def run_batch(model, proc, batch):
    texts, imgs = [], []
    for img, pn in batch:
        msg = [{"role": "system", "content": SYS_PROMPT},
               {"role": "user", "content": [{"type": "image", "image": img},
                                            {"type": "text", "text": f"\n{make_train_prompt(pn)}"}]}]
        texts.append(proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)); imgs.append(img)
    inp = proc(text=texts, images=imgs, return_tensors="pt", padding=True)
    inp = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inp.items()}
    il = inp["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=1024, do_sample=False, temperature=None, top_p=None)
    return [proc.decode(out[i][il:], skip_special_tokens=True) for i in range(len(batch))]


def code_of(comp):
    m = re.match(r"([CR]\d+)", comp)
    return m.group(1) if m else ""


def product_name(comp):
    c = code_of(comp)
    if c.startswith("C"): return f"{c} capacitor"
    if c.startswith("R"): return f"{c} resistor"
    return "electronic component"


def folder_label(lbl):
    l = lbl.lower()
    if "defect" in l: return "defect"
    if any(k in l for k in ("good", "normal", "100 percent", "all good")): return "good"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--system-prompt-file", default="", help="read system prompt text from this file (default: 'Please answer by yes or no')")
    ap.add_argument("--folder-code-prefix", default="", help="keep only files whose top-level component-folder code starts with this (e.g. 'R' for resistors)")
    a = ap.parse_args()

    global SYS_PROMPT
    if a.system_prompt_file:
        SYS_PROMPT = open(a.system_prompt_file).read().strip("\n")
        print(f"[{time.strftime('%H:%M:%S')}] custom system prompt ({len(SYS_PROMPT)} chars) from {a.system_prompt_file}", flush=True)

    exts = (".tiff", ".tif", ".bmp", ".jpg", ".jpeg", ".png")
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(a.image_dir) for f in fs if f.lower().endswith(exts))
    if a.folder_code_prefix:
        def _fc(fp):
            c = os.path.relpath(fp, a.image_dir).split(os.sep)[0]
            m = re.match(r"([CR]\d+)", c)
            return m.group(1) if m else ""
        files = [f for f in files if _fc(f).startswith(a.folder_code_prefix)]
        print(f"[{time.strftime('%H:%M:%S')}] folder-code-prefix '{a.folder_code_prefix}' -> {len(files)} files", flush=True)
    if a.max_images:
        files = files[:a.max_images]
    print(f"[{time.strftime('%H:%M:%S')}] {len(files)} images | ckpt={a.checkpoint}", flush=True)
    model, proc = load_model(a.checkpoint, a.base_model)
    print(f"[{time.strftime('%H:%M:%S')}] model loaded on {model.device}", flush=True)

    out = open(a.output, "w")
    t0 = time.time(); done = 0; batch = []; meta = []

    def flush():
        nonlocal done
        if not batch:
            return
        try:
            raws = run_batch(model, proc, batch)
        except Exception as e:
            raws = [f"<<ERROR {e}>>"] * len(batch)
        for m, raw in zip(meta, raws):
            tg = extract_tags(raw)
            rec = dict(path=m["path"], component=m["comp"], folder_label=m["label"],
                       verdict=tg.get("answer") or fuzzy_yes_no(raw), pred_type=tg.get("type", ""),
                       pred_location=tg.get("location", ""), reasoning=tg.get("reasoning", "")[:1200])
            out.write(json.dumps(rec) + "\n")
        out.flush(); done += len(batch)
        print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(files)} ({done/max(1e-9,time.time()-t0)*60:.0f}/min)", flush=True)
        batch.clear(); meta.clear()

    for fp in files:
        parts = os.path.relpath(fp, a.image_dir).split(os.sep)
        comp = parts[0] if parts else ""
        lbl = folder_label(parts[1]) if len(parts) > 1 else "unknown"
        try:
            img = Image.open(fp).convert("RGB")
        except Exception as e:
            out.write(json.dumps(dict(path=fp, component=comp, folder_label=lbl, verdict="", error=str(e))) + "\n")
            continue
        batch.append((img, product_name(comp))); meta.append(dict(path=fp, comp=comp, label=lbl))
        if len(batch) >= a.batch_size:
            flush()
    flush()
    out.close()
    print(f"[{time.strftime('%H:%M:%S')}] DONE {done} -> {a.output}", flush=True)


if __name__ == "__main__":
    main()
