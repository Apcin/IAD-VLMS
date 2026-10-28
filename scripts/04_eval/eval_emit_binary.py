#!/usr/bin/env python
"""EMIT (InternVL3-8B, difficulty-aware GRPO) on OUR protocol: single-image, ZERO-SHOT, NO RAG,
binary anomaly detection, balanced accuracy on DS-MVTec / VisA.

What this is / isn't:
  - EMIT's paper number (avg 71.44 over 7 MMAD MCQ subtasks) uses a NORMAL reference image (one-shot)
    + a RAG domain-knowledge block. Here we deny it BOTH: single query image only, rag="". This is a
    supported EMIT mode (its own loader has is_one_shot=False), so unlike JUDO it is NOT amputated --
    but it is still a harder setting than EMIT's headline, and must be labelled "single-image, no-RAG".
  - We evaluate ONLY MMAD's "Anomaly Detection" questions (the binary "Is there any defect?" A/B MCQ),
    and score with OUR balanced accuracy (BA = 0.5*(recall + specificity)) so it is comparable to our
    models. The prompt is EMIT's own single-image template with rag="" (verbatim from their script).
  - InternVL3-8B, so OFF-backbone vs our Qwen2.5-VL-7B: report as a cross-architecture reference.

Run:  CUDA_VISIBLE_DEVICES=1 python eval_emit_binary.py --subdataset DS-MVTec --out <path.json>
"""
import sys, os, re, json, argparse
sys.path.insert(0, "/bulk/aacudad/reasoning_traces/emit_eval/swift_stub")   # stub: swift.utils
sys.path.insert(0, "/bulk/aacudad/reasoning_traces/EMIT")                   # EMIT model code
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import AutoTokenizer
from model.internvl_chat import CustomizedInternVLChatModel, InternVLChatConfig

MODEL_DIR = "/bulk/aacudad/reasoning_traces/EMIT-8B"
MMAD_JSON = "/bulk/aacudad/reasoning_traces/MMAD_repo/dataset/MMAD/mmad.json"
IMG_ROOT  = "/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/MMAD"
SYSPROMPT = "/bulk/aacudad/reasoning_traces/EMIT/sysprompt.txt"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# --- standard InternVL preprocessing (inlined from EMIT/model/dataset.py to avoid its video deps) ---
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff, best = float('inf'), (1, 1); area = width * height
    for ratio in target_ratios:
        tar = ratio[0] / ratio[1]; diff = abs(aspect_ratio - tar)
        if diff < best_diff: best_diff, best = diff, ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]: best = ratio
    return best

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    w, h = image.size; ar = w / h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    tar = find_closest_aspect_ratio(ar, target_ratios, w, h, image_size)
    tw, th = image_size * tar[0], image_size * tar[1]; blocks = tar[0] * tar[1]
    resized = image.resize((tw, th)); out = []
    for i in range(blocks):
        box = ((i % (tw // image_size)) * image_size, (i // (tw // image_size)) * image_size,
               ((i % (tw // image_size)) + 1) * image_size, ((i // (tw // image_size)) + 1) * image_size)
        out.append(resized.crop(box))
    if use_thumbnail and len(out) != 1: out.append(image.resize((image_size, image_size)))
    return out

def load_image_tiles(path, input_size, max_num):
    img = Image.open(path).convert('RGB')
    tf = build_transform(input_size)
    tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([tf(t) for t in tiles])   # [K,3,H,W]

# --- EMIT's single-image prompt (verbatim else-branch of evaluate_batch_mmad_choice.py, rag="") ---
INSTRUCT = "Do not include any text in your response."   # think=False (their argparse default)
def make_prompt(question_text):
    rag = ""   # NO RAG
    return ("Query image: <image>\nTo answer a multiple-choice question, please inspect the product "
            "in the given image. There is only one correct option.\n{}\nSelect your answer by "
            "responding with the letter corresponding to the correct option, such as 'A'. {}\n\n"
            "Question:{}").format(rag, INSTRUCT, question_text)

LETTER_RE = re.compile(r'\b([A-E])\b')
def parse_letter(text, options):
    m = LETTER_RE.findall(text)
    if m: return m[0].upper()
    t = (text or "").strip().lower()             # fallback: match answer text
    for k, v in options.items():
        if v.strip().lower().rstrip('.') in t: return k
    return None

def norm_yesno(s):
    s = (s or "").strip().lower().rstrip('.')
    if s in ("yes", "a" ): return "yes"   # only used with resolved option text
    if s in ("no", "b"):   return "no"
    return "yes" if "yes" in s else ("no" if "no" in s else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdataset", required=True, choices=["DS-MVTec", "VisA"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-num", type=int, default=6)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)
    # Config surgery for transformers 5.0 compat:
    #  - flash_attn is not installed and tf5 hard-fails instead of falling back -> force eager attention
    #    on every sub-config (the LLM is built from config.llm_config, so it must be set THERE).
    cfg = InternVLChatConfig.from_pretrained(MODEL_DIR)
    for sub in [cfg, getattr(cfg, "llm_config", None), getattr(cfg, "vision_config", None)]:
        if sub is not None:
            sub._attn_implementation = "eager"
            sub.attn_implementation = "eager"
            if hasattr(sub, "_attn_implementation_autoset"):
                sub._attn_implementation_autoset = False
    # NB: InternViT __init__ did .item() on a linspace (patched); tf5 also meta-inits by default, so
    # force low_cpu_mem_usage=False and load on CPU then move to GPU.
    model = CustomizedInternVLChatModel.from_pretrained(
        MODEL_DIR, config=cfg, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False, attn_implementation="eager").eval().cuda()
    model.system_message = open(SYSPROMPT).read()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    gen = dict(num_beams=1, max_new_tokens=64, min_new_tokens=1, do_sample=False)

    data = json.load(open(MMAD_JSON))
    # collect this sub-dataset's Anomaly-Detection questions
    items = []
    for img_path, v in data.items():
        if img_path.split("/")[0] != a.subdataset: continue
        for c in v["conversation"]:
            if c.get("type") != "Anomaly Detection": continue
            opts = c["Options"]
            qtext = f"{c['Question']} \n" + "".join(f"{k}. {val}\n" for k, val in opts.items())
            items.append({"image": img_path, "options": opts, "gt_letter": c["Answer"], "qtext": qtext})
    print(f"[emit] {a.subdataset}: {len(items)} Anomaly-Detection questions", flush=True)

    results = []; tp = tn = fp = fn = bad = 0
    for s in range(0, len(items), a.batch_size):
        batch = items[s:s + a.batch_size]
        pv_list, npl, prompts = [], [], []
        for it in batch:
            tiles = load_image_tiles(os.path.join(IMG_ROOT, it["image"]), image_size, a.max_num)
            # batch_chat wants num_patches_list[i] to be a LIST of per-image tile counts (it iterates
            # it), so a single-image sample is [K], not K.
            pv_list.append(tiles); npl.append([tiles.size(0)]); prompts.append(make_prompt(it["qtext"]))
        pv = torch.cat(pv_list, 0).to(torch.bfloat16).cuda()
        outs = model.batch_chat(tokenizer=tok, pixel_values=pv, comp_pixel_values=None,
                                questions=prompts, use_comps=[False] * len(batch),
                                num_patches_list=npl, generation_config=gen, verbose=False)
        for it, out in zip(batch, outs):
            pl = parse_letter(out, it["options"])
            gt = norm_yesno(it["options"][it["gt_letter"]])
            pred = norm_yesno(it["options"].get(pl, "")) if pl else None
            if pred is None: bad += 1; pred = "no"   # unparseable -> count as 'no' (conservative), flagged
            if   gt == "yes" and pred == "yes": tp += 1
            elif gt == "yes" and pred == "no":  fn += 1
            elif gt == "no"  and pred == "no":  tn += 1
            elif gt == "no"  and pred == "yes": fp += 1
            results.append({"image": it["image"], "gt": gt, "pred": pred, "pred_letter": pl, "raw": out[:200]})
        done = s + len(batch)
        if done % 80 < a.batch_size:
            rec = tp / (tp + fn) if (tp + fn) else 0; spc = tn / (tn + fp) if (tn + fp) else 0
            print(f"  [{done}/{len(items)}] BA={0.5*(rec+spc)*100:.2f} recall={rec*100:.1f}% spec={spc*100:.1f}% bad={bad}", flush=True)

    rec = tp / (tp + fn) if (tp + fn) else 0; spc = tn / (tn + fp) if (tn + fp) else 0
    ba = 0.5 * (rec + spc)
    summary = {"model": "EMIT-8B (InternVL3, single-image zero-shot, no-RAG)", "subdataset": a.subdataset,
               "prompt_mode": "emit_singleimg_norag_binary", "metrics": {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
               "n": len(items), "unparseable": bad, "BA": round(ba * 100, 2),
               "recall": round(rec * 100, 2), "specificity": round(spc * 100, 2)}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({**summary, "results": results}, open(a.out, "w"), indent=1)
    print(f"\n*** EMIT {a.subdataset} (single-image, no-RAG): BA={ba*100:.2f} "
          f"recall={rec*100:.1f}% spec={spc*100:.1f}% tp={tp} tn={tn} fp={fp} fn={fn} bad={bad} ***", flush=True)

if __name__ == "__main__":
    main()
