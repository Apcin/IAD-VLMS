#!/usr/bin/env python
"""Build self-contained per-model TRACE VIEWERS from the shipped eval JSONs.

For each of the reported/released models, render a single self-contained HTML that shows, per
sample: the product image, the red GT-mask overlay (anomalies), the ground-truth type, and the
model's OWN generated reasoning trace + verdict (correct/incorrect). Images are JPEG thumbnails
base64-embedded so each file opens anywhere with no external assets.

A FIXED, product-diverse sample of image_ids is used across ALL models (same images everywhere),
so a reader can look up the same part and compare reasoning quality across checkpoints.

Output: repository_tu_delft_vlms/docs/trace_viewers/<label>.html  (+ index.html)
Run:    python build_trace_viewer.py [--per-bench 200 --anom-frac 0.70 --maxpx 384 --quality 70]
Data/overlay logic mirrors scripts/explainability_judge.py.
"""
import io, re, json, argparse, base64, collections, random, html
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path("/bulk/aacudad/reasoning_traces")
RES  = ROOT / "repository_tu_delft_vlms/results"
OUTDIR = ROOT / "repository_tu_delft_vlms/docs/trace_viewers"
MMAD_JSON = ROOT / "MMAD_repo/dataset/MMAD/mmad.json"
MMAD_IMG  = ROOT / "reasoning_traces_gen/data/MMAD"

MODELS = [  # (label, display, ds_json, visa_json|None)
 ("base",    "Qwen2.5-VL-7B - zero-shot base (69.01 / 53.79)",
   "qwen25vl_baseline_eval/eval_dsmvtec_full_trainprompt.json", None),
 ("sft6k",   "SFT-6K ckpt-564 (80.16 / 64.78)",
   "sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json",
   "sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_visa_full_trainprompt.json"),
 ("armC",    "Arm-C SFT ckpt-376 - THESIS HEADLINE (82.80 / 72.07)",
   "sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json",
   "sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_visa_full_trainprompt.json"),
 ("sft_grpo","SFT+GRPO run2 ckpt-530 (82.73 / 70.39)",
   "grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json",
   "grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_visa_full_trainprompt.json"),
 ("grpo954", "GRPO-on-Arm-C ckpt-954 - research preview (82.95 / 72.62)",
   "grpo_sftprompt_kl0.1_sys_3ep/checkpoint-954/eval_dsmvtec_full_trainprompt.json",
   "grpo_sftprompt_kl0.1_sys_3ep/checkpoint-954/eval_visa_full_trainprompt.json"),
 ("iadr1",   "IAD-R1 - baseline, native prompt (81.92 / 71.34)",
   "iad_r1_qwen_recanon/eval_dsmvtec_full_iadr1native.json",
   "iad_r1_qwen_recanon/eval_visa_full_iadr1native.json"),
]

# ---- mask / overlay / type helpers (from explainability_judge.py) ----
def gt_type_from_mmad(entry, fallback):
    for q in entry.get("conversation", []):
        if "type of the defect" in q.get("Question", "").lower():
            ans = q.get("Answer", ""); opt = q.get("Options", {})
            v = opt.get(ans, ans) if isinstance(opt, dict) else ans
            v = str(v).strip().rstrip(".").strip()
            if v and v.lower() not in ("good", "none", "no defect"):
                return v
    return fallback.replace("_", " ")

def mmad_key(abspath):
    s = str(abspath); i = s.find("/MMAD/")
    return s[i + 6:] if i >= 0 else None

def _fg_mask(mask_path, size=None):
    im = Image.open(mask_path).convert("RGB")
    if size and im.size != size:
        im = im.resize(size, Image.NEAREST)
    return (np.array(im).sum(2) > 30)

def make_overlay(img, mask_path):
    base = img.convert("RGBA"); fg = _fg_mask(mask_path, base.size)
    red = np.zeros((base.size[1], base.size[0], 4), dtype=np.uint8); red[fg] = (255, 0, 0, 120)
    return Image.alpha_composite(base, Image.fromarray(red, "RGBA")).convert("RGB")

def thumb(im, maxpx, q):
    im = im.convert("RGB"); w, h = im.size; s = min(1.0, maxpx / max(w, h))
    if s < 1.0:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    bio = io.BytesIO(); im.save(bio, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(bio.getvalue()).decode()

def rows_of(p):
    fp = RES / p
    return json.load(open(fp))["results"] if fp.exists() else None

# ---- deterministic product-diverse sampling ----
def diverse(ids, k, meta, rng):
    byp = collections.defaultdict(list)
    for i in ids: byp[meta[i]["product"]].append(i)
    for v in byp.values(): rng.shuffle(v)
    order = sorted(byp, key=lambda p: -len(byp[p])); out = []
    while len(out) < k and any(byp.values()):
        for p in order:
            if byp[p]: out.append(byp[p].pop())
            if len(out) >= k: break
    return out

def build_meta(ref_rows, mmad):
    """image_id -> {product, gt, abspath, mask, gtype, renderable} for the whole benchmark."""
    meta = {}
    for r in ref_rows:
        iid = r["image_id"]; ap = r.get("absolute_path", ""); key = mmad_key(ap)
        gt = r.get("gt_answer"); prod = r.get("product", "?")
        img_ok = bool(ap) and Path(ap).exists()
        mask = None; gtype = ""
        entry = mmad.get(key) if key else None
        if entry:
            base = MMAD_IMG / key.split("/")[0] / key.split("/")[1]
            mpath = entry.get("mask_path") or ""
            if mpath and (base / mpath).exists(): mask = base / mpath
            gtype = gt_type_from_mmad(entry, key.split("/")[3] if key and len(key.split("/")) > 3 else "")
        # include any sample whose image exists; anomalies without a mask just show the plain image
        renderable = img_ok
        meta[iid] = {"product": prod, "gt": gt, "abspath": ap, "mask": mask,
                     "gtype": gtype, "renderable": renderable}
    return meta

CARD = """<div class="card {cls}" data-v="{verdict}" data-a="{anom}">
  <div class="hd"><span class="prod">{product}</span><span class="badge {cls}">{verdict_label}</span></div>
  {imgs}
  <div class="gt">GT: <b>{gt}</b>{gtype}</div>
  <div class="pred">model: <b>{pred}</b> &nbsp; <span class="tags">{tags}</span></div>
  <details><summary>reasoning trace</summary><pre>{trace}</pre></details>
</div>"""

PAGE = """<meta charset="utf-8"><title>Trace viewer - {display}</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}}
 header{{position:sticky;top:0;background:#161a20;padding:12px 16px;border-bottom:1px solid #2a2f37;z-index:5}}
 h1{{font-size:16px;margin:0 0 6px}} .sub{{color:#9aa4b2;font-size:12px}}
 .bar{{margin-top:8px}} .bar button{{background:#222836;color:#cbd5e1;border:1px solid #333c4a;border-radius:6px;padding:4px 10px;margin-right:6px;cursor:pointer;font-size:12px}}
 .bar button.on{{background:#2d6cdf;color:#fff;border-color:#2d6cdf}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px;padding:14px}}
 .card{{background:#161a20;border:1px solid #262c35;border-radius:10px;padding:10px;overflow:hidden}}
 .hd{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}}
 .prod{{font-size:12px;color:#9aa4b2;text-transform:uppercase;letter-spacing:.04em}}
 .imgs{{display:flex;gap:6px}} .imgs img{{width:100%;max-width:50%;border-radius:6px;background:#000}}
 .imgs.single img{{max-width:60%}}
 .badge{{font-size:11px;padding:2px 8px;border-radius:20px}}
 .ok .badge,.badge.ok{{background:#13361f;color:#5fd38a}} .bad .badge,.badge.bad{{background:#3a1620;color:#ff8098}}
 .card.ok{{border-color:#1f5133}} .card.bad{{border-color:#5a2130}}
 .gt,.pred{{font-size:12px;margin-top:6px;color:#c7ced8}} .tags{{color:#8b94a2}}
 details{{margin-top:6px}} summary{{cursor:pointer;font-size:12px;color:#7fb1ff}}
 pre{{white-space:pre-wrap;word-break:break-word;font-size:11.5px;line-height:1.35;background:#0c0e12;border:1px solid #222;border-radius:6px;padding:8px;max-height:320px;overflow:auto;color:#d6dce4}}
</style>
<header>
 <h1>{display}</h1>
 <div class="sub">{n} samples ({na} anomalies with GT-mask overlay, {nn} normal) - shared fixed set across all model viewers - the reasoning shown is THIS model's own generated trace. Overlay = true defect region in red.</div>
 <div class="bar">
  <button class="on" data-f="all">all</button><button data-f="ok">correct</button><button data-f="bad">incorrect</button>
  <button data-f="anom">anomalies</button><button data-f="norm">normal</button>
 </div>
</header>
<div class="grid">{cards}</div>
<script>
 const btns=[...document.querySelectorAll('.bar button')], cards=[...document.querySelectorAll('.card')];
 btns.forEach(b=>b.onclick=()=>{{btns.forEach(x=>x.classList.remove('on'));b.classList.add('on');const f=b.dataset.f;
  cards.forEach(c=>{{let s=true;
   if(f==='ok')s=c.dataset.v==='ok'; else if(f==='bad')s=c.dataset.v==='bad';
   else if(f==='anom')s=c.dataset.a==='1'; else if(f==='norm')s=c.dataset.a==='0';
   c.style.display=s?'':'none';}});}});
</script>"""

def esc(s): return html.escape(str(s or ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include EVERY sample (one file per model per benchmark)")
    ap.add_argument("--per-bench", type=int, default=200)
    ap.add_argument("--anom-frac", type=float, default=0.70)
    ap.add_argument("--maxpx", type=int, default=288)
    ap.add_argument("--quality", type=int, default=60)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    mmad = json.load(open(MMAD_JSON))
    rng = random.Random(a.seed)

    # per-benchmark: universe = image_ids present in ALL models that have that benchmark
    benches = [("DS-MVTec", 2), ("VisA", 3)]  # (name, index into MODELS tuple for json path)
    bench_ids = {}  # bench -> ordered list of image_ids to render
    imgcache = {}   # bench -> {iid: {orig, ov, gtype, product, gt}} (built ONCE, reused across models)
    for bench, idx in benches:
        model_rows = {}
        for m in MODELS:
            if not m[idx]: continue
            r = rows_of(m[idx])
            if r is not None: model_rows[m[0]] = {x["image_id"]: x for x in r}
        if not model_rows: continue
        universe = set.intersection(*[set(v) for v in model_rows.values()])
        ref = next(iter(model_rows.values()))
        meta = build_meta([ref[i] for i in universe], mmad)
        rend = [i for i in universe if meta[i]["renderable"]]
        if a.all:
            ids = sorted(rend, key=lambda i: (meta[i]["product"], i))   # every sample, grouped by product
        else:
            anom = [i for i in rend if meta[i]["gt"] == "yes"]; norm = [i for i in rend if meta[i]["gt"] == "no"]
            na = min(len(anom), int(round(a.per_bench * a.anom_frac))); nn = min(len(norm), a.per_bench - na)
            ids = diverse(anom, na, meta, rng) + diverse(norm, nn, meta, rng)
        cache = {}
        for iid in ids:
            md = meta[iid]; im = Image.open(md["abspath"])
            entry = {"orig": thumb(im, a.maxpx, a.quality), "ov": None,
                     "gtype": md["gtype"], "product": md["product"], "gt": md["gt"]}
            if md["gt"] == "yes" and md["mask"] is not None:
                try: entry["ov"] = thumb(make_overlay(im, md["mask"]), a.maxpx, a.quality)
                except Exception: pass
            cache[iid] = entry
        bench_ids[bench] = ids; imgcache[bench] = cache
        na = sum(meta[i]["gt"] == "yes" for i in ids)
        print(f"[{bench}] {len(ids)} samples ({na} anom / {len(ids)-na} normal) over {len({meta[i]['product'] for i in ids})} products")

    # one HTML per model PER benchmark
    SLUG = {"DS-MVTec": "dsmvtec", "VisA": "visa"}
    index = collections.OrderedDict()  # display -> [(bench, filename, n, mb), ...]
    for m in MODELS:
        label, display = m[0], m[1]
        for bench, idx in benches:
            if not m[idx] or bench not in bench_ids: continue
            rows = {x["image_id"]: x for x in rows_of(m[idx])}
            cards = []; na_tot = nn_tot = 0
            for iid in bench_ids[bench]:
                if iid not in rows: continue
                r = rows[iid]; c = imgcache[bench][iid]
                pred = r.get("pred_answer", "?"); gtv = c["gt"]
                ok = (pred == gtv); verdict = "ok" if ok else "bad"
                anom = "1" if gtv == "yes" else "0"
                img_tags = f'<img src="{c["orig"]}" loading="lazy">'
                single = "single"
                if c["ov"]:
                    img_tags += f'<img src="{c["ov"]}" loading="lazy">'; single = ""
                imgs_div = f'<div class="imgs {single}">{img_tags}</div>'
                tags = r.get("pred_tags") or {}
                # only compact structured tags here; 'reasoning' is the full trace (details section)
                # and 'answer' is already the verdict above -> both excluded
                tagstr = " ".join(f"{k}={esc(tags[k])}" for k in ("type", "location")
                                  if isinstance(tags, dict) and tags.get(k))
                cards.append(CARD.format(
                    cls=verdict, verdict=verdict, anom=anom,
                    verdict_label=("correct" if ok else "incorrect"),
                    product=esc(f"{bench} / {c['product']}"),
                    imgs=imgs_div,
                    gt=("anomaly" if gtv == "yes" else "normal"),
                    gtype=(f" ({esc(c['gtype'])})" if gtv == "yes" and c["gtype"] else ""),
                    pred=("anomaly" if pred == "yes" else "normal" if pred == "no" else esc(pred)),
                    tags=tagstr, trace=esc(r.get("pred_full", ""))))
                na_tot += (gtv == "yes"); nn_tot += (gtv == "no")
            n = len(cards); fn = f"{label}_{SLUG[bench]}.html"
            (OUTDIR / fn).write_text(PAGE.format(display=esc(f"{display}  -  {bench}"),
                                                 n=n, na=na_tot, nn=nn_tot, cards="\n".join(cards)))
            mb = (OUTDIR / fn).stat().st_size / 1048576
            print(f"  wrote {fn}  ({n} cards, {mb:.1f} MB)")
            index.setdefault(display, []).append((bench, fn, n, mb))

    parts = ["<meta charset='utf-8'><title>AnomalyThink trace viewers</title>",
        "<style>body{font-family:system-ui,Arial;margin:24px;max-width:900px;line-height:1.5}"
        "h2{margin:18px 0 4px;font-size:15px}li{margin:4px 0}</style>",
        "<h1>AnomalyThink - per-model trace viewers</h1>",
        "<p>Each page shows EVERY DS-MVTec or VisA sample for one model: the image, the true defect "
        "region overlaid in red, the ground-truth type, and that model's own generated reasoning "
        "trace and verdict. Self-contained (images embedded); open any file directly in a browser.</p>"]
    for display, items in index.items():
        parts.append(f"<h2>{esc(display)}</h2><ul>")
        for bench, fn, n, mb in items:
            parts.append(f'<li><a href="{fn}">{esc(bench)}</a> - {n} samples, {mb:.1f} MB</li>')
        parts.append("</ul>")
    (OUTDIR / "index.html").write_text("\n".join(parts))
    print(f"  wrote index.html -> {OUTDIR}")

if __name__ == "__main__":
    main()
