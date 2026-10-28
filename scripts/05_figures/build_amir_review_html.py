#!/usr/bin/env python
"""Build an HTML review report for Amir from a Phase 0/1a/1b pilot directory.

Two modes:
  --mode implicit    images referenced by absolute path (small file, only viewable on the server)
  --mode explicit    images embedded as base64 (big file, self-contained — emailable)

The page contains:
  - Header with toggle to show:
      * Rollout user prompt
      * Gemini judge system prompt + 5-level rubric
      * Gemini rewrite + correct system prompts
      * Local reward formulas (accuracy, format, type bucketing, location)
  - Filter bar: product, gt_type, faithfulness label, local_bucket, NG/OK, difficulty range, source_pool
  - Per-item cards: image (+mask), GT info, local scores, judge rationale + flags + obs,
                    chosen trace, Gemini-corrected/rewritten trace (if available), all-rollouts dropdown
"""

import argparse
import base64
import io
import json
import os
import re
from collections import defaultdict
from html import escape


def b64_image(path, max_dim=512):
    if not os.path.exists(path):
        return None
    from PIL import Image
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def find_mask(image_path):
    candidate = image_path.rsplit(".", 1)[0] + ".png"
    return candidate if os.path.exists(candidate) else None


# Reference content for the "show prompts/formulas" section — load these directly
# from the source scripts so they stay in sync with what we actually used.
def load_prompts():
    here = "/bulk/aacudad/reasoning_traces/Training"
    judge_src   = open(f"{here}/phase1a_gemini_judge.py").read()
    correct_src = open(f"{here}/phase1b_gemini_correct.py").read()
    # Pull out the JUDGE_SYSTEM, REWRITE_SYSTEM, CORRECT_SYSTEM strings
    def extract(src, marker):
        # Find `MARKER = """ ... """`
        m = re.search(rf'{marker}\s*=\s*"""(.+?)"""', src, re.DOTALL)
        return m.group(1).strip() if m else "(prompt not found)"
    return {
        "rollout_user_template": (
            "Analyze the provided image of the {PRODUCT}. "
            "Determine if there are any anomalies present. "
            "If an anomaly is detected, specify its type and location, "
            "and provide a detailed reasoning for your conclusion."
        ),
        "judge_system":   extract(judge_src,   "JUDGE_SYSTEM"),
        "rewrite_system": extract(correct_src, "REWRITE_SYSTEM"),
        "correct_system": extract(correct_src, "CORRECT_SYSTEM"),
    }


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1400px; margin: 0 auto; padding: 16px; background: #f5f5f7; color: #1d1d1f; }
h1 { margin-top: 0; }
.docs { background: white; border-radius: 8px; padding: 16px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.docs details { margin: 8px 0; }
.docs summary { cursor: pointer; font-weight: 600; padding: 6px 0; }
.docs pre { background: #fafafa; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 12px; white-space: pre-wrap; }
.filters { background: white; border-radius: 8px; padding: 12px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); display: flex; flex-wrap: wrap; gap: 12px; align-items: center; position: sticky; top: 0; z-index: 100; }
.filters select, .filters input { padding: 6px 8px; border-radius: 4px; border: 1px solid #ccc; }
.filters .count { margin-left: auto; font-weight: 600; }
.card { background: white; border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); display: grid; grid-template-columns: 320px 1fr; gap: 16px; }
.card img { max-width: 300px; max-height: 300px; border-radius: 4px; display: block; }
.card .imgs { display: flex; flex-direction: column; gap: 8px; }
.card .imgs .lbl { font-size: 11px; color: #666; }
.card .meta { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }
.tag { padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; background: #eee; }
.tag.ng { background: #ffe5e5; color: #cc0000; }
.tag.ok { background: #e5f3ff; color: #0066cc; }
.tag.bucket-good { background: #d4edda; color: #155724; }
.tag.bucket-correction { background: #f8d7da; color: #721c24; }
.tag.score-0 { background: #d63031; color: white; }
.tag.score-1 { background: #e17055; color: white; }
.tag.score-2 { background: #fdcb6e; }
.tag.score-3 { background: #74b9ff; }
.tag.score-4 { background: #00b894; color: white; }
.card h3 { margin: 8px 0 4px; font-size: 14px; }
.card .trace { background: #fafafa; padding: 8px 10px; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 12px; white-space: pre-wrap; max-height: 200px; overflow-y: auto; border-left: 3px solid #ddd; }
.card .trace.corrected { border-left-color: #00b894; }
.card .trace.original { border-left-color: #fdcb6e; }
.flags { color: #cc0000; font-size: 12px; }
.observations { color: #00897b; font-size: 12px; }
ul.bullets { margin: 4px 0; padding-left: 18px; }
ul.bullets li { margin: 2px 0; }
.rationale { font-style: italic; color: #555; margin: 6px 0; }
.scores { font-family: ui-monospace, monospace; font-size: 12px; color: #555; }
.expand-all { font-size: 12px; cursor: pointer; color: #0066cc; }
details.rollouts { margin-top: 8px; }
details.rollouts > summary { font-size: 12px; color: #0066cc; cursor: pointer; }
details.rollouts pre { margin: 4px 0; font-size: 11px; }
"""

JS = """
function applyFilters() {
  const filters = {
    product:    document.getElementById('f-product').value,
    gt_type:    document.getElementById('f-type').value,
    fscore:     document.getElementById('f-score').value,
    bucket:     document.getElementById('f-bucket').value,
    ngok:       document.getElementById('f-ngok').value,
    pool:       document.getElementById('f-pool').value,
  };
  const cards = document.querySelectorAll('.card');
  let shown = 0;
  cards.forEach(c => {
    let visible = true;
    for (const [k, v] of Object.entries(filters)) {
      if (v && v !== 'all' && c.dataset[k] !== v) { visible = false; break; }
    }
    c.style.display = visible ? '' : 'none';
    if (visible) shown++;
  });
  document.querySelector('.count').textContent = shown + ' / ' + cards.length + ' items';
}
window.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.filters select').forEach(s => s.addEventListener('change', applyFilters));
});
"""


def render_card(item, mode, image_cache):
    """Render one item card from the merged record."""
    image_path = item["image_path"]
    is_anomaly = item.get("is_anomaly", False)

    if mode == "implicit":
        img_html = f'<img src="file://{escape(image_path)}" alt="image"/>'
        mask_path = find_mask(image_path) if is_anomaly else None
        mask_html = f'<img src="file://{escape(mask_path)}" alt="mask"/>' if mask_path else ""
    else:  # explicit
        img_b64 = image_cache.get(image_path) or b64_image(image_path)
        if img_b64: image_cache[image_path] = img_b64
        img_html = f'<img src="{img_b64}" alt="image"/>' if img_b64 else ""
        mask_path = find_mask(image_path) if is_anomaly else None
        mask_b64 = image_cache.get(mask_path) if mask_path else None
        if mask_path and mask_b64 is None:
            mask_b64 = b64_image(mask_path)
            if mask_b64: image_cache[mask_path] = mask_b64
        mask_html = f'<img src="{mask_b64}" alt="mask"/>' if mask_b64 else ""

    fname = os.path.basename(image_path)
    product = item.get("product", "?")
    gt_answer = item.get("gt_answer", "?")
    gt_type = item.get("gt_type")
    gt_location = item.get("gt_location")
    bucket = item.get("local_bucket", "?")
    fscore = item.get("faithfulness_score", "")
    flabel = item.get("faithfulness_label", "?")
    pool = item.get("source_pool", "?")

    flag_html = ""
    if item.get("hallucination_flags"):
        flag_html = '<div class="flags"><b>Hallucination flags:</b><ul class="bullets>">' + \
                    "".join(f"<li>{escape(f)}</li>" for f in item["hallucination_flags"]) + "</ul></div>"
    obs_html = ""
    if item.get("grounded_observations"):
        obs_html = '<div class="observations"><b>Grounded observations:</b><ul class="bullets">' + \
                   "".join(f"<li>{escape(o)}</li>" for o in item["grounded_observations"]) + "</ul></div>"

    # The chosen trace (passed local reward or the best failed)
    chosen_trace = item.get("chosen_trace", item.get("original_trace", item.get("best_failed_attempt", "")))
    # Gemini's correction or rewrite, if available
    gemini_trace = item.get("gemini_output_trace", "")
    gemini_mode  = item.get("gemini_output_mode", "")  # "correct" or "rewrite"

    # All rollouts (8 of them)
    rollouts_html = ""
    if item.get("all_rollouts"):
        rollouts_html = "<details class='rollouts'><summary>Show all 8 rollouts</summary>"
        for i, r in enumerate(item["all_rollouts"]):
            badge = f"acc={r['acc']:.2f} fmt={r['format']:.0f} len={r['len']}"
            rollouts_html += f"<pre><b>Rollout {i+1} ({badge}):</b>\n{escape(r['text'][:1500])}</pre>"
        rollouts_html += "</details>"

    # Build data attrs for filtering
    data_attrs = (
        f'data-product="{escape(product)}" '
        f'data-gt_type="{escape(str(gt_type) if gt_type else "none")}" '
        f'data-fscore="{fscore}" '
        f'data-bucket="{bucket}" '
        f'data-ngok="{gt_answer}" '
        f'data-pool="{pool}"'
    )

    return f"""
<div class="card" {data_attrs}>
  <div class="imgs">
    <div class="lbl">image: {escape(fname)}</div>
    {img_html}
    {('<div class="lbl">GT defect mask (white = defect):</div>' + mask_html) if mask_html else ''}
  </div>
  <div>
    <div class="meta">
      <span class="tag {'ng' if is_anomaly else 'ok'}">{escape(gt_answer.upper())}</span>
      <span class="tag bucket-{bucket}">local: {escape(bucket)}</span>
      <span class="tag score-{fscore}">judge: {fscore} {escape(flabel)}</span>
      <span class="tag">product: {escape(product)}</span>
      {f'<span class="tag">type: {escape(gt_type)}</span>' if gt_type else ''}
      {f'<span class="tag">location: {escape(gt_location)}</span>' if gt_location else ''}
      <span class="tag">pool: {escape(pool)}</span>
      <span class="tag">difficulty: {item.get('difficulty', 0):.2f}</span>
    </div>
    <div class="scores">
      acc={item.get('chosen_acc') or item.get('original_acc') or 0:.2f}
      &nbsp;type_score={item.get('chosen_type_score') if item.get('chosen_type_score') is not None else '-'}
      &nbsp;loc_score={item.get('chosen_loc_score') if item.get('chosen_loc_score') is not None else '-'}
    </div>
    <div class="rationale"><b>Gemini rationale:</b> {escape(item.get('rationale','(none)'))}</div>
    {flag_html}
    {obs_html}
    <h3>Model's chosen trace</h3>
    <div class="trace original">{escape(chosen_trace)}</div>
    {f'<h3>Gemini {gemini_mode}</h3><div class="trace corrected">{escape(gemini_trace)}</div>' if gemini_trace else ''}
    {rollouts_html}
  </div>
</div>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot_dir", required=True,
                   help="Directory containing rollouts_raw.jsonl + bucket outputs + judge_report.jsonl + gemini_*.jsonl")
    p.add_argument("--output_html", required=True)
    p.add_argument("--mode", choices=["implicit", "explicit"], default="implicit")
    p.add_argument("--max_image_dim", type=int, default=512,
                   help="Resize images to this max dim before base64 (explicit mode only)")
    args = p.parse_args()

    pilot = args.pilot_dir
    print(f"[build] mode={args.mode}, pilot={pilot}")

    # Load everything
    items_by_path = {}

    # Load judge_input for fallback local_bucket lookup (judge_report may lack it
    # if records were generated before we propagated the field)
    bucket_by_path = {}
    if os.path.exists(f"{pilot}/judge_input.json"):
        with open(f"{pilot}/judge_input.json") as f:
            for it in json.load(f):
                bucket_by_path[it["image_path"]] = it.get("local_bucket", "good")

    # Start from judge_report (covers all 100)
    with open(f"{pilot}/judge_report.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("success"): continue
            # Fill in local_bucket if missing from raw record
            if not r.get("local_bucket"):
                r["local_bucket"] = bucket_by_path.get(r["image_path"], "good")
            items_by_path[r["image_path"]] = r

    # Attach rollouts_raw (all 8 per item)
    with open(f"{pilot}/rollouts_raw.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            ip = rec["image_path"]
            if ip in items_by_path:
                items_by_path[ip]["all_rollouts"] = rec["rollouts"]

    # Attach Gemini outputs if present
    for path, mode in [(f"{pilot}/gemini_corrected.jsonl", "correct"),
                       (f"{pilot}/gemini_rewritten.jsonl", "rewrite")]:
        if not os.path.exists(path):
            print(f"[build] (no {path}, skipping)")
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if not rec.get("success"): continue
                ip = rec["image_path"]
                if ip in items_by_path:
                    items_by_path[ip]["gemini_output_trace"] = rec.get("corrected_trace", "")
                    items_by_path[ip]["gemini_output_mode"] = mode

    items = list(items_by_path.values())
    print(f"[build] {len(items)} items total")

    # Build filter options
    products = sorted({i["product"] for i in items if i.get("product")})
    types    = sorted({i.get("gt_type") or "none" for i in items})
    buckets  = sorted({i.get("local_bucket", "?") for i in items})
    pools    = sorted({i.get("source_pool", "?") for i in items})

    prompts = load_prompts()

    image_cache = {}

    # Render cards (sort: bucket=correction first, then by judge_score asc)
    def sort_key(it):
        return (it.get("local_bucket") != "correction", it.get("faithfulness_score", 99))
    items.sort(key=sort_key)

    cards_html = "\n".join(render_card(it, args.mode, image_cache) for it in items)

    # Headers, docs, filters
    def opts(label, opts_list):
        ops = '<option value="all">all</option>' + "".join(f'<option value="{escape(str(o))}">{escape(str(o))}</option>' for o in opts_list)
        return f'<label>{escape(label)} <select id="f-{label.lower().replace(" ","_")}">{ops}</select></label>'

    docs_html = f"""
<div class="docs">
  <h1>Phase 0/1 Pilot Review — {os.path.basename(pilot)}</h1>
  <p>{len(items)} items rolled out from <b>run2/ckpt-530</b> with k=8 generations each, locally scored, then judged by <b>Gemini 3 Flash Preview</b> for visual faithfulness on a 5-level rubric.</p>
  <details><summary>📋 Rollout user prompt (per item, with product name substituted)</summary>
    <pre>{escape(prompts['rollout_user_template'])}</pre>
  </details>
  <details><summary>🧑‍⚖️ Gemini judge system prompt + 5-level rubric</summary>
    <pre>{escape(prompts['judge_system'])}</pre>
  </details>
  <details><summary>✍️ Gemini rewrite system prompt (for correct-but-weak traces)</summary>
    <pre>{escape(prompts['rewrite_system'])}</pre>
  </details>
  <details><summary>🛠️ Gemini correct system prompt (for wrong traces)</summary>
    <pre>{escape(prompts['correct_system'])}</pre>
  </details>
  <details><summary>📐 Local reward formulas</summary>
    <pre>FORMAT (consistency_reward) — binary 0 or 1
  For OK (gt="no"):   <think>...</think><answer>no</answer>   AND no <location>/<type> tags
  For NG (gt="yes"):  <think>...</think><location>...</location><type>...</type><answer>yes</answer>

ACCURACY (accuracy_reward):
  For OK (gt="no"):   1.0 if model's <answer> = "no", else 0.0          (range [0, 1])
  For NG (gt="yes"):  (type_score + loc_score) / 2.0 + (1.0 if model says "yes")   (range [0, 2])

TYPE SCORE (Nomic embedding similarity → bucketed):
  sim >= 0.90 → 1.0    sim >= 0.55 → 0.5
  sim >= 0.80 → 0.9    sim >= 0.40 → 0.2
  sim >= 0.70 → 0.7    else        → 0.0

LOCATION SCORE (binary):
  1.0 if model's <location> maps to same 3x3 grid cell as GT, else 0.0

PASS THRESHOLDS (bucketing):
  OK item passes if format=1 AND acc >= 1.0
  NG item passes if format=1 AND acc >= 1.5  (= correct yes + (type+loc)/2 >= 0.5)
  NG "weak-but-correct" rewrite eligible if passing AND acc < 1.8

PER-ITEM DIFFICULTY (k=8 rollouts):
  difficulty = (8 - num_passing) / 8   ∈ {{0.00 (easy) … 1.00 (impossible)}}</pre>
  </details>
  <details><summary>📊 Score distribution (this pilot)</summary>
    <pre>{escape(_score_distribution(items))}</pre>
  </details>
</div>
"""

    filters_html = f"""
<div class="filters">
  {opts('product',  products)}
  {opts('type',     types)}
  {opts('score',    [0,1,2,3,4])}
  {opts('bucket',   buckets)}
  {opts('ngok',     ['yes','no'])}
  {opts('pool',     pools)}
  <span class="count">{len(items)} / {len(items)} items</span>
</div>
"""

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Phase 0/1 review</title>
<style>{CSS}</style></head><body>
{docs_html}
{filters_html}
{cards_html}
<script>{JS}</script>
</body></html>"""

    with open(args.output_html, "w") as f:
        f.write(html)

    size_mb = os.path.getsize(args.output_html) / 1024 / 1024
    print(f"[build] wrote {args.output_html}  ({size_mb:.2f} MB)")


def _score_distribution(items):
    from collections import Counter
    by_bucket = defaultdict(Counter)
    for it in items:
        by_bucket[it.get("local_bucket", "?")][it.get("faithfulness_score", -1)] += 1
    lines = [f"{'local bucket':<20} {'0 HALL':>7} {'1 POOR':>7} {'2 ACC':>7} {'3 GOOD':>7} {'4 EXCL':>7} {'total':>7}"]
    lines.append("=" * 70)
    for b, row in sorted(by_bucket.items()):
        lines.append(f"{b:<20} " + " ".join(f"{row[s]:>7}" for s in [0,1,2,3,4]) + f" {sum(row.values()):>7}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
