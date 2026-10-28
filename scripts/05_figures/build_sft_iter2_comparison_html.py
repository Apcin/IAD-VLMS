#!/usr/bin/env python
"""Build a self-contained HTML comparing run2/ckpt-530 baseline vs SFT-Iter2 ep1.

Shows:
  - Per-product BalAcc bar chart (baseline vs SFT-Iter2)
  - Sample items per product with:
      * Image + mask-overlay toggle
      * Baseline prediction + full trace
      * SFT-Iter2 prediction + full trace
  - Filter bar: dataset, product, outcome category (both right / regression / improvement / both wrong)

Output: one HTML file with all images embedded as base64.

Usage:
  python Training/build_sft_iter2_comparison_html.py --output cmp.html
"""
import argparse
import base64
import io
import json
import os
import random
import re
from collections import defaultdict
from html import escape
from pathlib import Path

from PIL import Image

# ───────────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────────
BASELINE_DIR = "/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530"
SFT2_DIR     = "/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_iter2_clean/checkpoint-188"
DATASETS     = [
    ("dsmvtec", "eval_dsmvtec_full_trainprompt.json"),
    ("VisA",    "eval_visa_full_trainprompt.json"),
]
ITEMS_PER_PRODUCT_PER_CATEGORY = 1  # 4 categories × 1 each = 4 items per product
MAX_IMAGE_DIM = 600


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────
def load_eval(path):
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)


def b64_image(path, max_dim=MAX_IMAGE_DIM):
    if not path or not os.path.exists(path): return None
    img = Image.open(path)
    if img.mode != 'RGB': img = img.convert('RGB')
    w, h = img.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        img = img.resize((int(w*s), int(h*s)))
    buf = io.BytesIO(); img.save(buf, format='JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def b64_mask(path, max_dim=MAX_IMAGE_DIM):
    """Convert grayscale mask to red-tinted RGBA PNG for overlay.

    Mask conventions across datasets:
      Real-IAD : 0 / 255 binary
      DS-MVTec : 0 / 1 (or 0 / 2) binary — values are very low for defect pixels
      VisA     : 0 / 255 binary
    Treat ANY non-zero pixel as defect (threshold > 0). Resize via NEAREST so
    we don't blur the edges.
    """
    if not path or not os.path.exists(path): return None
    mask = Image.open(path).convert('L')
    w, h = mask.size
    px = mask.load()
    rgba = Image.new('RGBA', mask.size, (255, 0, 0, 0))
    out = rgba.load()
    for y in range(h):
        for x in range(w):
            if px[x, y] > 0:
                out[x, y] = (255, 0, 0, 180)  # red, ~70% opacity
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        rgba = rgba.resize((int(w*s), int(h*s)), resample=Image.NEAREST)
    buf = io.BytesIO(); rgba.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def find_mask(image_path):
    """Locate the GT defect mask for an image path. Three datasets, three conventions:

      Real-IAD : <dir>/<n>.jpg                              -> <dir>/<n>.png
      DS-MVTec : .../<prod>/image/<defect>/<n>.png          -> .../<prod>/mask/<defect>/<n>_mask.png
      VisA     : .../<prod>/test/<defect>/<n>.JPG           -> .../<prod>/ground_truth/<defect>/<n>.png

    The earlier same-dir `.png` fallback was wrong for DS-MVTec (returned the image itself).
    """
    if not image_path: return None
    ext = os.path.splitext(image_path)[1].lower()

    # Real-IAD: only when source is .jpg (avoid returning self for .png images)
    if ext == '.jpg':
        cand = image_path[:-4] + '.png'
        if os.path.exists(cand): return cand

    # DS-MVTec
    m = re.search(r'^(.*?/DS-MVTec/[^/]+)/image/(.+?)/(\d+)\.png$', image_path, re.IGNORECASE)
    if m:
        cand = f"{m.group(1)}/mask/{m.group(2)}/{m.group(3)}_mask.png"
        if os.path.exists(cand): return cand

    # VisA
    m = re.search(r'^(.*?/VisA/[^/]+)/test/(.+?)/(\d+)\.(?:JPG|jpg)$', image_path, re.IGNORECASE)
    if m:
        cand = f"{m.group(1)}/ground_truth/{m.group(2)}/{m.group(3)}.png"
        if os.path.exists(cand): return cand

    return None


def per_product_balacc(results):
    """Returns dict: product -> {balacc, ng_rec, ok_rec, n, n_ng, n_ok}."""
    by_p = defaultdict(lambda: {'tp':0, 'fn':0, 'fp':0, 'tn':0})
    for r in results:
        prod = r.get('product','?')
        gt = (r.get('gt_answer') or '').lower() == 'yes'
        pr = (r.get('pred_answer') or '').lower() == 'yes'
        c = by_p[prod]
        if gt and pr: c['tp'] += 1
        elif gt: c['fn'] += 1
        elif pr: c['fp'] += 1
        else: c['tn'] += 1
    out = {}
    for prod, c in by_p.items():
        n_ng = c['tp']+c['fn']; n_ok = c['tn']+c['fp']
        ng_rec = 100*c['tp']/n_ng if n_ng else None
        ok_rec = 100*c['tn']/n_ok if n_ok else None
        if ng_rec is None and ok_rec is None: continue
        if ng_rec is None: balacc = ok_rec
        elif ok_rec is None: balacc = ng_rec
        else: balacc = (ng_rec + ok_rec) / 2
        out[prod] = {'balacc':balacc, 'ng_rec':ng_rec, 'ok_rec':ok_rec,
                     'n':n_ng+n_ok, 'n_ng':n_ng, 'n_ok':n_ok}
    return out


def categorise(base_correct, sft_correct):
    if base_correct and sft_correct: return 'both_right'
    if base_correct and not sft_correct: return 'regression'
    if not base_correct and sft_correct: return 'improvement'
    return 'both_wrong'


# ───────────────────────────────────────────────────────────────────────────
# Build the HTML
# ───────────────────────────────────────────────────────────────────────────
CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1500px; margin: 0 auto; padding: 16px; background: #f5f5f7; color: #1d1d1f; }
h1, h2, h3 { margin-top: 0; }
.intro, .section, .product-block, .item {
  background: white; border-radius: 8px; padding: 16px; margin-bottom: 14px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.tag { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px;
       font-weight: 500; margin-right: 4px; background: #eee; }
.tag.dsmvtec { background: #d0e7ff; color: #003c8f; }
.tag.VisA { background: #ffd9c0; color: #803300; }
.tag.both_right { background: #d4edda; color: #155724; }
.tag.regression { background: #f8d7da; color: #721c24; }
.tag.improvement { background: #cce5ff; color: #004085; }
.tag.both_wrong { background: #e2e3e5; color: #383d41; }
.tag.ng { background: #fde7e9; color: #c00; }
.tag.ok { background: #e0f0e0; color: #1b6b1b; }

/* Bar chart */
.chart-wrap { overflow-x: auto; }
.chart-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px;
             font-family: ui-monospace, monospace; font-size: 12px; }
.chart-row .product { width: 120px; text-align: right; }
.chart-row .delta { width: 60px; font-weight: 600; }
.chart-row .delta.up { color: #0a7c3a; }
.chart-row .delta.down { color: #c00; }
.chart-row .bars { flex: 1; display: flex; flex-direction: column; gap: 2px; }
.chart-row .bar { height: 14px; border-radius: 2px; display: flex; align-items: center;
                 color: white; padding-left: 4px; font-size: 10px; font-weight: 600;
                 white-space: nowrap; box-shadow: inset 0 -1px 0 rgba(0,0,0,0.15); }
.chart-row .bar.base { background: #6c757d; }
.chart-row .bar.sft  { background: #0066cc; }
.chart-row .label-base, .chart-row .label-sft { width: 40px; font-size: 10px; }

/* Filters */
.filters { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; padding: 8px 0; }
.filters label { font-size: 13px; }
.filters select, .filters button {
  padding: 6px 10px; font-size: 13px; border: 1px solid #ccc;
  border-radius: 4px; background: white; cursor: pointer;
}
.filters button.active { background: #0066cc; color: white; border-color: #0066cc; }
.filters .stat { font-size: 12px; color: #555; margin-left: 8px; }

/* Items */
.product-block h2 { padding-bottom: 8px; border-bottom: 2px solid #f0f0f3; }
.item { display: grid; grid-template-columns: 480px 1fr; gap: 16px;
        background: #fafafa; border: 1px solid #e5e5e7; }
.viewer { display: flex; flex-direction: column; gap: 8px; }
.canvas { position: relative; width: 460px; height: 460px;
          background: #000; border-radius: 4px; overflow: hidden; }
.canvas img.base { position: absolute; left: 0; top: 0; width: 100%; height: 100%; object-fit: contain; }
.canvas img.mask { position: absolute; left: 0; top: 0; width: 100%; height: 100%;
                    object-fit: contain; opacity: 0; transition: opacity 0.15s; pointer-events: none; }
.ctl { display: flex; gap: 4px; }
.ctl button { padding: 6px 10px; font-size: 12px; border: 1px solid #ccc;
              background: white; border-radius: 4px; cursor: pointer; flex: 1; }
.ctl button.active { background: #0066cc; color: white; border-color: #0066cc; }
.ctl button:disabled { opacity: 0.4; cursor: not-allowed; }
.filename { font-size: 11px; color: #666; word-break: break-all; }

.details { display: flex; flex-direction: column; gap: 8px; }
.details h3 { font-size: 14px; margin: 0; }
.meta { font-size: 12px; color: #555; }
.trace-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.trace { background: #f7f7f9; border-left: 3px solid #999; padding: 8px 10px;
         font-family: ui-monospace, monospace; font-size: 10.5px;
         white-space: pre-wrap; max-height: 250px; overflow-y: auto; }
.trace.correct { border-left-color: #0a7c3a; }
.trace.wrong   { border-left-color: #c00; }
.trace-label { font-size: 12px; font-weight: 600; margin-bottom: 4px; }
.trace-label.correct { color: #0a7c3a; }
.trace-label.wrong   { color: #c00; }
.hidden { display: none; }
"""

JS = """
function setView(btn, viewerId, mode) {
  const v = document.getElementById(viewerId);
  const mask = v.querySelector('img.mask');
  v.parentElement.querySelectorAll('.ctl button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  if (!mask) return;
  if (mode === 'image') mask.style.opacity = 0;
  else if (mode === 'overlay') mask.style.opacity = 0.6;
  else if (mode === 'mask') mask.style.opacity = 1;
}
function applyFilters() {
  const ds = document.getElementById('f-dataset').value;
  const prod = document.getElementById('f-product').value;
  const cat = document.getElementById('f-outcome').value;
  const gt = document.getElementById('f-gt').value;
  let shown = 0;
  document.querySelectorAll('.item').forEach(el => {
    const ok = (ds === 'ALL' || el.dataset.dataset === ds)
            && (prod === 'ALL' || el.dataset.product === prod)
            && (cat === 'ALL' || el.dataset.outcome === cat)
            && (gt === 'ALL' || el.dataset.gt === gt);
    el.classList.toggle('hidden', !ok);
    if (ok) shown++;
  });
  // Hide empty product blocks
  document.querySelectorAll('.product-block').forEach(blk => {
    const visible = blk.querySelectorAll('.item:not(.hidden)').length;
    blk.classList.toggle('hidden', visible === 0);
  });
  document.getElementById('item-count').textContent = `${shown} items shown`;
}
"""


def render_chart(per_prod_baseline, per_prod_sft):
    """Bar chart showing baseline vs SFT-Iter2 per product, sorted by SFT delta."""
    rows = []
    for prod in set(per_prod_baseline) | set(per_prod_sft):
        b = per_prod_baseline.get(prod, {}).get('balacc')
        s = per_prod_sft.get(prod, {}).get('balacc')
        if b is None or s is None: continue
        rows.append((prod, b, s, s - b))
    rows.sort(key=lambda x: -x[3])  # by delta, biggest improvements first

    html = ['<div class="chart-wrap">']
    for prod, b, s, d in rows:
        delta_class = 'up' if d > 0 else ('down' if d < 0 else '')
        bar_b = f'<div class="bar base" style="width:{b}%">{b:.1f}%</div>'
        bar_s = f'<div class="bar sft"  style="width:{s}%">{s:.1f}%</div>'
        html.append(f'''
          <div class="chart-row">
            <span class="product">{escape(prod)}</span>
            <div class="bars">{bar_b}{bar_s}</div>
            <span class="delta {delta_class}">{d:+.1f}</span>
          </div>''')
    html.append('</div>')
    html.append('<p style="font-size:12px;color:#666;margin-top:8px">'
                'Gray = baseline (run2/ckpt-530) &nbsp; • &nbsp; Blue = SFT-Iter2 ep1. '
                'Right column: ΔBalAcc (green = improvement).</p>')
    return ''.join(html)


def render_item(rec, idx):
    img_b64 = b64_image(rec['image_path'])
    mask_b64 = b64_mask(rec['mask_path']) if rec['mask_path'] else None
    if img_b64 is None:
        return ''
    has_mask = mask_b64 is not None
    vid = f'v-{idx}'
    base_ok = rec['baseline_correct']
    sft_ok  = rec['sft_correct']
    outcome = rec['outcome']
    fn = os.path.basename(rec['image_path'])

    return f'''
<div class="item"
     data-dataset="{rec['dataset']}"
     data-product="{escape(rec['product'])}"
     data-outcome="{outcome}"
     data-gt="{rec['gt_answer']}">
  <div class="viewer">
    <div id="{vid}" class="canvas">
      <img class="base" src="{img_b64}"/>
      {f'<img class="mask" src="{mask_b64}"/>' if mask_b64 else ''}
    </div>
    <div class="ctl">
      <button class="active" onclick="setView(this,'{vid}','image')">Image</button>
      <button onclick="setView(this,'{vid}','overlay')" {'' if has_mask else 'disabled'}>Overlay mask</button>
      <button onclick="setView(this,'{vid}','mask')" {'' if has_mask else 'disabled'}>Mask only</button>
    </div>
    <div class="filename">{escape(fn)}</div>
  </div>
  <div class="details">
    <h3>
      <span class="tag {rec['dataset']}">{rec['dataset']}</span>
      <span class="tag">{escape(rec['product'])}</span>
      <span class="tag {'ng' if rec['gt_answer']=='yes' else 'ok'}">GT: {rec['gt_answer'].upper()}</span>
      <span class="tag {outcome}">{outcome.replace('_', ' ')}</span>
    </h3>
    <div class="meta">
      <b>GT type:</b> {escape(rec.get('gt_type') or '—')} &nbsp; • &nbsp;
      <b>GT location:</b> {escape(rec.get('gt_location') or '—')}
    </div>
    <div class="trace-pair">
      <div>
        <div class="trace-label {'correct' if base_ok else 'wrong'}">
          run2/ckpt-530 — pred: <b>{rec['baseline_pred'].upper()}</b> {'✓' if base_ok else '✗'}
        </div>
        <div class="trace {'correct' if base_ok else 'wrong'}">{escape(rec['baseline_trace'])}</div>
      </div>
      <div>
        <div class="trace-label {'correct' if sft_ok else 'wrong'}">
          SFT-Iter2 ep1 — pred: <b>{rec['sft_pred'].upper()}</b> {'✓' if sft_ok else '✗'}
        </div>
        <div class="trace {'correct' if sft_ok else 'wrong'}">{escape(rec['sft_trace'])}</div>
      </div>
    </div>
  </div>
</div>'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--items_per_category', type=int, default=ITEMS_PER_PRODUCT_PER_CATEGORY)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    # ─── Load eval JSONs ──────────────────────────────────────────────────
    all_items = []
    chart_data = {}
    for ds_label, ds_file in DATASETS:
        base = load_eval(os.path.join(BASELINE_DIR, ds_file))
        sft  = load_eval(os.path.join(SFT2_DIR, ds_file))
        if not base or not sft:
            print(f"[warn] missing eval JSON for {ds_label}, skipping")
            continue

        # Per-product BalAcc
        base_per = per_product_balacc(base['results'])
        sft_per  = per_product_balacc(sft['results'])
        chart_data[ds_label] = (base_per, sft_per)

        # Match by image_id
        base_by_id = {r['image_id']: r for r in base['results']}
        sft_by_id  = {r['image_id']: r for r in sft['results']}
        common_ids = set(base_by_id) & set(sft_by_id)

        # Group by product, then by outcome
        by_prod_outcome = defaultdict(lambda: defaultdict(list))
        for iid in common_ids:
            b = base_by_id[iid]; s = sft_by_id[iid]
            base_ok = b.get('correct', False)
            sft_ok  = s.get('correct', False)
            outcome = categorise(base_ok, sft_ok)
            prod = b.get('product', '?')
            # Extract GT type/location from gt_tags
            gt_tags_raw = b.get('gt_tags', {})
            rec = {
                'dataset': ds_label,
                'product': prod,
                'image_id': iid,
                'image_path': b.get('absolute_path', ''),
                'mask_path': find_mask(b.get('absolute_path', '')),
                'gt_answer': (b.get('gt_answer') or '').lower(),
                'gt_type': gt_tags_raw.get('type') if isinstance(gt_tags_raw, dict) else None,
                'gt_location': gt_tags_raw.get('location') if isinstance(gt_tags_raw, dict) else None,
                'baseline_pred': (b.get('pred_answer') or '').lower(),
                'baseline_trace': b.get('pred_full', ''),
                'baseline_correct': base_ok,
                'sft_pred': (s.get('pred_answer') or '').lower(),
                'sft_trace': s.get('pred_full', ''),
                'sft_correct': sft_ok,
                'outcome': outcome,
            }
            by_prod_outcome[prod][outcome].append(rec)

        # Sample items_per_category items per (product, outcome)
        for prod, outcomes in by_prod_outcome.items():
            for outcome in ('improvement', 'regression', 'both_right', 'both_wrong'):
                items = outcomes.get(outcome, [])
                if items:
                    sampled = random.sample(items, min(args.items_per_category, len(items)))
                    all_items.extend(sampled)

    print(f"Selected {len(all_items)} items across {len(chart_data)} datasets")

    # ─── Build HTML ───────────────────────────────────────────────────────
    # Per-product blocks
    by_ds_prod = defaultdict(lambda: defaultdict(list))
    for rec in all_items:
        by_ds_prod[rec['dataset']][rec['product']].append(rec)

    # Chart sections
    chart_sections = []
    for ds_label in chart_data:
        base_per, sft_per = chart_data[ds_label]
        chart_sections.append(f'''
<div class="section">
  <h2>{escape(ds_label)} — per-product BalAcc</h2>
  {render_chart(base_per, sft_per)}
</div>''')

    # Item blocks per (dataset, product)
    item_sections = []
    for ds_label in by_ds_prod:
        for prod, items in sorted(by_ds_prod[ds_label].items()):
            item_sections.append(f'''
<div class="product-block" data-dataset="{ds_label}" data-product="{escape(prod)}">
  <h2><span class="tag {ds_label}">{ds_label}</span> {escape(prod)} <span style="font-size:13px;color:#666;font-weight:normal">({len(items)} sample item{'s' if len(items)!=1 else ''})</span></h2>
  {''.join(render_item(r, f"{ds_label}-{prod}-{i}") for i, r in enumerate(items))}
</div>''')

    # Filter options
    all_products = sorted({rec['product'] for rec in all_items})
    product_options = '\n'.join([f'<option value="{escape(p)}">{escape(p)}</option>' for p in all_products])
    dataset_options = '\n'.join([f'<option value="{ds}">{ds}</option>' for ds in chart_data])

    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SFT-Iter2 vs run2/ckpt-530 comparison</title>
<style>{CSS}</style></head><body>

<div class="intro">
<h1>SFT-Iter2 (ep1) vs run2/ckpt-530 baseline</h1>
<p>Per-product BalAcc comparison + side-by-side trace inspection. Use filters to drill down.</p>
<p style="font-size:13px;color:#666">
  Baseline: <code>{escape(BASELINE_DIR)}</code><br>
  SFT-Iter2: <code>{escape(SFT2_DIR)}</code>
</p>
</div>

{''.join(chart_sections)}

<div class="section">
<h2>Sample items</h2>
<div class="filters">
  <label>Dataset:
    <select id="f-dataset" onchange="applyFilters()">
      <option value="ALL">All</option>
      {dataset_options}
    </select>
  </label>
  <label>Product:
    <select id="f-product" onchange="applyFilters()">
      <option value="ALL">All</option>
      {product_options}
    </select>
  </label>
  <label>Outcome:
    <select id="f-outcome" onchange="applyFilters()">
      <option value="ALL">All</option>
      <option value="improvement">Improvement (base wrong → SFT correct)</option>
      <option value="regression">Regression (base correct → SFT wrong)</option>
      <option value="both_right">Both right</option>
      <option value="both_wrong">Both wrong</option>
    </select>
  </label>
  <label>GT class:
    <select id="f-gt" onchange="applyFilters()">
      <option value="ALL">All</option>
      <option value="yes">NG (defect present)</option>
      <option value="no">OK (no defect)</option>
    </select>
  </label>
  <span class="stat" id="item-count">{len(all_items)} items shown</span>
</div>
</div>

{''.join(item_sections)}

<script>{JS}</script>
</body></html>'''

    with open(args.output, 'w') as f:
        f.write(html)
    sz_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Wrote {args.output}  ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
