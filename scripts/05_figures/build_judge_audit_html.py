#!/usr/bin/env python
"""Build a focused audit HTML for the 7 lenience patterns the sub-agents flagged.

For each pattern, one or more concrete examples are shown with:
  - The product image (with a TOGGLE BUTTON to overlay the GT mask)
  - The model's chosen trace
  - The ground-truth info
  - Gemini's original verdict (score + rationale + flags)
  - What the AUDIT AGENT claimed was wrong
  - Buttons to record YOUR verdict (Agree / Disagree / Borderline)

Images embedded as base64 so the file is self-contained and emailable.
"""

import argparse
import base64
import io
import json
import os
from html import escape
from PIL import Image


def b64_image(path, max_dim=600):
    if not os.path.exists(path):
        return None
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def b64_mask(path, max_dim=600):
    """Mask as PNG with alpha so overlay shows defect region semi-transparent."""
    if not os.path.exists(path):
        return None
    mask = Image.open(path).convert("L")
    w, h = mask.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        mask = mask.resize((int(w * s), int(h * s)))
    # Make a red-tinted RGBA mask: red where mask is white, transparent elsewhere
    rgba = Image.new("RGBA", mask.size, (255, 0, 0, 0))
    px = mask.load()
    out = rgba.load()
    for y in range(mask.height):
        for x in range(mask.width):
            if px[x, y] > 50:
                out[x, y] = (255, 0, 0, 140)  # semi-transparent red
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def find_mask(image_path):
    p = image_path.rsplit(".", 1)[0] + ".png"
    return p if os.path.exists(p) else None


# ────────────────────────────────────────────────────────────────────────────
# The seven flagged patterns with their concrete example items
# Each entry has:
#   pattern_id, pattern_name, fix_text,
#   examples: list of {image_path, agent_claim, agent_verdict}
# ────────────────────────────────────────────────────────────────────────────

PATTERNS = [
    {
        "id": 1,
        "name": "Off-by-one positional counts",
        "fix": "Verify ordinal counts independently — fabricated indices = HALLUCINATED",
        "examples": [
            {
                "image": "/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/Real-IAD/images/transistor1/NG/BX/S0034/transistor1_0034_NG_BX_C1_20230923195401.jpg",
                "agent_claim": "Trace says 'the FIFTH lead from the left in the top row is severely deformed' — but counting from left, the bent/twisted pin is the 4TH of the leads in the top row. Off-by-one positional count.",
            },
        ],
    },
    {
        "id": 2,
        "name": "Absence claims accepted without verification",
        "fix": "When trace asserts absence of X, actively scan for counter-evidence",
        "examples": [
            {
                "image": "/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/Real-IAD/images/bottle_cap/OK/S0326/bottle_cap_0326_OK_C1_20230925185139.jpg",
                "agent_claim": "Trace claims: 'no signs of contamination like dust or hair, but the material appears clear and clean.' Image clearly shows thin hair-like fibers/strands INSIDE the cap (most prominent on right side and center). Gemini scored EXCELLENT (4) despite the trace contradicting visible features.",
            },
        ],
    },
    {
        "id": 3,
        "name": "Geometry mislabels accepted",
        "fix": "Shape names (hexagonal/circular/etc) must match actual silhouette; mismatches cap at score 2",
        "examples": [
            {
                "image": "/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/Real-IAD/images/plastic_nut/NG/AK/S0044/plastic_nut_0044_NG_AK_C1_20231004133152.jpg",
                "agent_claim": "Trace calls part 'hexagonal outer profile with six points' — but the actual silhouette is a 6-pointed STAR/gear shape with concave curved sides, NOT a hexagon. A 6-pointed star is geometrically distinct from a hexagon.",
            },
            {
                "image": "/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/Real-IAD/images/plastic_nut/NG/QS/S0029/plastic_nut_0029_NG_QS_C1_20231004113115.jpg",
                "agent_claim": "Same product, same geometry hallucination: trace says 'hexagonal' for a 6-pointed star shape. This appears to be a learned product-label-driven fabrication ('nut' → hex by default).",
            },
        ],
    },
    {
        "id": 4,
        "name": "Omission-as-fabrication",
        "fix": "Trace cannot claim systematic inspection while ignoring dominant visible features",
        "examples": [
            {
                "image": "/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/Real-IAD/images/toy/NG/AK/S0024/toy_0024_NG_AK_C1_20230928100705.jpg",
                "agent_claim": "Trace describes 'orange-brown surface with white sesame seed patterns' but completely omits the prominent GREEN (lettuce) ring and YELLOW (cheese/bun) ring — half the visible toy. Trace claims to 'scan the printed area and plastic surface' yet ignores major colored features.",
            },
        ],
    },
    {
        "id": 5,
        "name": "2D anomalies described as 3D mechanism",
        "fix": "Distinguish appearance (color/stain) from mechanism (pit/dent/hole)",
        "examples": [
            {
                "image": "/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/Real-IAD/images/wooden_beads/NG/AK/S0122/wooden_beads_0122_NG_AK_C1_20231013153943.jpg",
                "agent_claim": "Trace calls the feature a 'pit/indentation' and 'drill bit not properly finished' — image clearly shows an orange/tan STAIN (color discoloration), not a depression. The wood surface is intact. GT type is 'Stain'. The trace invents a 3D physical mechanism (drill bit hole) for a 2D color anomaly.",
            },
        ],
    },
    {
        "id": 6,
        "name": "Stock language passes as grounded",
        "fix": "Generic phrases unsupported by image-specific landmarks don't count as grounded observations",
        "examples": [
            {
                "image": "/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/Real-IAD/images/u_block/OK/S0448/u_block_0448_OK_C1_20231005112811.jpg",
                "agent_claim": "Trace is highly generic — describes a 'matte finish' ring with 'no scratches/pits/discoloration'. Could apply to ANY ring image. No specific landmark cited (no text, no asymmetry, no fastener mentioned). Pure stock-language inspection that requires zero visual confirmation.",
            },
            {
                "image": "/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/Real-IAD/images/u_block/OK/S0419/u_block_0419_OK_C1_20231005112215.jpg",
                "agent_claim": "Trace says 'matte gray finish' — the actual ring is translucent/pale off-white with a slight bluish-gray tint. Calling it 'matte gray' is loose. Also: 'consistent fine-grained texture across entire surface' is unsupported stock language.",
            },
        ],
    },
    {
        "id": 7,
        "name": "Specificity treated as proof",
        "fix": "Specific claims (counts, sizes, exact colors) require independent verification, not endorsement",
        "examples": [
            {
                "image": "/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/Real-IAD/images/wooden_beads/OK/S0106/wooden_beads_0106_OK_C1_20231013095955.jpg",
                "agent_claim": "Trace claims 'bright RED LINEAR mark at top-center edge like a sharpie or ink stain'. Actual feature is a tiny faint PINKISH/ORANGE SPECK — NOT linear, NOT straight, NOT bright red. The trace's specificity ('like sharpie/ink stain') gives illusion of grounding while inventing the surface feature. Trace classifies sample as anomalous (GT is OK).",
            },
            {
                "image": "/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/Real-IAD/images/sim_card_set/NG/ZW/S0166/sim_card_set_0166_NG_ZW_C1_20230923142742.jpg",
                "agent_claim": "Trace says contamination on top-right of handle is 'orange'. Actual color in image is more red/pink. Color word fidelity drift accepted as 'EXCELLENT (4)' by Gemini.",
            },
        ],
    },
]


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1200px; margin: 0 auto; padding: 16px; background: #f5f5f7; color: #1d1d1f; }
h1 { margin-top: 0; }
.intro { background: white; border-radius: 8px; padding: 16px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.pattern { background: white; border-radius: 8px; padding: 16px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.pattern h2 { margin-top: 0; padding-bottom: 8px; border-bottom: 2px solid #f0f0f3; }
.fix { background: #eaf6ff; border-left: 4px solid #0066cc; padding: 8px 12px; margin: 8px 0 16px 0; font-family: ui-monospace, monospace; font-size: 13px; }
.example { background: #fafafa; border: 1px solid #e5e5e7; border-radius: 6px; padding: 12px; margin-bottom: 12px; display: grid; grid-template-columns: 480px 1fr; gap: 16px; }
.viewer { display: flex; flex-direction: column; gap: 8px; }
.viewer .canvas { position: relative; width: 460px; height: 460px; background: #000; border-radius: 4px; overflow: hidden; }
.viewer img.base { position: absolute; left: 0; top: 0; width: 100%; height: 100%; object-fit: contain; }
.viewer img.mask { position: absolute; left: 0; top: 0; width: 100%; height: 100%; object-fit: contain; opacity: 0; transition: opacity 0.15s ease; pointer-events: none; }
.viewer img.mask.visible { opacity: 1; }
.viewer .controls { display: flex; gap: 4px; }
.viewer .controls button { padding: 6px 10px; font-size: 12px; border: 1px solid #ccc; background: white; border-radius: 4px; cursor: pointer; flex: 1; }
.viewer .controls button.active { background: #0066cc; color: white; border-color: #0066cc; }
.viewer .filename { font-size: 11px; color: #666; word-break: break-all; }
.details h3 { margin: 0 0 6px; font-size: 14px; color: #cc0000; }
.details .claim { background: #fff5e6; border-left: 3px solid #ff9500; padding: 8px 10px; font-size: 13px; line-height: 1.5; margin: 6px 0 12px; }
.details .meta { font-size: 12px; color: #555; margin: 8px 0; }
.details .trace { background: #f7f7f9; border-left: 3px solid #999; padding: 8px 10px; font-family: ui-monospace, monospace; font-size: 11px; white-space: pre-wrap; max-height: 180px; overflow-y: auto; margin-top: 8px; }
.details .gemini { background: #ecffec; border-left: 3px solid #00b894; padding: 8px 10px; font-size: 12px; margin-top: 8px; }
.verdict { margin-top: 10px; }
.verdict button { padding: 6px 10px; font-size: 12px; border: 1px solid #ccc; background: white; border-radius: 4px; cursor: pointer; margin-right: 6px; }
.verdict button.agree-on { background: #00b894; color: white; border-color: #00b894; }
.verdict button.dis-on { background: #d63031; color: white; border-color: #d63031; }
.verdict button.brd-on { background: #fdcb6e; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 500; background: #eee; margin-right: 4px; }
"""

JS = """
function setView(btn, viewerId, mode) {
  const v = document.getElementById(viewerId);
  const mask = v.querySelector('img.mask');
  const buttons = v.parentElement.querySelectorAll('.controls button');
  buttons.forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  if (mode === 'image')        { mask.style.opacity = 0; }
  else if (mode === 'overlay') { mask.style.opacity = 0.6; }
  else if (mode === 'mask')    { mask.style.opacity = 1; }
}
function setVerdict(btn, id, choice) {
  document.querySelectorAll('#verdict-' + id + ' button').forEach(b => {
    b.classList.remove('agree-on'); b.classList.remove('dis-on'); b.classList.remove('brd-on');
  });
  btn.classList.add(choice + '-on');
  // Persist to localStorage so verdicts survive reload
  try { localStorage.setItem('verdict-' + id, choice); } catch(e) {}
}
window.addEventListener('DOMContentLoaded', () => {
  // Restore verdicts
  document.querySelectorAll('[id^="verdict-"]').forEach(div => {
    const id = div.id.replace('verdict-', '');
    const v = localStorage.getItem('verdict-' + id);
    if (!v) return;
    const btn = div.querySelector('button[data-choice="' + v + '"]');
    if (btn) btn.classList.add(v + '-on');
  });
});
"""


def render_example(pat_id, ex_idx, ex, judge_records):
    image_b64 = b64_image(ex["image"])
    if image_b64 is None:
        return f"<p>(image missing: {ex['image']})</p>"
    mask_path = find_mask(ex["image"])
    mask_b64 = b64_mask(mask_path) if mask_path else None

    # Look up the judge record for this image to attach Gemini's verdict
    rec = judge_records.get(ex["image"])
    if rec:
        gemini_score = rec.get("faithfulness_score")
        gemini_label = rec.get("faithfulness_label", "?")
        gemini_rat   = rec.get("rationale", "")
        gemini_flags = rec.get("hallucination_flags", []) or []
        gemini_obs   = rec.get("grounded_observations", []) or []
        chosen_trace = rec.get("chosen_trace", "(no trace found)")
        gt_answer    = rec.get("gt_answer", "?")
        gt_type      = rec.get("gt_type")
        gt_location  = rec.get("gt_location")
        product      = rec.get("product", "?")
    else:
        gemini_score = "?"
        gemini_label = "(not in judge_report)"
        gemini_rat = "n/a"
        gemini_flags = []
        gemini_obs = []
        chosen_trace = "(no trace found)"
        gt_answer = "?"
        gt_type = "?"
        gt_location = "?"
        product = "?"

    viewer_id = f"viewer-{pat_id}-{ex_idx}"
    verdict_id = f"{pat_id}-{ex_idx}"
    fname = os.path.basename(ex["image"])

    flags_html = "".join(f"<li>{escape(f)}</li>" for f in gemini_flags) or "<li><i>(no flags raised by Gemini)</i></li>"
    obs_html   = "".join(f"<li>{escape(o)}</li>" for o in gemini_obs) or "<li><i>(none)</i></li>"

    has_mask = mask_b64 is not None
    controls = f"""
      <button class="active" onclick="setView(this, '{viewer_id}', 'image')">Image</button>
      <button onclick="setView(this, '{viewer_id}', 'overlay')" {"" if has_mask else "disabled"}>Overlay mask</button>
      <button onclick="setView(this, '{viewer_id}', 'mask')" {"" if has_mask else "disabled"}>Mask only</button>
    """

    return f"""
<div class="example">
  <div class="viewer">
    <div id="{viewer_id}" class="canvas">
      <img class="base" src="{image_b64}"/>
      {f'<img class="mask" src="{mask_b64}"/>' if mask_b64 else ''}
    </div>
    <div class="controls">{controls}</div>
    <div class="filename">{escape(fname)}</div>
  </div>
  <div class="details">
    <h3>Audit agent's claim</h3>
    <div class="claim">{escape(ex['agent_claim'])}</div>

    <div class="meta">
      <span class="tag">product: {escape(product)}</span>
      <span class="tag">gt: {escape(gt_answer)}</span>
      {f'<span class="tag">type: {escape(gt_type)}</span>' if gt_type else ''}
      {f'<span class="tag">loc: {escape(gt_location)}</span>' if gt_location else ''}
      <span class="tag">Gemini: {gemini_score} {escape(gemini_label)}</span>
    </div>

    <details>
      <summary><b>Model's reasoning trace</b></summary>
      <div class="trace">{escape(chosen_trace)}</div>
    </details>
    <details open>
      <summary><b>Gemini's original verdict</b></summary>
      <div class="gemini">
        <b>Rationale:</b> {escape(gemini_rat)}<br><br>
        <b>Hallucination flags:</b><ul>{flags_html}</ul>
        <b>Grounded observations:</b><ul>{obs_html}</ul>
      </div>
    </details>

    <div class="verdict" id="verdict-{verdict_id}">
      <b>Your verdict:</b>
      <button data-choice="agree"  onclick="setVerdict(this, '{verdict_id}', 'agree')">✓ Agree with audit</button>
      <button data-choice="dis"    onclick="setVerdict(this, '{verdict_id}', 'dis')">✗ Disagree</button>
      <button data-choice="brd"    onclick="setVerdict(this, '{verdict_id}', 'brd')">~ Borderline</button>
    </div>
  </div>
</div>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge_report", default="/bulk/aacudad/reasoning_traces/Training/phase0_pilot_20260527_234244/judge_report.jsonl")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    # Load judge records keyed by image_path
    judge_records = {}
    with open(args.judge_report) as f:
        for line in f:
            r = json.loads(line)
            if r.get("success"):
                judge_records[r["image_path"]] = r

    pattern_blocks = []
    for pat in PATTERNS:
        examples_html = "\n".join(
            render_example(pat["id"], i, ex, judge_records)
            for i, ex in enumerate(pat["examples"])
        )
        pattern_blocks.append(f"""
<div class="pattern">
  <h2>Pattern {pat['id']}: {escape(pat['name'])}</h2>
  <div class="fix"><b>Proposed JUDGE_SYSTEM fix:</b><br>{escape(pat['fix'])}</div>
  {examples_html}
</div>
""")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Judge Lenience Audit</title>
<style>{CSS}</style></head><body>

<div class="intro">
<h1>Judge Lenience Audit — 7 patterns to verify</h1>
<p>For each pattern below, an example item is shown where our sub-agents claimed Gemini was too lenient. <b>Click "Overlay mask" to see the GT defect region superimposed in red.</b> Click "Mask only" to see just the mask.</p>
<p>After reviewing each example, mark whether you <b>Agree</b> with the audit, <b>Disagree</b>, or find it <b>Borderline</b>. Your verdicts persist locally (refresh-proof).</p>
<p><b>Total items:</b> 7 patterns, {sum(len(p['examples']) for p in PATTERNS)} concrete examples.</p>
</div>

{''.join(pattern_blocks)}

<script>{JS}</script>
</body></html>
"""

    with open(args.output, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Wrote {args.output}  ({size_mb:.2f} MB, {sum(len(p['examples']) for p in PATTERNS)} examples across {len(PATTERNS)} patterns)")


if __name__ == "__main__":
    main()
