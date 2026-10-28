"""
Smoke test for the GPT-5-mini trace-verifier:
  (1) confirms the API key works and gpt-5-mini can SEE images (original + red overlay),
  (2) measures real token usage for ONE realistic verification call,
  (3) prints an a-priori token estimate (so cost works even if the call fails on no funds),
  (4) extrapolates a rough cost to the full corpus.
Never prints the API key. Run with the env that has openai+PIL+dotenv:
  /users/aacudad/miniconda3/envs/FINETUNE/bin/python smoke_test_gpt5mini.py
"""
import os, io, math, base64, json, sys
from pathlib import Path
from PIL import Image, ImageDraw

ENV_PATH = "/bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/.env"
MODEL = "gpt-5-mini"
# ---- ASSUMED pricing (USD / 1M tokens) -- VERIFY at platform.openai.com/pricing ----
PRICE_IN_PER_M  = 0.25
PRICE_OUT_PER_M = 2.00

# ---- load key from .env without printing it ----
def load_key():
    for ln in Path(ENV_PATH).read_text().splitlines():
        if ln.strip().startswith("OPENAI_API_KEY"):
            return ln.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("No OPENAI_API_KEY in .env")

# ---- pick a real product image + build red overlay (real mask if present, else synthetic) ----
def get_image_and_overlay():
    root = Path("/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/Real-IAD/images")
    cand = root / "bottle_cap/NG/ZW/S0015/bottle_cap_0015_NG_ZW_C1_20230926112810.jpg"
    if not cand.exists():
        # bounded fallback search for any C1 NG image
        cand = None
        for p in root.rglob("*_NG_*_C1_*.jpg"):
            cand = p; break
    if cand is None or not cand.exists():
        raise SystemExit("No Real-IAD image found for the smoke test")
    img = Image.open(cand).convert("RGB")
    mask_path = cand.with_suffix(".png")
    synthetic = False
    overlay = Image.open(cand).convert("RGBA")
    if mask_path.exists():
        m = Image.open(mask_path).convert("L").resize(img.size, Image.NEAREST)
        red = Image.new("RGBA", img.size, (0, 0, 0, 0))
        px = red.load(); mp = m.load()
        for x in range(img.width):
            for y in range(img.height):
                if mp[x, y] > 127: px[x, y] = (255, 0, 0, 128)
        overlay = Image.alpha_composite(overlay, red)
    else:
        synthetic = True
        d = ImageDraw.Draw(overlay, "RGBA")
        w, h = img.size
        d.ellipse([w*0.55, h*0.35, w*0.8, h*0.6], fill=(255, 0, 0, 110))
    return cand, img, overlay.convert("RGB"), synthetic

def b64(im):
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def est_image_tokens(w, h):
    # gpt-4o-style high-detail tiling (proxy; gpt-5-mini accounting may differ)
    if max(w, h) > 2048:
        s = 2048/max(w, h); w, h = w*s, h*s
    if min(w, h) > 768:
        s = 768/min(w, h); w, h = w*s, h*s
    return 85 + 170*math.ceil(w/512)*math.ceil(h/512)

SYS = ("You verify industrial-defect reasoning traces against the images. For the trace below, "
       "FIRST briefly state (a) what product you see, (b) where the RED-highlighted region is, "
       "(c) what defect it shows -- to confirm you can see both images -- then return JSON "
       '{"product":..., "red_region":..., "defect_seen":..., "status":"completely_correct|correct|wrong|completely_wrong"}.')

SAMPLE_TRACE = ("<think>Looking at this bottle cap. Scanning the rim and top surface. There is an "
                "irregular mark on the right side that breaks the smooth molded surface, consistent "
                "with a surface contamination or scratch. Edges elsewhere look intact.</think>"
                "<location>middle-right</location><type>Contamination</type><answer>Yes</answer>")

def main():
    img_path, img, overlay, synthetic = get_image_and_overlay()
    print(f"Test image: {img_path}")
    print(f"Image size: {img.size}  | overlay: {'SYNTHETIC red ellipse (no mask on disk)' if synthetic else 'REAL GT mask'}")

    # ---- a-priori token estimate for this request ----
    it = est_image_tokens(*img.size)
    est_in = 2*it + len(SYS+SAMPLE_TRACE)//4 + 40
    print(f"\n[a-priori estimate] ~{it} tokens/image x2 + text => ~{est_in} input tokens for this 1-trace call")

    # ---- real call ----
    try:
        from openai import OpenAI
        client = OpenAI(api_key=load_key())
        content = [
            {"type": "input_text", "text": "Verify this trace. Original image then RED-overlay image then the reasoning."},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64(img)}"},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64(overlay)}"},
            {"type": "input_text", "text": "Reasoning:\n"+SAMPLE_TRACE},
        ]
        resp = client.responses.create(
            model=MODEL,
            input=[{"role": "system", "content": SYS}, {"role": "user", "content": content}],
            reasoning={"effort": "low"},
            text={"format": {"type": "json_object"}},
        )
        txt = None
        for it_ in resp.output:
            if getattr(it_, "type", None) == "message":
                for c in it_.content:
                    if getattr(c, "type", None) == "output_text":
                        txt = c.text
        u = resp.usage
        rin = u.input_tokens; rout = u.output_tokens
        rreason = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", None)
        print("\n========== MODEL OUTPUT (proves it saw the images) ==========")
        print(txt)
        print("\n========== TOKEN USAGE (REAL) ==========")
        print(f"input={rin}  output={rout}  reasoning={rreason}")
        call_cost = rin/1e6*PRICE_IN_PER_M + rout/1e6*PRICE_OUT_PER_M
        print(f"this call cost (assumed pricing): ${call_cost:.5f}")
        per_in, per_out = rin, rout
        basis = "REAL measured usage"
    except Exception as e:
        print("\n[!] API call failed (expected if the account is unfunded):")
        print(f"    {type(e).__name__}: {str(e)[:300]}")
        per_in, per_out = est_in, 350   # fallback: estimate + assume ~350 out (incl. low-effort reasoning)
        basis = "A-PRIORI ESTIMATE (call failed)"

    # ---- extrapolate ----
    print(f"\n========== COST EXTRAPOLATION ({basis}) ==========")
    print(f"assumed pricing: ${PRICE_IN_PER_M}/1M in, ${PRICE_OUT_PER_M}/1M out  (VERIFY on OpenAI pricing page)")
    print(f"per-trace tokens used for estimate: in~{per_in}, out~{per_out}")
    print("NOTE: batching 15 traces/request AMORTIZES the system prompt; real per-trace input is dominated by the 2 images.")
    for n, label in [(10236, "C1 corpus (~'15K')"), (15000, "if 15,000 traces"), (5118, "NG-only (2 imgs)")]:
        c = (per_in*n)/1e6*PRICE_IN_PER_M + (per_out*n)/1e6*PRICE_OUT_PER_M
        print(f"  {label:24s} n={n:6d}  ->  ~${c:6.2f}")
    print("\n(OK/normal traces have NO mask => 1 image not 2 => ~40% cheaper than the NG estimate.)")

if __name__ == "__main__":
    main()
