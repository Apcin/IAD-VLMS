#!/usr/bin/env python
"""
Build the targeted Variety + Arm-C rehearsal mix for the continuation-SFT experiment.
  1,000 Real-IAD-Variety traces from EVAL-ALIGNED categories (electronics/connectors/
        components/switches/terminals + small metal hardware + zippers/buckles;
        toys/decor/ceramic/fabric excluded), 50/50 NG/OK, stratified across categories.
  2,000 Arm-C traces (the curated SFT corpus), 50/50 NG/OK, stratified across 30 products
        -> rehearsal anchor to prevent forgetting the 30-product distribution.
Total 3,000, shuffled (seed 42), ShareGPT format. -> variety_armC_mix_3k.json
"""
import json, re, random
random.seed(42)
ROOT = "/bulk/aacudad/reasoning_traces"
VAR = f"{ROOT}/Training/datasets_variety/variety_star_sft_6k.json"
ARM = f"{ROOT}/Training/datasets_sft_iter2/sft_iter2_train.json"   # = the Arm-C corpus
OUT = f"{ROOT}/Training/datasets_variety/variety_armC_mix_3k.json"

# Variety categories NOT relevant to MVTec/VisA (toys/models/decor/ceramic/nature/fabric) -> excluded.
EXCLUDE = {
 "lego_pin_connector_plate","lego_propeller","lego_reel","lego_technical_gear","lego_turbine",
 "slipper_model","bread_model","little_cow_model","jam_jar_model","aircraft_model_head","toy_tire",
 "meteor_hammer_arrowhead","purple_clay_pot","spherical_airstone","pencil_sharpener","suction_cup",
 "flower_copper_shape","flower_velvet_fabric","small_leaf","insect_metal_parts",
 "brooch_clasp_accessory","traceless_hair_clip",
}

def img_of(ex):  return ex["images"][0] if isinstance(ex["images"], list) else ex["images"]
def is_ng(ex):
    img, a = img_of(ex), ex["messages"][1]["content"]
    return ("/NG/" in img) or ("_NG_" in img) or ("<answer>Yes" in a)
def var_cat(ex):
    m = re.search(r"realiad-variety/([^/]+)/", img_of(ex)); return m.group(1) if m else "UNK"
def arm_prod(ex):
    m = re.search(r"/images/([^/]+)/", img_of(ex)); return m.group(1) if m else "UNK"

def roundrobin(groups, n):
    """Pick n items evenly across keys (stratified), shuffled within each key."""
    keys = sorted(groups)
    for k in keys: random.shuffle(groups[k])
    out, idx = [], {k: 0 for k in keys}
    while len(out) < n:
        progressed = False
        for k in keys:
            if len(out) >= n: break
            if idx[k] < len(groups[k]):
                out.append(groups[k][idx[k]]); idx[k] += 1; progressed = True
        if not progressed: break
    return out

# --- Variety: 500 NG + 500 OK from kept categories, stratified ---
var = json.load(open(VAR))
kept = [e for e in var if var_cat(e) not in EXCLUDE]
ng_g, ok_g = {}, {}
for e in kept:
    (ng_g if is_ng(e) else ok_g).setdefault(var_cat(e), []).append(e)
var_ng, var_ok = roundrobin(ng_g, 500), roundrobin(ok_g, 500)
var_sel = var_ng + var_ok
used_cats = sorted(set(var_cat(e) for e in var_sel))
print(f"[variety] kept categories={len(set(list(ng_g)+list(ok_g)))}  selected={len(var_sel)} (NG {len(var_ng)} / OK {len(var_ok)})  categories-used={len(used_cats)}")
print(f"[variety] excluded {len(EXCLUDE)} categories: {sorted(EXCLUDE)[:6]}...")

# --- Arm-C: 1000 NG + 1000 OK, stratified across products ---
arm = json.load(open(ARM))
ang, aok = {}, {}
for e in arm:
    (ang if is_ng(e) else aok).setdefault(arm_prod(e), []).append(e)
arm_ng, arm_ok = roundrobin(ang, 1000), roundrobin(aok, 1000)
arm_sel = arm_ng + arm_ok
print(f"[arm-C] products={len(set(list(ang)+list(aok)))}  selected={len(arm_sel)} (NG {len(arm_ng)} / OK {len(arm_ok)})")

mix = [{"messages": e["messages"], "images": e["images"]} for e in (var_sel + arm_sel)]
random.shuffle(mix)
json.dump(mix, open(OUT, "w"))
ng_total = sum(1 for e in mix if is_ng(e))
print(f"[mix] TOTAL {len(mix)}  NG/OK = {ng_total}/{len(mix)-ng_total}  -> {OUT}")
print(f"[mix] variety categories in mix: {used_cats}")
