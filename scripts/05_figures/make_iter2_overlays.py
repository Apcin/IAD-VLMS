#!/usr/bin/env python3
"""Regenerate iter2_sample_overlays.png cleanly: 4 representative DS-MVTec cases
comparing run-2 (GRPO) vs SFT-Iter2, GT defect mask overlaid in red, clean labels."""
import json, os
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT="/tmp/thesis_review/thesis_tud/figures/iter2_sample_overlays.png"
RUN2="/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json"
ITER2="/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_iter2_v2_frozen/checkpoint-378/eval_dsmvtec_full_trainprompt.json"

def index(p):
    return {r["image_id"]: r for r in json.load(open(p))["results"]}
b=index(RUN2); it=index(ITER2)
common=[k for k in b if k in it]

def mask_path(absp): return absp.replace("/image/","/mask/").replace(".png","_mask.png")
def has_mask(absp): return os.path.exists(mask_path(absp))

def find(gt, base_ok, iter_ok, product=None, need_mask=False):
    for k in common:
        rb,ri=b[k],it[k]
        if rb["gt_answer"]!=gt: continue
        if (rb["pred_answer"]==gt)!=base_ok: continue
        if (ri["pred_answer"]==gt)!=iter_ok: continue
        if product and rb["product"]!=product: continue
        if need_mask and not has_mask(rb["absolute_path"]): continue
        return rb,ri
    return None,None

# four pedagogical cases
cases=[
 ("both correct (hard product)", find("yes",True,True,need_mask=True)),
 ("OK over-predicted by Iter2",  find("no",True,False)),
 ("both miss (subtle defect)",   find("yes",False,False,need_mask=True)),
 ("Iter2 catches, run-2 misses", find("yes",False,True,need_mask=True)),
]

SZ=360
def render_img(absp):
    img=Image.open(absp).convert("RGB").resize((SZ,SZ))
    mp=mask_path(absp)
    if os.path.exists(mp):
        m=np.array(Image.open(mp).convert("L").resize((SZ,SZ)))>0
        if m.any():
            a=np.array(img).astype(float); red=np.zeros_like(a); red[...,0]=255
            a[m]=0.5*a[m]+0.5*red[m]; img=Image.fromarray(a.astype("uint8"))
    return img

plt.rcParams.update({"font.family":"DejaVu Sans"})
fig,axes=plt.subplots(2,2,figsize=(8.4,9.2))
def tick(ok): return "✓" if ok else "✗"
for ax,(label,(rb,ri)) in zip(axes.flat,cases):
    if rb is None: ax.axis("off"); ax.set_title(label+"\n(no matching sample)",fontsize=9); continue
    ax.imshow(render_img(rb["absolute_path"])); ax.set_xticks([]); ax.set_yticks([])
    gt=rb["gt_answer"]; prod=rb["product"]
    base_ok=rb["pred_answer"]==gt; iter_ok=ri["pred_answer"]==gt
    title=(f"{('NG' if gt=='yes' else 'OK')} · {prod} — {label}\n"
           f"GT: {gt}    run-2 (GRPO): {rb['pred_answer']} {tick(base_ok)}    "
           f"SFT-Iter2: {ri['pred_answer']} {tick(iter_ok)}")
    ax.set_title(title,fontsize=8.5,loc="center")
fig.suptitle("Representative DS-MVTec cases  (ground-truth defect mask overlaid in red)",
             fontsize=11,y=0.995)
fig.tight_layout(rect=[0,0,1,0.985])
fig.savefig(OUT,dpi=170,bbox_inches="tight")
print("wrote",OUT)
for label,(rb,ri) in cases:
    print(f"  {label}: {rb['image_id'] if rb else 'NONE'}")
