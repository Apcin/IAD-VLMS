#!/usr/bin/env python3
"""Generate the GRPO-on-Arm-C decoupling figure for thesis §6.13:
   (left) balanced accuracy vs step for 3 estimators, all below the SFT init;
   (right) mean training reward vs step, rising — i.e. reward/accuracy decoupling.
Outputs figures/grpo_estimator_decoupling.{png,pdf}
"""
import json, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/thesis_review/thesis_tud/figures"
RUNS = "/bulk/aacudad/reasoning_traces/outputs"
ARMS = [("ctrl","CTRL ($z$-score)","#0C2D48","o"),
        ("drgrpo","Dr.GRPO (no std)","#00A6D6","s"),
        ("g2rpo","G$^2$RPO (rank)","#E8A33D","^")]
INIT_DS, INIT_VA = 84.52, 71.89
STEPS = [20,40,60,80,100,120]

def bal(p):
    if not os.path.exists(p): return None
    try: m=json.load(open(p))["metrics"]
    except: return None
    tp,tn,fp,fn=m["tp"],m["tn"],m["fp"],m["fn"]
    return 100*((tp/(tp+fn))+(tn/(tn+fp)))/2 if (tp+fn and tn+fp) else None

def acc_series(arm, which):
    xs, ys = [], []
    f = "probe_dsmvtec.json" if which=="ds" else "probe_visa.json"
    for s in STEPS:
        v = bal(f"{RUNS}/grpo_probe_{arm}/checkpoint-{s}/{f}")
        if v is not None: xs.append(s); ys.append(v)
    return xs, ys

def reward_series(arm):
    p=f"{RUNS}/grpo_probe_{arm}/train.log"
    if not os.path.exists(p): return [],[]
    txt=open(p).read().replace("\r","\n")
    vals=[float(m.group(1)) for l in txt.split("\n")
          if "'reward':" in l for m in [re.search(r"'reward': '([0-9.eE+-]+)'",l)] if m]
    # smooth with trailing window of 5
    xs=list(range(1,len(vals)+1))
    sm=[sum(vals[max(0,i-4):i+1])/len(vals[max(0,i-4):i+1]) for i in range(len(vals))]
    return xs, sm

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.grid":True,
                     "grid.alpha":0.3,"axes.spines.top":False,"axes.spines.right":False})
fig,(axA,axB)=plt.subplots(1,2,figsize=(11,4.2))

# Panel A: balanced accuracy vs step (DS-MVTec solid, VisA dashed) + init lines
axA.axhline(INIT_DS,color="#0C2D48",ls=":",lw=1.5,alpha=.7)
axA.text(122,INIT_DS+0.15,"SFT init (DS-MVTec 84.5)",fontsize=8.5,color="#0C2D48",ha="right",va="bottom")
for arm,label,c,mk in ARMS:
    xs,ys=acc_series(arm,"ds")
    if xs: axA.plot(xs,ys,marker=mk,color=c,lw=2,ms=6,label=label)
axA.set_xlabel("GRPO step"); axA.set_ylabel("DS-MVTec balanced accuracy (\\%)")
axA.set_title("(a) Accuracy stays below the SFT init",fontsize=11,loc="left")
axA.set_xlim(0,128); axA.set_ylim(80.5,85.2); axA.legend(fontsize=8.5,loc="lower left")

# Panel B: mean training reward vs step (rising)
for arm,label,c,mk in ARMS:
    xs,ys=reward_series(arm)
    if xs: axB.plot(xs,ys,color=c,lw=2,label=label)
axB.set_xlabel("GRPO step"); axB.set_ylabel("Mean group reward (5-step MA)")
axB.set_title("(b) Training reward rises",fontsize=11,loc="left")
axB.set_xlim(0,122); axB.legend(fontsize=8.5,loc="lower right")

fig.suptitle("GRPO on the strong Arm-C SFT init: reward rises while accuracy does not (all three estimators)",
             fontsize=11.5,y=1.02)
fig.tight_layout()
for ext in ("png","pdf"):
    fig.savefig(f"{OUT}/grpo_estimator_decoupling.{ext}",dpi=200,bbox_inches="tight")
print("wrote", f"{OUT}/grpo_estimator_decoupling.png/.pdf")
