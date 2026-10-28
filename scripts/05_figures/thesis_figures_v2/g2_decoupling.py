import sys,os,json,re,glob; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
R='/bulk/aacudad/reasoning_traces/outputs'
ARMS=[("ctrl","CTRL, group z-score",C["grpo"]),("drgrpo","Dr.GRPO, no std",C["sft"]),("g2rpo","G2RPO, rank",C["kcr"])]
STEPS=[20,40,60,80,100,120]; INIT_DS,INIT_VA=84.52,71.89
def ba(p):
    try: m=json.load(open(p))['metrics']
    except Exception: return None
    tp,tn,fp,fn=(m.get(k,0) for k in ('tp','tn','fp','fn'))
    if not(tp+fn) or not(tn+fp): return None
    return 50*(tp/(tp+fn)+tn/(tn+fp))
def rew(arm):
    p=f"{R}/grpo_probe_{arm}/train.log"
    if not os.path.exists(p): return []
    t=open(p,errors='ignore').read().replace("\r","\n")
    v=[float(m.group(1)) for l in t.split("\n") if "'reward':" in l
       for m in [re.search(r"'reward': '?([0-9.eE+-]+)",l)] if m]
    return [(i+1,sum(v[max(0,i-4):i+1])/len(v[max(0,i-4):i+1])) for i in range(len(v))]
W=980; PW,PH=440,470; PY=100; GAPX=20
H=PY+PH+118
s=[txt(W/2,42,"Reward rises while accuracy does not","h"),
   txt(W/2,72,"Three advantage estimators, 120 GRPO steps, all from the same KCR checkpoint.","sub mut")]
# panel A: probe accuracy
x0=24
s.append(panel(x0,PY,PW,PH,C["rule2"],rx=26,sw=2))
s.append(txt(x0+18,PY+36,"(a) DS-MVTec probe accuracy","t18 b"))
A=Axes(x0+64,PY+74,PW-100,PH-160,0,126,80.5,86)
s.append(A.frame([81,82,83,84,85,86],STEPS,ylabel="balanced accuracy %",xlabel="GRPO step"))
s.append(A.hline(INIT_DS,C["mut"],None,dash="4 4"))
s.append(rect(A.x+4,A.y+2,214,20,fill="#fff",rx=4))
s.append(txt(A.x+10,A.y+17,f"KCR initialisation {INIT_DS:.2f}","t13",fill=C["mut"]))
for arm,lab,col in ARMS:
    pts=[(st,ba(f"{R}/grpo_probe_{arm}/checkpoint-{st}/probe_dsmvtec.json")) for st in STEPS]
    pts=[(a,b) for a,b in pts if b]
    s.append(A.poly(pts,col,3)); s.append(A.marks(pts,col))
# panel B: training reward
x1=x0+PW+GAPX+32
s.append(panel(x1,PY,PW,PH,C["rule2"],rx=26,sw=2))
s.append(txt(x1+18,PY+36,"(b) Mean group reward","t18 b"))
B=Axes(x1+64,PY+74,PW-100,PH-160,0,126,1.6,2.7)
s.append(B.frame([1.6,1.8,2.0,2.2,2.4,2.6],STEPS,yfmt=lambda v:f"{v:.1f}",ylabel="reward, 5-step mean",xlabel="GRPO step"))
for arm,lab,col in ARMS:
    r=rew(arm)
    if r: s.append(B.poly(r,col,2,op=0.9))
s.append(legend_row(24,H-72,[(l,c) for _,l,c in ARMS],gapx=300))
s.append(txt(24,H-42,"Only one of the eighteen post-initialisation checkpoints rises above the initialisation on DS-MVTec, and it is back below by step 40.","t13 mut"))
s.append(txt(24,H-20,"The training reward rises for all three estimators over the same span, which is the signature of reward and metric decoupling.","t13 mut"))
open('fig_decoupling.svg','w').write(head(W,H,"Reward rises while accuracy does not")+"".join(s)+foot())
print("  fig_decoupling.svg H =",H)
