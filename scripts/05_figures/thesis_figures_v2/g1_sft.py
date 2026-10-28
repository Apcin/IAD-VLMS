import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
# measured from the per-sample eval files, DS-MVTec / VisA per epoch
EP={"7B frozen":  [(71.00,65.45),(65.40,64.32),(71.66,64.28),(72.60,66.94)],
    "7B unfrozen":[(70.46,57.13),(68.90,59.18),(72.08,58.60),(70.27,58.16)],
    "3B frozen":  [(69.56,59.65),(66.74,58.42),(69.10,63.30),(68.65,64.85)],
    "3B unfrozen":[(56.61,53.32),(62.16,61.52),(62.45,60.73),(68.58,59.60)]}
BASE={"7B":69.08,"3B":56.14}
COL={"7B frozen":C["kcr"],"7B unfrozen":C["sft"],"3B frozen":C["grpo"],"3B unfrozen":C["base"]}
DASH={"7B frozen":None,"7B unfrozen":"6 4","3B frozen":None,"3B unfrozen":"6 4"}
GRID=[("3B","Frozen","6K",69.08,57.22),("3B","Frozen","15K",69.56,59.65),("3B","Unfrozen","15K",68.58,59.60),
      ("7B","Frozen","15K",72.60,66.94),("7B","Frozen","6K",80.16,64.78),("7B","Unfrozen","15K",72.08,58.60)]
W=980; PW,PH=900,368; PY=100; GAP=20
H=PY+2*PH+GAP+118
s=[txt(W/2,42,"The four-factor SFT ablation","h"),
   txt(W/2,72,"Freezing the vision encoder wins, and the curated 6K split beats the 15K union at 7B.","sub mut")]
# --- panel 1: per epoch on the matched 15K cells ---
x0=(W-PW)/2; y0=PY
s.append(panel(x0,y0,PW,PH,C["rule2"],rx=26,sw=2))
s.append(txt(x0+26,y0+38,"Balanced accuracy by epoch, 15K cells","hl"))
s.append(txt(x0+PW-26,y0+38,"DS-MVTec","t15 mut e"))
A=Axes(x0+80,y0+80,PW-300,PH-160,0.6,4.4,54,75)
s.append(A.frame(range(56,75,4),[1,2,3,4],xfmt=lambda v:f"epoch {v}",ylabel="balanced accuracy %"))
for k,v in EP.items():
    pts=[(i+1,d) for i,(d,_) in enumerate(v)]
    s.append(A.poly(pts,COL[k],3,DASH[k])); s.append(A.marks(pts,COL[k]))
for sz,col in (("7B",C["kcr"]),("3B",C["grpo"])):
    s.append(A.hline(BASE[sz],col,f"{sz} base {BASE[sz]:.2f}",dash="2 4"))
s.append(legend_row(x0+40,y0+PH-24,[("7B frozen",C["kcr"]),("7B unfrozen",C["sft"]),
                                    ("3B frozen",C["grpo"]),("3B unfrozen",C["base"])],gapx=218,
                    dashes={"7B unfrozen":"6 4","3B unfrozen":"6 4"}))

# --- panel 2: best epoch per cell ---
y1=PY+PH+GAP
s.append(panel(x0,y1,PW,PH,C["rule2"],rx=26,sw=2))
s.append(txt(x0+26,y1+38,"Best epoch per configuration","hl"))
B=Axes(x0+80,y1+76,PW-130,PH-166,0,1,50,84)
s.append(B.frame(range(50,85,5),ylabel="balanced accuracy %"))
s.append(B.bars([g[0] for g in GRID],
    [("DS-MVTec",C["purple"],C["purple"],[g[3] for g in GRID]),
     ("VisA",C["purple"],C["purple_f"],[g[4] for g in GRID])],
    bw=40,labels=[f"{g[1]} {g[2]}" for g in GRID]))
hx=B.X(0)+ (B.w/6)*4.5
s.append(rect(B.X(0)+(B.w/6)*4+4,B.y-4,B.w/6-8,B.h+8,fill="none",stroke=C["purple"],sw=2,rx=6,extra='stroke-dasharray="5 4"'))
s.append(txt(B.X(0)+(B.w/6)*4.5,B.y-14,"headline","t13 b c",fill=C["purple"]))
s.append(legend_row(x0+40,y1+PH-14,[("DS-MVTec",C["purple"]),("VisA",C["purple_f"])],gapx=200,swatch="box"))
s.append(txt(x0+PW-26,y1+PH-14,"bars start at 50, which is chance for balanced accuracy","t13 mut e"))
s.append(txt(40,H-38,"Top: every epoch of the four matched 15K cells, against each backbone's zero-shot baseline.","t13 mut"))
s.append(txt(40,H-18,"Bottom: the DS-MVTec-best epoch of each of the six populated cells, with VisA reported at that same checkpoint.","t13 mut"))
open('fig_sft.svg','w').write(head(W,H,"The four-factor SFT ablation")+"".join(s)+foot())
print("  fig_sft.svg H =",H)
