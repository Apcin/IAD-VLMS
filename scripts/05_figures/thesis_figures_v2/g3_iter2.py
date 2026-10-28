import sys,os,json; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
D=json.load(open('iter2_perprod.json')); rows=sorted(D["rows"],key=lambda r:r[2]-r[1])
ova,ovb=D["overall"]
W=980; PW,PH=440,560; PY=100; GAPX=36
H=PY+PH+112
s=[txt(W/2,42,"Refinement on top of the RL policy regresses","h"),
   txt(W/2,72,"The cleaned corpus continued from the GRPO checkpoint, per product on DS-MVTec.","sub mut")]
# left: diverging deltas
x0=24
s.append(panel(x0,PY,PW,PH,C["rule2"],rx=26,sw=2))
s.append(txt(x0+18,PY+36,"Change per product","t18 b"))
L=x0+112; RB=x0+PW-72; mid=(L+RB)/2; sc=(RB-L)/2/7.0
s.append(txt(x0+PW-16,PY+56,"change","t13 mut e"))
for g in (-6,-3,0,3,6):
    gx=mid+g*sc
    s.append(line(gx,PY+62,gx,PY+PH-100,C["rule"] if g else C["rule2"],1 if g else 1.5))
    s.append(txt(gx,PY+56,f"{g:+d}" if g else "0","t13 mut c"))
yy=PY+80
for p,a,b in rows:
    d=b-a; col=C["good"] if d>0.05 else (C["bad"] if d<-0.05 else C["mut"])
    s.append(txt(x0+104,yy+11,p,"t13 e"))
    w=abs(d)*sc
    s.append(rect(mid if d>=0 else mid-w, yy, max(w,1.2), 15, fill=col, rx=2))
    s.append(txt(x0+PW-16, yy+12, f"{d:+.1f}", "t13 b e" if abs(d)>3 else "t13 e", fill=col))
    yy+=24
s.append(line(x0+16,yy+4,x0+PW-16,yy+4,C["rule2"],1))
s.append(txt(x0+104,yy+26,"all products","t13 b e"))
dw=abs(ovb-ova)*sc
s.append(rect(mid-dw,yy+16,max(dw,1.2),15,fill=C["bad"],rx=2))
s.append(txt(x0+PW-16,yy+28,f"{ovb-ova:+.2f}","t15 b e",fill=C["bad"]))
# right: the cause
x1=x0+PW+GAPX+32
s.append(panel(x1,PY,PW,PH,C["rule2"],rx=26,sw=2))
s.append(txt(x1+18,PY+36,"Why","t18 b"))
box=[("Kept items",  "3,557 of the model's own rollouts", "near-zero loss, almost no gradient", C["kcr_f"], C["kcr"]),
     ("Patched items","2,443 teacher-written traces",     "94.8 % anomalous, high loss",        C["sft_f"], "#8a5b00"),
     ("Net effect",  "the corpus is balanced 50/50",      "the gradient is likely about 95 % anomalous",C["base_f"],C["bad"])]
by=PY+66
for t,a,b,fil,col in box:
    s.append(rect(x1+22,by,PW-44,96,fill=fil,rx=10))
    s.append(txt(x1+38,by+28,t,"t16 b"))
    s.append(txt(x1+38,by+52,a,"t13"))
    s.append(txt(x1+38,by+74,b,"t13 b",fill=col))
    by+=106
s.append(txt(x1+22,by+10,"Anomaly recall therefore rises each epoch while","t15"))
s.append(txt(x1+22,by+32,"normal recall falls. Every regression above is on a","t15"))
s.append(txt(x1+22,by+54,"product whose normal class is subtle.","t15"))
s.append(rect(x1+22,by+70,PW-44,68,fill=C["kcr_f"],rx=10))
s.append(txt(x1+38,by+96,"The same corpus trained from base","t15 b"))
s.append(txt(x1+38,by+120,f"reaches 82.80 %, the best model in this thesis.","t15 b",fill=C["kcr"]))
s.append(txt(24,H-40,f"Baseline is the production GRPO checkpoint at {ova:.2f} %. The refinement run is epoch 1 of the cleaned 6,000-item pool at {ovb:.2f} %.","t13 mut"))
s.append(txt(24,H-18,"Five products improve, eight regress and two are unchanged.","t13 mut"))
open('fig_iter2.svg','w').write(head(W,H,"Refinement on top of the RL policy regresses")+"".join(s)+foot())
print("  fig_iter2.svg H =",H)
