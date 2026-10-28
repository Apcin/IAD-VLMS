import sys,os,json; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
K=json.load(open('grpo_curve.json'))
def smooth(idx,w=25):
    v=[(k[0],k[idx]) for k in K if k[idx] is not None]
    return [(x,sum(b for _,b in v[max(0,i-w+1):i+1])/len(v[max(0,i-w+1):i+1])) for i,(x,_) in enumerate(v)]
def raw(idx): return [(k[0],k[idx]) for k in K if k[idx] is not None]
SFT=[(1,79.24),(2,76.66),(3,80.16),(4,77.07)]          # 6K frozen 7B, per epoch, DS-MVTec
W=980; PW,PH=440,330; PY=100; GX=36; GY=20
H=PY+2*PH+GY+146
s=[txt(W/2,42,"Training and reward dynamics","h"),
   txt(W/2,72,"The production GRPO run on the 6K-frozen SFT checkpoint, 1,060 logged steps.","sub mut")]
def pan(px,py,title,sub=None):
    o=[panel(px,py,PW,PH,C["rule2"],rx=26,sw=2), txt(px+18,py+34,title,"t18 b")]
    if sub: o.append(txt(px+PW-18,py+34,sub,"t13 mut e"))
    return "".join(o)
# (a) total reward and its two components
x0,y0=24,PY
s.append(pan(x0,y0,"(a) Reward","dashed line: max 3.0"))
A=Axes(x0+62,y0+66,PW-96,PH-140,0,1080,0.4,3.05)
s.append(A.frame([0.5,1.0,1.5,2.0,2.5,3.0],[0,265,530,795,1060],yfmt=lambda v:f"{v:.1f}",ylabel="reward",xlabel="training step"))
s.append(A.hline(3.0,C["bad"],None,dash="4 4"))
s.append(A.poly(raw(1),C["kcr"],1,op=0.22)); s.append(A.poly(smooth(1),C["kcr"],2.6))
s.append(A.poly(smooth(3),C["grpo"],2)); s.append(A.poly(smooth(4),C["sft"],2))

# (b) KL
x1=x0+PW+GX+32
s.append(pan(x1,y0,"(b) KL to the reference policy","max 0.197"))
B=Axes(x1+62,y0+66,PW-96,PH-140,0,1080,0,0.22)
s.append(B.frame([0,0.05,0.10,0.15,0.20],[0,265,530,795,1060],yfmt=lambda v:f"{v:.2f}",ylabel="KL divergence",xlabel="training step"))
s.append(B.poly(raw(2),C["purple"],1,op=0.22)); s.append(B.poly(smooth(2),C["purple"],2.6))
s.append(txt(B.x+10,B.Y(0.175)+3,"bounded throughout, no spike or collapse","t13 mut"))
# (c) completion length
y1=y0+PH+GY
s.append(pan(x0,y1,"(c) Completion length","tokens"))
Cx=Axes(x0+62,y1+66,PW-96,PH-140,0,1080,145,195)
s.append(Cx.frame([150,160,170,180,190],[0,265,530,795,1060],ylabel="completion length, tokens",xlabel="training step"))
s.append(Cx.poly(raw(5),C["mut"],1,op=0.22)); s.append(Cx.poly(smooth(5),C["mut"],2.6))
s.append(txt(Cx.x+10,Cx.Y(186)+3,"negligible length collapse","t13 mut"))
# (d) SFT validation accuracy by epoch
s.append(pan(x1,y1,"(d) SFT accuracy by epoch","DS-MVTec"))
Dx=Axes(x1+62,y1+66,PW-96,PH-140,0.6,4.4,75,82)
s.append(Dx.frame([76,78,80,82],[1,2,3,4],xfmt=lambda v:f"epoch {v}",ylabel="balanced accuracy %",xlabel="supervised epoch"))
s.append(Dx.poly(SFT,C["mut"],3)); s.append(Dx.marks(SFT,C["mut"]))
best=max(SFT,key=lambda t:t[1])
s.append(circ(Dx.X(best[0]),Dx.Y(best[1]),10,"none",stroke=C["kcr"],sw=2.5))
s.append(rect(Dx.X(best[0])-26,Dx.Y(best[1])-32,52,18,fill="#fff",rx=3))
s.append(txt(Dx.X(best[0]),Dx.Y(best[1])-19,f"{best[1]:.2f}","t13 b c",fill=C["kcr"]))
s.append(txt(Dx.x+8,Dx.Y(75.6)+3,"selected checkpoint, epoch 3","t13 mut"))
# legend for (a) as a header row under the panel title, outside the plot area
lx=x0+PW-18-301; lyy=y0+52
for lab,col,wd,dash in (("total",C["kcr"],34,None),("accuracy",C["grpo"],56,None),("format",C["sft"],44,None)):
    s.append(line(lx,lyy-5,lx+22,lyy-5,col,3 if not dash else 1.5,dash=dash)); s.append(txt(lx+27,lyy,lab,"t13")); lx+=22+27+wd+10
s.append(txt(24,H-70,"Faint lines are the raw per-step values, solid lines a 25-step trailing mean.","t13 mut"))
s.append(txt(24,H-48,"The reward saturates near 2.1 while the KL stays bounded, which is the standard healthy profile.","t13 mut"))
s.append(txt(24,H-26,"Training past epoch 1 of GRPO costs 0.70 points on DS-MVTec, which is why ckpt-530 is the reported checkpoint.","t13 mut"))
open('fig_dynamics.svg','w').write(head(W,H,"Training and reward dynamics")+"".join(s)+foot())
print("  fig_dynamics.svg H =",H)
