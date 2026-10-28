import json,sys,os,math
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
D=json.load(open('figdata.json')); A=D["arms"]
ARMC={"A, rejection sampling":C["base"],"B, STaR":"#8a5b00","C, KCR":C["kcr"]}
# arm A's line colour is deliberately light, so its value label uses a darker tone
LABC={"A, rejection sampling":"#5d6b74","B, STaR":"#8a5b00","C, KCR":C["kcr"]}
W=980; PW,PH=900,346; PY=100; GAP=20
H=PY+2*PH+GAP+108
s=[head(W,H,"Teacher-intervention arms, balanced accuracy by epoch")]
s.append(txt(W/2,42,"How much teacher help do the rollouts need?","h"))
s.append(txt(W/2,72,"The same images each time. Only the level of teacher intervention changes.","sub mut"))
for idx,(name,lo,hi) in enumerate((("DS-MVTec",74.5,83.5),("VisA",63.0,73.5))):
    x0=(W-PW)/2; y0=PY+idx*(PH+GAP)
    s.append(panel(x0,y0,PW,PH,C["rule2"],rx=26,sw=2))
    s.append(txt(x0+26,y0+38,name,"hl"))
    s.append(txt(x0+26,y0+60,"balanced accuracy %","t13 mut"))
    L,R=x0+74,x0+PW-236; T,B=y0+80,y0+PH-46
    pw,ph=R-L,B-T
    y=lambda v: T+(hi-v)/(hi-lo)*ph
    xs=lambda e: L+(e-1)/3*pw
    for v in [v for v in range(math.ceil(lo),int(hi)+1) if v%2==0]:
        s.append(line(L,y(v),R,y(v),C["rule"],1)); s.append(txt(L-10,y(v)+5,str(v),"t13 mut e"))
    for e in range(1,5):
        s.append(line(xs(e),T-6,xs(e),B,C["rule"],1)); s.append(txt(xs(e),B+24,f"epoch {e}","t15 c"))
    for lab,vals,col in (("SFT headline",D["refs"]["SFT headline"],C["purple"]),
                         ("SFT+GRPO headline",D["refs"]["SFT+GRPO headline"],C["grpo"])):
        v=vals[idx]
        if lo<v<hi:
            s.append(line(L,y(v),R+16,y(v),col,1.6,dash="7 5"))
            s.append(txt(R+22,y(v)+5,f"{lab}  {v:.2f}","t13",fill=col))
    lbls=[]
    for arm,col in ARMC.items():
        pts=[(xs(e+1),y(A[arm]["ep"][e][idx])) for e in range(4)]
        s.append("<polyline points=\""+" ".join(f"{a:.1f},{b:.1f}" for a,b in pts)+
                 f"\" fill=\"none\" stroke=\"{col}\" stroke-width=\"3\"/>")
        best=max(range(4),key=lambda e:A[arm]["ep"][e][idx])
        for e,(a,b) in enumerate(pts):
            s.append(circ(a,b,6 if e==best else 4.5,col,stroke="#fff",sw=2))
        lbls.append((pts[best],col,arm,A[arm]["ep"][best][idx]))
    for (a,b),col,arm,val in lbls:                      # labels last, on top of every line
        up = arm=="C, KCR" or b>(T+B)/2
        ly2=b-20 if up else b+30
        s.append(circ(a,b,10,"none",stroke=col,sw=2))
        s.append(rect(a-36,ly2-19,72,30,fill="#fff",rx=4))
        s.append(txt(a,ly2,f"{val:.2f}","t13 b c",fill=LABC[arm]))

lx=40
for arm,col in ARMC.items():
    s.append(line(lx,H-76,lx+26,H-76,col,3)); s.append(circ(lx+13,H-76,5.5,col))
    s.append(txt(lx+34,H-71,f"{arm}  ({A[arm]['n']:,})","t15")); lx+=290
s.append(txt(40,H-42,"The ringed marker is each arm's best epoch. Dashed lines are the two established headlines of this chapter.","t13 mut"))
s.append(txt(40,H-20,"Arm C is the only arm that clears both headlines, and it does so at epoch 2 on both benchmarks.","t13 mut"))
s.append(foot()); open('fig_arms.svg','w').write("".join(s)); print("fig_arms ok")
