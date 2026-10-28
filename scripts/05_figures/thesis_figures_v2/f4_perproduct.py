# superseded 7 Sep 2026: Figure 6.7 is now the heat map drawn by alt_figs.alt_perproduct (Adnane's pick); this file is kept for the dot-ladder version.
import json,sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
D=json.load(open('figdata.json'))["perproduct"]
BB=["Qwen2.5-VL-7B","LLaVA-OneVision-7B"]; ST=["SFT","SFT+GRPO","KCR"]
W=980; PW=448; GAPX=36; PY=104; GAPY=22
RH=28
def panel_h(nrows): return 92+nrows*RH+46
H=PY+panel_h(15)+GAPY+panel_h(12)+112
s=[head(W,H,"Per-product balanced accuracy for the three trained stages")]
s.append(txt(W/2,42,"Where the gain actually comes from","h"))
s.append(txt(W/2,72,"Balanced accuracy per product. The gain is concentrated on a few products, not spread evenly.","sub mut"))
lo,hi=45,100
yoff=PY
for bench in ("DS-MVTec","VisA"):
    nrows=len(D[bench][BB[0]]["rows"])
    ph=panel_h(nrows)
    for pi,bb in enumerate(BB):
        x0=24+pi*(PW+GAPX)
        s.append(panel(x0,yoff,PW,ph,C["rule2"],rx=26,sw=2))
        s.append(txt(x0+16,yoff+34,bb,"hl" if False else "t18 b"))
        s.append(txt(x0+PW-12,yoff+34,f"{bench},  n={D[bench][bb]['n']:,}","t13 mut e"))
        LB=x0+112; RB=x0+PW-58
        X=lambda v: LB+(v-lo)/(hi-lo)*(RB-LB)
        for g in (50,60,70,80,90,100):
            s.append(line(X(g),yoff+72,X(g),yoff+ph-40,C["rule"],1))
            if g<90: s.append(txt(X(g),yoff+66,str(g),"t13 mut c"))
        rows=D[bench][bb]["rows"]; ov=D[bench][bb]["overall"]
        yy=yoff+86
        for r in rows+[dict(product="all products",**ov)]:
            is_tot = r["product"]=="all products"
            cy=yy+RH/2
            if is_tot:
                s.append(line(x0+12,yy-2,x0+PW-12,yy-2,C["rule2"],1))
                cy+=6
            vals=[r[st] for st in ST]
            s.append(txt(x0+104,cy+6,r["product"],"t15 b e" if is_tot else "t15 e"))
            s.append(line(X(min(vals)),cy,X(max(vals)),cy,C["rule2"],1.5))
            for st in ST:
                ck,_=STAGE_C[st]
                s.append(circ(X(r[st]),cy,5.5 if st=="KCR" else 4.5,C[ck],stroke="#fff",sw=1.5))
            d=r["KCR"]-r["SFT"]
            col=C["good"] if d>0.05 else (C["bad"] if d<-0.05 else C["mut"])
            s.append(txt(x0+PW-12,cy+6,f"{d:+.1f}","t15 b e" if is_tot else "t15 e",fill=col))
            yy+=RH+(6 if is_tot else 0)
        s.append(txt(x0+PW-12,yoff+66,"KCR minus SFT","t13 mut e"))
    yoff+=ph+GAPY
lx=24
for st in ST:
    ck,_=STAGE_C[st]
    s.append(circ(lx+8,H-84,6,C[ck])); s.append(txt(lx+22,H-79,st,"t15")); lx+=190
s.append(txt(24,H-54,"Each row is one product. The three dots are the three trained stages and the grey line spans them.","t13 mut"))
s.append(txt(24,H-32,"The right-hand column is the KCR gain over SFT in percentage points, green when positive.","t13 mut"))
s.append(txt(24,H-10,"KCR wins overall on all four panels, but it loses on individual products, which the overall number hides.","t13 mut"))
s.append(foot()); open('fig_perproduct.svg','w').write("".join(s)); print("ok")
