import json,sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
D=json.load(open('figdata.json'))
BB=["Qwen2.5-VL-7B","LLaVA-OneVision-7B"]
ST=["Base","SFT","SFT+GRPO","KCR"]
NOTE={"Qwen2.5-VL-7B":"the backbone every controlled study in this chapter uses",
      "LLaVA-OneVision-7B":"the same loop run again, on this backbone's own rollouts"}
W=980; PW,PH=900,404; PY=100; GAP=22
H=PY+2*PH+GAP+96
s=[head(W,H,"Balanced accuracy across the four training stages, two backbones")]
s.append(txt(W/2,42,"One ladder, two backbones","h"))
s.append(txt(W/2,72,"KCR matches SFT+GRPO on both backbones with no reinforcement step, and leads on VisA.","sub mut"))
for pi,bb in enumerate(BB):
    x0=(W-PW)/2; y0=PY+pi*(PH+GAP)
    s.append(panel(x0,y0,PW,PH,C["rule2"],rx=26,sw=2))
    s.append(txt(x0+26,y0+40,bb,"hl"))
    s.append(txt(x0+26,y0+64,NOTE[bb],"t15 mut"))
    L,R=x0+78,x0+PW-26; T,B=y0+96,y0+PH-118
    pw,ph=R-L,B-T; lo,hi=50,92
    y=lambda v: T+(hi-v)/(hi-lo)*ph
    s.append(txt(L-10,T-12,"balanced accuracy %","t13 mut"))
    for v in range(50,91,10):
        s.append(line(L,y(v),R,y(v),C["rule"],1)); s.append(txt(L-10,y(v)+5,str(v),"t13 mut e"))
    s.append(line(L,y(50),R,y(50),C["rule2"],1.5))
    gw=pw/4; bw=54; gap=16
    dspts=[]
    for gi,st in enumerate(ST):
        gx=L+gi*gw; row=D["ladder"][bb][st]; ck,cf=STAGE_C[st]
        bx0=gx+(gw-(2*bw+gap))/2
        for bi,bench in enumerate(("ds","visa")):
            bx=bx0+bi*(bw+gap); v=row[bench]
            if v is None:
                s.append(rect(bx,y(lo)-20,bw,20,fill="none",stroke=C["rule2"],sw=1,rx=2,extra='stroke-dasharray="3 3"'))
                s.append(txt(bx+bw/2,y(lo)-28,"n/a","t13 mut c")); continue
            val=v["ba"]
            s.append(rect(bx,y(val),bw,y(lo)-y(val),fill=C[cf] if bi else C[ck],stroke=C[ck],sw=1.5,rx=2))
            s.append(rect(bx+bw/2-25,y(val)-27,50,18,fill="#fff",rx=3))
            s.append(txt(bx+bw/2,y(val)-14,f"{val:.2f}","t13 b c"))
            if bi==0: dspts.append((bx+bw/2,y(val)))
        s.append(txt(gx+gw/2,B+28,st,"t16 b c"))
    s.append("<polyline points=\""+" ".join(f"{a:.1f},{b:.1f}" for a,b in dspts)+
             f"\" fill=\"none\" stroke=\"{C['kcr']}\" stroke-width=\"2\" stroke-dasharray=\"5 4\" opacity=\"0.5\"/>")
    for a,b in dspts: s.append(circ(a,b,3,C["kcr"]))
    g=D["ladder"][bb]["SFT+GRPO"]; k=D["ladder"][bb]["KCR"]; f=D["ladder"][bb]["SFT"]
    R=lambda a,b_,bench: round(a[bench]["ba"],2)-round(b_[bench]["ba"],2)   # match the printed bars
    by=y0+PH-58
    s.append(rect(x0+26,by-26,PW-52,44,fill=C["kcr_f"],rx=8))
    s.append(txt(x0+42,by+4,f"KCR over SFT:  DS {k['ds']['ba']-f['ds']['ba']:+.2f} pp,  VisA {k['visa']['ba']-f['visa']['ba']:+.2f} pp","t15 b"))
    s.append(line(x0+PW/2-6,by-14,x0+PW/2-6,by+10,C["kcr"],1))
    s.append(txt(x0+PW/2+18,by+4,f"KCR over SFT+GRPO:  DS {R(k,g,'ds'):+.2f} pp,  VisA {R(k,g,'visa'):+.2f} pp","t15 b"))
ly=H-56
s.append(legend_swatch(40,ly,C["base"],C["base"],"solid: DS-MVTec, 1,670 images",w=26,h=15,cls="t15"))
s.append(legend_swatch(330,ly,C["base_f"],C["base"],"pale: VisA, 2,141 images",w=26,h=15,cls="t15"))
s.append(line(620,ly-5,646,ly-5,C["kcr"],2,dash="5 4")); s.append(circ(633,ly-5,3,C["kcr"]))
s.append(txt(654,ly,"the DS-MVTec ladder","t15"))
s.append(txt(40,H-22,"The 50 line is chance. Best epoch per configuration, selected on DS-MVTec.","t13 mut"))
s.append(txt(40,H-4,"The LLaVA-OneVision DS-MVTec cells carry the contamination caveat discussed in the text. VisA is clean for every row.","t13 mut"))
s.append(foot()); open('fig_ladder.svg','w').write("".join(s))
print("fig_ladder.svg rebuilt at 980px")
