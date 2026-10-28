import json,sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
D=json.load(open('figdata.json'))
BB=["Qwen2.5-VL-7B","LLaVA-OneVision-7B"]; ST=["Base","SFT","SFT+GRPO","KCR"]
W=980; PW,PH=448,552; PY=100; H=PY+PH+118
s=[head(W,H,"Sensitivity and specificity behind each balanced accuracy")]
s.append(txt(W/2,42,"Balanced accuracy hides how the model gets there","h"))
s.append(txt(W/2,72,"Sensitivity is the share of defects caught, specificity the share of good parts passed.","sub mut"))
for pi,(bench,bn) in enumerate((("ds","DS-MVTec"),("visa","VisA"))):
    x0=24+pi*(PW+36)
    s.append(panel(x0,PY,PW,PH,C["rule2"],rx=28,sw=2))
    s.append(txt(x0+16,PY+38,bn,"hl"))
    LB=x0+118; RB=x0+PW-86          # bar track
    sc=lambda v: (v/100)*(RB-LB)
    for g in (0,50,100):
        gx=LB+sc(g)
        s.append(line(gx,PY+84,gx,PY+PH-24,C["rule"],1))
        s.append(txt(gx,PY+76,str(g),"t13 mut c"))
    s.append(txt(x0+16,PY+62,"model","t13 mut"))
    s.append(txt(LB,PY+62,"sensitivity  /  specificity  %","t13 mut"))
    s.append(txt(x0+PW-16,PY+62,"bal-acc","t13 mut e"))
    yy=PY+96
    for bb in BB:
        s.append(txt(x0+16,yy+14,bb,"t15 b")); yy+=28
        for st in ST:
            v=D["ladder"][bb][st][bench]; ck,cf=STAGE_C[st]
            s.append(txt(x0+24,yy+18,st,"t15"))
            if v is None:
                s.append(txt(LB,yy+22,"no per-sample file","t13 mut")); yy+=50; continue
            def bar(val,ty,fill,stroke,cls):
                out=[rect(LB,ty,sc(val),15,fill=fill,**({"stroke":stroke,"sw":1} if stroke else {}),rx=2)]
                if val>92:   # label inside, so it never runs past the 100 gridline
                    out.append(txt(LB+sc(val)-7,ty+12,f"{val:.1f}","t13 e",fill="#3d4a52"))
                else:
                    out.append(txt(LB+sc(val)+7,ty+13,f"{val:.1f}",cls))
                return "".join(out)
            s.append(bar(v["tpr"],yy+2,C[ck],None,"t13"))
            s.append(bar(v["tnr"],yy+21,C[cf],C[ck],"t13 mut"))
            s.append(txt(x0+PW-16,yy+26,f"{v['ba']:.2f}","t16 b e"))
            yy+=50
        yy+=8
s.append(txt(24,H-88,"In each pair the upper solid bar is sensitivity and the lower pale bar is specificity.","t13 mut"))
s.append(txt(24,H-64,"On VisA the two KCR models reach a similar balanced accuracy from different operating points.","t15"))
s.append(txt(24,H-40,"Qwen2.5-VL catches 57.1 % of defects and passes 87.1 % of good parts. LLaVA-OneVision sits","t15"))
s.append(txt(24,H-16,"in the middle, 72.5 % and 72.8 %. One balanced-accuracy number cannot separate the two.","t15"))
s.append(foot()); open('fig_operating.svg','w').write("".join(s))
print("ok")
