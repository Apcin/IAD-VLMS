import json,sys,os,textwrap
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
IM=json.load(open('explain_images.json'))
TR_KCR=("I am inspecting a blue PCB module with two circular components on the top surface. I'll start by "
 "checking the main board substrate and the solder joints around the components. The central area and the "
 "left-side pins appear to be in good condition. However, looking at the bottom edge of the board, there is "
 "a clear structural defect. Several metal leads are bent outward from their intended positions, extending "
 "beyond the footprint of the board. This deformation would likely prevent proper mounting onto a larger "
 "assembly. Based on this visual evidence, the part is rejected.")
TR_IAD=("During my thorough inspection of this component, focusing particularly on its lower section against "
 "our established normative standards, it is evident that certain expected elements appear to be absent at "
 "what would typically house them. Upon closer scrutiny around these designated spots labeled 'T' and 'R', "
 "one notices discrepancies suggesting potential issues related to part integrity. This observation aligns "
 "precisely with identifying 'missing parts' as per defined criteria provided earlier today.")
AX=["visual grounding","defect faithfulness","evidence before conclusion","coherence","conciseness"]
KCR_AX=[2,2,2,2,2]; IAD_AX=[0,0,1,1,1]
KCR_MEAN=[1.79,1.47,2.00,1.94,1.91]; IAD_MEAN=[1.47,1.06,1.74,1.79,0.89]  # shared set, 47 VisA images, outputs/explainability_shared/NOTE.md
W=980; PX,PW2=24,932; LH=18; CHARS=56
def band(x,y,w,h,col,n,title,sub=None):
    o=[panel(x,y,w,h,col,rx=22,sw=2), numbadge(x+34,y+32,n,col,r=17), txt(x+62,y+39,title,"t18 b")]
    if sub: o.append(txt(x+PW2-16,y+39,sub,"t13 mut e"))
    return o
s=[txt(W/2,42,"What the reasoning trace gives the inspector","h"),
   txt(W/2,72,"One VisA image both models call correctly, so only the explanation differs.","sub mut")]
y=96
# ---- 1 ----
iw,ih=274,209; AH=56+ih+34
s+=band(PX,y,PW2,AH,C["grpo"],1,"What the inspector sees","VisA / pcb1 / bad / 008    ground truth: bent pin")
for i,(k,cap) in enumerate((("input","the image the model gets"),("overlay","ground-truth defect region"),("grid","3x3 grid, true cell shaded"))):
    ix=48+i*292
    s.append(f'<image x="{ix}" y="{y+56}" width="{iw}" height="{ih}" href="{IM[k]}" preserveAspectRatio="xMidYMid slice"/>')
    s.append(rect(ix,y+56,iw,ih,fill="none",stroke=C["rule2"],sw=1))
    s.append(txt(ix+iw/2,y+56+ih+22,cap,"t13 mut c"))
y+=AH+12
# ---- 2 ----
wk,wi=textwrap.wrap(TR_KCR,CHARS),textwrap.wrap(TR_IAD,CHARS)
nl=max(len(wk),len(wi)); card=30+nl*LH+12; BH=56+card+12+26+18
s+=band(PX,y,PW2,BH,C["kcr"],2,"The two traces, verbatim")
for i,(nm,lines,col,fill,tags) in enumerate((
        ("Qwen2.5-VL-7B, KCR",wk,C["kcr"],C["kcr_f"],"<location> bottom-center    <type> deformation"),
        ("IAD-R1 released checkpoint, our prompt",wi,C["sft"],C["sft_f"],"<location> bottom    <type> missing parts"))):
    cx=40+i*466; cw=440
    s.append(rect(cx,y+56,cw,card,fill=fill,rx=10))
    s.append(txt(cx+14,y+78,nm,"t15 b"))
    for j,ln in enumerate(lines): s.append(txt(cx+14,y+98+j*LH,ln,"t13"))
    ty=y+56+card+12
    s.append(rect(cx,ty,cw,26,fill="#fff",stroke=col,sw=1,rx=6))
    s.append(txt(cx+12,ty+18,tags,"t13 b",fill=col))
y+=BH+12
# ---- 3 ----
CH=62+2*26+18
s+=band(PX,y,PW2,CH,C["purple"],3,"Scored mechanically, outside the judge")
s.append(txt(600,y+58,"KCR","t13 b",fill=C["kcr"]))
s.append(txt(778,y+58,"IAD-R1","t13 b",fill=C["sft"]))
for j,(lab,a,b) in enumerate((("location, predicted 3x3 cell against the true cell","bottom-center, hit","bottom, coarser hit"),
                              ("type, embedding similarity to the true defect name","0.54","0.41"))):
    yy=y+82+j*26
    s.append(txt(40,yy,lab,"t13 mut"))
    s.append(txt(600,yy,a,"t13 b",fill=C["kcr"]))
    s.append(txt(778,yy,b,"t13 b",fill=C["sft"]))
y+=CH+12
# ---- 4 ----
DH=134+5*26+52
s+=band(PX,y,PW2,DH,C["purple"],4,"Blind judge, five axes scored 0 to 2")
s.append(txt(40,y+58,"The judge sees the image, the overlay and the true defect name, and is told the verdict is already correct.","t13 mut"))
s.append(txt(40,y+76,"It is not told which model wrote the trace.","t13 mut"))
CA,CB=430,556; MA,MB=690,826
s.append(txt(CA,y+96,"this image","t13 mut")); s.append(txt(MA,y+96,"mean over the 47 shared VisA images","t13 mut"))
for hx in (CA,MA): s.append(txt(hx,y+112,"KCR","t13 b",fill=C["kcr"]))
for hx in (CB,MB): s.append(txt(hx,y+112,"IAD-R1","t13 b",fill=C["sft"]))
def bars(x,v,col,mx,wd):
    return rect(x,0,0,0)
for i,ax in enumerate(AX):
    yy=y+126+i*26
    s.append(txt(40,yy+12,ax,"t13"))
    for x,v,col in ((CA,KCR_AX[i],C["kcr"]),(CB,IAD_AX[i],C["sft"])):
        s.append(rect(x,yy,40,15,fill="none",stroke="#c9d2d8",sw=1,rx=3))
        if v: s.append(rect(x,yy,40*(v/2),15,fill=col,rx=3))
        s.append(txt(x+46,yy+12,str(v),"t13 b",fill=col))
    for x,v,col in ((MA,KCR_MEAN[i],C["kcr"]),(MB,IAD_MEAN[i],C["sft"])):
        s.append(rect(x,yy,66,15,fill="none",stroke="#c9d2d8",sw=1,rx=3))
        s.append(rect(x,yy,66*(v/2),15,fill=col,rx=3))
        s.append(txt(x+72,yy+12,f"{v:.2f}","t13",fill=col))
ty=y+126+5*26+8
s.append(line(40,ty,916,ty,C["rule"],1))
s.append(txt(40,ty+22,"total out of 10","t15 b"))
for x,v,col in ((CA,sum(KCR_AX),C["kcr"]),(CB,sum(IAD_AX),C["sft"]),
                (MA,sum(KCR_MEAN),C["kcr"]),(MB,sum(IAD_MEAN),C["sft"])):
    s.append(txt(x,ty+22,f"{v:.2f}" if isinstance(v,float) else str(v),"t15 b",fill=col))
y+=DH+10
H=y+46
s.append(txt(PX,H-30,"The judge scored the released checkpoint zero on visual grounding and zero on defect faithfulness here. Its trace reports missing parts","t13 mut"))
s.append(txt(PX,H-11,"on an image whose defect is a bent pin. VisA carries no contamination caveat, so this comparison sits on the clean benchmark.","t13 mut"))
open('fig_explain.svg','w').write(head(W,H,"Explanation quality on one matched VisA case")+"".join(s)+foot())
print("fig_explain.svg H =",H, round(os.path.getsize('fig_explain.svg')/1024),"KB")
