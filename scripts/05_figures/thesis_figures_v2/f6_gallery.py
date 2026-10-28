import json,sys,os,textwrap
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
G=json.load(open('gallery.json'))
ROWA=[c for c in G if c['oks']['KCR']]; ROWB=[c for c in G if not c['oks']['KCR']]
W=980; MX=24; GAPX=16; CW=(W-2*MX-2*GAPX)/3; IMH=196; QL=5; LH=17; WRAP=33
CELLH=IMH+26+28+QL*LH+14
def mark(x,y,ok):
    c=C["good"] if ok else C["bad"]
    if ok: d=f"M {x} {y+4} l 3.5 4.5 L {x+10} {y-4}"
    else:  d=f"M {x} {y-4} l 9 9 M {x+9} {y-4} l -9 9"
    return f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
def row(y,cells,col,fillc,head,sub):
    o=[rect(MX,y,W-2*MX,32,fill=fillc,rx=8),txt(MX+14,y+21,head,"t16 b"),
       txt(W-MX-14,y+21,sub,"t13 e",fill="#3c4a52")]
    for i,c in enumerate(cells):
        cx=MX+i*(CW+GAPX); cy=y+42; cid=f"cp{i}{'a' if cells is ROWA else 'b'}"
        o.append(rect(cx,cy,CW,CELLH,fill="#fff",stroke=col,sw=1.5,rx=10))
        o.append(f'<clipPath id="{cid}"><rect x="{cx+1}" y="{cy+1}" width="{CW-2}" height="{IMH}" rx="9"/></clipPath>')
        o.append(f'<image x="{cx+1}" y="{cy+1}" width="{CW-2}" height="{IMH}" href="{c["img"]}" '
                 f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{cid})"/>')
        o.append(line(cx+1,cy+IMH+1,cx+CW-1,cy+IMH+1,C["rule"],1))
        o.append(txt(cx+12,cy+IMH+22,f"{c['bench']} / {c['product']}","t13 b"))
        vy=cy+IMH+46
        for st,lx,mx in (("SFT",12,42),("SFT+GRPO",96,178),("KCR",216,254)):
            o.append(txt(cx+lx,vy,st,"t13 b" if st=="KCR" else "t13"))
            o.append(mark(cx+mx,vy-5,c['oks'][st]))
        for k,ln in enumerate(textwrap.wrap('"'+c['quote'].rstrip('.')+'."',WRAP)[:QL]):
            o.append(txt(cx+12,cy+IMH+70+k*LH,ln,"t13"))
    return o
s=[txt(W/2,42,"What the corrective corpus fixes, and what it costs","h"),
   txt(W/2,72,"Six Qwen2.5-VL-7B cases where the trained stages disagree. Every image below carries a real defect.","sub mut")]
y=96
s+=row(y,ROWA,C["kcr"],C["kcr_f"],"KCR catches defects the two earlier stages both miss","42 such images on DS-MVTec")
y+=42+CELLH+20
s+=row(y,ROWB,C["bad"],"#fbe4e2","and it misses defects both earlier stages catch","107 such images on VisA")
y+=42+CELLH+20
H=y+72
s.append(txt(MX,H-50,"Quoted text is the model's own trace, verbatim. In the top row it names the evidence and the region.","t13 mut"))
s.append(txt(MX,H-30,"In the bottom row it asserts uniformity on an image that carries a defect.","t13 mut"))
s.append(txt(MX,H-10,"This is the sensitivity-for-specificity trade of the KCR checkpoint, seen case by case.","t13 mut"))
open('fig_gallery.svg','w').write(head(W,H,"Qualitative cases where the trained stages disagree")+"".join(s)+foot())
print("H =",H, round(os.path.getsize('fig_gallery.svg')/1024),"KB")
