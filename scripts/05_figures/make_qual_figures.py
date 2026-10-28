#!/usr/bin/env python3
"""Build qualitative-example image panels for the thesis:
  - anomaly: original | original+red-GT-mask-overlay  (pill + bottle)
  - normal:  single clean image (transistor + cable)
Saves into thesis figures/ as qual_*.png
"""
from PIL import Image, ImageDraw
import numpy as np, os
DATA="/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/MMAD/DS-MVTec"
OUT="/tmp/thesis_review/thesis_tud/figures"
SZ=420

def load(p): return Image.open(p).convert("RGB").resize((SZ,SZ))

def overlay(img_p, mask_p):
    img=load(img_p)
    m=Image.open(mask_p).convert("L").resize((SZ,SZ))
    a=np.array(img).astype(float); mk=np.array(m)>0
    red=np.zeros_like(a); red[...,0]=255
    al=0.50
    a[mk]=(1-al)*a[mk]+al*red[mk]
    out=Image.fromarray(a.astype("uint8"))
    # bright bounding box around the GT region so small defects are unmistakable
    ys,xs=np.where(mk)
    if len(xs):
        pad=10
        x0,x1=max(0,xs.min()-pad),min(SZ-1,xs.max()+pad)
        y0,y1=max(0,ys.min()-pad),min(SZ-1,ys.max()+pad)
        d=ImageDraw.Draw(out)
        for w in range(3):
            d.rectangle([x0-w,y0-w,x1+w,y1+w],outline=(50,255,50))
    return out

def hcat(imgs, gap=10):
    h=imgs[0].height; w=sum(i.width for i in imgs)+gap*(len(imgs)-1)
    canvas=Image.new("RGB",(w,h),(255,255,255)); x=0
    for im in imgs:
        canvas.paste(im,(x,0)); x+=im.width+gap
    return canvas

def anomaly(name, prod, defect, num):
    ip=f"{DATA}/{prod}/image/{defect}/{num}.png"
    mp=f"{DATA}/{prod}/mask/{defect}/{num}_mask.png"
    comp=hcat([load(ip), overlay(ip,mp)])
    comp.save(f"{OUT}/qual_anom_{name}.png"); print("wrote", f"qual_anom_{name}.png")

def normal(name, prod, num):
    load(f"{DATA}/{prod}/image/good/{num}.png").save(f"{OUT}/qual_norm_{name}.png")
    print("wrote", f"qual_norm_{name}.png")

anomaly("pill","pill","combined","001")
anomaly("bottle","bottle","broken_large","009")
normal("transistor","transistor","007")
normal("cable","cable","029")
