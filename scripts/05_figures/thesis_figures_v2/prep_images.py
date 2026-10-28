import os
import io,base64,os
from PIL import Image, ImageDraw
R=(os.environ.get('WORK_DIR','/bulk/aacudad/reasoning_traces')+'/reasoning_traces_gen/data/MMAD')
IMG=f'{R}/VisA/pcb1/test/bad/008.JPG'
MSK=f'{R}/VisA/pcb1/ground_truth/bad/008.png'
def b64(im,q=82,px=460):
    im=im.convert("RGB").copy(); im.thumbnail((px,px))
    b=io.BytesIO(); im.save(b,"JPEG",quality=q,optimize=True)
    return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode(), im.size
src=Image.open(IMG).convert("RGB")
m=Image.open(MSK).convert("L").resize(src.size)
# red overlay where the mask is set
ov=src.copy(); red=Image.new("RGB",src.size,(214,32,42))
ov=Image.composite(Image.blend(src,red,0.62),ov,m.point(lambda v:255 if v>10 else 0))
d=ImageDraw.Draw(ov)
bb=m.point(lambda v:255 if v>10 else 0).getbbox()
if bb: d.rectangle(bb,outline=(214,32,42),width=6)
# 3x3 grid version with the ground-truth cells shaded
gr=src.copy(); dg=ImageDraw.Draw(gr,"RGBA")
W,H=src.size
cells=set()
if bb:
    for cx in range(3):
        for cy in range(3):
            x0,y0,x1,y1=cx*W/3,cy*H/3,(cx+1)*W/3,(cy+1)*H/3
            if not (bb[2]<x0 or bb[0]>x1 or bb[3]<y0 or bb[1]>y1): cells.add((cx,cy))
for (cx,cy) in cells:
    dg.rectangle([cx*W/3,cy*H/3,(cx+1)*W/3,(cy+1)*H/3],fill=(0,185,92,80))
for i in (1,2):
    dg.line([i*W/3,0,i*W/3,H],fill=(255,255,255,220),width=5)
    dg.line([0,i*H/3,W,i*H/3],fill=(255,255,255,220),width=5)
NAME={(0,0):"top-left",(1,0):"top-center",(2,0):"top-right",(0,1):"middle-left",(1,1):"center",
      (2,1):"middle-right",(0,2):"bottom-left",(1,2):"bottom-center",(2,2):"bottom-right"}
out={}
out['input'],sz=b64(src); out['size']=sz
out['overlay'],_=b64(ov); out['grid'],_=b64(gr)
out['gt_cells']=sorted(NAME[c] for c in cells)
import json; json.dump(out,open('explain_images.json','w'))
print("  ground-truth cells:",out['gt_cells'])
print("  mask bbox:",bb,"| thumb size:",sz)
print("  payload KB:", round(sum(len(v) for k,v in out.items() if isinstance(v,str))/1024))
