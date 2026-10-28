"""Chapter 3 worked example: one anomalous AnomalyThink trace in the Figure 6.9 style (image with mask, four-tag trace, decisive sentence shaded).
Usage: python f10_trace_example.py <image_id substring>  -> trace_example_<product>.svg (+ pdf into THESIS_FIGURES)"""
import sys,os,json,re,subprocess
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
import f8_pairs
from f8_pairs import trace_card,sentences,key_sentence,DEFS,CELLS
from f9_appendix_examples import gt_cells
from PIL import Image
import numpy as np
W_=os.environ.get('WORK_DIR','.'); OUT=os.environ.get('THESIS_FIGURES','thesis'); CAIRO=os.environ.get('CAIROSVG','cairosvg')
ROOT=f'{W_}/reasoning_traces_gen/data/Real-IAD/'
def realiad_mask(p):
    q=os.path.splitext(p)[0]+'.png'; return q if os.path.exists(q) else None
import f9_appendix_examples as f9
f8_pairs.mask_for=realiad_mask; f9.mask_for=realiad_mask  # Real-IAD keeps the mask next to the image
key=sys.argv[1]
d=json.load(open(f'{W_}/Training/datasets_small_new_v4/combined_6k_train.json'))
e=[x for x in d if key in x['images'][0]][0]; img=ROOT+e['images'][0].split('Real-IAD/')[-1]
t=e['messages'][-1]['content']; think=re.search(r'<think>(.*?)</think>',t,re.S).group(1).strip()
mt=re.search(r'<type>(.*?)</type>',t,re.S); ml=re.search(r'<location>(.*?)</location>',t,re.S); typ=mt.group(1).strip() if mt else None; loc=ml.group(1).strip() if ml else None; ans=re.search(r'<answer>(.*?)</answer>',t,re.S).group(1).strip(); normal=typ is None
product=e['images'][0].split('/')[1] if e['images'][0].startswith('Real-IAD') else e['images'][0].split('images/')[1].split('/')[0]
rec={'absolute_path':img}; cells=[] if normal else gt_cells(rec)
sents=sentences(think); ki=key_sentence(sents,'sft' if normal else 'kcr')
W=980; IMG=300; x1=24+IMG+18; cwid=W-x1-24
h=trace_card([],x1,24,cwid,f"AnomalyThink trace, {product}",C['sft'],C['sft_f'],(f"<answer> {ans}" if normal else f"<answer> {ans}, <type> {typ}, <location> {loc}"),sents,ki,"tint",hl_fill="#c9ecd5")
H=24+max(h,IMG+88)+24; o=[head(W,H),DEFS]
f8_pairs.image_panel(o,24,24,IMG,rec,cells,[])
o.append(txt(24,24+IMG+22,f"{product}, Real-IAD C1 view","t13 b")); o.append(txt(24,24+IMG+40,"ground truth: normal" if normal else "ground truth: defective","t13 b"))
if not normal: o.append(txt(24,24+IMG+58,"shaded cell: mask cells (>5 % of the mask)","t13 mut")); o.append(txt(24,24+IMG+76,"red circle: the mask region, arrow to it","t13 mut"))
else: o.append(txt(24,24+IMG+58,"no mask, no location or type tag","t13 mut"))
trace_card(o,x1,24,cwid,f"AnomalyThink trace, {product}",C['sft'],C['sft_f'],(f"<answer> {ans}" if normal else f"<answer> {ans}, <type> {typ}, <location> {loc}"),sents,ki,"tint",hl_fill="#c9ecd5",fixed_h=max(h,IMG+88))
o.append(foot()); name=f'trace_example_{product}'; open(name+'.svg','w').write(''.join(o))
subprocess.run([CAIRO,name+'.svg','-o',f'{OUT}/{name}.pdf'],check=True); print('installed',f'{OUT}/{name}.pdf','| image',e['images'][0],'| cells',cells,'| type',typ,'| loc',loc)
