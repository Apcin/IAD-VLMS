import os
import json,sys,os,io,base64,re
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
os.chdir((os.environ.get('WORK_DIR','/bulk/aacudad/reasoning_traces')+''))
from registry import REG
from PIL import Image
ST=['SFT','SFT+GRPO','KCR']
def load(p): return {r['image_id']:r for r in json.load(open(p))['results']}
bb='Qwen2.5-VL-7B'
PICK={'ds':["DS-MVTec/bottle/image/broken_large/011.png",
            "DS-MVTec/carpet/image/color/002.png",
            "DS-MVTec/cable/image/cut_outer_insulation/008.png"],
      'visa':["VisA/candle/test/bad/002.JPG",
              "VisA/cashew/test/bad/062.JPG",
              "VisA/macaroni1/test/bad/008.JPG"]}
def thumb(p,px=430,q=80):
    im=Image.open(p).convert("RGB"); im.thumbnail((px,px))
    b=io.BytesIO(); im.save(b,"JPEG",quality=q,optimize=True)
    return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode(), im.size
def key_sentence(t,want_defect):
    ss=[x.strip() for x in re.split(r'(?<=[.!?])\s+',t or '') if x.strip()]
    if not ss: return ""
    ss=[x for x in ss if not re.match(r'(Based on|Therefore|In conclusion|Given )',x,re.I)]
    if want_defect:
        cand=[x for x in ss
              if re.search(r'\b(crack|chip|broken|scratch|stain|discolor|missing|damag|bent|thread|contamin|irregular|smudge|gouge|tear)\w*\b',x,re.I)
              and not re.search(r'\bno\b|\bnot\b|without',x[:46],re.I)]
    else:
        cand=[x for x in ss
              if re.search(r'\b(no |appears? (?:to be )?(?:uniform|consistent|correct|normal|intact)|free of|without any|shows no)\b',x,re.I)]
    if not cand: return min(ss,key=len) if ss else ""
    return min(cand,key=len)
out=[]
for bench,ids in PICK.items():
    d={s:load(REG[bb][s][bench]) for s in ST}
    for iid in ids:
        r=d['KCR'][iid]; t=r.get('pred_tags') or {}
        img,sz=thumb(r['absolute_path'])
        kcr_right = r['pred_answer']==r['gt_answer']
        out.append(dict(bench='DS-MVTec' if bench=='ds' else 'VisA',
            image_id=iid, product=r['product'], gt=r['gt_answer'], img=img,
            defect=iid.split('/')[3] if bench=='ds' else 'defect',
            preds={s:d[s][iid]['pred_answer'] for s in ST},
            oks={s:(d[s][iid]['pred_answer']==d[s][iid]['gt_answer']) for s in ST},
            type=t.get('type'), loc=t.get('location'),
            quote=key_sentence(t.get('reasoning'), kcr_right)))
json.dump(out,open((os.environ.get('WORK_DIR','/bulk/aacudad/reasoning_traces')+'/thesis_figures_v2/gallery.json'),'w'))
for c in out:
    print(f"  {c['image_id']:44s} gt={c['gt']} KCR={c['preds']['KCR']:3s} SFT={c['preds']['SFT']:3s} GRPO={c['preds']['SFT+GRPO']:3s}")
    print(f"      \"{c['quote'][:150]}\"")
print("\n  payload MB:", round(sum(len(c['img']) for c in out)/1e6,2))
