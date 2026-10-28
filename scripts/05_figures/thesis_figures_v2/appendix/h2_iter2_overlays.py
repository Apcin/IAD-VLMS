"""Appendix F overlays and Appendix D per-product plot: SFT+GRPO (ckpt-530) against the post-GRPO refinement on the cleaned 6K pool (iter2_clean ckpt-188)."""
import json,os,numpy as np,collections
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W=os.environ.get('WORK_DIR','/bulk/aacudad/reasoning_traces'); OUT=os.environ.get('THESIS_FIGURES','/bulk/aacudad/ol_thesis/figures')
B=f'{W}/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_%s_full_trainprompt.json'; R=f'{W}/outputs/sft_qwen25vl_7b_iter2_clean/checkpoint-188/eval_%s_full_trainprompt.json'
NB,NR='SFT+GRPO','refinement'
def load(p): return {r['image_id']:r for r in json.load(open(p))['results']}
# ---- overlays on DS-MVTec ----
b=load(B%'dsmvtec'); r=load(R%'dsmvtec'); ids=sorted(set(b)&set(r))
import glob
def mask_path(p):
    parts=p.split('/'); root='/'.join(parts[:-3]); g=glob.glob(f"{root}/mask/{parts[-2]}/{os.path.splitext(parts[-1])[0]}*.png"); return g[0] if g else '/nonexistent'

used=set()
def find(gt,base_ok,ref_ok,need_mask=False,prefer=()):
    order=list(prefer)+[p for p in sorted({b[i]['product'] for i in ids}) if p not in prefer]
    for prod in order:
        if prod in used: continue
        for i in ids:
            rb,rr=b[i],r[i]
            if rb['product']!=prod or rb['gt_answer']!=gt or (rb['pred_answer']==gt)!=base_ok or (rr['pred_answer']==gt)!=ref_ok: continue
            if need_mask and not os.path.exists(mask_path(rb['absolute_path'])): continue
            used.add(prod); return rb,rr
    return None,None
cases=[("both correct",find("yes",True,True,True,prefer=('tile',))),(f"OK over-predicted by the {NR}",find("no",True,False,prefer=('transistor','pill'))),("both miss",find("yes",False,False,True,prefer=('cable','screw'))),(f"{NR} catches, {NB} misses",find("yes",False,True,True,prefer=('zipper','capsule','hazelnut')))]
SZ=360
def render(p):
    img=Image.open(p).convert('RGB').resize((SZ,SZ)); mp=mask_path(p)
    if os.path.exists(mp):
        m=np.array(Image.open(mp).convert('L').resize((SZ,SZ)))>0
        if m.any(): a=np.array(img).astype(float); a[m,0]=0.5*a[m,0]+127.5; a[m,1]*=0.5; a[m,2]*=0.5; img=Image.fromarray(a.astype('uint8'))
    return img
plt.rcParams.update({"font.family":"DejaVu Sans"}); fig,axes=plt.subplots(2,2,figsize=(8.4,9.8)); tick=lambda ok:'✓' if ok else '✗'
picked=[]
for ax,(label,(rb,rr)) in zip(axes.flat,cases):
    ax.set_xticks([]); ax.set_yticks([])
    if rb is None: ax.axis('off'); ax.set_title(label+'\n(no matching sample)',fontsize=9); continue
    ax.imshow(render(rb['absolute_path'])); gt=rb['gt_answer']
    ax.set_title(f"{'NG' if gt=='yes' else 'OK'} · {rb['product']} — {label}\nGT: {gt}    {NB}: {rb['pred_answer']} {tick(rb['pred_answer']==gt)}    {NR}: {rr['pred_answer']} {tick(rr['pred_answer']==gt)}",fontsize=8.5)
    picked.append((label,rb['product'],rb['absolute_path'].split('/')[-2],rb['image_id']))
fig.suptitle('Representative DS-MVTec cases  (ground-truth defect mask overlaid in red)',fontsize=11,y=0.995); fig.tight_layout(rect=[0,0,1,0.985],h_pad=2.5); fig.savefig(f'{OUT}/iter2_sample_overlays.png',dpi=170,bbox_inches='tight')
for p in picked: print('case:',p)
# ---- per-product VisA ----
b=load(B%'visa'); r=load(R%'visa'); ids=sorted(set(b)&set(r))
def ba(rows):
    tp=sum(1 for x in rows if x['gt_answer']=='yes' and x['pred_answer']=='yes'); fn=sum(1 for x in rows if x['gt_answer']=='yes' and x['pred_answer']!='yes'); tn=sum(1 for x in rows if x['gt_answer']=='no' and x['pred_answer']=='no'); fp=sum(1 for x in rows if x['gt_answer']=='no' and x['pred_answer']!='no')
    return 50*(tp/(tp+fn)+tn/(tn+fp))
prods=sorted({b[i]['product'] for i in ids}); rows=[]
for p in prods:
    sub=[i for i in ids if b[i]['product']==p]; rows.append((p,ba([b[i] for i in sub]),ba([r[i] for i in sub])))
rows.sort(key=lambda x:x[2]-x[1]); ob,orr=ba([b[i] for i in ids]),ba([r[i] for i in ids])
fig,ax=plt.subplots(figsize=(8.5,4.2)); x=np.arange(len(rows)); w=0.38
ax.bar(x-w/2,[v[1] for v in rows],w,color='0.6',label=f'{NB} (ckpt-530), overall {ob:.2f}'); ax.bar(x+w/2,[v[2] for v in rows],w,color='tab:blue',label=f'post-GRPO {NR} on the cleaned 6K pool (ep 1), overall {orr:.2f}')
for i,(p,vb,vr) in enumerate(rows): ax.text(i,max(vb,vr)+1,f"{vr-vb:+.1f}",ha='center',fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels([v[0] for v in rows],rotation=35,ha='right'); ax.set_ylabel('balanced accuracy (%)'); ax.set_ylim(40,105); ax.axhline(50,color='0.7',ls=':',lw=0.8)
ax.set_title(f'Per-product balanced accuracy on VisA: {NB} against the post-GRPO {NR}'); ax.legend(fontsize=8,frameon=False,loc='upper left'); fig.tight_layout(); fig.savefig(f'{OUT}/per_product_iter2_visa.png',dpi=170)
print('visa overall',round(ob,2),round(orr,2),'net',round(orr-ob,2)); [print(f"  {p:12s} {vb:6.2f} {vr:6.2f} {vr-vb:+.1f}") for p,vb,vr in rows]
