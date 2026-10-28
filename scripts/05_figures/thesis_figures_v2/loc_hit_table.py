import os
"""Full-benchmark 3x3 location hit rate per stage (same gt_cells/pred_cells rule as the judge). Writes loc_hit.json + tab_loc_hit.tex."""
import json,re,os; from PIL import Image; import numpy as np
R=(os.environ.get('WORK_DIR','/bulk/aacudad/reasoning_traces')+'/'); src=open(R+'Training/explainability_judge_multi.py').read(); i0=src.index('\nCELLS'); i1=src.index('}',i0)+1; exec(src[i0:i1])
def gt_cells(mp,_c={}):
    if mp in _c: return _c[mp]
    fg=(np.array(Image.open(mp).convert('RGB')).sum(2)>30); H,W=fg.shape; tot=int(fg.sum()); c=set()
    if tot:
        for r in range(3):
            for k in range(3):
                if int(fg[r*H//3:(r+1)*H//3,k*W//3:(k+1)*W//3].sum())/tot>0.05: c.add((r,k))
    _c[mp]=c; return c
def pred_cells(loc):
    return {CELLS[t.strip()] for t in re.split(r"[,/;]| and ",(loc or '').lower()) if t.strip() in CELLS}
mmad=json.load(open(R+'MMAD_repo/dataset/MMAD/mmad.json')); DATA=R+'reasoning_traces_gen/data/MMAD/'
M=[('\\qwenvl-7B','SFT','outputs/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_{b}_full_trainprompt.json'),
   ('\\qwenvl-7B','SFT$+$GRPO','outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_{b}_full_trainprompt.json'),
   ('\\qwenvl-7B','KCR','outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_{b}_full_trainprompt.json'),
   ('LLaVA-OneVision-7B$^{\\dagger}$','SFT','outputs/sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188/eval_{b}_full_trainprompt.json'),
   ('LLaVA-OneVision-7B$^{\\dagger}$','SFT$+$GRPO','outputs/grpo_llava_ov_from_ep1/checkpoint-530/eval_{b}_full_trainprompt_vllm.json'),
   ('LLaVA-OneVision-7B$^{\\dagger}$','KCR','outputs/sft_llava_ov_7b_frozen_iad_sft_llava_iter1_C_original/checkpoint-376/eval_{b}_full_trainprompt_vllm.json')]
out=[]; rows=[]
for bb,st,pat in M:
    rec={'backbone':bb,'stage':st}
    for b in ('dsmvtec','visa'):
        tp=hit=anom=0
        for r in json.load(open(R+pat.format(b=b)))['results']:
            if str(r['gt_answer']).lower()!='yes': continue
            e=mmad.get(r['image_id']); mp=os.path.join(DATA,r['image_id'].split('/')[0],r['image_id'].split('/')[1],e['mask_path']) if e else None
            if not mp or not os.path.exists(mp): continue
            gc=gt_cells(mp)
            if not gc: continue
            anom+=1
            if str(r['pred_answer']).lower()=='yes':
                tp+=1; loc=(re.search(r'<location>(.*?)</location>',r['pred_full'] or '',re.S|re.I) or [None,''])[1]
                if pred_cells(loc)&gc: hit+=1
        rec[b]=dict(hit=hit,tp=tp,anom=anom,hit_tp=100*hit/tp,hit_all=100*hit/anom)
    out.append(rec)
json.dump(out,open(R+'thesis_figures_v2/loc_hit.json','w'),indent=1)
def best(key,b): 
    v=[r[b][key] for r in out]; return v
L=[]; last=None
for r in out:
    if r['backbone']!=last: L.append("\\midrule" if last else ""); L.append(f"\\multicolumn{{5}}{{@{{}}l}}{{{r['backbone']}}}\\\\"); last=r['backbone']
    d,v=r['dsmvtec'],r['visa']
    L.append(f"\\quad {r['stage']} & {d['hit_tp']:.1f}\\,\\% & {d['hit_all']:.1f}\\,\\% & {v['hit_tp']:.1f}\\,\\% & {v['hit_all']:.1f}\\,\\%\\\\")
open(R+'thesis_figures_v2/tab_loc_hit.tex','w').write('\n'.join(l for l in L if l)); print('\n'.join(l for l in L if l))
