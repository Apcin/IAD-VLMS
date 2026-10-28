import os
"""Full-benchmark type similarity per stage, judge rule (substring=1 else nomic cosine). Writes type_sim.json + tab_type_sim.tex."""
import json,re,os,numpy as np,collections
from sentence_transformers import SentenceTransformer
R=(os.environ.get('WORK_DIR','/bulk/aacudad/reasoning_traces')+'/'); mmad=json.load(open(R+'MMAD_repo/dataset/MMAD/mmad.json'))
nomic=SentenceTransformer("nomic-ai/nomic-embed-text-v1.5",trust_remote_code=True,device="cpu")
def gt_type(entry,fallback):
    for q in entry.get("conversation",[]):
        if "type of the defect" in q.get("Question","").lower():
            ans=q.get("Answer",""); opt=q.get("Options",{}); v=opt.get(ans,ans) if isinstance(opt,dict) else ans; v=str(v).strip().rstrip(".").strip()
            if v and v.lower() not in ("good","none","no defect"): return v
    return fallback.replace("_"," ")
cache={}
def type_sim(pred,gt):
    pred=(pred or "").strip().lower(); gt=(gt or "").strip().lower().replace("_"," ")
    if not pred: return 0.0
    if pred==gt or pred in gt or gt in pred: return 1.0
    k=(pred,gt)
    if k not in cache: e=nomic.encode([f"search_query: {pred}",f"search_query: {gt}"],normalize_embeddings=True); cache[k]=float(np.dot(e[0],e[1]))
    return cache[k]
M=[('\\qwenvl-7B','SFT','outputs/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_{b}_full_trainprompt.json'),
   ('\\qwenvl-7B','SFT$+$GRPO','outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_{b}_full_trainprompt.json'),
   ('\\qwenvl-7B','KCR','outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_{b}_full_trainprompt.json'),
   ('LLaVA-OneVision-7B$^{\\dagger}$','SFT','outputs/sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188/eval_{b}_full_trainprompt.json'),
   ('LLaVA-OneVision-7B$^{\\dagger}$','SFT$+$GRPO','outputs/grpo_llava_ov_from_ep1/checkpoint-530/eval_{b}_full_trainprompt_vllm.json'),
   ('LLaVA-OneVision-7B$^{\\dagger}$','KCR','outputs/sft_llava_ov_7b_frozen_iad_sft_llava_iter1_C_original/checkpoint-376/eval_{b}_full_trainprompt_vllm.json')]
out=[]; L=[]; last=None; pairs=collections.Counter(); vocab={}
for bb,st,pat in M:
    rec={'backbone':bb,'stage':st}
    for b in ('dsmvtec','visa'):
        sims=[]; P=collections.Counter(); G=collections.Counter()
        for r in json.load(open(R+pat.format(b=b)))['results']:
            if str(r['gt_answer']).lower()!='yes' or str(r['pred_answer']).lower()!='yes': continue
            e=mmad.get(r['image_id'])
            if not e: continue
            gt=gt_type(e,r['image_id'].split('/')[-2]); pt=(re.search(r'<type>(.*?)</type>',r['pred_full'] or '',re.S|re.I) or [None,''])[1]
            s=type_sim(pt,gt); sims.append(s); P[pt.strip().lower()]+=1; G[gt.lower()]+=1
            if st=='KCR' and 'qwenvl' in bb and s<0.999: pairs[(b,gt.lower(),pt.strip().lower())]+=1
        a=np.array(sims); rec[b]=dict(n=len(a),mean=float(a.mean()),match=float(100*(a>=0.999).mean()),n_pred_vocab=len(P),n_gt_vocab=len(G))
        if st=='KCR' and 'qwenvl' in bb: vocab[b]=dict(pred=P.most_common(6),gt=G.most_common(6))
    out.append(rec)
    if bb!=last: L.append("\\midrule" if last else ""); L.append(f"\\multicolumn{{5}}{{@{{}}l}}{{{bb}}}\\\\"); last=bb
    d,v=rec['dsmvtec'],rec['visa']; L.append(f"\\quad {st} & {d['mean']:.2f} & {d['match']:.1f}\\,\\% & {v['mean']:.2f} & {v['match']:.1f}\\,\\%\\\\")
json.dump(dict(rows=out,vocab=vocab,top_confusions=[dict(bench=b,gt=g,pred=p,n=n) for (b,g,p),n in pairs.most_common(12)]),open(R+'thesis_figures_v2/type_sim.json','w'),indent=1)
open(R+'thesis_figures_v2/tab_type_sim.tex','w').write('\n'.join(l for l in L if l)); print('\n'.join(l for l in L if l)); print("confusions:",pairs.most_common(8)); print("vocab sizes (Qwen KCR):",{b:(out[2][b]['n_pred_vocab'],out[2][b]['n_gt_vocab']) for b in ('dsmvtec','visa')})
