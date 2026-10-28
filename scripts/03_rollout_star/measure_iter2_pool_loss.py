"""Per-item cross-entropy of the SFT+GRPO checkpoint (ckpt-530) on the 6K pool of Table 6.10 / Figure 6.4,
split into kept (the model's own rollouts) and patched (teacher-corrected or rewritten). Assistant tokens only,
same chat template and image budget as training. Writes a JSONL with one line per item and a summary."""
import json,os,sys,time,torch,collections
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
W='${WORK_DIR}'; CK=f'{W}/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530'
OUT=f'{W}/outputs/iter2_pool_loss_measurement'; os.makedirs(OUT,exist_ok=True)
PHASE0=f'{W}/Training/phase0_full_10k_20260529_015821'; HELD=f'{W}/Training/phase0_heldout_20260601'
def paths(p,):
    s=set()
    if os.path.exists(p):
        for l in open(p):
            r=json.loads(l)
            if r.get('corrected_trace'): s.add(r['image_path'])
    return s
corr=set(); rew=set()
for d in (PHASE0,HELD): corr|=paths(f'{d}/gemini_corrected.jsonl'); rew|=paths(f'{d}/gemini_rewritten.jsonl')
data=json.load(open(f'{W}/Training/datasets_sft_iter2/sft_iter2_train.json'))
def kind(x):
    p=x['images'][0]
    return 'corrected' if p in corr else 'rewritten' if p in rew else 'kept'
kinds=[kind(x) for x in data]; print('breakdown',collections.Counter(kinds),flush=True)
proc=AutoProcessor.from_pretrained(CK,min_pixels=256*28*28,max_pixels=262144)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(CK,torch_dtype=torch.bfloat16,device_map='cuda').eval()
done=set()
outp=f'{OUT}/per_item_loss.jsonl'
if os.path.exists(outp):
    for l in open(outp): done.add(json.loads(l)['idx'])
f=open(outp,'a'); t0=time.time()
for i,(x,k) in enumerate(zip(data,kinds)):
    if i in done: continue
    user=x['messages'][0]['content'].replace('<image>','').strip(); asst=x['messages'][1]['content']
    msgs=[{'role':'user','content':[{'type':'image'},{'type':'text','text':user}]},{'role':'assistant','content':[{'type':'text','text':asst}]}]
    full=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False)
    prompt=proc.apply_chat_template(msgs[:1],tokenize=False,add_generation_prompt=True)
    img=Image.open(x['images'][0]).convert('RGB')
    enc=proc(text=[full],images=[img],return_tensors='pt').to('cuda')
    encp=proc(text=[prompt],images=[img],return_tensors='pt')
    npl=encp['input_ids'].shape[1]
    labels=enc['input_ids'].clone(); labels[:,:npl]=-100
    with torch.no_grad(): out=model(**enc,labels=labels)
    ntok=int((labels!=-100).sum())
    rec={'idx':i,'kind':k,'image':x['images'][0],'anomalous':'<answer>yes' in asst.lower().replace(' ',''),'n_assistant_tokens':ntok,'mean_ce':float(out.loss),'sum_ce':float(out.loss)*ntok}
    f.write(json.dumps(rec)+'\n'); f.flush()
    if (i+1)%100==0: print(f'{i+1}/{len(data)} {time.time()-t0:.0f}s',flush=True)
f.close()
recs=[json.loads(l) for l in open(outp)]
summ={}
for k in ('kept','corrected','rewritten','patched'):
    rs=[r for r in recs if (r['kind']==k if k!='patched' else r['kind']!='kept')]
    if not rs: continue
    tok=sum(r['n_assistant_tokens'] for r in rs); ce=sum(r['sum_ce'] for r in rs)
    summ[k]={'n_items':len(rs),'n_anomalous':sum(r['anomalous'] for r in rs),'assistant_tokens':tok,'mean_ce_per_token':ce/tok,'mean_ce_per_item':sum(r['mean_ce'] for r in rs)/len(rs),'share_of_total_summed_ce':ce/sum(r['sum_ce'] for r in recs)}
json.dump(summ,open(f'{OUT}/summary.json','w'),indent=1); print(json.dumps(summ,indent=1))
