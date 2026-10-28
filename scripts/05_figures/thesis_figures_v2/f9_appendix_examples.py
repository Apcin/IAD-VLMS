"""Appendix example figures in the house style (same kit as Figure 6.9 / f8_pairs.py).
F.1 success gallery, F.2 failure galleries, F.3 triptych, F.4 overlays, G.1-G.5 teacher against student.
Selection: F.1 from KCR images the explainability judge scored 10/10 (shared set or own-100), one per product family;
F.2 from KCR wrong verdicts and from judge scores <= 5; F.3 from images the judge scored SFT+GRPO 10/10 where SFT answered no."""
import json,os,sys,glob,ast,re,io,base64,subprocess,collections
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
from f8_pairs import image_panel,trace_card,sentences,key_sentence,wrap_words,tw,DEFS,mask_for,CELLS,W_,KEY
from PIL import Image
import numpy as np
OUT=os.environ.get('THESIS_FIGURES','thesis'); CAIRO=os.environ.get('CAIROSVG','cairosvg')
EV={'kcr':f'{W_}/outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_%s_full_trainprompt.json',
    'sft':f'{W_}/outputs/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_%s_full_trainprompt.json',
    'grpo':f'{W_}/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_%s_full_trainprompt.json',
    'ref':f'{W_}/outputs/sft_qwen25vl_7b_iter2_clean/checkpoint-188/eval_%s_full_trainprompt.json',
    'base':f'{W_}/outputs/baseline_eval/7b_base_%s.json',
    'gem':f'{W_}/outputs/gemini_25flash_eval/eval_%s_full_gemini25flash.json'}
def load(key):
    out={}
    for b in ('dsmvtec','visa'):
        for r in json.load(open(EV[key]%b))['results']:
            if isinstance(r.get('pred_tags'),str):
                try: r['pred_tags']=ast.literal_eval(r['pred_tags'])
                except Exception: r['pred_tags']={}
            r['pred_tags']=r.get('pred_tags') or {}; out[r['image_id']]=r
    return out
def gt_cells(rec):
    """3x3 cells holding more than five percent of the mask"""
    mp=mask_for(rec['absolute_path'])
    if not mp: return []
    m=np.array(Image.open(mp).convert('L'))>0; h,w=m.shape; tot=m.sum()
    if not tot: return []
    out=[]
    for name,(cx,cy) in CELLS.items():
        s=m[cy*h//3:(cy+1)*h//3, cx*w//3:(cx+1)*w//3].sum()
        if s>0.05*tot: out.append(name)
    return out
def tags_line(r):
    t=r['pred_tags']; a=r.get('pred_answer') or '?'
    s=f"<answer> {a}"
    if t.get('type'): s+=f", <type> {t['type']}"
    if t.get('location'): s+=f", <location> {t['location']}"
    return s
def reasoning(r):
    t=r['pred_tags'].get('reasoning')
    if t: return t
    f=r.get('pred_full') or ''
    f=re.sub(r'<answer>.*?</answer>','',f,flags=re.S); f=re.sub(r'</?think>','',f); return f.strip()
FIND=re.compile(r"(there is|there are|i see|i find|i notice|i can see|is visible|i spot|shows a|has a|reveals|has been|is a clear|is a significant|however)",re.I)
SCAN=re.compile(r"(i'll|i will|scan|checking|check for|look for|looking for|inspect for|examine|examining|start by|begin by)",re.I)
def decisive(sents,pred):
    if pred!='yes' or not sents: return key_sentence(sents,'sft')
    best=None
    for i,s in enumerate(sents):
        sc=0
        if KEY.search(s): sc+=2
        else: sc-=5
        if FIND.search(s): sc+=2
        if SCAN.search(s) and not re.search(r"however",s,re.I): sc-=3
        if re.search(r"\b(no|without|free of)\b[^.]{0,40}"+KEY.pattern,s,re.I): sc-=4
        if best is None or sc>best[0]: best=(sc,i)
    return best[1] if best and best[0]>0 else key_sentence(sents,'kcr')
HL={'good':"#c9ecd5",'bad':"#fbd5d7",'weak':"#fdebc0"}
# ---------------------------------------------------------------- card grid (F.1, F.2)
def card(o,x,y,cw,rec,status,cells,mode_label,ql=4,fixed_h=None):
    """image on top, product line, verdict mark, tags, decisive sentence shaded"""
    IMG=cw-2
    frame_at=len(o); image_panel(o,x+1,y+1,IMG,rec,cells,[])
    sents=sentences(reasoning(rec)); ki=decisive(sents,rec.get('pred_answer'))
    q=sents[ki] if sents else ''
    lines=wrap_words([q],cw-24)[:ql]
    if len(wrap_words([q],cw-24))>ql: lines[-1]=lines[-1][:-1]+[('…',0)]
    tl=wrap_words([tags_line(rec)],cw-24)
    h=IMG+2+26+18*len(tl)+8+18*len(lines)+14
    if fixed_h: h=fixed_h-18
    h+=18
    o.insert(frame_at,rect(x,y,cw,h,fill="#fff",stroke=C['rule2'],sw=1.2,rx=8))
    o.append(txt(x+12,y+IMG+22,f"{rec['sub_dataset']} / {rec['product']}","t13 b"))
    col={'good':C['good'],'bad':C['bad'],'weak':"#d4a017"}[status]
    sub=rec['image_id'].split('/')[-2] if rec['sub_dataset']=='DS-MVTec' else ('bad' if rec['gt_answer']=='yes' else 'good')
    o.append(txt(x+12,y+IMG+40,sub,"t13 mut"))
    bw=tw(mode_label)+16; o.append(rect(x+8,y+8,bw,22,fill=col,rx=11)); o.append(txt(x+8+bw/2,y+24,mode_label,"t13 b c",fill="#fff"))
    ty=y+IMG+2+26+18
    for j,ln in enumerate(tl): o.append(txt(x+12,ty+18*j+14,' '.join(w for w,_ in ln),"t13 mut"))
    ty+=18*len(tl)+8
    for j,ln in enumerate(lines):
        s=' '.join(w for w,_ in ln); by=ty+18*j+14
        o.append(rect(x+10,by-13,tw(s)+4,17,fill=HL[status],rx=2)); o.append(txt(x+12,by,s,"t13"))
    return h
def grid(cards,cols,out_name,title=None):
    W=980; MX=24; GAP=14; cw=(W-2*MX-(cols-1)*GAP)/cols
    body=[]; y=24; rows=[cards[i:i+cols] for i in range(0,len(cards),cols)]
    for row in rows:
        hs=[card([],MX+j*(cw+GAP),y,cw,*c) for j,c in enumerate(row)]; hr=max(hs)
        for j,c in enumerate(row): card(body,MX+j*(cw+GAP),y,cw,*c,fixed_h=hr)
        y+=hr+GAP
    H=y+10
    o=[head(W,H),DEFS]+body+[foot()]; open(out_name,'w').write(''.join(o)); return H
# ---------------------------------------------------------------- selection
def pick_f1(kcr):
    sh=json.load(open(f'{W_}/outputs/explainability_shared/raw_results_all.json')); own=json.load(open(f'{W_}/outputs/explainability_multi/raw_results.json'))
    kj={}
    for src in (own,sh):
        for r in src:
            if r['model']=='armC_finalsft': kj[r['image_id']]=r
    fam=['cable','bottle','zipper','wood','carpet','grid','pill','pcb1','pcb3','fryum','candle','macaroni1']
    pool=collections.defaultdict(list)
    for iid,r in kj.items():
        if r['j_overall']>=9.5 and kcr[iid]['pred_answer']=='yes': pool[r['product']].append(r)
    sel=[]
    for p in fam:
        rs=sorted(pool[p],key=lambda r:(-r['loc_met'],-(r['type_sim'] or 0),r['image_id']))
        sel.append(rs[0]['image_id'])
    return sel
def pick_f2(kcr):
    sh=json.load(open(f'{W_}/outputs/explainability_shared/raw_results_all.json')); own=json.load(open(f'{W_}/outputs/explainability_multi/raw_results.json'))
    kj={}
    for src in (own,sh):
        for r in src:
            if r['model']=='armC_finalsft': kj[r['image_id']]=r
    weak=[r['image_id'] for r in sorted(kj.values(),key=lambda r:(r['j_overall'],r['image_id'])) if r['j_overall']<=5 and kcr[r['image_id']]['pred_answer']=='yes']
    # one product each, lowest score first
    seen=set(); w4=[]
    for iid in weak:
        p=iid.split('/')[1]
        if p in seen: continue
        seen.add(p); w4.append(iid)
        if len(w4)==4: break
    fp=collections.defaultdict(list); fn=collections.defaultdict(list)
    for iid,r in kcr.items():
        if r['gt_answer']=='no' and r['pred_answer']!='no' and r['pred_tags'].get('reasoning') and iid!='VisA/macaroni2/test/good/0016.JPG': fp[r['product']].append(iid)
        if r['gt_answer']=='yes' and r['pred_answer']!='yes' and r['pred_tags'].get('reasoning') and mask_for(r['absolute_path']): fn[r['product']].append(iid)
    fp8=[sorted(fp[p])[0] for p in ['macaroni2','pcb3','capsules','transistor','pcb4','hazelnut','capsule','bottle']]
    fn8=[sorted(fn[p])[0] for p in ['pcb1','pcb2','cable','zipper','macaroni1','screw','candle','cashew']]
    return fp8,fn8,w4
# ---------------------------------------------------------------- F.3 triptych
def triptych(iid,base,sft,grpo,out_name):
    W=980; IMG=286; x1=24+IMG+18; cwid=(W-x1-24-2*12)/3
    recs=[("Base",C['base'],C['base_f'],base[iid]),("SFT",C['sft'],C['sft_f'],sft[iid]),("SFT+GRPO",C['grpo'],C['grpo_f'],grpo[iid])]
    cells=gt_cells(sft[iid]); body=[]; hs=[]
    for i,(nm,col,fill,r) in enumerate(recs):
        sents=sentences(reasoning(r)); ki=decisive(sents,r['pred_answer']); ok=(r['pred_answer']==r['gt_answer'])
        hs.append(trace_card([],x1+i*(cwid+12),24,cwid,nm,col,fill,tags_line(r),sents,ki,"tint",hl_fill=HL['good' if ok else 'bad']))
    hc=max(hs)
    for i,(nm,col,fill,r) in enumerate(recs):
        sents=sentences(reasoning(r)); ki=decisive(sents,r['pred_answer']); ok=(r['pred_answer']==r['gt_answer'])
        trace_card(body,x1+i*(cwid+12),24,cwid,nm,col,fill,tags_line(r),sents,ki,"tint",hl_fill=HL['good' if ok else 'bad'],fixed_h=hc)
    H=24+max(hc,IMG+70)+24; o=[head(W,H),DEFS]
    image_panel(o,24,24,IMG,sft[iid],cells,[])
    r=sft[iid]; o.append(txt(24,24+IMG+22,f"{r['product']}, ground truth: defective","t13 b")); o.append(txt(24,24+IMG+40,"shaded cell: annotated location","t13 mut")); o.append(txt(24,24+IMG+58,"arrow: the defect itself","t13 mut"))
    o+=body; o.append(foot()); open(out_name,'w').write(''.join(o))
# ---------------------------------------------------------------- F.4 overlays
def overlays(grpo,ref,out_name,alpha=0.30):
    b={k:v for k,v in grpo.items() if v['sub_dataset']=='DS-MVTec'}; r=ref; ids=sorted(set(b)&set(r)); used=set()
    def find(gt,base_ok,ref_ok,need_mask=False,prefer=()):
        order=list(prefer)+[p for p in sorted({b[i]['product'] for i in ids}) if p not in prefer]
        for prod in order:
            if prod in used: continue
            for i in ids:
                rb,rr=b[i],r[i]
                if rb['product']!=prod or rb['gt_answer']!=gt or (rb['pred_answer']==gt)!=base_ok or (rr['pred_answer']==gt)!=ref_ok: continue
                if need_mask and not mask_for(rb['absolute_path']): continue
                used.add(prod); return rb,rr
        return None,None
    cases=[("both correct",find("yes",True,True,True,prefer=('tile',))),("normal part, over-predicted by the refinement",find("no",True,False,prefer=('transistor','pill'))),("both miss",find("yes",False,False,True,prefer=('cable','screw'))),("refinement catches, SFT+GRPO misses",find("yes",False,True,True,prefer=('zipper','capsule','hazelnut')))]
    W=980; MX=24; GAP=16; cw=(W-2*MX-GAP)/2; IMG=cw-2; o=[head(W,10)]; y=24; ch=IMG+2+26+22+22+22+12
    for k,(label,(rb,rr)) in enumerate(cases):
        x=MX+(k%2)*(cw+GAP); yy=y+(k//2)*(ch+GAP)
        o.append(rect(x,yy,cw,ch,fill="#fff",stroke=C['rule2'],sw=1.2,rx=8))
        im=Image.open(rb['absolute_path']).convert('RGB'); W0,H0=im.size; side=min(W0,H0); box=((W0-side)//2,(H0-side)//2,(W0-side)//2+side,(H0-side)//2+side); im=im.crop(box)
        mp=mask_for(rb['absolute_path'])
        if mp:
            m=Image.open(mp).convert('L').crop(box).resize(im.size).point(lambda v:255 if v>0 else 0); ov=Image.new('RGB',im.size,(217,54,62)); im=Image.composite(Image.blend(im,ov,alpha),im,m)
        im.thumbnail((520,520)); bio=io.BytesIO(); im.save(bio,'JPEG',quality=85); uri="data:image/jpeg;base64,"+base64.b64encode(bio.getvalue()).decode()
        o.append(f'<image x="{x+1}" y="{yy+1}" width="{IMG}" height="{IMG}" href="{uri}"/>')
        gt='defective' if rb['gt_answer']=='yes' else 'normal'
        o.append(txt(x+12,yy+IMG+22,f"{rb['product']} / {rb['image_id'].split('/')[-2]} · ground truth {gt}","t13 b"))
        o.append(txt(x+12,yy+IMG+44,label,"t13 mut"))
        def verdict(name,rec,xx,yv):
            ok=rec['pred_answer']==rec['gt_answer']; o.append(txt(xx,yv,f"{name}: {rec['pred_answer']}","t13")); o.append(txt(xx+tw(f"{name}: {rec['pred_answer']}")+8,yv,"correct" if ok else "wrong","t13 b",fill=C['good'] if ok else C['bad']))
        verdict("SFT+GRPO",rb,x+12,yy+IMG+66); verdict("refinement",rr,x+12,yy+IMG+88)
    H=y+2*ch+GAP+10; o[0]=head(W,H); o.append(foot()); open(out_name,'w').write(''.join(o)); return cases
# ---------------------------------------------------------------- G teacher vs student
def teacher_student(iid,gem,stu,out_name):
    W=980; IMG=286; x1=24+IMG+18; cwid=(W-x1-24-14)/2
    t=gem[iid]; s=stu[iid]; gt=s['gt_answer']; cells=gt_cells(s) if gt=='yes' else []
    tr=reasoning(t); bare=not tr; tr=tr or "(bare <answer> tag, no reasoning)"; ts=sentences(tr); ks=-1 if bare else (decisive(ts,t['pred_answer']) if t['pred_answer']=='yes' else len(ts)-1)
    ss=sentences(reasoning(s)); kk=decisive(ss,s['pred_answer'])
    tok=t['pred_answer']==gt; sok=s['pred_answer']==gt
    h1=trace_card([],x1,24,cwid,"Teacher, Gemini 2.5 Flash",C['purple'],C['purple_f'],tags_line(t),ts,ks,"tint",hl_fill=HL['good' if tok else 'bad'])
    h2=trace_card([],x1+cwid+14,24,cwid,"Student, SFT+GRPO",C['grpo'],C['grpo_f'],tags_line(s),ss,kk,"tint",hl_fill=HL['good' if sok else 'bad'])
    hc=max(h1,h2); o=[head(W,24+max(hc,IMG+70)+24),DEFS]
    image_panel(o,24,24,IMG,s,cells,[])
    o.append(txt(24,24+IMG+22,f"{s['product']}, ground truth: {'defective' if gt=='yes' else 'normal'}","t13 b"))
    if gt=='yes': o.append(txt(24,24+IMG+40,"shaded cell: annotated location","t13 mut")); o.append(txt(24,24+IMG+58,"arrow: the defect itself","t13 mut"))
    else: o.append(txt(24,24+IMG+40,"no annotation, the part is clean","t13 mut"))
    trace_card(o,x1,24,cwid,"Teacher, Gemini 2.5 Flash",C['purple'],C['purple_f'],tags_line(t),ts,ks,"tint",hl_fill=HL['good' if tok else 'bad'],fixed_h=hc)
    trace_card(o,x1+cwid+14,24,cwid,"Student, SFT+GRPO",C['grpo'],C['grpo_f'],tags_line(s),ss,kk,"tint",hl_fill=HL['good' if sok else 'bad'],fixed_h=hc)
    o.append(foot()); open(out_name,'w').write(''.join(o))
def pdf(name):
    subprocess.run([CAIRO,f'{name}.svg','-o',f'{OUT}/{name}.pdf'],check=True); print('installed',f'{OUT}/{name}.pdf')
if __name__=='__main__':
    which=sys.argv[1:] or ['f1','f2','f3','f4','g']
    kcr=load('kcr'); report={}
    if 'f1' in which:
        sel=pick_f1(kcr); cards=[(kcr[i],'good',gt_cells(kcr[i]),'correct') for i in sel]
        grid(cards,4,'app_f1_success.svg'); pdf('app_f1_success'); report['f1']=sel
    if 'f2' in which:
        fp8,fn8,w4=pick_f2(kcr)
        grid([(kcr[i],'bad',[],'false alarm') for i in fp8],4,'app_f2_false_alarms.svg'); pdf('app_f2_false_alarms')
        grid([(kcr[i],'bad',gt_cells(kcr[i]),'missed defect') for i in fn8]+[(kcr[i],'weak',gt_cells(kcr[i]),'weak explanation') for i in w4],4,'app_f2_misses.svg'); pdf('app_f2_misses')
        report['f2']={'fp':fp8,'fn':fn8,'weak':w4}
    if 'f3' in which:
        base=load('base'); sft=load('sft'); grpo=load('grpo'); iid=os.environ.get('F3_ID','DS-MVTec/cable/image/poke_insulation/001.png')
        triptych(iid,base,sft,grpo,'app_f3_triptych.svg'); pdf('app_f3_triptych'); report['f3']=iid
    if 'f4' in which:
        grpo=load('grpo'); ref=load('ref'); cases=overlays(grpo,ref,'app_f4_overlays.svg'); pdf('app_f4_overlays'); report['f4']=[(l,rb['image_id'],rb['pred_answer'],rr['pred_answer']) for l,(rb,rr) in cases]
    if 'g' in which:
        gem=load('gem'); stu=load('grpo')
        G=['DS-MVTec/bottle/image/good/005.png','DS-MVTec/capsule/image/faulty_imprint/016.png','DS-MVTec/hazelnut/image/cut/007.png','VisA/fryum/test/bad/046.JPG','VisA/macaroni2/test/good/0016.JPG']
        for k,iid in enumerate(G,1): teacher_student(iid,gem,stu,f'app_g{k}.svg'); pdf(f'app_g{k}')
        report['g']=G
    old=json.load(open('app_examples_selection.json')) if os.path.exists('app_examples_selection.json') else {}; old.update(report); report=old
    json.dump(report,open('app_examples_selection.json','w'),indent=1); print(json.dumps(report,indent=1))
