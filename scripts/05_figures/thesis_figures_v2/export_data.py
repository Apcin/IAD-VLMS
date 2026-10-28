import os
import json,sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
os.chdir((os.environ.get('WORK_DIR','/bulk/aacudad/reasoning_traces')+''))
from registry import REG
ST=['SFT','SFT+GRPO','KCR']
def load(p): return {r['image_id']:r for r in json.load(open(p))['results']}
def ba(rs):
    tp=sum(1 for r in rs if r['gt_answer']=='yes' and r['pred_answer']=='yes')
    fn=sum(1 for r in rs if r['gt_answer']=='yes' and r['pred_answer']!='yes')
    tn=sum(1 for r in rs if r['gt_answer']=='no'  and r['pred_answer']=='no')
    fp=sum(1 for r in rs if r['gt_answer']=='no'  and r['pred_answer']!='no')
    if not(tp+fn) or not(tn+fp): return None
    return dict(ba=50*(tp/(tp+fn)+tn/(tn+fp)), tpr=100*tp/(tp+fn), tnr=100*tn/(tn+fp),
                tp=tp,tn=tn,fp=fp,fn=fn,n=tp+tn+fp+fn)
out={"ladder":{},"perproduct":{}}
for bb in REG:
    out["ladder"][bb]={}
    for s in ['Base']+ST:
        row={}
        for bench in ('ds','visa'):
            p=REG[bb][s][bench]
            row[bench]=None if not p else ba(list(load(p).values()))
        out["ladder"][bb][s]=row
for bench,bn in (('ds','DS-MVTec'),('visa','VisA')):
    out["perproduct"][bn]={}
    for bb in REG:
        d={s:load(REG[bb][s][bench]) for s in ST}
        ids=set.intersection(*[set(d[s]) for s in ST])
        rows=[]
        for p in sorted({d['KCR'][i]['product'] for i in ids}):
            sub=[i for i in ids if d['KCR'][i]['product']==p]
            v={s:ba([d[s][i] for i in sub]) for s in ST}
            if None in v.values(): continue
            rows.append({"product":p,"n":len(sub),**{s:round(v[s]['ba'],2) for s in ST}})
        ov={s:round(ba([d[s][i] for i in ids])['ba'],2) for s in ST}
        out["perproduct"][bn][bb]={"rows":rows,"overall":ov,"n":len(ids)}
# teacher-ablation arms, from the thesis table (per-epoch, DS / VisA)
out["arms"]={
 "A, rejection sampling":{"n":2978,"ep":[[78.33,66.15],[78.69,69.35],[77.32,70.21],[79.01,68.83]]},
 "B, STaR":              {"n":4369,"ep":[[75.83,64.08],[78.84,67.44],[80.75,67.46],[79.46,68.73]]},
 "C, KCR":               {"n":6000,"ep":[[79.71,68.76],[82.80,72.07],[81.26,72.01],[78.96,71.53]]},
}
out["refs"]={"SFT headline":[80.16,64.78],"SFT+GRPO headline":[82.73,70.39],"IAD-R1":[81.92,71.34]}
json.dump(out,open((os.environ.get('WORK_DIR','/bulk/aacudad/reasoning_traces')+'/thesis_figures_v2/figdata.json'),'w'),indent=1)
print("wrote figdata.json")
for bb in out["ladder"]:
    print(f"  {bb}")
    for s in ['Base']+ST:
        r=out["ladder"][bb][s]
        f=lambda x: f"{x['ba']:6.2f} (tpr {x['tpr']:5.1f} tnr {x['tnr']:5.1f})" if x else "   n/a"
        print(f"    {s:9s} DS {f(r['ds'])}   VisA {f(r['visa'])}")
