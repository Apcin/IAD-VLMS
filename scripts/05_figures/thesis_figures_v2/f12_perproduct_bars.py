"""Per-product balanced accuracy as standing bars in the house style: one stage against KCR, DS-MVTec and VisA panels.
Usage: python f12_perproduct_bars.py Base|SFT|SFT+GRPO   -> perproduct_bars_<stage>_vs_kcr.svg (+pdf into THESIS_FIGURES)"""
import sys,os,json,collections,subprocess
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
W_=os.environ.get('WORK_DIR','.'); OUT=os.environ.get('THESIS_FIGURES','thesis'); CAIRO=os.environ.get('CAIROSVG','cairosvg')
O=f'{W_}/outputs/'
F={'Base':{'DS-MVTec':O+'baseline_eval/7b_base_dsmvtec.json','VisA':O+'baseline_eval/7b_base_visa.json'},
   'SFT':{'DS-MVTec':O+'sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json','VisA':O+'sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_visa_full_trainprompt.json'},
   'SFT+GRPO':{'DS-MVTec':O+'grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json','VisA':O+'grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_visa_full_trainprompt.json'},
   'KCR':{'DS-MVTec':O+'sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json','VisA':O+'sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_visa_full_trainprompt.json'}}
OVERALL={'Base':(69.08,53.79),'SFT':(80.16,64.78),'SFT+GRPO':(82.73,70.39),'KCR':(82.80,72.07)}
def perprod(p):
    r=json.load(open(p))['results']; d=collections.defaultdict(lambda:[0,0,0,0])
    for x in r:
        k=x['product']
        if x['gt_answer']=='yes': d[k][0]+=1; d[k][1]+= x['pred_answer']=='yes'
        else: d[k][2]+=1; d[k][3]+= x['pred_answer']=='no'
    return {k:50*(v[1]/v[0]+v[3]/v[2]) for k,v in d.items()}
A=sys.argv[1] if len(sys.argv)>1 else 'Base'; B='KCR'
ca,fa=STAGE_C[A]; cb,fb=STAGE_C[B]
W=980; PW=932; PH=330; s=[]; y=24
for bi,bench in enumerate(('DS-MVTec','VisA')):
    pa=perprod(F[A][bench]); pb=perprod(F[B][bench]); prods=sorted(pa,key=lambda p:pb[p]-pa[p],reverse=True)
    x0=24; s.append(panel(x0,y,PW,PH,C['rule2'],rx=26,sw=2)); s.append(txt(x0+18,y+36,bench,'t18 b'))
    i=0 if bench=='DS-MVTec' else 1
    s.append(txt(x0+PW-18,y+36,f'overall {A} {OVERALL[A][i]:.2f}, {B} {OVERALL[B][i]:.2f}','t13 mut e'))
    ax=Axes(x0+62,y+62,PW-90,PH-180,0,len(prods),0,100)
    s.append(ax.frame([0,25,50,75,100],xticks=[],yfmt=lambda v:f'{v:.0f}',ylabel=None))
    s.append(ax.hline(50,C['rule2'],None,dash='4 4'))
    gw=ax.w/len(prods); bw=min(20,(gw-14)/2)
    for gi,p in enumerate(prods):
        gx=ax.x+gi*gw; xa=gx+gw/2-bw-2; xb=gx+gw/2+2
        for xx,v,col,fil in ((xa,pa[p],C[ca],C[fa]),(xb,pb[p],C[cb],C[fb])):
            s.append(rect(xx,ax.Y(v),bw,ax.Y(0)-ax.Y(v),fill=fil,stroke=col,sw=1.5,rx=2))
        d=pb[p]-pa[p]; s.append(txt(gx+gw/2,ax.Y(max(pa[p],pb[p]))-8,f'{d:+.1f}','t13 b c',fill=C['good'] if d>=0 else C['bad']))
        s.append(f'<text x="{gx+gw/2:.1f}" y="{ax.y+ax.h+14}" class="t13" text-anchor="end" transform="rotate(-38 {gx+gw/2:.1f} {ax.y+ax.h+14})">{esc(p)}</text>')
    s.append(txt(x0+30,y+PH-16,'balanced accuracy, %. Label above each pair: KCR minus '+A+', percentage points. Products sorted by gain.','t13 mut'))
    if bi==0: s.append(legend_row(x0+PW-420,y+36,[(A,C[ca],C[fa]),(B,C[cb],C[fb])],gapx=120,swatch='rect'))
    y+=PH+18
name=f'perproduct_bars_{A.replace("+","plus").lower()}_vs_kcr'
open(name+'.svg','w').write(head(W,y+6,f'Per-product balanced accuracy, {A} against KCR')+''.join(s)+foot())
subprocess.run([CAIRO,name+'.svg','-o',f'{OUT}/{name}.pdf'],check=True); print('wrote',name)
