"""Alternative representations of the Chapter 6 figures, same data, same kit. Output alt_*.svg"""
import sys,os,json,re; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
D=json.load(open('figdata.json')); L=D['ladder']; R=D['refs']
ST=['Base','SFT','SFT+GRPO','KCR']; SC={'Base':C['base'],'SFT':C['sft'],'SFT+GRPO':C['grpo'],'KCR':C['kcr']}
BB={'Qwen2.5-VL-7B':('Qwen2.5-VL-7B',None),'LLaVA-OneVision-7B':('LLaVA-OneVision-7B','7 5')}
def save(name,s,W,H): open(name,'w').write(head(W,H)+''.join(s)+foot())
# ---------------------------------------------------------------- 1. ladder as dot-and-line, benchmarks side by side
def alt_ladder():
    W=980; PW,PH=440,330; PY=24; GX=36; s=[]
    for k,(bench,key,lo,hi,ticks) in enumerate((('DS-MVTec','ds',50,90,[50,60,70,80,90]),('VisA','visa',50,90,[50,60,70,80,90]))):
        x0=24+k*(PW+GX); s.append(panel(x0,PY,PW,PH+150,C['rule2'],rx=26,sw=2)); s.append(txt(x0+18,PY+36,bench,'t18 b'))
        ax=Axes(x0+120,PY+70,PW-150,PH-30,-0.4,3.4,lo,hi); s.append(ax.frame(ticks,ylabel='balanced accuracy %'))
        for i,st in enumerate(ST): s.append(txt(ax.X(i),ax.y+ax.h+22,st,'t13 c b'))
        rv=R['IAD-R1'][0 if key=='ds' else 1]; s.append(ax.hline(rv,C['purple'],None)); s.append(txt(ax.x+4,ax.Y(rv)-6,f'IAD-R1 {rv:.2f}','t13',fill=C['purple']))
        for bb,(lab,dash) in BB.items():
            pts=[(i,L[bb][st][key]['ba']) for i,st in enumerate(ST)]
            s.append(ax.poly(pts,C['ink'] if dash is None else C['mut'],2.2,dash=dash))
            for i,st in enumerate(ST):
                v=L[bb][st][key]['ba']; s.append(circ(ax.X(i),ax.Y(v),7 if dash is None else 5,SC[st],stroke='#fff',sw=2))
        for r,(bb,(lab,dash)) in enumerate(BB.items()):
            yy=PY+PH+94+r*22; s.append(txt(ax.x-16,yy,'Qwen, solid' if dash is None else 'LLaVA, dashed','t13 e mut'))
            for i,st in enumerate(ST): s.append(txt(ax.X(i),yy,f"{L[bb][st][key]['ba']:.2f}",'t13 c b' if dash is None else 't13 c'))
    save('alt_ladder.svg',s,W,PY+PH+150+24)
# ---------------------------------------------------------------- 2. SFT grid as a heat map
def alt_sft():
    GRID=[("3B","Frozen","6K",69.08,57.22),("3B","Frozen","15K",69.56,59.65),("3B","Unfrozen","15K",68.58,59.60),("7B","Frozen","15K",72.60,66.94),("7B","Frozen","6K",80.16,64.78),("7B","Unfrozen","15K",72.08,58.60)]
    W=980; s=[]; x0=24; y0=24; s.append(panel(x0,y0,W-48,470,C['rule2'],rx=26,sw=2)); s.append(txt(x0+18,y0+36,'Best epoch per configuration','t18 b'))
    cols=[('DS-MVTec',3),('VisA',4)]; rw=54; cw=200; gx=x0+330; gy=y0+90
    def shade(v):  # 50 -> pale, 85 -> full purple
        t=max(0,min(1,(v-50)/35)); r=int(226+(155-226)*t); g=int(217+(63-217)*t); b=int(236+(191-236)*t); return f'#{r:02x}{g:02x}{b:02x}'
    for j,(nm,_) in enumerate(cols): s.append(txt(gx+j*cw+cw/2,gy-14,nm,'t15 b c'))
    for i,(sz,fr,dat,ds,va) in enumerate(GRID):
        y=gy+i*rw; s.append(txt(gx-14,y+rw/2+5,f"{sz}  {fr}  {dat}",'t15 e' + (' b' if (sz,fr,dat)==('7B','Frozen','6K') else '')))
        for j,(nm,idx) in enumerate(cols):
            v=(ds,va)[j]; s.append(rect(gx+j*cw,y,cw-6,rw-6,fill=shade(v),rx=6)); s.append(txt(gx+j*cw+(cw-6)/2,y+rw/2+6,f"{v:.2f}",'t16 b c',fill='#fff' if v>70 else C['ink']))
    s.append(rect(gx-8,gy+4*rw-4,2*cw+10,rw+2,fill='none',stroke=C['purple'],sw=2.5,rx=8,extra='stroke-dasharray="7 5"')); s.append(txt(gx+2*cw+10,gy+4*rw+rw/2+5,'headline','t13 b',fill=C['purple']))
    s.append(txt(x0+18,y0+440,'shade: balanced accuracy from 50 (pale) to 85 (dark). Cells are best epochs on DS-MVTec, VisA at that epoch.','t13 mut'))
    save('alt_sft.svg',s,W,y0+470+24)
# ---------------------------------------------------------------- 3. arms as dumbbells, best epoch only, both benchmarks
def alt_arms():
    A=D['arms']; W=980; s=[]; x0=24; y0=24; PH=500; s.append(panel(x0,y0,W-48,PH,C['rule2'],rx=26,sw=2)); s.append(txt(x0+18,y0+36,'Teacher-intervention arms at the DS-MVTec-selected epoch','t18 b'))
    ax=Axes(x0+250,y0+110,W-48-300,PH-240,62,86,0,3); s.append(ax.frame([],xticks=[65,70,75,80,85],xgrid=True,xfmt=lambda v:f'{v}'))
    ACOL={'A, rejection sampling':C['base'],'B, STaR':C['sft'],'C, KCR':C['kcr']}
    for lab,(vals,col) in {'SFT':(R['SFT headline'],C['purple']),'SFT+GRPO':(R['SFT+GRPO headline'],C['grpo'])}.items():
        for v in vals: s.append(line(ax.X(v),ax.y,ax.X(v),ax.y+ax.h,col,1.6,dash='7 5'))
        k=0 if lab=='SFT' else 1
        s.append(txt(ax.X(vals[0]),ax.y-10-18*k,f'{lab} DS {vals[0]:.2f}','t13 c',fill=col)); s.append(txt(ax.X(vals[1]),ax.y+ax.h+44+18*k,f'{lab} VisA {vals[1]:.2f}','t13 c',fill=col))
    for i,(arm,d) in enumerate(A.items()):
        best=max(range(4),key=lambda e:d['ep'][e][0]); ds,va=d['ep'][best]; yy=ax.Y(2.5-i)
        s.append(txt(ax.x-14,yy+5,f"{arm} ({d['n']:,})",'t15 e b' if 'KCR' in arm else 't15 e'))
        s.append(line(ax.X(va),yy,ax.X(ds),yy,ACOL[arm],4)); s.append(circ(ax.X(va),yy,8,'#fff',stroke=ACOL[arm],sw=3)); s.append(circ(ax.X(ds),yy,8,ACOL[arm],stroke='#fff',sw=2))
        s.append(txt(ax.X(va)-12,yy+5,f'{va:.2f}','t13 e')); s.append(txt(ax.X(ds)+12,yy+5,f'{ds:.2f}  ep {best+1}','t13'))
    s.append(txt(x0+18,y0+PH-24,'open circle: VisA    filled circle: DS-MVTec    dashed verticals: the SFT and SFT+GRPO headlines','t13 mut'))
    save('alt_arms.svg',s,W,y0+PH+24)
# ---------------------------------------------------------------- 4. iter2 as paired dots
def alt_iter2():
    I=json.load(open('iter2_perprod.json')); rows=sorted(I['rows'],key=lambda r:r[2]-r[1]); ova,ovb=I['overall']
    W=980; s=[]; x0=24; y0=24; PH=690; s.append(panel(x0,y0,W-48,PH,C['rule2'],rx=26,sw=2)); s.append(txt(x0+18,y0+36,'SFT+GRPO checkpoint against the refinement, per product on DS-MVTec','t18 b'))
    ax=Axes(x0+170,y0+80,W-48-340,PH-190,55,100,0,len(rows)+1); s.append(ax.frame([],xticks=[60,70,80,90,100],xgrid=True,xfmt=lambda v:f'{v}'))
    for i,(p,a,b) in enumerate(rows+[('all products',ova,ovb)]):
        yy=ax.Y(len(rows)-i+0.5) if i<len(rows) else ax.Y(0.3); d=b-a; col=C['good'] if d>0 else (C['bad'] if d<0 else C['mut'])
        s.append(txt(ax.x-14,yy+5,p,'t15 e b' if p=='all products' else 't15 e'))
        s.append(line(ax.X(a),yy,ax.X(b),yy,col,3)); s.append(circ(ax.X(a),yy,6,C['grpo'],stroke='#fff',sw=2)); s.append(circ(ax.X(b),yy,6,C['purple'],stroke='#fff',sw=2))
        s.append(txt(ax.x+ax.w+16,yy+5,f'{d:+.1f}','t15 b' if abs(d)>=3 or p=='all products' else 't15',fill=col))
    s.append(txt(ax.x+ax.w+16,ax.y-14,'change','t13 mut'))
    s.append(legend_row(x0+18,y0+PH-28,[('SFT+GRPO ckpt-530',C['grpo']),('refinement, epoch 1',C['purple'])],gapx=230,swatch='dot'))
    save('alt_iter2.svg',s,W,y0+PH+24)
# ---------------------------------------------------------------- 5. decoupling as reward vs accuracy scatter
def alt_decoupling():
    RT=os.environ.get('WORK_DIR','..')+'/outputs'; ARMS=[("ctrl","CTRL, group z-score",C["grpo"]),("drgrpo","Dr.GRPO, no std",C["sft"]),("g2rpo","G2RPO, rank",C["kcr"])]; STEPS=[20,40,60,80,100,120]
    def ba(p):
        try: m=json.load(open(p))['metrics']
        except Exception: return None
        tp,tn,fp,fn=(m.get(k,0) for k in ('tp','tn','fp','fn')); return 50*(tp/(tp+fn)+tn/(tn+fp))
    def rew(arm):
        t=open(f"{RT}/grpo_probe_{arm}/train.log",errors='ignore').read().replace("\r","\n")
        return [float(m.group(1)) for l in t.split("\n") if "'reward':" in l for m in [re.search(r"'reward': '?([0-9.eE+-]+)",l)] if m]
    W=980; s=[]; x0=24; y0=24; PH=580; s.append(panel(x0,y0,W-48,PH,C['rule2'],rx=26,sw=2)); s.append(txt(x0+18,y0+36,'Reward against probe accuracy, one point per checkpoint','t18 b'))
    ax=Axes(x0+90,y0+90,W-48-150,PH-220,1.9,2.6,80.5,86); s.append(ax.frame([81,82,83,84,85,86],xticks=[2.0,2.2,2.4,2.6],xgrid=True,xfmt=lambda v:f'{v:.1f}',ylabel='DS-MVTec probe balanced accuracy %',xlabel='mean group reward, 20-step window before the checkpoint'))
    s.append(ax.hline(84.52,C['mut'],None)); s.append(txt(ax.x+ax.w-4,ax.Y(84.52)-6,'KCR initialisation 84.52','t13 e mut'))
    for arm,lab,col in ARMS:
        rv=rew(arm); pts=[]
        for st in STEPS:
            b=ba(f"{RT}/grpo_probe_{arm}/checkpoint-{st}/probe_dsmvtec.json")
            if b is None or len(rv)<st: continue
            pts.append((sum(rv[st-20:st])/20,b,st))
        s.append(ax.poly([(a,b) for a,b,_ in pts],col,1.6,dash='4 4',op=0.7))
        for a,b,st in pts:
            s.append(circ(ax.X(a),ax.Y(b),6+(st==120)*3,col,stroke='#fff',sw=2))
            if st==20: s.append(txt(ax.X(a)+10,ax.Y(b)-9,'step 20','t13',fill=col))
    s.append(legend_row(x0+18,y0+PH-26,[(l,c) for _,l,c in ARMS],gapx=300,swatch='dot')); s.append(txt(x0+18,y0+PH-52,'large dot: step 120. Dashed lines join checkpoints in step order.','t13 mut'))
    save('alt_decoupling.svg',s,W,y0+PH+24)
# ---------------------------------------------------------------- 6. operating point as TPR vs TNR
def alt_operating():
    W=980; PW,PH=440,470; PY=24; GX=36; s=[]
    for k,(bench,key) in enumerate((('DS-MVTec','ds'),('VisA','visa'))):
        x0=24+k*(PW+GX); s.append(panel(x0,PY,PW,PH,C['rule2'],rx=26,sw=2)); s.append(txt(x0+18,PY+36,bench,'t18 b'))
        ax=Axes(x0+70,PY+70,PW-110,PH-160,0,100,50,100); s.append(ax.frame([50,60,70,80,90,100],xticks=[0,25,50,75,100],xgrid=True,ylabel='specificity %',xlabel='sensitivity %'))
        for b in (60,70,80,90):  # iso-balanced-accuracy lines: tnr = 2b - tpr
            pts=[(tpr,2*b-tpr) for tpr in range(0,101,5) if 50<=2*b-tpr<=100]
            if len(pts)>1:
                s.append(ax.poly(pts,C['rule'],1,dash='3 4')); m=pts[len(pts)//4]
                s.append(rect(ax.X(m[0])-22,ax.Y(m[1])-9,44,16,fill='#fff')); s.append(txt(ax.X(m[0]),ax.Y(m[1])+4,f'BA {b}','t13 mut c'))
        for bb,(lab,dash) in BB.items():
            pts=[(L[bb][st][key]['tpr'],L[bb][st][key]['tnr']) for st in ST]
            s.append(ax.poly(pts,C['ink'] if dash is None else C['mut'],1.8,dash=dash,op=0.8))
            for st,(a,b) in zip(ST,pts): s.append(circ(ax.X(a),ax.Y(b),7 if dash is None else 5,SC[st],stroke='#fff' if dash is None else SC[st],sw=2))
        s.append(txt(x0+18,PY+PH-22,'large, solid: Qwen   small, dashed: LLaVA   diagonals: equal BA','t13 mut'))
    s.append(legend_row(24,PY+PH+30,[(st,SC[st]) for st in ST],gapx=180,swatch='dot'))
    save('alt_operating.svg',s,W,PY+PH+60)
# ---------------------------------------------------------------- 7. per-product as heat map
def alt_perproduct():
    P=D['perproduct']; W=980; s=[]; y=24
    def shade(v):
        t=max(0,min(1,(v-50)/45)); r=int(240+(58-240)*t); g=int(246+(125-246)*t); b=int(240+(58-240)*t); return f'#{r:02x}{g:02x}{b:02x}'
    for bench in ('DS-MVTec','VisA'):
        for bb in ('Qwen2.5-VL-7B','LLaVA-OneVision-7B'):
            rows=P[bench][bb]['rows']; rh=24; ph=90+len(rows)*rh+30; x0=24 if bb.startswith('Qwen') else 24+466+20
            if bb.startswith('Qwen'): yb=y
            s.append(panel(x0,yb,466,ph,C['rule2'],rx=20,sw=2)); s.append(txt(x0+16,yb+30,bb,'t16 b')); s.append(txt(x0+450,yb+30,bench,'t13 mut e'))
            gx=x0+120; cw=76
            for j,st in enumerate(('SFT','SFT+GRPO','KCR')): s.append(txt(gx+j*cw+cw/2,yb+58,st,'t13 b c'))
            s.append(txt(gx+3*cw+40,yb+58,'KCR-SFT','t13 b c'))
            for i,r in enumerate(rows):
                yy=yb+70+i*rh; s.append(txt(gx-8,yy+16,r['product'],'t13 e'))
                for j,st in enumerate(('SFT','SFT+GRPO','KCR')):
                    v=r[st]; s.append(rect(gx+j*cw,yy+2,cw-3,rh-4,fill=shade(v),rx=3)); s.append(txt(gx+j*cw+(cw-3)/2,yy+16,f'{v:.1f}','t13 c',fill='#fff' if v>82 else C['ink']))
                d=r['KCR']-r['SFT']; s.append(txt(gx+3*cw+40,yy+16,f'{d:+.1f}','t13 c b',fill=C['good'] if d>0 else C['bad']))
            if not bb.startswith('Qwen'): y=yb+ph+20
    s.append(txt(24,y+4,'shade: balanced accuracy from 50 (white) to 95 (dark green)','t13 mut'))
    save('alt_perproduct.svg',s,W,y+30)
# ---------------------------------------------------------------- 8. dynamics as two panels
def alt_dynamics():
    K=json.load(open('grpo_curve.json'))
    def smooth(idx,w=25):
        v=[(k[0],k[idx]) for k in K if k[idx] is not None]; return [(x,sum(b for _,b in v[max(0,i-w+1):i+1])/len(v[max(0,i-w+1):i+1])) for i,(x,_) in enumerate(v)]
    def raw(idx): return [(k[0],k[idx]) for k in K if k[idx] is not None]
    W=980; PW,PH=440,330; PY=24; GX=36; s=[]
    x0=24; s.append(panel(x0,PY,PW,PH,C['rule2'],rx=26,sw=2)); s.append(txt(x0+18,PY+36,'(a) Reward','t18 b')); s.append(txt(x0+PW-18,PY+36,'max 3.0','t13 mut e'))
    ax=Axes(x0+70,PY+70,PW-100,PH-140,0,1060,0.5,3.05); s.append(ax.frame([0.5,1.0,1.5,2.0,2.5,3.0],xticks=[0,265,530,795,1060],xfmt=str,yfmt=lambda v:f'{v:.1f}',xlabel='training step'))
    s.append(ax.hline(3.0,C['bad'],None,dash='4 4')); s.append(ax.poly(raw(1),C['kcr'],0.8,op=0.25)); s.append(ax.poly(smooth(1),C['kcr'],2.6)); s.append(ax.poly(smooth(3),C['grpo'],2.2)); s.append(ax.poly(smooth(4),C['sft'],2.2))
    x1=x0+PW+GX; s.append(panel(x1,PY,PW,PH,C['rule2'],rx=26,sw=2)); s.append(txt(x1+18,PY+36,'(b) KL to the reference policy','t18 b'))
    ax=Axes(x1+80,PY+70,PW-110,PH-140,0,1060,0,0.2); s.append(ax.frame([0,0.05,0.1,0.15,0.2],xticks=[0,265,530,795,1060],xfmt=str,yfmt=lambda v:f'{v:.2f}',xlabel='training step'))
    s.append(ax.poly(raw(2),C['purple'],0.8,op=0.25)); s.append(ax.poly(smooth(2),C['purple'],2.6))
    s.append(legend_row(24,PY+PH+30,[('total reward',C['kcr']),('accuracy',C['grpo']),('format',C['sft']),('KL',C['purple'])],gapx=180))
    save('alt_dynamics.svg',s,W,PY+PH+60)
if __name__=='__main__':
    for f in (alt_ladder,alt_sft,alt_arms,alt_iter2,alt_decoupling,alt_operating,alt_perproduct,alt_dynamics):
        try: f(); print('ok',f.__name__)
        except Exception as e: print('FAIL',f.__name__,repr(e))
