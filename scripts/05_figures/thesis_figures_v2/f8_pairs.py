"""Figure 6.9 candidates: one image, the SFT trace and the KCR trace side by side, decisive sentence highlighted.
Usage: python f8_pairs.py  -> pairs_A_01.svg, pairs_B_01.svg, pairs_C_01.svg (+ _03 for variant A)"""
import json,sys,os,re,io,base64,glob
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
from PIL import Image
import numpy as np
from PIL import ImageFont
_F={}
def _font(px): 
    if px not in _F: _F[px]=ImageFont.truetype(os.environ.get('MEASURE_FONT','/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf'),px)
    return _F[px]
# class t13 renders at 15px (svgkit); cairosvg resolves Verdana to Noto Sans on this box, so widths are measured in Noto Sans
PX={13:15,15:17,16:19}
WSCALE=float(os.environ.get('WSCALE','1.0'))
def tw(s,px=13): return _font(PX.get(px,px)).getlength(s)*WSCALE
W_=os.environ.get('WORK_DIR','.')
SFT=f'{W_}/outputs/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_%s_full_trainprompt.json'
KCR=f'{W_}/outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_%s_full_trainprompt.json'
CELLS={"top-left":(0,0),"top-center":(1,0),"top-right":(2,0),"middle-left":(0,1),"center":(1,1),"middle-right":(2,1),"bottom-left":(0,2),"bottom-center":(1,2),"bottom-right":(2,2)}
KEY=re.compile(r"(indentation|missing|chip|chipped|broken|break in|crack|defect|scratch|dent|deform|bent|damage|fracture|hole|contamin)",re.I)
def load_pair(image_id,bench):
    s={r['image_id']:r for r in json.load(open(SFT%bench))['results']}[image_id]; k={r['image_id']:r for r in json.load(open(KCR%bench))['results']}[image_id]
    return s,k
def mask_for(abspath):
    p=abspath.split('/'); root='/'.join(p[:-3]); stem=os.path.splitext(p[-1])[0]
    g=glob.glob(f"{root}/mask/{p[-2]}/{stem}*.png")+glob.glob(f"{root}/ground_truth/{p[-2]}/{stem}*.png")
    return g[0] if g else None
def b64(im,px=520,q=85):
    im=im.copy(); im.thumbnail((px,px)); b=io.BytesIO(); im.convert('RGB').save(b,'JPEG',quality=q)
    return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode(), im.size
def sentences(t): return [x.strip() for x in re.split(r'(?<=[.!?])\s+',t.strip()) if x.strip()]
def key_sentence(sents,kind):
    if kind=='sft': return len(sents)-1
    for i,s in enumerate(sents):
        if KEY.search(s) and not re.search(r"\b(no|without|free of)\b[^.]{0,40}"+KEY.pattern,s,re.I): return i
    return len(sents)-1
def wrap_words(sents,maxw,ki=None,px=13):
    """lines of [(word,sent_idx)], greedy by measured width; the key sentence starts and ends on its own line"""
    lines=[]; cur=[]; 
    def flush():
        nonlocal cur
        if cur: lines.append(cur); cur=[]
    for si,s in enumerate(sents):
        if ki is not None and (si==ki or si==ki+1): flush()
        for w in s.split():
            if cur and tw(' '.join(x for x,_ in cur)+' '+w,px)*1.03>maxw: flush()
            cur.append((w,si))
    flush(); return lines
def trace_card(out,x,y,w,name,col,fill,tags,sents,ki,mode="tint",lh=18,px=13,title_h=48,hl_fill=None,fixed_h=None):
    pad=12; maxw=w-2*pad; lines=wrap_words(sents,maxw,ki,px); tl=wrap_words([tags],maxw); title_h=30+18*len(tl); h=fixed_h or (title_h+12+len(lines)*lh+12)
    out.append(rect(x,y,w,h,fill="#fff",stroke=col,sw=2,rx=12))
    out.append(rect(x,y,w,title_h,fill=fill,rx=12)); out.append(rect(x,y+title_h-12,w,12,fill=fill))
    out.append(txt(x+pad,y+22,name,"t16 b"))
    for j,ln in enumerate(tl): out.append(txt(x+pad,y+40+18*j,' '.join(wd for wd,_ in ln),"t13 mut"))
    ty=y+title_h+12
    for li,ln in enumerate(lines):
        s=' '.join(wd for wd,_ in ln); by=ty+li*lh+lh-4
        if ln[0][1]==ki:
            ww=tw(s,px)
            if mode=="tint": out.append(rect(x+pad-2,by-lh+5,ww+4,lh-1,fill=hl_fill,rx=2))
            elif mode=="underline": out.append(line(x+pad,by+3,x+pad+ww,by+3,col,2.5))
            elif mode=="bar": out.append(rect(x+4,by-lh+4,4,lh,fill=col))
        out.append(txt(x+pad,by,s,f"t{px}"))
    return h

def mask_components(m,maxside=256):
    """connected components (8-neighbour) of a binary mask, computed on a downsampled copy, coordinates mapped back to full size"""
    mh,mw=m.shape; f=max(1,int(np.ceil(max(mh,mw)/maxside)))
    small=m[::f,::f]; h,w=small.shape; lab=np.zeros((h,w),dtype=np.int32); comps=[]; cur=0
    sy,sx=np.nonzero(small)
    for y0,x0 in zip(sy,sx):
        if lab[y0,x0]: continue
        cur+=1; stack=[(y0,x0)]; lab[y0,x0]=cur; pts=[]
        while stack:
            yy,xx=stack.pop(); pts.append((yy,xx))
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny,nx=yy+dy,xx+dx
                    if 0<=ny<h and 0<=nx<w and small[ny,nx] and not lab[ny,nx]: lab[ny,nx]=cur; stack.append((ny,nx))
        ys=np.array([p[0] for p in pts])*f; xs=np.array([p[1] for p in pts])*f
        comps.append({'xs':xs,'ys':ys,'cx':xs.mean(),'cy':ys.mean(),'n':len(pts)})
    return comps
def image_panel(out,x,y,size,rec,gt_cells,kcr_cells):
    im=Image.open(rec['absolute_path']).convert('RGB'); W0,H0=im.size; side=min(W0,H0)
    im=im.crop(((W0-side)//2,(H0-side)//2,(W0-side)//2+side,(H0-side)//2+side)); uri,(w,h)=b64(im)
    out.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="{uri}"/>')
    # true cell shading + grid
    for c in gt_cells:
        cx,cy=CELLS[c]; out.append(rect(x+cx*size/3,y+cy*size/3,size/3,size/3,fill="#00b95c",extra='fill-opacity="0.28"'))
    for i in (1,2):
        out.append(line(x+i*size/3,y,x+i*size/3,y+size,"#ffffff",1.2,extra='stroke-opacity="0.8"')); out.append(line(x,y+i*size/3,x+size,y+i*size/3,"#ffffff",1.2,extra='stroke-opacity="0.8"'))
    # arrow to the mask centroid
    mp=mask_for(rec['absolute_path'])
    if mp:
        m=np.array(Image.open(mp).convert('L'))>0; mh,mw=m.shape; ys,xs=np.nonzero(m)
        if len(xs):
            # connected components on a downsampled mask, clustered by the Real-IAD mask-spread rule:
            # components whose centroids lie within 20 % of the image diagonal belong to one cluster (Single / Multi-Close),
            # farther apart they are Spread and each cluster gets its own circle
            comps=mask_components(m); diag=(mw*mw+mh*mh)**0.5
            clusters=[[c] for c in comps]
            merged=True
            while merged:
                merged=False
                for i in range(len(clusters)):
                    for j in range(i+1,len(clusters)):
                        if any(((a['cx']-b['cx'])**2+(a['cy']-b['cy'])**2)**0.5<=0.2*diag for a in clusters[i] for b in clusters[j]):
                            clusters[i]+=clusters[j]; del clusters[j]; merged=True; break
                    if merged: break
            clusters.sort(key=lambda cl:-sum(c['n'] for c in cl))
            for k,cl in enumerate(clusters):
                px=np.concatenate([c['xs'] for c in cl]); py=np.concatenate([c['ys'] for c in cl])
                cxm=(px.mean()-(W0-side)//2)/side*size+x; cym=(py.mean()-(H0-side)//2)/side*size+y
                rad=float(np.sqrt(((px-px.mean())**2+(py-py.mean())**2).max()))/side*size+6
                rad=max(14.0,min(rad,size*0.45))
                if k==0:
                    sx=x+size*0.18 if cxm>x+size/2 else x+size*0.82; sy=y+size*0.18 if cym>y+size/2 else y+size*0.82
                    dx,dy=cxm-sx,cym-sy; L=(dx*dx+dy*dy)**0.5
                    if L>rad+24: ex,ey=cxm-dx/L*(rad+6),cym-dy/L*(rad+6)
                    else: ex,ey=cxm-dx/L*10,cym-dy/L*10
                    out.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="#d9363e" stroke-width="4" marker-end="url(#arr)"/>')
                out.append(circ(cxm,cym,rad,"none","#d9363e",3))
    out.append(rect(x,y,size,size,fill="none",stroke=C["rule2"],sw=1.5,rx=6))
DEFS='<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="#d9363e"/></marker></defs>'
def variant_A(e,s,k,out_name,mode="tint"):
    """image left, two trace cards side by side"""
    W=980; IMG=286; x1=24+IMG+18; cwid=(W-x1-24-14)/2
    sents_s=sentences(s['pred_tags']['reasoning']); sents_k=sentences(k['pred_tags']['reasoning']); ks,kk=key_sentence(sents_s,'sft'),key_sentence(sents_k,'kcr')
    body=[]; ktags=f"<answer> yes, <type> {k['pred_tags'].get('type','')}, <location> {k['pred_tags'].get('location','')}"
    h1=trace_card([],x1,24,cwid,"SFT",C["sft"],C["sft_f"],"<answer> no",sents_s,ks,mode,hl_fill="#fbd5d7")
    h2=trace_card([],x1+cwid+14,24,cwid,"KCR",C["kcr"],C["kcr_f"],ktags,sents_k,kk,mode,hl_fill="#c9ecd5")
    hc=max(h1,h2)
    trace_card(body,x1,24,cwid,"SFT",C["sft"],C["sft_f"],"<answer> no",sents_s,ks,mode,hl_fill="#fbd5d7",fixed_h=hc)
    trace_card(body,x1+cwid+14,24,cwid,"KCR",C["kcr"],C["kcr_f"],ktags,sents_k,kk,mode,hl_fill="#c9ecd5",fixed_h=hc)
    H=24+max(hc,IMG+70)+24
    o=[head(W,H),DEFS]
    image_panel(o,24,24,IMG,s,e['gt_cells'],e['kcr_cells'])
    o.append(txt(24,24+IMG+22,f"{e['product']}, ground truth: defective","t13 b")); o.append(txt(24,24+IMG+40,"shaded cell: annotated location","t13 mut")); o.append(txt(24,24+IMG+58,"arrow: the defect itself","t13 mut"))
    o+=body; o.append(foot()); open(out_name,'w').write(''.join(o))
def variant_B(e,s,k,out_name):
    """image top-left, verdict strip beside it, both full traces below"""
    W=980; IMG=230; sents_s=sentences(s['pred_tags']['reasoning']); sents_k=sentences(k['pred_tags']['reasoning'])
    ks,kk=key_sentence(sents_s,'sft'),key_sentence(sents_k,'kcr'); cwid=(W-48-14)/2
    y2=24+IMG+70; body=[]; ktags=f"<answer> yes, <type> {k['pred_tags'].get('type','')}, <location> {k['pred_tags'].get('location','')}"
    h1=trace_card(body,24,y2,cwid,"SFT, full reasoning",C["sft"],C["sft_f"],"<answer> no",sents_s,ks,"bar")
    h2=trace_card(body,24+cwid+14,y2,cwid,"KCR, full reasoning",C["kcr"],C["kcr_f"],ktags,sents_k,kk,"bar")
    H=y2+max(h1,h2)+24
    o=[head(W,H),DEFS]
    image_panel(o,24,24,IMG,s,e['gt_cells'],e['kcr_cells']); o.append(txt(24,24+IMG+22,f"{e['product']}, ground truth: defective","t13 b")); o.append(txt(24,24+IMG+40,"shaded cell: annotated location","t13 mut")); o.append(txt(24,24+IMG+58,"arrow: the defect itself","t13 mut"))
    sx=24+IMG+22; sw_=W-sx-24
    for i,(nm,col,fill,verdict,quote) in enumerate((("SFT",C["sft"],C["sft_f"],"answers no",sents_s[ks]),("KCR",C["kcr"],C["kcr_f"],f"answers yes, {k['pred_tags'].get('type','')}, {k['pred_tags'].get('location','')}",sents_k[kk]))):
        yy=24+i*(IMG/2+4); hh=IMG/2-4
        o.append(rect(sx,yy,sw_,hh,fill=fill,stroke=col,sw=2,rx=12)); o.append(txt(sx+14,yy+24,nm,"t16 b")); o.append(txt(sx+60,yy+24,verdict,"t13 mut"))
        for j,ln in enumerate(wrap_words([quote],sw_-28)[:3]): o.append(txt(sx+14,yy+48+j*18,' '.join(w for w,_ in ln),"t13"))
    o+=body; o.append(foot()); open(out_name,'w').write(''.join(o))
if __name__=='__main__':
    idx=json.load(open(f'{W_}/defense_prep_20260901/sft_vs_kcr_selected/index.json'))
    ALL=os.environ.get('ALL_PAIRS')=='1'
    for e in idx:
        if not ALL and e['i'] not in (1,3): continue
        bench='visa' if e['bench']=='visa' else 'dsmvtec'; s,k=load_pair(e['image_id'],bench); tag=f"{e['i']:02d}"
        name={1:'pairs_cashew',3:'pairs_screw'}.get(e['i'],f'pairs_svg/pair_{tag}') if not ALL else f'pairs_svg/pair_{tag}'
        os.makedirs('pairs_svg',exist_ok=True)
        variant_A(e,s,k,f'{name}.svg','tint'); print('pair',tag,'->',name+'.svg')
    # thesis PDFs (cairosvg resolves Verdana to Noto Sans here, the width measurement above matches that)
    import subprocess
    out=os.environ.get('THESIS_FIGURES','thesis')
    names=[f"pairs_svg/pair_{e['i']:02d}" for e in idx] if ALL else ['pairs_cashew','pairs_screw']
    os.makedirs(f'{out}/pairs_svg',exist_ok=True)
    for n in names:
        subprocess.run([os.environ.get('CAIROSVG','cairosvg'),f'{n}.svg','-o',f'{out}/{n}.pdf'],check=True); print('installed',f'{out}/{n}.pdf')
