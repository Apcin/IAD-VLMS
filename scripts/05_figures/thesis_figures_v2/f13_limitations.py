"""Figure 7.1 in the house style: the KCR detector failing in both directions on the same three DS-MVTec products.
Top row false positives on good parts, bottom row false negatives with the ground-truth mask cells. Same card as Appendix F.2."""
import sys,os,subprocess
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import f9_appendix_examples as f9
from f9_appendix_examples import load,gt_cells,card,head,foot,DEFS,C,txt
OUT=os.environ.get('THESIS_FIGURES','thesis'); CAIRO=os.environ.get('CAIROSVG','cairosvg')
kcr=load('kcr')
FP=['DS-MVTec/pill/image/good/023.png','DS-MVTec/transistor/image/good/000.png','DS-MVTec/cable/image/good/041.png']
FN=['DS-MVTec/pill/image/color/000.png','DS-MVTec/transistor/image/bent_lead/008.png','DS-MVTec/cable/image/bent_wire/000.png']
for iid in FP: assert kcr[iid]['gt_answer']=='no' and kcr[iid]['pred_answer']=='yes', iid
for iid in FN: assert kcr[iid]['gt_answer']=='yes' and kcr[iid]['pred_answer']=='no', iid
W=980; MX=24; GAP=14; cols=3; cw=(W-2*MX-(cols-1)*GAP)/cols; body=[]; y=24
for label,ids,cells_on in (('false alarm',FP,False),('missed defect',FN,True)):
    body.append(txt(MX,y+14,('Top row: false positives on defect-free parts' if not cells_on else 'Bottom row: false negatives on the same three products, ground-truth mask cells shaded'),"t15 b")); y+=26
    cards=[(kcr[i],'bad',gt_cells(kcr[i]) if cells_on else [],label) for i in ids]
    hs=[card([],MX+j*(cw+GAP),y,cw,*c,ql=5) for j,c in enumerate(cards)]; hr=max(hs)
    for j,c in enumerate(cards): card(body,MX+j*(cw+GAP),y,cw,*c,ql=5,fixed_h=hr)
    y+=hr+GAP+6
H=y+4
open('fig_limitations.svg','w').write(head(W,H)+DEFS+''.join(body)+foot())
subprocess.run([CAIRO,'fig_limitations.svg','-o',f'{OUT}/fig_limitations.pdf'],check=True); print('wrote fig_limitations',H)
