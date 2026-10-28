import os
"""Strip in-canvas title/subtitle/footer prose from the house-style SVGs and crop the viewBox.
Writes thesis/<name>.svg + .pdf + .png, installs the PDF into ol_thesis/figures, and lists what was stripped."""
import re,glob,os,subprocess
os.makedirs('thesis',exist_ok=True); CAIRO=os.environ.get('CAIROSVG','cairosvg'); OUT=os.environ.get('THESIS_FIGURES','thesis')
TEXT=re.compile(r'<text\b[^>]*>.*?</text>',re.S)
def attr(el,k):
    m=re.search(rf'\b{k}="([^"]*)"',el); return m.group(1) if m else None
report=[]
import sys
ONLY=set(sys.argv[1:])
for f in sorted(glob.glob('fig_*.svg')):
    if f=='fig_flow.svg': continue
    if ONLY and f[:-4] not in ONLY: continue
    s=open(f).read(); W,H=map(int,re.search(r'viewBox="0 0 (\d+) (\d+)"',s).groups())
    stripped=[]
    def keep(el):
        y=float(attr(el,'y') or 0); cls=attr(el,'class') or ''; body=re.sub(r'<[^>]+>','',el).strip()
        if cls=='h' and y<60: stripped.append(('title',body)); return False
        if cls=='sub mut' and y<90: stripped.append(('subtitle',body)); return False
        if y>H-110 and cls=='t13 mut': stripped.append(('footer',body)); return False
        if y>H-110 and cls=='t15' and len(body)>60: stripped.append(('footer',body)); return False
        return True
    s2=TEXT.sub(lambda m: m.group(0) if keep(m.group(0)) else '',s)
    # remaining geometry extent
    ys=[float(attr(el,'y') or 0)+4 for el in TEXT.findall(s2)]
    for el in re.findall(r'<rect\b[^>]*>',s2):
        y,h=attr(el,'y'),attr(el,'height')
        if y and h and float(h)<H-1: ys.append(float(y)+float(h))
    for el in re.findall(r'<(?:line|path|polyline|circle)\b[^>]*>',s2):
        for k in ('y1','y2','cy'):
            v=attr(el,k)
            if v: ys.append(float(v)+6)
    top=88; bot=int(max(ys))+14; newH=bot-top
    s2=re.sub(r'viewBox="0 0 \d+ \d+" width="\d+" height="\d+"',f'viewBox="0 {top} {W} {newH}" width="{W}" height="{newH}"',s2,1)
    name=f[:-4]; open(f'thesis/{f}','w').write(s2)
    subprocess.run([CAIRO,f'thesis/{f}','-o',f'thesis/{name}.pdf'],check=True)
    subprocess.run([CAIRO,f'thesis/{f}','-o',f'thesis/{name}.png','-s','2'],check=True)
    subprocess.run(['cp',f'thesis/{name}.pdf',f'{OUT}/{name}.pdf'],check=True)
    report.append(f"\n### {name}  ({H} -> {newH} px)"); report+= [f"  [{k}] {t}" for k,t in stripped]
    print(f"  {name}: {H}->{newH}px, stripped {len(stripped)}")
open('thesis/STRIPPED.txt','w').write('\n'.join(report)); print(open('thesis/STRIPPED.txt').read())
