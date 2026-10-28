import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from svgkit import *
W=980; H=560
s=[txt(W/2,42,"The post-GRPO refinement pipeline","h"),
   txt(W/2,72,"How the cleaned 6,000-item pool of Section 6.7 was assembled.","sub mut")]
def node(x,y,w,h,fill,stroke,title,sub=None,sub2=None):
    o=[rect(x,y,w,h,fill=fill,stroke=stroke,sw=2,rx=12), txt(x+w/2,y+(28 if sub else h/2+6),title,"t16 b c")]
    if sub:  o.append(txt(x+w/2,y+50,sub,"t13 c"))
    if sub2: o.append(txt(x+w/2,y+70,sub2,"t13 c"))
    return "".join(o)
def arrow(x1,y1,x2,y2,label=None,col=None,lw=90):
    col=col or C["rule2"]
    o=[f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{col}" stroke-width="2.2" marker-end="url(#ah)"/>']
    if label:
        cx=(x1+x2)/2
        o.append(rect(cx-lw/2,min(y1,y2)-26,lw,18,fill="#fff",rx=3))
        o.append(txt(cx,min(y1,y2)-13,label,"t13 mut c"))
    return "".join(o)
s.append(f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{C["rule2"]}"/></marker></defs>')
# rubric strip
RB=[("0","Hallucinated",C["bad"]),("1","Poor",C["bad"]),("2","Acceptable",C["kcr"]),
    ("3","Good",C["kcr"]),("4","Excellent",C["kcr"])]
s.append(txt(40,118,"Gemini-3-Flash faithfulness rubric","t16 b"))
for i,(n,lab,col) in enumerate(RB):
    bx=40+i*182
    fill=C["kcr_f"] if i>=2 else "#fbe4e2"
    s.append(rect(bx,132,168,52,fill=fill,rx=10))
    s.append(txt(bx+84,154,n,"t16 b c",fill=C["kcr"] if i>=2 else C["bad"]))
    s.append(txt(bx+84,174,lab,"t13 c"))
s.append(line(40,196,388,196,C["bad"],2.5)); s.append(txt(214,214,"scores 0 or 1, rewrite or correct","t13 c",fill=C["bad"]))
s.append(line(404,196,950,196,C["kcr"],2.5)); s.append(txt(677,214,"scores 2 or above, keep","t13 c",fill=C["kcr"]))
# flow row
Y=252
s.append(node(40,Y,190,86,C["grpo_f"],C["grpo"],"GRPO ckpt-530","the production","checkpoint"))
s.append(arrow(232,Y+43,270,Y+43))
s.append(node(272,Y,190,86,"#fff",C["rule2"],"Roll out k = 8","10,236 items","81,888 rollouts"))
s.append(arrow(464,Y+43,544,Y+43,"passes",lw=62))
s.append(node(546,Y,180,86,C["kcr_f"],C["kcr"],"3,557 kept","correct and","well grounded"))
s.append(arrow(367,Y+90,367,Y+140,None))
s.append(node(272,Y+142,190,86,C["purple_f"],C["purple"],"Gemini judge","scores faithfulness","0 to 4"))
s.append(arrow(464,Y+185,544,Y+185,"fails or weak",lw=88))
s.append(node(546,Y+142,180,86,"#fbe4e2",C["bad"],"2,443 patched","corrected or","rewritten"))
s.append(arrow(728,Y+43,790,Y+43)); s.append(arrow(728,Y+185,790,Y+185))
s.append(node(792,Y+58,148,114,C["base_f"],C["rule2"],"6,000","3,000 anomalous","3,000 normal"))
s.append(txt(40,H-58,"The patched pool it is drawn from is 94.8 % anomalous, which is the gradient asymmetry of Section 6.7.","t13 mut"))
s.append(txt(40,H-36,"This is the earlier variant of the routing. Section 6.6 applies the same three-way routing to the same rollouts.","t13 mut"))
s.append(txt(40,H-14,"All counts are items, not rollouts.","t13 mut"))
open('fig_flow.svg','w').write(head(W,H,"The post-GRPO refinement pipeline")+"".join(s)+foot())
print("  fig_flow.svg")
