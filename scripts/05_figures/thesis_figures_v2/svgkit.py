"""House style for the thesis figures, matched to all_svg/rl_correction_pipeline.svg."""
FONT = "Verdana, 'DejaVu Sans', Arial, sans-serif"
# Canvas is 980px wide so that at \textwidth (448pt) the scale is 0.457 and the
# smallest label still prints at 6.9pt. Nothing below 15px.
STYLE = """
  .num{font-size:22px;font-weight:bold;fill:#fff;text-anchor:middle}
  .h{font-size:32px;font-weight:bold;fill:#000;text-anchor:middle}
  .hl{font-size:26px;font-weight:bold;fill:#000}
  .sub{font-size:19px;fill:#000;text-anchor:middle}
  .b{font-weight:bold}
  .t13{font-size:15px;fill:#000}
  .t15{font-size:17px;fill:#000}
  .t16{font-size:19px;fill:#000}
  .t18{font-size:21px;fill:#000}
  .c{text-anchor:middle}
  .e{text-anchor:end}
  .mut{fill:#3b4750}
  .ic{fill:none;stroke:#000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}
"""
# palette lifted from the existing figures
C = dict(
  base   = "#9aa5ad", base_f = "#e3e7ea",
  sft    = "#d4a017", sft_f  = "#fde3c0",
  grpo   = "#1f8ef1", grpo_f = "#cfe3f7",
  kcr    = "#3a7d3a", kcr_f  = "#d4e9d4",
  purple = "#9b3fbf", purple_f="#e2d9ec",
  good   = "#00b95c", bad = "#d9363e", warn = "#f2c200",
  rule   = "#c8d0d6", rule2 = "#8d979e", ink = "#000", mut = "#5b6770",
)
STAGE_C = {"Base":("base","base_f"),"SFT":("sft","sft_f"),
           "SFT+GRPO":("grpo","grpo_f"),"KCR":("kcr","kcr_f")}

def head(w,h,title=None):
    t = f"<title>{esc(title)}</title>" if title else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="{FONT}">{t}<style>{STYLE}</style>'
            f'<rect width="{w}" height="{h}" fill="#fff"/>')
def foot(): return "</svg>"
def esc(s):
    return (str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
def txt(x,y,s,cls="t16",extra="",fill=None):
    # NOTE: a CSS rule like .t13{fill:#000} overrides a fill="..." presentation attribute,
    # so colour has to go through an inline style to take effect.
    st=f' style="fill:{fill}"' if fill else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}"{st}{(" "+extra) if extra else ""}>{esc(s)}</text>'
def rect(x,y,w,h,fill="none",stroke=None,sw=2,rx=0,extra=""):
    st=f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    r=f' rx="{rx}"' if rx else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}"{st}{r}{(" "+extra) if extra else ""}/>'
def line(x1,y1,x2,y2,stroke="#c8d0d6",sw=1,dash=None,extra=""):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}{(" "+extra) if extra else ""}/>'
def circ(cx,cy,r,fill,stroke=None,sw=2):
    st=f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"{st}/>'
def panel(x,y,w,h,stroke,rx=28,fill="#fff",sw=3):
    return rect(x,y,w,h,fill=fill,stroke=stroke,sw=sw,rx=rx)
def numbadge(cx,cy,n,fill="#1f8ef1",r=20):
    return circ(cx,cy,r,fill)+f'<text x="{cx:.1f}" y="{cy+8:.1f}" class="num">{n}</text>'
def legend_swatch(x,y,fill,stroke,label,w=22,h=13,cls="t15"):
    return (rect(x,y-h+2,w,h,fill=fill,stroke=stroke,sw=1.5,rx=2)
            + txt(x+w+8,y,label,cls))

# ---------------------------------------------------------------------------
# Chart primitives, so the regenerated plots share the house style exactly.
# ---------------------------------------------------------------------------
class Axes:
    """Linear axes with the house grid, ticks and labels."""
    def __init__(self,x,y,w,h,xlo,xhi,ylo,yhi):
        self.x,self.y,self.w,self.h=x,y,w,h
        self.xlo,self.xhi,self.ylo,self.yhi=xlo,xhi,ylo,yhi
    def X(self,v): return self.x+(v-self.xlo)/(self.xhi-self.xlo)*self.w
    def Y(self,v): return self.y+self.h-(v-self.ylo)/(self.yhi-self.ylo)*self.h
    def frame(self,yticks,xticks=None,xfmt=str,yfmt=str,ylabel=None,xlabel=None,xgrid=False):
        o=[]
        for v in yticks:
            o.append(line(self.x,self.Y(v),self.x+self.w,self.Y(v),C["rule"],1))
            o.append(txt(self.x-10,self.Y(v)+5,yfmt(v),"t13 mut e"))
        if xticks:
            for v in xticks:
                if xgrid: o.append(line(self.X(v),self.y,self.X(v),self.y+self.h,C["rule"],1))
                o.append(txt(self.X(v),self.y+self.h+22,xfmt(v),"t13 c"))
        if ylabel: o.append(txt(self.x-10,self.y-14,ylabel,"t13 mut"))
        if xlabel: o.append(txt(self.x+self.w/2,self.y+self.h+46,xlabel,"t15 c"))
        return "".join(o)
    def poly(self,pts,col,sw=3,dash=None,op=1.0):
        d=f' stroke-dasharray="{dash}"' if dash else ""
        p=" ".join(f"{self.X(a):.1f},{self.Y(b):.1f}" for a,b in pts)
        return (f'<polyline points="{p}" fill="none" stroke="{col}" stroke-width="{sw}"'
                f'{d} stroke-linejoin="round" stroke-linecap="round" opacity="{op}"/>')
    def marks(self,pts,col,r=5):
        return "".join(circ(self.X(a),self.Y(b),r,col,stroke="#fff",sw=2) for a,b in pts)
    def hline(self,v,col,label=None,dash="7 5"):
        o=[line(self.x,self.Y(v),self.x+self.w,self.Y(v),col,1.6,dash=dash)]
        if label: o.append(txt(self.x+self.w+8,self.Y(v)+5,label,"t13",fill=col))
        return "".join(o)
    def bars(self,groups,series,bw=None,gap=10,labels=None):
        """groups: list of x-category names. series: list of (name, colour, fill, [values])."""
        n=len(groups); gw=self.w/n; ns=len(series)
        bw=bw or min(46,(gw-2*gap-(ns-1)*6)/ns)
        o=[]
        for gi in range(n):
            gx=self.x+gi*gw
            x0=gx+(gw-(ns*bw+(ns-1)*6))/2
            for si,(nm,col,fil,vals) in enumerate(series):
                v=vals[gi]
                if v is None: continue
                bx=x0+si*(bw+6)
                o.append(rect(bx,self.Y(v),bw,self.Y(self.ylo)-self.Y(v),fill=fil,stroke=col,sw=1.5,rx=2))
                o.append(txt(bx+bw/2,self.Y(v)-8,f"{v:.1f}","t13 b c"))
            o.append(txt(gx+gw/2,self.y+self.h+24,groups[gi],"t15 b c"))
            if labels: o.append(txt(gx+gw/2,self.y+self.h+44,labels[gi],"t13 mut c"))
        return "".join(o)
def legend_row(x,y,items,gapx=200,swatch="line",dashes=None):
    o=[]
    for i,it in enumerate(items):
        lab,col=it[0],it[1]
        dash=(dashes or {}).get(lab)
        lx=x+i*gapx
        if swatch=="line":
            o.append(line(lx,y-5,lx+26,y-5,col,3,dash=dash)); o.append(circ(lx+13,y-5,5.5,col))
        else:
            o.append(rect(lx,y-13,22,15,fill=col,rx=2))
        o.append(txt(lx+34,y,lab,"t15"))
    return "".join(o)
