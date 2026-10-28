import os
"""Figure 4.1: SFT stage next to GRPO stage. House style taken from all_svg (method_overview_v4, anomalythink_generation, rl_correction_pipeline)."""
import re,html
ov=open('/bulk/aacudad/reasoning_traces/all_svg/method_overview_v4.svg').read()
syms=re.findall(r'<symbol\b.*?</symbol>',ov,re.S); marks=re.findall(r'<marker\b.*?</marker>',ov,re.S)
mk=lambda i,c: f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="{c}"/></marker>'
defs='<defs>'+''.join(syms)+''.join(marks)+mk('arrCoral','#7a8b99')+'</defs>'
W,H=1600,760; s=[]; A=s.append
G,Gs,Gf="#00B95C","#7BB661","#F3F9EE"; P,Ps,Pf="#7B57B5","#9B7FC4","#FBF8FD"; INK,MUT="#111","#555"; CO,COs,COf="#3b4a5a","#9aa5ad","#F5F7F9"
def R(x,y,w,h,fill="#fff",stroke="#CFCFCF",sw=1.5,rx=12,dash=None): A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{(" stroke-dasharray=%s%s%s"%(chr(34),dash,chr(34))) if dash else ""}/>')
def T(x,y,t,size=13,anchor="start",bold=False,fill=INK,italic=False,mono=False): A(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{fill}"{" font-weight=%sbold%s"%(chr(34),chr(34)) if bold else ""}{" font-style=%sitalic%s"%(chr(34),chr(34)) if italic else ""}{" font-family=%sConsolas, DejaVu Sans Mono, monospace%s"%(chr(34),chr(34)) if mono else ""}>{html.escape(t)}</text>')
def AR(pts,col,marker,sw=4,dash=None): A(f'<path d="M'+' L'.join(f'{x} {y}' for x,y in pts)+f'" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"{(" stroke-dasharray=%s%s%s"%(chr(34),dash,chr(34))) if dash else ""} marker-end="url(#{marker})"/>')
def badge(x,y,n,col): A(f'<circle cx="{x}" cy="{y}" r="12" fill="{col}"/>'); T(x,y+4,str(n),12,"middle",True,"#fff")
def pill(x,y,w,t,fill,tc="#fff",size=11): R(x,y,w,20,fill,fill,1,10); T(x+w/2,y+14,t,size,"middle",True,tc)
def comp(x,y,w,label,frozen,col):
    R(x,y,w,50,"#fff",col if not frozen else "#8fb0dc",1.5,10); T(x+14,y+22,label,14,bold=True)
    if frozen: R(x+w-30,y+8,20,20,"#E8F0FF","#5B7FC7",1,4); A(f'<use href="#snow" x="{x+w-26}" y="{y+12}" width="12" height="12"/>'); T(x+14,y+40,"frozen, no gradient",12,fill=MUT)
    else: pill(x+w-72,y+9,62,"trained",col); T(x+14,y+40,"receives gradient",12,fill=MUT)
A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Verdana, Geneva, Arial, sans-serif">{defs}<rect width="{W}" height="{H}" fill="#fff"/>')
# ═══ container 1: SFT ═══
ax,ay,aw,ah=40,30,740,660; R(ax,ay,aw,ah,Gf,Gs,3,40); T(ax+aw/2,ay+46,"1  SFT",24,"middle",True); T(ax+aw/2,ay+70,"Supervised fine-tuning on the gold traces",14,"middle",fill=MUT)
# data card
R(ax+30,ay+95,680,120); A(f'<use href="#db" x="{ax+42}" y="{ay+104}" width="56" height="48"/>'); T(ax+108,ay+124,"AnomalyThink-6K",15,bold=True); T(ax+108,ay+142,"6,000 instances, one C1 image and one gold trace each",12,fill=MUT)
R(ax+42,ay+160,300,44,"#F5F5F5","#DDDDDD",1,6); T(ax+52,ay+178,"user",11,bold=True,fill=MUT); T(ax+88,ay+178,"Analyze the provided image ...",11,mono=True); T(ax+52,ay+196,"image",11,bold=True,fill=MUT); T(ax+88,ay+196,"one test image, no reference, no mask",11,fill=MUT)
R(ax+350,ay+160,354,44,"#F3F9EE",Gs,1,6); T(ax+360,ay+178,"assistant, gold",11,bold=True,fill="#0b7a3b"); T(ax+470,ay+178,"normal trace: <think> <answer> only",10,mono=True,fill=MUT); T(ax+360,ay+196,"<think> six phases </think> <location> <type> <answer>",9.5,mono=True,fill="#0b7a3b")
AR([(ax+370,ay+215),(ax+370,ay+245)],Gs,"arrGreen")
# model card
R(ax+30,ay+250,340,230); A(f'<use href="#qwen" x="{ax+42}" y="{ay+258}" width="34" height="34"/>'); T(ax+86,ay+280,"Qwen2.5-VL-7B",15,bold=True)
comp(ax+42,ay+300,316,"vision encoder (ViT)",True,Gs); comp(ax+42,ay+356,316,"projector",False,G); comp(ax+42,ay+412,316,"language model",False,G)
# loss card
AR([(ax+370,ay+365),(ax+410,ay+365)],Gs,"arrGreen")
R(ax+420,ay+250,290,230,COf,COs,1.5,12); badge(ax+445,ay+275,"L",CO); T(ax+464,ay+280,"cross-entropy loss",15,bold=True,fill=CO)
T(ax+436,ay+306,"per assistant token, the negative",12); T(ax+436,ay+323,"log-likelihood of the gold token",12); T(ax+436,ay+340,"given its prefix (teacher forcing)",12); T(ax+436,ay+359,"user and image tokens are masked out",11.5,fill=MUT)
R(ax+436,ay+372,258,44,"#fff",COs,1,8); T(ax+565,ay+390,"gradient flows back to",12,"middle",fill=CO); T(ax+565,ay+408,"projector and language model only",12,"middle",True,CO)
AR([(ax+436,ay+394),(ax+372,ay+394)],"#7a8b99","arrCoral",2.5,"6 5")
# output pill
R(ax+215,ay+520,310,56,"#fff",Gs,2.5,12); A(f'<use href="#qwen" x="{ax+228}" y="{ay+530}" width="36" height="36"/>'); T(ax+276,ay+542,"output:",12,fill=MUT); T(ax+276,ay+564,"the 6K SFT model",16,bold=True)
AR([(ax+200,ay+480),(ax+200,ay+548),(ax+213,ay+548)],Gs,"arrGreen",3)
# ═══ container 2: GRPO ═══
bx,by,bw,bh=820,30,740,660; R(bx,by,bw,bh,Pf,Ps,3,40); T(bx+bw/2,by+46,"2  GRPO",24,"middle",True); T(bx+bw/2,by+70,"Reinforcement learning from the model's own rollouts",14,"middle",fill=MUT)
# init + data
R(bx+30,by+95,330,54); A(f'<use href="#qwen" x="{bx+42}" y="{by+104}" width="36" height="36"/>'); T(bx+90,by+116,"initialisation and reference policy",11,fill=MUT); T(bx+90,by+136,"the 6K SFT model",15,bold=True)
R(bx+380,by+95,330,54); A(f'<use href="#db" x="{bx+392}" y="{by+102}" width="46" height="40"/>'); T(bx+448,by+116,"GRPO split, 4,236 prompts",13,bold=True); T(bx+448,by+136,"one image, one question, 23 products",12,fill=MUT)
# step 1 rollouts
badge(bx+45,by+180,1,P); T(bx+64,by+185,"sample G = 4 rollouts from the current policy",15,bold=True)
# one real group: run-2 policy on Real-IAD button_battery S0050 (AK), first four of its eight rollouts in phase0_full_10k rollouts_raw.jsonl
ex=[("3.00","format 1  verdict 1  type 1.00  loc 1"),("0.00","format 0  verdict 0  type 0.00  loc 0"),("2.50","format 1  verdict 1  type 0.00  loc 1"),("2.00","format 1  verdict 1  type 0.00  loc 0")]
for i,(r,d) in enumerate(ex):
    x=bx+30+i*172; R(x,by+198,160,78,"#fff",Ps,1.5,10); A(f'<use href="#robot" x="{x+8}" y="{by+204}" width="42" height="28"/>'); T(x+58,by+216,f"rollout {i+1}",12,bold=True); T(x+58,by+236,f"R = {r}",14,bold=True,fill="#4a3a66")
    a,b_=d.split("  type "); T(x+10,by+254,a.replace("  "," · "),12,fill=MUT); T(x+10,by+270,"type "+b_.replace("  loc ",", location "),12,fill=MUT)
# step 2 reward
badge(bx+45,by+306,2,P); T(bx+64,by+311,"score every rollout against the gold tags",15,bold=True)
R(bx+30,by+324,680,54,"#fff",Ps,1.5,10); T(bx+370,by+346,"anomalous:  R  =  format (0/1)  +  verdict (0/1)  +  ½ · type (0 to 1)  +  ½ · location (0/1)",12,"middle",True,"#4a3a66"); T(bx+370,by+366,"normal: format + verdict, max 2.0  ·  two unweighted reward functions, four bounded sub-signals, max 3.0",12,"middle",fill=MUT)
# step 3 advantage
badge(bx+45,by+400,3,P); T(bx+64,by+405,"compare within the group",15,bold=True)
R(bx+30,by+418,330,54,"#fff",Ps,1.5,10); T(bx+195,by+441,"Â  =  ( R  -  mean )  /  std",15,"middle",True,"#4a3a66"); T(bx+195,by+461,"no value network, the group is the baseline",12,"middle",fill=MUT)
R(bx+380,by+418,330,54,"#fff",Ps,1.5,10,"6 4"); T(bx+545,by+441,"KL to the reference is logged, not in the loss",13,"middle",True,"#4a3a66"); T(bx+545,by+461,"β = 0, the reference stays the SFT model",12,"middle",fill=MUT)
# step 4 update
badge(bx+45,by+504,4,P); T(bx+64,by+509,"update the policy with the clipped surrogate, ε = 0.2, lr 1e-6, batch 8",15,bold=True)
for i,lab in enumerate(("vision encoder","projector","language model")): comp(bx+30+i*230,by+522,220,lab,False,P)
# output pill
R(bx+215,by+596,310,56,"#fff",Ps,2.5,12); A(f'<use href="#qwen" x="{bx+228}" y="{by+606}" width="36" height="36"/>'); T(bx+276,by+618,"output:",12,fill=MUT); T(bx+276,by+640,"the SFT + GRPO model",16,bold=True)
AR([(bx+195,by+572),(bx+195,by+624),(bx+213,by+624)],Ps,"arrPurple",3)
# bridge 1 -> 2
AR([(ax+525,ay+548),(ax+745,ay+548),(ax+745,ay+122),(bx+20,by+122)],Gs,"arrGreen",4)
# footnote outside the frames
T(40,H-40,"The two training stages of the thesis. Stage 1 fits the gold traces by cross-entropy with the vision encoder frozen. Stage 2 starts from that checkpoint, samples groups of four rollouts, scores them, and updates all three blocks.",13)
A('</svg>'); open('fig_stages.svg','w').write(''.join(s)); print("svg written")
