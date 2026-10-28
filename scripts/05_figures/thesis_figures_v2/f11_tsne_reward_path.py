import os,json,numpy as np,torch
os.environ['HF_HUB_OFFLINE']='1'; os.environ.setdefault('HF_HOME','')
from transformers import AutoTokenizer, AutoModel
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
tok=AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v2-moe',trust_remote_code=True)
model=AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v2-moe',trust_remote_code=True,torch_dtype=torch.float32).cuda().eval()
def emb(w):
    inp=tok(f"search_query: {w}",return_tensors='pt',truncation=True,max_length=512).to('cuda')
    with torch.no_grad(): o=model(**inp)[0]
    m=inp['attention_mask'].unsqueeze(-1); e=(o*m).sum(1)/m.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(e,p=2,dim=1)[0].cpu().numpy()
# families: corpus strings (lower-cased, as the reward sees them) plus illustrative variants a model might emit
FAM={
 'scratch':['scratch','scratches','a scratch','surface scratch','scratched surface','fine scratch','scratch mark','light scratching'],
 'contamination':['contamination','contaminant','contaminated area','dirt','dust particle','residue','foreign residue','smudge'],
 'missing parts':['missing parts','missing part','missing component','absent component','part missing','incomplete assembly','missing piece','missing pin'],
 'pit':['pit','pits','pitting','pinhole','small pit','surface pit','pitted surface','tiny hole'],
 'damage':['damage','damaged area','surface damage','physical damage','broken','fracture','crack','cracked','torn edge'],
 'chip':['chip','chipping','chipped edge','chipped corner','chip on the edge','small chip'],
 'deformation':['deformation','deformed','bend','bent','warped','misalignment','misaligned pin','twisted'],
 'stain':['stain','stains','staining','discoloration','discoloured patch','dark spot','blotch','ink mark'],
 'abrasion':['abrasion','abraded surface','scuff','scuff mark','wear mark','rubbed area','worn surface'],
 'foreign objects':['foreign objects','foreign object','foreign material','extra component','unexpected object','debris','stray fibre'],
 'flash':['flash','excess material','burr','overmolding','excess plastic','flashing','moulding flash','poor finish'],
}
pairs=json.load(open('./results/anomaly_type_analysis/final_corpus/types_final_anom.json'))
cnt=Counter(s.strip().lower() for _,s in pairs)
names=[];fam=[]
for f,vs in FAM.items():
    for v in vs: names.append(v); fam.append(f)
E=np.stack([emb(n) for n in names]); uf=list(FAM); pal=plt.cm.tab20(np.linspace(0,1,len(uf))); col={f:pal[i] for i,f in enumerate(uf)}
xy=TSNE(n_components=2,random_state=0,perplexity=12,init='pca',max_iter=3000,learning_rate='auto').fit_transform(E)
xy=(xy-xy.min(0))/(xy.max(0)-xy.min(0))*100
from adjustText import adjust_text
fig,ax=plt.subplots(figsize=(12,9.5))
texts=[]
for (x,y),n,f in zip(xy,names,fam):
    inc=cnt.get(n,0)>0
    ax.scatter(x,y,s=50+min(cnt.get(n,0),1600)*0.22,color=col[f],edgecolor='k',lw=.5,alpha=.9 if inc else .55,marker='o' if inc else 'D',zorder=3)
    texts.append(ax.text(x,y,n+(f' ({cnt[n]})' if inc else ''),fontsize=9,ha='center',va='center',zorder=4,
                 bbox=dict(boxstyle='round,pad=0.12',fc='white',ec='none',alpha=.8)))
adjust_text(texts,x=xy[:,0],y=xy[:,1],ax=ax,expand=(1.35,1.8),force_text=(0.8,1.4),force_points=(0.6,1.0),arrowprops=dict(arrowstyle='-',color='#888',lw=.6),min_arrow_len=3,time_lim=60)
h=[plt.Line2D([],[],marker='o',ls='',color=col[f],markeredgecolor='k',markersize=9,label=f) for f in uf]
h+= [plt.Line2D([],[],marker='o',ls='',color='w',markeredgecolor='k',markersize=9,label='circle: string in the corpus (count)'),plt.Line2D([],[],marker='D',ls='',color='w',markeredgecolor='k',markersize=8,label='diamond: added variant, illustrative')]
ax.legend(handles=h,fontsize=9,loc='lower left',bbox_to_anchor=(0.0,-0.14),ncol=5,frameon=False,handletextpad=0.4,columnspacing=1.2)
ax.set_title('')
ax.set_xticks([]); ax.set_yticks([]); ax.margins(0.06); fig.tight_layout(); fig.savefig(os.path.dirname(os.path.abspath(__file__))+'/tsne_full.png',dpi=200,bbox_inches='tight')
# report cross-family nearest neighbours in embedding space
C=E@E.T; np.fill_diagonal(C,-1)
bad=[(names[i],names[j],round(C[i,j],2)) for i in range(len(names)) for j in range(i+1,len(names)) if fam[i]!=fam[j] and C[i,j]>=0.55]
print('n strings',len(names),'| cross-family pairs with cosine >= 0.55 (would get >= 0.5 credit):',bad)
