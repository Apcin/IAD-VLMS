"""Appendix H, Figure H.1: SFT balanced accuracy by epoch, 7B frozen, 6K against 15K. Values from the per-epoch eval files (g1_sft.py EP table)."""
import os,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT=os.environ.get('THESIS_FIGURES','thesis')
E=[1,2,3,4]; DS6=[79.24,76.66,80.16,77.07]; DS15=[71.00,65.40,71.66,72.60]; VA15=[65.45,64.32,64.28,66.94]; VA6=(3,64.78)
fig,(a,b)=plt.subplots(1,2,figsize=(10,4),dpi=150)
a.plot(E,DS6,'o-',color='tab:blue',label='6K SFT'); a.plot(E,DS15,'s--',color='tab:red',label='15K SFT'); a.set_title('DS-MVTec (development benchmark)'); a.set_xlabel('Epoch'); a.set_ylabel('Balanced accuracy (%)'); a.set_xticks(E); a.grid(alpha=.3); a.legend(loc='lower left')
b.plot(E,VA15,'s--',color='tab:red',label='15K SFT'); b.scatter([VA6[0]],[VA6[1]],s=120,color='tab:blue',zorder=3,label='6K SFT (epoch 3 only)'); b.set_title('VisA (held-out benchmark)'); b.set_xlabel('Epoch'); b.set_xticks(E); b.grid(alpha=.3); b.legend(loc='upper right'); b.set_ylim(63,68.5)
fig.suptitle('SFT balanced accuracy vs. epoch (full schema, train prompt)'); fig.tight_layout()
fig.savefig(f'{OUT}/curve_sft_ba.png',bbox_inches='tight'); print('written',f'{OUT}/curve_sft_ba.png')
