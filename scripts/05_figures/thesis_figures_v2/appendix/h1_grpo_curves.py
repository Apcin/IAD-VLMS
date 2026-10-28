"""Appendix H: reward / KL / completion-length curves of the SFT+GRPO run (run-2 trainer_state)."""
import json,os,numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W=os.environ.get('WORK_DIR','/bulk/aacudad/reasoning_traces'); OUT=os.environ.get('THESIS_FIGURES','/bulk/aacudad/ol_thesis/figures')
lh=[e for e in json.load(open(f'{W}/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-1060/trainer_state.json'))['log_history'] if 'reward' in e]
st=np.array([e['step'] for e in lh]); R=np.array([e['reward'] for e in lh]); A=np.array([e['rewards/accuracy_reward'] for e in lh]); C=np.array([e['rewards/consistency_reward'] for e in lh]); KL=np.array([e['kl'] for e in lh]); L=np.array([e['completion_length'] for e in lh])
ma=lambda x,k=25: np.convolve(x,np.ones(k)/k,mode='valid')
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9})
fig,ax=plt.subplots(figsize=(8,3.6))
ax.plot(st,R,color='0.75',lw=0.7,label='total reward, per step'); ax.plot(st[24:],ma(R),color='k',lw=1.6,label='total reward, 25-step mean')
ax.plot(st[24:],ma(A),color='tab:blue',lw=1.3,label='accuracy reward (max 2.0), 25-step mean'); ax.plot(st[24:],ma(C),color='tab:green',lw=1.3,label='consistency reward (max 1.0), 25-step mean')
ax.axhline(3.0,color='0.5',ls=':',lw=0.8); ax.axvline(530,color='0.5',ls='--',lw=0.8); ax.text(535,2.92,'ckpt-530',fontsize=8,color='0.3')
ax.set_xlabel('training step'); ax.set_ylabel('reward'); ax.set_ylim(0,3.1); ax.set_title('GRPO reward dynamics (7B, 6K-frozen init, SFT+GRPO run)'); ax.legend(loc='lower right',fontsize=7.5,frameon=False)
fig.tight_layout(); fig.savefig(f'{OUT}/curve_grpo_reward.png',dpi=170); print('wrote curve_grpo_reward.png')
fig,(a1,a2)=plt.subplots(1,2,figsize=(9,3.4))
a1.plot(st,KL,color='tab:red',alpha=0.3,lw=0.7); a1.plot(st[24:],ma(KL),color='tab:red',lw=1.6); a1.axvline(530,color='0.5',ls='--',lw=0.8); a1.set_xlabel('training step'); a1.set_ylabel('KL to the reference (k3)'); a1.set_title('KL divergence, raw and 25-step mean')
a2.plot(st,L,color='tab:purple',alpha=0.3,lw=0.7); a2.plot(st[24:],ma(L),color='tab:purple',lw=1.6); a2.axvline(530,color='0.5',ls='--',lw=0.8); a2.set_xlabel('training step'); a2.set_ylabel('completion length (tokens)'); a2.set_ylim(140,200); a2.set_title('Mean completion length')
fig.suptitle('GRPO KL and completion length (7B, 6K-frozen init, SFT+GRPO run)',fontsize=10); fig.tight_layout(); fig.savefig(f'{OUT}/curve_grpo_kl.png',dpi=170); print('wrote curve_grpo_kl.png')
print('MA start %.2f end %.2f | KL max %.3f | length min %.0f max %.0f'%(ma(R)[0],ma(R)[-1],KL.max(),L.min(),L.max()))
