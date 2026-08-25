import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

d=json.load(open('all_results_combined.json'))
FFTS=[512,1024,2048]; HOPS=[128,256,512]; SEEDS=[42,123,456,789,1024]
cfgs=[(f,h) for f in FFTS for h in HOPS]; names=[f"{a}/{b}" for a,b in cfgs]
def vals(f,h,key): return np.array([[r for r in d if r['fft']==f and r['hop']==h and r['seed']==s][0][key] for s in SEEDS])

# sequential ramp (validated: lightness band PASS, CVD separation PASS) + secondary encoding
COL={512:"#7EB6E8",1024:"#3182BD",2048:"#08519C"}
MRK={512:"o",1024:"s",2048:"^"}; LS={512:":",1024:"--",2048:"-"}
INK="#1a1a1a"; MUTED="#6b6b6b"; GRID="#dcdcdc"
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":9,
    "axes.edgecolor":MUTED,"axes.labelcolor":INK,"text.color":INK,
    "xtick.color":MUTED,"ytick.color":MUTED,"axes.linewidth":0.8,
    "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})

def style(ax):
    ax.grid(True,axis="y",color=GRID,lw=0.6,zorder=0); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)

# ---- FIG 1: accuracy vs hop, one line per n_fft, seg & track panels
fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),sharex=True)
for ax,key,title in zip(axes,["seg_test_acc","track_test_acc"],["(a) Segment level","(b) Track level"]):
    for f in FFTS:
        m=[vals(f,h,key).mean() for h in HOPS]; sd=[vals(f,h,key).std(ddof=1) for h in HOPS]
        ax.errorbar(range(3),m,yerr=sd,color=COL[f],marker=MRK[f],ls=LS[f],lw=1.6,ms=6,
                    capsize=3,elinewidth=0.9,zorder=3,label=f"n_fft = {f}",
                    markeredgecolor="white",markeredgewidth=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels(HOPS); ax.set_xlabel("hop_length")
    ax.set_title(title,fontsize=9,loc="left",pad=6); style(ax)
axes[0].set_ylabel("Accuracy (mean ± SD, 5 seeds)")
axes[0].legend(frameon=False,fontsize=8,loc="lower left")
fig.tight_layout()
fig.savefig("figs/fig_accuracy_vs_params.png",dpi=400,bbox_inches="tight")
fig.savefig("figs/fig_accuracy_vs_params.pdf",bbox_inches="tight"); plt.close(fig)

# ---- FIG 2: box plots across 5 seeds
fig,axes=plt.subplots(2,1,figsize=(7.2,5.4),sharex=True)
for ax,key,title in zip(axes,["seg_test_acc","track_test_acc"],["(a) Segment level","(b) Track level"]):
    data=[vals(f,h,key) for f,h in cfgs]
    bp=ax.boxplot(data,patch_artist=True,widths=0.55,
        medianprops=dict(color=INK,lw=1.4),whiskerprops=dict(color=MUTED,lw=0.9),
        capprops=dict(color=MUTED,lw=0.9),flierprops=dict(marker="",ls="none"))
    for patch,(f,h) in zip(bp["boxes"],cfgs):
        patch.set_facecolor(COL[f]); patch.set_alpha(0.55); patch.set_edgecolor(MUTED); patch.set_linewidth(0.9)
    for i,(f,h) in enumerate(cfgs):
        y=vals(f,h,key); ax.scatter(np.full_like(y,i+1)+np.linspace(-.13,.13,len(y)),y,
            s=13,color=INK,alpha=.75,zorder=4,linewidths=0)
    ax.set_ylabel("Accuracy"); ax.set_title(title,fontsize=9,loc="left",pad=6); style(ax)
axes[1].set_xticks(range(1,10)); axes[1].set_xticklabels(names,rotation=0,fontsize=8)
axes[1].set_xlabel("Configuration (n_fft / hop_length)")
hs=[plt.Line2D([],[],marker="s",ls="",ms=7,color=COL[f],alpha=.7,label=f"n_fft = {f}") for f in FFTS]
axes[0].legend(handles=hs,frameon=False,fontsize=8,loc="lower left",ncol=3)
fig.tight_layout()
fig.savefig("figs/fig_seed_boxplots.png",dpi=400,bbox_inches="tight")
fig.savefig("figs/fig_seed_boxplots.pdf",bbox_inches="tight"); plt.close(fig)

# ---- FIG 3: critical-difference diagram (Nemenyi, segment accuracy)
M=np.array([[vals(f,h,"seg_test_acc")[i] for f,h in cfgs] for i in range(5)])
ranks=np.array([stats.rankdata(-row) for row in M]); mr=ranks.mean(axis=0)
k,N=9,5; CD=3.102*np.sqrt(k*(k+1)/(6*N))
order=np.argsort(mr)
fig,ax=plt.subplots(figsize=(7.2,2.6))
ax.set_xlim(0.5,9.5); ax.set_ylim(0,1); ax.axis("off")
ax.hlines(0.78,1,9,color=MUTED,lw=1.0)
for x in range(1,10):
    ax.vlines(x,0.75,0.81,color=MUTED,lw=0.9); ax.text(x,0.86,str(x),ha="center",va="bottom",fontsize=8,color=MUTED)
ax.text(5,0.96,"Mean rank (1 = best)",ha="center",fontsize=8.5,color=INK)
left=order[:5]; right=order[5:][::-1]
for i,idx in enumerate(left):
    y=0.66-i*0.105; f=cfgs[idx][0]
    ax.plot([mr[idx],mr[idx],0.95],[0.75,y,y],color=COL[f],lw=1.3)
    ax.text(0.9,y,f"{names[idx]}  ({mr[idx]:.2f})",ha="right",va="center",fontsize=8,color=INK)
for i,idx in enumerate(right):
    y=0.66-i*0.105; f=cfgs[idx][0]
    ax.plot([mr[idx],mr[idx],9.05],[0.75,y,y],color=COL[f],lw=1.3)
    ax.text(9.1,y,f"({mr[idx]:.2f})  {names[idx]}",ha="left",va="center",fontsize=8,color=INK)
ax.plot([1,1+CD],[0.13,0.13],color=INK,lw=2.2)
ax.vlines([1,1+CD],0.10,0.16,color=INK,lw=1.6)
ax.text(1+CD/2,0.02,f"CD = {CD:.2f}  ($\\alpha$=0.05, k=9, N=5)",ha="center",fontsize=8,color=INK)
fig.tight_layout()
fig.savefig("figs/fig_critical_difference.png",dpi=400,bbox_inches="tight")
fig.savefig("figs/fig_critical_difference.pdf",bbox_inches="tight"); plt.close(fig)
print("OK")
