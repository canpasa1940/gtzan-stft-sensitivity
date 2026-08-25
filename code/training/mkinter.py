import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ab=json.load(open('ablation.json')); rec=json.load(open('all_results_combined.json'))
FFTS=[512,1024,2048]; SEEDS=[42,123,456,789,1024]
def v(nf,mels,key):
    r=[x for x in ab if x['fft']==nf] if mels==64 else [x for x in rec if x['fft']==nf and x['hop']==256]
    return np.array([[y for y in r if y['seed']==s][0][key] for s in SEEDS])
# two levels of an ordered factor -> sequential ramp, plus distinct markers/linestyles
COL={128:"#08519C",64:"#7EB6E8"}; EDGE={128:"#062F5C",64:"#3182BD"}
MRK={128:"s",64:"o"}; LS={128:"-",64:"--"}
INK="#1a1a1a"; MUTED="#6b6b6b"; GRID="#dcdcdc"
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":9,
 "axes.edgecolor":MUTED,"axes.labelcolor":INK,"text.color":INK,"xtick.color":MUTED,
 "ytick.color":MUTED,"axes.linewidth":0.8,"figure.facecolor":"white","axes.facecolor":"white"})
fig,axes=plt.subplots(1,2,figsize=(7.2,3.1))
for ax,key,title in zip(axes,["seg_test_acc","track_test_acc"],
                        ["(a) Segment level","(b) Track level (majority voting)"]):
    for mels in [128,64]:
        m=[v(f,mels,key).mean() for f in FFTS]; sd=[v(f,mels,key).std(ddof=1) for f in FFTS]
        ax.errorbar(range(3),m,yerr=sd,color=COL[mels],marker=MRK[mels],ls=LS[mels],lw=1.8,ms=6.5,
            capsize=3,elinewidth=0.9,zorder=3,label=f"n_mels = {mels}",
            markeredgecolor="white",markeredgewidth=0.8)
        for x,(yy,ss) in enumerate(zip(m,sd)):
            for i,val in enumerate(v(FFTS[x],mels,key)):
                ax.scatter(x+(-0.055 if mels==128 else 0.055),val,s=7,color=EDGE[mels],alpha=.55,zorder=2,linewidths=0)
    ax.set_xticks(range(3)); ax.set_xticklabels(FFTS); ax.set_xlabel("n_fft")
    ax.set_title(title,fontsize=9,loc="left",pad=6)
    ax.grid(True,axis="y",color=GRID,lw=0.6,zorder=0); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("Accuracy (mean ± SD, 5 seeds)")
axes[0].legend(frameon=False,fontsize=8,loc="lower left")
axes[0].annotate("crossover", xy=(1.0,0.7877), xytext=(1.35,0.7735), fontsize=8, color=INK,
    arrowprops=dict(arrowstyle="->",color=MUTED,lw=0.9))
fig.tight_layout()
fig.savefig("figs/fig_nfft_nmels_interaction.png",dpi=400,bbox_inches="tight")
fig.savefig("figs/fig_nfft_nmels_interaction.pdf",bbox_inches="tight")
print("OK")
