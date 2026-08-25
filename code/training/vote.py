import json, numpy as np, pandas as pd
from collections import Counter
from scipy import stats
m=json.load(open('work/meta_for_analysis.json'))
tracks=np.array(m['test_tracks_per_segment']); y=np.array(m['y_test'])
utracks=list(dict.fromkeys(tracks.tolist()))              # first-appearance order
tidx={t:i for i,t in enumerate(utracks)}
groups=[np.where(tracks==t)[0] for t in utracks]
ytrack=np.array([y[g[0]] for g in groups])
assert all(len(set(y[g].tolist()))==1 for g in groups)
FFTS=[512,1024,2048]; HOPS=[128,256,512]; SEEDS=[42,123,456,789,1024]
rows=[]; tie_detail=[]
for nf in FFTS:
  for hp in HOPS:
    for sd in SEEDS:
        P=np.load(f'work/probs/probs_{nf}_{hp}_{sd}.npy')
        pred=P.argmax(1)
        seg_acc=(pred==y).mean()
        hard=[];ties=0;tie_correct=0;tie_resolved_wrong=0
        for g in groups:
            c=Counter(pred[g].tolist()); top=c.most_common()
            best=top[0][1]; tied=[k for k,v in c.items() if v==best]
            lab=top[0][0]                      # Counter.most_common ties -> first inserted
            if len(tied)>1:
                ties+=1
            hard.append(lab)
        hard=np.array(hard)
        soft=np.array([P[g].mean(0).argmax() for g in groups])          # mean softmax
        logsum=np.array([np.log(P[g]+1e-12).sum(0).argmax() for g in groups])
        maxconf=np.array([P[g][P[g].max(1).argmax()].argmax() for g in groups])
        med=np.array([np.median(P[g],axis=0).argmax() for g in groups])
        # ties that hard voting got wrong but soft got right
        for i,g in enumerate(groups):
            c=Counter(pred[g].tolist()); best=c.most_common()[0][1]
            tied=[k for k,v in c.items() if v==best]
            if len(tied)>1:
                tie_detail.append(dict(cfg=f"{nf}/{hp}",seed=sd,track=utracks[i].split('/')[-1],
                    true=int(ytrack[i]),hard=int(hard[i]),soft=int(soft[i]),
                    n_tied=len(tied),hard_ok=bool(hard[i]==ytrack[i]),soft_ok=bool(soft[i]==ytrack[i])))
        from sklearn.metrics import f1_score
        rows.append(dict(n_fft=nf,hop=hp,seed=sd,seg_acc=seg_acc,
            hard_acc=(hard==ytrack).mean(), soft_acc=(soft==ytrack).mean(),
            logsum_acc=(logsum==ytrack).mean(), maxconf_acc=(maxconf==ytrack).mean(),
            median_acc=(med==ytrack).mean(),
            hard_f1=f1_score(ytrack,hard,average='macro'), soft_f1=f1_score(ytrack,soft,average='macro'),
            n_ties=ties))
df=pd.DataFrame(rows); df.to_csv('figs/aggregation_per_seed.csv',index=False)
pd.DataFrame(tie_detail).to_csv('figs/tie_cases.csv',index=False)
g=df.groupby(['n_fft','hop']).agg(['mean','std'])
print("=== Konfigurasyon bazinda (5 tohum ortalamasi) ===")
print("cfg       seg     HARD    SOFT    d(pp)   logsum  maxconf median  ties/150")
for (nf,hp),s in df.groupby(['n_fft','hop']):
    print("%4d/%-4d %.4f  %.4f  %.4f  %+5.2f   %.4f  %.4f  %.4f  %.1f"%(
        nf,hp,s.seg_acc.mean(),s.hard_acc.mean(),s.soft_acc.mean(),
        (s.soft_acc.mean()-s.hard_acc.mean())*100,
        s.logsum_acc.mean(),s.maxconf_acc.mean(),s.median_acc.mean(),s.n_ties.mean()))
print()
print("GENEL  hard=%.4f  soft=%.4f  fark=%+.2f pp"%(df.hard_acc.mean(),df.soft_acc.mean(),
    (df.soft_acc.mean()-df.hard_acc.mean())*100))
print("Toplam beraberlik vakasi (45 kosu x 150 parca = 6750): %d  (%.2f%%)"%(df.n_ties.sum(),100*df.n_ties.sum()/6750))
print()
# Friedman on soft
cfgs=[(a,b) for a in FFTS for b in HOPS]
def mat(col): return np.array([[df[(df.n_fft==a)&(df.hop==b)&(df.seed==s)][col].values[0] for a,b in cfgs] for s in SEEDS])
for col,name in [('hard_acc','Track Acc (HARD)'),('soft_acc','Track Acc (SOFT)'),('soft_f1','Track F1 (SOFT)')]:
    M=mat(col); chi,p=stats.friedmanchisquare(*[M[:,j] for j in range(9)])
    print("%-20s chi2(8)=%7.3f  p=%.4f  W=%.3f"%(name,chi,p,chi/(5*8)))
print()
# --- Wilcoxon: mean-probability vs hard voting -------------------------------
# 45 kosu bagimsiz DEGIL: ayni bes tohum degeri dokuz konfigurasyonda tekrar
# ediliyor (capraz tasarim) ve hepsi ayni test bolmesinde degerlendiriliyor.
# Bu yuzden kosu duzeyindeki test SADECE BETIMLEYICI olarak raporlanir;
# makalede raporlanan test konfigurasyon duzeyindeki eslesmis testtir.
w=stats.wilcoxon(df.soft_acc,df.hard_acc)
print("[betimleyici] Wilcoxon soft vs hard (45 kosu): stat=%.1f p=%.4f"%(w.statistic,w.pvalue))
print("              soft>hard: %d, soft<hard: %d, esit: %d"%((df.soft_acc>df.hard_acc).sum(),(df.soft_acc<df.hard_acc).sum(),(df.soft_acc==df.hard_acc).sum()))
print()
# MAKALEDE RAPORLANAN TEST — dokuz konfigurasyon ortalamasi, iki yonlu tam test
scfg=np.array([df[(df.n_fft==a)&(df.hop==b)].soft_acc.mean() for a,b in cfgs])
hcfg=np.array([df[(df.n_fft==a)&(df.hop==b)].hard_acc.mean() for a,b in cfgs])
wc=stats.wilcoxon(scfg,hcfg,alternative="two-sided",method="exact")
print("[RAPORLANAN] Wilcoxon soft vs hard (9 konfigurasyon, tam test): W=%.1f p=%.4f"%(wc.statistic,wc.pvalue))
print("              soft>hard: %d/9   ortalama fark = %+.2f pp"%((scfg>hcfg).sum(),(scfg-hcfg).mean()*100))
