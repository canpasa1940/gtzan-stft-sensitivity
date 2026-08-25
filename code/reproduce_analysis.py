#!/usr/bin/env python3
"""
reproduce_analysis.py — revizyonda raporlanan TUM turetilmis tablolari
birincil kaynaklardan yeniden uretir ve mevcut CSV'lerle karsilastirir.

Birincil kaynaklar (hicbiri bu betik tarafindan uretilmedi):
    all_results_combined.json   45 ana kosunun Colab ciktisi (result.json'larin birlesimi)
    ablation.json               15 ablasyon kosusunun Colab ciktisi
    meta_for_analysis.json      test segment->parca eslesmesi + gercek etiketler
    probs/probs_<nfft>_<hop>_<seed>.npy        45 test olasilik dizisi
    probs_val/probs_<nfft>_<hop>_<seed>.npy    45 dogrulama olasilik dizisi
    probs_val/meta_val.json                    dogrulama eslesmesi + etiketler

Uretilen (out/ klasorune yazilir, varsa figs/ ile karsilastirilir):
    per_seed_results.csv                aggregation_per_seed.csv
    aggregation_summary.csv             aggregation_validation_per_seed.csv
    tie_cases.csv                       posthoc_nemenyi_segment.csv
    posthoc_conover_holm_segment.csv    track_wilson_ci.csv
    computational_cost.csv              ablation_summary.csv
    ablation_two_way.csv                friedman_results.csv

Kullanim:
    python3 reproduce_analysis.py                     # varsayilan yollar
    python3 reproduce_analysis.py --root . --out out --compare figs

Gerekenler: numpy, pandas, scipy, scikit-posthocs, statsmodels
    pip install numpy pandas scipy scikit-posthocs statsmodels
"""
import argparse, glob, json, os, sys
from collections import Counter
import numpy as np
import pandas as pd

FFTS = [512, 1024, 2048]
HOPS = [128, 256, 512]
SEEDS = [42, 123, 456, 789, 1024]
CFGS = [(nf, hp) for nf in FFTS for hp in HOPS]
CFG_LABELS = ["%d/%d" % c for c in CFGS]
EPS = 1e-12          # vote.py ile ayni sayisal koruma

n_ok = n_bad = 0


def compare(name, df, cmp_dir):
    """Uretilen tabloyu mevcut CSV ile karsilastir."""
    global n_ok, n_bad
    p = os.path.join(cmp_dir, name) if cmp_dir else None
    if not p or not os.path.exists(p):
        print("  %-38s uretildi (karsilastirilacak dosya yok)" % name)
        return
    old = pd.read_csv(p)
    new = df.reset_index(drop=True)
    if list(old.columns) != list(new.columns) or len(old) != len(new):
        print("  %-38s >>> YAPI FARKLI <<< (%s satir/sutun)" % (name, "eski %dx%d yeni %dx%d"
              % (len(old), len(old.columns), len(new), len(new.columns))))
        n_bad += 1
        return
    worst, col = 0.0, ""
    for c in old.columns:
        if pd.api.types.is_numeric_dtype(old[c]) and pd.api.types.is_numeric_dtype(new[c]):
            d = np.nanmax(np.abs(old[c].to_numpy(float) - new[c].to_numpy(float))) if len(old) else 0.0
            if d > worst:
                worst, col = d, c
        else:
            if not (old[c].astype(str) == new[c].astype(str)).all():
                print("  %-38s >>> METIN SUTUNU FARKLI <<< (%s)" % (name, c))
                n_bad += 1
                return
    good = worst <= 5e-4
    n_ok += good
    n_bad += (not good)
    print("  %-38s en buyuk fark %.2e (%s)  %s"
          % (name, worst, col or "-", "OK" if good else ">>> FARKLI <<<"))


def save(df, name, out_dir, cmp_dir):
    df.to_csv(os.path.join(out_dir, name), index=False)
    compare(name, df, cmp_dir)


# ------------------------------------------------------------------ yukleme
def load_runs(root, results_dir):
    if results_dir:
        js = sorted(glob.glob(os.path.join(results_dir, "*", "seed_*", "result.json")))
        runs = [json.load(open(p)) for p in js]
    else:
        runs = json.load(open(os.path.join(root, "all_results_combined.json")))
    assert len(runs) == 45, "45 kosu bekleniyordu, %d bulundu" % len(runs)
    return runs


def load_split(root, sub, meta_name, tracks_key, y_key):
    m = json.load(open(os.path.join(root, sub, meta_name)) if sub else
                  open(os.path.join(root, meta_name)))
    tracks = np.array(m[tracks_key])
    y_seg = np.array(m[y_key])
    ut = list(dict.fromkeys(tracks.tolist()))
    groups = [np.where(tracks == t)[0] for t in ut]
    y_track = np.array([y_seg[g[0]] for g in groups])
    return y_seg, groups, y_track, [t.split("/")[-1] for t in ut]


def rules(P, groups):
    """Bes toplama kuralinin parca duzeyi tahminleri + beraberlik sayisi."""
    pred = P.argmax(1)
    hard, ties, tie_rows = [], 0, []
    for i, g in enumerate(groups):
        c = Counter(pred[g].tolist())
        best = c.most_common()[0][1]
        tied = [k for k, v in c.items() if v == best]
        if len(tied) > 1:
            ties += 1
            tie_rows.append((i, len(tied)))
        hard.append(c.most_common()[0][0])        # beraberlik -> ilk gorulen sinif
    return (pred,
            np.array(hard),
            np.array([P[g].mean(0).argmax() for g in groups]),
            np.array([np.log(P[g] + EPS).sum(0).argmax() for g in groups]),
            np.array([np.median(P[g], 0).argmax() for g in groups]),
            np.array([P[g][P[g].max(1).argmax()].argmax() for g in groups]),
            ties, tie_rows)


def macro_f1(y, p, K=10):
    f = []
    for k in range(K):
        tp = np.sum((p == k) & (y == k)); fp = np.sum((p == k) & (y != k)); fn = np.sum((p != k) & (y == k))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


# ------------------------------------------------------------------ maliyet
def conv_macs(H, W, cin, cout, k=3):
    return H * W * cin * cout * k * k


def model_macs(F, T):
    """Ileri gecis MAC sayisi; mimari Tablo 5 ile ayni."""
    m = conv_macs(F, T, 1, 32)
    F2, T2 = F // 2, T // 2
    m += conv_macs(F2, T2, 32, 64)
    F4, T4 = F2 // 2, T2 // 2
    m += conv_macs(F4, T4, 64, 64)
    m += 64 * 64 + 64 * 10                      # Dense-1 + Dense-2
    return m


# ------------------------------------------------------------------ ana
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="out")
    ap.add_argument("--compare", default="figs", help="mevcut CSV klasoru (bos = karsilastirma yok)")
    ap.add_argument("--results", default=None, help="experiment_results klasoru (result.json'lar)")
    ap.add_argument("--mel128-rerun", default=None,
                    help="ablation_results_mel128/ablation_all_results.json — verilirse "
                         "2x3 faktoriyelin mel=128 kolu bu ayni-ortam kosularindan kurulur")
    a = ap.parse_args()
    root, out, cmp_dir = a.root, a.out, a.compare
    os.makedirs(out, exist_ok=True)

    try:
        from scipy import stats
    except ImportError:
        sys.exit("scipy gerekli:  pip install scipy scikit-posthocs statsmodels")

    y_seg, groups, y_track, names = load_split(root, "", "meta_for_analysis.json",
                                               "test_tracks_per_segment", "y_test")
    n_tracks = len(groups)
    print("test: %d segment, %d parca\n" % (len(y_seg), n_tracks))

    # ---------------------------------------------------------- 1. per_seed_results
    print("[1] per_seed_results.csv  <- all_results_combined.json")
    runs = load_runs(root, a.results)
    per_seed = pd.DataFrame([{
        "n_fft": r["fft"], "hop_length": r["hop"], "seed": r["seed"],
        "seg_acc": r["seg_test_acc"], "seg_f1": r["seg_f1_macro"],
        "track_acc": r["track_test_acc"], "track_f1": r["track_f1_macro"],
        "best_epoch": r["best_epoch"], "total_epochs": r["total_epochs"],
        "train_time_sec": r["train_time_sec"]} for r in runs])
    per_seed = per_seed.sort_values(["n_fft", "hop_length", "seed"], key=lambda s:
                                    s.map({v: i for i, v in enumerate(SEEDS)}) if s.name == "seed" else s
                                    ).reset_index(drop=True)
    save(per_seed, "per_seed_results.csv", out, cmp_dir)

    # ---------------------------------------------------------- 2. toplama (test)
    print("\n[2] aggregation_*.csv  <- probs/")
    rows, tie_detail, acc = [], [], {}
    for nf, hp in CFGS:
        for sd in SEEDS:
            P = np.load(os.path.join(root, "probs", "probs_%d_%d_%d.npy" % (nf, hp, sd)))
            pred, hard, soft, logs, medi, maxc, ties, trs = rules(P, groups)
            acc[(nf, hp, sd)] = dict(
                seg=(pred == y_seg).mean(), hard=(hard == y_track).mean(),
                soft=(soft == y_track).mean(), logsum=(logs == y_track).mean(),
                median=(medi == y_track).mean(), maxconf=(maxc == y_track).mean(),
                hard_f1=macro_f1(y_track, hard), soft_f1=macro_f1(y_track, soft), ties=ties,
                seg_f1=macro_f1(y_seg, pred))
            rows.append(dict(n_fft=nf, hop=hp, seed=sd, seg_acc=acc[(nf, hp, sd)]["seg"],
                             hard_acc=acc[(nf, hp, sd)]["hard"], soft_acc=acc[(nf, hp, sd)]["soft"],
                             logsum_acc=acc[(nf, hp, sd)]["logsum"], maxconf_acc=acc[(nf, hp, sd)]["maxconf"],
                             median_acc=acc[(nf, hp, sd)]["median"], hard_f1=acc[(nf, hp, sd)]["hard_f1"],
                             soft_f1=acc[(nf, hp, sd)]["soft_f1"], n_ties=ties))
            for i, nt in trs:
                tie_detail.append(dict(cfg="%d/%d" % (nf, hp), seed=sd, track=names[i],
                                       true=int(y_track[i]), hard=int(hard[i]), soft=int(soft[i]),
                                       n_tied=nt, hard_ok=bool(hard[i] == y_track[i]),
                                       soft_ok=bool(soft[i] == y_track[i])))
    agg = pd.DataFrame(rows)
    save(agg, "aggregation_per_seed.csv", out, cmp_dir)
    save(pd.DataFrame(tie_detail), "tie_cases.csv", out, cmp_dir)

    summ = []
    for nf, hp in CFGS:
        s = agg[(agg.n_fft == nf) & (agg.hop == hp)]
        summ.append(dict(n_fft=nf, hop=hp,
                         seg_acc=round(s.seg_acc.mean(), 4),
                         hard_acc=round(s.hard_acc.mean(), 4), hard_sd=round(s.hard_acc.std(), 4),
                         soft_acc=round(s.soft_acc.mean(), 4), soft_sd=round(s.soft_acc.std(), 4),
                         logsum_acc=round(s.logsum_acc.mean(), 4),
                         median_acc=round(s.median_acc.mean(), 4),
                         maxconf_acc=round(s.maxconf_acc.mean(), 4),
                         hard_f1=round(s.hard_f1.mean(), 4), soft_f1=round(s.soft_f1.mean(), 4),
                         ties=round(s.n_ties.mean(), 4),
                         soft_minus_hard_pp=round(100 * (s.soft_acc.mean() - s.hard_acc.mean()), 2)))
    save(pd.DataFrame(summ), "aggregation_summary.csv", out, cmp_dir)

    # ---------------------------------------------------------- 3. toplama (dogrulama)
    print("\n[3] aggregation_validation_per_seed.csv  <- probs_val/")
    mv = json.load(open(os.path.join(root, "probs_val", "meta_val.json")))
    lab = json.load(open(os.path.join(root, "meta_for_analysis.json")))["label_map"]
    tk = "val_tracks_per_segment" if "val_tracks_per_segment" in mv else "tracks_per_segment"
    trv = np.array(mv[tk])
    # dogrulama etiketleri dosya adinin tur onekinden turetilir (label_map ile ayni kodlama)
    yv = np.array([lab[t.split("/")[-1].split(".")[0]] for t in trv])
    utv = list(dict.fromkeys(trv.tolist()))
    gv = [np.where(trv == t)[0] for t in utv]
    ytv = np.array([yv[g[0]] for g in gv])
    vrows = []
    for nf, hp in CFGS:
        for sd in SEEDS:
            P = np.load(os.path.join(root, "probs_val", "probs_%d_%d_%d.npy" % (nf, hp, sd)))
            pred, hard, soft, logs, medi, maxc, ties, _ = rules(P, gv)
            vrows.append(dict(n_fft=nf, hop=hp, seed=sd, seg_acc=(pred == yv).mean(),
                              hard=(hard == ytv).mean(), soft=(soft == ytv).mean(),
                              logsum=(logs == ytv).mean(), median=(medi == ytv).mean(),
                              maxconf=(maxc == ytv).mean(), soft_f1=macro_f1(ytv, soft), n_ties=ties))
    save(pd.DataFrame(vrows), "aggregation_validation_per_seed.csv", out, cmp_dir)

    # ---------------------------------------------------------- 4. Friedman
    print("\n[4] friedman_results.csv")

    def mat(key, src=acc):
        return np.array([[src[(nf, hp, sd)][key] for nf, hp in CFGS] for sd in SEEDS])

    N, k = len(SEEDS), 9
    fr = []
    for lbl, key in [("segment accuracy", "seg"), ("segment macro-F1", "seg_f1"),
                     ("track accuracy (hard)", "hard"), ("track macro-F1 (hard)", "hard_f1"),
                     ("track accuracy (mean prob.)", "soft"), ("track macro-F1 (mean prob.)", "soft_f1"),
                     ("track accuracy (log prob.)", "logsum")]:
        M = mat(key)
        chi, p = stats.friedmanchisquare(*[M[:, j] for j in range(9)])
        W = chi / (N * (k - 1))                                # Kendall W (uyum katsayisi)
        # Iman-Davenport F duzeltmesi: chi2 yaklasimi k ve N'e duyarli oldugu icin
        # Demsar (2006) bunu onerir. df1 = k-1, df2 = (k-1)(N-1)
        F = ((N - 1) * chi) / (N * (k - 1) - chi)
        pF = stats.f.sf(F, k - 1, (k - 1) * (N - 1))
        fr.append(dict(metric=lbl, chi2=round(chi, 3), df_chi2=k - 1, p_chi2=round(p, 4),
                       F_iman_davenport=round(F, 3), df1=k - 1, df2=(k - 1) * (N - 1),
                       p_F=round(pF, 4), kendall_w=round(W, 3)))
    # Kosu duzeyi Wilcoxon: 45 kosu bagimsiz DEGIL (ayni bes tohum dokuz
    # konfigurasyonda tekrarlaniyor, hepsi ayni test bolmesinde degerlendiriliyor).
    # Makale konfigurasyon duzeyi testi raporluyor; kosu duzeyi yalnizca
    # betimleyici sayim olarak verilir.
    sruns = np.array([acc[c]["soft"] for c in acc]); hruns = np.array([acc[c]["hard"] for c in acc])
    w_run = stats.wilcoxon(sruns, hruns)
    fr.append(dict(metric="wilcoxon: mean prob. vs hard (45 runs, descriptive)",
                   chi2=round(float(w_run.statistic), 3), df_chi2=np.nan,
                   p_chi2=round(float(w_run.pvalue), 6), F_iman_davenport=np.nan,
                   df1=np.nan, df2=np.nan, p_F=np.nan, kendall_w=np.nan))
    print("   kosu duzeyi (betimleyici): %d iyi / %d kotu / %d esit"
          % ((sruns > hruns).sum(), (sruns < hruns).sum(), (sruns == hruns).sum()))

    # Konfigurasyon duzeyi Wilcoxon — MAKALEDE RAPORLANAN TEST
    scfg = np.array([np.mean([acc[(nf, hp, s_)]["soft"] for s_ in SEEDS]) for nf, hp in CFGS])
    hcfg = np.array([np.mean([acc[(nf, hp, s_)]["hard"] for s_ in SEEDS]) for nf, hp in CFGS])
    w_cfg = stats.wilcoxon(scfg, hcfg, alternative="two-sided", method="exact")
    fr.append(dict(metric="wilcoxon: mean prob. vs hard (9 configurations, exact)",
                   chi2=round(float(w_cfg.statistic), 3), df_chi2=np.nan,
                   p_chi2=round(float(w_cfg.pvalue), 4), F_iman_davenport=np.nan,
                   df1=np.nan, df2=np.nan, p_F=np.nan, kendall_w=np.nan))
    print("   konfigurasyon duzeyi (raporlanan): W=%.0f  p=%.4f  | soft>hard %d/9  | +%.2f pp"
          % (w_cfg.statistic, w_cfg.pvalue, (scfg > hcfg).sum(), 100 * (scfg - hcfg).mean()))
    frd = pd.DataFrame(fr)
    save(frd, "friedman_results.csv", out, cmp_dir)
    print(frd.to_string(index=False))

    # ---------------------------------------------------------- 5. post-hoc
    print("\n[5] posthoc_*.csv")
    try:
        import scikit_posthocs as sp
        M = mat("seg")
        nem = sp.posthoc_nemenyi_friedman(M)
        con = sp.posthoc_conover_friedman(M, p_adjust="holm")
        for d in (nem, con):
            d.index = CFG_LABELS; d.columns = CFG_LABELS
        nem_r = nem.round(4).reset_index(); nem_r.columns = ["Unnamed: 0"] + CFG_LABELS
        con_r = con.round(4).reset_index(); con_r.columns = ["Unnamed: 0"] + CFG_LABELS
        save(nem_r, "posthoc_nemenyi_segment.csv", out, cmp_dir)
        save(con_r, "posthoc_conover_holm_segment.csv", out, cmp_dir)
        # parca duzeyi (ortalama olasilik) post-hoc — omnibus anlamli oldugu icin
        Mt = mat("soft")
        nemt = sp.posthoc_nemenyi_friedman(Mt)
        cont = sp.posthoc_conover_friedman(Mt, p_adjust="holm")
        for d in (nemt, cont):
            d.index = CFG_LABELS; d.columns = CFG_LABELS
        nt = nemt.round(4).reset_index(); nt.columns = ["Unnamed: 0"] + CFG_LABELS
        ct = cont.round(4).reset_index(); ct.columns = ["Unnamed: 0"] + CFG_LABELS
        save(nt, "posthoc_nemenyi_track_soft.csv", out, cmp_dir)
        save(ct, "posthoc_conover_holm_track_soft.csv", out, cmp_dir)
        tl = [(CFG_LABELS[i], CFG_LABELS[j], round(nemt.iloc[i, j], 4), round(cont.iloc[i, j], 4))
              for i in range(9) for j in range(i + 1, 9)
              if nemt.iloc[i, j] < 0.05 or cont.iloc[i, j] < 0.05]
        print("   Parca (ortalama olasilik) anlamli cift:", len(tl), tl)

        nl = [(CFG_LABELS[i], CFG_LABELS[j]) for i in range(9) for j in range(i + 1, 9)
              if nem.iloc[i, j] < 0.05]
        cl = [(CFG_LABELS[i], CFG_LABELS[j]) for i in range(9) for j in range(i + 1, 9)
              if con.iloc[i, j] < 0.05]
        print("   Nemenyi anlamli cift:", len(nl), nl)
        print("   Conover-Holm anlamli cift:", len(cl), cl)
        q = 3.102                                    # Nemenyi q_0.05, k=9
        print("   CD = %.2f  (q=%.3f, k=9, N=5)" % (q * np.sqrt(9 * 10 / (6 * 5)), q))
    except ImportError:
        print("   scikit-posthocs kurulu degil - atlandi")

    # ---------------------------------------------------------- 6. Wilson GA
    print("\n[6] track_wilson_ci.csv")
    z = stats.norm.ppf(0.975)
    wr = []
    for nf, hp in CFGS:
        mean_acc = np.mean([acc[(nf, hp, s)]["hard"] for s in SEEDS])
        k = int(round(mean_acc * n_tracks))
        n = n_tracks
        # Wilson araligi bir binom SAYIMI gerektirir; ortalama dogruluk tam sayi
        # bir sayima yuvarlanip aralik o sayim uzerinden hesaplanir.
        p_hat = k / n
        d = 1 + z * z / n
        c = (p_hat + z * z / (2 * n)) / d
        h = z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / d
        wr.append(dict(config="%d/%d" % (nf, hp), track_acc=round(mean_acc, 4), correct_tracks=k,
                       wilson_lo=round(c - h, 4), wilson_hi=round(c + h, 4),
                       ci_width_pp=round(100 * 2 * h, 2)))
    save(pd.DataFrame(wr), "track_wilson_ci.csv", out, cmp_dir)

    # ---------------------------------------------------------- 7. maliyet
    print("\n[7] computational_cost.csv  (MAC analitik; sureler olculen degerlerdir)")
    T_of = {128: 517, 256: 259, 512: 130}
    cr = []
    for nf, hp in CFGS:
        T = T_of[hp]
        m = model_macs(128, T)
        cr.append(dict(config="%d/%d" % (nf, hp), input="128x%d" % T,
                       MMACs=round(m / 1e6, 1), MFLOPs=round(2 * m / 1e6, 1)))
    cost = pd.DataFrame(cr)
    old_p = os.path.join(cmp_dir, "computational_cost.csv") if cmp_dir else None
    if old_p and os.path.exists(old_p):                 # olculen sutunlari devral
        old = pd.read_csv(old_p)
        for c in ["feat_extract_s", "train_s_per_run", "s_per_epoch", "comparable"]:
            if c in old.columns:
                cost[c] = old[c]
    save(cost, "computational_cost.csv", out, cmp_dir)

    # ---------------------------------------------------------- 8. ablasyon
    print("\n[8] ablation_*.csv  <- ablation.json"
          + ("  [mel=128 kolu: AYNI ORTAM yeniden kosumu]" if a.mel128_rerun
             else "  [mel=128 kolu: ana izgaradan — ORTAM KARISIKLIGI VAR]"))
    abl = json.load(open(os.path.join(root, "ablation.json")))
    ab = pd.DataFrame([{"n_fft": r["fft"], "n_mels": r["n_mels"], "seed": r["seed"],
                        "seg_acc": r["seg_test_acc"], "track_hard": r["track_test_acc"],
                        "track_soft": r.get("track_test_acc_soft", np.nan),   # JSON anahtari: track_test_acc_soft
                        "best_epoch": r["best_epoch"], "total_epochs": r["total_epochs"],
                        "train_time_sec": r["train_time_sec"]} for r in abl])
    save(ab, "ablation_per_seed.csv", out, cmp_dir)

    # mel=128 kolu: varsayilan olarak ana izgaradan (hop=256).
    # --mel128-rerun verilirse ayni-ortam kosularindan kurulur ve ortam etkisi olculur.
    base = per_seed[per_seed.hop_length == 256][["n_fft", "seed", "seg_acc", "track_acc"]].copy()
    base["n_mels"] = 128
    if a.mel128_rerun:
        rr = json.load(open(a.mel128_rerun))
        assert len(rr) == 15, "15 kosu bekleniyordu, %d bulundu" % len(rr)
        new128 = pd.DataFrame([{"n_fft": r["fft"], "seed": r["seed"], "n_mels": 128,
                                "seg_acc": r["seg_test_acc"],
                                "track_acc": r["track_test_acc"]} for r in rr])
        print("\n[8a] ORTAM ETKISI  (yeni mel=128 ayni ortamda  vs  eski mel=128 ana izgaradan)")
        env = []
        for nf in FFTS:
            o = base[base.n_fft == nf]; n_ = new128[new128.n_fft == nf]
            env.append(dict(n_fft=nf,
                            seg_old=round(o.seg_acc.mean(), 4), seg_new=round(n_.seg_acc.mean(), 4),
                            seg_shift_pp=round(100 * (n_.seg_acc.mean() - o.seg_acc.mean()), 2),
                            trk_old=round(o.track_acc.mean(), 4), trk_new=round(n_.track_acc.mean(), 4),
                            trk_shift_pp=round(100 * (n_.track_acc.mean() - o.track_acc.mean()), 2)))
        envd = pd.DataFrame(env)
        save(envd, "environment_effect.csv", out, cmp_dir)
        print(envd.to_string(index=False))
        sh = envd.seg_shift_pp
        print("   segment kaymasi: ortalama %+.2f pp  (aralik %+.2f .. %+.2f)"
              % (sh.mean(), sh.min(), sh.max()))
        base = new128[["n_fft", "seed", "seg_acc", "track_acc", "n_mels"]]
    two = pd.concat([
        base.rename(columns={"seg_acc": "seg", "track_acc": "trk"})[["n_fft", "n_mels", "seed", "seg", "trk"]],
        ab.rename(columns={"seg_acc": "seg", "track_hard": "trk"})[["n_fft", "n_mels", "seed", "seg", "trk"]],
    ]).sort_values(["n_mels", "n_fft", "seed"], ascending=[False, True, True]).reset_index(drop=True)
    save(two, "ablation_two_way.csv", out, cmp_dir)

    # ---- planlanmis esli karsilastirmalar: 512 vs 2048, her mel seviyesinde
    # Isaret konvansiyonu: fark = 512 - 2048 (makaledeki yon).
    print("\n[8b] ablation_paired_tests.csv  (esli t, Holm)")
    prows, praw = [], []
    for M in (128, 64):
        sm_ = two[(two.n_mels == M) & (two.n_fft == 512)].sort_values("seed").seg.to_numpy()
        lg = two[(two.n_mels == M) & (two.n_fft == 2048)].sort_values("seed").seg.to_numpy()
        t = stats.ttest_rel(sm_, lg)                      # 512 - 2048
        praw.append(float(t.pvalue))
        prows.append(dict(n_mels=M, contrast="512 minus 2048",
                          diff_pp=round(100 * float(np.mean(sm_ - lg)), 2),
                          t=round(float(t.statistic), 3), df=len(SEEDS) - 1,
                          p=round(float(t.pvalue), 4)))
    order = np.argsort(praw)                              # Holm (2 karsilastirma)
    adj = [0.0, 0.0]
    adj[order[0]] = min(1.0, 2 * praw[order[0]])
    adj[order[1]] = min(1.0, max(adj[order[0]], praw[order[1]]))
    for r, a_ in zip(prows, adj):
        r["p_holm"] = round(a_, 4)
    save(pd.DataFrame(prows), "ablation_paired_tests.csv", out, cmp_dir)

    # ---- iki yonlu TEKRARLI OLCUM ANOVA'si (blok = tohum)
    print("\n[8c] ablation_rm_anova.csv  (tekrarli olcum, blok = tohum)")
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import AnovaRM
        rows_a = []
        for dep, lbl in [("seg", "segment accuracy"), ("trk", "track accuracy (hard)")]:
            rm = AnovaRM(two, depvar=dep, subject="seed",
                         within=["n_fft", "n_mels"]).fit().anova_table
            # SS paylari OLS ayristirmasindan; toplam SS tohum bilesenlerini de
            # icerdigi icin bu paylar TANIMLAYICIDIR ve %100'e toplanmaz.
            # Artik payi RM modelinde tek bir hata terimi DEGILDIR -> raporlanmaz.
            oa = sm.stats.anova_lm(ols("%s ~ C(n_fft)*C(n_mels)" % dep, data=two).fit(), typ=2)
            tot = float(oa["sum_sq"].sum())
            for src, key in [("n_fft", "C(n_fft)"), ("n_mels", "C(n_mels)"),
                             ("n_fft:n_mels", "C(n_fft):C(n_mels)")]:
                rows_a.append(dict(
                    level=lbl, source=src,
                    # _F_exact: yuvarlanmamis F. p_gg BUNDAN hesaplanir; yuvarlanmis
                    # F kullanilirsa p_gg dorduncu basamakta kayiyor (or. 0.9836 vs 0.9829).
                    _F_exact=float(rm.loc[src, "F Value"]),
                    F=round(float(rm.loc[src, "F Value"]), 3),
                    df1=int(rm.loc[src, "Num DF"]), df2=int(rm.loc[src, "Den DF"]),
                    p=round(float(rm.loc[src, "Pr > F"]), 4),
                    pct_total_ss=round(100 * float(oa.loc[key, "sum_sq"]) / tot, 1)))
        # Greenhouse-Geisser: uc seviyeli n_fft faktoru kurellik varsayimi tasir.
        # eps < 1 ise serbestlik dereceleri eps ile carpilir (Demsar disi, klasik RM-ANOVA pratigi).
        def gg_eps(X):
            k = X.shape[1]
            S = np.cov(X, rowvar=False)
            Sb = S - S.mean(0)[None, :] - S.mean(1)[:, None] + S.mean()
            ev = np.linalg.eigvalsh(Sb)
            ev = ev[ev > 1e-12]
            return float((ev.sum() ** 2) / ((k - 1) * (ev ** 2).sum()))

        for r in rows_a:
            if r["source"] == "n_mels":
                r["gg_eps"] = np.nan; r["p_gg"] = r["p"]      # 2 seviye -> kurellik konusu yok
                continue
            dep = "seg" if r["level"].startswith("segment") else "trk"
            if r["source"] == "n_fft":
                X = np.array([[two[(two.seed == s_) & (two.n_fft == f)][dep].mean()
                               for f in FFTS] for s_ in SEEDS])
            else:                                            # etkilesim: fark skorlari
                X = np.array([[two[(two.seed == s_) & (two.n_fft == f) & (two.n_mels == 128)][dep].iloc[0] -
                               two[(two.seed == s_) & (two.n_fft == f) & (two.n_mels == 64)][dep].iloc[0]
                               for f in FFTS] for s_ in SEEDS])
            e = gg_eps(X)
            r["gg_eps"] = round(e, 3)
            r["p_gg"] = round(float(stats.f.sf(r["_F_exact"], r["df1"] * e, r["df2"] * e)), 4)

        rmd = pd.DataFrame(rows_a).drop(columns=["_F_exact"])
        save(rmd, "ablation_rm_anova.csv", out, cmp_dir)
        print(rmd.to_string(index=False))
    except ImportError:
        print("   statsmodels kurulu degil - ANOVA atlandi")

    print("\n" + "=" * 78)
    print("  %d tablo mevcut CSV ile ayni, %d tablo FARKLI" % (n_ok, n_bad))
    print("  cikti klasoru:", os.path.abspath(out))
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
