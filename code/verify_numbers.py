#!/usr/bin/env python3
"""
verify_numbers.py — makalede raporlanan her sayiyi ham veriden yeniden hesaplar
ve iddia edilen degerle karsilastirir.

Kullanim (revision_work klasorunun icinden):
    python3 verify_numbers.py
    python3 verify_numbers.py --results /path/to/experiment_results   # epoch istatistikleri icin
    python3 verify_numbers.py --audio   /path/to/genres_original      # kopya denetimi icin

Gereken dosyalar (revision_work icinde):
    meta_for_analysis.json      test segment->parca eslesmesi ve gercek etiketler
    probs/probs_<nfft>_<hop>_<seed>.npy   45 test olasilik dizisi (1500 x 10)

Her satir OK ise makaledeki sayi ham veriyle uyusuyor demektir.
"""
import argparse, glob, hashlib, json, os, sys
from collections import Counter
import numpy as np

ok_count = 0
bad_count = 0


def chk(label, got, want, tol=5e-5, fmt="%.4f"):
    """Hesaplanan degeri iddia edilenle karsilastir ve yazdir."""
    global ok_count, bad_count
    good = abs(got - want) <= tol
    ok_count += good
    bad_count += (not good)
    print("  %-46s hesaplanan=%s  makale=%s  %s"
          % (label, fmt % got, fmt % want, "OK" if good else ">>> FARKLI <<<"))


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ---------------------------------------------------------------- yukleme
def load_test_data(root):
    meta_p = os.path.join(root, "meta_for_analysis.json")
    if not os.path.exists(meta_p):
        sys.exit("meta_for_analysis.json bulunamadi: %s" % meta_p)
    m = json.load(open(meta_p))
    tracks = np.array(m["test_tracks_per_segment"])
    y_seg = np.array(m["y_test"])
    # parcalar ilk gorunme sirasina gore gruplanir (vote.py ile ayni)
    utracks = list(dict.fromkeys(tracks.tolist()))
    groups = [np.where(tracks == t)[0] for t in utracks]
    y_track = np.array([y_seg[g[0]] for g in groups])
    assert all(len(set(y_seg[g].tolist())) == 1 for g in groups), \
        "bir parcanin segmentleri farkli etiketler tasiyor"
    names = [t.split("/")[-1] for t in utracks]
    return y_seg, groups, y_track, names


FFTS = [512, 1024, 2048]
HOPS = [128, 256, 512]
SEEDS = [42, 123, 456, 789, 1024]


def all_runs(root):
    for nf in FFTS:
        for hp in HOPS:
            for sd in SEEDS:
                p = os.path.join(root, "probs", "probs_%d_%d_%d.npy" % (nf, hp, sd))
                if not os.path.exists(p):
                    sys.exit("eksik dosya: %s" % p)
                yield nf, hp, sd, np.load(p)


# ---------------------------------------------------------------- ana analiz
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="revision_work klasoru")
    ap.add_argument("--results", default=None, help="Colab experiment_results klasoru")
    ap.add_argument("--audio", default=None, help="GTZAN genres_original klasoru")
    a = ap.parse_args()
    root = a.root

    y_seg, groups, y_track, names = load_test_data(root)
    n_tracks = len(groups)
    print("test: %d segment, %d parca" % (len(y_seg), n_tracks))

    seg, hard, soft, logs, medi, maxc, ties = {}, {}, {}, {}, {}, {}, 0
    for nf, hp, sd, P in all_runs(root):
        pred = P.argmax(1)
        seg[(nf, hp, sd)] = (pred == y_seg).mean()
        h = []
        for g in groups:
            c = Counter(pred[g].tolist())
            best = c.most_common()[0][1]
            if sum(1 for v in c.values() if v == best) > 1:
                ties += 1
            h.append(c.most_common()[0][0])          # beraberlik -> ilk gorulen sinif
        hard[(nf, hp, sd)] = (np.array(h) == y_track).mean()
        soft[(nf, hp, sd)] = (np.array([P[g].mean(0).argmax() for g in groups]) == y_track).mean()
        logs[(nf, hp, sd)] = (np.array([np.log(P[g] + 1e-12).sum(0).argmax() for g in groups]) == y_track).mean()
        medi[(nf, hp, sd)] = (np.array([np.median(P[g], 0).argmax() for g in groups]) == y_track).mean()
        maxc[(nf, hp, sd)] = (np.array([P[g][P[g].max(1).argmax()].argmax() for g in groups]) == y_track).mean()

    def cfg_means(d):
        return np.array([np.mean([d[(nf, hp, s)] for s in SEEDS]) for nf in FFTS for hp in HOPS])

    # ---- 3.1 Overall Performance
    section("3.1  Overall Performance Evaluation")
    print("  NOT: segment dogruluklari ONNX cikarimindan; makaledeki degerler")
    print("       TensorFlow degerlendirmesinden gelir. Iki ondalikta ayni.")
    chk("45 kosu ortalamasi, segment", float(np.mean(list(seg.values()))), 0.7871)
    chk("45 kosu ortalamasi, parca (sert oylama)", float(np.mean(list(hard.values()))), 0.8519)
    cs, ct = cfg_means(seg), cfg_means(hard)
    chk("konfigurasyon ortalamasi, segment  en dusuk", float(cs.min()), 0.7639)
    chk("konfigurasyon ortalamasi, segment  en yuksek", float(cs.max()), 0.7991)
    chk("konfigurasyon ortalamasi, parca    en dusuk", float(ct.min()), 0.8413)
    chk("konfigurasyon ortalamasi, parca    en yuksek", float(ct.max()), 0.8600)
    chk("yayilim, segment (puan)", float(100 * (cs.max() - cs.min())), 3.52, tol=0.01, fmt="%.2f")
    chk("yayilim, parca   (puan)", float(100 * (ct.max() - ct.min())), 1.87, tol=0.01, fmt="%.2f")

    # ---- 3.4 Aggregation
    section("3.4  Temporal Aggregation Strategy")
    for nm, d, want in [("sert oylama", hard, 0.8519), ("ortalama olasilik", soft, 0.8637),
                        ("log-olasilik", logs, 0.8662), ("medyan", medi, 0.8585),
                        ("maksimum guven", maxc, 0.8330)]:
        chk("45 kosu ortalamasi, %s" % nm, float(np.mean(list(d.values()))), want)
    wins = sum(1 for nf in FFTS for hp in HOPS
               if np.mean([soft[(nf, hp, s)] for s in SEEDS]) > np.mean([hard[(nf, hp, s)] for s in SEEDS]))
    chk("yumusak > sert olan konfigurasyon sayisi", float(wins), 9.0, tol=0.5, fmt="%.0f")
    chk("yumusak toplamanin kazanci (puan)",
        float(100 * (np.mean(list(soft.values())) - np.mean(list(hard.values())))), 1.19,
        tol=0.02, fmt="%.2f")
    n_dec = 45 * n_tracks
    chk("toplam parca duzeyi karar sayisi", float(n_dec), 6750.0, tol=0.5, fmt="%.0f")
    chk("beraberlik sayisi", float(ties), 167.0, tol=0.5, fmt="%.0f")
    chk("beraberlik orani (%)", float(100 * ties / n_dec), 2.47, tol=0.01, fmt="%.2f")

    # olasilik tabanli kurallarda tam beraberlik var mi
    tie_prob = 0
    margin = 1.0
    for nf, hp, sd, P in all_runs(root):
        for g in groups:
            for s in (P[g].mean(0), np.log(P[g] + 1e-12).sum(0)):
                t = np.sort(s)[::-1]
                tie_prob += int(t[0] == t[1])
                if s.max() <= 1.0:
                    margin = min(margin, float(t[0] - t[1]))
    chk("olasilik tabanli kurallarda tam beraberlik", float(tie_prob), 0.0, tol=0.5, fmt="%.0f")
    print("  %-46s %.3e" % ("ortalama-olasilikta en kucuk ilk-iki farki", margin))

    # ---- 2.1.1 duyarlilik kontrolu
    section("2.1.1  Kopya duyarlilik kontrolu")
    bad = [i for i, n in enumerate(names)
           if n.startswith("hiphop.00039") or n.startswith("metal.00045")]
    print("  disarida birakilan test parcalari:", [names[i] for i in bad])
    keep = np.array([i for i in range(n_tracks) if i not in bad])
    kh, ks = [], []
    for nf, hp, sd, P in all_runs(root):
        pred = P.argmax(1)
        h = np.array([Counter(pred[g].tolist()).most_common()[0][0] for g in groups])
        s = np.array([P[g].mean(0).argmax() for g in groups])
        kh.append((h[keep] == y_track[keep]).mean())
        ks.append((s[keep] == y_track[keep]).mean())
    chk("148 parca, sert oylama", float(np.mean(kh)), 0.8498)
    chk("148 parca, yumusak toplama", float(np.mean(ks)), 0.8619)
    chk("dusus, sert oylama (puan)", float(100 * (np.mean(kh) - np.mean(list(hard.values())))),
        -0.20, tol=0.02, fmt="%+.2f")
    chk("dusus, yumusak toplama (puan)", float(100 * (np.mean(ks) - np.mean(list(soft.values())))),
        -0.18, tol=0.02, fmt="%+.2f")

    # ---- 2.5 epoch istatistikleri (opsiyonel)
    if a.results:
        section("2.5  Training Configuration (result.json dosyalarindan)")
        js = sorted(glob.glob(os.path.join(a.results, "*", "seed_*", "result.json")))
        if not js:
            print("  result.json bulunamadi:", a.results)
        else:
            be = np.array([json.load(open(p))["best_epoch"] for p in js])
            te = np.array([json.load(open(p))["total_epochs"] for p in js])
            print("  bulunan kosu sayisi:", len(js), "(45 olmali)")
            chk("en iyi epoch, en dusuk", float(be.min()), 53.0, tol=0.5, fmt="%.0f")
            chk("en iyi epoch, en yuksek", float(be.max()), 112.0, tol=0.5, fmt="%.0f")
            chk("en iyi epoch, ortalama", float(be.mean()), 85.8, tol=0.05, fmt="%.2f")
            chk("durma epoch'u, en dusuk", float(te.min()), 63.0, tol=0.5, fmt="%.0f")
            chk("durma epoch'u, en yuksek", float(te.max()), 120.0, tol=0.5, fmt="%.0f")
            chk("durma epoch'u, ortalama", float(te.mean()), 95.8, tol=0.05, fmt="%.2f")
            chk("en iyi epoch > 100 olan kosu", float((be > 100).sum()), 9.0, tol=0.5, fmt="%.0f")
            chk("butceye ulasan kosu (120)", float((te == 120).sum()), 1.0, tol=0.5, fmt="%.0f")
            chk("erken durdurmayla biten kosu", float((te < 120).sum()), 44.0, tol=0.5, fmt="%.0f")

    # ---- 2.1.1 kopya denetimi (opsiyonel, ham ses gerekir)
    if a.audio:
        section("2.1.1  Bit duzeyinde kopya denetimi (ham PCM)")
        import wave
        h2f = {}
        n_read = 0
        for p in sorted(glob.glob(os.path.join(a.audio, "*", "*.wav"))):
            try:
                with wave.open(p, "rb") as w:
                    raw = w.readframes(w.getnframes())
            except Exception:
                print("  cozulemedi (beklenen: jazz.00054.wav):", os.path.basename(p))
                continue
            n_read += 1
            h2f.setdefault(hashlib.sha256(raw).hexdigest(), []).append(os.path.basename(p))
        dup = [v for v in h2f.values() if len(v) > 1]
        chk("cozulebilen dosya sayisi", float(n_read), 999.0, tol=0.5, fmt="%.0f")
        chk("bit-ozdes grup sayisi", float(len(dup)), 14.0, tol=0.5, fmt="%.0f")
        chk("bu gruplardaki kayit sayisi", float(sum(len(v) for v in dup)), 28.0, tol=0.5, fmt="%.0f")
        for v in dup:
            print("     ", " + ".join(v))

    # ---- 2.3.1 mel filtre bankasi (opsiyonel, librosa gerekir)
    try:
        import librosa
        section("2.3.1  Conditioning of the mel filter bank")
        want = {(128, 512): (109, 64, 2.5), (128, 1024): (128, 26, 5.0), (128, 2048): (128, 0, 10.0),
                (64, 512): (64, 14, 5.0), (64, 1024): (64, 0, 9.5), (64, 2048): (64, 0, 19.5)}
        for (M, N), (r_, f_, s_) in want.items():
            F = librosa.filters.mel(sr=22050, n_fft=N, n_mels=M, fmin=0, fmax=11025,
                                    htk=False, norm="slaney")
            sup = (F > 0).sum(1)
            chk("n_mels=%3d n_fft=%4d  rank" % (M, N), float(np.linalg.matrix_rank(F)), float(r_), tol=0.5, fmt="%.0f")
            chk("n_mels=%3d n_fft=%4d  <=2 bin filtre" % (M, N), float((sup <= 2).sum()), float(f_), tol=0.5, fmt="%.0f")
            chk("n_mels=%3d n_fft=%4d  medyan destek" % (M, N), float(np.median(sup)), float(s_), tol=0.05, fmt="%.1f")
    except ImportError:
        print("\n(librosa kurulu degil - mel filtre bankasi tablosu atlandi)")

    section("SONUC")
    print("  %d kontrol OK, %d kontrol FARKLI" % (ok_count, bad_count))
    sys.exit(1 if bad_count else 0)


if __name__ == "__main__":
    main()
