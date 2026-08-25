import numpy as np, json, os, wave, sys, time
import melnp
SR=22050; DUR=30; SEG=3; NSEG=10; SEGLEN=SR*SEG
BASE=os.path.expanduser("~/mnt/NEW PAPER GTZAN")
AUDIO=BASE+"/Arşiv/GTZAN DATA/genres_original"
REF=BASE+"/Arşiv/extracted_features_with_segments_new/fft1024_hop256_mel128_seg3"

def load_wav(p):
    with wave.open(p,'rb') as w:
        assert w.getsampwidth()==2, p
        sr=w.getframerate(); ch=w.getnchannels()
        raw=w.readframes(w.getnframes())
    y=np.frombuffer(raw,dtype='<i2').astype(np.float32)/32768.0
    if ch>1: y=y.reshape(-1,ch).mean(1)
    assert sr==SR, (p,sr)
    n=SR*DUR
    if len(y)<n: y=np.pad(y,(0,n-len(y)))
    return y[:n]

def segs_of(y):
    return [y[i*SEGLEN:(i+1)*SEGLEN] for i in range(NSEG)]

def feat(seg,n_fft,hop,n_mels):
    S=melnp.logmel(seg,SR,n_fft,hop,n_mels).astype(np.float32)
    return (S-S.mean())/(S.std()+1e-8)

if __name__=="__main__":
    mode=sys.argv[1]
    if mode=="verify":
        segs=json.load(open(REF+"/segments.json")); X=np.load(REF+"/X.npy",mmap_mode='r')
        tracks=list(dict.fromkeys(s["track_path"] for s in segs))
        idx={}
        for i,s in enumerate(segs): idx.setdefault(s["track_path"],[]).append(i)
        rng=np.random.default_rng(0); pick=rng.choice(len(tracks),6,replace=False)
        worst=0
        for k in pick:
            t=tracks[k]; p=AUDIO+"/"+"/".join(t.split("/")[-2:])
            y=load_wav(p)
            for j,seg in enumerate(segs_of(y)):
                F=feat(seg,1024,256,128); G=np.asarray(X[idx[t][j],:,:,0],dtype=np.float32)
                d=np.abs(F-G).max(); worst=max(worst,d)
            print("  %-22s maks|fark|=%.4e"%(t.split("/")[-1],worst))
        print("EN KOTU FARK: %.4e"%worst)
    else:
        n_fft,hop,n_mels=int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
        out=BASE+"/revision_work/feat_mel%d_fft%d_hop%d"%(n_mels,n_fft,hop)
        os.makedirs(out,exist_ok=True)
        segs=json.load(open(REF+"/segments.json"))
        order=[s["track_path"] for s in segs]
        tracks=list(dict.fromkeys(order))
        W=1+(SEGLEN)//hop
        done=os.path.join(out,"X.npy")
        mm=np.lib.format.open_memmap(done,mode=('r+' if os.path.exists(done) else 'w+'),
                                     dtype=np.float32,shape=(len(order),n_mels,W,1))
        st=int(sys.argv[5]); en=int(sys.argv[6]); t0=time.time()
        idx={}
        for i,s in enumerate(segs): idx.setdefault(s["track_path"],[]).append(i)
        for k in range(st,min(en,len(tracks))):
            t=tracks[k]; y=load_wav(AUDIO+"/"+"/".join(t.split("/")[-2:]))
            for j,seg in enumerate(segs_of(y)):
                mm[idx[t][j],:,:,0]=feat(seg,n_fft,hop,n_mels)
        mm.flush()
        print("parca %d-%d bitti  %.1fs"%(st,min(en,len(tracks)),time.time()-t0))
