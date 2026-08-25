import numpy as np, onnxruntime as ort, sys, os, time
BASE=os.path.expanduser("~/mnt/NEW PAPER GTZAN")
FEAT=BASE+"/Arşiv/extracted_features_with_segments_new"
OUT=BASE+"/revision_work/probs_val"
nf,hp=int(sys.argv[1]),int(sys.argv[2]); seeds=sys.argv[3].split(",")
BS={128:16,256:64,512:150}[hp]
d=f"{FEAT}/fft{nf}_hop{hp}_mel128_seg3/track_level_split"
X=np.load(d+"/X_val.npy", mmap_mode='r'); y=np.load(d+"/y_val.npy")
so=ort.SessionOptions(); so.intra_op_num_threads=0
for sd in seeds:
    o=f"{OUT}/probs_{nf}_{hp}_{sd}.npy"
    if os.path.exists(o): print("skip",nf,hp,sd); continue
    t=time.time()
    s=ort.InferenceSession(f"{BASE}/revision_work/onnx/{nf}_{hp}_{sd}.onnx", so, providers=["CPUExecutionProvider"])
    inp=s.get_inputs()[0].name; P=[]
    for i in range(0,len(X),BS):
        P.append(s.run(None,{inp:np.ascontiguousarray(X[i:i+BS],dtype=np.float32)})[0])
    P=np.concatenate(P).astype(np.float32); np.save(o,P)
    print("%d/%d seed%s valsegacc=%.4f %.1fs"%(nf,hp,sd,(P.argmax(1)==y).mean(),time.time()-t))
np.save(OUT+"/y_val.npy", y)
