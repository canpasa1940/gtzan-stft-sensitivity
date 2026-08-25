import os, glob, keras, numpy as np, warnings
warnings.filterwarnings("ignore")
root="/mnt/user-data/uploads/NEW PAPER GTZAN/Arşiv/experiment_results"
W={128:517,256:259,512:130}
files=sorted(glob.glob(root+"/*/seed_*/best_model.keras"))
print("bulundu:",len(files))
ok=0
for f in files:
    parts=f.split("/"); cfg=parts[-3]; seed=parts[-2].replace("seed_","")
    nf=int(cfg.split("_")[0][3:]); hp=int(cfg.split("_")[1][3:])
    out=f"/home/claude/onnx/{nf}_{hp}_{seed}.onnx"
    if os.path.exists(out): ok+=1; continue
    m=keras.saving.load_model(f, compile=False)
    _=m.predict(np.zeros((1,128,W[hp],1),dtype='float32'),verbose=0)
    m.export(out, format='onnx', input_signature=[keras.InputSpec(shape=(None,128,W[hp],1),dtype='float32')], verbose=False)
    ok+=1
print("cevrildi:",ok)
