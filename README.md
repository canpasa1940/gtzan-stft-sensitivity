# STFT Parameter Sensitivity in a Lightweight CNN Pipeline for Music Genre Classification

Code, stored model outputs and analysis scripts for the study

> **A Controlled Sensitivity Analysis of STFT Parameters in a Lightweight CNN Pipeline for Music Genre Classification**
> Can Paşa, Music Technology Programme, İnönü University

Nine STFT configurations (3 `n_fft` × 3 `hop_length`) are trained five times each
on GTZAN with a fixed track-level partition and a single CNN architecture, giving
45 main-grid runs. Two further analyses are included: a comparison of five
track-level aggregation rules applied post hoc to the stored model outputs, and a
2×3 ablation on the interaction between `n_fft` and `n_mels` (30 additional runs).

Everything reported in the paper can be recomputed from the files in this
repository. **No number in the paper is produced by hand.**

---

## Quick start

```bash
git clone https://github.com/canpasa1940/gtzan-stft-sensitivity.git
cd gtzan-stft-sensitivity
pip install -r requirements.txt

# 1) Check every number reported in the paper against the raw records
python code/verify_numbers.py --root data

# 2) Regenerate every derived table
python code/reproduce_analysis.py --root data --out out \
       --mel128-rerun data/mel128_rerun.json --compare results/tables
```

`verify_numbers.py` prints one line per check and ends with a pass/fail count
(23 core checks; 23 pass). If the optional `librosa` dependency is installed,
it also runs 18 mel-filter-bank checks (41 checks in total; 41 pass).
`reproduce_analysis.py` writes 17 CSV files to `out/` and compares each against
the copy in `results/tables/`, reporting the largest absolute difference per
table (17 tables; 17 identical).

Both scripts print their progress messages in Turkish. The words that matter:
`OK` / `FARKLI` = matches / differs, `kontrol` = check, `tablo` = table,
`kosu` = run, `konfigurasyon` = configuration, `hesaplanan` = recomputed,
`makale` = as reported in the paper, `en buyuk fark` = largest difference,
`uretildi` = written (no stored copy to compare against). The CSV files
themselves, and all column names, are in English.

To publish this directory as a repository:

```bash
git init
git add .
git commit -m "Code, stored model outputs and analysis scripts"
git branch -M main
git remote add origin https://github.com/canpasa1940/gtzan-stft-sensitivity.git
git push -u origin main
```

The repository is about 11 MB, so it needs no Git LFS.

---

## Repository layout

```
code/
  verify_numbers.py        recomputes every paper number from the raw records
  reproduce_analysis.py    regenerates all derived tables (17 CSVs)
  extract_dev.py           independent NumPy re-implementation of the front end
  melnp.py                 mel filter bank / log-mel, no librosa dependency
  run_infer.py             ONNX inference over the test partition
  run_infer_val.py         ONNX inference over the validation partition
  training/
    featextract_src.py     feature extraction as executed
    train_src.py           training loop as executed (EPOCHS = 120)
    vote.py                the five aggregation rules
    conv.py                Keras -> ONNX conversion
    mkfigs.py, mkinter.py  figure generation

notebooks/
  Data_Prep.ipynb              partitioning and segmentation
  Feature_Extraction.ipynb     main-grid log-mel extraction (local, Python 3.12/3.13)
  single_config_train_v2.ipynb the notebook that produced the 45 main-grid runs
  n_mels_ablation_colab.ipynb  ablation, n_mels = 64  (15 runs)
  mel128_rerun_colab.ipynb     ablation, n_mels = 128 (15 runs, same-stack rerun)

data/
  all_results_combined.json  45 main-grid runs, as written by the training loop
  ablation.json              15 ablation runs at n_mels = 64
  mel128_rerun.json          15 ablation runs at n_mels = 128
  meta_for_analysis.json     test segment -> track mapping and ground-truth labels
  track_list.json            the fixed train/validation/test track lists
  exact_duplicates.json      sample-identical groups found in the decoded audio
  probs/                     45 test-set softmax arrays (1500 x 10, float32)
  probs_val/                 45 validation-set softmax arrays

results/
  tables/    17 CSV files — every derived table in the paper
  figures/   the figures as they appear in the paper
```

---

## Experimental setup

| | |
|---|---|
| Dataset | GTZAN, 999 usable tracks (`jazz.00054.wav` does not decode) |
| Segments | 9,990 (999 × 10 × 3 s), non-overlapping |
| Partition | track level, 699 / 150 / 150 tracks, `random_state=42`, fixed across all configurations |
| Grid | `n_fft` ∈ {512, 1024, 2048} × `hop_length` ∈ {128, 256, 512} |
| Seeds | 42, 123, 456, 789, 1024 |
| Front end | log-mel, Slaney scale (`htk=False`), `n_mels=128`, `power=2.0`, `ref=np.max`, `top_db=80`, per-segment z-score |
| Model | 3 × (Conv2D–BN–SpatialDropout–MaxPool) → GAP → Dense(64) → Dropout(0.5) → Dense(10); **61,194 parameters** |
| Training | 120 epochs max, early stopping (patience 10, `restore_best_weights=True`), `ReduceLROnPlateau`, batch 32, L2 5e-4, label smoothing |

The full front-end specification, including every setting left at its library
default, is given in Table 2 of the paper.

---

## Notes for anyone reusing these files

These are the things that are easy to get wrong. They are documented here
because they affect the numbers.

**GTZAN audio is not redistributed here.** The dataset is available from its
original sources. `data/track_list.json` gives the exact partition, so the split
can be reproduced without guessing.

**The stored probability arrays come from ONNX, the reported segment accuracies
from TensorFlow.** They agree on 42 of the 45 runs exactly. In three runs (all
`2048/128`) they differ by a single segment out of 1500 — 3 differing segment
decisions out of 67,500 (0.004%). Track-level predictions are identical in
45/45 runs. Mean segment accuracy is 78.7141% under TensorFlow and 78.7126%
under ONNX; both round to 78.71%.

**Do not read four-decimal standard deviations off `results/tables/ablation_per_seed.csv`.**
That file carries float32 rounding from the stored records and drifts in the
fourth decimal (for example the mel=64 / `n_fft`=512 mean-probability SD is
0.0144 there and 0.0145 in the primary record). Use `data/ablation.json` and
`data/mel128_rerun.json` as the source.

**The signed-rank comparison of aggregation rules is computed over the nine
configuration means, not the 45 runs.** The same five seed values are repeated
across all nine configurations (a crossed design, not a nested one) and all 45
runs are scored on the same 150-track test partition, so the runs are not
mutually independent. `reproduce_analysis.py` emits both: the run-level test is
labelled *descriptive*, the configuration-level exact test (W = 0, p = 0.0039) is
the one reported in the paper.

**The `n_mels`=128 arm of the ablation was retrained rather than taken from the
main grid.** The main grid ran locally months earlier under a different software
stack; reusing it would have confounded the `n_mels` main effect with the runtime
environment. Both ablation arms ran under TensorFlow 2.20.0 and librosa 0.11.0 on
Colab GPU runtimes. The measured environment shift on the retrained cells is
+0.11 / +0.16 / +0.33 percentage points (mean +0.20), smaller than the
seed-to-seed spread within any cell and not significant in any cell (all p ≥ 0.41).

**Greenhouse–Geisser p-values must be computed from the unrounded F.** Using the
rounded value (F = 0.008) gives 0.9836; the correct value from F = 0.008426 is
0.9829. `reproduce_analysis.py` keeps an unrounded column for this reason.

**Reducing `n_mels` changes more than the filter bank.** Halving it from 128 to 64
also halves the input height and the CNN forward-pass cost (237.2 → 118.6 MMACs
per segment at `hop_length`=256), and changes the frequency-axis sampling seen by
the convolutional layers. The parameter count stays at 61,194 because of global
average pooling, but the cost does not. The ablation therefore measures the joint
effect of these changes rather than isolating filter-bank conditioning.

**The exact librosa patch version used for the main-grid extraction was not
recorded.** The extraction log constrains it to the 0.10 or 0.11 release series.
Eight of the nine configurations were extracted under Python 3.12; the
`n_fft`=512, `hop_length`=128 configuration was re-extracted later under Python
3.13. Stored features from both environments were checked against an independent
NumPy implementation on randomly selected tracks and agreed to within a maximum
absolute difference of 1.2e-6 in each case. The ablation versions are recorded
exactly (librosa 0.11.0, TensorFlow 2.20.0).

---

## Headline results

| | |
|---|---|
| Segment-level accuracy, configuration means | 76.39% – 79.91% |
| Track-level accuracy, hard voting | 84.13% – 86.00% |
| Track-level accuracy, mean-probability aggregation | 84.93% – 88.00% |
| Mean-probability vs hard voting | better in 9/9 configurations, +1.19 pp (W = 0, p = 0.0039) |
| `n_fft` main effect (ablation, segment level) | F(2,8) = 0.008, p = 0.9916 |
| `n_mels` main effect (ablation, segment level) | F(1,4) = 4.588, p = 0.0989 |
| `n_fft` × `n_mels` interaction | F(2,8) = 11.067, p = 0.0050 (GG ε = 0.971, p = 0.0055) |
| CNN forward-pass cost | depends on `hop_length` only: 475.6 / 237.2 / 119.2 MMACs |

Neither main effect is significant; the interaction is. The association between
`n_fft` and segment-level accuracy is positive at 128 mel bands and negative
at 64.

---

## Requirements

Python 3.10 or later. See `requirements.txt`. The analysis scripts need only
NumPy, pandas, SciPy, scikit-learn, scikit-posthocs and statsmodels — TensorFlow
is **not** required to reproduce any table, because the model outputs are stored.

---

## Licence

Code is released under the MIT Licence (`LICENSE`). The stored outputs under
`data/` and `results/` are released under CC BY 4.0. GTZAN itself is not
included and remains under its own terms.

---

## Citation

See `CITATION.cff`. The manuscript is under review; this file will be updated
with the final reference on acceptance.
