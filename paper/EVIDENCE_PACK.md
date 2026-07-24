# Evidence pack

Every number intended for the manuscript, traced to its source file and
independently re-verified.

**Status of this document.** Sections marked ✅ VERIFIED were recomputed from the
underlying per-item arrays and agree with the stored value. Sections marked
⏳ PENDING A100 could not be checked on the Mac clone because the source file is
not in this repository. Nothing in this document is reconstructed from memory: if
a number has no traceable source it is listed under UNTRACEABLE rather than
given a value.

Verification scripts:

    python scripts/60_verify_evidence.py      # stored vs recomputed, all sections
    python scripts/62_crosscheck_claims.py    # prose/figure claims vs results files
    python scripts/61_collect_descriptors.py  # A100 only: raw-data descriptors

Run counts as of the last Mac run: **117 numbers traced, 97 independently
recomputed, 0 stored-vs-recomputed mismatches, 11 cross-file findings.**

---

## 1. Screen — primary metric (`mean_ratio_pairs`)

Headline configuration: `filt0`, `NMIN=200`, `d=10`, `seed=0`.

### 1.1 Metric definition and why `pairs` is primary

Both quantities come from `causalbench/scripts/03_screen.py`:

| quantity | numerator | denominator |
|---|---|---|
| `mean_ratio_vs_ctrl` | `median(‖mean(Zа) − mean(Zr)‖)` where `Zr` is drawn from `ref_pool` | `median(‖mean(Za) − mean(Zb)‖)` (split-half of the same env) |
| `mean_ratio_pairs` | `median(‖mean(Za) − mean(Zb)‖)` across **two different environments** | same split-half denominator |

**Confirmed at `03_screen.py:63`:** in the step-0 branch `ref_pool = ctrl_rows[:0]`
— an empty array. The subsequent draw therefore falls back to the pseudo-environment's
own rows (`rng.choice(ref_pool if len(ref_pool) else rows, ...)`), so under step-0
`mean_ratio_vs_ctrl` compares an environment against itself and returns ≈0.67–0.72,
which is a construction artifact, not a calibrated null. Only `mean_ratio_pairs`
is calibrated to ≈1.0 by step-0. **`mean_ratio_pairs` is the primary metric.**

### 1.2 Norman 2019 ✅ VERIFIED

Source: `causalbench/results/screen/norman.json`

| quantity | value | JSON key path |
|---|---:|---|
| **`mean_ratio_pairs` (PRIMARY)** | **3.7147385914720377** | `$.screen[nmin=200,d=10].mean_ratio_pairs` |
| `mean_ratio_vs_ctrl` (secondary) | 3.1895884263497645 | `$.screen[nmin=200,d=10].mean_ratio_vs_ctrl` |
| step-0 gate `mean_ratio_pairs` | 1.0070856615811838 | `$.step0_gate[nmin=200,d=10].mean_ratio_pairs` |
| step-0 `mean_ratio_vs_ctrl` (degenerate) | 0.7216995131517371 | `$.step0_gate[nmin=200,d=10].mean_ratio_vs_ctrl` |
| environments surviving NMIN | 101 | `$.screen[nmin=200,d=10].n_envs` |
| cells per half (`n_fit`) | 100 | `$.screen[nmin=200,d=10].n_fit` |

Step-0 gate across all four Norman configurations (`n=200/100 × d=5/10`),
`mean_ratio_pairs`: **0.996, 1.007, 1.020, 1.029** — all inside [0.9, 1.1].
`summary.step0_pass = true` reproduces. ✅

> ⚠️ **The README and `norman_verdict.txt` both quote 3.19, which is the
> SECONDARY metric.** Under the primary metric the Norman value is **3.71**.
> See DISCREPANCY D1.

### 1.3 CausalBench K562 / RPE1 ⏳ PENDING A100

Files `causalbench/results/screen/{k562,rpe1}_filt0_n200_d10.json` and their
`_STEP0` counterparts exist in the A100 working tree but were **never committed**
to this repository, so they cannot be verified here.

Values observed during an earlier A100 session (recorded for comparison only,
**not verified**):

| dataset | `mean_ratio_pairs` | `mean_ratio_vs_ctrl` | `coef_ratio_vs_ctrl` | `n_envs` |
|---|---:|---:|---:|---:|
| K562 | 4.270 | 3.077 | 1.179 | 385 |
| RPE1 | 2.059 | 1.671 | 1.069 | 146 |

Step-0 gate (`mean_ratio_pairs`): K562 **1.018**, RPE1 **0.926**;
`mean_ratio_vs_ctrl` under step-0: K562 0.672, RPE1 0.678.

`n_envs` for both is independently corroborated by the split files
(`results/splits/k562_zeroshot_split.json` `$.n_usable = 385`;
`rpe1_zeroshot_split.json` `$.n_usable = 146`) ✅.

### 1.4 HCP — different construction ⏳ PENDING A100

**HCP's ratio is NOT produced by `03_screen.py`.** It comes from
`hcp/scripts/mean_shift_v2.py`, a separate fMRI pipeline. What can be established
from the results files alone:

- The HCP JSONs expose `mean_ratio`, `within_median`, `between_median`.
- They contain **no `mean_ratio_pairs` key and no step-0 gate entry.**
- `mean_ratio` is exactly `between_median / within_median` (verified to 1e-9
  where the files were available).

That is structurally the **`vs_ctrl` form**, not the `pairs` form.

**Comparability verdict: HCP does not currently have a value of the paper's
primary metric.** Placing HCP on the same axis as K562/RPE1/Norman compares a
`vs_ctrl`-shaped quantity against a `pairs`-shaped one. See DISCREPANCY D5. The
line-by-line source comparison must be completed on the A100 (`scripts/61`
prints the full script).

Values recorded from an earlier A100 session (**not verified**):

| encoding | d | `mean_ratio` | `within_median` | `between_median` |
|---|---:|---:|---:|---:|
| LR | 10 | 0.7483833300732464 | 4.884750555542272 | 3.6556658873346146 |
| LR | 20 | 0.7603583245858845 | 5.403239061317536 | 4.108397800001169 |
| LR | 30 | 0.7628674458454061 | 5.718036885151888 | 4.362104193826406 |
| RL | 10 | 0.7949708307350283 | 5.3041699350649685 | 4.216660379639154 |
| RL | 20 | 0.7846324154240907 | 5.92861169044841 | 4.651780910788822 |
| RL | 30 | 0.7922280954383233 | 6.18595331953082 | 4.900686016803068 |

Note every value is **below 1.0**, not equal to 1.0.

---

## 2. Screen — sensitivity to configuration ⏳ PENDING A100

Requires the 16 files `causalbench/results/screen/{k562,rpe1}_filt{0,1}_n{100,200}_d{5,10}.json`,
none of which are committed. `scripts/60_verify_evidence.py` §2 emits the full
table once they are present; it also reports the min/max spread per dataset so
the stability of the headline number is quantified rather than asserted.

Only two data points are currently on record (from the A100 session, unverified):
K562 `filt0/n200/d10` pairs = 4.270; RPE1 `filt0/n200/d10` pairs = 2.059.
**A range for K562 cannot be stated from one configuration.**

---

## 3. Effective dimensionality

Definitions from `causalbench/scripts/04_spectrum.py`:

- `s_sig = svd(M) / sqrt(k)` where `M` is the (k × d) stack of per-environment
  mean shifts; `s_noi` likewise for a control-derived null `N` of matched shape.
- `n_dims_above_2x = count(s_sig[i] / s_noi[i] > 2.0)` — a **count of components**.
- `participation_ratio = (Σ s_sig)² / Σ s_sig²` — an **effective rank of the
  signal spectrum alone**, computed without reference to the null.

### 3.1 Norman 2019 ✅ VERIFIED

Source: `causalbench/results/spectrum/norman.json`, `nmin=200`, `seed=0`.
All five derived quantities per d were recomputed from `sing_signal` /
`sing_noise` and agree exactly.

| d | `n_dims_above_2x` | `n_dims_above_noise` | `participation_ratio` | `s2_over_s1` | `sing_ratio[0]` |
|---:|---:|---:|---:|---:|---:|
| 10 | 10 / 10 | 10 / 10 | 4.446065832169267 | 0.6582844209194609 | 11.945008095462393 |
| 20 | 20 / 20 | 20 / 20 | 7.347425728612681 | 0.6679978502032126 | 11.948279299930848 |
| 50 | **39 / 50** | 50 / 50 | **13.226402291236367** | 0.6833521368943763 | 11.441271378681018 |

`n_envs = 101`, `n_null = 101` at every d.

### 3.2 Why 39 and 13.2 are not in conflict

They measure different things and are not expected to match:

- **39/50** counts how many components exceed twice the null. It is a threshold
  count over a *ratio* of two spectra, so it grows as more of the spectrum's tail
  clears a fixed bar. At d=50 Norman still has 39 components above 2× noise
  because its null spectrum decays fast.
- **13.23** is the participation ratio of the *signal spectrum alone*. It is
  dominated by the leading components and is insensitive to a long shallow tail.
  For a spectrum with one large and many small values it stays low.

The two therefore diverge exactly when the signal spectrum has a long tail that
sits above the null but carries little energy. That is a property of the
construction, readable from the code — **no anomaly, and no discrepancy.**
Whether the paper reports the count or the effective rank is a presentation
choice; they must not be described as the same quantity.

### 3.3 CausalBench and HCP ⏳ PENDING A100

`causalbench/results/spectrum/{k562,rpe1}_filt0_n200_d{10,20,50}.json` are not
committed. Values from an earlier A100 session (**not verified**):

| dataset | d | `dims_above_2x` | `participation_ratio` | `s2_over_s1` |
|---|---:|---:|---:|---:|
| K562 | 10 | 9 / 10 | 5.05 | 0.620 |
| K562 | 20 | 14 / 20 | 7.28 | 0.634 |
| K562 | 50 | 15 / 50 | 11.93 | 0.649 |
| RPE1 | 10 | 6 / 10 | 3.57 | 0.239 |
| RPE1 | 20 | 9 / 20 | 5.61 | 0.247 |
| RPE1 | 50 | 9 / 50 | 11.04 | 0.250 |

HCP (`hcp/results/mean_shift_{LR,RL}.json`, `max_possible_dims = 6`, capped by
the 7-task design): LR `dims_above_2x` = 0 at d=10, 20, 30; RL = 1, 0, 0.
Participation ratio ≈ 4.5–5.0. **No d yields more than one component above 2×
noise.** Note the HCP spectrum has at most 6 entries regardless of d, so
"dims above 2× / d" is not on the same denominator as the CausalBench datasets.

---

## 4. Intervention subspace vs control PCA ✅ VERIFIED

Source: `results/cf_estimator/k562.json`, `nmin=200`.

| quantity | value | key |
|---|---:|---|
| environments | 385 | `$.n_envs` |
| stable environments | 298 | `$.n_stable` |
| **control-cell gate** `gate_cos_median` | **−0.006828483048328142** | `$.gate_cos_median` |
| `gate_cos_mean` | −0.009455955587287365 | `$.gate_cos_mean` |
| split-half `stab_cos_median` | 0.6842780722525882 | `$.stab_cos_median` |
| `stab_cos_q10` | 0.06632494098453602 | `$.stab_cos_q10` |
| `stab_cos_q90` | 0.8803670910740762 | `$.stab_cos_q90` |
| `frac_cos_gt_0p3` | 0.7168831168831169 | `$.frac_cos_gt_0p3` |
| `rank1_median` | 0.025447037879799767 | `$.rank1_median` |
| `shift_norm_median` | 2.5391828429016745 | `$.shift_norm_median` |

Held-out reconstruction R², `$.basis.<d>`. `gain` recomputed as
`shift_basis_r2 − control_pca_r2` for every d; all five agree to 1e-9 ✅.

| d | `shift_basis_r2` | `control_pca_r2` | `gain` (stored = recomputed) |
|---:|---:|---:|---:|
| 5 | 0.7029684174159108 | 0.4846019163349933 | 0.21836650108091749 |
| 10 | 0.7642257409720432 | 0.592387419064074 | 0.17183832190796922 |
| 15 | 0.78473985776114 | 0.6193275985937475 | 0.16541225916739244 |
| 20 | 0.7992939864369504 | 0.630621176988112 | 0.16867280944883833 |
| 30 | 0.8165562129322685 | 0.6492430406894609 | 0.16731317224280762 |

The control-cell gate returning ≈0 is the intended known-answer result: two halves
of the control pool share no common shift direction.

---

## 5. Zero-shot baselines, canonical split ✅ VERIFIED

Source: `results/zeroshot_canonical/k562.json`.
Split provenance confirmed:

| quantity | value | verified against |
|---|---:|---|
| `split_file` | `k562_zeroshot_split.json` | `$.split_file` |
| `n_train` | **308** | matches `results/splits/k562_zeroshot_split.json $.n_train` ✅ and `len($.train_perturbations)` ✅ |
| `n_heldout` | **77** | matches split `$.n_heldout` ✅ and `len($.heldout_perturbations)` ✅ |
| `n_usable` | 385 | equals `n_train + n_heldout` ✅ and equals `count(cells_per_perturbation ≥ 200)` ✅ |
| `d_latent` | 15 | `$.d_latent` |
| split `seed` | 0 | split `$.seed` |
| split `nmin` | 200 | split `$.nmin` |

**All 24 baseline statistics were recomputed from the 462-row
`$.per_perturbation` array (6 methods × 77 held-out genes) and every one agrees
exactly.** ✅

| method | median R² | mean R² | median cos | `frac_beats_gmean` |
|---|---:|---:|---:|---:|
| `zero` | 0.0 | 0.0 | 0.0 | 0.36363636363636365 |
| `global_mean` | **0.12902286583297684** | −0.04360618578319341 | 0.3777093107368002 | 0.0 |
| `corr_prop` | −0.18752402582055216 | −0.28580222935470173 | 0.13457430570772097 | 0.24675324675324675 |
| `nn_corr` | −0.24278568997495253 | −1.3799998480864606 | 0.22765124994501498 | 0.2727272727272727 |
| `ridge_basis` | **0.22558014602202647** | 0.033191204082618844 | 0.5010799951288413 | 0.7012987012987013 |
| `CEILING` | **0.5274920724871424** | 0.3168453803721219 | 0.7690887650072865 | 0.8311688311688312 |

`_headroom = 0.3984692066541655`, recomputed as `CEILING − global_mean` ✅.
Ridge capture = `(0.22558 − 0.12902) / 0.39847` = **24.232055721397924 %** ✅.

> ⚠️ `frac_beats_gmean` here is a **per-gene** comparison (each gene's R² against
> that same gene's `global_mean` R²), not a comparison against the constant
> 0.129. The model files use the constant. See DISCREPANCY D3.

---

## 6. The model result ✅ VERIFIED (except epoch-10)

Sources: `results/model/zeroshot_{shift,random}.json`,
`results/model/zeroshot_nodag_{shift,random}.json`. Each has a 77-entry
`$.per_gene` array; `final_median_r2`, `final_mean_r2`, `frac_beats_gmean` and
`frac_beats_ridge` were all recomputed from it and agree exactly. ✅

| arm | best median R² | final median R² | final mean R² | `frac_beats_gmean` | `frac_beats_ridge` | per-gene R² range |
|---|---:|---:|---:|---:|---:|---|
| full / shift | 0.009694068092076957 | 0.012570545400028665 | −0.0518394092777882 | 0.0 | 0.0 | [−0.7075, 0.1128] |
| full / random | −0.005696563328918991 | −0.005466684895471641 | −0.006320124530840101 | 0.0 | 0.0 | [−0.0427, 0.0221] |
| no-DAG / shift | −0.04224525495841136 | −0.04793495530415348 | −0.10737029539395729 | 0.0 | 0.0 | [−0.7025, 0.0520] |
| no-DAG / random | 0.013044818906979505 | 0.013023438700739853 | 0.013743959567562424 | 0.0 | 0.0 | [−0.0304, 0.0704] |

Across all four arms: **0 of 77 held-out genes exceed 0.129, and 0 of 77 exceed
0.226.** Recomputed directly from the per-gene arrays ✅.

`results/model/SUMMARY.json` was cross-checked field by field against the four
source files and the baselines file: **28 fields, all agree.** ✅

### 6.1 DAG condition number ✅ VERIFIED

Source: `results/model/dag_stability.csv`, 223 rows, epochs 0–14,
shift-init, `log_every=50`.

| quantity | value |
|---|---:|
| `cond2` min | 1.000471 |
| `cond2` max | **1.812932** |
| `cond2` final | 1.569744 |
| `inv_norm_2` min | 1.000219 |
| `inv_norm_2` max | 1.357119 |
| `G_frobenius` min | 1.20544 |
| `G_frobenius` max | 1.587544 |

Zero NaNs in all three columns. The condition number never approaches 10³.

### 6.2 Epoch-10 value ⏳ PENDING A100

`results/model/trajectories/*.json` were generated on the A100 but **never
committed**, and `logs/` is gitignored. The epoch-10 value therefore **cannot be
traced from this repository for any arm.**

Values seen in an A100 session (**not verified, and note they disagree with an
earlier record — see DISCREPANCY D7**):

| arm | peak | at epoch | final | at epoch |
|---|---:|---:|---:|---:|
| full / shift | +0.0097 | 20 | −0.0055 | 99 |
| full / random | −0.0057 | 0 | −0.0938 | 99 |
| no-DAG / shift | −0.0422 | 20 | −0.0664 | 99 |
| no-DAG / random | +0.0130 | 0 | −0.0804 | 99 |

An earlier handoff recorded a shift-init epoch-10 value of **−1.09**, and a
later note recorded **−0.997** for the no-DAG shift-init arm. Neither is
traceable to a committed file. **Do not use either number until the trajectory
JSONs are committed.**

---

## 7. Dataset descriptors

### 7.1 From the split files ✅ VERIFIED

| quantity | K562 | RPE1 | source |
|---|---:|---:|---|
| genes recorded in `cells_per_perturbation` | 1158 | 651 | `results/splits/<ds>_zeroshot_split.json` |
| environments clearing NMIN=200 (`n_usable`) | **385** | **146** | `$.n_usable`, recomputed from the cell counts ✅ |
| `n_train` | 308 | 117 | `$.n_train` |
| `n_heldout` | 77 | 29 | `$.n_heldout` |
| cells/env median (all recorded genes) | 165.0 | 146.0 | recomputed from `$.cells_per_perturbation` |
| cells/env min | 101.0 | 101.0 | recomputed |
| cells/env max | 1996.0 | 3580.0 | recomputed |
| seed | 0 | 0 | `$.seed` |

Note the median above is over **all** recorded genes including those below NMIN;
the median over the 385/146 that clear NMIN is a different number and is not
stored anywhere. `scripts/61` computes it from the raw data.

### 7.2 Cells, features, control cells, batches ⏳ PENDING A100

Not present in any results file. `scripts/61_collect_descriptors.py` reads
`dataset_{k562,rpe1}.npz`, `Norman2019_raw.h5ad` and `causalbench_k562.h5ad`
and emits: total cells, features, excluded cells, control cells, unique
perturbations, cells-per-perturbation quantiles, value range, fraction exactly
zero, whether values are integers, and how many targets appear among the feature
columns.

Preprocessing facts already established and traceable:

- **Norman 2019** is CPA-preprocessed and log-normalised despite the `_raw`
  filename — values span 0–8.5 and are non-integer. 5000 highly-variable genes.
  0 of 105 single-perturbation targets appear among the 5000 feature columns
  (verified during the Norman screen run; recorded in
  `causalbench/results/screen/norman.json` as
  `target_column_drop_is_noop: true`) ✅.
- **CausalBench** is CRISPR **interference**; **Norman** is CRISPR
  **activation**. Different perturbation direction.
- Batch identity (`gem_group`) exists only in the raw `.h5ad`, not in the `.npz`.

---

## 8. Third-party provenance

### 8.1 Commit hashes ✅ VERIFIED

| repository | commit | source |
|---|---|---|
| discrepancy_vae (Zhang et al., NeurIPS 2023) | `4451fdbc9d0aa3a1dee4e7d1b743a434e98fa58a` | `vendor_analysis/DVAE_COMMIT.txt` |
| causalbench_repo (Chevalley et al., NeurIPS 2023) | `1a2143cffdc85f835b41ce8d52034be1bf903e71` | `vendor_analysis/CAUSALBENCH_COMMIT.txt` |

`COMMITS.txt` at the repository root mirrors both and agrees ✅.

### 8.2 Our modifications to their tree ✅ VERIFIED

`vendor_analysis/my_changes.patch`: **4 hunks, touching only `src/dataset.py`.**
Both hunks are dataset-path string replacements (`/home/jzhang/...` →
`/workspace/external/...`). No change to model or loss code.

### 8.3 Our training used their `loss_function` unmodified ✅ VERIFIED

`model/cb_train.py`:
- `from train import loss_function` → **True**
- defines its own `loss_function` → **False**

Same two checks pass for `scripts/27_no_causal_ablation.py`.

### 8.4 Hyperparameters actually used ✅ VERIFIED (our defaults)

From `model/cb_train.py` argparse defaults:

| flag | default |
|---|---|
| `--zdim` | 15 |
| `--epochs` | 100 |
| `--batch` | 128 |
| `--lr` | 1e-3 |
| `--mxAlpha` | 10.0 |
| `--mxBeta` | 2.0 |
| `--mxTemp` | 5.0 |
| `--MMD_sigma` | 1000.0 |
| `--kernel_num` | 10 |
| `--lmbda` | 1e-3 |
| `--init` | `shift` |
| `--device` | `cuda:0` |

⏳ **PENDING A100:** the corresponding defaults in *their* `src/train.py`, to
confirm ours match theirs where we did not deliberately change them.
`scripts/61` prints their argparse defaults and the `loss_function` signature.

---

# DISCREPANCIES AND OPEN ITEMS

11 findings. Machine-readable copy: `paper/crosscheck_findings.json`.

### D1 — Norman headline uses the SECONDARY metric (HIGH)

| | |
|---|---|
| Claimed | **3.19** |
| Claim sources | `README.md` headline table; `causalbench/results/screen/norman_verdict.txt` |
| Actual | `mean_ratio_pairs` = **3.7147385914720377** (primary); `mean_ratio_vs_ctrl` = 3.1895884263497645 (secondary) |
| Authority | `causalbench/results/screen/norman.json $.screen[nmin=200,d=10]` |

Both the README table and the saved verdict sentence quote `vs_ctrl`. If
`mean_ratio_pairs` is the primary metric, both must be restated as 3.71 —
**or** the paper must state explicitly that it reports `vs_ctrl`.

### D2 — Figure 2 mixes two metrics on one axis (HIGH)

`scripts/50_figures.py::fig2_mean_shift_ratio` plots bar heights from
`mean_ratio_vs_ctrl` (6 references in the file) while its null line and gate
band come from `mean_ratio_pairs` (4 references). Already known; **still
unfixed in the script.** Every remaining `mean_ratio_vs_ctrl` use in that file
must be reviewed: it also drives the FIG 2 printout and the reference-comparison
block.

### D3 — `frac_beats_gmean` denotes two different quantities (HIGH)

| file | definition |
|---|---|
| `scripts/25_zeroshot_baselines.py` | per-gene: `r2 > gm_by_g[gene]`, i.e. that gene's own `global_mean` R² |
| `model/cb_train.py` | constant: `np.mean(zs > 0.129)` |

Both are internally correct and both recompute exactly ✅ — but they are not the
same statistic. The baselines' `ridge_basis` value of **0.701** and the models'
**0.000** are answers to different questions and must not be tabulated in one
column.

### D4 — The model and the baselines are scored against different targets (HIGH)

| file | target of the R² |
|---|---|
| `scripts/25_zeroshot_baselines.py` | **half B** of the held-out perturbation's cells (`A, B = shift(cells[:h]), shift(cells[h:])`; every method scored against `B`) |
| `model/cb_train.py::zeroshot_eval` | the **full-data** mean shift (`truth = X[iv==g].mean(0) - ctrl_mu`) |

The R² *formula* is identical in both files ✅ (`1 − RSS / Σtrue²`), but the
targets differ. The model's target is the less noisy of the two, so model R² and
baseline R² are not on a common scale — yet `cb_train.py` hard-codes the
constants **0.129** and **0.226**, which were produced by the half-B evaluation,
as its comparison thresholds. This is the most consequential item in the
document and it affects every "model vs ridge" statement.

### D5 — HCP has no value of the primary metric (HIGH)

HCP's ratio comes from `hcp/scripts/mean_shift_v2.py`, not `03_screen.py`. Its
JSONs expose `mean_ratio`, `within_median`, `between_median` and **no
`mean_ratio_pairs`, no step-0 gate**. `mean_ratio == between_median /
within_median`, which is structurally the `vs_ctrl` form. Placing HCP beside
K562/RPE1/Norman on a `mean_ratio_pairs` axis compares different quantities.
Full line-by-line source comparison is pending on the A100.

### D6 — README HCP value 1.00 is not traceable (MEDIUM)

`README.md` states HCP mean-shift ratio = **1.00**. Every recorded HCP
`mean_ratio` is **below** 1.0 (LR 0.748 / 0.760 / 0.763; RL 0.795 / 0.785 /
0.792 at d = 10 / 20 / 30). No results file contains 1.00. Either the README is
rounding a sub-null value up to the null, or it is quoting a different quantity.

### D7 — Epoch-10 crash value: three different numbers on record (MEDIUM)

- Handoff document: shift-init epoch-10 = **−1.09**
- Later session note: no-DAG shift-init epoch-10 = **−0.997**
- A100 trajectory parse: full/shift **peak** +0.0097 at epoch 20; no epoch-10
  value quoted

None is traceable to a committed file. `results/model/trajectories/` is not in
the repository and `logs/` is gitignored. **Unusable until committed.**

### D8 — README CEILING rounds the wrong way (LOW)

`README.md` gives CEILING as **+0.528**; the stored value is
**0.5274920724871424**, which rounds to 0.527. The same 0.528 appears in
`norman_verdict.txt` and in the FIG 6 baseline line. Cosmetic, but it is
propagated to three places.

### D9 — README K562 range "2.0 – 5.6" is not traceable (MEDIUM)

No committed file supports it. Only one K562 configuration is on record
(`filt0/n200/d10`: pairs 4.270, vs_ctrl 3.077). The range presumably spans the
16-configuration sweep, but neither the configurations nor the metric are stated,
and the files are uncommitted.

### D10 — README "CausalBench effective dim ~15" is ambiguous and untraceable (MEDIUM)

Not in any committed file. Additionally ambiguous between `dims_above_2x`
(K562 = 15 at d=50, from the unverified A100 record) and `participation_ratio`
(K562 = 11.93 at d=50). These are different quantities that happen to be close
for K562 — but for Norman they are 39 and 13.2, so the ambiguity is not benign.

### D11 — README "HCP effective dim 3 (group only)" is not traceable (MEDIUM)

The qualifier "group only" appears in no results file key. The HCP JSONs report
`n_dims_above_2x` of 0 (LR, all d) and 1 then 0 (RL). The value 3 matches
`n_dims_above_noise` at d=10 and d=20 for LR — i.e. the **1×** threshold, not
the 2× threshold used for every other dataset.

---

## Where `mean_ratio_vs_ctrl` is still in use

| location | use |
|---|---|
| `scripts/50_figures.py` fig2 | bar heights, printout, reference table |
| `scripts/50_figures.py` fig4 | `coef_ratio_vs_ctrl` for K562 (separate metric, but same `vs_ctrl` construction) |
| `README.md` | Norman 3.19, and the K562/HCP entries |
| `causalbench/results/screen/norman_verdict.txt` | the saved verdict sentence |
| `causalbench/results/screen/norman.json` | `summary.primary_mean_ratio_vs_ctrl` is labelled *primary* |

The last item is worth noting: the key **name** inside `norman.json` asserts that
`vs_ctrl` is primary. That label contradicts the current metric decision.

---

## What is still needed to close this document

1. Commit `causalbench/results/screen/{k562,rpe1}_*.json` (16 files) and
   `causalbench/results/spectrum/{k562,rpe1}_*.json` (6 files) — closes §1.3,
   §2, §3.3, D9, D10.
2. Commit `results/model/trajectories/*.json` — closes §6.2 and D7.
3. Run `scripts/61_collect_descriptors.py` — closes §7.2, §8.4, D5, D6, D11.
4. Re-run `scripts/60_verify_evidence.py` and `scripts/62_crosscheck_claims.py`
   on the A100 with everything present, and append the totals here.
