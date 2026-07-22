# meridian-causalbench

Meridian Objective 2: benchmarked-negative test of a causal-representation-learning
(CRL) model for zero-shot intervention prediction on CausalBench K562
(Replogle 2022 Perturb-seq CRISPRi).

Model: `CMVAE_CB`, a subclass of the discrepancy-VAE (Zhang et al., NeurIPS 2023).

## Result

The model beats neither the linear-ridge baseline (median R² 0.226) nor the
mean baseline (0.129) on any held-out perturbation. The signal exists: the
split-half ceiling is 0.528. Removing the causal DAG layer changes nothing,
matching the authors' own Table 3 (causal layer contributes ~0) on our task.
The DAG condition number stays in [1.0, 1.8] throughout training
(`results/model/dag_stability.csv`), so numerical instability is not the cause.

Baselines (median R² vs no-effect null on 77 held-out K562 perturbations):

| method       | median R² |
|--------------|----------:|
| zero         |    +0.000 |
| global_mean  |    +0.129 |
| corr_prop    |    −0.188 |
| nn_corr      |    −0.243 |
| ridge_basis  |    +0.226 |
| CEILING      |    +0.528 |

Headroom = 0.399 R² units. Ridge captures 24.2% of it.

Model runs (best median R² over training):

| variant                       | best median R² | beats gmean | beats ridge |
|-------------------------------|---------------:|------------:|------------:|
| full model, shift-init        |        +0.0097 |          0% |          0% |
| full model, random-init       |        −0.0057 |          0% |          0% |
| no-DAG ablation, shift-init   |        −0.0422 |          0% |          0% |
| no-DAG ablation, random-init  |        +0.0130 |          0% |          0% |

Consolidated table: `results/model/SUMMARY.json`.

**Verdict.** Benchmarked negative: CRL model beats neither ridge (0.226) nor
mean (0.129) on any held-out perturbation; causal layer contributes nothing
(no-DAG identical); signal exists (ceiling 0.528).

## Repository layout

```
model/
  cb_data.py     CausalBench loader, perturbation-level zero-shot split
  cb_init.py     shift-basis + nonlinear-decoder fit
  cb_model.py    CMVAE_CB subclass, decoder-probe zero-shot readout
  cb_train.py    training loop; --init {shift,random}
scripts/
  25_zeroshot_baselines.py     reference baselines (0.129, 0.226)
  27_no_causal_ablation.py     no-DAG ablation training
  28_export_nodag_results.py   export ablation results to canonical JSON
  29_dag_condnum_log.py        log DAG condition number during training
  30_results_summary.py        write results/model/SUMMARY.json
results/
  splits/k562_zeroshot_split.json      canonical split (77 held-out of 385)
  model/zeroshot_{shift,random}.json   full-model results
  model/zeroshot_nodag_{shift,random}.json  no-DAG results
  model/dag_stability.csv              DAG condition-number log
  model/SUMMARY.json                   consolidated Table 1 source
  zeroshot_canonical/k562.json         baselines
vendor_analysis/
  DVAE_COMMIT.txt          upstream discrepancy_vae commit
  CAUSALBENCH_COMMIT.txt   upstream causalbench_repo commit
  my_changes.patch         path-only edits applied to discrepancy_vae
  scripts/                 vendor scripts used for repro
COMMITS.txt                upstream commit hashes (mirror of vendor_analysis)
```

## Reproducing

Data is A100-local and not in this repo (gitignored: `*.npz`, `*.h5ad`, `*.pt`,
`data/`).

Prerequisites on the A100:

- `causalbench/data/dataset_k562.npz` from the CausalBench repo
  (Replogle 2022 K562).
- `external/discrepancy_vae/datasets/causalbench_k562.h5ad` (AnnData for the
  loader).
- `external/discrepancy_vae/src/` (Zhang 2023's model, unmodified except path
  fixes; see `vendor_analysis/my_changes.patch`).
- Environment: `/workspace/venvs/dvae` (torch + discrepancy_vae dependencies).

Pipeline:

```bash
source /workspace/venvs/dvae/bin/activate

# 1. baselines
python scripts/25_zeroshot_baselines.py

# 2. full model, both inits
python model/cb_train.py --init shift  --epochs 100
python model/cb_train.py --init random --epochs 100

# 3. no-DAG ablation, both inits
python scripts/27_no_causal_ablation.py --init shift  --epochs 100
python scripts/27_no_causal_ablation.py --init random --epochs 100
python scripts/28_export_nodag_results.py

# 4. DAG condition-number diagnostic
python scripts/29_dag_condnum_log.py --epochs 15 --log_every 50

# 5. consolidated table
python scripts/30_results_summary.py
```

The canonical split at `results/splits/k562_zeroshot_split.json` is the single
source of truth; never regenerate.

## Upstream

- discrepancy_vae: Zhang et al., NeurIPS 2023.
  Commit hash in `vendor_analysis/DVAE_COMMIT.txt` and `COMMITS.txt`.
- CausalBench: Chevalley et al., NeurIPS 2023.
  Commit hash in `vendor_analysis/CAUSALBENCH_COMMIT.txt` and `COMMITS.txt`.

Our modifications to `discrepancy_vae/src/` are dataset-path fixes only; see
`vendor_analysis/my_changes.patch`.
