"""Pure descriptive profiler for the real datasets. A100. NO TEST, NO VERDICT.

WHAT THIS IS NOT. It does not import 80_ranktest_core, does not call the rank
test, and does not emit any assumption verdict, pass/fail, or recommendation.
It reports counts and spectra. Phase B remains NOT AUTHORISED and nothing here
changes that.

WHAT IT REPORTS, per dataset (per ARM for Frangieh, never pooled):

  1. Environment attrition. How many environments survive each
     NMIN in {50, 125, 500, 2000, 8000}, so the curve can be read against
     both measured power floors (hard 125, soft 8000).
  2. Cells per environment. min, q25, median, q75, max, plus the counts above
     125 and above 8000.
  3. Latent dimension from CONTROL CELLS ONLY, by three estimators that do not
     agree:
       participation ratio   (sum lam)^2 / sum lam^2
       effective rank        exp(-sum p log p),  p = lam / sum lam
       n components reaching 80 / 90 / 95% of variance
     All are reported side by side, with the spectrum's own truncation bound,
     because the disagreement between them is the point.
  4. fMRI in equivalent terms: subjects, conditions, timepoints.

PARSING RULES ARE INHERITED, NOT REDISCOVERED. Frangieh goes through
41_screen_frangieh.load_metadata / load_expression, which already encode the
row-2 SCP TYPE skip, MOI == 1 only, the TRAILING guide-index regex
^(.*)_\\d+$, the pooled NO_SITE_* / ONE_NON-GENE_SITE_* controls, and the
chunked float32 read. Replogle goes through 03_screen.load, Norman through
40_screen_norman.load_norman. fit_pca / project are imported from 03_screen.py
via importlib so any projection arithmetic is byte-identical to the screen.

ONE HONEST CAVEAT ON THE SPECTRUM. fit_pca computes an SVD but returns only
(mu, W) and discards the singular values, and all three latent-dimension
estimators need the full spectrum. The spectrum is therefore computed here by
an SVD using the IDENTICAL centering (mu = Xc.mean(0)), and the top-d
directions are asserted to match fit_pca's W up to sign, so the reported
spectrum provably belongs to the same decomposition the screen uses.

A second caveat, stated because it bounds every number in section 3: when the
control-cell count n is below the gene count p, the covariance has at most
n - 1 non-zero eigenvalues, so participation ratio and effective rank are
capped by n - 1 regardless of the true latent dimension. Control pools are
subsampled to --n-spec-cap for tractability, which lowers that cap further.
Both n and the resulting rank bound are reported next to every estimate, and
--n-spec-cap is swept over two values so the n-sensitivity is visible rather
than hidden.

Usage (A100, cb venv, one dataset at a time):
    nohup python -u causalbench/scripts/84_dataset_descriptives.py \\
        --dataset k562 > k562_desc.log 2>&1 &
    ... --dataset frangieh --hvg 5000
    ... --dataset hcp
"""
import argparse
import datetime
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CTRL_LABEL = "non-targeting"
NMIN_SET = (50, 125, 500, 2000, 8000)
POWER_FLOOR_HARD = 125          # measured, lfc/power_hard
POWER_FLOOR_SOFT = 8000         # measured, lfc/power_soft
SPEC_CAPS = (2000, 8000)        # spectrum subsample sizes, for n-sensitivity
VAR_TARGETS = (0.80, 0.90, 0.95)

HCP_TS = Path("/workspace/meridian-identifiability/hcp/ts")
HCP_TASKS = ["WM", "GAMBLING", "MOTOR", "LANGUAGE", "SOCIAL", "RELATIONAL",
             "EMOTION"]
HCP_ENCS = ["LR", "RL"]
HCP_NFRAMES = 176               # matches mean_shift_v2 / 70_hcp_ceiling


def _load_module(path, name):
    """Import a sibling script, neutralising 03_screen.py's module-scope
    os.makedirs("/workspace/...") so this file stays importable off the A100."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    real = os.makedirs
    os.makedirs = lambda *a, **k: None
    try:
        spec.loader.exec_module(mod)
    finally:
        os.makedirs = real
    return mod


RIO = _load_module(HERE / "84_results_io.py", "_results_io")
SCREEN = _load_module(HERE / "03_screen.py", "_screen03_desc")
fit_pca = SCREEN.fit_pca          # imported, never reimplemented
project = SCREEN.project


# --------------------------------------------------------------- descriptives
def env_attrition(counts):
    """How many environments clear each NMIN. Pure counting."""
    c = np.asarray(list(counts), dtype=np.int64)
    return {str(m): int((c >= m).sum()) for m in NMIN_SET}


def cell_distribution(counts):
    c = np.asarray(list(counts), dtype=np.int64)
    if c.size == 0:
        return dict(n_environments=0)
    q25, q50, q75 = (float(x) for x in np.percentile(c, [25, 50, 75]))
    return dict(
        n_environments=int(c.size),
        min=int(c.min()), q25=q25, median=q50, q75=q75, max=int(c.max()),
        total_cells=int(c.sum()),
        n_above_hard_floor_125=int((c >= POWER_FLOOR_HARD).sum()),
        n_above_soft_floor_8000=int((c >= POWER_FLOOR_SOFT).sum()),
        frac_above_hard_floor_125=float((c >= POWER_FLOOR_HARD).mean()),
        frac_above_soft_floor_8000=float((c >= POWER_FLOOR_SOFT).mean()),
    )


def spectrum_estimators(Xc, cap, rng, check_against_fit_pca=True):
    """Latent-dimension estimators from control cells only.

    Returns the three estimators plus the bound they cannot exceed. Nothing
    here is a verdict: the estimators disagree by construction and all are
    reported.
    """
    n_all, p = Xc.shape
    n_use = int(min(cap, n_all))
    idx = rng.choice(n_all, size=n_use, replace=False) if n_use < n_all \
        else np.arange(n_all)
    X = np.asarray(Xc[idx], dtype=np.float64)

    mu = X.mean(0)                              # identical centering to fit_pca
    Xcen = X - mu
    # economy SVD; singular values are what fit_pca discards
    sv = np.linalg.svd(Xcen, full_matrices=False, compute_uv=False)
    lam = (sv ** 2) / max(n_use - 1, 1)
    lam = lam[lam > 0]
    tot = float(lam.sum())
    if tot <= 0:
        return dict(n_control_used=n_use, n_genes=int(p), error="zero variance")

    pr = float(tot ** 2 / float((lam ** 2).sum()))
    pk = lam / tot
    eff_rank = float(np.exp(-float((pk * np.log(pk)).sum())))
    cum = np.cumsum(lam) / tot
    ncomp = {f"{int(t*100)}pct": int(np.searchsorted(cum, t) + 1)
             for t in VAR_TARGETS}

    out = dict(
        n_control_used=n_use, n_control_available=int(n_all), n_genes=int(p),
        rank_bound=int(min(n_use - 1, p)),
        participation_ratio=pr,
        effective_rank_exp_spectral_entropy=eff_rank,
        n_components_for_variance=ncomp,
        top_eigenvalue_share=float(lam[0] / tot),
        n_nonzero_eigenvalues=int(lam.size),
    )
    if check_against_fit_pca:
        # prove the spectrum belongs to the SAME decomposition the screen uses
        d_chk = int(min(10, p, n_use - 1))
        if d_chk >= 2:
            mu2, W = fit_pca(X, d_chk)
            _, _, Vt = np.linalg.svd(Xcen, full_matrices=False)
            agree = np.allclose(np.abs(np.sum(Vt[:d_chk].T * W, axis=0)),
                                np.ones(d_chk), atol=1e-6)
            out["matches_fit_pca_basis"] = bool(agree)
            out["fit_pca_mu_identical"] = bool(np.allclose(mu, mu2))
    return out


def latent_dim_block(Xc, rng):
    """Spectrum estimators at each cap, so n-sensitivity is visible."""
    return {str(cap): spectrum_estimators(Xc, cap, rng) for cap in SPEC_CAPS}


def profile_perturb(X, iv, label, rng):
    """One perturbation block: attrition, distribution, control spectrum."""
    iv = np.asarray(iv, dtype=object)
    ctrl_rows = np.where(iv == CTRL_LABEL)[0]
    targets, counts = np.unique(iv[iv != CTRL_LABEL], return_counts=True)
    print(f"[{label}] {X.shape[0]} cells x {X.shape[1]} features; "
          f"{len(ctrl_rows)} control cells; {len(targets)} targets", flush=True)
    block = dict(
        label=label,
        n_cells=int(X.shape[0]), n_features=int(X.shape[1]),
        n_control_cells=int(len(ctrl_rows)),
        n_environments_total=int(len(targets)),
        environment_attrition=env_attrition(counts),
        cells_per_environment=cell_distribution(counts),
        latent_dimension_from_controls=(
            latent_dim_block(X[ctrl_rows], rng) if len(ctrl_rows) >= 50
            else {"error": f"only {len(ctrl_rows)} control cells"}),
    )
    a = block["environment_attrition"]
    print(f"[{label}] surviving: " +
          "  ".join(f"n>={m}:{a[str(m)]}" for m in NMIN_SET), flush=True)
    return block


# -------------------------------------------------------------------- loaders
def do_replogle(ds, rng, args):
    X, iv, vn = SCREEN.load(ds, False)
    return [profile_perturb(np.asarray(X), iv, ds, rng)]


def do_norman(rng, args):
    nor = _load_module(HERE / "40_screen_norman.py", "_norman40")
    got = nor.load_norman()
    X, iv = np.asarray(got[0]), np.asarray(got[1], dtype=object)
    return [profile_perturb(X, iv, "norman", rng)]


def do_frangieh(rng, args):
    """PER ARM, never pooled: each arm has its own control pool."""
    fr = _load_module(HERE / "41_screen_frangieh.py", "_frangieh41")
    md, meta = fr.load_metadata()
    X, vn, cell_order = fr.load_expression(fr.EXPR_CSV,
                                           list(md["NAME"].astype(str)),
                                           hvg=args.hvg)
    md = md.set_index(md["NAME"].astype(str)).reindex(cell_order)
    ok = md["iv"].notna().to_numpy()
    X, md = X[ok], md.loc[ok]
    iv = md["iv"].to_numpy().astype(object)
    arms = md["condition"].to_numpy().astype(str)
    blocks = []
    for arm in sorted(set(arms)):
        sel = arms == arm
        blocks.append(profile_perturb(np.ascontiguousarray(X[sel]), iv[sel],
                                      f"frangieh:{arm}", rng))
    blocks.append(dict(label="frangieh:loader_metadata", loader_meta=meta,
                       note="per-arm only; the pooled profile is deliberately "
                            "not computed"))
    return blocks


def do_hcp(rng, args):
    """fMRI in the equivalent terms: subjects, conditions, timepoints.

    Environment == task condition. 'Cells per environment' == timepoints
    pooled over subjects and encodings. There is NO control condition in this
    set, so the spectrum is taken over ALL pooled runs and is labelled as such
    rather than being passed off as a control-only spectrum.
    """
    if not HCP_TS.is_dir():
        sys.exit(f"[fatal] HCP_TS not a directory: {HCP_TS}")
    subs = sorted({p.name.split("_")[0] for p in HCP_TS.glob("*.npy")})

    def load_run(s, t, e):
        p = HCP_TS / f"{s}_{t}_{e}.npy"
        if not p.exists():
            return None
        x = np.load(p).astype(np.float64)
        return x[:HCP_NFRAMES] if x.shape[0] >= HCP_NFRAMES else None

    complete = [s for s in subs
                if all(load_run(s, t, e) is not None
                       for t in HCP_TASKS for e in HCP_ENCS)]
    print(f"[hcp] {len(subs)} subjects present, {len(complete)} with all "
          f"{len(HCP_TASKS)}x{len(HCP_ENCS)} runs", flush=True)

    per_task, pooled = {}, []
    for t in HCP_TASKS:
        n_tp = 0
        for s in complete:
            for e in HCP_ENCS:
                x = load_run(s, t, e)
                if x is not None:
                    n_tp += x.shape[0]
                    pooled.append(x)
        per_task[t] = n_tp
    Xall = np.vstack(pooled) if pooled else np.zeros((0, 0))
    counts = list(per_task.values())
    block = dict(
        label="hcp",
        modality="fMRI",
        n_subjects_present=len(subs), n_subjects_complete=len(complete),
        n_conditions=len(HCP_TASKS), conditions=HCP_TASKS,
        encodings=HCP_ENCS, frames_per_run=HCP_NFRAMES,
        n_regions=int(Xall.shape[1]) if Xall.size else None,
        timepoints_per_condition=per_task,
        timepoints_per_subject_run=HCP_NFRAMES,
        total_timepoints=int(Xall.shape[0]) if Xall.size else 0,
        environment_attrition=env_attrition(counts),
        cells_per_environment=cell_distribution(counts),
        latent_dimension_from_pooled_runs=(
            latent_dim_block(Xall, rng) if Xall.shape[0] >= 50 else {}),
        spectrum_caveat=("HCP has no control condition, so this spectrum is "
                         "over ALL pooled runs, not a control pool. It is not "
                         "comparable to the Perturb-seq control spectra."),
        scaling_caveat=("raw concatenated frames; the subject_pooled z-scoring "
                        "documented in PATHS_hcp.md is NOT applied here"),
    )
    a = block["environment_attrition"]
    print(f"[hcp] surviving conditions: " +
          "  ".join(f"n>={m}:{a[str(m)]}" for m in NMIN_SET), flush=True)
    return [block]


LOADERS = {"k562": lambda r, a: do_replogle("k562", r, a),
           "rpe1": lambda r, a: do_replogle("rpe1", r, a),
           "norman": do_norman, "frangieh": do_frangieh, "hcp": do_hcp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--hvg", type=int, default=None, help="Frangieh only")
    ap.add_argument("--n-spec-cap", type=int, default=None,
                    help="override the spectrum subsample caps (default 2000,8000)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.n_spec_cap:
        global SPEC_CAPS
        SPEC_CAPS = (a.n_spec_cap,)

    rng = np.random.default_rng(a.seed)
    print(f"[start] dataset={a.dataset} seed={a.seed} hvg={a.hvg} "
          f"spec_caps={SPEC_CAPS}", flush=True)
    blocks = LOADERS[a.dataset](rng, a)

    payload = dict(
        dataset=a.dataset, blocks=blocks,
        nmin_set=list(NMIN_SET),
        power_floors=dict(hard=POWER_FLOOR_HARD, soft=POWER_FLOOR_SOFT,
                          source="measured under LFC on the simulator; "
                                 "lfc/power_hard and lfc/power_soft"),
        contains_no_test=True,
        disclaimer=("DESCRIPTIVE ONLY. No rank test was run and no assumption "
                    "verdict is expressed or implied. Phase B is not "
                    "authorised."),
    )
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    meta = RIO.make_meta(
        "descriptive", "descriptives", ts,
        dict(alpha=None, B=None, n_e=None, d=None, d_latent=None, D=None,
             n_env=None, seeds=[a.seed], draws_per_point=None,
             dataset=a.dataset, hvg=a.hvg, nmin_set=list(NMIN_SET),
             spec_caps=list(SPEC_CAPS)),
        status="CURRENT",
        note=f"descriptive profile of {a.dataset}; no test, no verdict")
    meta["migrated_from"] = None
    path = RIO.write_results(payload, meta, suffix=f"__{a.dataset}")
    print(f"[write] {path}", flush=True)


if __name__ == "__main__":
    main()
