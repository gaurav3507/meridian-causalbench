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


def data_fingerprint(X):
    """Cheap, exact fingerprint of the INPUT array.

    If a number differs between the Mac and the A100 the first question is
    whether the two machines were even given the same bytes. Shape, dtype and
    full-precision sums answer that, so a data difference is never mistaken for
    version drift.
    """
    X = np.asarray(X)
    return dict(shape=list(X.shape), dtype=str(X.dtype),
                sum=float(np.asarray(X, dtype=np.float64).sum()),
                sumsq=float((np.asarray(X, dtype=np.float64) ** 2).sum()),
                nan_count=int(np.isnan(X).sum()))


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
        data_fingerprint_controls=(data_fingerprint(X[ctrl_rows])
                                   if len(ctrl_rows) else None),
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


def do_abide(rng, args):
    """fMRI, meridian-side. Environments are SITES; timepoints are the
    per-subject frames. Small enough to profile on both machines, which is why
    it is the default --overlap-check target.

    There is no control condition, so the spectrum is over all pooled frames
    and is labelled as such, exactly as for HCP.
    """
    path = Path(args.abide_npz)
    if not path.exists():
        sys.exit(f"[fatal] ABIDE npz not found: {path}  (pass --abide-npz)")
    z = np.load(path, allow_pickle=True)
    X = np.asarray(z["X"])                      # (subjects, timepoints, regions)
    sites = np.asarray(z["site_ids"]).astype(str)
    n_sub, n_tp, n_reg = X.shape
    per_site = {}
    for s_ in sorted(set(sites)):
        per_site[s_] = int((sites == s_).sum()) * int(n_tp)
    flat = X.reshape(-1, n_reg).astype(np.float64)
    counts = list(per_site.values())
    block = dict(
        label="abide", modality="fMRI", source=str(path),
        n_subjects=int(n_sub), n_conditions=len(per_site),
        conditions=sorted(per_site), timepoints_per_subject=int(n_tp),
        n_regions=int(n_reg), total_timepoints=int(flat.shape[0]),
        timepoints_per_condition=per_site,
        environment_attrition=env_attrition(counts),
        cells_per_environment=cell_distribution(counts),
        latent_dimension_from_pooled_frames=latent_dim_block(flat, rng),
        data_fingerprint=data_fingerprint(flat),
        spectrum_caveat=("no control condition; spectrum is over ALL pooled "
                         "frames and is not comparable to the Perturb-seq "
                         "control spectra"),
        environment_caveat=("environment == acquisition SITE, not an "
                            "intervention; this is a descriptive analogue only"),
    )
    a_ = block["environment_attrition"]
    print(f"[abide] {n_sub} subjects, {len(per_site)} sites, {n_tp} frames each; "
          f"surviving: " + "  ".join(f"n>={m}:{a_[str(m)]}" for m in NMIN_SET),
          flush=True)
    return [block]


LOADERS = {"k562": lambda r, a: do_replogle("k562", r, a),
           "rpe1": lambda r, a: do_replogle("rpe1", r, a),
           "norman": do_norman, "frangieh": do_frangieh, "hcp": do_hcp,
           "abide": do_abide}



# ===================== CROSS-ENVIRONMENT REPRODUCTION GATE ===================
# The A100 `cb` venv differs from the Mac venv, and SVD-derived quantities
# depend on the BLAS backend (Mac: accelerate; A100: whatever numpy was built
# against). Without a fixed reference the two machines cannot be compared and a
# version difference would read as a finding.
#
# --selftest generates ONE fixed synthetic covariance from ONE fixed seed and
# recomputes all three dimension estimators. No bootstrap, no resampling: the
# only RNG use is generating the data, and the spectrum cap is set above n so
# the subsample path never draws.

SELFTEST_SEED = 20260810
SELFTEST_N, SELFTEST_P, SELFTEST_K = 1500, 60, 7

# ---------------------------------------------------------------------------
# PLACEHOLDER -- NOT YET PINNED.  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# These are the reference values the A100 must reproduce. They are DELIBERATELY
# None until Gaurav runs --selftest on the Mac and supplies the numbers to
# embed here. While they are None the selftest does NOT pass: it prints a
# copy-paste block and exits 2, so an unpinned gate can never be mistaken for a
# green one.
SELFTEST_EXPECTED = {
    "participation_ratio": None,                  # float, tol 1e-8
    "effective_rank_exp_spectral_entropy": None,  # float, tol 1e-8
    "n_components_for_variance": None,            # dict of ints, EXACT
}
SELFTEST_TOL = 1e-8
# END PLACEHOLDER  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ---------------------------------------------------------------------------


def _selftest_data():
    """Deterministic. Same bytes on every machine for a given numpy version."""
    rng = np.random.default_rng(SELFTEST_SEED)
    Q = np.linalg.qr(rng.standard_normal((SELFTEST_P, SELFTEST_P)))[0][:, :SELFTEST_K]
    scale = np.linspace(3.0, 0.5, SELFTEST_K)
    Z = rng.standard_normal((SELFTEST_N, SELFTEST_K)) * scale
    return Z @ Q.T + 0.05 * rng.standard_normal((SELFTEST_N, SELFTEST_P))


def run_selftest():
    """Fail loudly on any mismatch. Never warn-and-continue."""
    X = _selftest_data()
    fp = data_fingerprint(X)
    # cap above n so spectrum_estimators takes the no-subsample path: the only
    # rng passed is therefore never used to draw.
    got = spectrum_estimators(X, SELFTEST_N + 1, np.random.default_rng(0))
    v = RIO.versions()
    print("=" * 74)
    print("SELFTEST  fixed synthetic covariance, no bootstrap, no resampling")
    print("=" * 74)
    print(f"  platform : {RIO.platform_tag()}   blas={v['platform'].get('numpy_blas')}")
    print(f"  versions : numpy {v['numpy']}  scipy {v['scipy']}  "
          f"sklearn {v['sklearn']}  scanpy {v['scanpy']}")
    print(f"  data     : seed={SELFTEST_SEED} n={SELFTEST_N} p={SELFTEST_P} "
          f"k={SELFTEST_K}")
    print(f"  fingerprint sum={fp['sum']!r} sumsq={fp['sumsq']!r}")
    print("  computed :")
    print(f"    participation_ratio                  {got['participation_ratio']!r}")
    print(f"    effective_rank_exp_spectral_entropy  "
          f"{got['effective_rank_exp_spectral_entropy']!r}")
    print(f"    n_components_for_variance            {got['n_components_for_variance']}")

    if any(v_ is None for v_ in SELFTEST_EXPECTED.values()):
        print("\n  STATUS: NOT PINNED -- reference constants are still the")
        print("  placeholder. This is NOT a pass. Paste the block below into")
        print("  SELFTEST_EXPECTED in this file, commit, then rerun on both")
        print("  machines.\n")
        print("SELFTEST_EXPECTED = {")
        print(f'    "participation_ratio": {got["participation_ratio"]!r},')
        print(f'    "effective_rank_exp_spectral_entropy": '
              f'{got["effective_rank_exp_spectral_entropy"]!r},')
        print(f'    "n_components_for_variance": {got["n_components_for_variance"]!r},')
        print("}")
        return 2

    fails = []
    for key, tol_exact in (("participation_ratio", False),
                           ("effective_rank_exp_spectral_entropy", False),
                           ("n_components_for_variance", True)):
        exp, act = SELFTEST_EXPECTED[key], got[key]
        if tol_exact:
            if act != exp:
                fails.append(f"{key}: expected {exp} got {act} (must match EXACTLY)")
        else:
            delta = abs(float(act) - float(exp))
            if not delta <= SELFTEST_TOL:
                fails.append(f"{key}: expected {exp!r} got {act!r} "
                             f"|delta|={delta:.3e} > tol {SELFTEST_TOL:g}")
    if fails:
        print("\n  SELFTEST FAILED:")
        for f in fails:
            print(f"    {f}")
        print("\n  Do NOT compare results across machines until this passes.")
        return 1
    print(f"\n  SELFTEST PASSED (tol {SELFTEST_TOL:g} continuous, exact counts)")
    return 0


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def diff_against_other_platform(payload, dataset, this_tag):
    """Find prior artefacts for this dataset from a DIFFERENT platform and
    report every differing field. Silence here means the machines agree."""
    others = []
    for path, prior in RIO.iter_results(gate="descriptives",
                                        statistic="descriptive"):
        m = prior.get("meta", {})
        if prior.get("dataset") != dataset:
            continue
        tag = (m.get("config", {}).get("extra", {}) or {}).get("platform_tag") \
            or prior.get("platform_tag")
        if tag and tag != this_tag:
            others.append((path, prior, tag))
    if not others:
        print(f"[overlap] no prior {dataset} artefact from another platform; "
              f"nothing to diff. Run this on the other machine and rerun.")
        return None
    report = []
    a = _flatten(payload.get("blocks"))
    for path, prior, tag in others:
        b = _flatten(prior.get("blocks"))
        keys = sorted(set(a) | set(b))
        diffs = []
        for k in keys:
            va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
            if isinstance(va, float) and isinstance(vb, float):
                if abs(va - vb) <= 1e-12 or (va == vb):
                    continue
            elif va == vb:
                continue
            diffs.append(dict(field=k, this=va, other=vb))
        print(f"\n[overlap] vs {path.name}  ({tag})")
        print(f"[overlap]   {len(diffs)} differing field(s) of {len(keys)}")
        for d in diffs[:40]:
            print(f"    {d['field']}\n      this ({this_tag}): {d['this']}"
                  f"\n      other ({tag}): {d['other']}")
        if len(diffs) > 40:
            print(f"    ... {len(diffs)-40} more")
        report.append(dict(compared_to=str(path.name), other_platform=tag,
                           n_fields=len(keys), n_differing=len(diffs),
                           differing=diffs))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(LOADERS),
                    help="profile one dataset")
    ap.add_argument("--selftest", action="store_true",
                    help="fixed-seed reproduction gate; exits non-zero on "
                         "mismatch or while the constants are unpinned")
    ap.add_argument("--overlap-check", dest="overlap_check",
                    choices=sorted(LOADERS), nargs="?", const="abide",
                    help="profile a dataset, tag it with the platform, and "
                         "diff against any prior run from another machine "
                         "(default: abide)")
    ap.add_argument("--abide-npz",
                    default="/Users/gauravgoyal/Downloads/meridian/data/"
                            "processed/abide_harmonized.npz",
                    help="path to the ABIDE npz (differs per machine)")
    ap.add_argument("--hvg", type=int, default=None, help="Frangieh only")
    ap.add_argument("--n-spec-cap", type=int, default=None,
                    help="override the spectrum subsample caps (default 2000,8000)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(run_selftest())
    dataset = a.dataset or a.overlap_check
    if not dataset:
        ap.error("give --dataset, --selftest, or --overlap-check")
    a.dataset = dataset
    if a.n_spec_cap:
        global SPEC_CAPS
        SPEC_CAPS = (a.n_spec_cap,)

    rng = np.random.default_rng(a.seed)
    print(f"[start] dataset={a.dataset} seed={a.seed} hvg={a.hvg} "
          f"spec_caps={SPEC_CAPS}", flush=True)
    blocks = LOADERS[a.dataset](rng, a)

    payload = dict(
        dataset=a.dataset, blocks=blocks,
        platform_tag=RIO.platform_tag(),
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
             spec_caps=list(SPEC_CAPS), platform_tag=RIO.platform_tag(),
             overlap_check=bool(a.overlap_check)),
        status="CURRENT",
        note=f"descriptive profile of {a.dataset}; no test, no verdict")
    meta["migrated_from"] = None
    if a.overlap_check:
        payload["overlap_report"] = diff_against_other_platform(
            payload, a.dataset, RIO.platform_tag())
    path = RIO.write_results(payload, meta, suffix=f"__{a.dataset}")
    print(f"[write] {path}", flush=True)
    if a.overlap_check:
        r = payload.get("overlap_report")
        if r:
            tot = sum(x["n_differing"] for x in r)
            print(f"[overlap] TOTAL differing fields across "
                  f"{len(r)} comparison(s): {tot}", flush=True)


if __name__ == "__main__":
    main()
