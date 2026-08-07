"""PHASE B -- rank diagnostic on real perturbation data. A100. NOT RUN ON THE MAC.

Environments are perturbations; environment 0 is the pooled control cells.

    H0(k):  rank( Sigma_e - Sigma_0 ) <= k,  k = 2

r_hat > 2 falsifies the CRL assumption bundle for that environment.
r_hat <= 2 is NOT evidence the bundle holds -- covariance is mean-centred, so
a pure shift intervention gives r_hat = 0 trivially. Only rejection carries a
claim. This is the second-order complement to the mean-shift screen in
03_screen.py, not a replacement for it.

=============================================================================
THIS SCRIPT REFUSES TO RUN WHILE GATE 0 IS FAILING.
=============================================================================
Gate 0 (calibration, 81_ranktest_oracle.py) currently FAILS: on pure control
data with no intervention anywhere, r_hat > 0 fires at 1-(1-alpha)^d rather
than alpha, because r_hat sums d marginal level-alpha tests with no
multiplicity control. Its null distribution is approximately Binomial(d,
alpha), not a point mass at 0. The falsification event r_hat > 2 inherits
this: measured false-rejection rates on data with NO intervention were 0.000
at d=5, 0.015 at d=10 and 0.090 at d=20.

require_gate0_pass() below reads the Gate 0 JSON and aborts unless the
verdict is PASS. Do not bypass it. Every number this script could produce
today would carry that uncalibrated false-rejection rate, which at d=20 means
roughly one environment in eleven is falsely called a bundle violation.

When the statistic is fixed, the rank_diagnostic call sites here do not
change, but the READOUT does, so re-read the interpretation of r_hat before
quoting anything from this script.

DATA HANDLING -- inherited, not rediscovered
--------------------------------------------
Replogle K562/RPE1 : 03_screen.load(ds, filt)
Norman             : 40_screen_norman.load_norman()
Frangieh           : 41_screen_frangieh.load_metadata() + load_expression(),
                     which already encode all of:
                       * RNA_metadata.csv row 2 is SCP's TYPE convention row
                         (skiprows=[1]) or it becomes a phantom cell
                       * MOI == 1 only
                       * target = sgRNA with the TRAILING _<digits> guide
                         index stripped, ^(.*)_\\d+$ -- NOT split on the first
                         underscore, which would mangle NO_SITE_1 -> "NO"
                       * controls = pooled NO_SITE_* and ONE_NON-GENE_SITE_*
                       * chunked float32 read of the ~218k-cell dense CSV
                     Frangieh is screened PER ARM (Co-culture / Control /
                     IFNg), each with its own control basis and reference
                     pool. Never pooled: a pooled run puts the arm effect into
                     the numerator.
                     NOTE the arm label is "IFNγ" with U+03B3, verified from
                     41_screen_frangieh.py:83, not ASCII "IFNg".

fit_pca / project reach this script through 80_ranktest_core, which imports
them from 03_screen.py via importlib. PCA is never reimplemented, so the
arithmetic is byte-identical to the existing screen.

SIZING
------
d in {5, 10, 20}. An environment is scored only if n_e >= 10 * d: covariance
estimation needs far more cells than the mean-shift screen did, and a d x d
covariance from fewer cells is dominated by estimation noise. Environments
that drop out are COUNTED AND REPORTED per dataset per d -- that attrition is
itself a result, not bookkeeping.

The control pool is split disjointly into a PCA basis and a reference pool.
The reference pool must satisfy n_p >= 3 * n_match or the environment is
skipped and reported, never scored by recycling cells.

Usage (A100, long runs via nohup only):
    python causalbench/scripts/82_ranktest_real.py --dataset k562 --dry-run
    nohup python causalbench/scripts/82_ranktest_real.py --dataset k562 &
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "ranktest"
GATE0_JSON = RESULTS / "gate0.json"

CTRL_LABEL = "non-targeting"
D_SET = (5, 10, 20)
ALPHA = 0.05
B_NULL = 500
SEED = 0
MIN_CELLS_PER_D = 10          # n_e >= MIN_CELLS_PER_D * d to be scored
BASIS_FRAC = 1.0 / 3.0        # control cells given to the PCA basis


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


CORE = _load_module(HERE / "80_ranktest_core.py", "_ranktest_core")
rank_diagnostic = CORE.rank_diagnostic
standardise = CORE.standardise
REF_SPLIT_FACTOR = CORE.REF_SPLIT_FACTOR


# --------------------------------------------------------------- gate 0 lock
def require_gate0_pass(force_flag=False):
    if force_flag:
        sys.exit("--i-know-gate0-failed is not implemented on purpose. "
                 "Fix the statistic, rerun 81_ranktest_oracle.py --gate 0, "
                 "then run this.")
    if not GATE0_JSON.exists():
        sys.exit(f"REFUSING TO RUN: {GATE0_JSON} not found. Gate 0 must pass "
                 f"before any real-data number is computed.")
    verdict = json.load(open(GATE0_JSON)).get("verdict")
    if verdict != "PASS":
        sys.exit(
            f"REFUSING TO RUN: Gate 0 verdict is {verdict!r}, not 'PASS'.\n"
            f"  The r_hat readout is not calibrated: on control-only data with\n"
            f"  no intervention, r_hat>0 fires at ~1-(1-alpha)^d instead of\n"
            f"  alpha, and r_hat>2 -- the event that falsifies the bundle --\n"
            f"  fired at up to 0.09 at d=20 with nothing to detect.\n"
            f"  Every environment this script scored would inherit that rate.\n"
            f"  See {GATE0_JSON}."
        )
    print(f"[gate0] verdict=PASS, proceeding", flush=True)


# -------------------------------------------------------------------- loaders
def load_replogle(ds, filt=False):
    screen = _load_module(HERE / "03_screen.py", "_screen03_loader")
    X, iv, vn = screen.load(ds, filt)
    return X, np.asarray(iv, dtype=object), vn, None


def load_norman():
    nor = _load_module(HERE / "40_screen_norman.py", "_norman40")
    got = nor.load_norman()
    X, iv, vn = got[0], got[1], got[2]
    return np.asarray(X), np.asarray(iv, dtype=object), list(vn), None


def load_frangieh(hvg=None):
    fr = _load_module(HERE / "41_screen_frangieh.py", "_frangieh41")
    md, _meta = fr.load_metadata()
    X, vn, cell_order = fr.load_expression(fr.EXPR_CSV,
                                           list(md["NAME"].astype(str)), hvg=hvg)
    md = md.set_index(md["NAME"].astype(str)).reindex(cell_order)
    ok = md["iv"].notna().to_numpy()
    X, md = X[ok], md.loc[ok]
    iv = md["iv"].to_numpy().astype(object)
    arms = md["condition"].to_numpy().astype(str)
    return X, iv, list(vn), arms


LOADERS = {
    "k562": lambda: load_replogle("k562"),
    "rpe1": lambda: load_replogle("rpe1"),
    "norman": load_norman,
    "frangieh": load_frangieh,
}


# ----------------------------------------------------------------- the sweep
def split_controls(ctrl_rows, rng):
    """Disjoint PCA basis and reference pool, by INDEX, from the control cells."""
    idx = np.array(ctrl_rows, copy=True)
    rng.shuffle(idx)
    cut = int(len(idx) * BASIS_FRAC)
    basis_idx, ref_idx = idx[:cut], idx[cut:]
    assert not (set(basis_idx.tolist()) & set(ref_idx.tolist())), \
        "control split produced overlapping basis and reference pools"
    return basis_idx, ref_idx


def run_block(X, iv, vn, tag, rng, no_drop=False, standardise_cols=False,
              b_null=B_NULL, dry_run=False):
    """One dataset, or one Frangieh arm. Sweeps d and every environment."""
    gidx = {g: i for i, g in enumerate(vn)}
    ctrl_rows = np.where(iv == CTRL_LABEL)[0]
    targets, counts = np.unique(iv[iv != CTRL_LABEL], return_counts=True)

    block = dict(tag=tag, n_cells=int(X.shape[0]), n_genes=int(X.shape[1]),
                 n_control_cells=int(len(ctrl_rows)),
                 n_targets_total=int(len(targets)),
                 no_drop=bool(no_drop), standardised=bool(standardise_cols),
                 per_d={}, runs=[])

    if len(ctrl_rows) < 10:
        block["aborted"] = "not_enough_control_cells"
        print(f"[{tag}] ABORT: {len(ctrl_rows)} control cells", flush=True)
        return block

    basis_idx, ref_idx = split_controls(ctrl_rows, rng)
    block["n_basis"] = int(len(basis_idx))
    block["n_ref_pool"] = int(len(ref_idx))

    Xw = X
    if standardise_cols:
        Xw = standardise(X, X[basis_idx])

    for d in D_SET:
        need = MIN_CELLS_PER_D * d
        surviving, dropped_small, dropped_pool = [], [], []
        for g, n_e in zip(targets, counts):
            if n_e < need:
                dropped_small.append((str(g), int(n_e)))
                continue
            n_match = int(min(n_e, len(ref_idx) // REF_SPLIT_FACTOR))
            if len(ref_idx) < REF_SPLIT_FACTOR * n_match or n_match < 2:
                dropped_pool.append((str(g), int(n_e)))
                continue
            surviving.append((str(g), int(n_e), n_match))

        block["per_d"][str(d)] = dict(
            d=d, n_cells_required=need,
            n_env_surviving=len(surviving),
            n_env_dropped_too_few_cells=len(dropped_small),
            n_env_dropped_control_pool_too_small=len(dropped_pool),
            env_surviving=[s[0] for s in surviving],
            env_dropped_too_few_cells=[s[0] for s in dropped_small],
        )
        print(f"[{tag}] d={d:<3} need n_e>={need:<4} "
              f"SURVIVING {len(surviving):>4} / {len(targets)} environments "
              f"(dropped {len(dropped_small)} too-few-cells, "
              f"{len(dropped_pool)} control-pool-too-small)", flush=True)

        if dry_run:
            continue

        band_cache = {}
        for g, n_e, n_match in surviving:
            drop = set()
            if not no_drop and g in gidx:
                drop = {gidx[g]}
            env_rows = np.where(iv == g)[0]
            # Band depends on (d, n_match, drop) only -- not on which
            # environment -- so it is cached, not recomputed per environment.
            key = (d, n_match, tuple(sorted(drop)))
            r = rank_diagnostic(
                Xw[env_rows], Xw[basis_idx], Xw[ref_idx], d, n_match,
                b_null, ALPHA, rng, drop=drop,
                basis_idx=basis_idx, ref_pool_idx=ref_idx,
                null_band=band_cache.get(key))
            band_cache.setdefault(key, r["band"])
            block["runs"].append(dict(
                d=d, env=str(g), n_env=int(n_e), n_match=r["n_match"],
                r_hat=r["r_hat"], reject_bundle=r["reject_bundle"],
                lam=r["lam"], band=r["band"],
                target_in_columns=bool(g in gidx)))

        scored = [r for r in block["runs"] if r["d"] == d]
        if scored:
            rej = sum(r["reject_bundle"] for r in scored)
            block["per_d"][str(d)].update(
                n_scored=len(scored), n_reject_bundle=rej,
                frac_reject_bundle=rej / len(scored),
                r_hat_median=float(np.median([r["r_hat"] for r in scored])))
            print(f"[{tag}] d={d:<3} r_hat>2 in {rej}/{len(scored)} "
                  f"({rej/len(scored):.3f}), median r_hat="
                  f"{block['per_d'][str(d)]['r_hat_median']:.1f}", flush=True)
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--dry-run", action="store_true",
                    help="validate loaders, print shapes and surviving counts, exit")
    ap.add_argument("--no-drop", action="store_true",
                    help="disable the targeted-gene column drop in project()")
    ap.add_argument("--standardise", action="store_true")
    ap.add_argument("--hvg", type=int, default=None, help="Frangieh only")
    ap.add_argument("--b-null", type=int, default=B_NULL)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not a.dry_run:
        require_gate0_pass()

    rng = np.random.default_rng(a.seed)
    print(f"[load] dataset={a.dataset}", flush=True)
    X, iv, vn, arms = (LOADERS[a.dataset]() if a.dataset != "frangieh"
                       else load_frangieh(hvg=a.hvg))
    print(f"[load] X={X.shape} dtype={X.dtype}  "
          f"{int((iv == CTRL_LABEL).sum())} control cells  "
          f"{len(set(iv[iv != CTRL_LABEL]))} targets", flush=True)

    out = dict(dataset=a.dataset, seed=a.seed, alpha=ALPHA, B_null=a.b_null,
               d_set=list(D_SET), min_cells_per_d=MIN_CELLS_PER_D,
               ref_split_factor=REF_SPLIT_FACTOR, basis_frac=BASIS_FRAC,
               no_drop=bool(a.no_drop), standardised=bool(a.standardise),
               dry_run=bool(a.dry_run), per_arm=(arms is not None), blocks=[])

    if arms is None:
        out["blocks"].append(run_block(X, iv, vn, a.dataset, rng,
                                       no_drop=a.no_drop,
                                       standardise_cols=a.standardise,
                                       b_null=a.b_null, dry_run=a.dry_run))
    else:
        for arm in sorted(set(arms)):
            sel = arms == arm
            print(f"\n{'=' * 60}\n  ARM {arm}  ({int(sel.sum())} cells)\n{'=' * 60}",
                  flush=True)
            out["blocks"].append(run_block(
                np.ascontiguousarray(X[sel]), iv[sel], vn, f"{a.dataset}:{arm}",
                rng, no_drop=a.no_drop, standardise_cols=a.standardise,
                b_null=a.b_null, dry_run=a.dry_run))

    # ---- attrition table, printed prominently rather than buried in JSON
    print("\n" + "=" * 78)
    print(f"SURVIVING ENVIRONMENTS -- {a.dataset}   (n_e >= {MIN_CELLS_PER_D} * d)")
    print("=" * 78)
    print(f"  {'block':<28}{'d':>4}{'targets':>10}{'surviving':>11}{'dropped':>9}")
    for b in out["blocks"]:
        for d in D_SET:
            p = b.get("per_d", {}).get(str(d))
            if not p:
                continue
            print(f"  {b['tag']:<28}{d:>4}{b['n_targets_total']:>10}"
                  f"{p['n_env_surviving']:>11}"
                  f"{p['n_env_dropped_too_few_cells'] + p['n_env_dropped_control_pool_too_small']:>9}")
    print("=" * 78)

    if a.dry_run:
        print("\n[dry-run] loaders validated, nothing computed, exiting.")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = ("_nodrop" if a.no_drop else "") + ("_std" if a.standardise else "")
    path = Path(a.out) if a.out else RESULTS / f"real_{a.dataset}{tag}.json"
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.rename(tmp, path)
    print(f"[write] {path}", flush=True)


if __name__ == "__main__":
    main()
