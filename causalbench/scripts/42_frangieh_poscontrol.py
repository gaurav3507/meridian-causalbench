"""Positive control for the Frangieh environment-validity screen.

CONTROL ONLY -- this script deliberately inflates the between-env contrast
by using the experimental ARM (Co-culture / Control / IFNγ) as the
environment label instead of the perturbation target. Arms have well-known
large transcriptional differences (IFNγ vs Co-culture in particular). If
the screen pipeline cannot detect a difference under this construction,
the pipeline is broken and no other Frangieh number is trustworthy.

The output is labeled `label = "POSITIVE_CONTROL_ONLY"` and must never
appear as a scientific result.

Design
------
* Same loader, same MOI==1 filter, same PCA machinery, same null
  construction as 41_screen_frangieh.py.
* Basis: sgRNA non-targeting cells pooled across ALL three arms.
* Environments: cells within each arm are split into contiguous NMIN-cell
  pseudo-envs labeled "{arm}_chunk_{i:04d}". Both perturbed and control
  cells within an arm are eligible (the arm effect is present in both).
  Pooling into pseudo-envs is required because 03_screen.py aborts at
  fewer than 4 envs, and we have only 3 arms.
* The "non-targeting" label is preserved for the PCA-basis cells only, so
  they are not double-counted as envs.

Expected outcome
----------------
mean_ratio_pairs well above 1.0 (say, >= 1.5) for a healthy pipeline. If
the number lands near the null band [0.90, 1.10] the pipeline is broken.

Usage (A100):
    python causalbench/scripts/42_frangieh_poscontrol.py
    python causalbench/scripts/42_frangieh_poscontrol.py --nmin 200
"""
import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

CAUSALBENCH = Path("/workspace/meridian-identifiability/causalbench")
SCREEN_DIR = CAUSALBENCH / "results/screen"
FRANGIEH_SCRIPT = CAUSALBENCH / "scripts/41_screen_frangieh.py"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Re-use 41's loader / regex / metadata parsing and 03's screen kernel by
# importing 41 (which itself imports 03).
FR = _load_module(FRANGIEH_SCRIPT, "_frangieh41")
load_metadata = FR.load_metadata
load_expression = FR.load_expression
screen_run = FR.screen_run
ARMS = FR.ARMS
CTRL_LABEL = FR.CTRL_LABEL
EXPR_CSV = FR.EXPR_CSV
META_CSV = FR.META_CSV
SEED = FR.SEED


def build_arm_chunk_iv(iv_all, cond_all, nmin, seed):
    """Return a new iv array where:
      - cells originally labeled non-targeting (basis cells) stay as
        CTRL_LABEL. They are pooled across arms and used ONLY for the
        PCA basis and reference pool.
      - every other cell gets label "{arm}_chunk_{i:04d}", chunked from
        that arm's cells into NMIN-size pseudo-envs.
      - leftover cells that do not complete an NMIN chunk get label
        "excluded" (screen_run's excluded-name path).
    """
    rng = np.random.default_rng(seed)
    iv_new = np.array(iv_all, dtype=object).copy()
    is_ctrl = iv_all == CTRL_LABEL
    for arm in ARMS:
        arm_mask = (cond_all == arm) & (~is_ctrl)
        arm_idx = np.where(arm_mask)[0]
        rng.shuffle(arm_idx)
        n_chunks = len(arm_idx) // nmin
        for i in range(n_chunks):
            chunk_rows = arm_idx[i * nmin:(i + 1) * nmin]
            iv_new[chunk_rows] = f"{arm}_chunk_{i:04d}"
        leftover = arm_idx[n_chunks * nmin:]
        iv_new[leftover] = "excluded"
    return iv_new


def atomic_write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.rename(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hvg", type=int, default=None)
    ap.add_argument("--nmin", type=int, default=100)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(
        SCREEN_DIR / "frangieh_poscontrol.json"))
    a = ap.parse_args()

    out_path = Path(a.out)
    if out_path.exists():
        print(f"[skip] {out_path} already exists", flush=True)
        return

    md, meta = load_metadata()
    X, vn, cell_order = load_expression(EXPR_CSV,
                                         list(md["NAME"].astype(str)),
                                         hvg=a.hvg)
    md = md.set_index(md["NAME"].astype(str)).reindex(cell_order)
    ok = md["iv"].notna().to_numpy()
    X, md = X[ok], md.loc[ok]
    iv_orig = md["iv"].to_numpy().astype(object)
    cond_all = md["condition"].to_numpy().astype(str)
    print(f"[align] {X.shape[0]} cells aligned", flush=True)

    iv_pos = build_arm_chunk_iv(iv_orig, cond_all, a.nmin, a.seed)
    us, cs = np.unique(iv_pos, return_counts=True)
    per_arm_chunk_counts = {}
    for arm in ARMS:
        n_chunks = int(sum(1 for g in us if g.startswith(f"{arm}_chunk_")))
        per_arm_chunk_counts[arm] = n_chunks
    n_ctrl = int(cs[us == CTRL_LABEL].sum()) if CTRL_LABEL in us else 0
    n_excl = int(cs[us == "excluded"].sum()) if "excluded" in us else 0
    n_env_labels = int(sum(1 for g in us
                            if g not in (CTRL_LABEL, "excluded")))
    print(f"[env] pseudo-envs per arm: {per_arm_chunk_counts}   "
          f"total envs={n_env_labels}", flush=True)
    print(f"[env] basis cells (non-targeting pooled): {n_ctrl}   "
          f"excluded (leftover): {n_excl}", flush=True)

    # STEP-0 (should return ~1.0 on control pseudo-envs) and POSCONTROL
    # (should return >> 1 if the pipeline is intact).
    print("\n[step0] control pseudo-envs from the basis pool", flush=True)
    r_step0 = screen_run(X, iv_pos, vn, a.nmin, a.d, step0=True,
                          seed=a.seed, no_drop=False)
    if "aborted" in r_step0:
        print(f"[step0] ABORTED: {r_step0['aborted']}", flush=True)
    else:
        print(f"[step0] mean_ratio_pairs={r_step0['mean_ratio_pairs']:.3f}  "
              f"n_envs={r_step0['n_envs']}", flush=True)

    print("\n[poscontrol] arm-chunk pseudo-envs (CONTROL ONLY)", flush=True)
    r_pos = screen_run(X, iv_pos, vn, a.nmin, a.d, step0=False,
                        seed=a.seed, no_drop=False)
    if "aborted" in r_pos:
        print(f"[poscontrol] ABORTED: {r_pos['aborted']}", flush=True)
    else:
        print(f"[poscontrol] mean_ratio_pairs={r_pos['mean_ratio_pairs']:.3f}"
              f"  coef_ratio_pairs={r_pos['coef_ratio_pairs']:.3f}"
              f"  n_envs={r_pos['n_envs']}", flush=True)

    # Verdict
    verdict = "PIPELINE_BROKEN"
    if "mean_ratio_pairs" in r_pos:
        v = r_pos["mean_ratio_pairs"]
        if v >= 1.5:
            verdict = "PIPELINE_INTACT_CLEAR"
        elif v >= 1.2:
            verdict = "PIPELINE_INTACT_MARGINAL"
        elif v >= 1.1:
            verdict = "PIPELINE_WEAK"
        else:
            verdict = "PIPELINE_BROKEN"

    out = dict(
        label="POSITIVE_CONTROL_ONLY",
        dataset="frangieh",
        source_metadata=str(META_CSV),
        source_expression=str(EXPR_CSV),
        seed=int(a.seed), nmin=int(a.nmin), d=int(a.d),
        env_label_source="arm-chunk pseudo-envs of NMIN cells each",
        arm_chunk_counts=per_arm_chunk_counts,
        n_basis_cells=int(n_ctrl),
        n_excluded_leftover=int(n_excl),
        n_environments=int(n_env_labels),
        step0=r_step0,
        poscontrol=r_pos,
        expected_ratio_note=("arms have well-known large transcriptional "
                              "differences (IFNγ vs Co-culture in particular); "
                              "a healthy pipeline is expected to give "
                              "mean_ratio_pairs >= 1.5 here."),
        verdict=verdict,
        do_not_report_as_result=(
            "This value is a deliberate positive control constructed by "
            "labelling cells by arm rather than perturbation. It is NOT a "
            "detectability estimate for perturbations and must never appear "
            "in a paper table as a Frangieh dataset result."),
        **meta,
    )
    atomic_write_json(out_path, out)

    print("\n" + "=" * 78, flush=True)
    print("FRANGIEH POSITIVE CONTROL (CONTROL ONLY, never report as result)",
          flush=True)
    print("=" * 78, flush=True)
    print(f"  step0 (control pseudo-envs): "
          f"mean_ratio_pairs={r_step0.get('mean_ratio_pairs', 'ABORT')}")
    if "mean_ratio_pairs" in r_pos:
        print(f"  poscontrol (arm chunks):     "
              f"mean_ratio_pairs={r_pos['mean_ratio_pairs']:.3f}")
    else:
        print(f"  poscontrol (arm chunks):     ABORT "
              f"({r_pos.get('aborted')})")
    print(f"  n_environments={n_env_labels}   "
          f"per_arm_chunks={per_arm_chunk_counts}")
    print(f"  VERDICT: {verdict}")
    print(f"\n[write] {out_path}", flush=True)


if __name__ == "__main__":
    main()
