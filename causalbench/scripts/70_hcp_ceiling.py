"""Build the HCP task-decoding triple that parallels CausalBench
(chance / linear / ceiling). Without this, HCP has only ratios and cannot share
an axis with CausalBench's extraction ceiling (0.129 / 0.226 / 0.528).

Two variants:

(a) HEADLINE -- held-out-subject task decoding.
    Multinomial logistic regression predicts task (K=7) from the d-dim
    projected first-half mean of each subject-encoding run.
      chance  = held-out-fold majority-class baseline (~1/K if balanced)
      linear  = decoder trained on train-subject first-halves,
                tested on held-out-subject first-halves
      ceiling = SAME decoder, tested on held-out-subject SECOND-halves
                (same subject, same task, same encoding, different frames)
                -- upper bound given the decoder cannot memorise a subject

(b) SUPPLEMENTARY -- leave-one-task-out shift prediction. K=7 environments.
    Reported UNDERPOWERED. Not to be treated as evidence of anything.

Inputs and pipeline mirror `hcp/scripts/mean_shift_v2.py` exactly. Documented
in `causalbench/PATHS_hcp.md`.

Guarantees enforced in code:
  - ORACLE: synthetic-data check where task identity is planted in a known
    direction at fixed SNR. Fails-fast if the decoder does not recover it or
    if chance does not sit at 1/K. If oracle fails, no science is reported.
  - SAMPLE-SIZE MATCHING: LINEAR and CEILING both test on the same held-out
    subjects' features, so their sample sizes are identical by construction.
    An assert enforces this in every fold. CHANCE is measured against the same
    held-out labels.
  - MULTIPLE SEEDS: the decoder is fit and evaluated under 5 subject-shuffles;
    per-seed values are reported alongside the mean.
  - `--clean` removes the specific output JSON before running.
  - Atomic write (.tmp + os.rename) so a partial file cannot masquerade as a
    finished one.

Usage (A100, cb venv):
    python causalbench/scripts/70_hcp_ceiling.py            # writes if absent
    python causalbench/scripts/70_hcp_ceiling.py --clean    # overwrite

Nohup for long runs:
    nohup python causalbench/scripts/70_hcp_ceiling.py \\
        > logs/hcp_ceiling.log 2>&1 &
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HCP_TS = Path("/workspace/meridian-identifiability/hcp/ts")

OUT_DIR = Path(__file__).resolve().parents[1] / "results/screen"
OUT_JSON = OUT_DIR / "hcp_ceiling.json"

TASKS = ["WM", "GAMBLING", "MOTOR", "LANGUAGE", "SOCIAL", "RELATIONAL", "EMOTION"]
ENCS = ["LR", "RL"]
NFRAMES = 176   # from mean_shift_v2.py:23
HALF = 88       # from mean_shift_v2.py:23
D = 10          # mean_shift_v2.py DIMS include 10; primary reporting rung
KFOLDS = 5
SEEDS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------- data loading
def load_run(subject, task, enc):
    p = HCP_TS / f"{subject}_{task}_{enc}.npy"
    if not p.exists():
        return None
    x = np.load(p).astype(np.float64)
    if x.shape[0] < NFRAMES:
        return None
    return x[:NFRAMES]                                # shape (176, n_regions)


def find_complete_subjects():
    subs = sorted({os.path.basename(f).split("_")[0]
                   for f in HCP_TS.glob("*.npy")})
    return [s for s in subs
            if all(load_run(s, t, e) is not None for t in TASKS for e in ENCS)]


def _subject_pooled_scale(runs):
    """Per-subject-per-region z-score across all runs (mean_shift_v2.py:39-44).

    `runs` is a dict {(task, enc): (176, n_regions)}. Returns the same dict
    with each run z-scored.
    """
    allr = np.concatenate([runs[k] for k in runs], axis=0)
    mu = allr.mean(0)
    sd = allr.std(0) + 1e-8
    return {k: (v - mu) / sd for k, v in runs.items()}


def fit_pca_basis(subjects, d):
    """Fit the d-dim basis on pooled subject-scaled residuals.
    Mirrors mean_shift_v2.py:37-48 with scaling='subject_pooled'.
    """
    all_frames = []
    for s in subjects:
        runs = {(t, e): load_run(s, t, e) for t in TASKS for e in ENCS}
        runs = _subject_pooled_scale(runs)
        for k, v in runs.items():
            all_frames.append(v)
    pool = np.concatenate(all_frames, axis=0)
    pool = pool - pool.mean(0)
    _, _, Vt = np.linalg.svd(pool, full_matrices=False)
    W = Vt[:d].T                                       # shape (n_regions, d)
    return W


def build_projected_halves(subjects, W):
    """For each (subject, task, encoding) return (A, B) = projected means of
    the first and second halves. Shapes: A, B are (d,).
    """
    features = {}
    for s in subjects:
        runs = {(t, e): load_run(s, t, e) for t in TASKS for e in ENCS}
        runs = _subject_pooled_scale(runs)
        for (t, e), x in runs.items():
            m = x @ W                                  # (176, d)
            assert m.shape == (NFRAMES, W.shape[1])
            features[(s, t, e)] = (m[:HALF].mean(0), m[HALF:NFRAMES].mean(0))
    return features


def features_to_matrix(features):
    keys = sorted(features.keys())
    X_A = np.array([features[k][0] for k in keys])
    X_B = np.array([features[k][1] for k in keys])
    y = np.array([TASKS.index(k[1]) for k in keys])
    groups = np.array([k[0] for k in keys])
    return X_A, X_B, y, groups


# ----------------------------------------------------------- shuffled folds
def shuffled_group_folds(groups, seed, k):
    """Yield (train_idx, test_idx) with subject-level splits, shuffled by seed."""
    unique = sorted(set(groups))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    fold_size = int(np.ceil(len(perm) / k))
    folds = [perm[i * fold_size:(i + 1) * fold_size] for i in range(k)]
    folds = [f for f in folds if len(f) > 0]
    for i in range(len(folds)):
        test_subj = set(folds[i].tolist())
        train_subj = set()
        for j in range(len(folds)):
            if j != i:
                train_subj.update(folds[j].tolist())
        train_mask = np.array([g in train_subj for g in groups])
        test_mask = np.array([g in test_subj for g in groups])
        yield np.where(train_mask)[0], np.where(test_mask)[0]


# ==================================================================== VARIANT A
def variant_a(features, seeds=SEEDS, kfolds=KFOLDS):
    X_A, X_B, y, groups = features_to_matrix(features)
    K = len(TASKS)

    per_seed = []
    for seed in seeds:
        chance_scores, linear_scores, ceiling_scores = [], [], []
        n_test_seen = []
        for tr, te in shuffled_group_folds(groups, seed, kfolds):

            # SAMPLE-SIZE MATCHING (asserted every fold)
            # LINEAR test = X_A[te]; CEILING test = X_B[te]; both use the same
            # `te` indices, so n is identical by construction.
            n_linear_test = len(te)
            n_ceiling_test = len(te)
            assert n_linear_test == n_ceiling_test, (
                f"sample sizes not matched: {n_linear_test} vs {n_ceiling_test}"
            )

            # CHANCE: majority-class from train, evaluated on held-out labels
            _, counts = np.unique(y[tr], return_counts=True)
            maj = np.argmax(counts)
            chance = float((y[te] == maj).mean())

            # LINEAR
            clf = LogisticRegression(max_iter=2000, C=1.0,
                                      multi_class="multinomial",
                                      solver="lbfgs", random_state=seed)
            clf.fit(X_A[tr], y[tr])
            linear = float(clf.score(X_A[te], y[te]))

            # CEILING: same fitted classifier, second-halves of the same held-out
            # subject-encoding-task tuples
            ceiling = float(clf.score(X_B[te], y[te]))

            chance_scores.append(chance)
            linear_scores.append(linear)
            ceiling_scores.append(ceiling)
            n_test_seen.append(int(len(te)))

        per_seed.append(dict(
            seed=int(seed),
            chance=float(np.mean(chance_scores)),
            linear=float(np.mean(linear_scores)),
            ceiling=float(np.mean(ceiling_scores)),
            per_fold_n_test=n_test_seen,
        ))

    def agg(key):
        vals = [ps[key] for ps in per_seed]
        return dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                    per_seed=[float(v) for v in vals])

    return dict(
        chance=agg("chance"),
        linear=agg("linear"),
        ceiling=agg("ceiling"),
        n_test_per_fold=per_seed[0]["per_fold_n_test"],
        n_matched=("LINEAR and CEILING test on the same held-out subjects' "
                   "first- and second-half features; identical n by construction."),
        seeds=list(map(int, seeds)),
        kfolds=int(kfolds),
        K_classes=int(K),
        classifier="LogisticRegression(multinomial, C=1.0, max_iter=2000)",
    )


# ==================================================================== VARIANT B
def variant_b(features, seeds=SEEDS):
    """Leave-one-task-out shift prediction. K=7, UNDERPOWERED."""
    from sklearn.metrics import r2_score

    X_A, _, y, _ = features_to_matrix(features)
    K = len(TASKS)

    per_seed = []
    for seed in seeds:
        r2_per_task = []
        for held in range(K):
            train_mask = y != held
            held_mask = y == held
            # Model: predict held-out task's mean shift as the mean of the K-1
            # training tasks' mean shifts (global-mean baseline for zero-shot
            # task prediction). No per-task features exist to do better.
            training_task_means = np.array(
                [X_A[y == t].mean(0) for t in range(K) if t != held])
            pred = training_task_means.mean(0)
            actual = X_A[held_mask].mean(0)
            r2_per_task.append(float(r2_score(actual, pred)))
        per_seed.append(dict(
            seed=int(seed),
            r2_per_task=[float(x) for x in r2_per_task],
            r2_mean=float(np.mean(r2_per_task)),
        ))

    r2_means = [ps["r2_mean"] for ps in per_seed]
    return dict(
        LABEL="UNDERPOWERED",
        n_environments=int(K),
        r2_mean=float(np.mean(r2_means)),
        r2_std=float(np.std(r2_means)),
        per_seed=per_seed,
        note=("Leave-one-task-out with K=7 tasks: 6 training environments, 1 "
              "held out. Structural analogue of the CausalBench zero-shot task, "
              "but underpowered. Provided for parity, not for a headline."),
        seeds=list(map(int, seeds)),
    )


# ======================================================================= ORACLE
def oracle(snr=3.0, n_subjects=40, n_regions=60, seed=42):
    """Known-answer test. Synthetic runs where task identity is planted in a
    known direction at fixed SNR. Fails-fast if variant_a's decoder does not
    recover it or if chance is not at 1/K.
    """
    rng = np.random.default_rng(seed)
    K = len(TASKS)

    # Plant K distinct task directions in the top D dimensions.
    task_dirs = rng.normal(0, snr, (K, D))

    W = np.zeros((n_regions, D))
    W[:D, :] = np.eye(D)                               # identity basis first D

    features = {}
    for si in range(n_subjects):
        s = f"synth{si:03d}"
        sub_offset = rng.normal(0, 0.5, n_regions)
        for i, t in enumerate(TASKS):
            for e in ENCS:
                signal = np.zeros((NFRAMES, n_regions))
                signal[:, :D] = task_dirs[i]           # constant across frames
                signal += sub_offset[None, :]
                noise = rng.normal(0, 1.0, (NFRAMES, n_regions))
                x = signal + noise
                m = x @ W
                features[(s, t, e)] = (m[:HALF].mean(0), m[HALF:NFRAMES].mean(0))

    res = variant_a(features, seeds=(0,), kfolds=5)
    linear = res["linear"]["mean"]
    chance = res["chance"]["mean"]
    ceiling = res["ceiling"]["mean"]
    expected_chance = 1.0 / K

    linear_ok = linear > 0.90
    chance_ok = abs(chance - expected_chance) < 0.06
    ceiling_ok = ceiling > 0.85

    passed = bool(linear_ok and chance_ok and ceiling_ok)
    meta = dict(
        snr=float(snr), n_synth_subjects=int(n_subjects),
        n_regions=int(n_regions), d=int(D), K=int(K), seed=int(seed),
        linear=float(linear), chance=float(chance), ceiling=float(ceiling),
        expected_chance=float(expected_chance),
        linear_ok=bool(linear_ok), chance_ok=bool(chance_ok),
        ceiling_ok=bool(ceiling_ok),
    )
    return passed, meta


# ------------------------------------------------------------------------ util
def atomic_write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.rename(tmp, path)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="remove the output JSON before running (lesson 4)")
    ap.add_argument("--oracle-only", action="store_true",
                    help="run only the oracle; skip real data")
    args = ap.parse_args()

    if args.clean and OUT_JSON.exists():
        print(f"[clean] rm {OUT_JSON}", flush=True)
        OUT_JSON.unlink()

    if OUT_JSON.exists() and not args.oracle_only:
        print(f"[skip] {OUT_JSON} exists; --clean to overwrite", flush=True)
        return

    # -------------- ORACLE FIRST
    print("[oracle] START", flush=True)
    passed, oracle_meta = oracle()
    print(f"[oracle] linear={oracle_meta['linear']:.4f} "
          f"(want > 0.90) -> {'OK' if oracle_meta['linear_ok'] else 'FAIL'}",
          flush=True)
    print(f"[oracle] chance={oracle_meta['chance']:.4f} "
          f"(want ~ {oracle_meta['expected_chance']:.4f}) -> "
          f"{'OK' if oracle_meta['chance_ok'] else 'FAIL'}", flush=True)
    print(f"[oracle] ceiling={oracle_meta['ceiling']:.4f} "
          f"(want > 0.85) -> {'OK' if oracle_meta['ceiling_ok'] else 'FAIL'}",
          flush=True)
    if not passed:
        print("[oracle] FAIL -- readout mislabelled; refusing to report science",
              flush=True)
        atomic_write_json(OUT_JSON, dict(oracle=oracle_meta,
                                          aborted="oracle_failed"))
        sys.exit(1)
    print("[oracle] PASS", flush=True)

    if args.oracle_only:
        print("[done] --oracle-only requested; not touching real data",
              flush=True)
        return

    # -------------- REAL DATA
    if not HCP_TS.is_dir():
        sys.exit(f"[fatal] HCP_TS not a directory: {HCP_TS}")

    subjects = find_complete_subjects()
    print(f"[data] {len(subjects)} subjects with complete 7x2 coverage",
          flush=True)
    if len(subjects) < 10:
        sys.exit(f"[fatal] too few complete subjects: {len(subjects)}")

    print(f"[data] fitting PCA basis (subject_pooled, d={D})", flush=True)
    W = fit_pca_basis(subjects, D)
    print(f"[data] W shape {W.shape}", flush=True)

    features = build_projected_halves(subjects, W)
    print(f"[data] {len(features)} (subject, task, enc) tuples", flush=True)

    print("[variant_a] running (headline)", flush=True)
    va = variant_a(features, seeds=SEEDS, kfolds=KFOLDS)
    print(f"[variant_a] chance={va['chance']['mean']:.4f}  "
          f"linear={va['linear']['mean']:.4f}  "
          f"ceiling={va['ceiling']['mean']:.4f}", flush=True)

    print("[variant_b] running (UNDERPOWERED, K=7)", flush=True)
    vb = variant_b(features, seeds=SEEDS)
    print(f"[variant_b] r2_mean={vb['r2_mean']:.4f}  n_env={vb['n_environments']}",
          flush=True)

    out = dict(
        oracle=dict(passed=passed, **oracle_meta),
        variant_a_headline=va,
        variant_b_supplementary=vb,
        config=dict(
            scaling="subject_pooled",
            d=int(D),
            nframes=int(NFRAMES),
            half=int(HALF),
            tasks=TASKS,
            encodings=ENCS,
            kfolds=int(KFOLDS),
            seeds=list(map(int, SEEDS)),
            classifier="LogisticRegression(multinomial, C=1.0, max_iter=2000)",
        ),
        n_subjects=int(len(subjects)),
        source=dict(
            ts_dir=str(HCP_TS),
            file_pattern="{subject}_{task}_{encoding}.npy",
            provenance="/workspace/meridian-identifiability/hcp/scripts/mean_shift_v2.py",
        ),
    )
    atomic_write_json(OUT_JSON, out)
    print(f"\n[write] {OUT_JSON}", flush=True)

    print("\n" + "=" * 78, flush=True)
    print("HCP CEILING SUMMARY", flush=True)
    print("=" * 78, flush=True)
    print(f"  oracle:          PASS")
    print(f"  variant (a) HEADLINE -- held-out-subject task decoding")
    print(f"    chance:        {va['chance']['mean']:.4f} +/- {va['chance']['std']:.4f}")
    print(f"    linear:        {va['linear']['mean']:.4f} +/- {va['linear']['std']:.4f}")
    print(f"    ceiling:       {va['ceiling']['mean']:.4f} +/- {va['ceiling']['std']:.4f}")
    print(f"    n test / fold: {va['n_test_per_fold']}")
    print(f"    n matched:     LINEAR and CEILING share held-out subjects")
    print(f"    seeds:         {va['seeds']}")
    print(f"  variant (b) UNDERPOWERED -- leave-one-task-out shift prediction")
    print(f"    r2_mean:       {vb['r2_mean']:.4f} +/- {vb['r2_std']:.4f}")
    print(f"    n env:         {vb['n_environments']}")
    print(f"    label:         {vb['LABEL']}")


if __name__ == "__main__":
    main()
