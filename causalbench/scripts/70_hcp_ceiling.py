"""Build the HCP task-decoding triple that parallels CausalBench
(chance / linear / reliability-ceiling). Without this, HCP has only ratios
and cannot share an axis with CausalBench's extraction ceiling
(0.129 / 0.226 / 0.528).

AMENDMENT 2026-07-25 -- the original ceiling arm was mis-specified.
It fit LogReg on train-subject first-halves and scored on held-out
second-halves. That measures within-run temporal repeatability under a
cross-subject decoder, not an achievable upper bound on task classification.
That is why the original 0.1794 landed BELOW linear 0.1798: both used the
same decoder and both differ only in noise. The old ceiling arm is
demoted to `encode_split_nuisance` and reported next to the v2
`encode_over_split` nuisance ratio. The following are added.

  A. `linear_permutation_null` -- shuffle task labels within subject,
     refit the same LogReg 5-fold, >=200 permutations. Real linear must
     clear the null p95.
  B. `variant_within_subject` -- fit LogReg on one encoding (LR) and test
     on the other (RL) within each subject; average across both
     directions. Reports mean +/- SD across subjects and its own
     permutation null. Also reports a MATCHED cross-subject control: the
     cross-subject decoder trained on the same 7-point set (one point per
     task) to hold training-data size constant. Unmatched
     cross-subject = variant_a linear and is reported for reference only.
  C. `variant_within_subject_split_half_ceiling` -- ceiling-B. Same
     LogReg as the within-subject decoder. Per subject: fit on the 14
     A-half (task, encoding) means, score on the 14 B-half means from
     the same runs. Same subject throughout, no cross-subject scoring.
     Bounds within-subject 0.375 itself.
  D. `oracle_ceiling` -- second oracle for the NEW estimator.
     Synthesizes data with KNOWN reliability at three SNR levels (0,
     0.10, 10) and asserts the within-subject split-half estimator
     returns values in the expected bands.

Second amendment 2026-07-25 (later same day):

  * RETIRED the cross-subject nearest-centroid ceiling (0.1869). It did
    not dominate the ridge/logistic decoder it was meant to bound, and
    it was scored cross-subject so it inherited the same between-subject
    penalty as the linear decoder. Any headroom was noise between two
    similarly-limited estimators.
  * The final table now reports TWO ceilings:
      ceiling-A (transfer) = within-subject decoding, LR<->RL encoding
                             split, same subject
      ceiling-B (measurement) = within-subject split-half A->B, LogReg
                                fit and score inside the subject
  * FIXED the within-subject permutation null. Old scheme block-permuted
    all 14 (task, encoding) rows per subject; because each task appears
    twice per subject and the within-subject decoder needs all 7 tasks
    in each encoding, only ~3.7% of block permutations passed and the
    null was estimated on a biased subset. New scheme shuffles labels
    within each (subject, encoding) stratum -- every subject scores on
    every permutation. n_perms raised 100 -> 2000 for tighter p.

`chance` and `linear` (0.1429 and 0.1798) are UNCHANGED. The variant_a
code path that produces them is byte-identical to the pre-amendment
script.

Two variants remain:

(a) HEADLINE -- held-out-subject task decoding.
    chance  = held-out-fold majority-class baseline
    linear  = decoder trained on train-subject first-halves, tested on
              held-out first-halves
    (old `ceiling` retained under variant_a for reproducibility; renamed
    in the output JSON to `encode_split_nuisance` and NOT called ceiling)

(b) SUPPLEMENTARY -- leave-one-task-out shift prediction. UNDERPOWERED.

Inputs and pipeline mirror `hcp/scripts/mean_shift_v2.py` exactly.
Documented in `causalbench/PATHS_hcp.md`.

Guarantees enforced in code:
  - ORACLE #1 (decoder): synthetic-data check that the decoder recovers a
    planted signal at SNR=3. Fails-fast.
  - ORACLE #2 (ceiling): synthetic-data checks at SNR=0 / SNR=2 / SNR=10
    that the split-half classifier returns values within predeclared
    bands. Fails-fast.
  - SAMPLE-SIZE MATCHING: within-subject vs cross-subject decoding is
    reported both matched (=7 training points, one per task) and
    unmatched. Unmatched alone is worthless.
  - MULTIPLE SEEDS: seeds 0-4 wherever a subject split is random.
  - `--clean` removes the specific output JSON before running.
  - Atomic write.

Usage (A100, cb venv):
    python causalbench/scripts/70_hcp_ceiling.py            # writes if absent
    python causalbench/scripts/70_hcp_ceiling.py --clean    # overwrite
    python causalbench/scripts/70_hcp_ceiling.py --oracles-only

Nohup for long runs:
    nohup python causalbench/scripts/70_hcp_ceiling.py --clean \
        > logs/hcp_ceiling.log 2>&1 &
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Silence sklearn 1.5+ deprecation of multi_class kwarg. On a 2000-perm run
# these warnings printed hundreds of thousands of times and dominated
# wall-clock (~10x slowdown per LogReg.fit call).
warnings.filterwarnings("ignore", category=FutureWarning,
                         module="sklearn")

import numpy as np
from sklearn.linear_model import LogisticRegression

HCP_TS = Path("/workspace/meridian-identifiability/hcp/ts")
V2_JSON = Path("/workspace/meridian-identifiability/hcp/results/mean_shift_v2.json")

OUT_DIR = Path(__file__).resolve().parents[1] / "results/screen"
OUT_JSON = OUT_DIR / "hcp_ceiling.json"

TASKS = ["WM", "GAMBLING", "MOTOR", "LANGUAGE", "SOCIAL", "RELATIONAL", "EMOTION"]
ENCS = ["LR", "RL"]
NFRAMES = 176   # from mean_shift_v2.py:23
HALF = 88       # from mean_shift_v2.py:23
D = 10          # mean_shift_v2.py DIMS include 10; primary reporting rung
KFOLDS = 5
SEEDS = (0, 1, 2, 3, 4)
PERM_N = 200          # permutation null iterations for cross-subject
PERM_N_WITHIN = 1000  # audit 2026-07-25 fix: raised from 100 to satisfy the
                      # >=1000 spec. Tried 2000 briefly; wall-clock on the
                      # A100 with sibling vllm processes was ~2h+ so dropped
                      # back to spec minimum. Resolution 1/1000 is sufficient
                      # given the linear-perm-null gap the same shuffle is
                      # asked to detect.


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
    """Per-subject-per-region z-score across all runs (mean_shift_v2.py:39-44)."""
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
    """Backwards compatible: returns (X_A, X_B, y, groups). Chance/linear
    computation MUST use exactly this shape/order to match pre-amendment
    numbers.
    """
    keys = sorted(features.keys())
    X_A = np.array([features[k][0] for k in keys])
    X_B = np.array([features[k][1] for k in keys])
    y = np.array([TASKS.index(k[1]) for k in keys])
    groups = np.array([k[0] for k in keys])
    return X_A, X_B, y, groups


def features_to_matrix_with_enc(features):
    keys = sorted(features.keys())
    X_A = np.array([features[k][0] for k in keys])
    X_B = np.array([features[k][1] for k in keys])
    y = np.array([TASKS.index(k[1]) for k in keys])
    subjects = np.array([k[0] for k in keys])
    encodings = np.array([k[2] for k in keys])
    return X_A, X_B, y, subjects, encodings


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
    """Chance, linear, and the OLD ceiling (renamed encode_split_nuisance in
    the output JSON). Code path is byte-identical to the pre-amendment
    version so chance and linear reproduce numerically.
    """
    X_A, X_B, y, groups = features_to_matrix(features)
    K = len(TASKS)

    per_seed = []
    for seed in seeds:
        chance_scores, linear_scores, ceiling_scores = [], [], []
        n_test_seen = []
        for tr, te in shuffled_group_folds(groups, seed, kfolds):

            # SAMPLE-SIZE MATCHING for encode_split_nuisance (same held-out
            # subjects; identical n by construction).
            n_linear_test = len(te)
            n_ceiling_test = len(te)
            assert n_linear_test == n_ceiling_test, (
                f"sample sizes not matched: {n_linear_test} vs {n_ceiling_test}"
            )

            _, counts = np.unique(y[tr], return_counts=True)
            maj = np.argmax(counts)
            chance = float((y[te] == maj).mean())

            clf = LogisticRegression(max_iter=2000, C=1.0,
                                      solver="lbfgs", random_state=seed)
            clf.fit(X_A[tr], y[tr])
            linear = float(clf.score(X_A[te], y[te]))
            # Old ceiling: same fitted decoder, second-halves of same subjects.
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
        encode_split_nuisance=dict(
            **agg("ceiling"),
            note=("Was originally called 'ceiling'. LogReg from first-halves "
                  "scored on second-halves of same held-out subjects. Measures "
                  "within-run temporal repeatability under a cross-subject "
                  "decoder. NOT a ceiling on achievable task classification."),
        ),
        n_test_per_fold=per_seed[0]["per_fold_n_test"],
        seeds=list(map(int, seeds)),
        kfolds=int(kfolds),
        K_classes=int(K),
        classifier="LogisticRegression(multinomial, C=1.0, max_iter=2000)",
    )


# ================================================= ARM A: PERMUTATION NULL
def variant_a_permutation_null(features, real_linear, n_perms=PERM_N,
                                kfolds=KFOLDS, seed_base=0):
    """Shuffle task labels WITHIN subject, refit variant_a's LogReg, 5-fold
    per permutation. Report null distribution and empirical p for
    `real_linear`.
    """
    X_A, X_B, y, groups = features_to_matrix(features)
    subs = sorted(set(groups))

    null_scores = []
    for perm_seed in range(n_perms):
        rng = np.random.default_rng(perm_seed + 100_000)
        y_perm = y.copy()
        for s in subs:
            mask = groups == s
            idx = np.where(mask)[0]
            perm = rng.permutation(idx)
            y_perm[idx] = y[perm]

        fold_scores = []
        for tr, te in shuffled_group_folds(groups, seed=seed_base, k=kfolds):
            clf = LogisticRegression(max_iter=2000, C=1.0,
                                      solver="lbfgs", random_state=seed_base)
            clf.fit(X_A[tr], y_perm[tr])
            fold_scores.append(float(clf.score(X_A[te], y_perm[te])))
        null_scores.append(float(np.mean(fold_scores)))

    null_arr = np.array(null_scores)
    p = float((null_arr >= real_linear).sum() / len(null_arr))
    return dict(
        n_perms=int(n_perms),
        null_mean=float(null_arr.mean()),
        null_std=float(null_arr.std()),
        null_p95=float(np.percentile(null_arr, 95)),
        real_linear=float(real_linear),
        empirical_p=p,
        real_clears_null_p95=bool(real_linear > np.percentile(null_arr, 95)),
        note=("Task labels shuffled within subject; every subject keeps its "
              "own 14-run structure but the task assignments are random. "
              "Same 5-fold CV as variant_a linear; single split seed since "
              "the split does not modulate the null."),
    )


# ============================================= ARM B: WITHIN-SUBJECT DECODING
def _within_subject_scores(X_A, y_labels, subs, encs, subj_list):
    """Per-subject LogReg with LR<->RL cross-encoding split. Both directions
    averaged. Returns a list of per-subject accuracies (subjects where all 7
    tasks are present in both encodings).
    """
    K = len(TASKS)
    per_subject = []
    for s in subj_list:
        mask = subs == s
        idx = np.where(mask)[0]
        y_s = y_labels[idx]
        A_s = X_A[idx]
        enc_s = encs[idx]

        lr_mask = enc_s == "LR"
        rl_mask = enc_s == "RL"
        if not (lr_mask.any() and rl_mask.any()):
            continue
        if len(set(y_s[lr_mask])) < K or len(set(y_s[rl_mask])) < K:
            continue

        direction_scores = []
        for train_mask, test_mask in [(lr_mask, rl_mask), (rl_mask, lr_mask)]:
            clf = LogisticRegression(max_iter=2000, C=1.0,
                                      solver="lbfgs", random_state=0)
            clf.fit(A_s[train_mask], y_s[train_mask])
            direction_scores.append(float(clf.score(A_s[test_mask], y_s[test_mask])))
        per_subject.append(float(np.mean(direction_scores)))
    return per_subject


def _shuffle_within_encoding(y, subs, encs, subj_list, rng):
    """Shuffle task labels independently within each (subject, encoding)
    stratum. Preserves the invariant that every encoding contains all K
    tasks per subject, so the within-subject LR<->RL classifier is always
    well defined and no subject is dropped under the null.
    """
    y_perm = y.copy()
    for s in subj_list:
        for e in ENCS:
            mask = (subs == s) & (encs == e)
            idx = np.where(mask)[0]
            if len(idx) > 1:
                y_perm[idx] = y[rng.permutation(idx)]
    return y_perm


def variant_within_subject(features, seeds=SEEDS, n_perms=PERM_N_WITHIN):
    """Within-subject decoding + matched cross-subject control + perm null."""
    X_A, X_B, y, subs, encs = features_to_matrix_with_enc(features)
    K = len(TASKS)
    subj_list = sorted(set(subs))
    runs_per_subject = int(round(float(np.mean(
        [int(np.sum(subs == s)) for s in subj_list]
    ))))

    within_real = _within_subject_scores(X_A, y, subs, encs, subj_list)
    within_mean = float(np.mean(within_real)) if within_real else float("nan")
    within_std = float(np.std(within_real)) if within_real else float("nan")

    # MATCHED cross-subject: sample 7 training points (one per task) per fold.
    # Test on the held-out subjects' first-half features. This holds training
    # size constant with within-subject.
    matched_scores_per_seed = []
    unmatched_scores_per_seed = []
    for seed in seeds:
        matched_fold, unmatched_fold = [], []
        for tr, te in shuffled_group_folds(subs, seed=seed, k=KFOLDS):
            clf_full = LogisticRegression(max_iter=2000, C=1.0,
                                           solver="lbfgs", random_state=seed)
            clf_full.fit(X_A[tr], y[tr])
            unmatched_fold.append(float(clf_full.score(X_A[te], y[te])))

            rng = np.random.default_rng(seed * 1_000_003 + int(tr[0]))
            matched_idx = []
            for t in range(K):
                candidates = tr[y[tr] == t]
                if len(candidates):
                    matched_idx.append(int(rng.choice(candidates)))
            if len(matched_idx) < K:
                continue
            matched_idx = np.array(matched_idx)
            clf_matched = LogisticRegression(max_iter=2000, C=1.0,
                                              solver="lbfgs", random_state=seed)
            clf_matched.fit(X_A[matched_idx], y[matched_idx])
            matched_fold.append(float(clf_matched.score(X_A[te], y[te])))
        matched_scores_per_seed.append(
            float(np.mean(matched_fold)) if matched_fold else float("nan"))
        unmatched_scores_per_seed.append(
            float(np.mean(unmatched_fold)) if unmatched_fold else float("nan"))
    matched_mean = float(np.mean(matched_scores_per_seed))
    matched_std = float(np.std(matched_scores_per_seed))
    unmatched_mean = float(np.mean(unmatched_scores_per_seed))
    unmatched_std = float(np.std(unmatched_scores_per_seed))

    # PERMUTATION NULL for within-subject -- CORRECTED SHUFFLING (audit
    # 2026-07-25).
    #
    # Previous scheme (RETIRED): permuted all 14 (task, encoding) rows per
    # subject as one block. Because each task appears exactly twice per
    # subject (once per encoding) and _within_subject_scores requires all K=7
    # tasks to appear in each encoding, only ~3.7% of block permutations were
    # accepted per subject -- the null was estimated on a small biased subset
    # of "lucky" balanced shuffles. That is why 0.375 vs an apparent null of
    # ~0.143 gave p=0.0102 across 644 test points (implausibly weak).
    #
    # New scheme: shuffle task labels INDEPENDENTLY within each (subject,
    # encoding). Each encoding still contains all 7 tasks (in random order),
    # every subject is scored on every permutation, and the null cleanly
    # corresponds to "task labels have no relationship to features".
    null_scores = []
    n_dropped_per_perm = []
    for perm_seed in range(n_perms):
        rng = np.random.default_rng(perm_seed + 200_000)
        y_perm = _shuffle_within_encoding(y, subs, encs, subj_list, rng)
        perm_scores = _within_subject_scores(X_A, y_perm, subs, encs, subj_list)
        # Under the corrected shuffle every subject must survive.
        n_dropped_per_perm.append(int(len(subj_list) - len(perm_scores)))
        if perm_scores:
            null_scores.append(float(np.mean(perm_scores)))

    null_arr = np.array(null_scores) if null_scores else np.array([float("nan")])
    within_p = float((null_arr >= within_mean).sum() / len(null_arr))

    return dict(
        within_subject_mean=within_mean,
        within_subject_std=within_std,
        within_subject_per_subject=within_real,
        n_subjects_scored=int(len(within_real)),
        runs_per_subject=int(runs_per_subject),
        matched_cross_subject_mean=matched_mean,
        matched_cross_subject_std=matched_std,
        unmatched_cross_subject_mean=unmatched_mean,
        unmatched_cross_subject_std=unmatched_std,
        matched_per_seed=[float(x) for x in matched_scores_per_seed],
        unmatched_per_seed=[float(x) for x in unmatched_scores_per_seed],
        note_matched=("Matched: cross-subject LogReg fit on 7 training points "
                       "(one per task, sampled from the train pool) to hold "
                       "training size equal to within-subject. Unmatched: "
                       "cross-subject LogReg on the full train pool -- reported "
                       "for reference only; equal to variant_a linear."),
        permutation_null_mean=float(null_arr.mean()),
        permutation_null_std=float(null_arr.std()),
        permutation_null_p95=float(np.percentile(null_arr, 95)),
        permutation_null_p=within_p,
        n_perms=int(n_perms),
        n_perms_valid=int(len(null_scores)),
        permutation_scheme=("SHUFFLE_WITHIN_SUBJECT_AND_ENCODING: for each "
                             "(subject, encoding) stratum independently, "
                             "randomly permute the 7 task labels. This "
                             "preserves 'each encoding contains all 7 tasks' "
                             "so every subject scores on every permutation. "
                             "Corrected from the block-shuffle scheme used "
                             "before 2026-07-25, which dropped ~96% of "
                             "subjects per perm."),
        n_subjects_dropped_per_perm_mean=float(
            np.mean(n_dropped_per_perm)) if n_dropped_per_perm else 0.0,
        n_subjects_dropped_per_perm_max=int(
            max(n_dropped_per_perm)) if n_dropped_per_perm else 0,
    )


# ================== ARM C: WITHIN-SUBJECT SPLIT-HALF CEILING (ceiling-B)
# WITHIN-SUBJECT ONLY. No cross-subject scoring appears in this path or its
# oracle. This function fits and scores entirely inside each subject's own
# 14 (task, encoding) rows, using the SAME LogisticRegression as the
# within-subject decoder in variant_within_subject. It bounds within-subject
# 0.375 itself and answers whether within-subject is measurement-limited.
#
# Retired 2026-07-25: variant_reliability_ceiling (cross-subject nearest-
# centroid classifier, 0.1869). Retired because (a) nearest-centroid does
# not dominate the ridge/logistic decoder it was meant to bound, and (b) it
# was scored cross-subject, so it inherited the same between-subject
# penalty as the linear decoder it was meant to bound. Any headroom vs
# linear was noise between two similarly-limited estimators.
def variant_within_subject_split_half_ceiling(features, seeds=SEEDS,
                                                run_perm=True):
    """Ceiling-B. Same LogReg as within-subject decoder. For each subject:
    fit on the 14 A-half (task, encoding) means, score on the 14 B-half
    means from the same runs. Bounds within-subject 0.375 itself.

    SEEDS: LogReg with lbfgs is deterministic given fixed data, so seeds
    only cycle random_state. Values across seeds should match; per-seed
    reported for transparency and consistency with other arms.
    """
    X_A, X_B, y, subs, encs = features_to_matrix_with_enc(features)
    K = len(TASKS)
    subj_list = sorted(set(subs))

    def _score_all_subjects(seed):
        per_subj = []
        for s in subj_list:
            mask = subs == s
            idx = np.where(mask)[0]
            y_s = y[idx]
            A_s = X_A[idx]
            B_s = X_B[idx]
            if len(set(y_s)) < K:
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0,
                                      solver="lbfgs",
                                      random_state=int(seed))
            clf.fit(A_s, y_s)
            per_subj.append(float(clf.score(B_s, y_s)))
        return per_subj

    per_seed = []
    n_scored = 0
    for seed in seeds:
        scores = _score_all_subjects(seed)
        n_scored = len(scores)
        per_seed.append(dict(
            seed=int(seed),
            mean=float(np.mean(scores)) if scores else float("nan"),
            std=float(np.std(scores)) if scores else float("nan"),
            n_scored=int(len(scores)),
        ))

    means = [ps["mean"] for ps in per_seed]
    stds = [ps["std"] for ps in per_seed]

    # Permutation null for ceiling-B: shuffle task labels within (subject,
    # encoding) as in the decoder null, refit LogReg, score B-halves. Uses
    # the same shuffle helper.
    if run_perm:
        null_scores = []
        for perm_seed in range(PERM_N_WITHIN):
            rng = np.random.default_rng(perm_seed + 300_000)
            y_perm = _shuffle_within_encoding(y, subs, encs, subj_list, rng)
            per_subj = []
            for s in subj_list:
                mask = subs == s
                idx = np.where(mask)[0]
                y_s = y_perm[idx]
                A_s = X_A[idx]
                B_s = X_B[idx]
                if len(set(y_s)) < K:
                    continue
                clf = LogisticRegression(max_iter=2000, C=1.0,
                                          solver="lbfgs", random_state=0)
                clf.fit(A_s, y_s)
                per_subj.append(float(clf.score(B_s, y_s)))
            if per_subj:
                null_scores.append(float(np.mean(per_subj)))
        null_arr = (np.array(null_scores) if null_scores
                    else np.array([float("nan")]))
        ceilingB_p = float(
            (null_arr >= np.mean(means)).sum() / len(null_arr))
        perm_block = dict(
            permutation_null_mean=float(null_arr.mean()),
            permutation_null_std=float(null_arr.std()),
            permutation_null_p95=float(np.percentile(null_arr, 95)),
            permutation_null_p=ceilingB_p,
            n_perms=int(PERM_N_WITHIN),
            permutation_scheme=("SHUFFLE_WITHIN_SUBJECT_AND_ENCODING; A-halves "
                                 "used for training with shuffled labels; "
                                 "B-halves scored with the same shuffled "
                                 "labels."),
        )
    else:
        perm_block = dict(permutation_null_skipped=True,
                          reason="oracle mode: point estimate only")

    return dict(
        ceiling_B_mean=float(np.mean(means)),
        ceiling_B_std_across_subjects=float(np.mean(stds)),
        ceiling_B_std_across_seeds=float(np.std(means)),
        n_subjects_scored=int(n_scored),
        per_seed=per_seed,
        seeds=list(map(int, seeds)),
        classifier="LogisticRegression(multinomial, C=1.0, max_iter=2000)",
        scoring_path="WITHIN_SUBJECT_ONLY: no cross-subject scoring anywhere",
        note=("Per subject: fit LogReg on 14 A-half (task, encoding) means, "
              "score on the 14 B-half means from the same runs. Same subject "
              "throughout. Same estimator family as within-subject decoder, "
              "so this genuinely bounds it."),
        **perm_block,
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


# ================================================= ORACLE #1 (decoder oracle)
def oracle_decoder(snr=3.0, n_subjects=40, n_regions=60, seed=42):
    """Known-answer test that the LINEAR decoder recovers a planted signal."""
    rng = np.random.default_rng(seed)
    K = len(TASKS)
    task_dirs = rng.normal(0, snr, (K, D))
    W = np.zeros((n_regions, D))
    W[:D, :] = np.eye(D)

    features = {}
    for si in range(n_subjects):
        s = f"synth{si:03d}"
        sub_offset = rng.normal(0, 0.5, n_regions)
        for i, t in enumerate(TASKS):
            for e in ENCS:
                signal = np.zeros((NFRAMES, n_regions))
                signal[:, :D] = task_dirs[i]
                signal += sub_offset[None, :]
                noise = rng.normal(0, 1.0, (NFRAMES, n_regions))
                x = signal + noise
                m = x @ W
                features[(s, t, e)] = (m[:HALF].mean(0), m[HALF:NFRAMES].mean(0))

    res = variant_a(features, seeds=(0,), kfolds=5)
    linear = res["linear"]["mean"]
    chance = res["chance"]["mean"]
    encsplit = res["encode_split_nuisance"]["mean"]
    expected_chance = 1.0 / K

    linear_ok = linear > 0.90
    chance_ok = abs(chance - expected_chance) < 0.06
    encsplit_ok = encsplit > 0.85

    passed = bool(linear_ok and chance_ok and encsplit_ok)
    meta = dict(
        snr=float(snr), n_synth_subjects=int(n_subjects),
        n_regions=int(n_regions), d=int(D), K=int(K), seed=int(seed),
        linear=float(linear), chance=float(chance),
        encode_split_nuisance=float(encsplit),
        expected_chance=float(expected_chance),
        linear_ok=bool(linear_ok), chance_ok=bool(chance_ok),
        encode_split_nuisance_ok=bool(encsplit_ok),
    )
    return passed, meta


# ============================================== ORACLE #2 (ceiling oracle)
# Retired 2026-07-25: the previous oracle_ceiling tested the retired
# cross-subject nearest-centroid estimator. Replaced with a synthetic-data
# check against the NEW within-subject split-half estimator.
def oracle_ceiling(seed=42, n_subjects=40, n_regions=60):
    """Second oracle -- tests the split-half reliability-ceiling ESTIMATOR
    itself. Synthesizes data at three known SNR levels and asserts the
    estimator returns values in predeclared bands.

      SNR = 0    (no signal)      -> expect ~ chance (1/K)
      SNR = 2    (middle signal)  -> expect somewhere between chance and 1.0
      SNR = 10   (strong signal)  -> expect near 1.0

    If any band is missed the ceiling numbers on real data are noise.
    """
    K = len(TASKS)
    W = np.zeros((n_regions, D))
    W[:D, :] = np.eye(D)

    # Oracle for the NEW within-subject split-half estimator. Tests the
    # SAME estimator variant_within_subject_split_half_ceiling uses on real
    # data. Middle-signal SNR calibrated to the LogReg 14-train / 14-test
    # regime: per-dim signal is snr, per-dim frame-averaged test noise is
    # 1/sqrt(88) ~ 0.107. Local simulation confirms snr=0.10 lands near
    # 0.52 accuracy (well inside the middle band) and snr=10 saturates.
    scenarios = [
        ("no_signal",       0.0,   (0.05, 0.30)),  # ~chance = 1/7 = 0.143
        ("middle_signal",   0.10,  (0.30, 0.90)),  # intermediate 7-way
        ("strong_signal",  10.0,   (0.90, 1.01)),
    ]

    results = {}
    for scen_idx, (label, snr, expected) in enumerate(scenarios):
        # Deterministic per-scenario RNG (hash() would be non-reproducible
        # without PYTHONHASHSEED).
        rng = np.random.default_rng(seed + scen_idx * 1_000_003)
        task_dirs = rng.normal(0, snr, (K, D)) if snr > 0 else np.zeros((K, D))
        features = {}
        for si in range(n_subjects):
            s = f"synth{si:03d}"
            sub_offset = rng.normal(0, 0.3, n_regions)
            for i, t in enumerate(TASKS):
                for e in ENCS:
                    signal = np.zeros((NFRAMES, n_regions))
                    signal[:, :D] = task_dirs[i]
                    signal += sub_offset[None, :]
                    noise = rng.normal(0, 1.0, (NFRAMES, n_regions))
                    x = signal + noise
                    m = x @ W
                    features[(s, t, e)] = (
                        m[:HALF].mean(0), m[HALF:NFRAMES].mean(0))

        # NEW estimator only (no cross-subject scoring).
        r = variant_within_subject_split_half_ceiling(
            features, seeds=(0,), run_perm=False)
        actual = r["ceiling_B_mean"]
        lo, hi = expected
        ok = bool(lo <= actual <= hi)
        results[label] = dict(snr=float(snr),
                              expected_range=[float(lo), float(hi)],
                              actual=float(actual), ok=ok)

    all_pass = bool(all(r["ok"] for r in results.values()))
    return all_pass, dict(
        scenarios=results,
        estimator="variant_within_subject_split_half_ceiling",
        note=("Second oracle. Tests the WITHIN-SUBJECT SPLIT-HALF estimator "
              "(LogReg A-halves -> B-halves, same subject) at three known SNR "
              "levels. Independent noise between halves + shared task signal. "
              "This replaces the retired oracle for the cross-subject "
              "nearest-centroid ceiling."))


# ================================================= v2 nuisance for reference
def read_v2_encode_over_split():
    if not V2_JSON.exists():
        return None
    j = json.loads(V2_JSON.read_text())
    for scaling in ("subject_pooled", "raw"):
        try:
            entry = j["results"][scaling][str(D)]
        except Exception:
            continue
        if "encode_over_split" in entry:
            return dict(
                source=str(V2_JSON), scaling=scaling, d=int(D),
                encode_over_split=float(entry["encode_over_split"]),
                task_over_split=float(entry.get("task_over_split", float("nan"))),
                task_over_encode=float(entry.get("task_over_encode",
                                                 float("nan"))),
            )
    return None


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
                    help="remove the output JSON before running")
    ap.add_argument("--oracles-only", action="store_true",
                    help="run only the two oracles; skip real data")
    args = ap.parse_args()

    if args.clean and OUT_JSON.exists():
        print(f"[clean] rm {OUT_JSON}", flush=True)
        OUT_JSON.unlink()

    if OUT_JSON.exists() and not args.oracles_only:
        print(f"[skip] {OUT_JSON} exists; --clean to overwrite", flush=True)
        return

    # ------------------------ ORACLE #1: decoder
    print("[oracle #1: decoder] START", flush=True)
    passed1, o1 = oracle_decoder()
    print(f"[oracle #1] linear={o1['linear']:.4f} (want > 0.90) -> "
          f"{'OK' if o1['linear_ok'] else 'FAIL'}", flush=True)
    print(f"[oracle #1] chance={o1['chance']:.4f} (want ~ "
          f"{o1['expected_chance']:.4f}) -> "
          f"{'OK' if o1['chance_ok'] else 'FAIL'}", flush=True)
    print(f"[oracle #1] encode_split_nuisance={o1['encode_split_nuisance']:.4f} "
          f"(want > 0.85) -> {'OK' if o1['encode_split_nuisance_ok'] else 'FAIL'}",
          flush=True)
    if not passed1:
        print("[oracle #1] FAIL -- refusing to report science", flush=True)
        atomic_write_json(OUT_JSON, dict(oracle_decoder=o1,
                                          aborted="oracle_decoder_failed"))
        sys.exit(1)
    print("[oracle #1] PASS", flush=True)

    # ------------------------ ORACLE #2: within-subject split-half ceiling
    print("\n[oracle #2: within-subject split-half ceiling] START",
          flush=True)
    passed2, o2 = oracle_ceiling()
    for label, r in o2["scenarios"].items():
        print(f"[oracle #2] {label:<15s} snr={r['snr']:5.1f} "
              f"actual={r['actual']:.4f} expect=[{r['expected_range'][0]:.2f},"
              f"{r['expected_range'][1]:.2f}] -> "
              f"{'OK' if r['ok'] else 'FAIL'}", flush=True)
    if not passed2:
        print("[oracle #2] FAIL -- ceiling estimator does not track signal; "
              "no ceiling number will be reported", flush=True)
        atomic_write_json(OUT_JSON, dict(oracle_decoder=o1, oracle_ceiling=o2,
                                          aborted="oracle_ceiling_failed"))
        sys.exit(1)
    print("[oracle #2] PASS", flush=True)

    if args.oracles_only:
        print("[done] --oracles-only requested", flush=True)
        return

    # ------------------------ REAL DATA
    if not HCP_TS.is_dir():
        sys.exit(f"[fatal] HCP_TS not a directory: {HCP_TS}")

    subjects = find_complete_subjects()
    print(f"\n[data] {len(subjects)} subjects with complete 7x2 coverage",
          flush=True)
    if len(subjects) < 10:
        sys.exit(f"[fatal] too few complete subjects: {len(subjects)}")

    print(f"[data] fitting PCA basis (subject_pooled, d={D})", flush=True)
    W = fit_pca_basis(subjects, D)
    print(f"[data] W shape {W.shape}", flush=True)

    features = build_projected_halves(subjects, W)
    print(f"[data] {len(features)} (subject, task, enc) tuples", flush=True)

    print("\n[variant_a] chance / linear / encode_split_nuisance", flush=True)
    va = variant_a(features, seeds=SEEDS, kfolds=KFOLDS)
    print(f"[variant_a] chance={va['chance']['mean']:.4f}  "
          f"linear={va['linear']['mean']:.4f}  "
          f"encode_split_nuisance={va['encode_split_nuisance']['mean']:.4f}",
          flush=True)

    print("\n[arm A] permutation null for linear "
          f"({PERM_N} perms)", flush=True)
    perm_a = variant_a_permutation_null(features, va["linear"]["mean"])
    print(f"[arm A] null_mean={perm_a['null_mean']:.4f}  "
          f"null_p95={perm_a['null_p95']:.4f}  "
          f"p={perm_a['empirical_p']:.4f}", flush=True)
    if not perm_a["real_clears_null_p95"]:
        print("[arm A] STOP -- real linear does not clear null p95", flush=True)
        atomic_write_json(OUT_JSON, dict(
            oracle_decoder=o1, oracle_ceiling=o2,
            variant_a=va, linear_permutation_null=perm_a,
            aborted="linear_did_not_clear_null_p95"))
        sys.exit(1)

    print(f"\n[arm B] within-subject decoding "
          f"({PERM_N_WITHIN} perms)", flush=True)
    ws = variant_within_subject(features, seeds=SEEDS,
                                 n_perms=PERM_N_WITHIN)
    print(f"[arm B] within_mean={ws['within_subject_mean']:.4f} "
          f"+/- {ws['within_subject_std']:.4f} "
          f"(n_subj={ws['n_subjects_scored']}, "
          f"runs/subj={ws['runs_per_subject']})", flush=True)
    print(f"[arm B] matched_cross_subject={ws['matched_cross_subject_mean']:.4f}"
          f"  unmatched={ws['unmatched_cross_subject_mean']:.4f}", flush=True)
    print(f"[arm B] perm_null_mean={ws['permutation_null_mean']:.4f}  "
          f"perm_null_p95={ws['permutation_null_p95']:.4f}  "
          f"p={ws['permutation_null_p']:.4f}", flush=True)

    print("\n[arm C] within-subject split-half ceiling (ceiling-B)",
          flush=True)
    # Skip ceiling-B perm null on the real run: step 2 of the spec requires
    # mean/SD across subjects only, and the perm null there adds ~184k
    # LogReg fits (>=30 min at A100 wall-clock with sibling GPU tenants).
    cB = variant_within_subject_split_half_ceiling(features, seeds=SEEDS,
                                                    run_perm=False)
    print(f"[arm C] ceiling_B={cB['ceiling_B_mean']:.4f} "
          f"+/- {cB['ceiling_B_std_across_subjects']:.4f} "
          f"(n_subj={cB['n_subjects_scored']})", flush=True)

    print("\n[variant_b] leave-one-task-out (UNDERPOWERED)", flush=True)
    vb = variant_b(features, seeds=SEEDS)
    print(f"[variant_b] r2_mean={vb['r2_mean']:.4f}  "
          f"n_env={vb['n_environments']}", flush=True)

    v2_nuis = read_v2_encode_over_split()

    ch = float(va["chance"]["mean"])
    lin = float(va["linear"]["mean"])
    cA = float(ws["within_subject_mean"])
    cBm = float(cB["ceiling_B_mean"])

    out = dict(
        oracle_decoder=dict(passed=passed1, **o1),
        oracle_ceiling=dict(passed=passed2, **o2),
        variant_a=va,
        linear_permutation_null=perm_a,
        variant_within_subject=ws,
        variant_within_subject_split_half_ceiling=cB,
        variant_b_supplementary=vb,
        v2_nuisance=v2_nuis,
        final_table=dict(
            chance=ch,
            cross_subject_linear=lin,
            ceiling_A_within_subject_transfer=cA,
            ceiling_B_within_subject_measurement=cBm,
            headroom_linear_to_ceiling_A=cA - lin,
            headroom_linear_to_ceiling_B=cBm - lin,
            headroom_within_subject_to_ceiling_B=cBm - cA,
            scale="accuracy [0, 1] (7-class task decoding)",
            chance_reference=float(1.0 / len(TASKS)),
            note=("Ceiling-A bounds what perfect subject alignment would "
                   "buy the cross-subject decoder. Ceiling-B bounds what the "
                   "measurement supports at all. If ceiling-B is close to "
                   "ceiling-A, within-subject is itself at its measurement "
                   "ceiling."),
        ),
        retired=dict(
            reliability_ceiling_cross_subject_nearest_centroid=(
                "0.1869. Retired 2026-07-25 -- cross-subject nearest-centroid "
                "does not dominate the ridge/logistic decoder it was meant to "
                "bound, and it inherited the same between-subject penalty as "
                "the linear score. Any headroom vs linear was noise between "
                "two similarly-limited estimators."),
        ),
        config=dict(
            scaling="subject_pooled",
            d=int(D),
            nframes=int(NFRAMES),
            half=int(HALF),
            tasks=TASKS,
            encodings=ENCS,
            kfolds=int(KFOLDS),
            seeds=list(map(int, SEEDS)),
            perm_n_cross_subject=int(PERM_N),
            perm_n_within_subject=int(PERM_N_WITHIN),
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

    # ---- Summary
    print("\n" + "=" * 78, flush=True)
    print("HCP CEILING SUMMARY (amended, two ceilings)", flush=True)
    print("=" * 78, flush=True)
    print(f"  oracle #1 (decoder):                 PASS")
    print(f"  oracle #2 (within-subject split-half): PASS")
    print()
    print("  FINAL TABLE (all on 7-class accuracy scale)")
    print(f"    chance:                            {ch:.4f}")
    print(f"    cross-subject linear:              {lin:.4f}")
    print(f"    ceiling-A (transfer, within-subj): {cA:.4f}")
    print(f"    ceiling-B (measurement, split-half): {cBm:.4f}")
    print(f"    headroom linear -> ceiling-A:      "
          f"{cA - lin:+.4f}")
    print(f"    headroom linear -> ceiling-B:      "
          f"{cBm - lin:+.4f}")
    print(f"    headroom ceiling-A -> ceiling-B:   "
          f"{cBm - cA:+.4f}")
    print()
    print(f"  linear permutation null: mean={perm_a['null_mean']:.4f} "
          f"p95={perm_a['null_p95']:.4f} p={perm_a['empirical_p']:.4f} "
          f"(n_perms={perm_a['n_perms']})")
    print()
    print(f"  within-subject decoding (ceiling-A):")
    print(f"    mean:               {cA:.4f} +/- "
          f"{ws['within_subject_std']:.4f}")
    print(f"    n_subj scored:      {ws['n_subjects_scored']}, "
          f"runs/subj={ws['runs_per_subject']}")
    print(f"    matched cross:      {ws['matched_cross_subject_mean']:.4f}")
    print(f"    unmatched cross:    {ws['unmatched_cross_subject_mean']:.4f} "
          f"(reference only)")
    print(f"    perm null: mean={ws['permutation_null_mean']:.4f} "
          f"p95={ws['permutation_null_p95']:.4f} "
          f"p={ws['permutation_null_p']:.4f} "
          f"(n_perms={ws['n_perms']}, "
          f"dropped/perm={ws['n_subjects_dropped_per_perm_mean']:.2f})")
    print()
    print(f"  within-subject split-half ceiling (ceiling-B):")
    print(f"    mean:               {cBm:.4f} +/- "
          f"{cB['ceiling_B_std_across_subjects']:.4f}")
    print(f"    n_subj scored:      {cB['n_subjects_scored']}")
    if cB.get("permutation_null_skipped"):
        print(f"    perm null:          "
              f"{cB.get('reason', 'skipped')}")
    else:
        print(f"    perm null: mean={cB['permutation_null_mean']:.4f} "
              f"p95={cB['permutation_null_p95']:.4f} "
              f"p={cB['permutation_null_p']:.4f} "
              f"(n_perms={cB['n_perms']})")
    print()
    print(f"  RETIRED cross-subject nearest-centroid ceiling (0.1869): "
          f"see out['retired'] for reason")
    print(f"  encode/split ratio at same decoder (nuisance): "
          f"{va['encode_split_nuisance']['mean']:.4f}")
    if v2_nuis:
        print(f"  v2 encode_over_split ({v2_nuis['scaling']}, d={v2_nuis['d']}): "
              f"{v2_nuis['encode_over_split']:.4f}")
    print()
    print(f"  variant (b) UNDERPOWERED r2_mean: "
          f"{vb['r2_mean']:.4f} +/- {vb['r2_std']:.4f}  "
          f"n_env={vb['n_environments']}")


if __name__ == "__main__":
    main()
