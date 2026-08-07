"""Rank diagnostic for the multi-environment CRL assumption bundle.

WHAT IS BEING TESTED
--------------------
For each environment e, against the observational/control environment 0:

    H0(k):  rank( Sigma_e - Sigma_0 ) <= k,   with k = 2

Under the assumption bundle behind multi-environment CRL identifiability
theorems -- linear-Gaussian latent SCM, linear mixing, a SINGLE-NODE
intervention, and a mixing map shared across environments -- the observed
covariance difference is a rank-<=2 update, and rank 1 when the intervened
node is a source.

Sketch, latent space. Sigma = M D M^T with M = (I - B)^-1.
A hard intervention on node i zeroes row i of B, so by Sherman-Morrison
M_e = M + u v^T with u proportional to M e_i -- a rank-1 update to M -- and
D_e = D + delta e_i e_i^T. Expanding M_e D_e M_e^T - M D M^T leaves a column
space contained in span{M D v, u}, hence rank <= 2. For a source node v = 0
and only the delta term survives, hence rank 1. Linear mixing X = A Z maps
this to A (Sigma_Z,e - Sigma_Z,0) A^T, which can only lose rank.

THE DECISION -- reject_rank2, not a count
-----------------------------------------
H0(2) says at most two eigenvalues of Delta are non-zero, so the test is the
SINGLE comparison lam[2] > band[2] on the third eigenvalue. One test, one
alpha, no multiplicity to control.

The earlier statistic summed exceedances over all d eigenvalues and called
r_hat > 2 a rejection. That was d marginal level-alpha tests with no
multiplicity control: its null distribution was approximately Binomial(d,
alpha) rather than a point mass at 0, so it rejected on pure control data
with nothing to detect at up to 0.09 at d=20. Gate 0 caught it. It survives
for one commit as r_hat_marginal_DEPRECATED purely so that Gate 0 diff is
auditable, and must not be used for anything.

r_hat_stepdown is the accompanying rank ESTIMATE, by step-down: the number of
leading eigenvalues that clear their band before the first one that does not.
It is a descriptive readout, not the decision.

INTERPRETATION RULE -- this is the whole point of the diagnostic
----------------------------------------------------------------
* reject_rank2 True is INFORMATIVE. It falsifies the bundle: at least one of
  linear-Gaussian latents, linear mixing, single-node intervention, or a
  shared mixing map fails on this data.
* reject_rank2 False is NOT evidence that the bundle holds. Covariance is
  mean-centred, so a pure shift intervention moves the mean and leaves the
  covariance alone, producing no rejection trivially. Only REJECTION carries
  a claim.
* This is the SECOND-ORDER COMPLEMENT to the mean-shift screen in
  03_screen.py, not a replacement for it. The shift blind spot above is
  exactly what the mean-shift screen sees and this does not. Neither
  subsumes the other; they are run together.

DEVIATIONS FROM THE HANDOFF SPEC -- read before using any number from this
--------------------------------------------------------------------------
(1) fit_pca / project SIGNATURES. The handoff assumed
        P = fit_pca(X_basis, d)   -> (D, d)      and    Y = X @ P
    The functions in 03_screen.py are actually
        fit_pca(Xc, d)            -> (mu, W),  W is (D, d)
        project(X, mu, W, drop)   -> (X - mu) @ W_masked
    The NAMES match the handoff, so per the handoff's own stop-rule this is
    not a stop condition; only the call sites are adapted. PCA is imported,
    never reimplemented. The mu subtraction is irrelevant to this diagnostic
    -- np.cov centres each sample itself, so a constant offset cancels from
    Delta -- but it is kept so the arithmetic stays byte-identical to the
    screen.

(2) n_match. Resolved by amendment to
        n_match = min(n_e, n_p // 3)
    which supersedes the handoff's contradictory pair of
    n_match = min(n_e, floor(n_p/2)) and "raise if n_p < 3 * n_match" (those
    two could never both hold: whenever n_e was large the first gave
    floor(n_p/2), and n_p >= 3*floor(n_p/2) is false for every n_p >= 2).
    The n_match ARGUMENT is consequently inert -- it is not part of the min.
    It stays in the signature for compatibility and is echoed back as
    n_match_requested so a caller can see the request was not honoured.

(3) drop=. Exposed as a keyword-only argument, default empty, so Phase B can
    apply the screen's targeted-gene masking consistently to the environment,
    the reference AND the null draws. The handoff's positional signature is
    unchanged. With drop empty this is a no-op.

(4) null_band=. Additive, keyword-only, arithmetic-identical optimisation.
    The band depends only on (X_ref_pool, mu, W, n_match, alpha, B_null) and
    NOT on X_env, so a caller sweeping many environments against one fixed
    control pool may compute the band once and pass it back in. Gate 0 needs
    this: 200 splits x 3 d x 10 seeds x 2 scalings would otherwise recompute
    an identical band 12,000 times. Callers that reuse a band MUST report
    B_null, because the 200 decisions then share that band's Monte-Carlo
    error and are no longer independent.

Pure functions. No I/O, no CLI, no global seeding: every draw takes an
explicit rng.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants
# See deviation (2) in the module docstring. The null needs two DISJOINT
# subsets of size n_match drawn from X_ref_pool; 3 leaves margin so that
# successive null draws are not forced to recycle the same cells.
REF_SPLIT_FACTOR = 3

_SCREEN_PY = Path(__file__).resolve().parent / "03_screen.py"


def _load_screen_module(path=_SCREEN_PY):
    """Import 03_screen.py for fit_pca/project. PCA is NEVER reimplemented.

    03_screen.py does os.makedirs("/workspace/.../results/screen") at module
    scope, which raises OSError on any machine that is not the A100. Only
    that side effect is neutralised, by swapping os.makedirs for a no-op for
    the duration of exec_module. No function body is touched, so fit_pca and
    project are byte-identical to the screen's.
    """
    spec = importlib.util.spec_from_file_location("_ranktest_screen03", str(path))
    mod = importlib.util.module_from_spec(spec)
    real_makedirs = os.makedirs
    os.makedirs = lambda *a, **k: None
    try:
        spec.loader.exec_module(mod)
    finally:
        os.makedirs = real_makedirs
    return mod


_SCREEN = _load_screen_module()
fit_pca = _SCREEN.fit_pca        # fit_pca(Xc, d) -> (mu, W),  W is (D, d)
project = _SCREEN.project        # project(X, mu, W, drop) -> (X - mu) @ W_masked


# ------------------------------------------------------------------ helpers
def _spectrum(Ya, Yb):
    """|eigenvalues| of cov(Ya) - cov(Yb), sorted descending. Ya, Yb are (n, d)."""
    if Ya.shape[0] != Yb.shape[0]:
        raise ValueError(
            f"numerator/denominator sample sizes differ: {Ya.shape[0]} vs "
            f"{Yb.shape[0]}; the covariance difference would be biased by the "
            f"n-dependent estimation noise alone"
        )
    Delta = np.cov(Ya, rowvar=False) - np.cov(Yb, rowvar=False)
    return np.sort(np.abs(np.linalg.eigvalsh(Delta)))[::-1]


def _two_disjoint(n_pool, n_match, rng):
    """Two index sets of size n_match into [0, n_pool), guaranteed disjoint."""
    take = rng.choice(n_pool, size=2 * n_match, replace=False)
    return take[:n_match], take[n_match:]


def null_band_from_pool(Yp, n_match, B_null, alpha, rng):
    """(1 - alpha) quantile of each sorted null eigenvalue.

    Yp is the ALREADY-PROJECTED control reference pool, (n_p, d). Projection
    is linear, so projecting a subset of rows equals subsetting the projected
    matrix -- this is an exact restatement, not an approximation.

    Each of the B_null draws splits the pool into two disjoint halves of size
    n_match and takes the spectrum of their covariance difference. That is
    the same statistic as the real comparison, computed where the true rank
    is 0 by construction.
    """
    n_p, d = Yp.shape
    lam_null = np.empty((B_null, d))
    for b in range(B_null):
        ia, ib = _two_disjoint(n_p, n_match, rng)
        # LEAK GATE, per null draw: the two halves must not share a cell.
        assert not (set(ia.tolist()) & set(ib.tolist())), (
            f"null draw {b}: the two reference subsets share cells"
        )
        lam_null[b] = _spectrum(Yp[ia], Yp[ib])
    return np.quantile(lam_null, 1.0 - alpha, axis=0)


# ------------------------------------------------------------- entry point
def rank_diagnostic(X_env, X_basis, X_ref_pool, d, n_match, B_null, alpha, rng,
                    *, drop=(), basis_idx=None, ref_pool_idx=None,
                    null_band=None):
    """Estimate rank(Sigma_e - Sigma_0) against a resampled null band.

    X_env      : (n_e, D) environment cells
    X_basis    : (n_a, D) control cells used ONLY to fit the projection
    X_ref_pool : (n_p, D) control cells used for the reference covariance and
                 the null. MUST be row-disjoint from X_basis.
    d          : projection dimension
    n_match    : REQUESTED matched sample size; clamped down to
                 min(n_match, n_e, n_p // REF_SPLIT_FACTOR)
    B_null     : null resampling draws (ignored when null_band is supplied)
    alpha      : per-eigenvalue upper-tail level
    rng        : np.random.Generator. Every draw uses it; nothing is seeded
                 globally anywhere in this module.

    Keyword-only, see deviations (3) and (4) in the module docstring:
    drop, basis_idx, ref_pool_idx, null_band.

    The projection is fitted on CONTROL CELLS ONLY and applied unchanged to
    the environment, the reference and every null draw. Projection can only
    reduce rank, never increase it, so a rejection in the projected space is
    a valid rejection in the full space. The converse does not hold: failing
    to reject in d dimensions says nothing about the other D - d.

    Returns a dict; see the keys assembled at the end of this function.
    """
    X_env = np.asarray(X_env)
    X_basis = np.asarray(X_basis)
    X_ref_pool = np.asarray(X_ref_pool)

    n_e = X_env.shape[0]
    n_p = X_ref_pool.shape[0]

    # ---- LEAK GATE: basis vs reference pool, on INDEX SETS, not on values.
    if basis_idx is not None and ref_pool_idx is not None:
        shared = set(np.asarray(basis_idx).tolist()) & set(np.asarray(ref_pool_idx).tolist())
        if shared:
            raise ValueError(
                f"leak: X_basis and X_ref_pool share {len(shared)} row indices "
                f"(e.g. {sorted(shared)[:5]}); the projection would be fitted "
                f"on the same cells that define the null"
            )

    # ---- sample-size matching. See deviation (2).
    # n_match = min(n_e, n_p // 3). The n_match ARGUMENT is not part of this
    # min and is therefore inert; it is kept in the signature for
    # compatibility and echoed back as n_match_requested so a caller can see
    # that its request was not honoured.
    n_match_eff = int(min(n_e, n_p // REF_SPLIT_FACTOR))
    if n_match_eff < 2:
        raise ValueError(
            f"n_match_eff={n_match_eff} < 2 (n_e={n_e}, n_p={n_p}, "
            f"requested n_match={n_match}); not enough cells to form a covariance"
        )
    if d < 3:
        raise ValueError(
            f"d={d} < 3: reject_rank2 tests the THIRD eigenvalue, which does "
            f"not exist below d=3"
        )
    if n_p < REF_SPLIT_FACTOR * n_match_eff:
        raise ValueError(
            f"reference pool too small: n_p={n_p} < {REF_SPLIT_FACTOR} * "
            f"n_match={n_match_eff}; refusing to silently reuse cells across "
            f"the null draws"
        )

    # ---- projection, fitted on the basis controls alone.
    mu, W = fit_pca(X_basis, d)

    # Project the pools ONCE. project() is linear in its rows, so subsetting
    # after projecting is identical to projecting a subset.
    Yp_all = project(X_ref_pool, mu, W, drop)
    Ye_all = project(X_env, mu, W, drop)

    # ---- matched draws for the observed comparison.
    env_take = rng.choice(n_e, size=n_match_eff, replace=False)
    ref_take = rng.choice(n_p, size=n_match_eff, replace=False)
    Y_e = Ye_all[env_take]
    Y_0 = Yp_all[ref_take]

    # Numerator and denominator must carry identical n.
    if Y_e.shape[0] != Y_0.shape[0]:
        raise AssertionError(
            f"matched-n violation: environment n={Y_e.shape[0]} but reference "
            f"n={Y_0.shape[0]}"
        )
    assert Y_e.shape[0] == n_match_eff and Y_0.shape[0] == n_match_eff

    lam = _spectrum(Y_e, Y_0)

    # ---- null band.
    band_reused = null_band is not None
    if band_reused:
        band = np.asarray(null_band, dtype=float)
        if band.shape != (d,):
            raise ValueError(f"null_band has shape {band.shape}, expected ({d},)")
    else:
        band = null_band_from_pool(Yp_all, n_match_eff, B_null, alpha, rng)

    # ---- statistic.
    exceed = lam > band
    # THE DECISION. H0(2) says at most two eigenvalues are non-zero, so the
    # test is the SINGLE comparison on the third one. One test, so its null
    # rate is alpha -- unlike the old count over all d eigenvalues, whose
    # null rate was ~1-(1-alpha)^d because it never controlled multiplicity.
    reject_rank2 = bool(lam[2] > band[2])

    # Rank estimate by step-down: walk the sorted spectrum from the top and
    # stop at the FIRST eigenvalue that fails to clear its band. Eigenvalues
    # past a non-exceedance do not contribute, which is what separates this
    # from the deprecated marginal count below.
    r_hat_stepdown = 0
    for j in range(d):
        if not exceed[j]:
            break
        r_hat_stepdown += 1

    return dict(
        reject_rank2=reject_rank2,
        r_hat_stepdown=int(r_hat_stepdown),
        # DEPRECATED, retained for exactly one commit so the Gate 0 diff is
        # auditable against the failing run. Do not use: this is the
        # uncontrolled marginal count whose null rate is ~1-(1-alpha)^d.
        r_hat_marginal_DEPRECATED=int(exceed.sum()),
        lam=lam.tolist(),
        band=band.tolist(),
        exceed=exceed.tolist(),
        d=int(d),
        n_match=int(n_match_eff),
        n_match_requested=int(n_match),
        n_env=int(n_e),
        n_ref_pool=int(n_p),
        n_basis=int(X_basis.shape[0]),
        B_null=int(B_null),
        alpha=float(alpha),
        band_reused=bool(band_reused),
        ref_split_factor=int(REF_SPLIT_FACTOR),
        drop=sorted(int(x) for x in drop),
    )


def standardise(X, X_fit):
    """Column z-score of X using the mean/sd of X_fit (the CONTROL cells).

    Varsortability (Reisach et al., NeurIPS 2021): raw simulator output can
    carry the causal order in the marginal variances, and any result read off
    unstandardised data is provisional until it is reproduced here.

    Scaling is fitted on controls only, for the same reason the PCA basis is.
    Because it acts as Delta -> S^-1 Delta S^-1 with S diagonal and positive,
    it preserves rank EXACTLY, so the rank-<=2 theory is untouched by it.
    """
    mu = X_fit.mean(0)
    sd = X_fit.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd
