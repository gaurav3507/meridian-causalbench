"""Eigenvalue-spectrum figure: the defence of the dimension claim.

WHY THIS FIGURE EXISTS. A reviewer will object, correctly, that the
participation ratio of an OBSERVED control covariance is not the LATENT
dimension. A low-dimensional latent process pushed through a noisy mixing map
can present with high observed dimension, so a large participation ratio is on
its own compatible with a small latent dimension plus noise.

The defence is not a bigger number, it is the SHAPE of the spectrum. Under
signal-plus-isotropic-noise the eigenvalues separate into a few large signal
eigenvalues and a noise bulk, and the boundary between them is visible as a
knee whose position is predicted by Marchenko-Pastur. If the observed spectrum
decays smoothly with no knee, and no eigenvalue bulk sits inside the MP
support, then "low-dimensional latent plus noise" does not describe the data.
The figure has to make that visible rather than assert it.

STATUS: CANNOT RUN YET. The descriptives artefacts store only spectrum
SUMMARIES (participation_ratio, effective_rank_exp_spectral_entropy,
n_components_for_variance, top_eigenvalue_share, n_nonzero_eigenvalues,
rank_bound). The full eigenvalue array is NOT persisted anywhere, and it
cannot be reconstructed from those summaries: infinitely many spectra share a
given participation ratio and entropy. This script therefore reads an
`eigenvalues` key and exits with a clear message when it is absent. It does
NOT recompute, approximate, or synthesise a spectrum, because a fabricated
curve in a figure whose entire purpose is to show a real shape would be worse
than no figure.

To enable it, 85_dataset_descriptives.py must persist the eigenvalue array
(spectrum_estimators already computes it as `lam` and discards it), and the
A100 profiling must be re-run. That re-run is Gaurav's decision.

MARCHENKO-PASTUR, from first principles. For a p-variate sample covariance
built from n iid samples whose population covariance is sigma^2 * I, with
aspect ratio gamma = p / n, the sample eigenvalues converge to the
Marchenko-Pastur law supported on

    lambda_default = sigma^2 * (1 +/- sqrt(gamma))^2

so the upper edge of the pure-noise bulk is

    lambda_plus = sigma^2 * (1 + sqrt(p / n))^2

Any eigenvalue above lambda_plus is inconsistent with isotropic noise alone
and is evidence of signal; the bulk below it is what noise alone would
produce. When p > n the sample covariance is rank-deficient with at most
n - 1 non-zero eigenvalues, and the formula still gives the upper edge of the
non-zero bulk. sigma^2 is NOT taken as the mean of all eigenvalues, which is
inflated by the signal: it is estimated from the trailing fraction of the
spectrum (default: the median of the lower half), which is dominated by the
noise bulk. That choice is a judgement and is recorded on the figure.

Usage (once eigenvalues are persisted):
    python causalbench/scripts/90_spectrum_figure.py
    python causalbench/scripts/90_spectrum_figure.py --dir <artefact dir>
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_DIR = REPO / "causalbench/results/ranktest/descriptive/descriptives"
OUT_DIR = REPO / "causalbench/paper/figures"

# only the control-cell spectra defend the dimension claim; the fMRI pooled
# spectra are a different quantity and are deliberately not panelled here
CTRL_KEY = "latent_dimension_from_controls"
EIG_KEYS = ("eigenvalues", "eigenvalue_array", "lam", "spectrum")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def find_eigenvalues(dp):
    """Return the stored eigenvalue array, or None. Never reconstructs one."""
    for k in EIG_KEYS:
        v = dp.get(k)
        if isinstance(v, list) and len(v) > 2 and all(
                isinstance(x, (int, float)) for x in v[:3]):
            return v, k
    return None, None


def mp_upper_edge(n, p, sigma2):
    """Marchenko-Pastur upper edge: sigma^2 * (1 + sqrt(p/n))^2."""
    if not n or not p or sigma2 is None:
        return None
    gamma = float(p) / float(n)
    return float(sigma2) * (1.0 + gamma ** 0.5) ** 2


def sigma2_from_bulk(lam, frac=0.5):
    """Noise variance from the TRAILING fraction of the spectrum.

    The mean of all eigenvalues is inflated by the signal, so it would push
    the MP edge up and hide exactly what the figure is meant to show.
    """
    lam = sorted((float(x) for x in lam), reverse=True)
    tail = lam[int(len(lam) * (1.0 - frac)):] or lam
    mid = len(tail) // 2
    return (tail[mid] if len(tail) % 2 else 0.5 * (tail[mid - 1] + tail[mid]))


def collect(dirpath):
    """Perturb-seq control spectra only, keyed (dataset, cap)."""
    panels, missing = [], []
    for p in sorted(Path(dirpath).glob("*.json")):
        try:
            doc = json.load(open(p))
        except ValueError:
            continue
        if (doc.get("meta") or {}).get("status") != "CURRENT":
            continue
        ds = doc.get("dataset")
        arm = ((doc.get("meta") or {}).get("config") or {}).get(
            "extra", {}).get("arm")
        label = f"{ds}_{arm}" if arm else str(ds)
        for b in doc.get("blocks") or []:
            if not isinstance(b, dict) or CTRL_KEY not in b:
                continue
            for cap, dp in (b[CTRL_KEY] or {}).items():
                if not isinstance(dp, dict):
                    continue
                lam, key = find_eigenvalues(dp)
                if lam is None:
                    missing.append((label, cap, p.name))
                else:
                    panels.append(dict(label=label, cap=cap, lam=lam,
                                       eig_key=key,
                                       n=dp.get("n_control_used"),
                                       p=dp.get("n_genes"),
                                       pr=dp.get("participation_ratio")))
    return panels, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--outdir", default=str(OUT_DIR))
    ap.add_argument("--bulk-frac", type=float, default=0.5,
                    help="trailing fraction of the spectrum used to estimate "
                         "the noise variance for the MP edge")
    a = ap.parse_args()

    panels, missing = collect(a.dir)

    if not panels:
        print("=" * 74)
        print("SPECTRUM FIGURE NOT BUILT -- no eigenvalue array is stored.")
        print("=" * 74)
        print(f"  source dir : {rel(a.dir)}")
        print(f"  checked    : {len(missing)} control-spectrum block(s)")
        for label, cap, fn in missing[:12]:
            print(f"    {label:<22} cap={cap:<6} in {fn}")
        print(f"  looked for keys: {', '.join(EIG_KEYS)}")
        print("")
        print("  The artefacts persist only SUMMARIES of the spectrum")
        print("  (participation_ratio, effective_rank_exp_spectral_entropy,")
        print("  n_components_for_variance, top_eigenvalue_share). The full")
        print("  array cannot be reconstructed from those: infinitely many")
        print("  spectra share a given participation ratio and entropy.")
        print("")
        print("  NOTHING WAS APPROXIMATED OR SYNTHESISED. To enable this")
        print("  figure, persist the eigenvalue array in")
        print("  85_dataset_descriptives.spectrum_estimators (it already")
        print("  computes it as `lam` and discards it) and re-run the A100")
        print("  profiling.")
        return 2

    # ---- from here the data exists; build the figure
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels.sort(key=lambda d: (d["label"], str(d["cap"])))
    ncol = 3
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             squeeze=False)
    for ax, d in zip(axes.ravel(), panels):
        lam = np.sort(np.asarray(d["lam"], dtype=float))[::-1]
        share = lam / lam.sum()                      # comparable across p
        ax.semilogy(np.arange(1, len(share) + 1), share, lw=1.2,
                    label="observed")
        s2 = sigma2_from_bulk(lam, a.bulk_frac)
        edge = mp_upper_edge(d["n"], d["p"], s2)
        if edge:
            ax.axhline(edge / lam.sum(), ls="--", lw=1.0, color="crimson",
                       label="MP upper edge (noise)")
        if isinstance(d["pr"], (int, float)):
            ax.axvline(d["pr"], ls=":", lw=1.0, color="navy",
                       label=f"PR={d['pr']:.0f}")
        ax.set_title(f"{d['label']}  n={d['n']}  p={d['p']}", fontsize=9)
        ax.set_xlabel("eigenvalue index")
        ax.set_ylabel("share of total variance")
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Control-covariance spectra, generated {utc_now()}  "
                 f"(noise sigma^2 from trailing {a.bulk_frac:.0%} of spectrum)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "spectrum.pdf")
    fig.savefig(out / "spectrum.png", dpi=300)
    print(f"wrote {rel(out / 'spectrum.pdf')} and spectrum.png ({len(panels)} panels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
