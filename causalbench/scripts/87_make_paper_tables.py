"""Regenerate the paper's descriptive tables from CURRENT artefacts.

Reads only what is on disk. Runs no gate, no test, and no estimator: every
number here was computed by 85_dataset_descriptives.py and is copied through,
except the two attrition fractions, which are DERIVED here on purpose so the
table cannot silently inherit a stale precomputed value.

KEY NAMES ARE TAKEN FROM THE ARTEFACTS, NOT FROM MEMORY. Three that differ
from the obvious guess, and which this file therefore handles explicitly:

  * The dimension block is named per dataset TYPE, not uniformly:
        latent_dimension_from_controls      Perturb-seq (control cells)
        latent_dimension_from_pooled_runs   HCP
        latent_dimension_from_pooled_frames ABIDE
    Anything matching latent_dimension_* is accepted and the actual key is
    reported in the table, because "from controls" and "from pooled frames"
    are not the same quantity and must not be silently merged.
  * That block is nested BY SPECTRUM CAP (e.g. "2000", "8000"), so one
    dataset yields several rows, one per n_control_used. This is the whole
    point: the estimates are sample-size-dependent lower bounds.
  * Environment counts live at cells_per_environment.n_environments; the
    Perturb-seq-only n_environments_total is a different field and is not
    used as a substitute.

Usage:
    python causalbench/scripts/87_make_paper_tables.py
    python causalbench/scripts/87_make_paper_tables.py --dir <artefact dir>
"""
import argparse
import csv
import datetime
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_DIR = REPO / "causalbench/results/ranktest/descriptive/descriptives"
OUT_DIR = REPO / "causalbench/paper/tables"

NA = "n/a"

def rel(p):
    """Repo-relative when possible, absolute otherwise. --dir may be relative."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)

HARD_FLOOR, SOFT_FLOOR = 125, 8000

FOOTNOTE_B = ("Dimension estimates are sample-size-dependent lower bounds; "
              "compare only at matched n_control_used. Estimators disagree by "
              "up to ~1.8x and all three are reported.")
FOOTNOTE_A_FMRI = (
    "fMRI environments are acquisition sites or task conditions, not "
    "interventions; the attrition curve is a descriptive analogue and not an "
    "interventional one. HCP and ABIDE have no control condition, so their "
    "spectra are over pooled frames and are NOT comparable to the Perturb-seq "
    "control-cell spectra.")

FMRI_DATASETS = {"hcp", "abide"}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def script_commit():
    r = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%h",
                        "--", str(Path(__file__).relative_to(REPO))],
                       capture_output=True, text=True)
    return r.stdout.strip() or "uncommitted"


def render_ts(ts):
    """Artefact timestamps carry no tz marker. Render verbatim, say so."""
    if ts is None:
        return NA
    s = str(ts)
    return s if s.endswith("Z") else f"{s} (tz unmarked)"


def g(d, *path, default=NA):
    """Nested get. A missing key is NA, never 0 and never blank."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return default if cur is None else cur


def dim_blocks(block):
    """Yield (key_name, cap, payload) for every latent_dimension_* block."""
    for key, val in block.items():
        if not key.startswith("latent_dimension_") or not isinstance(val, dict):
            continue
        for cap, payload in sorted(val.items(),
                                   key=lambda kv: int(kv[0])
                                   if str(kv[0]).isdigit() else 0):
            if isinstance(payload, dict):
                yield key, cap, payload


def dataset_label(doc, block):
    """dataset name, with the Frangieh arm appended when present."""
    ds = doc.get("dataset") or NA
    arm = (g(doc, "meta", "config", "extra", "arm", default=None)
           or block.get("arm_flag"))
    if arm and arm != NA:
        return f"{ds}_{arm}"
    lbl = block.get("label") or ""
    if ":" in str(lbl):                      # e.g. "frangieh:ifng"
        return str(lbl).replace(":", "_")
    return ds


def load(dirpath):
    kept, skipped = [], []
    for p in sorted(Path(dirpath).glob("*.json")):
        try:
            doc = json.load(open(p))
        except ValueError as e:
            skipped.append((p.name, f"unparseable: {e}"))
            continue
        status = g(doc, "meta", "status", default=None)
        if status != "CURRENT":
            skipped.append((p.name, f"status={status!r}"))
            continue
        kept.append((p, doc))
    return kept, skipped


def build_table_a(kept):
    floors, rows = set(), []
    for _, doc in kept:
        for b in doc.get("blocks") or []:
            if not isinstance(b, dict) or "environment_attrition" not in b:
                continue
            floors |= {int(k) for k in b["environment_attrition"]}
    floors = sorted(floors)

    for _, doc in kept:
        for b in doc.get("blocks") or []:
            if not isinstance(b, dict) or "environment_attrition" not in b:
                continue
            att = b["environment_attrition"]
            cpe = b.get("cells_per_environment") or {}
            n_env = g(cpe, "n_environments")
            total = g(cpe, "total_cells")
            # DERIVED here, never read from the artefact
            n_hard = att.get(str(HARD_FLOOR))
            n_soft = att.get(str(SOFT_FLOOR))
            f_hard = (round(n_hard / n_env, 4)
                      if isinstance(n_hard, int) and isinstance(n_env, int)
                      and n_env else NA)
            f_soft = (round(n_soft / n_env, 4)
                      if isinstance(n_soft, int) and isinstance(n_env, int)
                      and n_env else NA)
            row = {"dataset": dataset_label(doc, b), "n_environments": n_env}
            for f in floors:
                row[f"surv_{f}"] = att.get(str(f), NA)
            row.update({
                "cells_min": g(cpe, "min"), "cells_q25": g(cpe, "q25"),
                "cells_median": g(cpe, "median"), "cells_q75": g(cpe, "q75"),
                "cells_max": g(cpe, "max"), "total_cells": total,
                "frac_above_hard_125": f_hard,
                "frac_above_soft_8000": f_soft,
            })
            rows.append(row)
    rows.sort(key=lambda r: str(r["dataset"]))
    return rows


def build_table_b(kept):
    rows = []
    for _, doc in kept:
        for b in doc.get("blocks") or []:
            if not isinstance(b, dict):
                continue
            for key, cap, dp in dim_blocks(b):
                rows.append({
                    "dataset": dataset_label(doc, b),
                    "spectrum_source": key.replace("latent_dimension_", ""),
                    "spec_cap": cap,
                    "n_features": g(b, "n_features",
                                    default=g(b, "n_regions",
                                              default=g(dp, "n_genes"))),
                    "n_control_available": g(dp, "n_control_available"),
                    "n_control_used": g(dp, "n_control_used"),
                    "rank_bound": g(dp, "rank_bound"),
                    "participation_ratio": g(dp, "participation_ratio"),
                    "effective_rank_exp_spectral_entropy":
                        g(dp, "effective_rank_exp_spectral_entropy"),
                    "n_comp_80pct": g(dp, "n_components_for_variance", "80pct"),
                    "n_comp_90pct": g(dp, "n_components_for_variance", "90pct"),
                    "n_comp_95pct": g(dp, "n_components_for_variance", "95pct"),
                    "top_eigenvalue_share": g(dp, "top_eigenvalue_share"),
                })
    rows.sort(key=lambda r: (str(r["dataset"]), str(r["spec_cap"])))
    return rows


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return NA if v is None else str(v)


def header_lines(kept, skipped, dirpath):
    commits = {}
    for p, doc in kept:
        commits.setdefault(g(doc, "meta", "git_commit", default=NA),
                           []).append(p.name)
    L = [f"generated   : {utc_now()}",
         f"generator   : 87_make_paper_tables.py @ {script_commit()}",
         f"source dir  : {rel(dirpath)}",
         f"artefacts   : {len(kept)} CURRENT, {len(skipped)} skipped"]
    if len(commits) > 1:
        L.append("")
        L.append("!! WARNING: MIXED PROVENANCE -- source artefacts do not share "
                 "one meta.git_commit.")
        L.append("!! Rows below were produced by DIFFERENT versions of the "
                 "profiler. Do not read them as one run.")
    for c, names in sorted(commits.items()):
        for n in sorted(names):
            L.append(f"  source    : {n}   meta.git_commit={c}")
    for n, why in skipped:
        L.append(f"  skipped   : {n}   ({why})")
    return L


def write_outputs(name, rows, header, footnotes, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else []

    csv_path = outdir / f"{name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        for line in header:
            w.writerow([f"# {line}"])
        if cols:
            w.writerow(cols)
            for r in rows:
                w.writerow([fmt(r[c]) for c in cols])
        for fn in footnotes:
            w.writerow([f"# {fn}"])

    md_path = outdir / f"{name}.md"
    L = [f"# {name}", ""]
    L += ["```"] + header + ["```", ""]
    if cols:
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in rows:
            L.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    else:
        L.append("_no CURRENT artefacts found_")
    L.append("")
    for fn in footnotes:
        L.append(f"> {fn}")
        L.append("")
    md_path.write_text("\n".join(L))
    return md_path, csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--outdir", default=str(OUT_DIR))
    a = ap.parse_args()

    kept, skipped = load(a.dir)
    head = header_lines(kept, skipped, a.dir)
    print("\n".join(head))

    rows_a = build_table_a(kept)
    has_fmri = any(str(r["dataset"]).split("_")[0] in FMRI_DATASETS
                   for r in rows_a)
    fa = [FOOTNOTE_A_FMRI] if has_fmri else []
    pa = write_outputs("table_a_environment_attrition", rows_a, head, fa, a.outdir)

    rows_b = build_table_b(kept)
    pb = write_outputs("table_b_dimension", rows_b, head, [FOOTNOTE_B], a.outdir)

    print(f"\nTable A: {len(rows_a)} rows -> {pa[0].name}, {pa[1].name}")
    print(f"Table B: {len(rows_b)} rows -> {pb[0].name}, {pb[1].name}")
    if skipped:
        print(f"skipped {len(skipped)} non-CURRENT artefact(s)")


if __name__ == "__main__":
    main()
