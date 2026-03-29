#!/usr/bin/env python3
"""
FreeSurfer QC — per-scanner outlier flagging and slice panel generation.

Reads a ``qc_stats.csv`` produced by the companion bash script
``collect_freesurfer_stats.sh``, performs per-scanner SD-based outlier
detection, generates a per-subject brain slice panel with segmentation
overlay, and produces a contact sheet with flagged subjects highlighted.

Usage
-----
    # Step 1: collect raw QC metrics (bash, requires FreeSurfer on PATH)
    bash scripts/collect_freesurfer_stats.sh \\
        --subjects_dir /path/to/freesurfer \\
        --output_dir   outputs/freesurfer_qc

    # Step 2: flag outliers and generate slice panels (Python)
    python scripts/freesurfer_qc.py \\
        --subjects_dir /path/to/freesurfer \\
        --output_dir   outputs/freesurfer_qc \\
        --stats_csv    outputs/freesurfer_qc/qc_stats.csv \\
        [--sd_threshold 3.0] \\
        [--euler_threshold -200] \\
        [--ncols 4]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


# ── Colourmap for segmentation overlay ────────────────────────────────────────

def _seg_cmap() -> ListedColormap:
    np.random.seed(42)
    colors = np.random.rand(256, 4)
    colors[:, 3] = 0.45
    colors[0, :] = [0, 0, 0, 0]   # background transparent
    return ListedColormap(colors)


_SEG_CMAP = _seg_cmap()


# ── Image helpers ──────────────────────────────────────────────────────────────

def _load_mgz(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return nib.as_closest_canonical(img).get_fdata()


def _normalize(vol: np.ndarray) -> np.ndarray:
    nonzero = vol[vol > 0]
    if nonzero.size == 0:
        return vol
    vmin, vmax = np.percentile(nonzero, [1, 99])
    return np.clip((vol - vmin) / (vmax - vmin + 1e-8), 0, 1)


# ── Per-scanner outlier detection ──────────────────────────────────────────────

def flag_outliers(
    df: pd.DataFrame,
    metric_cols: list[str],
    sd_threshold: float = 3.0,
    group_col: str = "scanner",
) -> dict[str, list[str]]:
    """
    Flag subjects where any metric exceeds sd_threshold SDs from the
    per-scanner mean. Returns {subject: [flag_reasons]}.
    """
    flags: dict[str, list[str]] = {s: [] for s in df["subject"]}

    for col in metric_cols:
        if df[col].isna().all():
            continue
        for scanner, grp in df.groupby(group_col):
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            if len(vals) < 3:
                continue
            mean, sd = vals.mean(), vals.std()
            if sd == 0:
                continue
            for _, row in grp.iterrows():
                val = pd.to_numeric(row[col], errors="coerce")
                if pd.isna(val):
                    continue
                z = abs(val - mean) / sd
                if z > sd_threshold:
                    flags[row["subject"]].append(
                        f"{col} outlier (scanner={scanner}, val={val:.1f}, z={z:.1f})"
                    )

    return flags


def flag_absolute(df: pd.DataFrame, euler_threshold: float) -> dict[str, list[str]]:
    """Scanner-agnostic hard-fail checks (log completion, missing files, Euler number)."""
    flags: dict[str, list[str]] = {s: [] for s in df["subject"]}

    for _, row in df.iterrows():
        subj = row["subject"]
        if row.get("log_complete", 1) == 0:
            flags[subj].append("recon-all did not finish")
        errs = pd.to_numeric(row.get("log_errors", 0), errors="coerce")
        if not pd.isna(errs) and errs > 0:
            flags[subj].append(f"{int(errs)} ERROR lines in log")
        missing = pd.to_numeric(row.get("missing_files", 0), errors="coerce")
        if not pd.isna(missing) and missing > 0:
            flags[subj].append(f"{int(missing)} missing output files")
        for hemi, col in [("lh", "euler_lh"), ("rh", "euler_rh")]:
            val = pd.to_numeric(row.get(col, 0), errors="coerce")
            if not pd.isna(val) and val < euler_threshold:
                flags[subj].append(f"Euler {hemi}={val:.0f} < {euler_threshold}")

    return flags


# ── Per-subject slice panel ────────────────────────────────────────────────────

def make_subject_panel(
    subj_path: Path,
    subj_name: str,
    output_path: Path,
    flag_reasons: list[str] | None = None,
    scanner_label: str | None = None,
) -> bool:
    """Generate a 3-plane brain slice panel with segmentation overlay."""
    brain_path = subj_path / "mri" / "brain.mgz"
    seg_path   = subj_path / "mri" / "aparc+aseg.mgz"

    if not brain_path.exists() or not seg_path.exists():
        print(f"  [SKIP] {subj_name} — missing brain.mgz or aparc+aseg.mgz")
        return False

    try:
        brain = _normalize(_load_mgz(brain_path))
        seg   = _load_mgz(seg_path)
    except Exception as e:
        print(f"  [ERROR] {subj_name}: {e}")
        return False

    mx, my, mz = [s // 2 for s in brain.shape]
    planes = [
        (brain[:, :, mz], seg[:, :, mz],   "Axial"),
        (brain[:, my, :], seg[:, my, :],   "Coronal"),
        (brain[mx, :, :], seg[mx, :, :],   "Sagittal"),
    ]

    flagged = bool(flag_reasons)
    border_color = "red" if flagged else "#00cc44"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("black")

    for ax, (bslice, sslice, label) in zip(axes, planes):
        ax.imshow(np.rot90(bslice), cmap="gray", interpolation="nearest")
        ax.imshow(np.rot90(sslice % 255), cmap=_SEG_CMAP,
                  interpolation="nearest", vmin=0, vmax=255)
        ax.set_title(label, color="white", fontsize=9)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)

    scanner_str = f"  [{scanner_label}]" if scanner_label else ""
    flag_str    = "  ⚠ FLAGGED" if flagged else ""
    fig.suptitle(f"{subj_name}{scanner_str}{flag_str}",
                 color="red" if flagged else "white",
                 fontsize=10, fontweight="bold", y=1.01)

    if flagged and flag_reasons:
        reasons = " | ".join(flag_reasons[:3])
        if len(flag_reasons) > 3:
            reasons += f" (+{len(flag_reasons) - 3} more)"
        fig.text(0.5, -0.01, reasons, color="red", fontsize=7,
                 ha="center", va="top")

    plt.tight_layout(pad=0.5)
    fig.savefig(str(output_path), dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return True


# ── Contact sheet ──────────────────────────────────────────────────────────────

def make_contact_sheet(
    panel_paths: list[Path],
    subject_names: list[str],
    flagged_set: set[str],
    output_path: Path,
    ncols: int = 4,
) -> None:
    """Tile all subject panels into a single contact sheet."""
    n = len(panel_paths)
    if n == 0:
        return

    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 2.4))
    fig.patch.set_facecolor("#111111")
    axes_flat = list(np.array(axes).flat)

    for ax, panel_path, subj in zip(axes_flat, panel_paths, subject_names):
        try:
            ax.imshow(mpimg.imread(str(panel_path)))
        except Exception:
            ax.set_facecolor("black")
            ax.text(0.5, 0.5, "LOAD ERROR", color="red",
                    ha="center", va="center", transform=ax.transAxes)
        border = "red" if subj in flagged_set else "#333333"
        lw = 3 if subj in flagged_set else 0.5
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(lw)
        ax.axis("off")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    legend = fig.legend(
        handles=[
            mpatches.Patch(facecolor="red",     label="Flagged"),
            mpatches.Patch(facecolor="#00cc44", label="Passed"),
        ],
        loc="lower center", ncol=2,
        facecolor="#222222", edgecolor="white", fontsize=10,
        bbox_to_anchor=(0.5, -0.01),
    )
    for text in legend.get_texts():
        text.set_color("white")

    plt.suptitle("FreeSurfer QC — Contact Sheet",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout(pad=0.3)
    fig.savefig(str(output_path), dpi=100, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"[freesurfer_qc] Contact sheet → {output_path}")


# ── Summary report ─────────────────────────────────────────────────────────────

def write_summary(
    df: pd.DataFrame,
    all_flags: dict[str, list[str]],
    output_path: Path,
) -> None:
    """Write plain-text flagging summary."""
    flagged = {s: r for s, r in all_flags.items() if r}

    with open(output_path, "w") as f:
        f.write("FreeSurfer QC — Flagging Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total subjects : {len(df)}\n")
        f.write(f"Flagged        : {len(flagged)}\n")
        f.write(f"Passed         : {len(df) - len(flagged)}\n\n")
        f.write("Per-scanner breakdown:\n")
        for scanner, grp in df.groupby("scanner"):
            n_flagged = sum(1 for s in grp["subject"] if all_flags.get(s))
            f.write(f"  {scanner}: {len(grp)} subjects, {n_flagged} flagged\n")
        f.write("\n" + "=" * 60 + "\nFLAGGED SUBJECTS\n" + "=" * 60 + "\n\n")
        for subj, reasons in sorted(flagged.items()):
            f.write(f"{subj}:\n")
            for r in reasons:
                f.write(f"  - {r}\n")
            f.write("\n")

    print(f"[freesurfer_qc] Summary → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

_SD_METRICS = [
    "icv", "total_gm", "total_wm",
    "euler_lh", "euler_rh",
    "talairach_tx", "talairach_ty", "talairach_tz",
]

_SKIP_SUBJECTS = {"fsaverage", "fsaverage5", "fsaverage6", "bert"}


def run_qc(
    subjects_dir: Path,
    output_dir: Path,
    stats_csv: Path,
    sd_threshold: float = 3.0,
    euler_threshold: float = -200.0,
    ncols: int = 4,
) -> None:
    subjects_dir = Path(subjects_dir)
    output_dir   = Path(output_dir)
    panels_dir   = output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(stats_csv)
    print(f"[freesurfer_qc] Loaded {len(df)} subjects from {stats_csv}")

    abs_flags = flag_absolute(df, euler_threshold=euler_threshold)
    sd_flags  = flag_outliers(df, _SD_METRICS, sd_threshold=sd_threshold)

    all_flags: dict[str, list[str]] = {
        s: abs_flags.get(s, []) + sd_flags.get(s, [])
        for s in set(abs_flags) | set(sd_flags)
    }
    flagged_set = {s for s, r in all_flags.items() if r}
    print(f"[freesurfer_qc] Flagged {len(flagged_set)} / {len(df)} subjects")

    write_summary(df, all_flags, output_dir / "qc_flagging_summary.txt")

    panel_paths, subject_names = [], []
    for _, row in df.iterrows():
        subj = row["subject"]
        if subj in _SKIP_SUBJECTS:
            continue
        panel_path = panels_dir / f"{subj}_qc.png"
        ok = make_subject_panel(
            subj_path=subjects_dir / subj,
            subj_name=subj,
            output_path=panel_path,
            flag_reasons=all_flags.get(subj, []),
            scanner_label=row.get("scanner"),
        )
        if ok:
            panel_paths.append(panel_path)
            subject_names.append(subj)

    make_contact_sheet(
        panel_paths, subject_names, flagged_set,
        output_dir / "contact_sheet.png", ncols=ncols,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FreeSurfer QC: per-scanner flagging + slice panel generation"
    )
    parser.add_argument("--subjects_dir",    required=True,
                        help="FreeSurfer SUBJECTS_DIR")
    parser.add_argument("--output_dir",      required=True,
                        help="Directory for QC outputs")
    parser.add_argument("--stats_csv",       required=True,
                        help="CSV produced by collect_freesurfer_stats.sh")
    parser.add_argument("--sd_threshold",    type=float, default=3.0,
                        help="SD threshold for per-scanner outlier detection (default: 3)")
    parser.add_argument("--euler_threshold", type=float, default=-200.0,
                        help="Absolute Euler number threshold (default: -200)")
    parser.add_argument("--ncols",           type=int,   default=4,
                        help="Contact sheet columns (default: 4)")
    args = parser.parse_args()

    run_qc(
        subjects_dir=Path(args.subjects_dir),
        output_dir=Path(args.output_dir),
        stats_csv=Path(args.stats_csv),
        sd_threshold=args.sd_threshold,
        euler_threshold=args.euler_threshold,
        ncols=args.ncols,
    )


if __name__ == "__main__":
    main()
