"""
Main pipeline entrypoint for cam-structural-harmony.

Orchestrates the full pipeline:
  1. Skull stripping (SynthStrip and/or ROBEX) — parallel
  2. Intensity normalisation (up to 6 methods per skull strip) — parallel
  3. FreeSurfer recon-all + aparc/aseg extraction — per variant
  4. NeuroCombat harmonisation — per variant
  5. Variance decomposition (ICC, CV, source partitioning) — per variant
  6. Comparison figures
  7. AI-assisted QC report

All outputs are organised under:
  output_dir/pipeline_variants/{skull_strip}_{norm}/
  output_dir/variance_results/
  output_dir/figures/
  output_dir/qc_report.md

Usage
-----
    python -m cam_harmony.run \
        --bids_dir /path/to/bids \
        --output_dir /path/to/outputs \
        --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import product
from pathlib import Path

import pandas as pd
import yaml


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(config_path: str | Path) -> dict:
    """Load and return the YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_bids_t1w_paths(bids_dir: Path) -> list[tuple[str, Path]]:
    """
    Discover T1w NIfTI images in a BIDS dataset.

    Handles both session-level and flat layouts:
      - Session: sub-XX/ses-YY/anat/*_T1w.nii.gz → label ``sub-XX_ses-YY``
      - Flat:    sub-XX/anat/*_T1w.nii.gz         → label ``sub-XX``

    For session datasets each scan is treated as a separate entry so that
    within-subject, cross-scanner comparisons (e.g. TRIO vs PRISMA sessions)
    are preserved as independent FreeSurfer subjects.

    Returns
    -------
    List of (subject_label, t1w_path) tuples, sorted by subject_label.
    """
    # Try session-level layout first
    t1w_files = sorted(bids_dir.glob("sub-*/ses-*/anat/*_T1w.nii.gz"))
    if not t1w_files:
        t1w_files = sorted(bids_dir.glob("sub-*/ses-*/anat/*_T1w.nii"))

    if t1w_files:
        subjects = []
        for path in t1w_files:
            parts = path.parts
            sub = next(p for p in parts if p.startswith("sub-"))
            ses = next(p for p in parts if p.startswith("ses-"))
            subjects.append((f"{sub}_{ses}", path))
        print(f"[run] Found {len(subjects)} scans (session BIDS) in {bids_dir}")
        return subjects

    # Fall back to flat (no-session) layout
    t1w_files = sorted(bids_dir.glob("sub-*/anat/*_T1w.nii.gz"))
    if not t1w_files:
        t1w_files = sorted(bids_dir.glob("sub-*/anat/*_T1w.nii"))

    subjects = []
    seen: set[str] = set()
    for path in t1w_files:
        parts = path.parts
        sub = next(p for p in parts if p.startswith("sub-"))
        if sub not in seen:
            seen.add(sub)
            subjects.append((sub, path))

    print(f"[run] Found {len(subjects)} subjects (flat BIDS) in {bids_dir}")
    return subjects


def load_participants_tsv(bids_dir: Path) -> pd.DataFrame | None:
    """Load participants.tsv if present (provides age, sex, scanner metadata)."""
    tsv_path = bids_dir / "participants.tsv"
    if tsv_path.exists():
        df = pd.read_csv(tsv_path, sep="\t")
        print(f"[run] Loaded participants.tsv: {df.shape[0]} rows, columns: {list(df.columns)}")
        return df
    print("[run] No participants.tsv found — metadata will be empty.")
    return None


# ── Per-variant pipeline ──────────────────────────────────────────────────────

def run_variant(
    skull_strip_method: str,
    norm_method: str,
    subjects: list[tuple[str, Path]],
    output_dir: Path,
    config: dict,
    participants: pd.DataFrame | None,
) -> Path:
    """
    Run skull strip → normalise → FreeSurfer → Combat → variance for one variant.

    Parameters
    ----------
    skull_strip_method : str
        "synthstrip" or "robex".
    norm_method : str
        Intensity normalisation method name.
    subjects : list of (subject_id, t1w_path)
    output_dir : Path
        Root output directory.
    config : dict
        Loaded from config.yaml.
    participants : pd.DataFrame or None
        Participant metadata (age, sex, scanner, etc.).

    Returns
    -------
    Path to the variant output directory.
    """
    from cam_harmony.skull_strip import batch_skull_strip
    from cam_harmony.intensity_norm import batch_normalize
    from cam_harmony.freesurfer import batch_recon_all, extract_aparc_aseg
    from cam_harmony.harmonise import run_combat, compute_combat_residuals
    from cam_harmony.variance import run_variance_analysis

    variant_label = f"{skull_strip_method}_{norm_method}"
    variant_dir = output_dir / "pipeline_variants" / variant_label
    variant_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[run] Variant: {variant_label}")
    print(f"{'='*60}")

    subject_ids = [s[0] for s in subjects]
    t1w_paths = [s[1] for s in subjects]

    # ── Step 1: Skull stripping ───────────────────────────────────────────────
    strip_dir = variant_dir / "skull_strip"
    ss_cfg = config.get("skull_strip", {}).get(skull_strip_method, {})

    stripped_results = batch_skull_strip(
        input_paths=t1w_paths,
        output_dir=strip_dir,
        method=skull_strip_method,
        n_jobs=config.get("freesurfer", {}).get("parallel_jobs", 1),
        no_csf=ss_cfg.get("no_csf", False),
        device=ss_cfg.get("device", "cpu"),
        synthstrip_bin=config.get("paths", {}).get("synthstrip_bin", "mri_synthstrip"),
    )

    stripped_paths = [r[0] for r in stripped_results]
    mask_paths = [r[1] for r in stripped_results]

    # ── Step 2: Intensity normalisation ──────────────────────────────────────
    norm_dir = variant_dir / "normalised"
    normalised_paths = batch_normalize(
        image_paths=stripped_paths,
        method=norm_method,
        output_dir=norm_dir,
        mask_paths=mask_paths,
    )

    # ── Step 3: FreeSurfer recon-all ──────────────────────────────────────────
    fs_dir = variant_dir / "freesurfer"
    fs_cfg = config.get("freesurfer", {})
    fs_subjects_dir = Path(fs_cfg.get("subjects_dir") or fs_dir)
    fs_home = config.get("paths", {}).get("freesurfer_home") or os.environ.get("FREESURFER_HOME")

    batch_recon_all(
        subject_ids=subject_ids,
        t1w_paths=normalised_paths,
        subjects_dir=fs_subjects_dir,
        n_jobs=fs_cfg.get("parallel_jobs", 1),
        freesurfer_home=fs_home,
    )

    roi_df = extract_aparc_aseg(fs_subjects_dir, subject_ids)
    roi_csv = variant_dir / "aparc_aseg_volumes.csv"
    roi_df.to_csv(roi_csv)

    # ── Merge participant metadata ────────────────────────────────────────────
    combat_cfg = config.get("combat", {})
    batch_col = combat_cfg.get("batch_col", "scanner")
    bio_covars = combat_cfg.get("biological_covariates", [])

    if participants is not None:
        part_id_col = participants.columns[0]

        # roi_df.index is the FreeSurfer subject label: sub-14_ses-PRISMA1
        # participants.tsv uses plain participant IDs: sub-14
        # Extract sub-XX from sub-XX_ses-YY so the join finds matches.
        sub_keys = (
            roi_df.index.to_series()
            .str.extract(r"^(sub-[^_]+)", expand=False)
            .fillna(roi_df.index.to_series())
        )
        join_cols = [c for c in [batch_col, *bio_covars] if c in participants.columns]
        if join_cols:
            meta = (
                participants.set_index(part_id_col)[join_cols]
                .reindex(sub_keys.values)
            )
            meta.index = roi_df.index   # restore FreeSurfer subject labels as index
            roi_df = roi_df.join(meta, how="left")

    # Fallback: derive batch_col from the session label when it is absent or all-NaN.
    # e.g. sub-14_ses-PRISMA1 → PRISMA;  sub-14_ses-TRIO2 → TRIO
    if batch_col not in roi_df.columns or roi_df[batch_col].isna().all():
        derived = (
            roi_df.index.to_series()
            .str.extract(r"ses-([A-Za-z]+)\d*$", expand=False)
        )
        if not derived.isna().all():
            roi_df[batch_col] = derived.values
            print(
                f"[run] '{batch_col}' derived from session labels: "
                f"{sorted(roi_df[batch_col].dropna().unique())}"
            )
        else:
            print(
                f"[run] Warning: '{batch_col}' not in participants.tsv and cannot be "
                "derived from session labels. Harmonisation and ICC will be skipped."
            )

    roi_cols = [c for c in roi_df.columns if c not in [batch_col, *bio_covars]]

    # ── Step 4: NeuroCombat harmonisation ─────────────────────────────────────
    batch_ready = (
        batch_col in roi_df.columns
        and roi_df[batch_col].notna().any()
        and roi_df[batch_col].nunique() >= 2
    )
    min_per_batch = roi_df[batch_col].value_counts().min() if batch_ready else 0

    if batch_ready and min_per_batch >= 2:
        harmonised_df = run_combat(
            data=roi_df,
            batch_col=batch_col,
            covariate_cols=bio_covars if all(c in roi_df for c in bio_covars) else None,
            parametric=combat_cfg.get("parametric", True),
            output_path=variant_dir / "harmonised_volumes.csv",
        )
        residuals = compute_combat_residuals(
            pre_harmonised=roi_df,
            post_harmonised=harmonised_df,
            batch_col=batch_col,
            roi_cols=roi_cols,
            output_path=variant_dir / "combat_residuals.json",
        )
    else:
        if batch_ready and min_per_batch < 2:
            print(
                f"[run] Warning: smallest batch has {min_per_batch} subject(s) — "
                "NeuroCombat requires ≥ 2 per batch. Skipping harmonisation."
            )
        else:
            print(f"[run] Warning: batch column '{batch_col}' unavailable — skipping harmonisation.")
        harmonised_df = roi_df
        residuals = {}

    # ── Step 5: Variance analysis ─────────────────────────────────────────────
    var_cfg = config.get("variance", {})
    variance_dir = output_dir / "variance_results" / variant_label
    variance_dir.mkdir(parents=True, exist_ok=True)

    subject_col = "subject_id"
    if subject_col not in roi_df.columns:
        roi_df.index.name = subject_col
        roi_df = roi_df.reset_index()

    run_variance_analysis(
        data=roi_df,
        roi_cols=roi_cols,
        batch_col=batch_col,
        subject_col=subject_col,
        variance_sources=[s for s in var_cfg.get("sources", []) if s in roi_df.columns],
        output_dir=variance_dir,
        variant_label=variant_label,
    )

    # ── Post-harmonisation ICC — for pre/post comparison ──────────────────────
    # Only meaningful when harmonisation actually ran (harmonised_df ≠ roi_df).
    if harmonised_df is not roi_df and batch_ready:
        from cam_harmony.variance import compute_icc_batch

        harm_df = harmonised_df.copy()
        if subject_col not in harm_df.columns:
            harm_df.index.name = subject_col
            harm_df = harm_df.reset_index()

        post_icc = compute_icc_batch(harm_df, roi_cols, subject_col, batch_col)
        post_icc_path = variance_dir / "icc_results_post.json"
        with open(post_icc_path, "w") as f:
            json.dump(post_icc.to_dict(), f, indent=2)
        print(f"[run] Post-harmonisation ICC saved to {post_icc_path.name}")

    # Save variant metadata for QC assistant
    variants_meta_path = output_dir / "variance_results" / "pipeline_variants.json"
    variants_meta: list = []
    if variants_meta_path.exists():
        with open(variants_meta_path) as f:
            variants_meta = json.load(f)
    variants_meta.append({
        "label": variant_label,
        "skull_strip": skull_strip_method,
        "norm": norm_method,
    })
    with open(variants_meta_path, "w") as f:
        json.dump(variants_meta, f, indent=2)

    print(f"[run] Variant complete: {variant_label}")
    return variant_dir


# ── Main entrypoint ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="cam-structural-harmony: full MRI harmonisation pipeline"
    )
    parser.add_argument(
        "--bids_dir", required=True,
        help="Path to BIDS-formatted dataset"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Root directory for all pipeline outputs"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--skull_strip", nargs="+", default=None,
        help="Override skull-strip methods from config (e.g. --skull_strip synthstrip)"
    )
    parser.add_argument(
        "--norm", nargs="+", default=None,
        help="Override normalisation methods from config"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print planned variants without running them"
    )
    args = parser.parse_args()

    bids_dir = Path(args.bids_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)

    skull_strip_methods = args.skull_strip or config.get("skull_strip", {}).get("methods", ["synthstrip"])
    norm_methods = args.norm or config.get("intensity_norm", {}).get("methods", ["zscore"])

    variants = list(product(skull_strip_methods, norm_methods))

    print(f"[run] Pipeline variants to process: {len(variants)}")
    for ss, nm in variants:
        print(f"       {ss} × {nm}")

    if args.dry_run:
        print("[run] Dry run complete — exiting.")
        return

    subjects = get_bids_t1w_paths(bids_dir)
    participants = load_participants_tsv(bids_dir)

    if not subjects:
        raise FileNotFoundError(f"No T1w images found in {bids_dir}")

    # ── Run all variants ──────────────────────────────────────────────────────
    completed_variants = []
    for skull_strip_method, norm_method in variants:
        variant_dir = run_variant(
            skull_strip_method=skull_strip_method,
            norm_method=norm_method,
            subjects=subjects,
            output_dir=output_dir,
            config=config,
            participants=participants,
        )
        completed_variants.append((skull_strip_method, norm_method, variant_dir))

    # ── Post-processing: figures ──────────────────────────────────────────────
    from cam_harmony.plotting import generate_all_figures

    print("\n[run] Generating comparison figures ...")
    generate_all_figures(
        results_dir=output_dir / "variance_results",
        figures_dir=output_dir / "figures",
    )

    # ── Post-processing: QC report ────────────────────────────────────────────
    from cam_harmony.qc_assistant import generate_qc_report

    var_cfg = config.get("variance", {})
    focus_rois = var_cfg.get("rois") or None

    print("\n[run] Generating AI-assisted QC report ...")
    generate_qc_report(
        results_dir=output_dir / "variance_results",
        output_path=output_dir / "qc_report.md",
        focus_rois=focus_rois,
    )

    print(f"\n[run] Pipeline complete. Outputs in: {output_dir}")
    print(f"      Variants completed: {len(completed_variants)}")
    print(f"      QC report: {output_dir / 'qc_report.md'}")


if __name__ == "__main__":
    main()
