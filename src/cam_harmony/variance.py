"""
Variance decomposition module for cam-structural-harmony.

Quantifies technical sources of variance in ROI volumes before and after
harmonisation. Provides three complementary metrics:

  ICC   : Intraclass correlation coefficient — how consistent is a measurement
          across sessions/scanners for the same subject? Target: ICC > 0.75.
  CV    : Coefficient of variation — spread of volumes within a scanner group
          as a fraction of the group mean.
  VD    : Variance decomposition — how much of total variance is attributable
          to each technical source (scanner, site, manufacturer, model, protocol)?

Uses pingouin for ICC computation (two-way mixed model, absolute agreement)
and a simple ANOVA-based decomposition for variance partitioning.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def compute_icc(
    data: pd.DataFrame,
    roi: str,
    subject_col: str,
    rater_col: str,
    icc_type: str = "ICC(A,1)",
) -> float:
    """
    Compute intraclass correlation coefficient for a single ROI.

    Uses a two-way mixed-effects model (absolute agreement), appropriate
    for scanner reliability studies where scanners are a fixed set.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format DataFrame with one row per subject-session.
    roi : str
        Column name of the ROI volume to evaluate.
    subject_col : str
        Column identifying subjects (targets).
    rater_col : str
        Column identifying the scanner or session (raters).
    icc_type : str
        ICC type from pingouin: "ICC(1,1)", "ICC(A,1)" (default, two-way
        absolute agreement), "ICC(C,1)", "ICC(1,k)", "ICC(A,k)", "ICC(C,k)".

    Returns
    -------
    float : ICC value in [0, 1].
    """
    try:
        import pingouin as pg
    except ImportError as e:
        raise ImportError("pingouin is required: pip install pingouin") from e

    # Drop duplicate (subject, rater) pairs so each subject has exactly one
    # measurement per scanner, then reshape to the long format pingouin expects.
    long = (
        data[[subject_col, rater_col, roi]]
        .drop_duplicates(subset=[subject_col, rater_col])
        .reset_index(drop=True)
    )

    icc_result = pg.intraclass_corr(
        data=long,
        targets=subject_col,
        raters=rater_col,
        ratings=roi,
        nan_policy="omit",
    )
    row = icc_result[icc_result["Type"] == icc_type]
    if row.empty:
        raise ValueError(f"ICC type '{icc_type}' not found in pingouin output.")
    return float(row["ICC"].values[0])


def compute_icc_batch(
    data: pd.DataFrame,
    roi_cols: list[str],
    subject_col: str,
    rater_col: str,
    icc_type: str = "ICC(A,1)",
) -> pd.Series:
    """
    Compute ICC for a list of ROIs. Returns a Series indexed by ROI name.

    Parameters
    ----------
    data : pd.DataFrame
    roi_cols : list of str
    subject_col : str
    rater_col : str
    icc_type : str

    Returns
    -------
    pd.Series with ROI names as index and ICC values as values.
    """
    icc_values = {}
    for roi in roi_cols:
        try:
            icc_values[roi] = compute_icc(data, roi, subject_col, rater_col, icc_type)
        except Exception as e:
            print(f"[variance] ICC failed for {roi}: {e}")
            icc_values[roi] = float("nan")
    return pd.Series(icc_values, name="ICC")


def compute_cv(
    data: pd.DataFrame,
    roi_cols: list[str],
    group_col: str,
) -> pd.DataFrame:
    """
    Compute coefficient of variation (CV = std/mean) per ROI per group.

    Parameters
    ----------
    data : pd.DataFrame
    roi_cols : list of str
    group_col : str
        Column to group by (e.g. "scanner", "site").

    Returns
    -------
    pd.DataFrame with shape (n_groups, n_rois). Values are CV × 100 (percent).
    """
    cv_df = (
        data.groupby(group_col)[roi_cols]
        .apply(lambda g: g.std() / g.mean() * 100)
    )
    return cv_df


def compute_cv_intra_inter(
    data: pd.DataFrame,
    roi_cols: list[str],
    scanner_col: str,
    session_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute intra-scanner and inter-scanner CV separately.

    These are the two complementary reliability metrics used in the pipeline:

    - **Intra-scanner CV** (test-retest): CV computed within each scanner-session
      group (e.g. all PRISMA run-1 subjects). Captures subject-level variability
      within a single acquisition context. Low intra-CV = stable measurement.
    - **Inter-scanner CV**: CV computed grouping all sessions of the same scanner
      (e.g. all PRISMA sessions together vs all TRIO sessions together). High
      inter-scanner CV = scanner has an outlier distribution.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain scanner_col and session_col as separate columns.
    roi_cols : list of str
    scanner_col : str
        Column identifying the scanner (e.g. "PRISMA", "TRIO").
    session_col : str
        Column identifying the session run number within a scanner (e.g. "1", "2").

    Returns
    -------
    (intra_cv_df, inter_cv_df)
        Both DataFrames have shape (n_groups, n_rois), values are CV × 100 (%).
    """
    data = data.copy()
    data["scanner_session"] = (
        data[scanner_col].astype(str) + "_" + data[session_col].astype(str)
    )
    intra_cv = compute_cv(data, roi_cols, group_col="scanner_session")
    inter_cv = compute_cv(data, roi_cols, group_col=scanner_col)
    return intra_cv, inter_cv


def compute_dice_scores(
    mask_paths_a: list[Path],
    mask_paths_b: list[Path],
) -> np.ndarray:
    """
    Compute pairwise Dice similarity coefficients between two sets of binary masks.

    Used to compare skull-stripping methods: a Dice score near 1 indicates
    near-identical brain masks; lower values indicate disagreement at the
    brain boundary.

    Parameters
    ----------
    mask_paths_a : list of Path
        Paths to NIfTI binary masks from method A (e.g. SynthStrip).
    mask_paths_b : list of Path
        Paths to NIfTI binary masks from method B (e.g. ROBEX).
        Must be the same length as mask_paths_a and in the same subject order.

    Returns
    -------
    np.ndarray of shape (n_subjects,) with Dice coefficients in [0, 1].
    """
    if len(mask_paths_a) != len(mask_paths_b):
        raise ValueError("mask_paths_a and mask_paths_b must have the same length.")

    dice_scores = []
    for path_a, path_b in zip(mask_paths_a, mask_paths_b):
        mask_a = nib.load(path_a).get_fdata() > 0
        mask_b = nib.load(path_b).get_fdata() > 0
        intersection = np.sum(mask_a & mask_b)
        union = np.sum(mask_a) + np.sum(mask_b)
        dice = (2.0 * intersection / union) if union > 0 else 1.0
        dice_scores.append(float(dice))

    return np.array(dice_scores)


def compute_scanner_statistics(
    data: pd.DataFrame,
    roi_cols: list[str],
    subject_col: str,
    scanner_col: str,
    session_col: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Paired statistical tests comparing ROI volumes across scanners and sessions.

    Uses pingouin repeated-measures ANOVA and pairwise t-tests to quantify
    scanner and session effects on each ROI. This mirrors the analysis from
    the pipeline development on the Cam-CAN Trio → Prisma transition.

    Parameters
    ----------
    data : pd.DataFrame
        Long format: one row per subject-scanner-session combination.
    roi_cols : list of str
    subject_col : str
    scanner_col : str
    session_col : str, optional
        If provided, tests session effects within each scanner in addition to
        the between-scanner comparison.

    Returns
    -------
    dict with keys:
        ``"anova"``          : rm-ANOVA results (scanner as within-factor)
        ``"scanner_ttest"``  : pairwise t-tests between scanner groups
        ``"session_ttest"``  : pairwise t-tests between sessions (if session_col given)
    """
    try:
        import pingouin as pg
    except ImportError as e:
        raise ImportError("pingouin is required: pip install pingouin") from e

    within = [scanner_col] if session_col is None else [scanner_col, session_col]

    anova_list, scanner_tt_list, session_tt_list = [], [], []

    for roi in roi_cols:
        try:
            anova = pg.rm_anova(
                data=data, dv=roi, subject=subject_col,
                within=within, detailed=True,
            )
            anova["roi"] = roi
            anova_list.append(anova)

            sc_tt = pg.pairwise_ttests(
                data=data, dv=roi, within=scanner_col,
                subject=subject_col, padjust="fdr_bh",
            )
            sc_tt["roi"] = roi
            scanner_tt_list.append(sc_tt)

            if session_col is not None:
                ses_tt = pg.pairwise_ttests(
                    data=data, dv=roi, within=session_col,
                    subject=subject_col, padjust="fdr_bh",
                )
                ses_tt["roi"] = roi
                session_tt_list.append(ses_tt)

        except Exception as e:
            print(f"[variance] Stats failed for {roi}: {e}")

    results: dict[str, pd.DataFrame] = {}
    if anova_list:
        results["anova"] = pd.concat(anova_list, ignore_index=True)
    if scanner_tt_list:
        results["scanner_ttest"] = pd.concat(scanner_tt_list, ignore_index=True)
    if session_tt_list:
        results["session_ttest"] = pd.concat(session_tt_list, ignore_index=True)

    return results


def decompose_variance(
    data: pd.DataFrame,
    roi_cols: list[str],
    sources: list[str],
) -> pd.DataFrame:
    """
    Partition total ROI variance into contributions from each source.

    Uses a sequential ANOVA (type I SS) decomposition. Each source is a
    categorical column in `data` (e.g. "scanner", "site", "manufacturer").
    Residual variance is also returned as "residual".

    Parameters
    ----------
    data : pd.DataFrame
    roi_cols : list of str
    sources : list of str
        Ordered list of variance sources to decompose. Column names in data.

    Returns
    -------
    pd.DataFrame with ROIs as rows, sources + "residual" as columns.
    Values are percentage of total variance explained.
    """
    missing = [s for s in sources if s not in data.columns]
    if missing:
        raise ValueError(f"Source columns not found in data: {missing}")

    results = []

    for roi in roi_cols:
        total_var = data[roi].var()
        if total_var == 0:
            results.append({s: 0.0 for s in sources} | {"residual (biological+noise)": 0.0, "roi": roi})
            continue

        explained = {}
        residual_data = data[roi].copy()

        for source in sources:
            group_means = data.groupby(source)[roi].transform("mean")
            current_mean = residual_data.mean()
            ss_source = ((group_means - current_mean) ** 2).sum()
            explained[source] = ss_source / (total_var * len(data)) * 100
            residual_data = residual_data - (group_means - current_mean)

        total_explained = sum(explained.values())
        residual_pct = max(0.0, 100.0 - total_explained)

        results.append({**explained, "residual (biological+noise)": residual_pct, "roi": roi})

    df = pd.DataFrame(results).set_index("roi")
    return df


def run_variance_analysis(
    data: pd.DataFrame,
    roi_cols: list[str],
    batch_col: str,
    subject_col: str,
    variance_sources: list[str],
    output_dir: Path,
    variant_label: str = "default",
    icc_type: str = "ICC(A,1)",
    session_col: str | None = None,
) -> dict:
    """
    Run the full variance analysis suite and save results as JSON.

    Computes ICC, CV, and variance decomposition for a set of ROIs, then
    writes results to output_dir for downstream QC reporting.

    Parameters
    ----------
    data : pd.DataFrame
        Wide-format: one row per subject, ROI cols + metadata cols.
    roi_cols : list of str
    batch_col : str
        Scanner column — used as the rater in ICC and the grouping variable in CV.
    subject_col : str
    variance_sources : list of str
        Columns to include in variance decomposition.
    output_dir : Path
        Directory to write JSON result files.
    variant_label : str
        Pipeline variant identifier (e.g. "synthstrip_zscore").
    icc_type : str
    session_col : str, optional
        If provided alongside batch_col, computes separate intra-scanner CV
        (within-session) and inter-scanner CV (across scanners), saving both
        as ``cv_intra.json`` and ``cv_inter.json`` in addition to the combined
        ``cv_by_scanner.json``.

    Returns
    -------
    dict with keys "icc", "cv_by_scanner", "variance_decomposition".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[variance] Analysing {len(roi_cols)} ROIs for variant: {variant_label}")

    # --- ICC ---
    icc_series = compute_icc_batch(data, roi_cols, subject_col, batch_col, icc_type)
    icc_dict = icc_series.to_dict()

    n_flagged = sum(1 for v in icc_dict.values() if not np.isnan(v) and v < 0.75)
    print(f"[variance] ICC: {n_flagged}/{len(roi_cols)} ROIs below 0.75 threshold")

    # --- CV ---
    cv_df = compute_cv(data, roi_cols, batch_col)
    cv_dict = cv_df.to_dict()

    if session_col is not None and session_col in data.columns:
        intra_cv, inter_cv = compute_cv_intra_inter(
            data, roi_cols, batch_col, session_col
        )
        with open(output_dir / "cv_intra.json", "w") as f:
            json.dump(intra_cv.to_dict(), f, indent=2)
        with open(output_dir / "cv_inter.json", "w") as f:
            json.dump(inter_cv.to_dict(), f, indent=2)
        print(f"[variance] Intra/inter scanner CV saved separately")

    # --- Variance decomposition ---
    avail_sources = [s for s in variance_sources if s in data.columns]
    vd_df = decompose_variance(data, roi_cols, avail_sources)
    vd_dict = vd_df.to_dict(orient="index")

    results = {
        "variant": variant_label,
        "icc": icc_dict,
        "cv_by_scanner": cv_dict,
        "variance_decomposition": vd_dict,
    }

    # Save
    with open(output_dir / "icc_results.json", "w") as f:
        json.dump(icc_dict, f, indent=2)
    with open(output_dir / "cv_by_scanner.json", "w") as f:
        json.dump(cv_dict, f, indent=2)

    vd_df.to_csv(output_dir / "variance_decomposition.csv")
    print(f"[variance] Results saved to {output_dir}")

    return results


def compute_variance_components(
    data: pd.DataFrame,
    roi_cols: list[str],
    subject_col: str,
    scanner_col: str,
) -> pd.DataFrame:
    """
    Decompose ROI variance into between-subject, between-scanner, and
    within-scanner (test-retest) components.

    Requires at least 2 rows per (subject, scanner) pair (i.e. session repeats).
    Components are computed directly from group statistics — not sequential ANOVA —
    so they are non-overlapping and sum to 100%.

    Returns
    -------
    pd.DataFrame with ROIs as index and columns:
        between_subject, between_scanner, within_scanner (all % of total).
    """
    results = []
    for roi in roi_cols:
        subj_means = data.groupby(subject_col)[roi].mean()
        subj_scan_means = data.groupby([subject_col, scanner_col])[roi].mean()

        var_bw_subj = float(subj_means.var())

        subj_mean_expanded = subj_scan_means.groupby(level=0).transform("mean")
        scanner_devs = subj_scan_means - subj_mean_expanded
        var_bw_scan = float((scanner_devs ** 2).mean())

        var_within = float(data.groupby([subject_col, scanner_col])[roi].var().mean())

        total = var_bw_subj + var_bw_scan + var_within
        if total == 0:
            results.append({"roi": roi, "between_subject": 0.0,
                             "between_scanner": 0.0, "within_scanner": 0.0})
        else:
            results.append({
                "roi": roi,
                "between_subject": round(var_bw_subj / total * 100, 1),
                "between_scanner": round(var_bw_scan / total * 100, 1),
                "within_scanner":  round(var_within  / total * 100, 1),
            })
    return pd.DataFrame(results).set_index("roi")


def detect_design(
    data: pd.DataFrame,
    subject_col: str,
    scanner_col: str,
    manufacturer_col: str | None = None,
    session_col: str | None = None,
) -> dict:
    """
    Inspect a DataFrame and return design flags for conditional figure selection.

    Returns
    -------
    dict with keys:
        n_scanners      : int  — number of unique scanners
        n_manufacturers : int  — number of unique manufacturers (1 if col absent)
        is_paired       : bool — every subject appears on every scanner
        has_sessions    : bool — any (subject, scanner) pair has >1 row
    """
    n_scanners = data[scanner_col].nunique()
    n_manufacturers = (
        data[manufacturer_col].nunique()
        if manufacturer_col and manufacturer_col in data.columns else 1
    )
    subjects_per_scanner = data.groupby(scanner_col)[subject_col].nunique()
    is_paired = bool((subjects_per_scanner == data[subject_col].nunique()).all())
    if session_col and session_col in data.columns:
        has_sessions = data[session_col].nunique() > 1
    else:
        has_sessions = bool(data.groupby([subject_col, scanner_col]).size().max() > 1)
    return {
        "n_scanners": n_scanners,
        "n_manufacturers": n_manufacturers,
        "is_paired": is_paired,
        "has_sessions": has_sessions,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run variance decomposition on ROI CSV")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--batch_col", required=True, help="Scanner/site column")
    parser.add_argument("--subject_col", required=True)
    parser.add_argument("--sources", nargs="+", required=True, help="Variance source columns")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variant", default="default")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, index_col=0)
    meta_cols = {args.batch_col, args.subject_col, *args.sources}
    rois = [c for c in df.columns if c not in meta_cols]

    run_variance_analysis(
        data=df,
        roi_cols=rois,
        batch_col=args.batch_col,
        subject_col=args.subject_col,
        variance_sources=args.sources,
        output_dir=Path(args.output_dir),
        variant_label=args.variant,
    )
