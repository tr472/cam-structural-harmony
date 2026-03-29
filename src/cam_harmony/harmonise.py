"""
Harmonisation module for cam-structural-harmony.

Wraps NeuroCombat (Johnson et al. 2007) via the neuroCombat-sklearn interface
to remove scanner/site batch effects from FreeSurfer-derived ROI volumes.

Biological covariates (age, sex, TIV) are preserved during harmonisation so
that clinically relevant variance is not removed along with technical variance.

The module can be called with a pandas DataFrame of ROI volumes, or pointed
at a CSV file produced by freesurfer.extract_aparc_aseg().
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def run_combat(
    data: pd.DataFrame,
    batch_col: str,
    covariate_cols: list[str] | None = None,
    parametric: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Apply NeuroCombat harmonisation to a DataFrame of ROI volumes.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame where rows are subjects and columns include ROI volumes plus
        metadata columns (batch_col, covariate_cols).
    batch_col : str
        Column name identifying the scanner/site batch (e.g. "scanner").
    covariate_cols : list of str, optional
        Biological covariates to preserve (e.g. ["age", "sex", "tiv"]).
        These are passed to ComBat as covariates and are NOT removed.
    parametric : bool
        Use parametric (Gaussian) ComBat (True) or non-parametric (False).
    output_path : Path, optional
        If provided, saves harmonised DataFrame as a CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with same shape as input; ROI columns replaced by harmonised
        values. Metadata columns are preserved unchanged.

    Raises
    ------
    ImportError
        If neuroCombat-sklearn is not installed.
    ValueError
        If batch_col or covariate_cols are missing from data.
    """
    try:
        from neurocombat_sklearn import CombatModel
    except ImportError as e:
        raise ImportError(
            "neuroCombat is required. Install with: pip install neuroCombat-sklearn"
        ) from e

    # Validate columns
    if batch_col not in data.columns:
        raise ValueError(f"batch_col '{batch_col}' not found in DataFrame columns.")
    if covariate_cols:
        missing = [c for c in covariate_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Covariate columns not found: {missing}")

    meta_cols = [batch_col] + (covariate_cols or [])
    roi_cols = [
        c for c in data.select_dtypes(include="number").columns if c not in meta_cols
    ]

    if not roi_cols:
        raise ValueError("No ROI columns found after excluding metadata columns.")

    # CombatModel expects (n_samples, n_features)
    roi_matrix = data[roi_cols].values  # shape: (n_subjects, n_rois)
    batch = data[batch_col].values.reshape(-1, 1)

    print(f"[combat] Harmonising {len(roi_cols)} ROIs across {len(np.unique(batch))} batches ...")
    print(f"[combat] Subjects: {roi_matrix.shape[0]}, parametric={parametric}")

    if covariate_cols:
        numeric_covs = [c for c in covariate_cols if pd.api.types.is_numeric_dtype(data[c])]
        categ_covs = [c for c in covariate_cols if not pd.api.types.is_numeric_dtype(data[c])]
        continuous_covs = data[numeric_covs].values if numeric_covs else None
        discrete_covs = data[categ_covs].values if categ_covs else None
    else:
        continuous_covs = None
        discrete_covs = None

    model = CombatModel()
    harmonised_matrix = model.fit_transform(
        roi_matrix, batch, discrete_covs, continuous_covs
    )
    harmonised_df = data.copy()
    harmonised_df[roi_cols] = harmonised_matrix

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        harmonised_df.to_csv(output_path)
        print(f"[combat] Harmonised data saved to {output_path}")

    return harmonised_df


def compute_combat_residuals(
    pre_harmonised: pd.DataFrame,
    post_harmonised: pd.DataFrame,
    batch_col: str,
    roi_cols: list[str] | None = None,
    output_path: Path | None = None,
) -> dict:
    """
    Quantify residual scanner variance remaining after harmonisation.

    Computes the ratio of between-scanner variance to total variance for each
    ROI, before and after ComBat. A well-harmonised ROI will show a large drop.

    Parameters
    ----------
    pre_harmonised : pd.DataFrame
    post_harmonised : pd.DataFrame
    batch_col : str
    roi_cols : list of str, optional
        ROIs to evaluate. Defaults to all numeric columns excluding batch_col.
    output_path : Path, optional
        Save residual summary as JSON.

    Returns
    -------
    dict mapping roi_name → {"pre": float, "post": float, "reduction_pct": float}
    """
    if roi_cols is None:
        roi_cols = [c for c in pre_harmonised.select_dtypes(include="number").columns
                    if c != batch_col]

    results = {}
    for roi in roi_cols:
        def _between_scanner_var(df: pd.DataFrame) -> float:
            groups = df.groupby(batch_col)[roi]
            grand_mean = df[roi].mean()
            n = len(df)
            between = sum(
                len(g) * (g.mean() - grand_mean) ** 2
                for _, g in groups
            ) / n
            total = df[roi].var()
            return float(between / total) if total > 0 else 0.0

        pre_ratio = _between_scanner_var(pre_harmonised)
        post_ratio = _between_scanner_var(post_harmonised)
        reduction = (pre_ratio - post_ratio) / pre_ratio * 100 if pre_ratio > 0 else 0.0

        results[roi] = {
            "pre": round(pre_ratio, 4),
            "post": round(post_ratio, 4),
            "reduction_pct": round(reduction, 1),
        }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[combat] Residual summary saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply NeuroCombat harmonisation")
    parser.add_argument("--input_csv", required=True, help="CSV with ROI volumes + metadata")
    parser.add_argument("--batch_col", required=True, help="Scanner/site column name")
    parser.add_argument("--covariates", nargs="+", help="Biological covariate column names")
    parser.add_argument("--output_csv", required=True, help="Path to write harmonised CSV")
    parser.add_argument("--no_parametric", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, index_col=0)
    harmonised = run_combat(
        data=df,
        batch_col=args.batch_col,
        covariate_cols=args.covariates,
        parametric=not args.no_parametric,
        output_path=Path(args.output_csv),
    )
    print(f"Harmonised shape: {harmonised.shape}")
