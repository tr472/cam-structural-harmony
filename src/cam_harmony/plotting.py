"""
Plotting module for cam-structural-harmony.

Generates comparison figures for pre/post harmonisation outputs and
pipeline variant evaluations. All plots are saved as publication-quality
PNG files with optional PDF output.

Key figures
-----------
plot_icc_comparison      : ICC bar chart comparing pipeline variants
plot_cv_heatmap          : CV% per scanner per ROI (heatmap)
plot_variance_pie        : Variance source breakdown per ROI
plot_roi_distributions   : Volume distributions pre/post harmonisation
plot_pipeline_summary    : Overview grid for a single variant
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import seaborn as sns


# ── Style defaults ────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_PALETTE = "Set2"


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_icc_comparison(
    icc_results: dict[str, dict[str, float]],
    output_path: Path,
    threshold: float = 0.75,
    top_n: int = 20,
    title: str = "ICC by pipeline variant",
) -> None:
    """
    Bar chart comparing ICC values across pipeline variants for top ROIs.

    Parameters
    ----------
    icc_results : dict
        {variant_label: {roi_name: icc_value}}
    output_path : Path
    threshold : float
        Horizontal line drawn at this ICC value (default 0.75).
    top_n : int
        Show the top N ROIs by mean ICC (prevents crowded axes).
    title : str
    """
    df = pd.DataFrame(icc_results)  # rows = ROIs, cols = variants

    # Focus on most interesting ROIs (highest mean ICC variance across methods)
    roi_spread = df.std(axis=1).nlargest(top_n).index
    df_plot = df.loc[roi_spread]

    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot.plot(kind="bar", ax=ax, colormap=_PALETTE, width=0.75)

    ax.axhline(threshold, color="crimson", linestyle="--", linewidth=1.2,
               label=f"Threshold ({threshold})")
    ax.set_ylabel("ICC")
    ax.set_xlabel("ROI")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    _save(fig, output_path)


def plot_cv_heatmap(
    cv_data: pd.DataFrame,
    output_path: Path,
    title: str = "CV (%) by scanner and ROI",
    top_n: int = 30,
) -> None:
    """
    Heatmap of coefficient of variation per scanner per ROI.

    Parameters
    ----------
    cv_data : pd.DataFrame
        Shape (n_scanners, n_rois). Values are CV × 100.
    output_path : Path
    top_n : int
        Show only the top N highest-CV ROIs.
    """
    # Select ROIs with highest mean CV (most scanner-sensitive)
    top_rois = cv_data.mean(axis=0).nlargest(top_n).index
    df_plot = cv_data[top_rois]

    fig, ax = plt.subplots(figsize=(14, max(4, len(df_plot) * 0.6)))
    sns.heatmap(
        df_plot,
        ax=ax,
        cmap="YlOrRd",
        annot=len(df_plot) <= 10,
        fmt=".1f",
        linewidths=0.3,
        cbar_kws={"label": "CV (%)"},
    )
    ax.set_title(title)
    ax.set_xlabel("ROI")
    ax.set_ylabel("Scanner")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    _save(fig, output_path)


def plot_variance_decomposition(
    vd_df: pd.DataFrame,
    output_path: Path,
    title: str = "Variance decomposition by source",
    top_n: int = 15,
) -> None:
    """
    Stacked bar chart showing variance source breakdown per ROI.

    Parameters
    ----------
    vd_df : pd.DataFrame
        Output of variance.decompose_variance(). Rows = ROIs, columns = sources.
        Values are percentage of total variance.
    output_path : Path
    top_n : int
        Show only the top N ROIs by technical (non-residual) variance.
    """
    technical_cols = [c for c in vd_df.columns if not c.startswith("residual")]
    technical_total = vd_df[technical_cols].sum(axis=1)
    top_rois = technical_total.nlargest(top_n).index
    df_plot = vd_df.loc[top_rois]

    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot.plot(kind="bar", stacked=True, ax=ax, colormap=_PALETTE, width=0.8)
    ax.set_ylabel("Variance explained (%)")
    ax.set_xlabel("ROI")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    _save(fig, output_path)


def plot_roi_distributions(
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    roi_cols: list[str],
    batch_col: str,
    output_path: Path,
    n_cols: int = 3,
) -> None:
    """
    Grid of violin plots showing ROI volume distributions pre/post harmonisation.

    Parameters
    ----------
    pre_data : pd.DataFrame
    post_data : pd.DataFrame
    roi_cols : list of str
        ROIs to plot (up to 12 for readability).
    batch_col : str
        Column used to colour by scanner/site.
    output_path : Path
    n_cols : int
        Number of columns in the subplot grid.
    """
    roi_cols = roi_cols[:12]  # cap for readability
    n_rois = len(roi_cols)
    n_rows = int(np.ceil(n_rois / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols * 2,
                              figsize=(n_cols * 7, n_rows * 4),
                              squeeze=False)

    for idx, roi in enumerate(roi_cols):
        row, col_pair = divmod(idx, n_cols)
        ax_pre = axes[row][col_pair * 2]
        ax_post = axes[row][col_pair * 2 + 1]

        for ax, df, label in [(ax_pre, pre_data, "Pre"), (ax_post, post_data, "Post")]:
            sns.violinplot(
                data=df, x=batch_col, y=roi, ax=ax,
                palette=_PALETTE, inner="box", linewidth=0.8,
            )
            ax.set_title(f"{roi} — {label}", fontsize=9)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30, labelsize=7)

    # Hide unused axes
    for idx in range(n_rois, n_rows * n_cols):
        row, col_pair = divmod(idx, n_cols)
        axes[row][col_pair * 2].set_visible(False)
        axes[row][col_pair * 2 + 1].set_visible(False)

    fig.suptitle("ROI distributions: pre vs post harmonisation", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, output_path)


def plot_pipeline_summary(
    icc_results: dict[str, float],
    cv_df: pd.DataFrame,
    vd_df: pd.DataFrame,
    output_path: Path,
    variant_label: str = "",
    icc_threshold: float = 0.75,
) -> None:
    """
    3-panel summary figure for a single pipeline variant.

    Panels: (1) ICC distribution, (2) mean CV per scanner, (3) variance sources.

    Parameters
    ----------
    icc_results : dict
        {roi_name: icc_value}
    cv_df : pd.DataFrame
        (n_scanners, n_rois)
    vd_df : pd.DataFrame
        (n_rois, n_sources)
    output_path : Path
    variant_label : str
    icc_threshold : float
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: ICC distribution
    icc_vals = [v for v in icc_results.values() if not np.isnan(v)]
    axes[0].hist(icc_vals, bins=20, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[0].axvline(icc_threshold, color="crimson", linestyle="--",
                    label=f"Threshold ({icc_threshold})")
    axes[0].set_xlabel("ICC")
    axes[0].set_ylabel("Number of ROIs")
    axes[0].set_title("ICC distribution")
    axes[0].legend(fontsize=9)

    # Panel 2: Mean CV per scanner
    mean_cv = cv_df.mean(axis=1).sort_values()
    mean_cv.plot(kind="barh", ax=axes[1], color="darkorange")
    axes[1].set_xlabel("Mean CV (%)")
    axes[1].set_title("Mean CV by scanner")

    # Panel 3: Mean variance decomposition
    mean_vd = vd_df.mean(axis=0).sort_values(ascending=False)
    mean_vd.plot(kind="bar", ax=axes[2], colormap=_PALETTE)
    axes[2].set_ylabel("Mean variance explained (%)")
    axes[2].set_title("Variance sources (mean across ROIs)")
    axes[2].tick_params(axis="x", rotation=30)

    title = f"Pipeline variant: {variant_label}" if variant_label else "Pipeline summary"
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)


# ── Pre-processing QC plots ───────────────────────────────────────────────────

def plot_intensity_histograms(
    intensities: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    n_bins: int = 140,
    sigma: float = 2.0,
    title: str = "Intensity distributions by normalisation method",
) -> None:
    """
    Multi-panel smoothed histogram comparing intensity distributions across methods.

    One subplot per normalisation method; each session/scanner plotted as a
    separate coloured line. Histograms are smoothed with a Gaussian kernel to
    remove noise from the bin count step.

    Parameters
    ----------
    intensities : dict
        Nested dict: ``{method_label: {session_label: intensity_values_array}}``.
        Intensity values should be 1-D numpy arrays of within-mask voxel values.
    output_path : Path
    n_bins : int
        Number of histogram bins (default 140).
    sigma : float
        Gaussian smoothing sigma applied to raw bin counts (default 2.0).
    title : str
    """
    methods = list(intensities.keys())
    n = len(methods)
    n_cols = min(n, 3)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    palette = plt.get_cmap("tab10")

    for ax, method in zip(axes, methods):
        session_arrays = intensities[method]
        for idx, (session_label, values) in enumerate(session_arrays.items()):
            hist, bins = np.histogram(values, bins=n_bins)
            smoothed = ndi.gaussian_filter1d(hist.astype(float), sigma=sigma)
            ax.plot(bins[:-1], smoothed, color=palette(idx % 10), label=session_label)

        ax.set_title(method, fontsize=10)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count (smoothed)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.4)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, output_path)


def plot_bland_altman(
    values_a: np.ndarray,
    values_b: np.ndarray,
    label_a: str = "Method A",
    label_b: str = "Method B",
    output_path: Path | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, dict]:
    """
    Bland-Altman plot for comparing two sets of measurements.

    Commonly used to compare skull-strip methods (volume of mask A vs B) or
    pre/post harmonisation ROI volumes.

    Parameters
    ----------
    values_a, values_b : np.ndarray
        Paired measurements. Must be the same length.
    label_a, label_b : str
        Labels for the two methods.
    output_path : Path, optional
        If provided, saves the figure.
    title : str, optional
        Defaults to ``"Bland-Altman: {label_a} vs {label_b}"``.

    Returns
    -------
    (fig, stats_dict) where stats_dict has keys:
        mean_diff, std_diff, upper_loa, lower_loa.
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    means = (values_a + values_b) / 2.0
    diffs = values_a - values_b

    mean_diff = float(np.mean(diffs))
    std_diff  = float(np.std(diffs, ddof=1))
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(means, diffs, alpha=0.6, color="steelblue", s=40)
    ax.axhline(mean_diff, color="crimson",  linestyle="--", linewidth=1.5,
               label=f"Mean diff ({mean_diff:.3f})")
    ax.axhline(upper_loa, color="grey", linestyle=":",  linewidth=1.2,
               label=f"+1.96 SD ({upper_loa:.3f})")
    ax.axhline(lower_loa, color="grey", linestyle=":",  linewidth=1.2,
               label=f"−1.96 SD ({lower_loa:.3f})")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)

    ax.set_xlabel(f"Mean of {label_a} and {label_b}")
    ax.set_ylabel(f"Difference ({label_a} − {label_b})")
    ax.set_title(title or f"Bland-Altman: {label_a} vs {label_b}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path is not None:
        _save(fig, output_path)

    stats = {
        "mean_diff": mean_diff,
        "std_diff":  std_diff,
        "upper_loa": upper_loa,
        "lower_loa": lower_loa,
    }
    return fig, stats


def plot_scanner_scatter(
    data: pd.DataFrame,
    scanner_col: str,
    roi_cols: list[str],
    scanners: tuple[str, str],
    output_path: Path,
    n_cols: int = 4,
    title: str | None = None,
) -> None:
    """
    Scatter plot of ROI volumes from one scanner against another.

    Each ROI is shown as a separate coloured point series. Values are scaled
    0-1 within each ROI so that all ROIs fit on the same axes regardless of
    absolute size. The identity line (y = x) indicates perfect agreement.

    Reproduces the between-scanner ROI comparison used during pipeline
    development on the Cam-CAN Trio → Prisma dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Wide format; one row per subject, ROI cols + scanner_col.
    scanner_col : str
        Column whose unique values identify scanners.
    roi_cols : list of str
        ROI columns to plot.
    scanners : (str, str)
        The two scanner labels to compare, e.g. ("TRIO", "PRISMA").
    output_path : Path
    n_cols : int
        Subplot grid columns (one panel per strip x norm combination is not
        needed here — a single panel is generated).
    title : str, optional
    """
    scanner_a, scanner_b = scanners
    group_a = data[data[scanner_col] == scanner_a][roi_cols]
    group_b = data[data[scanner_col] == scanner_b][roi_cols]

    # Align on index (subject)
    idx = group_a.index.intersection(group_b.index)
    if len(idx) == 0:
        print(f"[plotting] No matched subjects between {scanner_a} and {scanner_b} — skipping scatter.")
        return

    a = group_a.loc[idx]
    b = group_b.loc[idx]

    palette = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8, 8))

    for i, roi in enumerate(roi_cols):
        x = a[roi].values.astype(float)
        y = b[roi].values.astype(float)
        rng = max(x.max() - x.min(), y.max() - y.min(), 1e-8)
        x_s = (x - x.min()) / rng
        y_s = (y - y.min()) / rng
        ax.scatter(x_s, y_s, s=60, alpha=0.7, color=palette(i % 20),
                   edgecolors="k", linewidths=0.3, label=roi)

    ax.plot([0, 1], [0, 1], "r--", linewidth=1.2, label="Identity")
    ax.set_xlabel(f"{scanner_a} (scaled)")
    ax.set_ylabel(f"{scanner_b} (scaled)")
    ax.set_title(title or f"ROI volumes: {scanner_a} vs {scanner_b} (scaled per ROI)")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, output_path)


def plot_icc_pre_post(
    icc_pre: dict[str, float],
    icc_post: dict[str, float],
    output_path: Path,
    threshold: float = 0.75,
    top_n: int = 20,
    title: str = "ICC before and after NeuroCombat harmonisation",
    label_pre: str = "Pre-harmonisation",
    label_post: str = "Post-harmonisation",
) -> None:
    """
    Side-by-side bar chart comparing per-ROI ICC before and after harmonisation.

    Focuses on the ROIs that changed the most (by absolute delta) so the chart
    is not crowded. A well-calibrated harmonisation raises ICC for
    scanner-sensitive ROIs without degrading already-reliable ones.

    Parameters
    ----------
    icc_pre : dict
        {roi_name: icc_value} before harmonisation.
    icc_post : dict
        {roi_name: icc_value} after harmonisation.
    output_path : Path
    threshold : float
        Reliability threshold line (default 0.75).
    top_n : int
        Show the N ROIs with the largest absolute ICC change.
    title : str
    """
    all_rois = sorted(set(icc_pre) | set(icc_post))
    df = pd.DataFrame(
        {
            "Pre":  [icc_pre.get(r,  float("nan")) for r in all_rois],
            "Post": [icc_post.get(r, float("nan")) for r in all_rois],
        },
        index=all_rois,
    )
    df["delta"] = df["Post"] - df["Pre"]
    df_plot = df.loc[df["delta"].abs().nlargest(top_n).index].sort_values("delta", ascending=False)

    x = np.arange(len(df_plot))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, df_plot["Pre"],  width, label=label_pre,
           color="steelblue",  alpha=0.85)
    ax.bar(x + width / 2, df_plot["Post"], width, label=label_post,
           color="darkorange", alpha=0.85)
    ax.axhline(threshold, color="crimson", linestyle="--", linewidth=1.2,
               label=f"Threshold ({threshold})")

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ICC")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    _save(fig, output_path)


# ── Batch entry point ─────────────────────────────────────────────────────────

def generate_all_figures(
    results_dir: Path,
    figures_dir: Path,
    pre_data: pd.DataFrame | None = None,
    post_data: pd.DataFrame | None = None,
    batch_col: str = "scanner",
    roi_cols: list[str] | None = None,
) -> list[Path]:
    """
    Generate the full set of comparison figures from pipeline result files.

    Reads JSON outputs from variance.run_variance_analysis() and optional
    pre/post harmonisation DataFrames, then produces all standard figures.

    Parameters
    ----------
    results_dir : Path
        Contains icc_results.json, cv_by_scanner.json, variance_decomposition.csv.
    figures_dir : Path
        Output directory for figures.
    pre_data : pd.DataFrame, optional
        Raw (pre-harmonisation) ROI volumes with metadata columns.
    post_data : pd.DataFrame, optional
        Harmonised ROI volumes.
    batch_col : str
    roi_cols : list of str, optional

    Returns
    -------
    List of paths to generated figure files.
    """
    import json

    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    # ── Cross-variant ICC comparison ──────────────────────────────────────────
    # results_dir is typically output_dir/variance_results/, which contains
    # one subdirectory per pipeline variant (e.g. synthstrip_zscore/).
    # Load ICC from every variant subdirectory and plot them side-by-side.
    variant_icc: dict[str, dict[str, float]] = {}
    for variant_dir in sorted(results_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        vicc_path = variant_dir / "icc_results.json"
        if vicc_path.exists():
            with open(vicc_path) as f:
                variant_icc[variant_dir.name] = json.load(f)

    icc_data: dict | None = None
    cv_df: pd.DataFrame | None = None
    vd_df: pd.DataFrame | None = None

    if len(variant_icc) > 1:
        p = figures_dir / "icc_cross_variant.png"
        plot_icc_comparison(
            variant_icc, output_path=p,
            title="ICC across pipeline variants (skull-strip × normalisation)",
        )
        saved.append(p)
        # Use the variant with the highest mean ICC for the single-variant panels
        best = max(variant_icc, key=lambda v: np.nanmean(list(variant_icc[v].values())))
        icc_data = variant_icc[best]
        best_dir = results_dir / best
    elif len(variant_icc) == 1:
        best, icc_data = next(iter(variant_icc.items()))
        best_dir = results_dir / best
        p = figures_dir / "icc_comparison.png"
        plot_icc_comparison({best: icc_data}, output_path=p)
        saved.append(p)
    else:
        # Fall back: maybe results_dir IS a single variant directory
        best_dir = results_dir
        icc_path = results_dir / "icc_results.json"
        if icc_path.exists():
            with open(icc_path) as f:
                icc_data = json.load(f)
            p = figures_dir / "icc_comparison.png"
            plot_icc_comparison({"variant": icc_data}, output_path=p)
            saved.append(p)

    # ── CV heatmap (best variant or only variant) ─────────────────────────────
    cv_path = best_dir / "cv_by_scanner.json"
    if cv_path.exists():
        with open(cv_path) as f:
            cv_raw = json.load(f)
        cv_df = pd.DataFrame(cv_raw)
        p = figures_dir / "cv_heatmap.png"
        plot_cv_heatmap(cv_df.T, output_path=p)
        saved.append(p)

    # ── Variance decomposition ────────────────────────────────────────────────
    vd_path = best_dir / "variance_decomposition.csv"
    if vd_path.exists():
        vd_df = pd.read_csv(vd_path, index_col=0)
        p = figures_dir / "variance_decomposition.png"
        plot_variance_decomposition(vd_df, output_path=p)
        saved.append(p)

    # ── Pre/post ICC comparison ───────────────────────────────────────────────
    # For each variant that has both icc_results.json and icc_results_post.json,
    # generate a before/after bar chart. Uses the best variant when multiple exist.
    post_icc_path = best_dir / "icc_results_post.json"
    if icc_data is not None and post_icc_path.exists():
        with open(post_icc_path) as f:
            icc_post_data = json.load(f)
        p = figures_dir / "icc_pre_post.png"
        plot_icc_pre_post(icc_data, icc_post_data, output_path=p)
        saved.append(p)

    # ── 3-panel summary (requires all three) ─────────────────────────────────
    if icc_data is not None and cv_df is not None and vd_df is not None:
        p = figures_dir / "pipeline_summary.png"
        plot_pipeline_summary(icc_data, cv_df.T, vd_df, output_path=p)
        saved.append(p)

    # Pre/post distributions
    if pre_data is not None and post_data is not None:
        rois = roi_cols or [c for c in pre_data.select_dtypes(include="number").columns
                            if c != batch_col][:12]

        p = figures_dir / "roi_distributions.png"
        plot_roi_distributions(pre_data, post_data, rois, batch_col, output_path=p)
        saved.append(p)

        # Scanner scatter (requires exactly two scanners in batch_col)
        scanners = pre_data[batch_col].dropna().unique().tolist()
        if len(scanners) == 2:
            p = figures_dir / "scanner_scatter.png"
            plot_scanner_scatter(
                data=pre_data,
                scanner_col=batch_col,
                roi_cols=rois[:20],
                scanners=(scanners[0], scanners[1]),
                output_path=p,
            )
            saved.append(p)

        # Bland-Altman on first ROI as a method-comparison example
        if rois:
            roi = rois[0]
            vals_a = pre_data[pre_data[batch_col] == scanners[0]][roi].dropna().values
            vals_b = pre_data[pre_data[batch_col] == scanners[-1]][roi].dropna().values
            n = min(len(vals_a), len(vals_b))
            if n >= 3:
                p = figures_dir / f"bland_altman_{roi}.png"
                plot_bland_altman(
                    vals_a[:n], vals_b[:n],
                    label_a=str(scanners[0]),
                    label_b=str(scanners[-1]),
                    output_path=p,
                    title=f"Bland-Altman: {roi}",
                )
                saved.append(p)

    print(f"[plotting] {len(saved)} figures saved to {figures_dir}")
    return saved


# ── Normalized scanner agreement scatter ─────────────────────────────────────

def plot_normalized_scanner_agreement(
    data: pd.DataFrame,
    roi_cols: list,
    scanner_col: str,
    scanners: tuple,
    subject_col: str,
    output_path: Path,
    title: str | None = None,
) -> None:
    """
    Plot normalized ROI volumes for scanner A vs scanner B across all ROIs.

    Each point is one (subject, ROI) pair.  Volumes are normalized by each
    scanner's ROI mean so all ROIs share a common scale (~1.0 = mean) and
    can be overlaid regardless of absolute magnitude.  Points are coloured
    by subject to show that biological ordering is preserved across scanners.
    A y = x identity line marks perfect inter-scanner agreement.

    Parameters
    ----------
    data : pd.DataFrame
        One row per (subject, scanner) — typically roi_df_avg.
    roi_cols : list of str
    scanner_col : str
    scanners : tuple of str
        (scanner_A, scanner_B) — defines x and y axes.
    subject_col : str
    output_path : Path
    title : str, optional
    """
    sc_a, sc_b = scanners
    df_a = data[data[scanner_col] == sc_a].set_index(subject_col)
    df_b = data[data[scanner_col] == sc_b].set_index(subject_col)
    common_subjects = df_a.index.intersection(df_b.index)

    if len(common_subjects) == 0:
        print(f"[plotting] plot_normalized_scanner_agreement: no matched subjects — skipping.")
        return

    df_a = df_a.loc[common_subjects]
    df_b = df_b.loc[common_subjects]

    # Build long arrays: normalize each ROI by its cross-subject mean per scanner
    x_vals, y_vals, subj_idx = [], [], []
    for roi in roi_cols:
        col_a = df_a[roi]
        col_b = df_b[roi]
        mean_a = col_a.mean()
        mean_b = col_b.mean()
        if mean_a == 0 or mean_b == 0:
            continue
        x_vals.extend((col_a / mean_a).tolist())
        y_vals.extend((col_b / mean_b).tolist())
        subj_idx.extend(range(len(common_subjects)))

    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    c_arr = np.array(subj_idx)

    fig, ax = plt.subplots(figsize=(7, 7))

    sc = ax.scatter(
        x_arr, y_arr,
        c=c_arr,
        cmap="tab20" if len(common_subjects) <= 20 else "viridis",
        alpha=0.55,
        s=22,
        linewidths=0,
    )

    # Identity line
    lim_lo = max(0, min(x_arr.min(), y_arr.min()) - 0.05)
    lim_hi = max(x_arr.max(), y_arr.max()) + 0.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="crimson",
            linewidth=1.2, linestyle="--", label="Identity (y = x)")

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(f"{sc_a}  (volume / scanner mean)", fontsize=11)
    ax.set_ylabel(f"{sc_b}  (volume / scanner mean)", fontsize=11)
    ax.set_title(title or f"Normalized ROI agreement: {sc_a} vs {sc_b}\n"
                           f"({len(common_subjects)} subjects × {len(roi_cols)} ROIs, "
                           f"colour = subject)")
    ax.legend(fontsize=9, loc="upper left")

    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Subject index", fontsize=9)

    plt.tight_layout()
    _save(fig, output_path)


# ── Brain metric rendering ────────────────────────────────────────────────────

# Harvard-Oxford subcortical atlas parcel indices (0-based, maxprob-thr25-1mm)
# Index corresponds to label position in the atlas (background=0, labels start at 1)
_HO_SUBCORTICAL_LABELS: dict[str, int] = {
    "lateral_ventricle_L": 3,
    "lateral_ventricle_R": 14,
    "thalamus_L": 4,
    "thalamus_R": 15,
    "caudate_L": 5,
    "caudate_R": 16,
    "hippocampus_L": 9,
    "hippocampus_R": 19,
    "amygdala_L": 10,
    "amygdala_R": 20,
}


def plot_brain_metric(
    metric_dict: dict,
    output_path: Path,
    metric_name: str = "ICC",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdYlGn",
    title: str | None = None,
) -> None:
    """
    Render ROI-level metric values (ICC, CoV, etc.) onto the Harvard-Oxford
    subcortical atlas and save as a high-DPI brain image.

    Parameters
    ----------
    metric_dict : dict
        Mapping from ROI name to scalar metric value (e.g. results['icc']).
    output_path : Path
        Where to save the PNG.
    metric_name : str
        Label used in the colorbar title.
    vmin, vmax : float, optional
        Colormap range. Defaults to data min/max.
    cmap : str
        Matplotlib colormap name (default 'RdYlGn').
    title : str, optional
        Figure title. Defaults to f'{metric_name} by ROI (Harvard-Oxford subcortical)'.

    Notes
    -----
    ROIs not present in the Harvard-Oxford subcortical atlas (e.g. entorhinal
    cortex) are silently skipped and left uncoloured (NaN / transparent).
    Requires nilearn and nibabel.
    """
    try:
        import nibabel as nib
        from nilearn import datasets, plotting
    except ImportError as exc:
        raise ImportError(
            "nilearn is required for brain rendering. "
            "Install with: pip install nilearn"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load atlas
    atlas_obj = datasets.fetch_atlas_harvard_oxford(
        "sub-maxprob-thr25-1mm", symmetric_split=False
    )
    maps = atlas_obj["maps"]
    # nilearn may return a NIfTI image directly or a file path depending on version
    if isinstance(maps, str):
        atlas_img = nib.load(maps)
    else:
        atlas_img = maps  # already a Nifti1Image
    atlas_data = np.asarray(atlas_img.dataobj)

    # Build float metric volume (NaN for background / unmapped parcels)
    metric_vol = np.full(atlas_data.shape, np.nan, dtype=np.float32)

    mapped_count = 0
    for roi_name, value in metric_dict.items():
        parcel_idx = _HO_SUBCORTICAL_LABELS.get(roi_name)
        if parcel_idx is None:
            continue
        mask = atlas_data == parcel_idx
        if mask.any():
            metric_vol[mask] = float(value)
            mapped_count += 1

    if mapped_count == 0:
        print(f"[plotting] plot_brain_metric: no ROIs mapped to atlas — skipping.")
        return

    metric_img = nib.Nifti1Image(metric_vol, atlas_img.affine, atlas_img.header)

    # Colormap range
    valid_vals = [v for k, v in metric_dict.items() if k in _HO_SUBCORTICAL_LABELS]
    _vmin = vmin if vmin is not None else float(np.nanmin(valid_vals))
    _vmax = vmax if vmax is not None else float(np.nanmax(valid_vals))
    _title = title or f"{metric_name} by ROI (Harvard-Oxford subcortical)"

    # Z-slices that pass through subcortical structures (MNI mm)
    cut_coords = (-20, -10, 0, 10, 20)

    display = plotting.plot_stat_map(
        metric_img,
        display_mode="z",
        cut_coords=cut_coords,
        cmap=cmap,
        vmin=_vmin,
        vmax=_vmax,
        colorbar=True,
        annotate=True,
        draw_cross=False,
        title=_title,
    )
    display.savefig(str(output_path), dpi=150)
    display.close()
    print(f"[plotting] Brain metric map saved: {output_path.name} ({mapped_count} ROIs mapped)")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plotting] Saved: {path.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate figures from variance results")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--figures_dir", required=True)
    parser.add_argument("--pre_csv", default=None, help="Pre-harmonisation CSV")
    parser.add_argument("--post_csv", default=None, help="Post-harmonisation CSV")
    parser.add_argument("--batch_col", default="scanner")
    args = parser.parse_args()

    pre = pd.read_csv(args.pre_csv, index_col=0) if args.pre_csv else None
    post = pd.read_csv(args.post_csv, index_col=0) if args.post_csv else None

    generate_all_figures(
        results_dir=Path(args.results_dir),
        figures_dir=Path(args.figures_dir),
        pre_data=pre,
        post_data=post,
        batch_col=args.batch_col,
    )
