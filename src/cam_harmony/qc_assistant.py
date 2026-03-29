"""
QC Assistant — AI-powered interpretation of variance decomposition results.

Takes structured pipeline outputs (ICC values, CV by scanner group, ComBat
residuals) and generates a natural language quality control report using
Claude. Surfaces actionable flags and summarises cross-pipeline comparison
results without requiring manual inspection of every metric.

Usage
-----
    # Programmatic
    from cam_harmony.qc_assistant import generate_qc_report

    report = generate_qc_report(
        results_dir="outputs/variance_results",
        output_path="outputs/qc_report.md",
        focus_rois=["hippocampus", "lateral_ventricle", "entorhinal"]
    )

    # CLI
    python -m cam_harmony.qc_assistant \
        --results_dir outputs/variance_results \
        --output outputs/qc_report.md \
        --focus_rois hippocampus lateral_ventricle
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic


def load_variance_results(results_dir: str | Path) -> dict:
    """
    Load variance decomposition outputs into a structured summary dict.

    Expects results_dir to contain any of:
      icc_results.json         ICC per ROI per pipeline variant
      cv_by_scanner.json       Coefficient of variation per scanner group
      combat_residuals.json    Post-harmonisation residual variance
      pipeline_variants.json   Skull-strip x norm combinations that were run
    """
    results_dir = Path(results_dir)
    results = {}

    for fname in ["icc_results", "cv_by_scanner", "combat_residuals", "pipeline_variants"]:
        fpath = results_dir / f"{fname}.json"
        if fpath.exists():
            with open(fpath) as f:
                results[fname] = json.load(f)
        else:
            results[fname] = None

    if all(v is None for v in results.values()):
        raise FileNotFoundError(
            f"No variance result files found in {results_dir}. "
            "Run the full pipeline first or check the results directory path."
        )

    return results


def build_prompt(results: dict, focus_rois: list[str] | None = None) -> str:
    """
    Format pipeline results into a structured prompt for the QC report.

    Parameters
    ----------
    results : dict
        Output from load_variance_results.
    focus_rois : list of str, optional
        ROIs to highlight (e.g. ["hippocampus", "lateral_ventricle"]).
    """
    roi_section = ""
    if focus_rois:
        roi_section = (
            f"\nPay particular attention to these ROIs: {', '.join(focus_rois)}. "
            "Flag if any show poor reliability or high scanner sensitivity.\n"
        )

    return f"""You are a neuroimaging QC specialist reviewing outputs from
cam-structural-harmony, a pipeline that evaluates structural MRI harmonisation
across scanners. The pipeline tests combinations of skull-stripping methods
(SynthStrip, ROBEX) and intensity normalisation techniques before running
FreeSurfer and NeuroCombat harmonisation.

Your task is to produce a structured QC report that:
1. Identifies which pipeline variant (skull-strip x normalisation combination)
   performs best overall for removing scanner variance while preserving biology
2. Flags ROIs with poor reliability (ICC < 0.75) or high scanner sensitivity
3. Notes any preprocessing steps that appear to introduce rather than remove variance
4. Assesses how effectively NeuroCombat harmonisation removes scanner effects
5. Gives a clear, actionable recommendation for the preferred pipeline configuration
{roi_section}
Format your response as a markdown report with these sections:
## Executive summary
## Best performing pipeline variant
## ROIs of concern
## Harmonisation effectiveness
## Recommendations

Here are the pipeline results:

Pipeline variants tested:
{json.dumps(results.get("pipeline_variants"), indent=2)}

ICC results by variant (higher = more reliable across scanners, target > 0.75):
{json.dumps(results.get("icc_results"), indent=2)}

Coefficient of variation by scanner group (lower = less scanner sensitivity):
{json.dumps(results.get("cv_by_scanner"), indent=2)}

Post-harmonisation residual variance (lower = harmonisation more effective):
{json.dumps(results.get("combat_residuals"), indent=2)}
"""


def generate_qc_report(
    results_dir: str | Path,
    output_path: str | Path | None = None,
    focus_rois: list[str] | None = None,
) -> str:
    """
    Generate an AI-assisted QC report from pipeline variance results.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing variance decomposition outputs.
    output_path : str or Path, optional
        If provided, saves the report here as a markdown file.
    focus_rois : list of str, optional
        ROIs to highlight in the report.

    Returns
    -------
    str
        The generated QC report in markdown format.

    Example
    -------
    >>> report = generate_qc_report(
    ...     results_dir="outputs/variance_results",
    ...     output_path="outputs/qc_report.md",
    ...     focus_rois=["hippocampus", "lateral_ventricle", "entorhinal"]
    ... )
    """
    results = load_variance_results(results_dir)
    prompt = build_prompt(results, focus_rois=focus_rois)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    report = message.content[0].text

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"QC report saved to {output_path}")

    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate AI-assisted QC report from cam-structural-harmony outputs"
    )
    parser.add_argument(
        "--results_dir", required=True,
        help="Directory containing variance decomposition results"
    )
    parser.add_argument(
        "--output", default="outputs/qc_report.md",
        help="Path to save the markdown report"
    )
    parser.add_argument(
        "--focus_rois", nargs="+",
        help="ROIs to highlight (e.g. --focus_rois hippocampus lateral_ventricle)"
    )
    args = parser.parse_args()

    report = generate_qc_report(
        results_dir=args.results_dir,
        output_path=args.output,
        focus_rois=args.focus_rois,
    )
    print(report)


if __name__ == "__main__":
    main()
