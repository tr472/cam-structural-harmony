"""
FreeSurfer module for cam-structural-harmony.

Runs recon-all on skull-stripped T1w images and extracts volumetric ROI
statistics from the aparc/aseg parcellation. Designed to process BIDS
subject lists in parallel and collect outputs into tidy pandas DataFrames
for downstream harmonisation.

Requires FreeSurfer 7.x with FREESURFER_HOME set, or an explicit path in
config.yaml under paths.freesurfer_home.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def run_recon_all(
    subject_id: str,
    t1w_path: Path,
    subjects_dir: Path,
    freesurfer_home: str | None = None,
) -> Path:
    """
    Run FreeSurfer recon-all on a skull-stripped T1w image.

    Uses the standard external-skull-strip protocol:

    1. ``recon-all -autorecon1 -noskullstrip`` — imports the T1w, runs
       motion correction and talairach registration without skull stripping.
    2. Copy ``T1.mgz`` as the brainmask so FreeSurfer uses our external mask.
    3. ``recon-all -autorecon2 -careg -autorecon3`` — runs the full cortical
       and subcortical reconstruction using the provided brainmask.

    This matches the workflow used during pipeline development on the
    Cam-CAN travelling-heads dataset.

    Parameters
    ----------
    subject_id : str
        FreeSurfer subject label (used as ``-subjid`` argument).
        For session BIDS data this is typically ``sub-{id}_ses-{session}``.
    t1w_path : Path
        Path to the skull-stripped, intensity-normalised T1w NIfTI image.
    subjects_dir : Path
        FreeSurfer SUBJECTS_DIR. Output goes to subjects_dir/subject_id/.
    freesurfer_home : str, optional
        Path to FreeSurfer installation. Falls back to $FREESURFER_HOME.

    Returns
    -------
    Path
        Path to the completed subject directory (subjects_dir/subject_id).

    Raises
    ------
    RuntimeError
        If any recon-all stage exits non-zero.
    EnvironmentError
        If FreeSurfer cannot be located.
    """
    fs_home = freesurfer_home or os.environ.get("FREESURFER_HOME")
    if not fs_home:
        raise EnvironmentError(
            "FreeSurfer not found. Set FREESURFER_HOME or provide "
            "paths.freesurfer_home in config.yaml."
        )

    subjects_dir = Path(subjects_dir)
    subjects_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["FREESURFER_HOME"] = str(fs_home)
    env["SUBJECTS_DIR"] = str(subjects_dir)

    recon_bin = Path(fs_home) / "bin" / "recon-all"

    print(f"[freesurfer] autorecon1 (no skull strip) — {subject_id} ...")
    cmd1 = [
        str(recon_bin),
        "-autorecon1", "-noskullstrip",
        "-i", str(t1w_path),
        "-subjid", subject_id,
        "-sd", str(subjects_dir),
    ]
    r1 = subprocess.run(cmd1, capture_output=True, text=True, env=env)
    if r1.returncode != 0:
        raise RuntimeError(
            f"recon-all autorecon1 failed for {subject_id}:\n{r1.stderr[-2000:]}"
        )

    # Copy T1.mgz as brainmask so FreeSurfer uses our external skull strip
    subj_mri = subjects_dir / subject_id / "mri"
    shutil.copy(subj_mri / "T1.mgz", subj_mri / "brainmask.auto.mgz")
    shutil.copy(subj_mri / "brainmask.auto.mgz", subj_mri / "brainmask.mgz")
    print(f"[freesurfer] Brainmask set from T1.mgz — {subject_id}")

    print(f"[freesurfer] autorecon2 + autorecon3 — {subject_id} ...")
    cmd2 = [
        str(recon_bin),
        "-autorecon2", "-careg", "-autorecon3",
        "-subjid", subject_id,
    ]
    r2 = subprocess.run(cmd2, capture_output=True, text=True, env=env)
    if r2.returncode != 0:
        raise RuntimeError(
            f"recon-all autorecon2/3 failed for {subject_id}:\n{r2.stderr[-2000:]}"
        )

    print(f"[freesurfer] Completed: {subject_id}")
    return subjects_dir / subject_id


def batch_recon_all(
    subject_ids: list[str],
    t1w_paths: list[Path],
    subjects_dir: Path,
    n_jobs: int = 1,
    freesurfer_home: str | None = None,
) -> list[Path]:
    """
    Run recon-all in parallel across subjects.

    Parameters
    ----------
    subject_ids : list of str
    t1w_paths : list of Path
        Must be the same length as subject_ids and in the same order.
    subjects_dir : Path
    n_jobs : int
        Parallel workers. Recommended: ≤ number of CPU cores / 4 (recon-all
        itself is multi-threaded).
    freesurfer_home : str, optional

    Returns
    -------
    List of completed subject directory Paths.
    """
    results: dict[int, Path] = {}

    def _run(idx: int, sid: str, t1w: Path) -> tuple[int, Path]:
        return idx, run_recon_all(sid, t1w, subjects_dir, freesurfer_home)

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(_run, i, sid, t1w): i
            for i, (sid, t1w) in enumerate(zip(subject_ids, t1w_paths))
        }
        for future in as_completed(futures):
            idx, path = future.result()
            results[idx] = path

    return [results[i] for i in range(len(subject_ids))]


def extract_aparc_aseg(
    subjects_dir: Path,
    subject_ids: list[str],
    measure: str = "volume",
    parc: str = "aparc",
) -> pd.DataFrame:
    """
    Extract ROI statistics from aparc/aseg for a list of subjects.

    Calls `asegstats2table` / `aparcstats2table` to aggregate FreeSurfer stats
    files into a single DataFrame. Each row is a subject; columns are ROI labels.

    Parameters
    ----------
    subjects_dir : Path
        FreeSurfer SUBJECTS_DIR.
    subject_ids : list of str
    measure : str
        Statistic to extract: "volume" (default), "thickness", "area".
    parc : str
        Parcellation: "aparc" (Desikan-Killiany) or "aparc.a2009s" (Destrieux).

    Returns
    -------
    pd.DataFrame with shape (n_subjects, n_rois).
    """
    subjects_dir = Path(subjects_dir)

    env = os.environ.copy()
    env["SUBJECTS_DIR"] = str(subjects_dir)

    # Build subject list file for the stats table commands
    subj_list = " ".join(subject_ids)

    tables: dict[str, pd.DataFrame] = {}

    # --- Subcortical volumes from aseg ---
    aseg_cmd = [
        "asegstats2table",
        "--subjects", *subject_ids,
        "--meas", measure,
        "--tablefile", "/dev/stdout",
        "--delimiter", "tab",
    ]
    aseg_result = subprocess.run(aseg_cmd, capture_output=True, text=True, env=env)
    if aseg_result.returncode == 0:
        from io import StringIO
        tables["aseg"] = pd.read_csv(StringIO(aseg_result.stdout), sep="\t", index_col=0)
    else:
        print(f"[freesurfer] Warning: asegstats2table failed — {aseg_result.stderr[:500]}")

    # --- Cortical parcellation (left + right hemispheres) ---
    for hemi in ["lh", "rh"]:
        aparc_cmd = [
            "aparcstats2table",
            "--subjects", *subject_ids,
            "--hemi", hemi,
            "--parc", parc,
            "--meas", measure,
            "--tablefile", "/dev/stdout",
            "--delimiter", "tab",
        ]
        aparc_result = subprocess.run(aparc_cmd, capture_output=True, text=True, env=env)
        if aparc_result.returncode == 0:
            from io import StringIO
            df = pd.read_csv(StringIO(aparc_result.stdout), sep="\t", index_col=0)
            # Prefix columns to avoid clashes between hemispheres
            df.columns = [f"{hemi}_{col}" for col in df.columns]
            tables[hemi] = df
        else:
            print(f"[freesurfer] Warning: aparcstats2table ({hemi}) failed — {aparc_result.stderr[:500]}")

    if not tables:
        raise RuntimeError("All FreeSurfer stats extraction commands failed.")

    combined = pd.concat(tables.values(), axis=1)
    combined.index.name = "subject_id"
    print(f"[freesurfer] Extracted {combined.shape[1]} ROIs for {combined.shape[0]} subjects")
    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run recon-all on a single subject")
    parser.add_argument("--subject_id", required=True,
                        help="FreeSurfer subject label (e.g. sub-14_ses-PRISMA1)")
    parser.add_argument("--t1w", required=True, help="Path to skull-stripped T1w")
    parser.add_argument("--subjects_dir", required=True)
    args = parser.parse_args()

    out = run_recon_all(
        subject_id=args.subject_id,
        t1w_path=Path(args.t1w),
        subjects_dir=Path(args.subjects_dir),
    )
    print(f"Completed: {out}")
