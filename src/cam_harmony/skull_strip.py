"""
Skull stripping module for cam-structural-harmony.

Provides wrappers for SynthStrip and ROBEX, with a unified interface so the
pipeline can swap methods without changing downstream code.

SynthStrip  : Learning-based method (Hoopes et al. 2022). Robust to
              pathology and acquisition differences. Requires FreeSurfer 7.3+.
              Called as `mri_synthstrip` (the FreeSurfer 7.x binary name).
ROBEX       : Registration-based method (Iglesias et al. 2011). Faster than
              SynthStrip; slightly less robust to unusual anatomy. Uses the
              pyrobex Python package — no shell script dependency.

SynthStrip is called as a subprocess; ROBEX uses the pyrobex Python API
directly (nib.load → robex() → nib.save), removing the runROBEX.sh dependency.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def synthstrip(
    input_path: Path,
    output_path: Path,
    mask_path: Path,
    no_csf: bool = False,
    device: str = "cpu",
    synthstrip_bin: str = "mri_synthstrip",
) -> tuple[Path, Path]:
    """
    Run SynthStrip skull stripping on a single T1w image.

    Parameters
    ----------
    input_path : Path
        Path to the input NIfTI image.
    output_path : Path
        Path to write the skull-stripped image.
    mask_path : Path
        Path to write the binary brain mask.
    no_csf : bool
        If True, excludes CSF from the brain mask (--no-csf flag).
    device : str
        Compute device: "cpu" or "cuda".
    synthstrip_bin : str
        Name or path of the SynthStrip executable. FreeSurfer 7.x installs
        this as ``mri_synthstrip``.

    Returns
    -------
    (output_path, mask_path) on success.

    Raises
    ------
    RuntimeError
        If SynthStrip exits with a non-zero return code.
    """
    output_path = Path(output_path)
    mask_path = Path(mask_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        synthstrip_bin,
        "-i", str(input_path),
        "-o", str(output_path),
        "-m", str(mask_path),
    ]
    if no_csf:
        cmd.append("--no-csf")
    if device == "cuda":
        cmd += ["--gpu"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"SynthStrip failed on {input_path}:\n{result.stderr}"
        )

    print(f"[synthstrip] Stripped: {output_path.name}")
    return output_path, mask_path


def robex(
    input_path: Path,
    output_path: Path,
    mask_path: Path,
) -> tuple[Path, Path]:
    """
    Run ROBEX skull stripping on a single T1w image via the pyrobex Python API.

    Uses pyrobex (https://github.com/jcreinhold/pyrobex) rather than shelling
    out to runROBEX.sh, removing any dependency on ROBEX shell scripts being
    on PATH.

    Parameters
    ----------
    input_path : Path
        Path to the input NIfTI image.
    output_path : Path
        Path to write the skull-stripped image.
    mask_path : Path
        Path to write the binary brain mask.

    Returns
    -------
    (output_path, mask_path) on success.

    Raises
    ------
    RuntimeError
        If pyrobex raises an exception.
    """
    import nibabel as nib
    from pyrobex.robex import robex as _robex

    output_path = Path(output_path)
    mask_path = Path(mask_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    img = nib.load(input_path)
    stripped, mask = _robex(img)
    nib.save(stripped, output_path)
    nib.save(mask, mask_path)

    print(f"[robex] Stripped: {output_path.name}")
    return output_path, mask_path


def skull_strip(
    input_path: Path,
    output_dir: Path,
    method: str,
    subject_id: str | None = None,
    no_csf: bool = False,
    device: str = "cpu",
    synthstrip_bin: str = "mri_synthstrip",
) -> tuple[Path, Path]:
    """
    Unified skull stripping interface — dispatches to SynthStrip or ROBEX.

    Output filenames are ``{subject_id}_stripped.nii.gz`` and
    ``{subject_id}_mask.nii.gz``. The method is encoded in the parent
    directory (set by the pipeline variant), not the filename.

    Parameters
    ----------
    input_path : Path
        Path to the T1w NIfTI image.
    output_dir : Path
        Directory to write outputs.
    method : str
        "synthstrip" or "robex".
    subject_id : str, optional
        Filename prefix. Defaults to the input stem.
    no_csf : bool
        Passed to SynthStrip only (--no-csf flag).
    device : str
        Passed to SynthStrip only ("cpu" or "cuda").
    synthstrip_bin : str
        SynthStrip executable name/path (default: ``mri_synthstrip``).

    Returns
    -------
    (stripped_path, mask_path)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if subject_id is None:
        subject_id = input_path.name.replace(".nii.gz", "").replace(".nii", "")

    stripped_path = output_dir / f"{subject_id}_stripped.nii.gz"
    mask_path = output_dir / f"{subject_id}_mask.nii.gz"

    if method == "synthstrip":
        return synthstrip(
            input_path, stripped_path, mask_path,
            no_csf=no_csf, device=device, synthstrip_bin=synthstrip_bin,
        )
    elif method == "robex":
        return robex(input_path, stripped_path, mask_path)
    else:
        raise ValueError(f"Unknown skull stripping method: '{method}'. Choose 'synthstrip' or 'robex'.")


def batch_skull_strip(
    input_paths: list[Path],
    output_dir: Path,
    method: str,
    n_jobs: int = 1,
    **kwargs,
) -> list[tuple[Path, Path]]:
    """
    Run skull stripping on a list of images, optionally in parallel.

    Parameters
    ----------
    input_paths : list of Path
    output_dir : Path
    method : str
        "synthstrip" or "robex".
    n_jobs : int
        Number of parallel workers (uses concurrent.futures.ThreadPoolExecutor).
    **kwargs
        Forwarded to skull_strip().

    Returns
    -------
    List of (stripped_path, mask_path) tuples in the same order as input_paths.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[int, tuple[Path, Path]] = {}

    def _strip(idx: int, path: Path) -> tuple[int, tuple[Path, Path]]:
        return idx, skull_strip(path, output_dir, method, **kwargs)

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_strip, i, p): i for i, p in enumerate(input_paths)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return [results[i] for i in range(len(input_paths))]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Skull strip a single T1w image")
    parser.add_argument("--input", required=True, help="Input NIfTI path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--method", required=True, choices=["synthstrip", "robex"])
    parser.add_argument("--subject_id", default=None)
    parser.add_argument("--no_csf", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    stripped, mask = skull_strip(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        method=args.method,
        subject_id=args.subject_id,
        no_csf=args.no_csf,
        device=args.device,
    )
    print(f"Stripped: {stripped}")
    print(f"Mask:     {mask}")
