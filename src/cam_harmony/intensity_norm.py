"""
Intensity normalisation module for cam-structural-harmony.

Wraps the intensity-normalization package (Reinhold et al. 2019) with a
BIDS-native batch interface. Provides a unified API across 6 methods so
the pipeline can treat normalisation as a single configurable step.

Supported methods
-----------------
zscore      : Z-score standardisation within brain mask
fcm_wm      : Fuzzy C-means white matter peak normalisation (recommended for T1w)
kde         : Kernel density estimation of tissue modes
whitestripe : Normal-appearing white matter standardisation (Shinohara et al. 2016)
nyul        : Piecewise linear histogram matching (Nyul & Udupa 1999) — population method
minmax      : Min-max range scaling to [0, 1] — naive baseline
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from intensity_normalization import (
    FCMNormalizer,
    KDENormalizer,
    NyulNormalizer,
    WhiteStripeNormalizer,
    ZScoreNormalizer,
)

Method = Literal["zscore", "fcm_wm", "kde", "whitestripe", "nyul", "minmax"]

INDIVIDUAL_METHODS = {"zscore", "fcm_wm", "kde", "whitestripe", "minmax"}
POPULATION_METHODS = {"nyul"}


def normalize_image(
    img: nib.Nifti1Image,
    method: Method,
    mask: nib.Nifti1Image | None = None,
) -> nib.Nifti1Image:
    """
    Normalise a single T1w image using the specified method.

    For population methods (nyul), use batch_normalize instead — this
    function will raise an error if called with a population method, since
    those require fitting across subjects before transforming any individual.

    Parameters
    ----------
    img : nibabel image
        T1-weighted MRI volume.
    method : str
        One of: zscore, fcm_wm, kde, whitestripe, nyul, minmax.
    mask : nibabel image, optional
        Binary brain mask. Required for fcm_wm and whitestripe.

    Returns
    -------
    nibabel image with normalised intensities, same affine as input.
    """
    if method in POPULATION_METHODS:
        raise ValueError(
            f"Method '{method}' is a population method and requires fitting "
            "across all subjects. Use batch_normalize() instead."
        )

    if method == "zscore":
        normalizer = ZScoreNormalizer()
        return normalizer.fit_transform(img, mask=mask)

    elif method == "fcm_wm":
        normalizer = FCMNormalizer(tissue_type="wm")
        return normalizer.fit_transform(img, mask=mask)

    elif method == "kde":
        normalizer = KDENormalizer()
        return normalizer.fit_transform(img, mask=mask)

    elif method == "whitestripe":
        normalizer = WhiteStripeNormalizer()
        return normalizer.fit_transform(img, mask=mask)

    elif method == "minmax":
        return _minmax_normalize(img, mask=mask)

    else:
        raise ValueError(f"Unknown method: '{method}'. Choose from {list(Method.__args__)}")


def batch_normalize(
    image_paths: list[Path],
    method: Method,
    output_dir: Path,
    mask_paths: list[Path] | None = None,
) -> list[Path]:
    """
    Normalise a list of images and save outputs.

    Handles population methods (nyul) by fitting across all subjects before
    transforming, ensuring cross-subject comparability.

    Parameters
    ----------
    image_paths : list of Path
        Paths to T1w NIfTI images (.nii or .nii.gz).
    method : str
        Normalisation method.
    output_dir : Path
        Directory to write normalised images.
    mask_paths : list of Path, optional
        Brain masks corresponding to each image. Must match order of image_paths.

    Returns
    -------
    List of Paths to normalised output images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [nib.load(p) for p in image_paths]
    masks = [nib.load(p) for p in mask_paths] if mask_paths else [None] * len(images)

    output_paths = []

    if method == "nyul":
        normalizer = NyulNormalizer()
        print(f"[nyul] Fitting histogram landmarks across {len(images)} subjects...")
        normalizer.fit_population(images)
        normalized = [normalizer.transform(img) for img in images]

    elif method in INDIVIDUAL_METHODS:
        normalized = [
            normalize_image(img, method=method, mask=mask)
            for img, mask in zip(images, masks)
        ]

    else:
        raise ValueError(f"Unknown method: '{method}'")

    for img_path, norm_img in zip(image_paths, normalized):
        stem = img_path.name.replace(".nii.gz", "").replace(".nii", "")
        # Drop _stripped suffix — the pipeline variant directory already encodes
        # both the skull-strip method and the normalisation method, so the
        # filename only needs to carry the subject/session label.
        stem = stem.removesuffix("_stripped")
        out_path = output_dir / f"{stem}.nii.gz"
        nib.save(norm_img, out_path)
        output_paths.append(out_path)
        print(f"[{method}] Saved: {out_path.name}")

    return output_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Intensity-normalise a set of skull-stripped T1w images"
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Paths to skull-stripped NIfTI images"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write normalised images"
    )
    parser.add_argument(
        "--method", required=True,
        choices=["zscore", "fcm_wm", "kde", "whitestripe", "nyul", "minmax"],
        help="Normalisation method"
    )
    parser.add_argument(
        "--masks", nargs="+", default=None,
        help="Brain mask paths (same order as --inputs); required for fcm_wm and whitestripe"
    )
    cli_args = parser.parse_args()

    out_paths = batch_normalize(
        image_paths=[Path(p) for p in cli_args.inputs],
        method=cli_args.method,
        output_dir=Path(cli_args.output_dir),
        mask_paths=[Path(p) for p in cli_args.masks] if cli_args.masks else None,
    )
    for p in out_paths:
        print(p)


def _minmax_normalize(
    img: nib.Nifti1Image,
    mask: nib.Nifti1Image | None = None,
) -> nib.Nifti1Image:
    """
    Scale intensities to [0, 1] using min and max within the brain mask.
    Serves as a naive baseline for comparison against tissue-informed methods.
    """
    data = img.get_fdata().copy()

    if mask is not None:
        mask_data = mask.get_fdata().astype(bool)
        vmin = data[mask_data].min()
        vmax = data[mask_data].max()
    else:
        vmin = data.min()
        vmax = data.max()

    normalized = (data - vmin) / (vmax - vmin + 1e-8)
    return nib.Nifti1Image(normalized, img.affine, img.header)
