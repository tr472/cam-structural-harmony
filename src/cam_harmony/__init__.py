"""
cam-structural-harmony
======================
Modular pipeline for evaluating and harmonising structural MRI data across
scanners, sites, and acquisition protocols.
"""

from cam_harmony.intensity_norm import normalize_image, batch_normalize
from cam_harmony.harmonise import run_combat
from cam_harmony.variance import run_variance_analysis
from cam_harmony.qc_assistant import generate_qc_report

__version__ = "0.1.0"
__all__ = [
    "normalize_image",
    "batch_normalize",
    "run_combat",
    "run_variance_analysis",
    "generate_qc_report",
]
