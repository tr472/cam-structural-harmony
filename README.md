# cam-structural-harmony

A modular pipeline for evaluating and harmonising structural MRI data across
scanners, sites, and acquisition protocols.

Developed on the Cambridge Centre for Ageing and Neuroscience (Cam-CAN)
longitudinal dataset, which spans a Siemens Trio → Prisma scanner transition.
Designed to work with any BIDS-formatted structural MRI dataset.

## The problem

When a study spans multiple scanners — different manufacturers, field strengths,
or software versions — measured brain volumes shift in ways that have nothing to
do with the participant's biology. A 5% apparent change in hippocampal volume
across a scanner upgrade can swamp a real ageing effect you spent years measuring.

This pipeline systematically isolates and quantifies technical sources of variance
before applying harmonisation, so you know what you are correcting and how well
it worked.

## What it does
```
BIDS dataset
  └── skull stripping              (SynthStrip, ROBEX)
       └── intensity normalisation  (6 methods × 2 skull strips = 12 variants)
            └── FreeSurfer recon-all → aparc/aseg extraction
                 └── NeuroCombat harmonisation
                      └── variance decomposition
                           └── QC report (AI-assisted)
```

For each combination of skull-stripping method and normalisation technique, the
pipeline runs the full FreeSurfer → harmonisation → evaluation sequence. This
produces a structured comparison of how preprocessing choices interact with
scanner-related variance.

Variance sources decomposed: inter-scanner, intra-scanner, site, manufacturer,
scanner model, and study/protocol effects.

See [cam-dwi-harmony](#) for the companion diffusion MRI harmonisation pipeline.

## Installation
```bash
git clone https://github.com/yourusername/cam-structural-harmony
cd cam-structural-harmony
pip install -e .
```

Requires FreeSurfer 7.x (set `FREESURFER_HOME`), Python ≥ 3.9.

## Usage
```bash
# Run full pipeline on a BIDS dataset
python -m cam_harmony.run \
  --bids_dir /path/to/bids \
  --output_dir /path/to/outputs \
  --config config.yaml

# Generate AI-assisted QC report from completed outputs
python -m cam_harmony.qc_assistant \
  --results_dir /path/to/outputs/variance_results \
  --output outputs/qc_report.md \
  --focus_rois hippocampus lateral_ventricle entorhinal
```

## Configuration
```yaml
skull_strip:
  methods: [synthstrip, robex]

intensity_norm:
  methods: [zscore, fcm_wm, kde, whitestripe, nyul, minmax]

freesurfer:
  parallel_jobs: 4
  flags: "-all"

combat:
  batch_col: scanner
  biological_covariates: [age, sex, tiv]

variance:
  sources: [scanner, site, manufacturer, model, protocol]
```

## Outputs
```
outputs/
  pipeline_variants/      # per-method FreeSurfer and harmonisation outputs
  variance_results/       # ICC, CV, and variance decomposition tables
  figures/                # pre/post harmonisation comparisons
  qc_report.md            # AI-generated narrative summary
```

## Demo notebook vs real pipeline

`notebooks/demo.ipynb` runs the full pipeline API end-to-end without requiring
FreeSurfer, real MRI data, or a multi-scanner dataset. It is designed so
reviewers can execute it on any machine and see representative outputs. Several
steps are necessarily approximated:

| Step | Demo notebook | Real pipeline |
|------|---------------|---------------|
| Input data | Synthetic 3D numpy arrays (Gaussian noise shaped like brain volumes) | BIDS-formatted T1w NIfTI images |
| Skull stripping | Skipped — synthetic arrays have no skull | `mri_synthstrip` or `pyrobex` applied to each scan |
| Intensity normalisation | Applied to synthetic volumes; differences between methods are visible but not anatomically grounded | Applied to skull-stripped T1w images with real tissue intensity distributions |
| FreeSurfer recon-all | Skipped; ROI volumes are synthetic random data with explicit additive scanner offsets injected to simulate bias | Full `recon-all` (6–10 hours per subject on real data) |
| NeuroCombat | Runs on the synthetic ROI DataFrame — harmonisation demonstrably removes the injected offsets | Runs on real FreeSurfer `aparc+aseg` stats across 2+ scanner groups |
| Scanner / batch labels | Hard-coded as `"TRIO"` and `"PRISMA"` in the notebook | Read from `participants.tsv` or derived from BIDS session labels (e.g. `ses-PRISMA1 → PRISMA`) |
| ICC and CV values | Illustrative — computed from synthetic data with controlled properties | Validated on the Cam-CAN travelling-heads cohort; ICC target > 0.75 is a meaningful threshold |
| QC report | Generated from synthetic metrics; shows the report format and narrative structure | AI analysis of real variance decomposition results using `ANTHROPIC_API_KEY` |

**What the demo does faithfully represent:**
the module interfaces, the pipeline's data flow, the pre/post harmonisation
ICC comparison structure, and the output file layout. The synthetic scanner
effect (a fixed offset added to TRIO volumes) is deliberately sized to produce
a realistic harmonisation scenario — ICC improves from ~0.4 to ~0.85 after
ComBat — so the before/after figures are visually representative of what the
real pipeline produces.

**To run the real pipeline** you need: FreeSurfer 7.x (`FREESURFER_HOME` set),
a BIDS dataset with ≥ 2 scanner groups and ≥ 2 subjects per group, and either a
`participants.tsv` containing a `scanner` (or `site`) column, or BIDS session
labels that encode the scanner name (e.g. `ses-PRISMA1`, `ses-TRIO2`).

## Background

Developed as part of a PhD investigating scanner harmonisation effects in
longitudinal brain ageing studies. Methods draw on the ENIGMA harmonisation
protocols, neuroCombat-sklearn, and the intensity-normalization package
(Reinhold et al., 2019).

Reinhold, J.C., et al. "Evaluating the impact of intensity normalization on MR
image synthesis." Medical Imaging 2019: Image Processing. SPIE, 2019.
