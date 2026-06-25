# TML Assignment 4 — Watermark Forging Pipeline

This repository implements a validation-first pipeline for analyzing and forging unknown image watermarks across eight watermark groups.

The pipeline uses category-specific attacks selected from measured evidence rather than one universal model.

## 1. Repository structure

```text
tml-task4-watermark_forging/
├── dataset/
│   ├── clean_targets/
│   │   ├── 1.png
│   │   └── ...
│   └── watermarked_sources/
│       ├── WM_1/
│       ├── WM_2/
│       ├── WM_3/
│       ├── WM_4/
│       ├── WM_5/
│       ├── WM_6/
│       ├── WM_7/
│       └── WM_8/
│
├── common.py
├── check_dataset.py
├── inspect_outputs.py
├── diagnose_watermarks_validated.py
├── forge_specialized.py
├── forge_baseline.py
├── train_surrogate.py
├── forge_pgd.py
├── train_wm3_surrogate.py   # deprecated thin wrapper -> train_surrogate.py --category WM_3
├── forge_wm3_pgd.py         # deprecated thin wrapper -> forge_pgd.py --category WM_3
├── build_submission.py
│
├── diagnostics_validated/
├── baseline_candidates/
├── specialized_candidates/
├── surrogates/
├── pgd_candidates/
└── final_submission/
```

## 2. Environment setup

Activate the Conda environment:

```bash
conda activate tmltask4
```

Install the required dependencies:

```bash
python -m pip install \
    numpy \
    pillow \
    scipy \
    scikit-learn \
    PyWavelets \
    torch \
    torchvision
```

Optional LPIPS support for WM_3:

```bash
python -m pip install lpips
```

## 3. Dataset layout

The dataset must have this structure:

```text
dataset/
├── clean_targets/
└── watermarked_sources/
    ├── WM_1/
    ├── WM_2/
    ├── WM_3/
    ├── WM_4/
    ├── WM_5/
    ├── WM_6/
    ├── WM_7/
    └── WM_8/
```

Target mappings:

| Watermark group | Target image IDs |
|---|---:|
| WM_1 | 1–25 |
| WM_2 | 26–50 |
| WM_3 | 51–75 |
| WM_4 | 76–100 |
| WM_5 | 101–125 |
| WM_6 | 126–150 |
| WM_7 | 151–175 |
| WM_8 | 176–200 |

## 4. Validate the dataset

Run:

```bash
python check_dataset.py \
    --dataset /home/atml_team052/tml-task4-watermark_forging/dataset
```

The validation step should confirm:

- 200 clean target images;
- 25 source images per watermark group;
- valid PNG files;
- expected image resolutions;
- no corrupt or unreadable images.

## 5. Run watermark diagnostics

Run:

```bash
python diagnose_watermarks_validated.py \
    --dataset /home/atml_team052/tml-task4-watermark_forging/dataset \
    --output-dir /home/atml_team052/tml-task4-watermark_forging/diagnostics_validated
```

Outputs:

```text
diagnostics_validated/
├── validated_diagnostics.json
├── validated_summary.csv
└── oof_specificity_matrix.csv
```

The diagnostics include:

- out-of-fold detector evaluation;
- out-of-fold category-specificity scoring;
- out-of-fold transform-survival testing;
- permutation significance tests;
- PNG and JPEG provenance controls;
- matched-clean spectral normalization;
- corrected bootstrap confidence intervals;
- clean-normalized periodicity metrics.

A feature should only be treated as useful when:

1. its out-of-fold AUC is high;
2. its permutation p-value is low;
3. the intended category scores higher than clean images;
4. the intended category scores higher than other watermark groups;
5. the signal survives a PNG decode-and-save round trip;
6. the signal is not explained by raw image content.

## 6. Current attack routing

| Category | Primary attack | Fallback / ablation baseline |
|---|---|---|
| WM_1 | Cb-channel residual transfer | mean-residual baseline (`forge_baseline.py`) |
| WM_2 | surrogate-classifier + constrained PGD (no validated hand-crafted signal) | mean-residual baseline |
| WM_3 | residual CNN ensemble with constrained PGD | mean-residual baseline |
| WM_4 | coherent Fourier-phase template | mean-residual baseline |
| WM_5 | Cb/Cr **LSB-plane** bit transfer (LSB_auc ≈ 0.995, far stronger than the continuous residual signal) | mean-residual baseline |
| WM_6 | block-DCT coefficient distribution matching | mean-residual baseline |
| WM_7 | surrogate-classifier + constrained PGD (no validated hand-crafted signal) | mean-residual baseline |
| WM_8 | surrogate-classifier + constrained PGD (no validated hand-crafted signal) | mean-residual baseline |

WM_2/7/8 showed no statistically significant hand-crafted feature (residual/channel/LSB/DCT/spectral AUC all ≈0.5, high permutation p-values) — these are routed to the generic surrogate+PGD pipeline instead (same mechanism as WM_3, now parametrized by `--category`).

WM_5 showed near-perfect LSB separability (AUC ≈ 0.995) but no luma signal at all (Y_auc ≈ 0.5) — routed to a direct bit-plane transfer instead of the continuous residual approach used for WM_1.

The generic mean-residual baseline (`forge_baseline.py`) computes the mean high-pass residual across a group's 25 source images and additively transfers it to the matching clean targets. It can be run for **any** category and serves as: (a) the only attack used for ablation comparison purposes, and (b) a sanity-check baseline to verify specialized/PGD attacks actually outperform the naive approach before being included in the final submission.

Routing should be revised only when the validated diagnostics provide stronger evidence.

## 7. Generate specialized non-neural candidates

Run:

```bash
python forge_specialized.py \
    --dataset /home/atml_team052/tml-task4-watermark_forging/dataset \
    --output-dir /home/atml_team052/tml-task4-watermark_forging/specialized_candidates \
    --strength-grid 0.0025,0.005,0.01,0.02
```

This produces one complete 200-image candidate folder per strength:

```text
specialized_candidates/
├── strength_0.0025/
├── strength_0.005/
├── strength_0.01/
└── strength_0.02/
```

The script modifies:

- WM_1 targets with a Cb-channel residual template;
- WM_4 targets with a coherent Fourier-phase template;
- WM_5 targets with a direct Cb/Cr **LSB bit-plane** transfer (no strength scaling — a single bit per pixel is already imperceptible);
- WM_6 targets with selected 8×8 DCT coefficient adjustments.

All other targets (WM_2, WM_3, WM_7, WM_8) remain unchanged in this script's output — they are handled by the surrogate+PGD pipeline below.

## 8. Generate the generic mean-residual baseline

Run for any subset of categories (defaults to all 8):

```bash
python forge_baseline.py \
    --dataset /home/atml_team052/tml-task4-watermark_forging/dataset \
    --output-dir /home/atml_team052/tml-task4-watermark_forging/baseline_candidates \
    --strength-grid 0.01,0.02,0.04,0.08 \
    --categories WM_1,WM_2,WM_3,WM_4,WM_5,WM_6,WM_7,WM_8
```

This computes the mean high-pass residual across each group's 25 source images and additively transfers it to the matching clean targets. Use it as the ablation reference point for every category, and in particular as the working attack for any category where no stronger specialized or surrogate result is available.

## 9. Train a surrogate-classifier ensemble (WM_2, WM_3, WM_7, WM_8)

`train_surrogate.py` is parametrized by `--category` and is used for every group with no validated hand-crafted signal (WM_2, WM_7, WM_8), plus WM_3 which already used this mechanism:

```bash
python train_surrogate.py \
    --dataset /home/atml_team052/tml-task4-watermark_forging/dataset \
    --category WM_2 \
    --output-dir /home/atml_team052/tml-task4-watermark_forging/surrogates \
    --ensemble-size 5 \
    --epochs 40
```

Repeat with `--category WM_3`, `--category WM_7`, `--category WM_8`. Outputs land in per-category subfolders:

```text
surrogates/
├── wm_2/
│   ├── detector_0.pt ... detector_4.pt
│   └── metadata.json
├── wm_3/
├── wm_7/
└── wm_8/
```

Each detector uses a different random seed and train/validation split.

## 10. Generate PGD candidates

`forge_pgd.py` is parametrized the same way and only perturbs the target-image range belonging to `--category` (per `CATEGORY_RANGES` in `common.py`):

```bash
python forge_pgd.py \
    --dataset /home/atml_team052/tml-task4-watermark_forging/dataset \
    --category WM_2 \
    --models /home/atml_team052/tml-task4-watermark_forging/surrogates \
    --output-dir /home/atml_team052/tml-task4-watermark_forging/pgd_candidates \
    --eps-grid 0.0039215686,0.0078431373,0.011764706 \
    --steps 50 \
    --step-size 0.0009803922
```

Repeat for `WM_3`, `WM_7`, `WM_8`. The epsilon values correspond to:

| Folder | Perturbation bound |
|---|---:|
| `eps_0.003922` | 1/255 |
| `eps_0.007843` | 2/255 |
| `eps_0.011765` | 3/255 |

Start with the smallest epsilon that improves held-out surrogate AUC.

`train_wm3_surrogate.py` / `forge_wm3_pgd.py` remain as deprecated thin wrappers around `train_surrogate.py --category WM_3` / `forge_pgd.py --category WM_3` for backward compatibility.

## 11. Inspect candidate quality

```bash
python inspect_outputs.py \
    --clean-dir /home/atml_team052/tml-task4-watermark_forging/dataset/clean_targets \
    --forged-dir /home/atml_team052/tml-task4-watermark_forging/specialized_candidates/strength_0.005
```

```bash
python inspect_outputs.py \
    --clean-dir /home/atml_team052/tml-task4-watermark_forging/dataset/clean_targets \
    --forged-dir /home/atml_team052/tml-task4-watermark_forging/pgd_candidates/wm_2/eps_0.007843
```

Reject candidates that:

- are visibly altered;
- have unexpectedly high maximum pixel error;
- substantially change image structure;
- perform worse than the mean-residual baseline for that category.

## 12. Build the final submission

`build_submission.py` now takes a generic per-category routing file instead of fixed flags. Write a JSON file mapping each category to the candidate directory you want to use for it, e.g. `routing.json`:

```json
{
  "WM_1": "/home/atml_team052/tml-task4-watermark_forging/specialized_candidates/strength_0.005",
  "WM_2": "/home/atml_team052/tml-task4-watermark_forging/pgd_candidates/wm_2/eps_0.007843",
  "WM_3": "/home/atml_team052/tml-task4-watermark_forging/pgd_candidates/wm_3/eps_0.007843",
  "WM_4": "/home/atml_team052/tml-task4-watermark_forging/specialized_candidates/strength_0.005",
  "WM_5": "/home/atml_team052/tml-task4-watermark_forging/specialized_candidates/strength_0.005",
  "WM_6": "/home/atml_team052/tml-task4-watermark_forging/specialized_candidates/strength_0.005",
  "WM_7": "/home/atml_team052/tml-task4-watermark_forging/pgd_candidates/wm_7/eps_0.007843",
  "WM_8": "/home/atml_team052/tml-task4-watermark_forging/pgd_candidates/wm_8/eps_0.007843"
}
```

Any category omitted from the file (or whose file is missing for a given id) falls back to the unmodified clean target.

```bash
python build_submission.py \
    --dataset /home/atml_team052/tml-task4-watermark_forging/dataset \
    --routing /home/atml_team052/tml-task4-watermark_forging/routing.json \
    --output-dir /home/atml_team052/tml-task4-watermark_forging/final_submission \
    --zip /home/atml_team052/tml-task4-watermark_forging/final_submission.zip
```

The submission directory must contain exactly:

```text
1.png
2.png
...
200.png
```

The ZIP must contain the image files at the archive root.

## 13. Ablation strategy

Do not evaluate only one fully combined submission.

Test these variants separately:

1. mean-residual baseline only, for every category;
2. each specialized attack in isolation (WM_1, WM_4, WM_5, WM_6);
3. each surrogate+PGD attack in isolation (WM_2, WM_3, WM_7, WM_8);
4. combined best-performing routing across all 8 categories.

Only combine attacks that produce measurable improvements over the mean-residual baseline for that category.

Recommended initial sweeps:

### WM_1, WM_4 (continuous-strength specialized attacks)

```text
0.0025
0.005
0.01
0.02
```

### WM_5

No strength sweep — the LSB transfer is a binary bit-plane copy.

### WM_2, WM_3, WM_7, WM_8 (surrogate+PGD)

```text
1/255
2/255
3/255
```

### WM_6

Reduce the DCT interpolation strength if block artifacts or visible changes appear.

### All categories (baseline)

```text
0.01
0.02
0.04
0.08
```

## 13. HTCondor execution

Use absolute paths for all inputs and outputs.

Example wrapper:

```bash
#!/bin/bash
set -eo pipefail

PROJECT=/home/atml_team052/tml-task4-watermark_forging
PYTHON=/home/atml_team052/.conda/envs/tmltask4/bin/python

"$PYTHON" "$PROJECT/diagnose_watermarks_validated.py" \
    --dataset "$PROJECT/dataset" \
    --output-dir "$PROJECT/diagnostics_validated"
```

The Condor submit file should execute the wrapper through Bash:

```text
executable = /bin/bash
arguments = /home/atml_team052/tml-task4-watermark_forging/run_pipeline.sh
```

Relative output paths should be avoided because Condor executes jobs in temporary working directories.

## 14. Recommended execution order

```text
1. Validate dataset
2. Run validated diagnostics
3. Review specificity and provenance controls
4. Generate the mean-residual baseline for all categories
5. Generate specialized candidates (WM_1, WM_4, WM_5, WM_6)
6. Train surrogate ensembles for WM_2, WM_3, WM_7, WM_8
7. Generate PGD candidates for WM_2, WM_3, WM_7, WM_8
8. Inspect image quality
9. Run category ablations against the baseline
10. Select the best attack per category
11. Write routing.json and build the final submission ZIP
```

## 15. Limitations

The source images are not paired with clean versions of the same content.

Therefore:

- direct watermark residual recovery is uncertain;
- classifiers may learn image provenance instead of watermark structure;
- category-specific statistics must be validated out of fold;
- surrogate attacks may not transfer to the hidden detector;
- low perceptual distortion must take priority over aggressive optimization.

The pipeline is experimental and should be evaluated through controlled category-level ablations.
