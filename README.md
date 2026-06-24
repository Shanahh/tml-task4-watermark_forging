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
├── train_wm3_surrogate.py
├── forge_wm3_pgd.py
├── build_submission.py
│
├── diagnostics_validated/
├── specialized_candidates/
├── wm3_surrogates/
├── wm3_candidates/
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

| Category | Primary attack |
|---|---|
| WM_1 | Cb-channel residual transfer |
| WM_2 | unchanged baseline |
| WM_3 | residual CNN ensemble with constrained PGD |
| WM_4 | coherent Fourier-phase template |
| WM_5 | Cb/Cr residual transfer |
| WM_6 | block-DCT coefficient distribution matching |
| WM_7 | unchanged baseline |
| WM_8 | unchanged baseline |

This routing should be revised only when the validated diagnostics provide stronger evidence.

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
- WM_5 targets with Cb and Cr residual templates;
- WM_6 targets with selected 8×8 DCT coefficient adjustments.

All other targets remain unchanged.

## 8. Train the WM_3 surrogate ensemble

Run:

```bash
python train_wm3_surrogate.py \
    --dataset /home/atml_team052/tml-task4-watermark_forging/dataset \
    --output-dir /home/atml_team052/tml-task4-watermark_forging/wm3_surrogates \
    --ensemble-size 5 \
    --epochs 40
```

Outputs:

```text
wm3_surrogates/
├── wm3_detector_0.pt
├── wm3_detector_1.pt
├── wm3_detector_2.pt
├── wm3_detector_3.pt
├── wm3_detector_4.pt
└── metadata.json
```

Each detector uses a different random seed and train/validation split.

## 9. Generate WM_3 PGD candidates

Run:

```bash
python forge_wm3_pgd.py \
    --dataset /home/atml_team052/tml-task4-watermark_forging/dataset \
    --models /home/atml_team052/tml-task4-watermark_forging/wm3_surrogates \
    --output-dir /home/atml_team052/tml-task4-watermark_forging/wm3_candidates \
    --eps-grid 0.0039215686,0.0078431373,0.011764706 \
    --steps 50 \
    --step-size 0.0009803922
```

The epsilon values correspond to:

| Folder | Perturbation bound |
|---|---:|
| `eps_0.003922` | 1/255 |
| `eps_0.007843` | 2/255 |
| `eps_0.011765` | 3/255 |

Start with the smallest epsilon that improves held-out surrogate scores.

Optional LPIPS-constrained optimization:

```bash
python forge_wm3_pgd.py \
    --dataset /home/atml_team052/tml-task4-watermark_forging/dataset \
    --models /home/atml_team052/tml-task4-watermark_forging/wm3_surrogates \
    --output-dir /home/atml_team052/tml-task4-watermark_forging/wm3_candidates_lpips \
    --eps-grid 0.0039215686,0.0078431373 \
    --steps 50 \
    --step-size 0.0009803922 \
    --lpips-weight 1.0
```

## 10. Inspect candidate quality

Inspect specialized candidates:

```bash
python inspect_outputs.py \
    --clean-dir /home/atml_team052/tml-task4-watermark_forging/dataset/clean_targets \
    --forged-dir /home/atml_team052/tml-task4-watermark_forging/specialized_candidates/strength_0.005
```

Inspect WM_3 candidates:

```bash
python inspect_outputs.py \
    --clean-dir /home/atml_team052/tml-task4-watermark_forging/dataset/clean_targets \
    --forged-dir /home/atml_team052/tml-task4-watermark_forging/wm3_candidates/eps_0.007843
```

Reject candidates that:

- are visibly altered;
- have unexpectedly high maximum pixel error;
- substantially change image structure;
- perform worse than the baseline.

## 11. Build the final submission

Run:

```bash
python build_submission.py \
    --dataset /home/atml_team052/tml-task4-watermark_forging/dataset \
    --wm1-wm4-wm5-wm6 /home/atml_team052/tml-task4-watermark_forging/specialized_candidates/strength_0.005 \
    --wm3 /home/atml_team052/tml-task4-watermark_forging/wm3_candidates/eps_0.007843 \
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

## 12. Ablation strategy

Do not evaluate only one fully combined submission.

Test these variants separately:

1. unchanged or naive baseline;
2. WM_1 only;
3. WM_3 only;
4. WM_4 only;
5. WM_5 only;
6. WM_6 only;
7. combined best-performing categories.

Only combine attacks that produce measurable improvements.

Recommended initial sweeps:

### WM_1, WM_4, WM_5

```text
0.0025
0.005
0.01
0.02
```

### WM_3

```text
1/255
2/255
3/255
```

### WM_6

Reduce the DCT interpolation strength if block artifacts or visible changes appear.

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
4. Generate specialized candidates
5. Train the WM_3 surrogate ensemble
6. Generate WM_3 candidates
7. Inspect image quality
8. Run category ablations
9. Select the best strengths
10. Build the final submission ZIP
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
