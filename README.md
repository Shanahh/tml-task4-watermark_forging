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
├── check_surrogate_transfer.py
├── train_wm3_surrogate.py   # deprecated thin wrapper -> train_surrogate.py --category WM_3
├── forge_wm3_pgd.py         # deprecated thin wrapper -> forge_pgd.py --category WM_3
├── select_routing.py
├── build_submission.py
│
├── diagnostics_validated/
├── baseline_candidates/
├── specialized_candidates/
├── surrogates/
├── pgd_candidates/
├── transfer_check_wm_*.json
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
| WM_5 | Cb/Cr residual transfer **combined with** Cb/Cr LSB-plane bit transfer | mean-residual baseline |
| WM_6 | block-DCT coefficient distribution matching | mean-residual baseline |
| WM_7 | surrogate-classifier + constrained PGD (no validated hand-crafted signal) | mean-residual baseline |
| WM_8 | surrogate-classifier + constrained PGD (no validated hand-crafted signal) | mean-residual baseline |

WM_2/7/8 showed no statistically significant hand-crafted feature (residual/channel/LSB/DCT/spectral AUC all ≈0.5, high permutation p-values) — these are routed to the generic surrogate+PGD pipeline instead (same mechanism as WM_3, now parametrized by `--category`).

WM_5 showed near-perfect LSB separability (AUC ≈ 0.995) but no luma signal at all (Y_auc ≈ 0.5), and *also* strong Cb/Cr residual separability (AUC ≈ 0.96/0.91). Both domains are statistically independent evidence, so the attack applies both: the continuous residual transfer (as for WM_1) plus the LSB bit-plane copy. The LSB edit costs effectively zero extra perceptual budget, so there's no reason to pick one domain over the other — earlier versions of this attack used LSB only, which risked silently discarding a working continuous-domain signal if the real detector doesn't read literal LSB bits.

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
- WM_5 targets with a Cb/Cr residual template (scaled by strength) *and* a direct Cb/Cr **LSB bit-plane** copy (no strength scaling — a single bit per pixel is already imperceptible);
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

## 9. Train surrogate-classifier ensembles (WM_2, WM_3, WM_7, WM_8)

`train_surrogate.py` is parametrized by `--category` and `--arch` (`cnn_a`, `cnn_b`, `cnn_c` — three structurally different architectures) and is used for every group with no validated hand-crafted signal (WM_2, WM_7, WM_8), plus WM_3 which already used this mechanism.

`cnn_a` and `cnn_b` together form the attack ensemble `forge_pgd.py` actually optimizes against (attacking multiple diverse architectures at once transfers to an unseen model far better than attacking one — the standard ensemble-transferability trick). `cnn_c` is then a third, still-independent architecture used only by the transfer check in the next step, never part of the attack itself.

Train all three for each surrogate-driven category:

```bash
for ARCH in cnn_a cnn_b cnn_c; do
  python train_surrogate.py \
      --dataset /home/atml_team052/tml-task4-watermark_forging/dataset \
      --category WM_2 \
      --arch "$ARCH" \
      --output-dir /home/atml_team052/tml-task4-watermark_forging/surrogates \
      --ensemble-size 5 \
      --epochs 40
done
```

Repeat for `--category WM_3`, `WM_7`, `WM_8`. Outputs land in per-category, per-architecture subfolders:

```text
surrogates/
├── wm_2/
│   ├── cnn_a/{detector_0.pt ... detector_4.pt, metadata.json}
│   ├── cnn_b/{detector_0.pt ... detector_4.pt, metadata.json}
│   └── cnn_c/{detector_0.pt ... detector_4.pt, metadata.json}
├── wm_3/
├── wm_7/
└── wm_8/
```

Each detector uses a different random seed and train/validation split.

## 10. Check whether the attack is likely to transfer, before spending a submission on it

`check_surrogate_transfer.py` crafts PGD perturbations against the `cnn_a`+`cnn_b` ensemble (`--attack-archs`, comma-separated) and measures whether the independent `cnn_c` ensemble (`--holdout-arch`, never used in the attack) also raises its "watermarked" probability on the same images:

```bash
python check_surrogate_transfer.py \
    --dataset /home/atml_team052/tml-task4-watermark_forging/dataset \
    --category WM_2 \
    --attack-models /home/atml_team052/tml-task4-watermark_forging/surrogates --attack-archs cnn_a,cnn_b \
    --holdout-models /home/atml_team052/tml-task4-watermark_forging/surrogates --holdout-arch cnn_c \
    --eps 0.0078431373 \
    --output /home/atml_team052/tml-task4-watermark_forging/transfer_check_wm2.json
```

The script reports, and acts on, more than a single AUC-style number:

- **`holdout_genuine_flip_rate`** only counts images where the holdout model's probability actually *crossed* 0.5 because of the attack (`clean < 0.5` → `forged >= 0.5`). A naive "landed above 0.5" count is misleading whenever the holdout model is already miscalibrated and predicts >0.5 on many *clean* images before any attack at all — this was observed in practice for one category, where 88% of "flips" turned out to already be above 0.5 pre-attack.
- **`holdout_baseline_std`** flags a holdout model that collapsed to a near-constant output regardless of input (e.g. always predicting ≈0.35) — such a holdout can't judge transfer either way, and is reported as `"inconclusive"` rather than a false `"doesn't transfer"`.
- **`transfer_ratio`** (`holdout_lift / attack_lift`) and the verdict string distinguish three outcomes: `"likely to transfer"`, `"weak or partial transfer"` (some real signal carries over but not enough to cross the holdout's decision boundary at this eps/step budget — worth a bigger budget or a better surrogate, not a flat no), and `"unlikely to transfer"` (no real signal, matches a holdout near chance or collapsed).

Treat anything other than `"likely to transfer"` as a signal to improve the surrogate (more augmentation, larger negative set, bigger eps/step budget) before submitting, not as a reason to submit anyway. `select_routing.py` (step 13) automates exactly this decision.

## 11. Generate PGD candidates

`forge_pgd.py` is parametrized the same way (`--category`, `--archs`, defaulting to `cnn_a`) and only perturbs the target-image range belonging to `--category` (per `CATEGORY_RANGES` in `common.py`). `--archs` accepts a comma-separated list to attack several architectures simultaneously, matching whatever was used in the transfer check. The quality term in the PGD loss optimizes real LPIPS directly — matching `Sqlt = exp(-8*LPIPS)` exactly — falling back to an MSE+TV proxy with a printed warning if the `lpips` package isn't installed:

```bash
python forge_pgd.py \
    --dataset /home/atml_team052/tml-task4-watermark_forging/dataset \
    --category WM_2 \
    --models /home/atml_team052/tml-task4-watermark_forging/surrogates \
    --archs cnn_a,cnn_b \
    --output-dir /home/atml_team052/tml-task4-watermark_forging/pgd_candidates \
    --eps-grid 0.0039215686,0.0078431373,0.011764706 \
    --steps 50 \
    --step-size 0.0009803922 \
    --lpips-weight 10.0
```

Repeat for `WM_3`, `WM_7`, `WM_8`. The epsilon values correspond to:

| Folder | Perturbation bound |
|---|---:|
| `eps_0.003922` | 1/255 |
| `eps_0.007843` | 2/255 |
| `eps_0.011765` | 3/255 |

Start with the smallest epsilon that improves held-out surrogate AUC.

`train_wm3_surrogate.py` / `forge_wm3_pgd.py` remain as deprecated thin wrappers around `train_surrogate.py --category WM_3` / `forge_pgd.py --category WM_3` for backward compatibility (they only train/attack the single default `cnn_a` architecture).

## 12. Inspect candidate quality

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

## 13. Build the final submission

`build_submission.py` takes a generic per-category routing file instead of fixed flags: a JSON mapping each category to the candidate directory to use for it. Any category omitted from the file (or whose file is missing for a given id) falls back to the unmodified clean target.

Rather than writing `routing.json` by hand, `select_routing.py` builds it automatically: WM_1/4/5/6 always route to the specialized candidates (they don't depend on a black-box surrogate at all), and each surrogate-driven category (WM_2/3/7/8) routes to its PGD candidate **only if** its `transfer_check_<category>.json` verdict (step 10) says `"likely to transfer"` — otherwise it falls back to the mean-residual baseline automatically, with no manual editing required:

```bash
python select_routing.py \
    --specialized-dir /home/atml_team052/tml-task4-watermark_forging/specialized_candidates --specialized-strength 0.005 \
    --baseline-dir /home/atml_team052/tml-task4-watermark_forging/baseline_candidates --baseline-strength 0.02 \
    --pgd-dir /home/atml_team052/tml-task4-watermark_forging/pgd_candidates --pgd-eps 0.007843 \
    --transfer-checks-dir /home/atml_team052/tml-task4-watermark_forging \
    --output /home/atml_team052/tml-task4-watermark_forging/routing.json
```

It prints its reasoning per category (`transfer check passed`, `transfer check failed (...)`, or `no transfer check found`) so you can see exactly why each category landed where it did. Inspect the printed output and the resulting `routing.json` before treating it as final — a `"weak or partial transfer"` verdict currently also falls back to the baseline, even though step 10 suggests that case deserves more investment (bigger eps/step budget, a better surrogate) rather than an automatic fallback; edit `routing.json` by hand afterwards if you've separately validated a stronger candidate for such a category.

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

## 14. Ablation strategy

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

The residual half uses the same strength grid as WM_1/WM_4. The LSB half has no strength knob — it is a binary bit-plane copy applied at full strength regardless. Worth ablating the two halves independently (residual only, LSB only, both) in case one of them doesn't correspond to what the real detector reads.

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

## 15. HTCondor execution

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

## 16. Recommended execution order

```text
1. Validate dataset
2. Run validated diagnostics
3. Review specificity and provenance controls
4. Generate the mean-residual baseline for all categories
5. Generate specialized candidates (WM_1, WM_4, WM_5, WM_6)
6. Train surrogate ensembles (cnn_a, cnn_b, cnn_c) for WM_2, WM_3, WM_7, WM_8
7. Run the cross-surrogate transfer sanity check (cnn_a+cnn_b attack vs cnn_c holdout)
8. Generate PGD candidates (cnn_a+cnn_b ensemble attack) for WM_2, WM_3, WM_7, WM_8
9. Inspect image quality
10. Run category ablations against the baseline
11. Run select_routing.py to auto-route each category based on the transfer-check verdicts
12. Manually review and adjust routing.json for any "weak or partial transfer" categories
13. Build the final submission ZIP
```

## 17. Limitations

The source images are not paired with clean versions of the same content.

Therefore:

- direct watermark residual recovery is uncertain;
- classifiers may learn image provenance instead of watermark structure (use `check_surrogate_transfer.py` to catch this before submitting: if an independent holdout architecture doesn't agree with the attack ensemble, the perturbation is probably overfit to provenance, not the watermark);
- category-specific statistics must be validated out of fold;
- surrogate attacks may not transfer to the hidden detector, even when the transfer check above looks favorable — agreement between two of your own surrogates is necessary but not sufficient evidence of transfer to the real detector;
- low perceptual distortion must take priority over aggressive optimization;
- a hand-crafted attack with a higher diagnostic AUC is not necessarily what the real detector reads — when more than one statistically independent domain shows significant evidence for the same category (as with WM_5's LSB and residual signals), prefer combining them over picking only the highest-AUC one.

The pipeline is experimental and should be evaluated through controlled category-level ablations.
