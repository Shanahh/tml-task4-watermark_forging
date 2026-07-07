# TML Assignment 4 — Watermark Forgery Attack

This README explains **only how to recreate our best leaderboard result
(0.719245)**, as required by the task. It forges the watermark of each of the 8
source groups (WM_1–WM_8) onto its assigned batch of 25 clean target images.

The best submission is produced in two steps:

1. **Statistical estimation** (`forge_simple.py`) forges every group by adding an
   estimated watermark to the clean targets. This is the base for the 5 groups
   whose scheme we could not identify (WM_1/3/4/5/6).
2. **Known-scheme forging** (`forge_known_scheme.py`) overwrites the 3 groups we
   identified as instances of published, open-source watermarking schemes
   (WM_2 = RivaGAN, WM_7 = TrustMark-Q, WM_8 = TrustMark-P) by re-embedding the
   recovered message with the scheme's own encoder, and passes the other 5
   groups through from step 1 unchanged.

## 1. Requirements

Python 3.11 with:

```bash
pip install numpy pillow scipy PyWavelets torch torchvision lpips \
            bm3d invisible-watermark onnxruntime trustmark
```

- `bm3d` is used by the BM3D "regeneration" extraction for WM_5/WM_6.
- `invisible-watermark`+`onnxruntime` provide the RivaGAN encoder (WM_2).
- `trustmark` provides the TrustMark encoders (WM_7/WM_8); it downloads its model
  checkpoints from the internet on first use.
- If OpenCV fails to import on a headless server (`libgthread-2.0.so.0` error),
  install the headless build: `pip uninstall -y opencv-python && pip install opencv-python-headless`.

## 2. Dataset layout

Place the provided data under `dataset/`:

```text
dataset/
├── clean_targets/            # 1.png .. 200.png
└── watermarked_sources/
    ├── WM_1/ .. WM_8/        # 25 source images per group
```

Target mapping (fixed by the task): WM_1→1–25, WM_2→26–50, WM_3→51–75,
WM_4→76–100, WM_5→101–125, WM_6→126–150, WM_7→151–175, WM_8→176–200.

## 3. Recreate the best submission

Let `$PROJECT` be the repo root and `$DATASET=$PROJECT/dataset`.

### Step 1 — statistical-estimation base (all 8 groups)

```bash
python forge_simple.py \
  --dataset "$DATASET" \
  --output-dir "$PROJECT/simple_candidates" \
  --wm1-strength 0.5 \
  --wm2-strength 0.4 \
  --wm3-strength 0.4 \
  --wm4-strength 0.2 \
  --wm5-extraction regen --wm5-regen-sigma 0.01 --wm5-lpips-cap 0.043 \
  --wm6-extraction regen --wm6-regen-sigma 0.0075 --wm6-lpips-cap 0.046 \
  --wm7-strength 0.6 \
  --wm8-strength 0.2
```

Each group gets `delta = mean(sources) − mean(clean pool)` added to its clean
targets at the given per-category strength; WM_5/WM_6 instead use a BM3D
regeneration copy of the watermark, with the per-image perturbation sized by an
LPIPS cap. This writes 200 images to `simple_candidates/`.

### Step 2 — overwrite the identified groups with their real encoders

```bash
python forge_known_scheme.py \
  --dataset "$DATASET" \
  --base-dir "$PROJECT/simple_candidates" \
  --output-dir "$PROJECT/known_scheme_candidates" \
  --wm2-scheme rivaGan \
  --wm2-message 00010000101111110011101011101000 \
  --wm7-scheme trustmark \
  --wm7-message 1000100100110010100110011110110010010010111110000111011111111101011001111101011011010001010011100001 \
  --wm8-scheme trustmark --wm8-trustmark-model-type P \
  --wm8-message 1010001101101000000100011110001101011111111011001010000110001111111101011011011011101101100001111000
```

This re-embeds each recovered message with the scheme's own encoder on the clean
targets for WM_2/7/8, copies WM_1/3/4/5/6 through from `simple_candidates/`, and
writes the final 200 images to `known_scheme_candidates/`. It refuses to write
anything if a required package is missing and only replaces the output directory
after all 8 groups finish, so a partial run cannot silently corrupt the result.

### Step 3 — zip and submit

```bash
cd "$PROJECT/known_scheme_candidates"
zip -q "$PROJECT/submission.zip" *.png
cd "$PROJECT"
python -c "import zipfile; n=sorted(zipfile.ZipFile('submission.zip').namelist()); \
  assert n==sorted(f'{i}.png' for i in range(1,201)) and not any('/' in x for x in n); \
  print('OK: 200 flat files')"
```

Set your API key and the zip path in `submission.py` and run it to submit.

## 4. How the messages/schemes were found (`identify_scheme.py`)

The three scheme matches above were found by decoding all 25 sources of each
group with the real decoders of candidate open-source schemes and flagging a
match only when the 25 decodes agreed on a balanced, non-degenerate message
clearly above a clean-image control:

```bash
python identify_scheme.py --dataset "$DATASET" --trustmark-model-types C,Q,B,P
```

## 5. Repository layout

Active pipeline:

| File | Role |
|---|---|
| `forge_simple.py` | statistical-estimation forger (step 1) |
| `forge_known_scheme.py` | known-scheme forger / final submission assembly (step 2) |
| `identify_scheme.py` | scheme-identification harness |
| `sweep_regen_sigma.py` | picks the BM3D `--regen-sigma` for WM_5/WM_6 |
| `diagnose_watermarks_validated.py` | out-of-fold feature diagnostics (Figure 1 in the report) |
| `verify_submission.py` | checks a submission zip against the clean targets |
| `common.py` | shared dataset I/O, colour-space and denoising helpers |
| `submission.py` | provided leaderboard upload script |
| `report/` | the written report and its figure-generation script |
| `archive/` | superseded approaches, kept only for reference (see `archive/README.md`) |
