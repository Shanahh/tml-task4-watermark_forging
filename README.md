# Native-resolution Stable Diffusion LoRA attack

This code trains one LoRA adapter per watermark category at the category's
native resolution:

- WM_1–WM_4 and WM_6: 256×256
- WM_5: 128×128
- WM_7–WM_8: 512×512

The VAE, text encoder, and base U-Net remain frozen. Only LoRA weights in the
U-Net attention layers are trained.

## Install

Install a CUDA-enabled PyTorch build first, then:

```bash
pip install -r requirements.txt
```

You may need to authenticate with Hugging Face and accept the base model's
license:

```bash
hf auth login
```

## Dataset layout

```text
.
├── clean_targets/
└── watermarked_sources/
    ├── WM_1/
    ├── ...
    └── WM_8/
```

## Smoke test

```bash
python train.py \
  --dataset . \
  --category WM_1 \
  --steps 100 \
  --batch-size 2 \
  --grad-accum 2
```

## Train all eight adapters

```bash
python train.py \
  --dataset . \
  --category all \
  --steps 3000 \
  --rank 4 \
  --batch-size 2 \
  --grad-accum 4 \
  --learning-rate 1e-4 \
  --max-timestep 150 \
  --precision fp16
```

## Forge submission

```bash
python forge.py \
  --dataset . \
  --run-dir wm_lora_runs \
  --adapter-scale 0.75 \
  --strength 0.10 \
  --steps 20 \
  --precision fp16
```

This creates `submission.zip` with exactly `1.png` through `200.png`.

## Inspect distortion

```bash
python inspect_outputs.py
```

## Inference sweep without retraining

```bash
python forge.py --strength 0.05 --adapter-scale 0.50 --submission sub_005_050.zip
python forge.py --strength 0.10 --adapter-scale 0.75 --submission sub_010_075.zip
python forge.py --strength 0.15 --adapter-scale 1.00 --submission sub_015_100.zip
```

This is a WMCopier-inspired experimental approximation, not a faithful
reimplementation of the WMCopier paper. With only 25 examples per category,
the LoRA may learn source-image distribution features rather than only the
watermark. Treat the leaderboard as an empirical validation step.
