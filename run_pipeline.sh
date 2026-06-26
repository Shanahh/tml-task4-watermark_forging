#!/bin/bash
set -eo pipefail

# Usage:
#   ./run_pipeline.sh [--wm1-strength 0.03] [--wm3-strength 0.015] \
#                      [--wm4-strength 0.01] [--wm5-strength 0.07] [--wm6-strength 0.04]
#
# Strength is per-category, not a single shared value: run_lpips_sweep.sh
# showed these categories have very different LPIPS sensitivity per unit
# strength (e.g. WM_4's luma attack drops sharply by 0.02, while WM_6's DCT
# attack saturates at >=0.04 at almost no perceptual cost). A single shared
# strength necessarily over- or under-drives at least one category. The
# defaults below are forge_specialized.py's own defaults (the knee points
# found by the sweep) -- run ./run_lpips_sweep.sh first and override these if
# the underlying templates change.

PROJECT=/home/atml_team052/tml-task4-watermark_forging
ENV=/home/atml_team052/.conda/envs/tmltask4
PYTHON="$ENV/bin/python"

export PYTHONNOUSERSITE=1
export HF_HOME=/home/atml_team052/.cache/huggingface
export PIP_CACHE_DIR=/home/atml_team052/.cache/pip

WM1_STRENGTH="0.03"
WM3_STRENGTH="0.015"
WM4_STRENGTH="0.01"
WM5_STRENGTH="0.07"
WM6_STRENGTH="0.04"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wm1-strength) WM1_STRENGTH="$2"; shift 2 ;;
        --wm3-strength) WM3_STRENGTH="$2"; shift 2 ;;
        --wm4-strength) WM4_STRENGTH="$2"; shift 2 ;;
        --wm5-strength) WM5_STRENGTH="$2"; shift 2 ;;
        --wm6-strength) WM6_STRENGTH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Environment setup (idempotent: only built if missing)
# ---------------------------------------------------------------------------
if [ ! -x "$PYTHON" ]; then
    source /opt/conda/etc/profile.d/conda.sh

    export CONDA_ENVS_PATH=/home/atml_team052/.conda/envs
    export CONDA_PKGS_DIRS=/home/atml_team052/.conda/pkgs

    conda create -y -p "$ENV" python=3.11 pip

    conda install -y -p "$ENV" \
        pytorch==2.4.1 \
        torchvision==0.19.1 \
        torchaudio==2.4.1 \
        pytorch-cuda=12.1 \
        -c pytorch \
        -c nvidia
fi

"$PYTHON" -m pip install --upgrade pip setuptools wheel

"$PYTHON" -m pip install \
    diffusers==0.35.2 \
    transformers==4.56.2 \
    accelerate==1.10.1 \
    peft==0.17.1 \
    safetensors \
    pillow \
    "numpy<2" \
    tqdm \
    torchvision \
    PyWavelets \
    scikit-learn \
    scipy \
    lpips

"$PYTHON" - <<'PY'
import sys
import torch
import torchvision
import transformers
import diffusers
import peft

print("Python:", sys.executable)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("diffusers:", diffusers.__version__)
print("peft:", peft.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

DATASET="$PROJECT/dataset"

# ---------------------------------------------------------------------------
# 2. Validated diagnostics (re-run only if the report is missing/stale)
# ---------------------------------------------------------------------------
echo "=== [1/6] Diagnostics ==="
"$PYTHON" "$PROJECT/diagnose_watermarks_validated.py" \
    --dataset "$DATASET" \
    --output-dir "$PROJECT/diagnostics_validated"

# ---------------------------------------------------------------------------
# 3. Generic mean-residual baseline — every category, used as the ablation
#    reference and as the fallback attack for categories with no stronger
#    candidate.
# ---------------------------------------------------------------------------
echo "=== [2/6] Baseline (all categories) ==="
"$PYTHON" "$PROJECT/forge_baseline.py" \
    --dataset "$DATASET" \
    --output-dir "$PROJECT/baseline_candidates" \
    --strength-grid 0.01,0.02,0.04,0.08 \
    --categories WM_1,WM_2,WM_3,WM_4,WM_5,WM_6,WM_7,WM_8

# ---------------------------------------------------------------------------
# 4. Specialized hand-crafted attacks (validated signal exists):
#    WM_1 Cb residual, WM_3 combined Y/Cb/Cr residual, WM_4 Fourier-phase,
#    WM_5 residual+LSB bit-plane, WM_6 DCT. WM_3 also has a surrogate+PGD
#    path below (kept for ablation comparison) but routes here by default
#    in step 8, since this attack doesn't depend on a black-box proxy
#    model's transferability to the real detector.
# ---------------------------------------------------------------------------
echo "=== [3/6] Specialized candidates (WM_1, WM_3, WM_4, WM_5, WM_6) ==="
echo "strengths: WM_1=$WM1_STRENGTH WM_3=$WM3_STRENGTH WM_4=$WM4_STRENGTH WM_5=$WM5_STRENGTH WM_6=$WM6_STRENGTH"
"$PYTHON" "$PROJECT/forge_specialized.py" \
    --dataset "$DATASET" \
    --output-dir "$PROJECT/specialized_candidates" \
    --wm1-strength "$WM1_STRENGTH" \
    --wm3-strength "$WM3_STRENGTH" \
    --wm4-strength "$WM4_STRENGTH" \
    --wm5-strength "$WM5_STRENGTH" \
    --wm6-strength "$WM6_STRENGTH"

# ---------------------------------------------------------------------------
# 5. Surrogate-classifier ensembles for categories with no validated
#    hand-crafted signal: WM_2, WM_7, WM_8. (WM_3 is also trained here for
#    ablation comparison against its hand-crafted attack above, but routing
#    in step 8 prefers the hand-crafted attack for WM_3 by default.)
#
#    Three structurally different architectures are trained per category.
#    cnn_a + cnn_b together are the attack ensemble forge_pgd.py optimizes
#    against (ensemble attacks transfer to unseen models better than a single
#    architecture). cnn_c is then a still-independent holdout used only for
#    the transfer sanity check in step 6, never part of the attack itself.
# ---------------------------------------------------------------------------
echo "=== [4/7] Surrogate ensembles (WM_2, WM_3, WM_7, WM_8) ==="
for CAT in WM_2 WM_3 WM_7 WM_8; do
    for ARCH in cnn_a cnn_b cnn_c; do
        "$PYTHON" "$PROJECT/train_surrogate.py" \
            --dataset "$DATASET" \
            --category "$CAT" \
            --arch "$ARCH" \
            --output-dir "$PROJECT/surrogates" \
            --ensemble-size 5 \
            --epochs 40
    done
done

# ---------------------------------------------------------------------------
# 6. Cross-surrogate transfer sanity check, before spending a real submission
#    on these categories. Attacks cnn_a+cnn_b together and checks whether the
#    independent cnn_c holdout agrees -- see check_surrogate_transfer.py
#    docstring for what "agrees" means and the pitfalls it guards against
#    (miscalibrated holdout baselines, collapsed/degenerate holdouts).
#    Informational only: does not stop the pipeline. select_routing.py below
#    reads these verdicts and decides the actual routing.
# ---------------------------------------------------------------------------
echo "=== [5/7] Surrogate transfer sanity check (WM_2, WM_3, WM_7, WM_8) ==="
for CAT in WM_2 WM_3 WM_7 WM_8; do
    "$PYTHON" "$PROJECT/check_surrogate_transfer.py" \
        --dataset "$DATASET" \
        --category "$CAT" \
        --attack-models "$PROJECT/surrogates" --attack-archs cnn_a,cnn_b \
        --holdout-models "$PROJECT/surrogates" --holdout-arch cnn_c \
        --eps 0.0078431373 \
        --steps 50 \
        --step-size 0.0009803922 \
        --output "$PROJECT/transfer_check_${CAT,,}.json" || true
done

# ---------------------------------------------------------------------------
# 7. PGD candidates for the same surrogate-driven categories, attacking the
#    cnn_a+cnn_b ensemble together.
# ---------------------------------------------------------------------------
echo "=== [6/7] PGD candidates (WM_2, WM_3, WM_7, WM_8) ==="
for CAT in WM_2 WM_3 WM_7 WM_8; do
    "$PYTHON" "$PROJECT/forge_pgd.py" \
        --dataset "$DATASET" \
        --category "$CAT" \
        --models "$PROJECT/surrogates" \
        --archs cnn_a,cnn_b \
        --output-dir "$PROJECT/pgd_candidates" \
        --eps-grid 0.0039215686,0.0078431373,0.011764706 \
        --steps 50 \
        --step-size 0.0009803922
done

# ---------------------------------------------------------------------------
# 8. Build the final submission.
#
#    select_routing.py reads the transfer_check_*.json verdicts from step 6
#    and automatically falls back to the mean-residual baseline for any
#    surrogate category whose attack did not come back "likely to transfer" --
#    this is the actual decision logic, not a manual edit. WM_1/3/4/5/6 (the
#    validated hand-crafted attacks, including WM_3's by default now) are
#    always routed straight to specialized_candidates/ (now a single
#    per-category-strength output, not a strength_<value> subfolder), since
#    they don't depend on a black-box surrogate. Inspect routing.json
#    afterwards and re-run category ablations before treating this as final.
# ---------------------------------------------------------------------------
echo "=== [7/7] Build submission ==="

ROUTING="$PROJECT/routing.json"
"$PYTHON" "$PROJECT/select_routing.py" \
    --specialized-dir "$PROJECT/specialized_candidates" \
    --baseline-dir "$PROJECT/baseline_candidates" --baseline-strength 0.02 \
    --pgd-dir "$PROJECT/pgd_candidates" --pgd-eps 0.007843 \
    --transfer-checks-dir "$PROJECT" \
    --output "$ROUTING"

"$PYTHON" "$PROJECT/build_submission.py" \
    --dataset "$DATASET" \
    --routing "$ROUTING" \
    --output-dir "$PROJECT/final_submission" \
    --zip "$PROJECT/final_submission.zip"

echo "Pipeline complete. Submission zip: $PROJECT/final_submission.zip"
