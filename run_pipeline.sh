#!/bin/bash
set -eo pipefail

PROJECT=/home/atml_team052/tml-task4-watermark_forging
ENV=/home/atml_team052/.conda/envs/tmltask4
PYTHON="$ENV/bin/python"

export PYTHONNOUSERSITE=1
export HF_HOME=/home/atml_team052/.cache/huggingface
export PIP_CACHE_DIR=/home/atml_team052/.cache/pip

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
#    WM_1 Cb residual, WM_4 Fourier-phase, WM_5 LSB bit-plane, WM_6 DCT.
# ---------------------------------------------------------------------------
echo "=== [3/6] Specialized candidates (WM_1, WM_4, WM_5, WM_6) ==="
"$PYTHON" "$PROJECT/forge_specialized.py" \
    --dataset "$DATASET" \
    --output-dir "$PROJECT/specialized_candidates" \
    --strength-grid 0.0025,0.005,0.01,0.02

# ---------------------------------------------------------------------------
# 5. Surrogate-classifier ensembles for categories with no validated
#    hand-crafted signal: WM_2, WM_3, WM_7, WM_8.
# ---------------------------------------------------------------------------
echo "=== [4/6] Surrogate ensembles (WM_2, WM_3, WM_7, WM_8) ==="
for CAT in WM_2 WM_3 WM_7 WM_8; do
    "$PYTHON" "$PROJECT/train_surrogate.py" \
        --dataset "$DATASET" \
        --category "$CAT" \
        --output-dir "$PROJECT/surrogates" \
        --ensemble-size 5 \
        --epochs 40
done

# ---------------------------------------------------------------------------
# 6. PGD candidates for the same surrogate-driven categories.
# ---------------------------------------------------------------------------
echo "=== [5/6] PGD candidates (WM_2, WM_3, WM_7, WM_8) ==="
for CAT in WM_2 WM_3 WM_7 WM_8; do
    "$PYTHON" "$PROJECT/forge_pgd.py" \
        --dataset "$DATASET" \
        --category "$CAT" \
        --models "$PROJECT/surrogates" \
        --output-dir "$PROJECT/pgd_candidates" \
        --eps-grid 0.0039215686,0.0078431373,0.011764706 \
        --steps 50 \
        --step-size 0.0009803922
done

# ---------------------------------------------------------------------------
# 7. Build the final submission.
#
#    Routing below picks one candidate per category as a sane starting
#    point (strength_0.005 for specialized, eps_0.007843 for PGD, i.e. 2/255).
#    Re-run category ablations against baseline_candidates and edit
#    routing.json before submitting for real — this default routing is not
#    guaranteed to be the best-scoring combination.
# ---------------------------------------------------------------------------
echo "=== [6/6] Build submission ==="
ROUTING="$PROJECT/routing.json"
cat > "$ROUTING" <<JSON
{
  "WM_1": "$PROJECT/specialized_candidates/strength_0.005",
  "WM_2": "$PROJECT/pgd_candidates/wm_2/eps_0.007843",
  "WM_3": "$PROJECT/pgd_candidates/wm_3/eps_0.007843",
  "WM_4": "$PROJECT/specialized_candidates/strength_0.005",
  "WM_5": "$PROJECT/specialized_candidates/strength_0.005",
  "WM_6": "$PROJECT/specialized_candidates/strength_0.005",
  "WM_7": "$PROJECT/pgd_candidates/wm_7/eps_0.007843",
  "WM_8": "$PROJECT/pgd_candidates/wm_8/eps_0.007843"
}
JSON

"$PYTHON" "$PROJECT/build_submission.py" \
    --dataset "$DATASET" \
    --routing "$ROUTING" \
    --output-dir "$PROJECT/final_submission" \
    --zip "$PROJECT/final_submission.zip"

echo "Pipeline complete. Submission zip: $PROJECT/final_submission.zip"
