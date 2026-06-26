#!/bin/bash
set -eo pipefail

# Finds the real usable strength budget for the specialized (hand-crafted)
# attacks by directly measuring LPIPS at a wide range of strengths, instead
# of trusting the conservative default grid or a visual eyeball check.
#
# Usage:
#   ./run_lpips_sweep.sh [--categories WM_1,WM_3,WM_4,WM_5,WM_6] [--strength-grid 0.0025,...]
#
# Look at the printed table (or the JSON output) for where Sqlt starts
# dropping noticeably -- that's the real ceiling on --specialized-strength
# for run_pipeline.sh, not whatever the default grid happens to contain.

PROJECT=/home/atml_team052/tml-task4-watermark_forging
ENV=/home/atml_team052/.conda/envs/tmltask4
PYTHON="$ENV/bin/python"

export PYTHONNOUSERSITE=1
export HF_HOME=/home/atml_team052/.cache/huggingface
export PIP_CACHE_DIR=/home/atml_team052/.cache/pip

CATEGORIES="WM_1,WM_3,WM_4,WM_5,WM_6"
STRENGTH_GRID="0.0025,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5"
OUTPUT="$PROJECT/lpips_strength_sweep.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --categories) CATEGORIES="$2"; shift 2 ;;
        --strength-grid) STRENGTH_GRID="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --dataset) DATASET_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

DATASET="${DATASET_OVERRIDE:-$PROJECT/dataset}"

if [ ! -x "$PYTHON" ]; then
    echo "Conda env not found at $ENV -- run run_pipeline.sh first to set it up." >&2
    exit 1
fi

"$PYTHON" -m pip install --quiet lpips

echo "=== LPIPS-vs-strength sweep ==="
echo "categories: $CATEGORIES"
echo "strength grid: $STRENGTH_GRID"

"$PYTHON" "$PROJECT/sweep_lpips_strength.py" \
    --dataset "$DATASET" \
    --categories "$CATEGORIES" \
    --strength-grid "$STRENGTH_GRID" \
    --output "$OUTPUT"

echo "Sweep complete. Results: $OUTPUT"
