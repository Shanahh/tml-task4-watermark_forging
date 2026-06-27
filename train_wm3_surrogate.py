#!/usr/bin/env python3
"""Deprecated thin wrapper: use `train_surrogate.py --category WM_3` directly."""
from __future__ import annotations

import sys

from train_surrogate import ResidualCNN  # noqa: F401  (re-exported for forge_wm3_pgd.py)
from train_surrogate import main as _main

if __name__ == "__main__":
    sys.argv += ["--category", "WM_3"]
    _main()
