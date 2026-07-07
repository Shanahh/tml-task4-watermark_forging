# Archived / deprecated scripts

These scripts belong to earlier approaches that were **superseded** and are **not
part of the final submission pipeline**. They are kept only for reference and
reproducibility of the negative results discussed in the report. Each was
originally written to run from the repository root (it imports `common` from the
parent directory), so to run one, copy it back to the repo root first.

The final pipeline (see the top-level `README.md`) is: `forge_simple.py` for the
statistical-estimation base, then `forge_known_scheme.py` to overwrite the
identified groups (WM_2/7/8) with their real open-source encoders.

## Why each was retired

| Script | What it did | Why it was dropped |
|---|---|---|
| `forge_specialized.py` | Hand-crafted per-domain attacks (Cb residual, Fourier phase, block-DCT, LSB, block-statistic and WMCopier-style estimators) | All beaten by the raw mean-difference in `forge_simple.py`; its only still-needed helper (`estimate_content`) was moved into `common.py`. |
| `forge_baseline.py` | Mean high-pass-residual template transfer | The high-pass filter discarded real watermark signal; raw (non-high-pass) averaging in `forge_simple.py` scored higher. |
| `calibrate_strength.py` | Amplitude-calibrated the attack strength to the genuine watermark's own projection | Fixed a real 15–44σ overshoot but did not improve the score. |
| `train_surrogate.py`, `forge_pgd.py`, `check_surrogate_transfer.py` | Surrogate-classifier + constrained PGD, with a cross-architecture transfer check | The surrogates did not transfer to the hidden detector. |
| `train_wm3_surrogate.py`, `forge_wm3_pgd.py` | Deprecated WM_3-only wrappers around the surrogate pipeline | Superseded together with the surrogate pipeline. |
| `sweep_lpips_strength.py` | Swept LPIPS vs a shared strength grid | Superseded by per-image LPIPS capping (built into `forge_simple.py`) and `sweep_regen_sigma.py`. |
| `select_routing.py`, `build_submission.py` | Routing JSON + submission assembly for the old multi-candidate specialized pipeline | The final pipeline assembles the submission directly in `forge_known_scheme.py`. |
| `score_with_diagnostics.py` | Scored our own forgeries with the diagnostic classifiers | An analysis probe; not needed once scheme identification took over. |
| `check_lsb_roundtrip.py` | Verified the old WM_5 LSB-plane attack survived PNG save/reload | The LSB attack itself was dropped. |
