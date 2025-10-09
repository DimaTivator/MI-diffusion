# MINDE Paper Reproduction

[![arXiv](https://img.shields.io/badge/arXiv-2310.09031-b31b1b.svg)](https://arxiv.org/abs/2310.09031)

This repository contains code and experiments that implement and explore MINDE (Mutual Information Neural Diffusion Estimation) — a method that uses score-based diffusion models to estimate mutual information (MI) between random variables. The implementation in this repo is focused on the core ideas from the MINDE paper and includes lightweight training/evaluation scripts, data generation utilities and Jupyter notebooks used for experiments.

Note: The original MINDE project and paper artifacts are located in `papers/minde/` (from the paper authors). This repository collects an implementation that reproduces the MINDE approach and provides tools to run the estimator on synthetic datasets.

## Highlights
- Implements diffusion-based MI estimation using score (denoiser) models.
- Lightweight PyTorch modules for diffusion, denoiser networks and training loops.
- Data generation helpers for synthetic correlated Gaussian data used in experiments.
- Notebooks and scripts for benchmark/experiments.

## Repository layout

- `minde/` — core implementation used by this repo
  - `model.py` — Diffusion wrapper, denoiser model (MLP), training utilities
  - `mi.py` — MI estimation helpers (score functions, Monte-Carlo and step-based estimators)
  - `benchmark.py` — higher-level orchestration to train joint/independent models and compute MI estimates
- `minde_c/` — conditional-diffusion variant (conditional denoiser + estimators)
- `gen_data.py` — synthetic data generators (block-correlated, nd-correlated, random covariance, and analytical MI for Gaussians)
- `*.ipynb` — notebooks used to explore and benchmark the estimators

## Requirements

- Python 3.10 (this codebase was developed with CPython 3.10.x, adjust if needed)
- PyTorch (cpu/cuda as desired)
- numpy, tqdm

You can use `requirements.txt` from `papers/minde/` as a starting point:

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate
pip install -r papers/minde/requirements.txt
```

## Quick start — estimate MI on synthetic data

This repository includes a convenience function `estimate_MI` in `minde/benchmark.py` which trains two diffusion models (one on the joint distribution and one on an independent-product distribution) and returns an MI estimate.

Example (run inside a Python session or notebook):

```python
import torch
from gen_data import create_block_correlated_data
from minde.benchmark import estimate_MI

# Synthetic dataset: n_samples x n_dim. n_dim must be even here.
n_samples = 20000
n_dim = 8
dep_data, indep_data, cov = create_block_correlated_data(n_samples, n_dim, r=0.7)

# Split joint into A and B halves
n_half = n_dim // 2
A = dep_data[:, :n_half]
B = dep_data[:, n_half:]

# Estimate MI (this will train models; lower epochs for a quick run)
mi_est = estimate_MI(
    A, B,
    num_epochs=50,
    batch_size_train=256,
    device='cpu',       # use 'cuda' if you have a GPU
    method='steps',     # 'steps' or 'monte-carlo'
    dt=0.01,
    T=5,
)

print('Estimated MI:', mi_est)
```

Notes:
- `method='steps'` uses a discretized time integration of the squared-score-difference; `method='monte-carlo'` performs a Monte-Carlo estimate.
- Increase `num_epochs` and move to `device='cuda'` for production-quality runs.

## API notes

- `minde.model.Diffusion(denoiser, loss=None, T=5)` — Diffusion wrapper. Call with tensors to compute denoising loss during training. Use `.sample(...)` to run the generative sampling procedure provided a schedule.
- `minde.model.MLPDenoiser(data_dim, hidden_dim=256, add_dim=1)` — MLP denoiser; takes noisy input and time/scale `p` as an extra input.
- `minde.mi.score_func(...)`, `estimate_MI_steps(...)`, `estimate_MI_monte_carlo(...)` — lower-level helpers to compute score differences and MI estimates.
- `minde.benchmark.estimate_MI(A, B, ...)` — higher-level experiment: trains joint and marginal models and computes an MI estimate. Useful to reproduce the experiments quickly.

## Reproducing paper results

The original MINDE artifacts and experimental scripts are included under `papers/minde/` (with a `quickstart.ipynb` and config/scripts). To reproduce the paper results you will likely need to:

1. Install dependencies listed in `papers/minde/requirements.txt`.
2. Inspect `papers/minde/src/scripts/config.py` for experiment configurations.
3. Run the notebooks (e.g., `papers/minde/quickstart.ipynb`) or the provided scripts under `papers/minde/src/scripts/`.

This repository provides a compact reimplementation of the core estimator and utilities to run targeted experiments. For full-scale reproduction (ICLR experiments) follow the paper's README in `papers/minde/`.

## Tips and troubleshooting

- GPU usage: change `device='cuda'` in calls to take advantage of CUDA. Ensure you installed a CUDA-enabled PyTorch build.
- If training is slow, reduce `num_epochs` or `batch_size_train` for quick debugging. For final runs, use larger batch sizes and run on GPU.
- If you run into numerical issues during sampling, ensure p/t schedules and shapes match the model's expectations. The code uses p = exp(-t) as the time-to-scale mapping.

## Where to go next

- Inspect `minde/` to customize architectures (larger MLPs, add skip connections, or swap to Transformers for images).
- Use `minde_c/` if you want to run the conditional diffusion variant.
- Add small experiments in the notebooks or create scripts that sweep over `num_epochs`, `T`, and `dt` to check estimator convergence.

