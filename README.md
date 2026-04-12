# FMP: Framelet Message Passing

This repository contains the implementation of **FMP (Framelet Message Passing)** from the paper [*Framelet message passing*](https://doi.org/10.1016/j.acha.2025.101773).

The codebase builds on the continuous-depth graph learning framework already present in this repository and adds the framelet-based message passing components used by FMP.

## Repository layout

- `src/framelet_message_passing.py` – canonical FMP module containing framelet operator construction and the `UFGLevel` message passing layer.
- `src/function_laplacian_diffusion.py` – ODE function that uses `UFGLevel` when `--data_norm ufg` is enabled.
- `src/block_constant.py` – prepares and caches framelet operators for the constant ODE block.
- `src/run_GNN.py` – main training entry point for node classification experiments.
- `src/sMP2.py` – legacy experiment script that now reuses the shared FMP module.
- `test/` – unit tests for the original codebase.

## Installation

Create the environment from the provided Conda specification:

```bash
conda env create -f /home/runner/work/FMP/FMP/environment.yml
conda activate fmp
```

If you prefer a manual setup, install the PyTorch / PyG versions that match your CUDA configuration and then install the remaining dependencies from `environment.yml`.

## Data

Most citation-network datasets are downloaded automatically on first run.

Create a root-level data directory before training:

```bash
mkdir -p /home/runner/work/FMP/FMP/data
```

## Running FMP

The main FMP path is enabled through `--data_norm ufg`.

Example:

```bash
cd /home/runner/work/FMP/FMP/src
python run_GNN.py \
  --dataset Cora \
  --block constant \
  --function laplacian \
  --data_norm ufg \
  --init_scale_1 0.1 \
  --init_scale_2 0.1 \
  --init_scale_3 0.6
```

Useful FMP-related flags in `run_GNN.py`:

- `--data_norm ufg` enables the framelet-based diffusion path.
- `--init_scale_1`, `--init_scale_2`, `--init_scale_3` set the initial framelet coefficients.
- `--channel_mix` applies a linear channel mixing step inside each `UFGLevel`.

## Development notes

The active FMP implementation is now centralized in `src/framelet_message_passing.py`. If you need to adjust framelet construction, filter definitions, or the reusable message passing layer, start there.

## Validation

The repository includes `unittest`-based tests under `test/`. In an environment with the project dependencies installed, run:

```bash
cd /home/runner/work/FMP/FMP
python -m unittest discover -s test
```

## Citation

If you use this repository, please cite:

```bibtex
@article{framelet_message_passing_2025,
  title   = {Framelet message passing},
  journal = {Applied and Computational Harmonic Analysis},
  year    = {2025},
  doi     = {10.1016/j.acha.2025.101773}
}
```
