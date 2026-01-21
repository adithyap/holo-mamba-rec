# HoloMambaRec

HoloMambaRec is a minimal, single-file reference implementation of holographic binding with Mamba state space models for sequential recommendation. It includes data loaders for Amazon Beauty and MovieLens-1M, model definitions (HoloMamba, SASRec, GRU4Rec, and an item-only ablation), and scripts for benchmarking, ablations, compression profiling, and lightweight grid search.

## Repository contents
- `HoloMambaRec.py` — end-to-end script (data prep, models, training loops, evals, and plotting).
- `arch.png`, `bench.png`, `ablation.png` — diagrams and figure exports used in the paper draft.
- `learning_curve_*.png`, `compression_*.png`, `ablation_study.png` — auto-generated metrics/plots from the script.
- `grid_search_results.csv` — example CSV emitted by the grid search helper.
- `holo_mamba_rec_final.pdf` and supporting `.tex`/`.bib` files — paper draft assets.

## Setup
Python 3.9+ and a CUDA GPU are recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch mamba-ssm causal-conv1d pandas numpy matplotlib tqdm requests
# Optional (if you want parity with the Colab snippet in the code header):
# pip install datasets transformers google-generativeai
```

## Running the script
The main entry point is `HoloMambaRec.py`. It will download datasets on first run into `data/amazon-beauty` and `data/ml-1m`.

```bash
python HoloMambaRec.py
```

What it does:
- Runs the benchmark loop on Amazon-Beauty and ML-1M with HoloMamba, SASRec, and GRU4Rec.
- Runs the binding vs. item-only ablation.
- Runs a grid search and writes `grid_search_results.csv` (set `GRID_MAX_TRIALS=0` or a small integer to limit/skip it).
- Runs compression/latency evaluations and saves plots.

Notes:
- The full run is compute-heavy; use a GPU and consider trimming epochs or commenting out sections in `__main__` for quicker smoke tests.
- Figures and CSVs are written to the repository root (e.g., `learning_curve_amazon-beauty_hr.png`, `compression_runtime_ml-1m.png`).

## Customization tips
- Adjust global defaults in `GLOBAL_CONFIG` (e.g., `d_model`, `n_layers`, `batch_size`, `epochs`, `use_compression`).
- Modify dataset-specific overrides inside `run_benchmark` and `run_compression_evals`.
- To evaluate only cold-start behavior or compression, call `evaluate_cold_start` or `run_compression_evals` directly from a notebook after constructing your loaders.

## Attribution
If you use this codebase, please reference the HoloMambaRec project and include links back to this repository.
