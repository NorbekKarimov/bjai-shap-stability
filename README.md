# Class-dependent instability of SHAP explanations and a low-cost reliability gate

Code, data partitions and analysis outputs for the manuscript
"Class-Dependent Instability of SHAP Explanations in Imbalanced Forest Cover Classification
and a Low-Cost Reliability Gate for Decision Support" (submitted to the Babylonian Journal of Artificial Intelligence).

## Data
UCI Forest Covertype (581,012 cells, 54 features, 7 classes). It is not redistributed here;
the scripts download it with `sklearn.datasets.fetch_covtype`.

## Pipeline

| Script | Task in paper | Input | Output |
|---|---|---|---|
| `scripts/stage1_train.py` | Task 1 (Sec. 4.2 / 6.1) | Covertype | `results_v3/` (splits, hyperparameters, metrics, fitted models) |
| `scripts/stage2_shap.py` | Task 2 (Sec. 4.3 / 6.2) | `results_v3/` | `results_v3/shap/` (SHAP values per model, seed, background) |
| `scripts/stage2_pack.py` | - | `results_v3/` | `bjai_stage2_pack.npz` |
| `scripts/stage3_analysis.py` | Tasks 2-3 (Sec. 4.3-4.4 / 6.2-6.3) | `bjai_stage2_pack.npz` | `analysis_stage3/` |
| `scripts/stage4_gate.py` | Task 4 (Sec. 4.5 / 6.4, Table VIII) | `bjai_stage2_pack.npz` | `analysis_stage4/` |
| `scripts/stage4b_gate_variants.py` | Task 4 (Table IX) | `bjai_stage2_pack.npz` | `analysis_stage4b/` |

Every script accepts `--demo` (where available) for a fast run on a small subset.

## Repository contents
- `data/bjai_stage2_pack.npz` - all SHAP values, base values, explained-row indices, labels,
  features and predicted probabilities used in the analyses.
- `results/stage1/` - per-seed test metrics, tuning logs of the tuning rounds, selected hyperparameters.
- `results/stage1_splits/` - index files of the shared test set and of every training/validation split.
- `results/analysis_stage3/`, `results/analysis_stage4/`, `results/analysis_stage4b/` - all tables behind the paper.
- `results/figures/` - Figs. 1-5 of the paper.

## Reproducing the analyses (Stages 3-4b, a few minutes on a laptop)
```
pip install -r requirements.txt
cp data/bjai_stage2_pack.npz .
python scripts/stage3_analysis.py
python scripts/stage4_gate.py
python scripts/stage4b_gate_variants.py
```
Reproducing Stages 1-2 from scratch takes several hours on a CPU (mostly KernelSHAP for SVC).

## Environment used
AMD Ryzen 5 7535HS (6C/12T), 8 GB RAM, no GPU, Windows 10.0.26200, Python 3.11,
scikit-learn 1.9.1, XGBoost 3.2.0, SHAP 0.51.0, NumPy 2.4.6, pandas 3.1.1.

## Citation
[To be added after publication.]
