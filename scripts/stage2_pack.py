"""
BJAI article - pack Stage 2 outputs into ONE compressed file for analysis

Why: the analysis also needs the true labels and feature values of the explained rows,
which are not in results_v3/shap/. This script collects everything into one .npz.

Usage : python stage2_pack.py          -> bjai_stage2_pack.npz   (real run, reads results_v3/)
        python stage2_pack.py --demo   -> bjai_stage2_pack_demo.npz (reads results_demo/)

Contents of the .npz:
  shap__{model}__seed{s}__bg{b}   float32 (rows, 54, 7)   SHAP values, unchanged
  base__{model}__seed{s}__bg{b}   float32 (7,)            expected values
  explain_idx_tree / explain_idx_kernel / background_bg{b}  row indices into the full dataset
  y_explain_tree, X_explain_tree   labels (0..6) and features of the 700 tree rows
  X_background_bg{b}               features of the background rows
  proba_explain__{model}__seed{s}  float32 (rows, 7) test-set probabilities of the explained rows
  y_test                           int8 labels of the 50,000 shared test rows (for Stage 3)
  feature_names, timing_csv, config_json   text
Nothing is recomputed: values are copied from existing Stage 1/2 files.
"""
import argparse
import glob
import io
import os
import re

import numpy as np


def load_xy(demo):
    if demo:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=6000, n_features=54, n_informative=12,
                                   n_classes=7, n_clusters_per_class=1,
                                   weights=[.45, .35, .07, .005, .02, .03, .075], random_state=0)
        return X.astype(np.float32), y.astype(np.int64), [f"f{i}" for i in range(54)]
    from sklearn.datasets import fetch_covtype
    d = fetch_covtype(as_frame=True)
    return (d.data.to_numpy(dtype=np.float32), d.target.to_numpy().astype(np.int64) - 1,
            list(d.data.columns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    src = "results_demo" if a.demo else "results_v3"
    sh = os.path.join(src, "shap")
    out_file = "bjai_stage2_pack_demo.npz" if a.demo else "bjai_stage2_pack.npz"

    X, y, names = load_xy(a.demo)
    test_idx = np.load(os.path.join(src, "split_test_idx.npy"))
    pos = {r: i for i, r in enumerate(test_idx)}                 # dataset row -> test position
    ex_tree = np.load(os.path.join(sh, "explain_idx_tree.npy"))
    ex_kern = np.load(os.path.join(sh, "explain_idx_kernel.npy"))

    pack = {"explain_idx_tree": ex_tree, "explain_idx_kernel": ex_kern,
            "y_explain_tree": y[ex_tree].astype(np.int8), "X_explain_tree": X[ex_tree],
            "y_test": y[test_idx].astype(np.int8),
            "feature_names": np.array(names)}

    for f in sorted(glob.glob(os.path.join(sh, "background_bg*.npy"))):
        b = re.search(r"bg(\d+)", f).group(1)
        idx = np.load(f)
        pack[f"background_bg{b}"] = idx
        pack[f"X_background_bg{b}"] = X[idx]

    n = 0
    for f in sorted(glob.glob(os.path.join(sh, "shap_*_seed*_bg*.npy"))):
        m, s, b = re.search(r"shap_(\w+?)_seed(\d+)_bg(\d+)\.npy", os.path.basename(f)).groups()
        pack[f"shap__{m}__seed{s}__bg{b}"] = np.load(f).astype(np.float32)
        pack[f"base__{m}__seed{s}__bg{b}"] = np.load(
            os.path.join(sh, f"base_{m}_seed{s}_bg{b}.npy")).astype(np.float32)
        n += 1
        key = f"proba_explain__{m}__seed{s}"
        if key not in pack:
            rows = ex_tree if m in ("RF", "XGB") else ex_kern
            proba = np.load(os.path.join(src, f"proba_test_seed{s}_{m}.npy"))
            pack[key] = proba[[pos[r] for r in rows]].astype(np.float32)

    for name in ("timing.csv", "config.json"):
        p = os.path.join(sh, name)
        pack[name.replace(".", "_")] = np.array(open(p, encoding="utf-8").read() if os.path.exists(p) else "")

    np.savez_compressed(out_file, **pack)
    print(f"Packed {n} SHAP runs -> {out_file} ({os.path.getsize(out_file) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
