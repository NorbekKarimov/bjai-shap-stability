"""
BJAI article - Stage 2: SHAP explanations for RQ3 (feature influence) and RQ4 / G1 (explanation stability)

Reads the frozen Stage 1 v3 outputs (results_v3/) and computes SHAP values on a fixed,
class-balanced explanation set drawn from the shared test set.

Design (all explainers estimate interventional / marginal Shapley values with the SAME background rows):
  - RF, XGB : shap.TreeExplainer(feature_perturbation="interventional", model_output="raw")
              -> exact given the background; RF in probability units, XGB in per-class margin units
  - SVC, MLP: shap.KernelExplainer                                       -> sampling approximation
  Two sources of variation are separated (cf. Hwang et al., 2026):
  - model-induced     : models of different training seeds, background fixed (BG_SEEDS[0])
  - explainer-induced : the seed-0 model, background re-drawn with BG_SEEDS[1:]
  Attributions are stored for every class output; the analysis (Stage 3) chooses which output to use.

Usage : python stage2_shap.py          (real run, needs results_v3/)
        python stage2_shap.py --demo   (uses results_demo/ from 'stage1_train.py --demo'; NOT for the article)

Output (results_v3/shap/ ; resumable, existing files are skipped):
  explain_idx_tree.npy, explain_idx_kernel.npy    row indices into the full dataset
  background_bg{b}.npy                            background row indices
  shap_{model}_seed{s}_bg{b}.npy                  float32 array (n_explain, 54, 7)
  base_{model}_seed{s}_bg{b}.npy                  expected value per class
  timing.csv, config.json
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import shap

# ============================ CONFIG ============================
TREE_MODELS = ["RF", "XGB"]
KERNEL_MODELS = ["MLP", "SVC"]          # MLP first: it is fast; SVC is the slow one
MODEL_SEEDS = {"RF": list(range(10)), "XGB": list(range(10)),
               "MLP": list(range(10)), "SVC": [0, 1, 2, 3, 4]}   # SVC limited by run time
BG_SEEDS = [0, 1, 2]                    # [0] = fixed background; [1:] = explainer-induced runs
N_PER_CLASS_TREE = 100                  # 7 x 100 = 700 explained rows for RF/XGB
N_PER_CLASS_KERNEL = 20                 # 7 x 20  = 140 explained rows for SVC/MLP (subset of the 700)
N_BACKGROUND = 25                       # background rows (shared by all explainers)
KERNEL_NSAMPLES = 500                   # coalition samples per row for KernelSHAP
EXPLAIN_SEED = 7
# ================================================================


def load_X(demo):
    if demo:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=6000, n_features=54, n_informative=12,
                                   n_classes=7, n_clusters_per_class=1,
                                   weights=[.45, .35, .07, .005, .02, .03, .075], random_state=0)
        return X.astype(np.float32), y.astype(np.int64)
    from sklearn.datasets import fetch_covtype
    d = fetch_covtype(as_frame=True)
    return d.data.to_numpy(dtype=np.float32), d.target.to_numpy().astype(np.int64) - 1


def balanced_sample(idx, y, n_per_class, rng):
    out = []
    for c in np.unique(y[idx]):
        pool = idx[y[idx] == c]
        out.append(rng.choice(pool, size=min(n_per_class, len(pool)), replace=False))
    return np.sort(np.concatenate(out))


def to_3d(values, n_rows, n_feat):
    """Normalise SHAP output to (rows, features, classes)."""
    if isinstance(values, list):                       # older shap: list of (rows, features)
        values = np.stack(values, axis=-1)
    values = np.asarray(values)
    if values.ndim == 2:                               # single output
        values = values[..., None]
    if values.shape[0] != n_rows or values.shape[1] != n_feat:
        raise ValueError(f"unexpected SHAP shape {values.shape}")
    return values.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    global N_PER_CLASS_TREE, N_PER_CLASS_KERNEL, KERNEL_NSAMPLES
    src = "results_demo" if args.demo else "results_v3"
    if args.demo:
        N_PER_CLASS_TREE, N_PER_CLASS_KERNEL, KERNEL_NSAMPLES = 6, 2, 60
        for k in MODEL_SEEDS:
            MODEL_SEEDS[k] = MODEL_SEEDS[k][:2]
    out = os.path.join(src, "shap")
    os.makedirs(out, exist_ok=True)

    X, y = load_X(args.demo)
    n_feat = X.shape[1]
    test_idx = np.load(os.path.join(src, "split_test_idx.npy"))
    train0 = np.load(os.path.join(src, "split_seed0_train_idx.npy"))

    # ---------- fixed explanation sets ----------
    p_tree = os.path.join(out, "explain_idx_tree.npy")
    p_kern = os.path.join(out, "explain_idx_kernel.npy")
    if os.path.exists(p_tree):
        ex_tree, ex_kern = np.load(p_tree), np.load(p_kern)
    else:
        rng = np.random.default_rng(EXPLAIN_SEED)
        ex_tree = balanced_sample(test_idx, y, N_PER_CLASS_TREE, rng)
        ex_kern = balanced_sample(ex_tree, y, N_PER_CLASS_KERNEL, rng)
        np.save(p_tree, ex_tree)
        np.save(p_kern, ex_kern)
    print(f"Explain rows: tree models {len(ex_tree)}, kernel models {len(ex_kern)}; "
          f"per class (tree): {np.bincount(y[ex_tree]).tolist()}")

    # ---------- backgrounds (drawn from the seed-0 training set) ----------
    bgs = {}
    for b in BG_SEEDS:
        p = os.path.join(out, f"background_bg{b}.npy")
        if not os.path.exists(p):
            np.save(p, np.sort(np.random.default_rng(1000 + b).choice(train0, N_BACKGROUND,
                                                                       replace=False)))
        bgs[b] = np.load(p)

    # ---------- runs: (model, seed, bg) ----------
    runs = []
    for m in TREE_MODELS + KERNEL_MODELS:
        for s in MODEL_SEEDS[m]:
            runs.append((m, s, BG_SEEDS[0]))              # model-induced
        for b in BG_SEEDS[1:]:
            runs.append((m, MODEL_SEEDS[m][0], b))         # explainer-induced
    tpath = os.path.join(out, "timing.csv")

    for m, s, b in runs:
        f = os.path.join(out, f"shap_{m}_seed{s}_bg{b}.npy")
        if os.path.exists(f):
            print(f"skip {m} seed{s} bg{b}")
            continue
        model = joblib.load(os.path.join(src, "models", f"seed{s}_{m}.joblib"))
        Xb = X[bgs[b]]
        rows = ex_tree if m in TREE_MODELS else ex_kern
        t0 = time.time()
        if m in TREE_MODELS:
            # "raw": class probability for RF, per-class margin (log-odds) for XGB.
            # shap does not support model_output="probability" for multi:softprob with a background.
            ex = shap.TreeExplainer(model, data=Xb, feature_perturbation="interventional",
                                    model_output="raw")
            vals = ex.shap_values(X[rows], check_additivity=False)
        else:
            ex = shap.KernelExplainer(model.predict_proba, Xb)
            np.random.seed(10_000 * b + s)                 # KernelSHAP sampling is global-RNG based
            vals = ex.shap_values(X[rows], nsamples=KERNEL_NSAMPLES, silent=True)
        sec = time.time() - t0
        vals = to_3d(vals, len(rows), n_feat)
        np.save(f, vals)
        np.save(os.path.join(out, f"base_{m}_seed{s}_bg{b}.npy"),
                np.atleast_1d(np.asarray(ex.expected_value, dtype=np.float32)))
        pd.DataFrame([{"model": m, "seed": s, "bg": b, "rows": len(rows), "seconds": sec}]).to_csv(
            tpath, mode="a", index=False, header=not os.path.exists(tpath))
        print(f"{m} seed{s} bg{b}: {vals.shape} in {sec:.0f}s")

    json.dump({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "shap": shap.__version__, "demo": args.demo,
               "TREE_MODELS": TREE_MODELS, "KERNEL_MODELS": KERNEL_MODELS,
               "MODEL_SEEDS": MODEL_SEEDS, "BG_SEEDS": BG_SEEDS,
               "N_PER_CLASS_TREE": N_PER_CLASS_TREE, "N_PER_CLASS_KERNEL": N_PER_CLASS_KERNEL,
               "N_BACKGROUND": N_BACKGROUND, "KERNEL_NSAMPLES": KERNEL_NSAMPLES,
               "EXPLAIN_SEED": EXPLAIN_SEED, "tree_explainer": "interventional, model_output=raw (RF: probability, XGB: margin)",
               "kernel_explainer": "KernelExplainer(predict_proba)"},
              open(os.path.join(out, "config.json"), "w"), indent=2)
    print(f"Done. SHAP files in ./{out}/")


if __name__ == "__main__":
    main()
