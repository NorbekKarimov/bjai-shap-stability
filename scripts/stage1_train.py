"""
BJAI article - Stage 1 (v3): data split, model training and predictive evaluation (RQ1, RQ2)

Dataset : UCI Forest Covertype (581,012 x 54, 7 classes), via sklearn.datasets.fetch_covtype
Models  : Random Forest, XGBoost, SVC (RBF), MLP
Design  : fixed stratified test set (N_TEST) shared by all seeds;
          per seed a stratified train (N_TRAIN) and validation (N_VAL) sample from the remaining pool;
          hyperparameters tuned on seed 0 validation (macro-F1), then frozen for all seeds.

Usage   : python stage1_train.py            (real run; downloads Covertype on first use, ~11 MB)
          python stage1_train.py --demo     (tiny synthetic check that the pipeline runs; NOT for the article)

Outputs (folder results/):
  split_test_idx.npy, split_seed{s}_train_idx.npy, split_seed{s}_val_idx.npy
  hyperparams.json, tuning_log.csv
  metrics_per_seed.csv, per_class_recall.csv, confusion_seed{s}_{model}.csv
  proba_test_seed{s}_{model}.npy           (test-set class probabilities, used later for the gate, RQ5)
  models/seed{s}_{model}.joblib            (fitted models, used later for SHAP, RQ3-RQ4)
  environment.json                         (versions, CPU, timestamp)

Integrity: every number in the article comes from these files. Nothing is filled in by hand.
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, recall_score)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ============================ CONFIG ============================
SEEDS = list(range(10))   # v2: 10 seeds (Wilcoxon with n=5 cannot reach p<0.05)
N_TEST = 50_000
N_TRAIN = 20_000
N_VAL = 5_000
SPLIT_SEED = 2026          # fixes the shared test set
N_JOBS = 4                 # keep memory use moderate on an 8 GB machine
OUT = "results_v3"         # v1 (results/) and v2 (results_v2/) are kept unchanged
N_CONTINUOUS = 10          # first 10 Covertype columns are continuous; the other 44 are binary

GRIDS = {   # v3: XGB/SVC frozen at their v2 optima (interior of the v2 grid, see results_v2/tuning_log.csv);
            # RF/MLP extended because their v2 optima were still on the grid edge
    "RF":  {"n_estimators": [300], "max_features": [0.5, 0.7, 1.0], "min_samples_leaf": [1]},
    "XGB": {"n_estimators": [400], "max_depth": [10], "learning_rate": [0.1],
            "subsample": [0.8], "colsample_bytree": [0.8]},
    "SVC": {"C": [100], "gamma": ["scale"]},
    "MLP": {"hidden_layer_sizes": [(512, 256), (1024, 512)], "alpha": [1e-3, 1e-2]},
}
# ================================================================


def load_data(demo):
    if demo:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=6000, n_features=54, n_informative=12,
                                   n_classes=7, n_clusters_per_class=1,
                                   weights=[.45, .35, .07, .005, .02, .03, .075], random_state=0)
        return X.astype(np.float32), y.astype(np.int64), [f"f{i}" for i in range(54)]
    from sklearn.datasets import fetch_covtype
    d = fetch_covtype(as_frame=True)
    X = d.data.to_numpy(dtype=np.float32)
    y = d.target.to_numpy().astype(np.int64) - 1          # classes 1..7 -> 0..6
    return X, y, list(d.data.columns)


def build(name, params, seed, n_classes):
    if name == "RF":
        return RandomForestClassifier(random_state=seed, n_jobs=N_JOBS, **params)
    if name == "XGB":
        return XGBClassifier(random_state=seed, n_jobs=N_JOBS, tree_method="hist",
                             objective="multi:softprob", num_class=n_classes,
                             eval_metric="mlogloss", **params)
    scale = ColumnTransformer([("num", StandardScaler(), list(range(N_CONTINUOUS)))],
                              remainder="passthrough")
    if name == "SVC":
        clf = SVC(kernel="rbf", probability=True, random_state=seed, cache_size=500, **params)
    else:
        clf = MLPClassifier(early_stopping=True, max_iter=300, random_state=seed, **params)
    return Pipeline([("scale", scale), ("clf", clf)])


def evaluate(y_true, y_pred, n_classes):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }, recall_score(y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    global N_TEST, N_TRAIN, N_VAL, OUT
    if args.demo:
        N_TEST, N_TRAIN, N_VAL, OUT = 1500, 2000, 500, "results_demo"
        for g in GRIDS.values():
            if "n_estimators" in g:
                g["n_estimators"] = [50]

    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    X, y, feat_names = load_data(args.demo)
    n_classes = int(y.max()) + 1
    print(f"Data: {X.shape}, classes: {n_classes}, counts: {np.bincount(y).tolist()}")

    # ---------- fixed shared test set ----------
    all_idx = np.arange(len(y))
    pool_idx, test_idx = train_test_split(all_idx, test_size=N_TEST, stratify=y,
                                          random_state=SPLIT_SEED)
    np.save(os.path.join(OUT, "split_test_idx.npy"), test_idx)
    X_test, y_test = X[test_idx], y[test_idx]

    def seed_split(seed):
        tr, rest = train_test_split(pool_idx, train_size=N_TRAIN, stratify=y[pool_idx],
                                    random_state=seed)
        va, _ = train_test_split(rest, train_size=N_VAL, stratify=y[rest], random_state=seed)
        np.save(os.path.join(OUT, f"split_seed{seed}_train_idx.npy"), tr)
        np.save(os.path.join(OUT, f"split_seed{seed}_val_idx.npy"), va)
        return tr, va

    # ---------- tuning on seed 0 ----------
    hp_path = os.path.join(OUT, "hyperparams.json")
    if os.path.exists(hp_path):
        best = json.load(open(hp_path))
        print("Loaded frozen hyperparameters:", best)
    else:
        tr, va = seed_split(SEEDS[0])
        best, log = {}, []
        for name, grid in GRIDS.items():
            scores = []
            for params in ParameterGrid(grid):
                t0 = time.time()
                m = build(name, params, SEEDS[0], n_classes).fit(X[tr], y[tr])
                f1 = f1_score(y[va], m.predict(X[va]), average="macro")
                scores.append((f1, params))
                log.append({"model": name, "params": json.dumps(params, default=str),
                            "val_macro_f1": f1, "fit_s": time.time() - t0})
                print(f"  tune {name} {params}: val macro-F1 = {f1:.4f}")
            best[name] = max(scores, key=lambda s: s[0])[1]
        pd.DataFrame(log).to_csv(os.path.join(OUT, "tuning_log.csv"), index=False)
        json.dump(best, open(hp_path, "w"), indent=2, default=str)
        print("Frozen hyperparameters:", best)
    for name in best:                      # JSON turns tuples into lists
        if "hidden_layer_sizes" in best[name]:
            best[name]["hidden_layer_sizes"] = tuple(best[name]["hidden_layer_sizes"])

    # ---------- training / evaluation over seeds (resumable) ----------
    met_path = os.path.join(OUT, "metrics_per_seed.csv")
    rec_path = os.path.join(OUT, "per_class_recall.csv")
    done = set()
    if os.path.exists(met_path):
        prev = pd.read_csv(met_path)
        done = set(zip(prev.seed, prev.model))
    for seed in SEEDS:
        tr, _ = seed_split(seed)
        for name in GRIDS:
            if (seed, name) in done:
                print(f"skip seed {seed} {name} (already done)")
                continue
            t0 = time.time()
            m = build(name, best[name], seed, n_classes).fit(X[tr], y[tr])
            fit_s = time.time() - t0
            t0 = time.time()
            proba = m.predict_proba(X_test).astype(np.float32)
            pred_s = time.time() - t0
            y_pred = proba.argmax(1)
            scores, rec = evaluate(y_test, y_pred, n_classes)
            row = {"seed": seed, "model": name, **scores, "fit_s": fit_s,
                   "predict_s_per_1k": pred_s / len(y_test) * 1000}
            pd.DataFrame([row]).to_csv(met_path, mode="a", index=False,
                                       header=not os.path.exists(met_path))
            pd.DataFrame([{"seed": seed, "model": name,
                           **{f"recall_c{c}": r for c, r in enumerate(rec)}}]).to_csv(
                rec_path, mode="a", index=False, header=not os.path.exists(rec_path))
            pd.DataFrame(confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))).to_csv(
                os.path.join(OUT, f"confusion_seed{seed}_{name}.csv"))
            np.save(os.path.join(OUT, f"proba_test_seed{seed}_{name}.npy"), proba)
            joblib.dump(m, os.path.join(OUT, "models", f"seed{seed}_{name}.joblib"), compress=3)
            print(f"seed {seed} {name}: acc={scores['accuracy']:.4f} "
                  f"macroF1={scores['macro_f1']:.4f} fit={fit_s:.0f}s")

    import sklearn, xgboost
    json.dump({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "python": sys.version, "platform": platform.platform(),
               "processor": platform.processor(), "cpu_count": os.cpu_count(),
               "numpy": np.__version__, "pandas": pd.__version__,
               "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
               "config": {"SEEDS": SEEDS, "N_TEST": N_TEST, "N_TRAIN": N_TRAIN,
                          "N_VAL": N_VAL, "SPLIT_SEED": SPLIT_SEED, "N_JOBS": N_JOBS},
               "demo": args.demo, "feature_names": feat_names},
              open(os.path.join(OUT, "environment.json"), "w"), indent=2)
    print(f"Done. Results in ./{OUT}/")


if __name__ == "__main__":
    main()
