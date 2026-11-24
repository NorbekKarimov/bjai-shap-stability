"""
BJAI article - Stage 3: SHAP analysis for RQ3 (feature influence) and RQ4 / G1 (explanation stability)

Input : bjai_stage2_pack.npz  (from stage2_pack.py)
Output: folder analysis_stage3/  (CSV tables + PNG figures + summary.txt)

Design (stated here so it can be reported verbatim in the Methods section):
  Explained output : for every row, the attribution vector of its OBSERVED class
                     (phi_i = SHAP values of output y_i). Sensitivity check: the model's own predicted class.
  Local stability  : for each row, pairwise comparison of attribution vectors
                     - model-induced     : between models of different training seeds (background bg0)
                     - explainer-induced : between backgrounds bg0, bg1, bg2 (seed-0 model)
                     metrics (|phi| ranking unless stated):
                       top5_jaccard  |T5(a) & T5(b)| / |T5(a) | T5(b)|
                       rbo10         rank-biased overlap, depth 10, p = 0.9, normalised by its maximum
                       top1_agree    same most-important feature
                       sign5         share of a's top-5 features whose SHAP sign is the same in b
                     row score = mean over all pairs; class score = mean of row scores (+ bootstrap 95% CI).
  Class effect     : Kruskal-Wallis test of row scores across the 7 classes, per model and source;
                     epsilon-squared effect size.
  Confounder check : row scores split by whether the row is correctly classified by the majority of seeds.
  Sanity           : local accuracy (base + sum of SHAP = model output) checked for every run.

Rank metrics are unit-free, so RF/SVC/MLP (probability units) and XGB (margin units) are comparable.
Cross-model comparisons use the common 140-row kernel subset (20 rows per class).
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

PACK = "bjai_stage2_pack.npz"
OUT = "analysis_stage3"
CLASS_NAMES = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow",
               "Aspen", "Douglas-fir", "Krummholz"]
MODELS = ["RF", "XGB", "MLP", "SVC"]
K, RBO_DEPTH, RBO_P = 5, 10, 0.9
N_BOOT, BOOT_SEED = 2000, 11

os.makedirs(OUT, exist_ok=True)
z = np.load(PACK, allow_pickle=False)
fn = list(z["feature_names"])
y_tree = z["y_explain_tree"].astype(int)
ex_tree, ex_kern = z["explain_idx_tree"], z["explain_idx_kernel"]
kern_pos = np.searchsorted(ex_tree, ex_kern)              # kernel rows inside the tree set
y_kern = y_tree[kern_pos]
y_test = z["y_test"].astype(int)
prev = np.bincount(y_test, minlength=7) / len(y_test)      # class prevalence in the test set
keys = list(z.keys())


def runs(m):
    out = {}
    for k in keys:
        if k.startswith(f"shap__{m}__"):
            _, _, s, b = k.split("__")
            out[(int(s[4:]), int(b[2:]))] = z[k]
    return out


def softmax(a):
    e = np.exp(a - a.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# ---------------- sanity: local accuracy ----------------
# RF/MLP/SVC: base + sum(SHAP) must equal the predicted probability.
# XGB (margin units): XGBoost >= 2 stores a per-class intercept (log class prior) that shap's
# expected_value omits. That intercept is constant per class, so it cancels in every SHAP value;
# the check is therefore that (base + sum SHAP - log p) differs between classes only by a
# row-independent constant: we report the largest across-row SD of that difference.
san = []
for m in MODELS:
    for (s, b), v in runs(m).items():
        recon = v.sum(1).astype(np.float64) + z[f"base__{m}__seed{s}__bg{b}"][None, :]
        proba = z[f"proba_explain__{m}__seed{s}"].astype(np.float64)
        if m == "XGB":
            D = recon - np.log(np.clip(proba, 1e-30, 1))
            err = 0.0
            for k in range(1, recon.shape[1]):
                ok = (proba[:, k] > 1e-6) & (proba[:, 0] > 1e-6)
                if ok.sum() > 2:
                    err = max(err, float(np.std(D[ok, k] - D[ok, 0])))
        else:
            err = float(np.abs(recon - proba).max())
        san.append({"model": m, "seed": s, "bg": b, "max_abs_err": err})
pd.DataFrame(san).to_csv(f"{OUT}/sanity_local_accuracy.csv", index=False)


# ---------------- metrics ----------------
def topk(v, k):
    return set(np.argsort(-np.abs(v))[:k])


def rbo(a, b, depth=RBO_DEPTH, p=RBO_P):
    ra, rb_ = np.argsort(-np.abs(a)), np.argsort(-np.abs(b))
    s = sum(p ** (d - 1) * len(set(ra[:d]) & set(rb_[:d])) / d for d in range(1, depth + 1))
    return s / sum(p ** (d - 1) for d in range(1, depth + 1))


def pair_scores(a, b):
    ta, tb = topk(a, K), topk(b, K)
    top_a = np.argsort(-np.abs(a))[:K]
    return {"top5_jaccard": len(ta & tb) / len(ta | tb),
            "rbo10": rbo(a, b),
            "top1_agree": float(np.argmax(np.abs(a)) == np.argmax(np.abs(b))),
            "sign5": float(np.mean(np.sign(a[top_a]) == np.sign(b[top_a])))}


METRICS = ["top5_jaccard", "rbo10", "top1_agree", "sign5"]


def row_stability(vecs):
    """vecs: list of (rows, features) arrays -> DataFrame of per-row mean pairwise scores."""
    n = vecs[0].shape[0]
    acc = {m: np.zeros(n) for m in METRICS}
    pairs = list(itertools.combinations(range(len(vecs)), 2))
    for i in range(n):
        sc = [pair_scores(vecs[a][i], vecs[b][i]) for a, b in pairs]
        for m in METRICS:
            acc[m][i] = np.mean([s[m] for s in sc])
    return pd.DataFrame(acc)


def own_class(v, cls):
    return v[np.arange(len(cls)), :, cls]                  # (rows, features)


rows_all, glob_rows = [], []
for m in MODELS:
    R = runs(m)
    seeds = sorted(s for (s, b) in R if b == 0)
    y_m = y_tree if m in ("RF", "XGB") else y_kern
    pos_m = np.arange(len(y_tree)) if m in ("RF", "XGB") else kern_pos
    probas = np.stack([z[f"proba_explain__{m}__seed{s}"] for s in seeds])   # (S, rows, 7)
    majority_correct = (probas.argmax(2) == y_m[None]).mean(0) > 0.5

    for target in ("observed", "predicted"):
        def vec(s, b):
            cls = y_m if target == "observed" else z[f"proba_explain__{m}__seed{s}"].argmax(1)
            return own_class(R[(s, b)], cls)
        for source, combos in (("model", [(s, 0) for s in seeds]),
                               ("explainer", [(seeds[0], b) for b in (0, 1, 2)])):
            df = row_stability([vec(s, b) for s, b in combos])
            df["model"], df["target"], df["source"] = m, target, source
            df["cls"], df["row_pos"], df["correct"] = y_m, pos_m, majority_correct
            rows_all.append(df)

    # global importance (RQ3): mean |phi_observed| per seed
    for s in seeds:
        g = np.abs(own_class(R[(s, 0)], y_m)).mean(0)
        glob_rows.append(pd.DataFrame({"model": m, "seed": s, "feature": fn, "mean_abs_shap": g}))

rows = pd.concat(rows_all, ignore_index=True)
rows.to_csv(f"{OUT}/row_stability.csv", index=False)
glob = pd.concat(glob_rows, ignore_index=True)

# ---------------- RQ3: global importance ----------------
gi = (glob.groupby(["model", "feature"]).mean_abs_shap.agg(["mean", "min", "max"]).reset_index())
gi["rank"] = gi.groupby("model")["mean"].rank(ascending=False)
gi.sort_values(["model", "rank"]).to_csv(f"{OUT}/rq3_global_importance.csv", index=False)

# ---------------- RQ4 / G1: per-class stability ----------------
rng = np.random.default_rng(BOOT_SEED)


def boot_ci(x):
    b = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


cls_tab, kw_tab, conf_tab = [], [], []
for (m, t, src), d in rows.groupby(["model", "target", "source"]):
    for metric in METRICS:
        groups = [d.loc[d.cls == c, metric].to_numpy() for c in range(7)]
        for c, g in enumerate(groups):
            lo, hi = boot_ci(g)
            cls_tab.append({"model": m, "target": t, "source": src, "metric": metric,
                            "class": c, "class_name": CLASS_NAMES[c], "prevalence": prev[c],
                            "n": len(g), "mean": g.mean(), "ci_lo": lo, "ci_hi": hi})
        if all(np.ptp(g) == 0 for g in groups) and len({g[0] for g in groups}) == 1:
            H, p = 0.0, 1.0
        else:
            H, p = kruskal(*groups)
        n = sum(len(g) for g in groups)
        kw_tab.append({"model": m, "target": t, "source": src, "metric": metric,
                       "overall_mean": d[metric].mean(), "kruskal_H": H, "p": p,
                       "epsilon_sq": H / (n - 1), "n_rows": n})
        c1, c0 = d.loc[d.correct, metric], d.loc[~d.correct, metric]
        u_p = mannwhitneyu(c1, c0).pvalue if len(c0) > 0 and len(c1) > 0 else np.nan
        conf_tab.append({"model": m, "target": t, "source": src, "metric": metric,
                         "mean_correct": c1.mean(), "n_correct": len(c1),
                         "mean_incorrect": c0.mean() if len(c0) else np.nan, "n_incorrect": len(c0),
                         "mannwhitney_p": u_p})
pd.DataFrame(cls_tab).to_csv(f"{OUT}/rq4_class_stability.csv", index=False)
kw = pd.DataFrame(kw_tab)
kw.to_csv(f"{OUT}/rq4_class_effect_tests.csv", index=False)
pd.DataFrame(conf_tab).to_csv(f"{OUT}/rq4_correct_vs_incorrect.csv", index=False)

# ---------------- mediator check: attribution mass ----------------
# mass_i = sum_j |phi_ij| of the observed class (seed-0 model, bg0), i.e. how much attribution
# the row receives. OLS with HC3 standard errors (numpy implementation, no extra packages):
#   stability ~ majority + correct            (model A)
#   stability ~ majority + correct + log(mass) (model B)
from scipy.stats import spearmanr, t as t_dist


def ols_hc3(Xm, yv):
    Xm = np.column_stack([np.ones(len(yv)), Xm])
    XtX_inv = np.linalg.inv(Xm.T @ Xm)
    beta = XtX_inv @ Xm.T @ yv
    e = yv - Xm @ beta
    h = np.einsum("ij,jk,ik->i", Xm, XtX_inv, Xm)
    cov = XtX_inv @ (Xm.T * (e / (1 - h)) ** 2) @ Xm @ XtX_inv
    se = np.sqrt(np.diag(cov))
    p = 2 * t_dist.sf(np.abs(beta / se), len(yv) - Xm.shape[1])
    r2 = 1 - (e @ e) / ((yv - yv.mean()) @ (yv - yv.mean()))
    return beta, p, r2


med = []
for m in MODELS:
    y_m = y_tree if m in ("RF", "XGB") else y_kern
    phi = own_class(runs(m)[(0, 0)], y_m)
    logmass = np.log(np.abs(phi).sum(1))
    for src in ("model", "explainer"):
        d = rows[(rows.model == m) & (rows.target == "observed") & (rows.source == src)]
        yv = d["top5_jaccard"].to_numpy()
        maj = d.cls.isin([0, 1]).to_numpy(float)
        cor = d.correct.to_numpy(float)
        bA, pA, r2A = ols_hc3(np.column_stack([maj, cor]), yv)
        bB, pB, r2B = ols_hc3(np.column_stack([maj, cor, logmass]), yv)
        med.append({"model": m, "source": src, "spearman_logmass_stability": spearmanr(logmass, yv)[0],
                    "b_majority_A": bA[1], "p_majority_A": pA[1], "R2_A": r2A,
                    "b_majority_B": bB[1], "p_majority_B": pB[1],
                    "b_logmass_B": bB[3], "p_logmass_B": pB[3], "R2_B": r2B})
pd.DataFrame(med).to_csv(f"{OUT}/rq4_mass_mediation.csv", index=False)

# cross-model comparison on the common 140-row subset (model-induced, observed class)
sub = rows[(rows.target == "observed") & (rows.source == "model") & rows.row_pos.isin(kern_pos)]
sub.groupby("model")[METRICS].mean().to_csv(f"{OUT}/rq4_common_subset_by_model.csv")

# ---------------- figures ----------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ct = pd.DataFrame(cls_tab)
    order = np.argsort(prev)                                   # rarest class first
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, src in zip(axes, ["model", "explainer"]):
        for i, m in enumerate(MODELS):
            d = ct[(ct.model == m) & (ct.source == src) & (ct.target == "observed")
                   & (ct.metric == "top5_jaccard")].set_index("class").loc[order]
            x = np.arange(7) + (i - 1.5) * 0.18
            ax.errorbar(x, d["mean"], yerr=[d["mean"] - d.ci_lo, d.ci_hi - d["mean"]],
                        fmt="o", ms=4, capsize=2, label=m)
        ax.set_xticks(range(7))
        ax.set_xticklabels([f"{CLASS_NAMES[c]}\n({prev[c] * 100:.1f}%)" for c in order],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{src}-induced variation")
        ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("Top-5 Jaccard (mean, 95% CI)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Local SHAP stability by class (classes ordered by test-set prevalence)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_rq4_class_stability.png", dpi=200)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, m in zip(axes, MODELS):
        d = gi[gi.model == m].sort_values("mean", ascending=False).head(10)[::-1]
        ax.barh(d.feature, d["mean"], xerr=[d["mean"] - d["min"], d["max"] - d["mean"]], capsize=2)
        ax.set_title(m + (" (margin units)" if m == "XGB" else " (probability units)"), fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Global importance: mean |SHAP| of the observed class (bar = mean over seeds, whisker = min-max)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_rq3_global_importance.png", dpi=200)
except ImportError:
    print("matplotlib not installed: figures skipped")

# ---------------- summary ----------------
with open(f"{OUT}/summary.txt", "w", encoding="utf-8") as f:
    f.write("Sanity (max |base + sum SHAP - model output|) per model:\n")
    f.write(pd.DataFrame(san).groupby("model").max_abs_err.max().to_string() + "\n\n")
    f.write("Class effect tests (observed class):\n")
    f.write(kw[kw.target == "observed"].round(4).to_string(index=False) + "\n")
print(open(f"{OUT}/summary.txt", encoding="utf-8").read())
