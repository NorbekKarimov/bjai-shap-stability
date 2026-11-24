"""
BJAI article - Stage 4: reliability gate for predictions and explanations (RQ5 / G2)

Input : bjai_stage2_pack.npz          Output: analysis_stage4/

Setting: the seed-0 model is the deployed model. For each explained row the gate decides whether to
release (prediction, SHAP explanation) or to abstain. Rows are ranked by a gate score; releasing the
top-c fraction gives a risk-coverage curve.

Risks (evaluated on information the gate never sees):
  pred_risk  : seed-0 prediction is wrong
  expl_risk  : 1 - mean top-5 Jaccard between the seed-0 explanation and explanations of HELD-OUT
               retrains (RF/XGB/MLP: seeds 5-9; SVC: seeds 3-4), observed-class attributions, bg0
Gate scores (higher = more reliable; combinations are rank averages, so nothing is fitted):
  G_unc        : 1 - max class probability of the seed-0 model           (cheap baseline)
  G_unc_mass   : G_unc + attribution mass sum|phi| of the seed-0 explanation of the PREDICTED class
                 (cheap: one model, one explanation; the label is never used)
  G_unc_stab   : G_unc + agreement of the seed-0 explanation with REFERENCE retrains
                 (RF/XGB/MLP: seeds 1-4; SVC: seeds 1-2) -> expensive upper reference
  G_mass, G_stab alone are reported for completeness.
Evaluation: AURC (area under the risk-coverage curve; lower = better) and risk at 80 % coverage.
  Rows are weighted by test-set prevalence / (1/7) so that the class-balanced explanation set
  represents the natural class mix; unweighted results are also saved.
  95 % CIs by row bootstrap; paired bootstrap for differences between gates.

Note: the expl_risk target uses observed-class attributions (as in Stage 3), the cheap gate uses the
predicted class so that it can be computed at deployment time without labels.
"""
import itertools
import os

import numpy as np
import pandas as pd

PACK, OUT = "bjai_stage2_pack.npz", "analysis_stage4"
MODELS = ["RF", "XGB", "MLP", "SVC"]
SPLIT = {"RF": ([1, 2, 3, 4], [5, 6, 7, 8, 9]), "XGB": ([1, 2, 3, 4], [5, 6, 7, 8, 9]),
         "MLP": ([1, 2, 3, 4], [5, 6, 7, 8, 9]), "SVC": ([1, 2], [3, 4])}
K, N_BOOT, SEED = 5, 2000, 21
COVERAGES = np.linspace(0.1, 1.0, 91)

os.makedirs(OUT, exist_ok=True)
z = np.load(PACK, allow_pickle=False)
y_tree = z["y_explain_tree"].astype(int)
kern_pos = np.searchsorted(z["explain_idx_tree"], z["explain_idx_kernel"])
prev = np.bincount(z["y_test"].astype(int), minlength=7) / len(z["y_test"])


def shap_of(m, s, cls):
    v = z[f"shap__{m}__seed{s}__bg0"]
    return v[np.arange(len(cls)), :, cls]


def jaccard_rows(A, B):
    ta = np.argsort(-np.abs(A), 1)[:, :K]
    tb = np.argsort(-np.abs(B), 1)[:, :K]
    return np.array([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(ta, tb)])


def rank01(x):
    return pd.Series(x).rank(method="average").to_numpy() / len(x)


def risk_coverage(score, risk, w):
    order = np.argsort(-score, kind="stable")               # most reliable first
    r, ww = risk[order], w[order]
    cw = np.cumsum(ww)
    cum = np.cumsum(r * ww) / cw
    cov = cw / cw[-1]
    curve = np.interp(COVERAGES, cov, cum)
    return curve, _trapz(curve, COVERAGES) / (COVERAGES[-1] - COVERAGES[0]), np.interp(0.8, cov, cum)


_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
rng = np.random.default_rng(SEED)
res, diffs, curves = [], [], []
for m in MODELS:
    y = y_tree if m in ("RF", "XGB") else y_tree[kern_pos]
    p0 = z[f"proba_explain__{m}__seed0"]
    pred0 = p0.argmax(1)
    ref, held = SPLIT[m]
    phi0_obs = shap_of(m, 0, y)
    pred_risk = (pred0 != y).astype(float)
    expl_risk = 1 - np.mean([jaccard_rows(phi0_obs, shap_of(m, s, y)) for s in held], 0)

    unc = -(1 - p0.max(1))                                       # higher = more reliable
    mass = np.abs(shap_of(m, 0, pred0)).sum(1)
    stab = np.mean([jaccard_rows(shap_of(m, 0, pred0), shap_of(m, s, pred0)) for s in ref], 0)
    gates = {"G_unc": rank01(unc),
             "G_mass": rank01(mass),
             "G_stab": rank01(stab),
             "G_unc_mass": (rank01(unc) + rank01(mass)) / 2,
             "G_unc_stab": (rank01(unc) + rank01(stab)) / 2,
             "random": np.zeros(len(y))}

    for weighting in ("prevalence", "uniform"):
        w = prev[y] * 7 if weighting == "prevalence" else np.ones(len(y))
        for rname, risk in (("pred_risk", pred_risk), ("expl_risk", expl_risk)):
            point = {}
            for g, sc in gates.items():
                if g == "random":                                # expected value of random order
                    aurc = r80 = np.average(risk, weights=w)
                    curve = np.full(len(COVERAGES), aurc)
                else:
                    curve, aurc, r80 = risk_coverage(sc, risk, w)
                point[g] = aurc
                if weighting == "prevalence":
                    curves.append(pd.DataFrame({"model": m, "risk": rname, "gate": g,
                                                "coverage": COVERAGES, "risk_value": curve}))
                res.append({"model": m, "weighting": weighting, "risk": rname, "gate": g,
                            "AURC": aurc, "risk_at_80": r80, "full_coverage_risk": np.average(risk, weights=w)})
            # paired bootstrap: AURC differences against the uncertainty-only baseline
            n = len(y)
            boot = {g: [] for g in ("G_unc_mass", "G_unc_stab", "G_mass")}
            for _ in range(N_BOOT):
                i = rng.integers(0, n, n)
                base = risk_coverage(gates["G_unc"][i], risk[i], w[i])[1]
                for g in boot:
                    boot[g].append(risk_coverage(gates[g][i], risk[i], w[i])[1] - base)
            for g, b in boot.items():
                b = np.array(b)
                diffs.append({"model": m, "weighting": weighting, "risk": rname,
                              "gate_minus_G_unc": g, "delta_AURC": point[g] - point["G_unc"],
                              "ci_lo": np.percentile(b, 2.5), "ci_hi": np.percentile(b, 97.5),
                              "share_boot_better": float(np.mean(b < 0))})

pd.DataFrame(res).to_csv(f"{OUT}/rq5_gate_results.csv", index=False)
pd.DataFrame(diffs).to_csv(f"{OUT}/rq5_gate_differences.csv", index=False)
pd.concat(curves).to_csv(f"{OUT}/rq5_risk_coverage_curves.csv", index=False)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cv = pd.concat(curves)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
    for j, m in enumerate(MODELS):
        for i, rname in enumerate(["pred_risk", "expl_risk"]):
            ax = axes[i, j]
            for g, st in [("random", ":"), ("G_unc", "-"), ("G_unc_mass", "-"), ("G_unc_stab", "--")]:
                d = cv[(cv.model == m) & (cv.risk == rname) & (cv.gate == g)]
                ax.plot(d.coverage, d.risk_value, st, label=g)
            ax.set_title(f"{m} - {'prediction error' if i == 0 else 'explanation instability'}", fontsize=9)
            ax.grid(alpha=.3)
            if j == 0:
                ax.set_ylabel("risk among released rows")
            if i == 1:
                ax.set_xlabel("coverage")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Risk-coverage curves of the reliability gates (prevalence-weighted; lower is better)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_rq5_risk_coverage.png", dpi=200)
except ImportError:
    pass

r = pd.DataFrame(res)
print(r[r.weighting == "prevalence"].pivot_table(index=["risk", "model"], columns="gate", values="AURC").round(4))
d = pd.DataFrame(diffs)
print(d[d.weighting == "prevalence"].round(4).to_string(index=False))
