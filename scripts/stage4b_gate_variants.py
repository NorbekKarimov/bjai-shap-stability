"""
BJAI article - Stage 4b: robustness of the reliability gate to the choice of uncertainty score (RQ5)

Input : bjai_stage2_pack.npz          Output: analysis_stage4b/

Same setting as stage4_gate.py (deployed model = seed 0; explanation risk measured against held-out
retrains; prevalence and uniform weighting; AURC over coverage 0.1-1.0).
Three standard uncertainty scores of selective classification are compared, each alone and each
combined (rank average) with the attribution mass of the explanation of the predicted class:
  maxprob : maximum class probability (softmax response)
  margin  : difference between the two largest class probabilities
  entropy : negative Shannon entropy of the class-probability vector
Paired bootstrap (2,000 resamples of cells) for  AURC(score + mass) - AURC(score).
"""
import os
import numpy as np
import pandas as pd

PACK, OUT = "bjai_stage2_pack.npz", "analysis_stage4b"
MODELS = ["RF", "XGB", "MLP", "SVC"]
HELD = {"RF": [5, 6, 7, 8, 9], "XGB": [5, 6, 7, 8, 9], "MLP": [5, 6, 7, 8, 9], "SVC": [3, 4]}
K, N_BOOT, SEED = 5, 2000, 22
COVERAGES = np.linspace(0.1, 1.0, 91)
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

os.makedirs(OUT, exist_ok=True)
z = np.load(PACK, allow_pickle=False)
y_tree = z["y_explain_tree"].astype(int)
kern_pos = np.searchsorted(z["explain_idx_tree"], z["explain_idx_kernel"])
prev = np.bincount(z["y_test"].astype(int), minlength=7) / len(z["y_test"])


def shap_of(m, s, cls):
    return z[f"shap__{m}__seed{s}__bg0"][np.arange(len(cls)), :, cls]


def jaccard_rows(A, B):
    ta = np.argsort(-np.abs(A), 1)[:, :K]
    tb = np.argsort(-np.abs(B), 1)[:, :K]
    return np.array([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(ta, tb)])


def rank01(x):
    return pd.Series(x).rank(method="average").to_numpy() / len(x)


def aurc(score, risk, w):
    o = np.argsort(-score, kind="stable")
    r, ww = risk[o], w[o]
    cw = np.cumsum(ww)
    curve = np.interp(COVERAGES, cw / cw[-1], np.cumsum(r * ww) / cw)
    return _trapz(curve, COVERAGES) / (COVERAGES[-1] - COVERAGES[0])


rng = np.random.default_rng(SEED)
res, diffs = [], []
for m in MODELS:
    y = y_tree if m in ("RF", "XGB") else y_tree[kern_pos]
    p0 = z[f"proba_explain__{m}__seed0"]
    pred0 = p0.argmax(1)
    ps = np.sort(p0, 1)
    pc = np.clip(p0, 1e-12, 1)
    unc = {"maxprob": ps[:, -1], "margin": ps[:, -1] - ps[:, -2], "entropy": (pc * np.log(pc)).sum(1)}
    mass = rank01(np.abs(shap_of(m, 0, pred0)).sum(1))
    risks = {"pred_risk": (pred0 != y).astype(float),
             "expl_risk": 1 - np.mean([jaccard_rows(shap_of(m, 0, y), shap_of(m, s, y)) for s in HELD[m]], 0)}
    gates = {}
    for u, v in unc.items():
        gates[u] = rank01(v)
        gates[u + "+mass"] = (rank01(v) + mass) / 2
    for weighting in ("prevalence", "uniform"):
        w = prev[y] * 7 if weighting == "prevalence" else np.ones(len(y))
        for rname, risk in risks.items():
            pt = {g: aurc(sc, risk, w) for g, sc in gates.items()}
            for g, a in pt.items():
                res.append({"model": m, "weighting": weighting, "risk": rname, "gate": g, "AURC": a,
                            "random": np.average(risk, weights=w)})
            n = len(y)
            boot = {u: [] for u in unc}
            for _ in range(N_BOOT):
                i = rng.integers(0, n, n)
                for u in unc:
                    boot[u].append(aurc(gates[u + "+mass"][i], risk[i], w[i]) - aurc(gates[u][i], risk[i], w[i]))
            for u, b in boot.items():
                b = np.array(b)
                diffs.append({"model": m, "weighting": weighting, "risk": rname, "uncertainty": u,
                              "delta_AURC_mass": pt[u + "+mass"] - pt[u],
                              "ci_lo": np.percentile(b, 2.5), "ci_hi": np.percentile(b, 97.5)})

pd.DataFrame(res).to_csv(f"{OUT}/rq5b_gate_variants.csv", index=False)
pd.DataFrame(diffs).to_csv(f"{OUT}/rq5b_mass_deltas.csv", index=False)
d = pd.DataFrame(diffs)
print(d.round(4).to_string(index=False))
