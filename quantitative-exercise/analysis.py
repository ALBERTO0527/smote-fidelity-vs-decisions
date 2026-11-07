"""II2202 quantitative exercise: EDA, model and "how much data" on the Di Cicco et al. (2024) dataset.

Usage: python analysis.py <dataset_TNSM_06995_v2.csv>
Writes figures/*.pdf and results.json next to this file. Fixed seeds, reproducible.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # fixed PDF timestamp, so reruns give byte-identical figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
NAMES = ["IDU", "ODU", "Cable", "Power"]
CABLE = 2
SEEDS = range(30)
FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SCARCITY = [0.0, 0.5, 0.75, 0.9]
R = 5  # cost ratio of the research plan base case
# Okabe-Ito, plus a distinct marker and line style per series so the figures survive grayscale
COL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
MARK = ["o", "s", "^", "D"]
LS = ["-", "--", ":", "-."]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})


def cost_matrix(r):
    group = np.array([0, 1, 1, 0])  # IDU and power indoor, ODU and cable outdoor (plan base case)
    C = np.where(group[:, None] == group[None, :], 1.0, r)
    np.fill_diagonal(C, 0)
    return C


def expected_cost(y, p, C):
    return (confusion_matrix(y, p, labels=range(4)) * C).sum() / len(y)


def ci95(a):
    a = np.asarray(a)
    return a.mean(), stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))


def rf(seed, weighted=False):
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1,
                                  class_weight="balanced" if weighted else None)


def random_split(y, seed):
    """60/20/20 stratified, as in the research plan; validation unused here."""
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=0.6, stratify=y, random_state=seed)
    _, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    return tr, te


def group_split(y, groups, seed):
    """Same proportions, but identical windows never fall on both sides."""
    idx = np.arange(len(y))
    rest, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(idx, y, groups))
    tr_rel, _ = next(StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(rest, y[rest], groups[rest]))
    return rest[tr_rel], te


def subsample(tr, y, frac, seed):
    if frac == 1.0:
        return tr
    keep, _ = train_test_split(tr, train_size=frac, stratify=y[tr], random_state=seed)
    return keep


def make_scarce(tr, y, level, seed):
    cab = tr[y[tr] == CABLE]
    drop = np.random.default_rng(seed).choice(cab, int(round(level * len(cab))), replace=False)
    return np.setdiff1d(tr, drop)


def main(csv):
    df = pd.read_csv(csv)
    Xraw = df.drop(columns="label")
    y = df["label"].to_numpy()
    res = {}

    # ---------- EDA ----------
    const = Xraw.columns[Xraw.nunique() == 1]
    X = (Xraw.drop(columns=const) > 0).astype(int)  # Mode-1 binary: alarm on for >= 1 s
    nz = Xraw.to_numpy()[Xraw.to_numpy() > 0]
    groups = pd.factorize(Xraw.astype(str).agg(",".join, axis=1))[0]
    bin_groups = pd.factorize(X.astype(str).agg("".join, axis=1))[0]
    conflict = pd.Series(y).groupby(bin_groups).nunique()
    res["eda"] = {
        "rows": len(df), "features": Xraw.shape[1], "constant": len(const),
        "class_counts": dict(zip(NAMES, np.bincount(y).tolist())),
        "zero_fraction": float((Xraw.to_numpy() == 0).mean()),
        "nonzero_median": float(np.median(nz)), "nonzero_at_900": float((nz == 900).mean()),
        "nonzero_le_60": float((nz <= 60).mean()),
        "raw_duplicate_rows": int(df.duplicated().sum()), "raw_unique_windows": int(groups.max() + 1),
        "binary_unique_patterns": int(bin_groups.max() + 1),
        "binary_patterns_with_conflicting_labels": int((conflict > 1).sum()),
        "rows_in_conflicting_patterns": int(pd.Series(bin_groups).isin(conflict[conflict > 1].index).sum()),
    }

    active = X.sum(1)
    res["eda"]["active_per_window"] = {NAMES[c]: [float(np.median(active[y == c])), float(active[y == c].mean())] for c in range(4)}
    kw = stats.kruskal(*[active[y == c] for c in range(4)])
    res["eda"]["kruskal_active"] = {"H": float(kw.statistic), "p": float(kw.pvalue)}

    # association of each alarm with the class: chi-square on 2x4 tables, Cramer's V, Holm
    rows = []
    for col in X.columns:
        tab = pd.crosstab(X[col], y)
        chi2, p, *_ = stats.chi2_contingency(tab)
        rows.append((col, chi2, p, np.sqrt(chi2 / len(y)), *[X[col][y == c].mean() for c in range(4)]))
    assoc = pd.DataFrame(rows, columns=["alarm", "chi2", "p", "V", *NAMES]).sort_values("V", ascending=False)
    m = len(assoc)
    holm = np.maximum.accumulate((m - np.arange(m)) * np.sort(assoc.p.to_numpy())).clip(max=1)
    res["eda"]["assoc_tests"] = m
    res["eda"]["assoc_significant_holm"] = int((holm < 0.05).sum())
    res["eda"]["assoc_V_ge_0.3"] = int((assoc.V >= 0.3).sum())
    res["eda"]["top_alarms"] = assoc.head(8).round(3).to_dict("records")

    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    res["eda"]["pca"] = {"pc1": float(pca.explained_variance_ratio_[0]), "pc2": float(pca.explained_variance_ratio_[1]),
                         "k80": int(np.searchsorted(cum, 0.8) + 1), "k95": int(np.searchsorted(cum, 0.95) + 1)}

    # Figure 1: classes and sparsity
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4), gridspec_kw={"wspace": 0.45})
    ax[0].bar(NAMES, np.bincount(y), color="0.55", edgecolor="black", hatch=["", "//", "xx", ".."])
    for i, n in enumerate(np.bincount(y)):
        ax[0].text(i, n + 10, str(n), ha="center")
    ax[0].set_ylabel("Windows"); ax[0].set_title("(a) Windows per root cause", loc="left")
    act = np.sort((Xraw > 0).mean().to_numpy())[::-1]
    ax[1].plot(np.arange(1, len(act) + 1), np.where(act > 0, act, np.nan), color="black", marker=".", ms=3, lw=0.8)
    ax[1].set_yscale("log"); ax[1].axvline(len(act) - len(const) + 0.5, color="0.5", ls="--", lw=0.8)
    ax[1].text(len(act) - len(const) + 3, 0.3, f"{len(const)} alarms\nnever active", fontsize=8)
    ax[1].set_xlabel("Alarm, ranked by how often it is active"); ax[1].set_ylabel("Share of windows (log)")
    ax[1].set_title("(b) Alarm activity", loc="left")
    fig.savefig(FIG / "fig1_classes_sparsity.pdf"); plt.close(fig)

    # Figure 2: durations and active alarms per window
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
    ax[0].hist(nz, bins=np.arange(0, 931, 30), color="0.55", edgecolor="black")
    ax[0].set_yscale("log"); ax[0].set_xlabel("Seconds active in the 15-min window (non-zero only)")
    ax[0].set_ylabel("Counter values (log)"); ax[0].set_title("(a) Non-zero durations", loc="left")
    bp = ax[1].boxplot([active[y == c] for c in range(4)], tick_labels=NAMES, patch_artist=True,
                       medianprops={"color": "black"}, flierprops={"marker": ".", "ms": 3})
    for b, h in zip(bp["boxes"], ["", "//", "xx", ".."]):
        b.set(facecolor="0.85", hatch=h)
    ax[1].set_ylabel("Active alarms per window"); ax[1].set_title("(b) Alarms per window", loc="left")
    fig.savefig(FIG / "fig2_durations_active.pdf"); plt.close(fig)

    # Figure 3: PCA
    Z = pca.transform(X)[:, :2]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax[0].plot(np.arange(1, len(cum) + 1), cum, color="black", lw=1)
    for t in (0.8, 0.95):
        ax[0].axhline(t, color="0.5", ls=":", lw=0.8)
    ax[0].set_xlabel("Number of components"); ax[0].set_ylabel("Cumulative explained variance")
    ax[0].set_title("(a) PCA on binary alarms", loc="left")
    jit = np.random.default_rng(0).normal(0, 0.04, Z.shape)
    for c in range(4):
        ax[1].scatter(*(Z[y == c] + jit[y == c]).T, s=9, marker=MARK[c], facecolors="none", edgecolors=COL[c], lw=0.7, label=NAMES[c])
    ax[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})"); ax[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1, 1)); ax[1].set_title("(b) Windows on PC1-PC2", loc="left")
    fig.savefig(FIG / "fig3_pca.pdf"); plt.close(fig)

    # ---------- model: learning curves, random vs duplicate-aware split ----------
    Xn = X.to_numpy()
    C = cost_matrix(R)
    lc = {"random": {f: [] for f in FRACS}, "group": {f: [] for f in FRACS}}
    leak = []
    for s in SEEDS:
        for kind, (tr, te) in {"random": random_split(y, s), "group": group_split(y, groups, s)}.items():
            if kind == "random":
                leak.append(np.isin(groups[te], groups[tr]).mean())
            for f in FRACS:
                sub = subsample(tr, y, f, s)
                p = rf(s).fit(Xn[sub], y[sub]).predict(Xn[te])
                lc[kind][f].append(f1_score(y[te], p, average="macro"))
    res["leak_test_rows_with_copy_in_train"] = float(np.mean(leak))
    res["learning_curve"] = {k: {str(f): ci95(v) for f, v in d.items()} for k, d in lc.items()}

    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    n_tr = [round(f * 0.6 * len(y)) for f in FRACS]
    for i, (k, lab) in enumerate([("random", "Random split"), ("group", "Duplicate-aware split")]):
        m_, h_ = np.array([ci95(lc[k][f]) for f in FRACS]).T
        ax.errorbar(n_tr, m_, yerr=h_, color=COL[i], marker=MARK[i], ls=LS[i], ms=4, capsize=2, lw=1, label=lab)
    ax.set_xlabel("Training windows"); ax.set_ylabel("Macro-F1 on real test set")
    ax.legend(frameon=False); ax.grid(alpha=0.3, lw=0.5)
    fig.savefig(FIG / "fig4_learning_curve.pdf"); plt.close(fig)

    # ---------- scarcity of the cable class: plain vs class-weighted RF ----------
    sc = {lvl: {"plain": [], "weighted": []} for lvl in SCARCITY}
    for s in SEEDS:
        tr, te = random_split(y, s)
        for lvl in SCARCITY:
            sub = make_scarce(tr, y, lvl, s)
            for arm in ("plain", "weighted"):
                p = rf(s, arm == "weighted").fit(Xn[sub], y[sub]).predict(Xn[te])
                sc[lvl][arm].append((recall_score(y[te], p, labels=[CABLE], average="macro"),
                                     f1_score(y[te], p, average="macro"), expected_cost(y[te], p, C)))
    rho = 0.2 / 0.6  # n_test / n_train, Nadeau-Bengio correction for overlapping training sets
    J = len(SEEDS)
    res["scarcity"] = {}
    for lvl in SCARCITY:
        P, W = np.array(sc[lvl]["plain"]), np.array(sc[lvl]["weighted"])
        d = W[:, 2] - P[:, 2]
        sd = d.std(ddof=1)
        t = d.mean() / np.sqrt((1 / J + rho) * sd ** 2)
        k = stats.t.ppf(0.975, J - 1) + stats.t.ppf(0.8, J - 1)
        res["scarcity"][str(lvl)] = {
            "cable_train_rows": int(round((1 - lvl) * (y == CABLE).sum() * 0.6)),
            "plain": {"cable_recall": ci95(P[:, 0]), "macro_f1": ci95(P[:, 1]), "cost": ci95(P[:, 2])},
            "weighted": {"cable_recall": ci95(W[:, 0]), "macro_f1": ci95(W[:, 1]), "cost": ci95(W[:, 2])},
            "cost_diff_mean": float(d.mean()), "cost_diff_sd": float(sd),
            "corrected_t": float(t), "corrected_p": float(2 * stats.t.sf(abs(t), J - 1)),
            "mde_30_seeds": float(k * sd * np.sqrt(1 / J + rho)),
            "mde_infinite_seeds": float(stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * sd * np.sqrt(rho),
        }

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"wspace": 0.35})
    x = [round((1 - l) * (y == CABLE).sum() * 0.6) for l in SCARCITY]
    for i, arm in enumerate(("plain", "weighted")):
        for j, (metric, lab) in enumerate([(0, "Cable recall"), (2, f"Expected cost (r = {R})")]):
            m_, h_ = np.array([ci95(np.array(sc[l][arm])[:, metric]) for l in SCARCITY]).T
            ax[j].errorbar(np.array(x) + (i - 0.5) * 2.5, m_, yerr=h_, color=COL[i], marker=MARK[i], ls=LS[i], ms=4, capsize=2, lw=1,
                           label="Random Forest" if arm == "plain" else "Random Forest, class-weighted")
            ax[j].set_ylabel(lab); ax[j].set_xlabel("Cable windows left in training")
            ax[j].grid(alpha=0.3, lw=0.5)
    ax[0].set_title("(a) Rare-class recall", loc="left"); ax[1].set_title("(b) Decision cost", loc="left")
    ax[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(FIG / "fig5_scarcity.pdf"); plt.close(fig)

    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    # self-check: wrong-group errors cost r, same-group errors 1, correct 0
    C = cost_matrix(5)
    assert C[0, 0] == 0 and C[0, 3] == 1 and C[0, 2] == 5 and C[1, 2] == 1 and C[2, 3] == 5
    assert expected_cost([0, 2], [2, 2], C) == 2.5
    main(sys.argv[1])
