"""Pilot for the project: scarcity of the cable class with splits that keep duplicates together.

Usage: python pilot_group_split.py <dataset_TNSM_06995_v2.csv>
Four designs, 30 seeds each:
  random_rows    random stratified 60/20/20 split of the research plan (as in analysis.py)
  group_groups   identical windows kept on one side of the split, scarcity removes whole groups
  group_dedup    as group_groups, but the test set counts each distinct window once
  dedup_all      exact duplicates dropped first, then random stratified 60/20/20 split
Writes pilot_group_split.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

from analysis import CABLE, R, SCARCITY, SEEDS, ci95, cost_matrix, expected_cost, group_split, make_scarce, random_split

OUT = Path(__file__).resolve().parent
DESIGNS = ("random_rows", "group_groups", "group_dedup", "dedup_all")


def scarce_groups(tr, y, groups, level, seed):
    cab = np.unique(groups[tr][y[tr] == CABLE])
    drop = np.random.default_rng(seed).choice(cab, int(round(level * len(cab))), replace=False)
    return tr[~np.isin(groups[tr], drop)]


def one_seed(s, Xn, y, groups, bgroups, C):
    out = {}
    first = np.unique(groups, return_index=True)[1]
    for design in DESIGNS:
        if design == "random_rows":
            tr, te = random_split(y, s)
        elif design == "dedup_all":
            tr, te = (first[i] for i in random_split(y[first], s))
        else:
            tr, te = group_split(y, groups, s)
            if design == "group_dedup":
                te = te[np.isin(te, first)]
        for lvl in SCARCITY:
            sub = scarce_groups(tr, y, groups, lvl, s) if design.startswith("group") else make_scarce(tr, y, lvl, s)
            cab = sub[y[sub] == CABLE]
            row = {"cable_rows": len(cab), "cable_distinct_raw": len(np.unique(groups[cab])),
                   "cable_distinct_binary": len(np.unique(bgroups[cab])), "test_rows": len(te),
                   "test_cable": int((y[te] == CABLE).sum()), "train_rows": len(sub)}
            for arm in ("plain", "weighted"):
                m = RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=1,
                                           class_weight="balanced" if arm == "weighted" else None)
                p = m.fit(Xn[sub], y[sub]).predict(Xn[te])
                row[arm] = (recall_score(y[te], p, labels=[CABLE], average="macro"),
                            f1_score(y[te], p, average="macro"), expected_cost(y[te], p, C))
            out[(design, lvl)] = row
    return out


def main(csv):
    df = pd.read_csv(csv)
    Xraw = df.drop(columns="label")
    y = df["label"].to_numpy()
    X = (Xraw.loc[:, Xraw.nunique() > 1] > 0).astype(int)
    groups = pd.factorize(Xraw.astype(str).agg(",".join, axis=1))[0]
    bgroups = pd.factorize(X.astype(str).agg("".join, axis=1))[0]
    runs = Parallel(n_jobs=-1)(delayed(one_seed)(s, X.to_numpy(), y, groups, bgroups, cost_matrix(R)) for s in SEEDS)

    J = len(SEEDS)
    res = {}
    for design in DESIGNS:
        for lvl in SCARCITY:
            rows = [r[(design, lvl)] for r in runs]
            P = np.array([r["plain"] for r in rows]); W = np.array([r["weighted"] for r in rows])
            rho = np.mean([r["test_rows"] for r in rows]) / np.mean([r["train_rows"] for r in rows])
            d = W[:, 2] - P[:, 2]; sd = d.std(ddof=1)
            t = d.mean() / np.sqrt((1 / J + rho) * sd ** 2)
            k = stats.t.ppf(0.975, J - 1) + stats.t.ppf(0.8, J - 1)
            res[f"{design}@{lvl}"] = {
                **{c: float(np.mean([r[c] for r in rows])) for c in ("cable_rows", "cable_distinct_raw", "cable_distinct_binary", "test_rows", "test_cable")},
                "plain_cost": ci95(P[:, 2]), "weighted_cost": ci95(W[:, 2]),
                "plain_recall": ci95(P[:, 0]), "weighted_recall": ci95(W[:, 0]),
                "plain_f1": ci95(P[:, 1]), "plain_cost_sd": float(P[:, 2].std(ddof=1)),
                "diff": float(d.mean()), "diff_sd": float(sd), "p": float(2 * stats.t.sf(abs(t), J - 1)),
                "mde_30": float(k * sd * np.sqrt(1 / J + rho)),
                "mde_inf": float((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * sd * np.sqrt(rho)),
            }
    (OUT / "pilot_group_split.json").write_text(json.dumps(res, indent=1, default=float))
    for k, v in res.items():
        print(f"{k:22s} test {v['test_rows']:5.0f}/{v['test_cable']:4.1f} rows {v['cable_rows']:6.1f} distinct {v['cable_distinct_raw']:5.1f}/{v['cable_distinct_binary']:5.1f} "
              f"cost {v['plain_cost'][0]:.3f}->{v['weighted_cost'][0]:.3f} sd_plain {v['plain_cost_sd']:.3f} "
              f"diff {v['diff']:+.3f} sd_d {v['diff_sd']:.3f} p {v['p']:.3f} MDE30 {v['mde_30']:.3f} MDEinf {v['mde_inf']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
