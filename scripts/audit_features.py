"""Empirical feature-stack audit for ALTUS Layer 1.

Computes per-feature mutual information against triple-barrier labels, redundancy
(correlation), variance, per-TF informativeness, per-family scores, and a
'drop bottom-N features' stability experiment.

Usage:
    python3 scripts/audit_features.py [--start 2024-01-01] [--end 2025-01-01]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from altus.data import load_mnq  # noqa: E402
from altus.features.pipeline import build_features, FeatureSpec  # noqa: E402
from altus.features.structural import StructuralSpec, _FAMILY_REGISTRY  # noqa: E402
from altus.labels.triple_barrier import triple_barrier_labels  # noqa: E402


# Mapping each feature to its family/source ----------------------------------
def feature_to_family(col: str) -> str:
    """Map a feature column name to its source family."""
    # Multi-TF price-action features: tf{N}_*
    if col.startswith("tf"):
        # extract the timeframe number
        parts = col.split("_", 1)
        return f"price_tf{parts[0][2:]}"
    # Structural families — match by known column prefixes from each family
    for fam_name, mod in _FAMILY_REGISTRY.items():
        cols = getattr(mod, "FEATURE_COLUMNS", ())
        if col in cols:
            return fam_name
    # simmtm uses programmatic names
    if col.startswith("simmtm_"):
        return "simmtm"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-06-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--families", default="all_no_heavy",
                    help="'all' | 'all_no_heavy' | comma list. all_no_heavy = all except kronos+simmtm.")
    ap.add_argument("--out", default=str(REPO / "artifacts" / "feature_audit"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[audit] loading MNQ {args.start} → {args.end}", flush=True)
    df = load_mnq(args.start, args.end)
    print(f"[audit]   {len(df):,} 1m bars loaded", flush=True)

    # Decide which structural families to enable
    all_fams = set(_FAMILY_REGISTRY.keys())
    if args.families == "all":
        enabled = all_fams
    elif args.families == "all_no_heavy":
        # Skip kronos (deferred, no cache) — keep simmtm if cache exists
        enabled = all_fams - {"kronos"}
        if not (REPO / "artifacts" / "simmtm_embeddings.parquet").exists():
            enabled -= {"simmtm"}
    else:
        enabled = set(args.families.split(","))
    spec_struct = StructuralSpec(enabled=frozenset(enabled))
    spec_price = FeatureSpec()
    print(f"[audit] structural families: {sorted(enabled)}", flush=True)

    print(f"[audit] building features...", flush=True)
    tA = time.time()
    X = build_features(df, spec=spec_price, structural_spec=spec_struct)
    print(f"[audit]   features: {X.shape} in {time.time()-tA:.1f}s", flush=True)

    print(f"[audit] computing labels...", flush=True)
    lab = triple_barrier_labels(df)
    y_long = pd.Series(lab.long_tp, index=lab.index, name="long_tp")
    y_short = pd.Series(lab.short_tp, index=lab.index, name="short_tp")

    # Align features and labels on shared index
    shared = X.index.intersection(lab.index)
    X = X.loc[shared].copy()
    y_long = y_long.loc[shared].copy()
    y_short = y_short.loc[shared].copy()
    print(f"[audit]   aligned: X={X.shape}  y_long mean={y_long.mean():.3f}  y_short mean={y_short.mean():.3f}",
          flush=True)

    # Replace any residual inf/nan (build_features dropna'd but defensive)
    X = X.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    y_long = y_long.loc[X.index]
    y_short = y_short.loc[X.index]

    # SUBSAMPLE for MI speed — full ~140k rows is too slow for MI w/ 200 cols.
    # Use a deterministic 30k-row subsample.
    N_SAMPLE = 30000
    if len(X) > N_SAMPLE:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=N_SAMPLE, replace=False)
        idx.sort()
        X_s = X.iloc[idx]
        y_long_s = y_long.iloc[idx]
        y_short_s = y_short.iloc[idx]
    else:
        X_s = X
        y_long_s = y_long
        y_short_s = y_short
    print(f"[audit] MI on subsample of {len(X_s):,} rows / {X.shape[1]} features", flush=True)

    # ------------------------------------------------------------------
    # 1. Variance / scale check
    # ------------------------------------------------------------------
    print(f"[audit] [1/6] variance + scale checks", flush=True)
    stds = X_s.std()
    means = X_s.mean()
    abs_means = means.abs()
    near_constant = stds[stds < 1e-6].index.tolist()
    low_var = stds[stds < 1e-3].index.tolist()
    # Scale heterogeneity: ratio of max std to median std
    scale_summary = {
        "min_std": float(stds.min()),
        "max_std": float(stds.max()),
        "median_std": float(stds.median()),
        "scale_ratio_max_over_median": float(stds.max() / max(stds.median(), 1e-12)),
        "n_near_constant_lt_1e-6": len(near_constant),
        "n_low_var_lt_1e-3": len(low_var),
        "near_constant_cols": near_constant[:30],
        "low_var_cols": low_var[:30],
    }

    # ------------------------------------------------------------------
    # 2. Mutual information vs long_tp + short_tp
    # ------------------------------------------------------------------
    print(f"[audit] [2/6] MI(long_tp) ...", flush=True)
    from sklearn.feature_selection import mutual_info_classif
    tA = time.time()
    mi_long = mutual_info_classif(X_s.values, y_long_s.values, random_state=42, n_neighbors=3)
    print(f"[audit]   long MI done in {time.time()-tA:.1f}s", flush=True)
    tA = time.time()
    mi_short = mutual_info_classif(X_s.values, y_short_s.values, random_state=42, n_neighbors=3)
    print(f"[audit]   short MI done in {time.time()-tA:.1f}s", flush=True)
    mi_long = pd.Series(mi_long, index=X_s.columns, name="mi_long")
    mi_short = pd.Series(mi_short, index=X_s.columns, name="mi_short")
    mi_total = (mi_long + mi_short).rename("mi_total")

    families = pd.Series([feature_to_family(c) for c in X_s.columns],
                         index=X_s.columns, name="family")
    mi_df = pd.concat([families, mi_long, mi_short, mi_total, stds.rename("std")], axis=1)
    mi_df = mi_df.sort_values("mi_total", ascending=False)

    mi_df.to_csv(out_dir / "feature_mi.csv")
    print(f"[audit]   MI CSV → {out_dir / 'feature_mi.csv'}", flush=True)

    top10 = mi_df.head(10)
    bot10 = mi_df.tail(10).iloc[::-1]

    # Concentration: what fraction of total MI is in top-20?
    tot = mi_df["mi_total"].sum()
    top20_frac = mi_df["mi_total"].head(20).sum() / max(tot, 1e-12)
    top40_frac = mi_df["mi_total"].head(40).sum() / max(tot, 1e-12)
    bot80_frac = mi_df["mi_total"].tail(80).sum() / max(tot, 1e-12)

    # ------------------------------------------------------------------
    # 3. Redundancy / correlation
    # ------------------------------------------------------------------
    print(f"[audit] [3/6] correlation matrix ({X_s.shape[1]} cols)", flush=True)
    tA = time.time()
    # spearman is robust to scale but slow; use pearson on subsample
    corr = X_s.corr().abs()
    print(f"[audit]   corr done in {time.time()-tA:.1f}s", flush=True)

    # Pairs |r| > 0.8 (excluding diagonal & duplicates)
    triu = np.triu(np.ones_like(corr.values, dtype=bool), k=1)
    pairs = []
    rows, cols = np.where(triu & (corr.values > 0.8))
    for i, j in zip(rows, cols):
        pairs.append((corr.columns[i], corr.columns[j], float(corr.values[i, j])))
    pairs.sort(key=lambda x: -x[2])
    # Count of features involved in any |r|>0.8 pair
    redundant_feats = set()
    for a, b, _ in pairs:
        redundant_feats.add(a)
        redundant_feats.add(b)

    # Pairs > 0.9 + > 0.95
    n_pairs_90 = sum(1 for _, _, r in pairs if r > 0.90)
    n_pairs_95 = sum(1 for _, _, r in pairs if r > 0.95)

    redundancy_summary = {
        "n_features_total": int(X_s.shape[1]),
        "n_pairs_abs_corr_gt_0.8": len(pairs),
        "n_pairs_abs_corr_gt_0.9": n_pairs_90,
        "n_pairs_abs_corr_gt_0.95": n_pairs_95,
        "n_features_in_any_high_corr_pair": len(redundant_feats),
        "fraction_features_redundant": len(redundant_feats) / X_s.shape[1],
        "top_20_pairs": [(a, b, r) for a, b, r in pairs[:20]],
    }

    # ------------------------------------------------------------------
    # 4. Per-TF analysis
    # ------------------------------------------------------------------
    print(f"[audit] [4/6] per-TF block analysis", flush=True)
    tf_summary = {}
    for tf in (1, 3, 5, 15, 30, 60):
        block_cols = [c for c in X_s.columns if c.startswith(f"tf{tf}_")]
        if not block_cols:
            continue
        block_mi = mi_df.loc[block_cols, "mi_total"]
        tf_summary[f"tf{tf}"] = {
            "n_cols": len(block_cols),
            "mean_mi": float(block_mi.mean()),
            "median_mi": float(block_mi.median()),
            "max_mi": float(block_mi.max()),
            "sum_mi": float(block_mi.sum()),
            "top_feat": block_mi.idxmax(),
        }

    # ------------------------------------------------------------------
    # 5. Per-family scoring (Phase E specifically + all)
    # ------------------------------------------------------------------
    print(f"[audit] [5/6] per-family scoring", flush=True)
    family_summary = {}
    for fam in sorted(mi_df["family"].unique()):
        sub = mi_df[mi_df["family"] == fam]
        family_summary[fam] = {
            "n_cols": int(len(sub)),
            "mean_mi": float(sub["mi_total"].mean()),
            "median_mi": float(sub["mi_total"].median()),
            "max_mi": float(sub["mi_total"].max()),
            "sum_mi": float(sub["mi_total"].sum()),
        }

    # ------------------------------------------------------------------
    # 6. Drop-bottom-N experiment: stability of a simple linear probe
    # ------------------------------------------------------------------
    print(f"[audit] [6/6] drop-bottom-N experiment (linear probe AUC)", flush=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    def auc_with_features(cols):
        Xc = X_s[cols].values
        # train/test split — temporal: first 80% / last 20%
        n = len(Xc)
        split = int(n * 0.8)
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xc[:split])
        Xte = sc.transform(Xc[split:])
        ytr_l = y_long_s.values[:split]
        yte_l = y_long_s.values[split:]
        ytr_s = y_short_s.values[:split]
        yte_s = y_short_s.values[split:]
        try:
            m_l = LogisticRegression(max_iter=300, C=0.1, solver="lbfgs").fit(Xtr, ytr_l)
            m_s = LogisticRegression(max_iter=300, C=0.1, solver="lbfgs").fit(Xtr, ytr_s)
            auc_l = roc_auc_score(yte_l, m_l.predict_proba(Xte)[:, 1])
            auc_s = roc_auc_score(yte_s, m_s.predict_proba(Xte)[:, 1])
            return auc_l, auc_s
        except Exception as e:
            return None, None

    all_cols = mi_df.index.tolist()
    # Sort: high MI first → keep top, drop bottom
    keep_all = all_cols
    drop_bot50 = mi_df.head(len(all_cols) - 50).index.tolist()
    keep_top50 = mi_df.head(50).index.tolist()
    keep_top30 = mi_df.head(30).index.tolist()
    keep_top20 = mi_df.head(20).index.tolist()

    print(f"[audit]   probe: all {len(all_cols)} features", flush=True)
    a_all = auc_with_features(keep_all)
    print(f"[audit]   probe: drop bottom 50 → {len(drop_bot50)} features", flush=True)
    a_drop50 = auc_with_features(drop_bot50)
    print(f"[audit]   probe: top 50 features only", flush=True)
    a_top50 = auc_with_features(keep_top50)
    print(f"[audit]   probe: top 30 features only", flush=True)
    a_top30 = auc_with_features(keep_top30)
    print(f"[audit]   probe: top 20 features only", flush=True)
    a_top20 = auc_with_features(keep_top20)

    # Random feature dropout: stability test
    print(f"[audit]   random 10% dropout stability (5 trials)", flush=True)
    rng = np.random.default_rng(0)
    drop_pct = 0.1
    aucs_l, aucs_s = [], []
    for trial in range(5):
        keep_idx = rng.choice(len(all_cols),
                              size=int(len(all_cols) * (1 - drop_pct)),
                              replace=False)
        cols = [all_cols[i] for i in keep_idx]
        a_l, a_s = auc_with_features(cols)
        if a_l is not None:
            aucs_l.append(a_l)
            aucs_s.append(a_s)

    probe_summary = {
        "n_train": int(len(X_s) * 0.8),
        "n_test": int(len(X_s) * 0.2),
        "n_features_all": len(all_cols),
        "auc_all_features":        {"long": a_all[0], "short": a_all[1]},
        "auc_drop_bottom_50":      {"long": a_drop50[0], "short": a_drop50[1]},
        "auc_top_50_only":         {"long": a_top50[0], "short": a_top50[1]},
        "auc_top_30_only":         {"long": a_top30[0], "short": a_top30[1]},
        "auc_top_20_only":         {"long": a_top20[0], "short": a_top20[1]},
        "random_dropout_long_aucs":  aucs_l,
        "random_dropout_short_aucs": aucs_s,
        "random_dropout_std_long":   float(np.std(aucs_l)) if aucs_l else None,
        "random_dropout_std_short":  float(np.std(aucs_s)) if aucs_s else None,
    }

    # ------------------------------------------------------------------
    # Write the full report
    # ------------------------------------------------------------------
    report = {
        "config": {
            "start": args.start,
            "end": args.end,
            "n_bars": int(len(df)),
            "n_features": int(X_s.shape[1]),
            "n_samples_mi": int(len(X_s)),
            "n_samples_total": int(len(X)),
            "structural_families": sorted(enabled),
            "long_tp_mean": float(y_long.mean()),
            "short_tp_mean": float(y_short.mean()),
        },
        "concentration": {
            "total_mi": float(tot),
            "top20_mi_share": float(top20_frac),
            "top40_mi_share": float(top40_frac),
            "bottom80_mi_share": float(bot80_frac),
        },
        "top_10_features": [
            {"feature": idx, "family": row["family"],
             "mi_long": float(row["mi_long"]), "mi_short": float(row["mi_short"]),
             "mi_total": float(row["mi_total"])}
            for idx, row in top10.iterrows()
        ],
        "bottom_10_features": [
            {"feature": idx, "family": row["family"],
             "mi_long": float(row["mi_long"]), "mi_short": float(row["mi_short"]),
             "mi_total": float(row["mi_total"])}
            for idx, row in bot10.iterrows()
        ],
        "variance_scale": scale_summary,
        "redundancy": redundancy_summary,
        "per_tf": tf_summary,
        "per_family": family_summary,
        "drop_experiment": probe_summary,
    }

    out_json = out_dir / "audit_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[audit] report → {out_json}", flush=True)
    print(f"[audit] total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
