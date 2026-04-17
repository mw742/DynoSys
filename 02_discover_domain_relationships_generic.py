#!/usr/bin/env python3
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from covar_adjust import split_feature_sets, prepare_continuous_split, prepare_binary_split

ROOT = Path("/gpfs/group/ehlers/mwei/")
PHENO = os.environ.get("PHENO", "externalizing")
TASK_TYPE = os.environ.get("TASK_TYPE", "continuous")

INPUT_FILE = ROOT / "results" / PHENO / "unified_inputs" / f"{PHENO}_unified_weighted.tsv"
OUT_DIR = ROOT / "results" / PHENO / "relationships"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = f"{PHENO}_t" if TASK_TYPE == "continuous" else "binary_event"
EXCLUDE = {"FID", "IID", "src_subject_id", "eventname", "time_year"}

df = pd.read_csv(INPUT_FILE, sep="\t").copy()
df["IID"] = df["IID"].astype(str)
df = df.sort_values(["IID", "time_num"]).reset_index(drop=True)

candidate_cols = [c for c in df.columns if c not in EXCLUDE and c != TARGET]
for c in candidate_cols + [TARGET]:
    if c in df.columns:
        df[f"{c}__lag1"] = df.groupby("IID")[c].shift(1)

pred_cols = [f"{c}__lag1" for c in candidate_cols if f"{c}__lag1" in df.columns]
work = df.loc[df["time_num"] > 0].copy()
work = work[[TARGET] + pred_cols].copy()
work = work.apply(pd.to_numeric, errors="coerce")
work = work.dropna(subset=[TARGET]).reset_index(drop=True)

if work.shape[0] < 100:
    raise SystemExit("[ERROR] Too few rows after lagging")

X = work[pred_cols].copy()
y = work[TARGET].copy()

covar_cols, other_cols = split_feature_sets(X, list(X.columns))

if TASK_TYPE == "continuous":
    prep = prepare_continuous_split(X, X, y, y, covar_cols, other_cols)
    model = LassoCV(alphas=np.logspace(-3, 0.5, 40), cv=5, max_iter=20000, random_state=2026)
    model.fit(prep["xtr"], prep["ytr_res"])
    coef = model.coef_.ravel()
    feat_names = prep["feature_names"]
else:
    prep = prepare_binary_split(X, X, covar_cols, other_cols)
    model = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 12),
        cv=5,
        penalty="elasticnet",
        solver="saga",
        scoring="roc_auc",
        max_iter=20000,
        l1_ratios=[0.2, 0.5, 0.8, 0.95, 1.0],
        n_jobs=1,
        random_state=2026,
    )
    model.fit(prep["xtr"], y.astype(int))
    coef = model.coef_.ravel()
    feat_names = prep["feature_names"]

rows = []
for feat, beta in zip(feat_names, coef):
    if abs(beta) <= 1e-10:
        continue
    rows.append({
        "target": TARGET,
        "source_lag1": feat.replace("__lag1", ""),
        "coef": float(beta),
        "abs_coef": float(abs(beta)),
        "sign": "positive" if beta > 0 else "negative",
        "n_rows": int(work.shape[0]),
    })

edges = pd.DataFrame(rows).sort_values("abs_coef", ascending=False)
edges.to_csv(OUT_DIR / "lagged_domain_edges.csv", index=False)

with open(OUT_DIR / "relationship_settings.json", "w") as f:
    json.dump({
        "pheno": PHENO,
        "task_type": TASK_TYPE,
        "input_file": str(INPUT_FILE),
        "target": TARGET,
        "covariate_mode": "continuous=splitwise_style residualization on full analysis set; binary=raw_covars_plus_residualized_other_features",
    }, f, indent=2)

print("[INFO] Relationship discovery outputs written to:", OUT_DIR)
