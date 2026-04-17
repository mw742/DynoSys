#!/usr/bin/env python3
import json
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler


ROOT = Path("/gpfs/group/ehlers/mwei/")
PHENO = os.environ.get("PHENO", "externalizing")
TASK_TYPE = os.environ.get("TASK_TYPE", "continuous")

INPUT_DIR = ROOT / "results" / PHENO / "unified_inputs"
OUT_DIR = ROOT / "results" / PHENO / "state_space"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TABLE = f"{PHENO}_unified_weighted.tsv"
TARGET = f"{PHENO}_t" if TASK_TYPE == "continuous" else "binary_event"


def zscore_frame(df):
    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    arr = scl.fit_transform(imp.fit_transform(df))
    return pd.DataFrame(arr, columns=df.columns, index=df.index)

def choose_cols(df, prefixes, exact=None):
    exact = exact or []
    cols = []
    for c in df.columns:
        if c in exact:
            cols.append(c)
        elif any(c.startswith(p) for p in prefixes):
            cols.append(c)
    return cols

def domain_summary(df, cols, name):
    if len(cols) == 0:
        return pd.Series(np.nan, index=df.index, name=name)
    z = zscore_frame(df[cols].apply(pd.to_numeric, errors="coerce"))
    return z.mean(axis=1).rename(name)

def bootstrap_by_subject(df, iid_col, n_boot, seed):
    rng = np.random.default_rng(seed)
    ids = df[iid_col].drop_duplicates().tolist()
    for _ in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        out = []
        for i, sid in enumerate(sampled):
            tmp = df.loc[df[iid_col] == sid].copy()
            tmp[iid_col] = f"{sid}__boot{i}"
            out.append(tmp)
        yield pd.concat(out, axis=0, ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--n_latent", type=int, default=2)
    parser.add_argument("--n_boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    df = pd.read_csv(INPUT_DIR / args.table, sep="\t").copy()
    df["IID"] = df["IID"].astype(str)
    df = df.sort_values(["IID", "time_num"]).reset_index(drop=True)

    foundation_cols = [c for c in [
        "externalizing_prs", "internalizing_youth_prs", "internalizing_combo_prs",
        "anxiety_prs", "mdd_prs", "neuroticism_prs",
        "alcohol_prs", "any_prs", "cannabis_prs", "nicotine_prs",
        "CNT", "CNT2"
    ] if c in df.columns]

    internal_cols = choose_cols(df, prefixes=["neurorecognition_", "brain_", "smri_"])
    if len(internal_cols) == 0:
        raise SystemExit("[ERROR] No internal-state columns found.")

    df["environment_state"] = domain_summary(df, [c for c in df.columns if c.startswith("environment_")], "environment_state")
    df["family_state"] = domain_summary(df, [c for c in df.columns if c.startswith("general_family_context_")], "family_state")
    df["physical_state"] = domain_summary(df, [c for c in df.columns if c.startswith("physical_health_")], "physical_state")
    df["technology_state"] = domain_summary(df, [c for c in df.columns if c.startswith("technology_")], "technology_state")

    internal_matrix = zscore_frame(df[internal_cols].apply(pd.to_numeric, errors="coerce"))
    pca = PCA(n_components=min(args.n_latent, internal_matrix.shape[1], internal_matrix.shape[0]))
    latent = pca.fit_transform(internal_matrix)

    latent_targets = []
    for k in range(latent.shape[1]):
        col = f"latent_state_{k+1}"
        df[col] = latent[:, k]
        latent_targets.append(col)

    lag_cols = latent_targets + [
        "environment_state", "family_state", "physical_state", "technology_state", TARGET
    ] + foundation_cols

    for c in lag_cols:
        if c in df.columns:
            df[f"{c}_lag1"] = df.groupby("IID")[c].shift(1)

    work = df.loc[df["time_num"] > 0].copy()

    transition_rows = []
    outcome_rows = []

    transition_predictors = [f"{c}_lag1" for c in latent_targets] + [
        "environment_state_lag1", "family_state_lag1", "physical_state_lag1", "technology_state_lag1"
    ] + [f"{c}_lag1" for c in foundation_cols if f"{c}_lag1" in work.columns]

    outcome_predictors = latent_targets + [f"{c}_lag1" for c in foundation_cols if f"{c}_lag1" in work.columns] + [
        "environment_state_lag1", "family_state_lag1", "physical_state_lag1", "technology_state_lag1"
    ]
    if f"{TARGET}_lag1" in work.columns:
        outcome_predictors.append(f"{TARGET}_lag1")

    for boot_df in bootstrap_by_subject(work, "IID", args.n_boot, args.seed):
        for target in latent_targets:
            cols = [c for c in transition_predictors if c in boot_df.columns]
            tmp = boot_df[[target] + cols].dropna().copy()
            if tmp.shape[0] < 100:
                continue
            X = tmp[cols].to_numpy(dtype=float)
            y = tmp[target].to_numpy(dtype=float)
            mdl = RidgeCV(alphas=np.logspace(-3, 3, 25))
            mdl.fit(X, y)
            for feat, coef in zip(cols, mdl.coef_):
                transition_rows.append({"target": target, "feature": feat, "coef": float(coef)})

        cols = [c for c in outcome_predictors if c in boot_df.columns]
        tmp = boot_df[[TARGET] + cols].dropna().copy()
        if tmp.shape[0] < 100:
            continue

        X = tmp[cols].to_numpy(dtype=float)
        y = tmp[TARGET].to_numpy(dtype=float)

        if TASK_TYPE == "continuous":
            mdl = RidgeCV(alphas=np.logspace(-3, 3, 25))
            mdl.fit(X, y)
            coefs = mdl.coef_
        else:
            mdl = LogisticRegressionCV(
                Cs=np.logspace(-3, 2, 10),
                cv=5,
                penalty="elasticnet",
                solver="saga",
                scoring="roc_auc",
                max_iter=20000,
                l1_ratios=[0.2, 0.5, 0.8, 0.95, 1.0],
                n_jobs=1,
                random_state=args.seed,
            )
            mdl.fit(X, y.astype(int))
            coefs = mdl.coef_.ravel()

        for feat, coef in zip(cols, coefs):
            outcome_rows.append({"target": TARGET, "feature": feat, "coef": float(coef)})

    transition_df = pd.DataFrame(transition_rows)
    outcome_df = pd.DataFrame(outcome_rows)

    if not transition_df.empty:
        tr_sum = (
            transition_df.groupby(["target", "feature"])["coef"]
            .agg(
                mean_coef="mean",
                median_coef="median",
                ci_low=lambda s: np.quantile(s, 0.025),
                ci_high=lambda s: np.quantile(s, 0.975),
                sign_stability=lambda s: np.mean(np.sign(s) == np.sign(np.mean(s))),
            )
            .reset_index()
            .sort_values(["target", "sign_stability"], ascending=[True, False])
        )
        tr_sum.to_csv(OUT_DIR / "state_transition_coefficients.csv", index=False)

    if not outcome_df.empty:
        out_sum = (
            outcome_df.groupby(["target", "feature"])["coef"]
            .agg(
                mean_coef="mean",
                median_coef="median",
                ci_low=lambda s: np.quantile(s, 0.025),
                ci_high=lambda s: np.quantile(s, 0.975),
                sign_stability=lambda s: np.mean(np.sign(s) == np.sign(np.mean(s))),
            )
            .reset_index()
            .sort_values(["sign_stability"], ascending=False)
        )
        out_sum.to_csv(OUT_DIR / "state_to_outcome_coefficients.csv", index=False)

    keep_cols = ["IID", "time_num", TARGET, "environment_state", "family_state", "physical_state", "technology_state"] + latent_targets
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(OUT_DIR / "latent_state_trajectories.csv", index=False)

    with open(OUT_DIR / "state_space_settings.json", "w") as f:
        json.dump({
            "pheno": PHENO,
            "task_type": TASK_TYPE,
            "table": args.table,
            "target": TARGET,
            "n_latent": int(latent.shape[1]),
            "n_boot": args.n_boot,
            "foundation_cols": foundation_cols,
            "n_internal_features": len(internal_cols),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }, f, indent=2)

    print("[INFO] State-space outputs written to:", OUT_DIR)

if __name__ == "__main__":
    main()
