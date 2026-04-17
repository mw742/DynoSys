#!/usr/bin/env python3
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNetCV, LassoCV, LogisticRegressionCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from covar_adjust import split_feature_sets, prepare_continuous_split, prepare_binary_split

ROOT = Path("/gpfs/group/ehlers/mwei/")

PHENO = os.environ.get("PHENO", "externalizing")
TASK_TYPE = os.environ.get("TASK_TYPE", "continuous")
INPUT_DIR = ROOT / "results" / PHENO / "unified_inputs"
OUT_DIR = ROOT / "results" / PHENO / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TABLES = [
    f"{PHENO}_unified_pca.tsv",
    f"{PHENO}_unified_weighted.tsv",
    f"{PHENO}_unified_cluster.tsv",
    f"{PHENO}_unified_all.tsv",
]

MODEL_TYPE = os.environ.get("MODEL_TYPE", "both").lower()
N_REPEATS = int(os.environ.get("N_REPEATS", "5"))
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.20"))
MISSINGNESS_THRESHOLD = float(os.environ.get("MISSINGNESS_THRESHOLD", "0.70"))
INCLUDE_TIME = os.environ.get("INCLUDE_TIME", "1") == "1"
RANDOM_STATE = 2026

if TASK_TYPE == "continuous":
    TARGET = f"{PHENO}_t"
    EXCLUDE = {"FID", "IID", "src_subject_id", "eventname", "time_year", TARGET, f"delta_{PHENO}", f"{PHENO}_lag"}
else:
    TARGET = "binary_event"
    EXCLUDE = {"FID", "IID", "src_subject_id", "eventname", "time_year", "time_month", "time", "event", "binary_event"}

def predictor_cols(df):
    cols = [c for c in df.columns if c not in EXCLUDE]
    if not INCLUDE_TIME and "time_num" in cols:
        cols.remove("time_num")
    return cols

def safe_corr(y_true, y_pred):
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(y_true[ok], y_pred[ok])[0, 1])

def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))

def safe_auprc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_prob))

def prepare_matrix(df):
    y = pd.to_numeric(df[TARGET], errors="coerce")
    x = df[predictor_cols(df)].apply(pd.to_numeric, errors="coerce").copy()
    groups = df["IID"].astype(str)

    ok = y.notna()
    x = x.loc[ok].reset_index(drop=True)
    y = y.loc[ok].reset_index(drop=True)
    groups = groups.loc[ok].reset_index(drop=True)

    x = x.loc[:, x.isna().mean() <= MISSINGNESS_THRESHOLD].copy()
    x = x.loc[:, x.nunique(dropna=True) > 1].copy()

    return x, y, groups

def fit_regression_model(model_type, xtr, ytr, gtr):
    if model_type == "lasso":
        model = LassoCV(alphas=np.logspace(-3, 1, 60), cv=None, max_iter=50000, n_jobs=1, random_state=RANDOM_STATE)
    elif model_type == "elasticnet":
        model = ElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8, 0.95, 1.0],
            alphas=np.logspace(-3, 1, 60),
            cv=None,
            max_iter=50000,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(model_type)

    model.cv = list(GroupKFold(n_splits=5).split(xtr, ytr, gtr))
    model.fit(xtr, ytr)
    return model

def fit_binary_model(xtr, ytr):
    model = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 12),
        cv=5,
        penalty="elasticnet",
        solver="saga",
        scoring="roc_auc",
        max_iter=20000,
        l1_ratios=[0.2, 0.5, 0.8, 0.95, 1.0],
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    model.fit(xtr, ytr.astype(int))
    return model

all_rows = []

for table in TABLES:
    df = pd.read_csv(INPUT_DIR / table, sep="\t")
    x, y, groups = prepare_matrix(df)
    covar_cols, other_cols = split_feature_sets(df=x, predictor_cols=list(x.columns))

    splitter = GroupShuffleSplit(n_splits=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    if TASK_TYPE == "continuous":
        model_types = ["lasso", "elasticnet"] if MODEL_TYPE == "both" else [MODEL_TYPE]
    else:
        model_types = ["logistic_elasticnet"]

    for model_type in model_types:
        metrics_rows = []
        coef_rows = []

        for split_idx, (tr, te) in enumerate(splitter.split(x, y, groups=groups), start=1):
            xtr_df, xte_df = x.iloc[tr].copy(), x.iloc[te].copy()
            ytr, yte = y.iloc[tr].copy(), y.iloc[te].copy()
            gtr, gte = groups.iloc[tr].copy(), groups.iloc[te].copy()

            if TASK_TYPE == "continuous":
                prep = prepare_continuous_split(xtr_df, xte_df, ytr, yte, covar_cols, other_cols)
                model = fit_regression_model(model_type, prep["xtr"], prep["ytr_res"], gtr)
                pred = prep["yte_cov"] + model.predict(prep["xte"])
                row = {
                    "table": table,
                    "model_type": model_type,
                    "split": split_idx,
                    "n_train_subjects": int(gtr.nunique()),
                    "n_test_subjects": int(gte.nunique()),
                    "n_predictors_penalized": int(len(prep["feature_names"])),
                    "n_covariates": int(prep["n_covars"]),
                    "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
                    "mae": float(mean_absolute_error(yte, pred)),
                    "r2": float(r2_score(yte, pred)),
                    "corr": safe_corr(np.asarray(yte, dtype=float), np.asarray(pred, dtype=float)),
                    "n_nonzero": int((np.abs(model.coef_) > 1e-10).sum()),
                    "alpha": float(model.alpha_),
                    "l1_ratio": float(getattr(model, "l1_ratio_", 1.0)),
                }
                coefs = np.asarray(model.coef_).ravel()
                feat_names = prep["feature_names"]
            else:
                prep = prepare_binary_split(xtr_df, xte_df, covar_cols, other_cols)
                model = fit_binary_model(prep["xtr"], ytr)
                prob = model.predict_proba(prep["xte"])[:, 1]
                row = {
                    "table": table,
                    "model_type": model_type,
                    "split": split_idx,
                    "n_train_subjects": int(gtr.nunique()),
                    "n_test_subjects": int(gte.nunique()),
                    "n_predictors_total": int(len(prep["feature_names"])),
                    "n_covariates": int(prep["n_covars"]),
                    "event_rate_test": float(np.mean(yte)),
                    "auroc": safe_auc(np.asarray(yte, dtype=int), prob),
                    "auprc": safe_auprc(np.asarray(yte, dtype=int), prob),
                    "brier": float(brier_score_loss(np.asarray(yte, dtype=int), prob)),
                    "n_nonzero": int((np.abs(model.coef_.ravel()) > 1e-10).sum()),
                }
                coefs = np.asarray(model.coef_).ravel()
                feat_names = prep["feature_names"]

            metrics_rows.append(row)
            all_rows.append(row)

            for feat, coef in zip(feat_names, coefs):
                if abs(coef) > 1e-10:
                    coef_rows.append({
                        "table": table,
                        "model_type": model_type,
                        "split": split_idx,
                        "feature": feat,
                        "coef": float(coef),
                        "abs_coef": float(abs(coef)),
                        "sign": int(np.sign(coef)),
                    })

        pd.DataFrame(metrics_rows).to_csv(
            OUT_DIR / f"{table.replace('.tsv', '')}_{model_type}_metrics.csv",
            index=False
        )
        coef_df = pd.DataFrame(coef_rows)
        coef_df.to_csv(
            OUT_DIR / f"{table.replace('.tsv', '')}_{model_type}_coefficients.csv",
            index=False
        )

        if not coef_df.empty:
            stable = (
                coef_df.groupby("feature")
                .agg(
                    n_splits_selected=("split", "nunique"),
                    selection_rate=("split", lambda s: s.nunique() / N_REPEATS),
                    mean_coef=("coef", "mean"),
                    mean_abs_coef=("abs_coef", "mean"),
                )
                .reset_index()
                .sort_values(["selection_rate", "mean_abs_coef"], ascending=[False, False])
            )
            stable.to_csv(
                OUT_DIR / f"{table.replace('.tsv', '')}_{model_type}_stable_features.csv",
                index=False
            )

summary = pd.DataFrame(all_rows)
summary.to_csv(OUT_DIR / "benchmark_all_metrics.csv", index=False)

with open(OUT_DIR / "benchmark_settings.json", "w") as f:
    json.dump({
        "pheno": PHENO,
        "task_type": TASK_TYPE,
        "tables": TABLES,
        "target": TARGET,
        "n_repeats": N_REPEATS,
        "test_size": TEST_SIZE,
        "covariate_mode": "continuous=splitwise_residualization; binary=raw_covars_plus_residualized_other_features",
    }, f, indent=2)

print("[INFO] Benchmark outputs written to:", OUT_DIR)
