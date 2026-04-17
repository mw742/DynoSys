#!/usr/bin/env python3
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from covar_adjust import split_feature_sets, prepare_continuous_split

ROOT = Path("/gpfs/group/ehlers/mwei/")

PHENO = os.environ.get("PHENO", "externalizing")
TASK_TYPE = os.environ.get("TASK_TYPE", "continuous")
TARGET = os.environ.get("TARGET_NAME", f"{PHENO}_t_next")

if TASK_TYPE != "continuous":
    raise SystemExit(f"[ERROR] 04_train_forward_benchmarks_generic.py is only for continuous phenotypes. Got TASK_TYPE={TASK_TYPE}")

INPUT_DIR = ROOT / "results" / PHENO / "forward_inputs"
OUT_DIR = ROOT / "results" / PHENO / "forward_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TABLES = [
    f"{PHENO}_unified_pca_forward.tsv",
    f"{PHENO}_unified_weighted_forward.tsv",
    f"{PHENO}_unified_cluster_forward.tsv",
    f"{PHENO}_unified_all_forward.tsv",
]

N_REPEATS = int(os.environ.get("N_REPEATS", "10"))
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.20"))
MISSINGNESS_THRESHOLD = float(os.environ.get("MISSINGNESS_THRESHOLD", "0.70"))
INCLUDE_TIME = os.environ.get("INCLUDE_TIME", "1") == "1"
RANDOM_STATE = 2026

EXCLUDE = {
    "FID", "IID", "src_subject_id", "eventname", "time_year",
    f"{PHENO}_t", f"{PHENO}_t_next", f"delta_{PHENO}", f"{PHENO}_lag", f"delta_{PHENO}_next",
    "has_next_wave", "future_gap", "worsen_event_0p5sd", "worsen_event_1sd", f"high_{PHENO}_next_q75"
}

MODELS = ["ridge", "lasso", "elasticnet"]

def predictor_cols(df: pd.DataFrame):
    cols = [c for c in df.columns if c not in EXCLUDE]
    if not INCLUDE_TIME and "time_num" in cols:
        cols.remove("time_num")
    return cols

def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])

def prepare(df: pd.DataFrame):
    if TARGET not in df.columns:
        raise ValueError(f"Target {TARGET} not found in dataframe columns")

    y = pd.to_numeric(df[TARGET], errors="coerce")
    x = df[predictor_cols(df)].apply(pd.to_numeric, errors="coerce").copy()
    groups = df["IID"].astype(str)

    ok = y.notna()
    x = x.loc[ok].reset_index(drop=True)
    y = y.loc[ok].reset_index(drop=True)
    groups = groups.loc[ok].reset_index(drop=True)

    if x.shape[0] == 0:
        raise ValueError("No rows remain after filtering non-missing target values")

    keep_missing = x.columns[x.isna().mean() <= MISSINGNESS_THRESHOLD].tolist()
    x = x[keep_missing].copy()

    keep_nonconstant = x.columns[x.nunique(dropna=True) > 1].tolist()
    x = x[keep_nonconstant].copy()

    if x.shape[1] == 0:
        raise ValueError("No predictors remain after missingness / non-constant filtering")

    return x, y, groups

def make_model(name: str):
    if name == "ridge":
        return RidgeCV(alphas=np.logspace(-3, 3, 30))
    if name == "lasso":
        return LassoCV(alphas=np.logspace(-3, 1, 50), cv=None, max_iter=50000, n_jobs=1, random_state=RANDOM_STATE)
    if name == "elasticnet":
        return ElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8, 0.95, 1.0],
            alphas=np.logspace(-3, 1, 50),
            cv=None,
            max_iter=50000,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model: {name}")

def fit_grouped(model, xtr, ytr, gtr):
    if hasattr(model, "cv"):
        model.cv = list(GroupKFold(n_splits=5).split(xtr, ytr, gtr))
    model.fit(xtr, ytr)
    return model

def evaluate(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "corr": safe_corr(y, pred),
    }

def main():
    all_rows = []
    settings = {
        "pheno": PHENO,
        "task_type": TASK_TYPE,
        "tables": TABLES,
        "target": TARGET,
        "models": MODELS,
        "n_repeats": N_REPEATS,
        "test_size": TEST_SIZE,
        "missingness_threshold": MISSINGNESS_THRESHOLD,
        "include_time": INCLUDE_TIME,
        "random_state": RANDOM_STATE,
        "covariate_mode": "continuous=splitwise_residualization",
    }

    for table in TABLES:
        path = INPUT_DIR / table
        if not path.exists():
            print(f"[WARN] Missing table, skipping: {path}")
            continue

        df = pd.read_csv(path, sep="\t")
        x, y, groups = prepare(df)
        covar_cols, other_cols = split_feature_sets(df=x, predictor_cols=list(x.columns))

        splitter = GroupShuffleSplit(n_splits=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        for model_name in MODELS:
            coef_rows = []
            metrics_rows = []

            for split_idx, (tr, te) in enumerate(splitter.split(x, y, groups=groups), start=1):
                xtr_df, xte_df = x.iloc[tr].copy(), x.iloc[te].copy()
                ytr, yte = y.iloc[tr].copy(), y.iloc[te].copy()
                gtr, gte = groups.iloc[tr].copy(), groups.iloc[te].copy()

                prep = prepare_continuous_split(xtr_df, xte_df, ytr, yte, covar_cols, other_cols)
                model = fit_grouped(make_model(model_name), prep["xtr"], prep["ytr_res"], gtr)
                pred = prep["yte_cov"] + model.predict(prep["xte"])

                row = {
                    "table": table,
                    "target": TARGET,
                    "model_type": model_name,
                    "split": split_idx,
                    "n_train_rows": int(len(tr)),
                    "n_test_rows": int(len(te)),
                    "n_train_subjects": int(gtr.nunique()),
                    "n_test_subjects": int(gte.nunique()),
                    "n_predictors_penalized": int(len(prep["feature_names"])),
                    "n_covariates": int(prep["n_covars"]),
                }
                row.update(evaluate(yte, pred))

                coef = getattr(model, "coef_", None)
                if coef is not None:
                    coef = np.asarray(coef).ravel()
                    row["n_nonzero"] = int((np.abs(coef) > 1e-10).sum())

                    for f, c in zip(prep["feature_names"], coef):
                        if abs(c) > 1e-10:
                            coef_rows.append({
                                "table": table,
                                "target": TARGET,
                                "model_type": model_name,
                                "split": split_idx,
                                "feature": f,
                                "coef": float(c),
                                "abs_coef": float(abs(c)),
                                "sign": int(np.sign(c)),
                            })
                else:
                    row["n_nonzero"] = np.nan

                if hasattr(model, "alpha_"):
                    row["alpha"] = float(model.alpha_)
                if hasattr(model, "l1_ratio_"):
                    row["l1_ratio"] = float(model.l1_ratio_)

                metrics_rows.append(row)
                all_rows.append(row)

            metrics_df = pd.DataFrame(metrics_rows)
            metrics_df.to_csv(OUT_DIR / f"{table.replace('.tsv','')}_{model_name}_metrics.csv", index=False)

            coef_df = pd.DataFrame(coef_rows)
            coef_df.to_csv(OUT_DIR / f"{table.replace('.tsv','')}_{model_name}_coefficients.csv", index=False)

            if not coef_df.empty:
                stable = (
                    coef_df.groupby("feature")
                    .agg(
                        n_splits_selected=("split", "nunique"),
                        selection_rate=("split", lambda s: s.nunique() / N_REPEATS),
                        mean_coef=("coef", "mean"),
                        mean_abs_coef=("abs_coef", "mean"),
                        sign_consistency=("sign", lambda s: np.mean(np.sign(s) == np.sign(np.mean(s))))
                    )
                    .reset_index()
                    .sort_values(["selection_rate", "mean_abs_coef"], ascending=[False, False])
                )
                stable.to_csv(OUT_DIR / f"{table.replace('.tsv','')}_{model_name}_stable_features.csv", index=False)

    if len(all_rows) == 0:
        raise SystemExit("[ERROR] No forward benchmark results were produced")

    summary = pd.DataFrame(all_rows)
    summary.to_csv(OUT_DIR / "forward_benchmark_all_metrics.csv", index=False)

    (
        summary.groupby(["table", "target", "model_type"])
        .agg(
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_r2=("r2", "mean"),
            mean_corr=("corr", "mean"),
            mean_nonzero=("n_nonzero", "mean"),
        )
        .reset_index()
        .sort_values(["mean_r2", "mean_corr"], ascending=False)
        .to_csv(OUT_DIR / "forward_benchmark_summary.csv", index=False)
    )

    with open(OUT_DIR / "forward_benchmark_settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    print("[INFO] Forward benchmark outputs written to:", OUT_DIR)

if __name__ == "__main__":
    main()
