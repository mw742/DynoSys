#!/usr/bin/env python3
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

from covar_adjust import split_feature_sets, prepare_binary_split

ROOT = Path("/gpfs/group/ehlers/mwei/")

PHENO = os.environ.get("PHENO", "externalizing")
TASK_TYPE = os.environ.get("TASK_TYPE", "continuous")
TARGET = os.environ.get("HAZARD_TARGET", f"high_{PHENO}_next_q75")

if TASK_TYPE != "continuous":
    raise SystemExit(f"[ERROR] 05_train_discrete_time_hazard_generic.py is only for continuous phenotypes. Got TASK_TYPE={TASK_TYPE}")

INPUT_DIR = ROOT / "results" / PHENO / "forward_inputs"
OUT_DIR = ROOT / "results" / PHENO / "hazard_models"
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

def predictor_cols(df: pd.DataFrame):
    cols = [c for c in df.columns if c not in EXCLUDE]
    if not INCLUDE_TIME and "time_num" in cols:
        cols.remove("time_num")
    return cols

def safe_auc(y_true, p):
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, p))

def safe_auprc(y_true, p):
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, p))

def main():
    rows = []
    coef_rows = []

    settings = {
        "pheno": PHENO,
        "task_type": TASK_TYPE,
        "tables": TABLES,
        "target": TARGET,
        "n_repeats": N_REPEATS,
        "test_size": TEST_SIZE,
        "missingness_threshold": MISSINGNESS_THRESHOLD,
        "include_time": INCLUDE_TIME,
        "random_state": RANDOM_STATE,
        "model": "LogisticRegressionCV elasticnet saga",
        "covariate_mode": "raw_covars_plus_residualized_other_features",
    }

    for table in TABLES:
        path = INPUT_DIR / table
        if not path.exists():
            print(f"[WARN] Missing table, skipping: {path}")
            continue

        df = pd.read_csv(path, sep="\t")
        if TARGET not in df.columns:
            print(f"[WARN] Target {TARGET} not found in {table}, skipping")
            continue

        y = pd.to_numeric(df[TARGET], errors="coerce")
        x = df[predictor_cols(df)].apply(pd.to_numeric, errors="coerce").copy()
        groups = df["IID"].astype(str)

        ok = y.notna()
        x = x.loc[ok].reset_index(drop=True)
        y = y.loc[ok].astype(int).reset_index(drop=True)
        groups = groups.loc[ok].reset_index(drop=True)

        if x.shape[0] == 0:
            print(f"[WARN] No rows remain after target filtering for {table}")
            continue

        x = x.loc[:, x.isna().mean() <= MISSINGNESS_THRESHOLD].copy()
        x = x.loc[:, x.nunique(dropna=True) > 1].copy()
        if x.shape[1] == 0:
            print(f"[WARN] No predictors remain for {table}")
            continue

        covar_cols, other_cols = split_feature_sets(x, list(x.columns))

        splitter = GroupShuffleSplit(n_splits=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        for split_idx, (tr, te) in enumerate(splitter.split(x, y, groups=groups), start=1):
            xtr_df, xte_df = x.iloc[tr].copy(), x.iloc[te].copy()
            ytr, yte = y.iloc[tr].copy(), y.iloc[te].copy()
            gtr, gte = groups.iloc[tr].copy(), groups.iloc[te].copy()

            if ytr.nunique() < 2:
                print(f"[WARN] Train split has only one class for {table} split {split_idx}; skipping")
                continue
            if yte.nunique() < 2:
                print(f"[WARN] Test split has only one class for {table} split {split_idx}; skipping")
                continue

            prep = prepare_binary_split(xtr_df, xte_df, covar_cols, other_cols)

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
            model.fit(prep["xtr"], ytr)
            p = model.predict_proba(prep["xte"])[:, 1]

            row = {
                "table": table,
                "target": TARGET,
                "split": split_idx,
                "n_train_rows": int(len(tr)),
                "n_test_rows": int(len(te)),
                "n_train_subjects": int(gtr.nunique()),
                "n_test_subjects": int(gte.nunique()),
                "event_rate_train": float(ytr.mean()),
                "event_rate_test": float(yte.mean()),
                "auroc": safe_auc(yte, p),
                "auprc": safe_auprc(yte, p),
                "brier": float(brier_score_loss(yte, p)),
                "n_predictors_total": int(len(prep["feature_names"])),
                "n_covariates": int(prep["n_covars"]),
                "n_nonzero": int((np.abs(model.coef_.ravel()) > 1e-10).sum()),
            }

            try:
                row["best_C"] = float(np.ravel(model.C_)[0])
            except Exception:
                pass
            try:
                row["best_l1_ratio"] = float(np.ravel(model.l1_ratio_)[0])
            except Exception:
                pass

            rows.append(row)

            for f, c in zip(prep["feature_names"], model.coef_.ravel()):
                if abs(c) > 1e-10:
                    coef_rows.append({
                        "table": table,
                        "target": TARGET,
                        "split": split_idx,
                        "feature": f,
                        "coef": float(c),
                        "abs_coef": float(abs(c)),
                        "sign": int(np.sign(c)),
                    })

    if len(rows) == 0:
        raise SystemExit("[ERROR] No hazard model results were produced")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / f"hazard_{TARGET}_metrics.csv", index=False)

    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(OUT_DIR / f"hazard_{TARGET}_coefficients.csv", index=False)

    if not coef_df.empty:
        (
            coef_df.groupby(["table", "feature"])
            .agg(
                n_splits_selected=("split", "nunique"),
                selection_rate=("split", lambda s: s.nunique() / max(1, N_REPEATS)),
                mean_coef=("coef", "mean"),
                mean_abs_coef=("abs_coef", "mean")
            )
            .reset_index()
            .sort_values(["table", "selection_rate", "mean_abs_coef"], ascending=[True, False, False])
            .to_csv(OUT_DIR / f"hazard_{TARGET}_stable_features.csv", index=False)
        )

    with open(OUT_DIR / f"hazard_{TARGET}_settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    print("[INFO] Hazard model outputs written to:", OUT_DIR)

if __name__ == "__main__":
    main()
