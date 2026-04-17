#!/usr/bin/env python3
import json
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from covar_adjust import find_covariate_cols


ROOT = Path("/gpfs/group/ehlers/mwei/")
PHENO = os.environ.get("PHENO", "externalizing")
TASK_TYPE = os.environ.get("TASK_TYPE", "continuous")

RESULTS_DIR = ROOT / "results" / PHENO
OUT_DIR = RESULTS_DIR / "sem_style"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_UNIFIED = RESULTS_DIR / "unified_inputs" / f"{PHENO}_unified_weighted.tsv"
DEFAULT_LATENT = RESULTS_DIR / "state_space" / "latent_state_trajectories.csv"
TARGET = f"{PHENO}_t" if TASK_TYPE == "continuous" else "binary_event"

FOUNDATION_CANDIDATES = [
    "externalizing_prs", "internalizing_youth_prs", "internalizing_combo_prs",
    "anxiety_prs", "mdd_prs", "neuroticism_prs",
    "alcohol_prs", "any_prs", "cannabis_prs", "nicotine_prs",
]

MIN_N = 100


def bootstrap_subjects(df, iid_col, n_boot, seed):
    rng = np.random.default_rng(seed)
    ids = df[iid_col].drop_duplicates().tolist()
    for _ in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        pieces = []
        for i, sid in enumerate(sampled):
            tmp = df.loc[df[iid_col] == sid].copy()
            tmp[iid_col] = f"{sid}__b{i}"
            pieces.append(tmp)
        yield pd.concat(pieces, axis=0, ignore_index=True)


def clean_design_matrix(X: pd.DataFrame):
    X = X.copy()

    # remove duplicate columns
    X = X.loc[:, ~X.columns.duplicated()].copy()

    # numeric cast
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # drop all-NA columns
    X = X.loc[:, X.notna().any(axis=0)].copy()
    if X.shape[1] == 0:
        return X

    # drop constant columns
    nunq = X.nunique(dropna=True)
    keep = nunq[nunq > 1].index.tolist()
    X = X[keep].copy()
    if X.shape[1] == 0:
        return X

    # median impute for rank checks / model stability
    X = X.fillna(X.median(numeric_only=True))

    # add intercept first, then remove rank-deficient columns greedily
    Xc = sm.add_constant(X, has_constant="add")

    kept = []
    current = pd.DataFrame(index=Xc.index)

    for col in Xc.columns:
        test = pd.concat([current, Xc[[col]]], axis=1)
        arr = test.to_numpy(dtype=float)
        if np.linalg.matrix_rank(arr) == test.shape[1]:
            current = test
            kept.append(col)

    return current


def fit_model(df, y_col, x_cols, task_type):
    covars = [c for c in find_covariate_cols(df) if c != y_col and c not in x_cols]
    use_cols = []
    seen = set()
    for c in x_cols + covars:
        if c in df.columns and c not in seen:
            use_cols.append(c)
            seen.add(c)

    tmp = df[[y_col] + use_cols].copy()
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.dropna(subset=[y_col]).copy()

    if tmp.shape[0] < MIN_N:
        return None

    X_raw = tmp[use_cols].copy()
    X = clean_design_matrix(X_raw)
    if X.shape[1] <= 1:
        return None

    keep_idx = X.index
    y = tmp.loc[keep_idx, y_col].astype(float)

    if task_type == "continuous":
        try:
            return sm.OLS(y, X).fit()
        except Exception:
            return None

    # binary
    if y.nunique() < 2:
        return None

    try:
        return sm.Logit(y, X).fit(disp=False, maxiter=200)
    except (LinAlgError, OverflowError, ValueError):
        try:
            # robust fallback for separation / singular Hessian
            return sm.Logit(y, X).fit_regularized(alpha=1e-4, L1_wt=0.0, disp=False, maxiter=500)
        except Exception:
            return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified_table", default=str(DEFAULT_UNIFIED))
    parser.add_argument("--latent_table", default=str(DEFAULT_LATENT))
    parser.add_argument("--n_boot", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    uni = pd.read_csv(args.unified_table, sep="\t")
    lat = pd.read_csv(args.latent_table)

    uni["IID"] = uni["IID"].astype(str)
    lat["IID"] = lat["IID"].astype(str)

    df = uni.merge(lat, on=["IID", "time_num"], how="inner", suffixes=("", "_lat"))
    df = df.loc[:, ~df.columns.duplicated()].copy()

    foundation = next((c for c in FOUNDATION_CANDIDATES if c in df.columns), None)
    if foundation is None:
        raise SystemExit("[ERROR] No foundation PRS column found.")

    mediators = [
        c for c in
        ["latent_state_1", "latent_state_2", "environment_state", "family_state", "physical_state", "technology_state"]
        if c in df.columns
    ]
    if len(mediators) == 0:
        raise SystemExit("[ERROR] No mediator columns found.")

    rows = []
    for med in mediators:
        a_boot = []
        b_boot = []
        cprime_boot = []
        acme_boot = []

        for boot_df in bootstrap_subjects(df, "IID", args.n_boot, args.seed):
            m1 = fit_model(boot_df, med, [foundation], "continuous")
            if m1 is None or foundation not in m1.params.index:
                continue
            a = float(m1.params[foundation])

            m2 = fit_model(boot_df, TARGET, [foundation, med], TASK_TYPE)
            if m2 is None or med not in m2.params.index or foundation not in m2.params.index:
                continue
            b = float(m2.params[med])
            cprime = float(m2.params[foundation])

            if np.isfinite(a) and np.isfinite(b) and np.isfinite(cprime):
                a_boot.append(a)
                b_boot.append(b)
                cprime_boot.append(cprime)
                acme_boot.append(a * b)

        if len(acme_boot) == 0:
            rows.append({
                "pheno": PHENO,
                "task_type": TASK_TYPE,
                "foundation": foundation,
                "mediator": med,
                "a_mean": np.nan,
                "b_mean": np.nan,
                "cprime_mean": np.nan,
                "acme_mean": np.nan,
                "acme_ci_low": np.nan,
                "acme_ci_high": np.nan,
                "n_boot_kept": 0,
            })
            continue

        rows.append({
            "pheno": PHENO,
            "task_type": TASK_TYPE,
            "foundation": foundation,
            "mediator": med,
            "a_mean": float(np.mean(a_boot)),
            "b_mean": float(np.mean(b_boot)),
            "cprime_mean": float(np.mean(cprime_boot)),
            "acme_mean": float(np.mean(acme_boot)),
            "acme_ci_low": float(np.quantile(acme_boot, 0.025)),
            "acme_ci_high": float(np.quantile(acme_boot, 0.975)),
            "n_boot_kept": int(len(acme_boot)),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "sem_mediation_summary.csv", index=False)

    with open(OUT_DIR / "sem_settings.json", "w") as f:
        json.dump({
            "pheno": PHENO,
            "task_type": TASK_TYPE,
            "target": TARGET,
            "n_boot": args.n_boot,
            "foundation_candidates": FOUNDATION_CANDIDATES,
            "min_n": MIN_N,
            "binary_fallback": "statsmodels Logit.fit_regularized ridge-like fallback",
        }, f, indent=2)

    print("[INFO] SEM-style outputs written to:", OUT_DIR)

if __name__ == "__main__":
    main()
