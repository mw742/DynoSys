#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE_COVARS = {"sex", "age"} | {f"PC{i}" for i in range(1, 11)}

def find_covariate_cols(df: pd.DataFrame):
    cols = list(df.columns)
    site_cols = sorted([c for c in cols if c.startswith("site_")])
    covars = [c for c in ["sex", "age"] if c in cols]
    covars += [f"PC{i}" for i in range(1, 11) if f"PC{i}" in cols]
    covars += site_cols
    return covars

def split_feature_sets(df: pd.DataFrame, predictor_cols):
    covars = [c for c in find_covariate_cols(df) if c in predictor_cols]
    others = [c for c in predictor_cols if c not in covars]
    return covars, others

def _impute_df(train_df: pd.DataFrame, test_df: pd.DataFrame):
    if train_df.shape[1] == 0:
        return train_df.copy(), test_df.copy(), None

    # Columns completely missing in training data break SimpleImputer shape assumptions.
    all_na_cols = train_df.columns[train_df.isna().all()].tolist()
    keep_cols = [c for c in train_df.columns if c not in all_na_cols]

    if len(keep_cols) == 0:
        tr = pd.DataFrame(0.0, index=train_df.index, columns=train_df.columns)
        te = pd.DataFrame(0.0, index=test_df.index, columns=test_df.columns)
        return tr, te, None

    imp = SimpleImputer(strategy="median")

    tr_imp = imp.fit_transform(train_df[keep_cols])
    te_imp = imp.transform(test_df[keep_cols])

    tr = pd.DataFrame(tr_imp, columns=keep_cols, index=train_df.index)
    te = pd.DataFrame(te_imp, columns=keep_cols, index=test_df.index)

    # Re-add dropped all-NA columns as zeros so downstream code keeps stable shapes.
    for c in all_na_cols:
        tr[c] = 0.0
        te[c] = 0.0

    tr = tr[train_df.columns]
    te = te[test_df.columns]
    return tr, te, imp

def _scale_df(train_df: pd.DataFrame, test_df: pd.DataFrame):
    if train_df.shape[1] == 0:
        return train_df.copy(), test_df.copy(), None
    scl = StandardScaler()
    tr = pd.DataFrame(scl.fit_transform(train_df), columns=train_df.columns, index=train_df.index)
    te = pd.DataFrame(scl.transform(test_df), columns=test_df.columns, index=test_df.index)
    return tr, te, scl

def _residualize_against_covars(train_cov: pd.DataFrame, test_cov: pd.DataFrame,
                                train_x: pd.DataFrame, test_x: pd.DataFrame):
    if train_x.shape[1] == 0:
        return train_x.copy(), test_x.copy()
    if train_cov.shape[1] == 0:
        return train_x.copy(), test_x.copy()

    Ctr = np.column_stack([np.ones(len(train_cov)), train_cov.to_numpy(dtype=float)])
    Cte = np.column_stack([np.ones(len(test_cov)), test_cov.to_numpy(dtype=float)])

    Xtr = train_x.to_numpy(dtype=float)
    Xte = test_x.to_numpy(dtype=float)

    beta, *_ = np.linalg.lstsq(Ctr, Xtr, rcond=None)
    Xtr_res = Xtr - Ctr @ beta
    Xte_res = Xte - Cte @ beta

    tr = pd.DataFrame(Xtr_res, columns=train_x.columns, index=train_x.index)
    te = pd.DataFrame(Xte_res, columns=test_x.columns, index=test_x.index)
    return tr, te

def residualize_target_continuous(train_cov: pd.DataFrame, test_cov: pd.DataFrame,
                                  y_train: pd.Series, y_test: pd.Series):
    if train_cov.shape[1] == 0:
        return y_train.copy(), y_test.copy(), np.zeros(len(y_test), dtype=float)

    Ctr = np.column_stack([np.ones(len(train_cov)), train_cov.to_numpy(dtype=float)])
    Cte = np.column_stack([np.ones(len(test_cov)), test_cov.to_numpy(dtype=float)])

    ytr = np.asarray(y_train, dtype=float)
    yte = np.asarray(y_test, dtype=float)

    beta, *_ = np.linalg.lstsq(Ctr, ytr, rcond=None)
    ytr_cov = Ctr @ beta
    yte_cov = Cte @ beta

    ytr_res = ytr - ytr_cov
    yte_res = yte - yte_cov
    return pd.Series(ytr_res, index=y_train.index), pd.Series(yte_res, index=y_test.index), yte_cov

def prepare_continuous_split(xtr_df: pd.DataFrame, xte_df: pd.DataFrame,
                             ytr: pd.Series, yte: pd.Series,
                             covar_cols, other_cols):
    cov_tr = xtr_df[covar_cols].copy() if covar_cols else pd.DataFrame(index=xtr_df.index)
    cov_te = xte_df[covar_cols].copy() if covar_cols else pd.DataFrame(index=xte_df.index)
    oth_tr = xtr_df[other_cols].copy() if other_cols else pd.DataFrame(index=xtr_df.index)
    oth_te = xte_df[other_cols].copy() if other_cols else pd.DataFrame(index=xte_df.index)

    cov_tr, cov_te, _ = _impute_df(cov_tr, cov_te)
    oth_tr, oth_te, _ = _impute_df(oth_tr, oth_te)

    cov_tr, cov_te, _ = _scale_df(cov_tr, cov_te)
    oth_tr, oth_te, _ = _scale_df(oth_tr, oth_te)

    oth_tr_res, oth_te_res = _residualize_against_covars(cov_tr, cov_te, oth_tr, oth_te)
    ytr_res, yte_res, yte_cov = residualize_target_continuous(cov_tr, cov_te, ytr, yte)

    return {
        "xtr": oth_tr_res.to_numpy(dtype=float),
        "xte": oth_te_res.to_numpy(dtype=float),
        "ytr_res": np.asarray(ytr_res, dtype=float),
        "yte_res": np.asarray(yte_res, dtype=float),
        "yte_cov": np.asarray(yte_cov, dtype=float),
        "feature_names": list(oth_tr_res.columns),
        "n_covars": int(len(covar_cols)),
    }

def prepare_binary_split(xtr_df: pd.DataFrame, xte_df: pd.DataFrame, covar_cols, other_cols):
    cov_tr = xtr_df[covar_cols].copy() if covar_cols else pd.DataFrame(index=xtr_df.index)
    cov_te = xte_df[covar_cols].copy() if covar_cols else pd.DataFrame(index=xte_df.index)
    oth_tr = xtr_df[other_cols].copy() if other_cols else pd.DataFrame(index=xtr_df.index)
    oth_te = xte_df[other_cols].copy() if other_cols else pd.DataFrame(index=xte_df.index)

    cov_tr, cov_te, _ = _impute_df(cov_tr, cov_te)
    oth_tr, oth_te, _ = _impute_df(oth_tr, oth_te)

    cov_tr, cov_te, _ = _scale_df(cov_tr, cov_te)
    oth_tr, oth_te, _ = _scale_df(oth_tr, oth_te)

    oth_tr_res, oth_te_res = _residualize_against_covars(cov_tr, cov_te, oth_tr, oth_te)

    xtr_parts = []
    xte_parts = []
    feat_names = []

    if cov_tr.shape[1] > 0:
        xtr_parts.append(cov_tr.to_numpy(dtype=float))
        xte_parts.append(cov_te.to_numpy(dtype=float))
        feat_names += list(cov_tr.columns)

    if oth_tr_res.shape[1] > 0:
        xtr_parts.append(oth_tr_res.to_numpy(dtype=float))
        xte_parts.append(oth_te_res.to_numpy(dtype=float))
        feat_names += list(oth_tr_res.columns)

    if len(xtr_parts) == 0:
        raise ValueError("No predictors remain after covariate handling")

    xtr = np.column_stack(xtr_parts)
    xte = np.column_stack(xte_parts)

    return {
        "xtr": xtr,
        "xte": xte,
        "feature_names": feat_names,
        "n_covars": int(len(covar_cols)),
    }
