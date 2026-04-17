#!/usr/bin/env python3
"""
Multi-task shared-bottom neural model for mixed outcomes.

------------------------
1. Uses covariate-aware preprocessing aligned with covar_adjust.py.
2. Splits predictors into:
   - covariates: sex, age, PC1-10, site_*
   - non-covariates: all other predictors
3. Residualizes non-covariate predictors against covariates using training-only fits.
4. Residualizes continuous outcomes against covariates using training-only fits.
5. Uses a mixed architecture:
   - shared trunk learns from residualized non-covariate features
   - classification heads also receive scaled covariates directly
   - regression heads use only the shared trunk, matching the single-task logic better
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    raise SystemExit("[ERROR] PyTorch is required. Install torch in environment first.") from e

try:
    from covar_adjust import split_feature_sets
except ImportError as e:
    raise SystemExit("[ERROR] Could not import split_feature_sets from covar_adjust.py") from e

ROOT = Path("/gpfs/group/ehlers/mwei/")
FAMILY = os.environ.get("FAMILY", "weighted").lower()
INPUT_PATH = ROOT / FAMILY / f"multitask_{FAMILY}.tsv"
RUN_TAG = os.environ.get("RUN_TAG", "").strip()
if not RUN_TAG:
    RUN_TAG = datetime.now().strftime("run_%Y%m%d_%H%M%S")
OUT_DIR = ROOT / FAMILY / "shared_bottom" / RUN_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("SEED", "2026"))
N_SPLITS = int(os.environ.get("N_SPLITS", "5"))
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.20"))
VAL_SIZE = float(os.environ.get("VAL_SIZE", "0.20"))
EPOCHS = int(os.environ.get("EPOCHS", "120"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
LR = float(os.environ.get("LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
DROPOUT = float(os.environ.get("DROPOUT", "0.20"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "256"))
SHARED_DIM = int(os.environ.get("SHARED_DIM", "128"))
PATIENCE = int(os.environ.get("PATIENCE", "25"))
MIN_EPOCHS = int(os.environ.get("MIN_EPOCHS", "25"))
MISSINGNESS_THRESHOLD = float(os.environ.get("MISSINGNESS_THRESHOLD", "0.70"))
FORCE_ALL_FEATURES = os.environ.get("FORCE_ALL_FEATURES", "0") == "1"
MAX_SHARED_FEATURES = int(os.environ.get("MAX_SHARED_FEATURES", "0"))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CLASS_LOSS = os.environ.get("CLASS_LOSS", "bce").lower()
FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", "2.0"))
TIME_AS_FEATURE = os.environ.get("TIME_AS_FEATURE", "1") == "1"
MIN_DELTA = float(os.environ.get("MIN_DELTA", "1e-4"))
TASK_WEIGHTING = os.environ.get("TASK_WEIGHTING", "inverse_observed").lower()
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))

REG_TASKS = ["externalizing", "internalizing"]
CLS_TASKS = ["alcohol", "nicotine", "cannabis", "any"]
ALL_TASKS = REG_TASKS + CLS_TASKS

ID_EXCLUDE = {"FID", "IID", "src_subject_id", "eventname"}
LABEL_COLS = {f"y_{t}" for t in ALL_TASKS} | {f"mask_{t}" for t in ALL_TASKS}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


@dataclass
class ResidualizationBundle:
    covariate_cols: List[str]
    shared_feature_cols: List[str]
    used_dummy_shared_feature: bool
    cov_imputer: SimpleImputer | None
    cov_scaler: StandardScaler | None
    shared_imputer: SimpleImputer | None
    shared_scaler: StandardScaler | None
    x_beta: np.ndarray | None
    y_beta: Dict[str, np.ndarray]
    reg_target_stats: Dict[str, Dict[str, float]]


@dataclass
class PreparedSplit:
    x_shared_train: np.ndarray
    x_shared_val: np.ndarray
    x_shared_test: np.ndarray
    x_cov_train: np.ndarray
    x_cov_val: np.ndarray
    x_cov_test: np.ndarray
    y_train: Dict[str, np.ndarray]
    y_val: Dict[str, np.ndarray]
    y_test: Dict[str, np.ndarray]
    y_cov_val: Dict[str, np.ndarray]
    y_cov_test: Dict[str, np.ndarray]
    m_train: Dict[str, np.ndarray]
    m_val: Dict[str, np.ndarray]
    m_test: Dict[str, np.ndarray]
    groups_train: np.ndarray
    groups_val: np.ndarray
    groups_test: np.ndarray
    prep: ResidualizationBundle


class MTLDataset(Dataset):
    def __init__(self, x_shared: np.ndarray, x_cov: np.ndarray, y: Dict[str, np.ndarray], m: Dict[str, np.ndarray]):
        self.x_shared = torch.tensor(x_shared, dtype=torch.float32)
        self.x_cov = torch.tensor(x_cov, dtype=torch.float32)
        self.y = {k: torch.tensor(v, dtype=torch.float32) for k, v in y.items()}
        self.m = {k: torch.tensor(v, dtype=torch.float32) for k, v in m.items()}

    def __len__(self) -> int:
        return self.x_shared.shape[0]

    def __getitem__(self, idx: int):
        return {
            "x_shared": self.x_shared[idx],
            "x_cov": self.x_cov[idx],
            "y": {k: v[idx] for k, v in self.y.items()},
            "m": {k: v[idx] for k, v in self.m.items()},
        }


class SharedBottomMTL(nn.Module):
    def __init__(self, shared_input_dim: int, covar_input_dim: int, hidden_dim: int, shared_dim: int, dropout: float):
        super().__init__()
        self.covar_input_dim = covar_input_dim
        self.trunk = nn.Sequential(
            nn.Linear(shared_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim),
            nn.ReLU(),
            nn.BatchNorm1d(shared_dim),
            nn.Dropout(dropout),
        )
        reg_head_in = shared_dim
        cls_head_in = shared_dim + covar_input_dim

        self.reg_heads = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(reg_head_in, max(shared_dim // 2, 16)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(shared_dim // 2, 16), 1),
            ) for t in REG_TASKS
        })
        self.cls_heads = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(cls_head_in, max(cls_head_in // 2, 16)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(cls_head_in // 2, 16), 1),
            ) for t in CLS_TASKS
        })

    def forward(self, x_shared: torch.Tensor, x_cov: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.trunk(x_shared)
        if self.covar_input_dim > 0:
            z_cls = torch.cat([z, x_cov], dim=1)
        else:
            z_cls = z
        out: Dict[str, torch.Tensor] = {}
        for t, head in self.reg_heads.items():
            out[t] = head(z).squeeze(1)
        for t, head in self.cls_heads.items():
            out[t] = head(z_cls).squeeze(1)
        return out


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(brier_score_loss(y_true, y_prob))


def _make_dummy_feature(df: pd.DataFrame, name: str = "__dummy_shared_feature__") -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out[name] = 0.0
    return out


def _impute_df(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, SimpleImputer | None]:
    if train_df.shape[1] == 0:
        return train_df.copy(), eval_df.copy(), None

    all_na_cols = train_df.columns[train_df.isna().all()].tolist()
    keep_cols = [c for c in train_df.columns if c not in all_na_cols]

    if len(keep_cols) == 0:
        tr = pd.DataFrame(0.0, index=train_df.index, columns=train_df.columns)
        ev = pd.DataFrame(0.0, index=eval_df.index, columns=eval_df.columns)
        return tr, ev, None

    imp = SimpleImputer(strategy="median")
    tr_imp = imp.fit_transform(train_df[keep_cols])
    ev_imp = imp.transform(eval_df[keep_cols])

    tr = pd.DataFrame(tr_imp, columns=keep_cols, index=train_df.index)
    ev = pd.DataFrame(ev_imp, columns=keep_cols, index=eval_df.index)

    for c in all_na_cols:
        tr[c] = 0.0
        ev[c] = 0.0

    tr = tr[train_df.columns]
    ev = ev[eval_df.columns]
    return tr, ev, imp


def _scale_df(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler | None]:
    if train_df.shape[1] == 0:
        return train_df.copy(), eval_df.copy(), None
    scl = StandardScaler()
    tr = pd.DataFrame(scl.fit_transform(train_df), columns=train_df.columns, index=train_df.index)
    ev = pd.DataFrame(scl.transform(eval_df), columns=eval_df.columns, index=eval_df.index)
    return tr, ev, scl


def _design_with_intercept(cov_df: pd.DataFrame) -> np.ndarray:
    if cov_df.shape[1] == 0:
        return np.ones((cov_df.shape[0], 1), dtype=float)
    return np.column_stack([np.ones(len(cov_df)), cov_df.to_numpy(dtype=float)])


def _residualize_features_against_covars(
    train_cov: pd.DataFrame,
    eval_cov: pd.DataFrame,
    train_x: pd.DataFrame,
    eval_x: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None]:
    if train_x.shape[1] == 0:
        return train_x.copy(), eval_x.copy(), None
    if train_cov.shape[1] == 0:
        return train_x.copy(), eval_x.copy(), None

    ctr = _design_with_intercept(train_cov)
    cev = _design_with_intercept(eval_cov)
    xtr = train_x.to_numpy(dtype=float)
    xev = eval_x.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(ctr, xtr, rcond=None)
    xtr_res = xtr - ctr @ beta
    xev_res = xev - cev @ beta
    tr = pd.DataFrame(xtr_res, columns=train_x.columns, index=train_x.index)
    ev = pd.DataFrame(xev_res, columns=eval_x.columns, index=eval_x.index)
    return tr, ev, beta


def _residualize_continuous_target(
    train_cov: pd.DataFrame,
    eval_cov: pd.DataFrame,
    y_train: pd.Series,
    y_eval: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if train_cov.shape[1] == 0:
        beta = np.array([0.0], dtype=float)
        ytr = np.asarray(y_train, dtype=float)
        yev = np.asarray(y_eval, dtype=float)
        return ytr, yev, np.zeros_like(yev, dtype=float), beta

    ctr = _design_with_intercept(train_cov)
    cev = _design_with_intercept(eval_cov)
    ytr = np.asarray(y_train, dtype=float)
    yev = np.asarray(y_eval, dtype=float)

    obs = np.isfinite(ytr)
    if obs.sum() == 0:
        beta = np.zeros(ctr.shape[1], dtype=float)
        ytr_cov = np.zeros_like(ytr, dtype=float)
        yev_cov = np.zeros_like(yev, dtype=float)
    else:
        beta, *_ = np.linalg.lstsq(ctr[obs], ytr[obs], rcond=None)
        ytr_cov = ctr @ beta
        yev_cov = cev @ beta

    ytr_res = ytr - ytr_cov
    yev_res = yev - yev_cov
    return ytr_res, yev_res, yev_cov, beta


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feature_cols = [
        c for c in df.columns
        if c not in ID_EXCLUDE and c not in LABEL_COLS and not c.startswith("raw_")
    ]
    if not TIME_AS_FEATURE and "time_num" in feature_cols:
        feature_cols.remove("time_num")
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce").copy()

    # Force-all mode: keep every numeric predictor that is not entirely missing.
    if FORCE_ALL_FEATURES:
        keep_cols = [c for c in x.columns if x[c].notna().any()]
    else:
        keep_cols = x.columns[x.isna().mean() <= MISSINGNESS_THRESHOLD].tolist()

    x = x[keep_cols]

    # Still remove constant columns because they add no information.
    keep_cols = [c for c in x.columns if x[c].nunique(dropna=True) > 1]
    x = x[keep_cols]

    # Optional cap if you want a larger-but-bounded shared input.
    if MAX_SHARED_FEATURES > 0 and x.shape[1] > MAX_SHARED_FEATURES:
        variances = x.var(axis=0, skipna=True).sort_values(ascending=False)
        keep_cols = variances.head(MAX_SHARED_FEATURES).index.tolist()
        x = x[keep_cols]

    return x, list(x.columns)


def split_train_val_test(df: pd.DataFrame, split_seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = df["IID"].astype(str).to_numpy()
    idx = np.arange(df.shape[0])
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=split_seed)
    trval_idx, test_idx = next(gss_outer.split(idx, groups=groups))

    groups_trval = groups[trval_idx]
    idx_trval = idx[trval_idx]
    inner_val_frac = VAL_SIZE / max(1e-8, (1.0 - TEST_SIZE))
    inner_val_frac = min(max(inner_val_frac, 0.05), 0.40)
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=inner_val_frac, random_state=split_seed)
    tr_inner, val_inner = next(gss_inner.split(idx_trval, groups=groups_trval))

    train_idx = idx_trval[tr_inner]
    val_idx = idx_trval[val_inner]
    return train_idx, val_idx, test_idx


def _prepare_shared_and_covariate_matrices(
    x_train: pd.DataFrame,
    x_eval: pd.DataFrame,
    covar_cols: List[str],
    other_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, SimpleImputer | None, StandardScaler | None, SimpleImputer | None, StandardScaler | None, np.ndarray | None, bool]:
    cov_tr = x_train[covar_cols].copy() if covar_cols else pd.DataFrame(index=x_train.index)
    cov_ev = x_eval[covar_cols].copy() if covar_cols else pd.DataFrame(index=x_eval.index)
    oth_tr = x_train[other_cols].copy() if other_cols else pd.DataFrame(index=x_train.index)
    oth_ev = x_eval[other_cols].copy() if other_cols else pd.DataFrame(index=x_eval.index)

    cov_tr, cov_ev, cov_imp = _impute_df(cov_tr, cov_ev)
    oth_tr, oth_ev, oth_imp = _impute_df(oth_tr, oth_ev)

    cov_tr, cov_ev, cov_scl = _scale_df(cov_tr, cov_ev)
    oth_tr, oth_ev, oth_scl = _scale_df(oth_tr, oth_ev)

    oth_tr_res, oth_ev_res, x_beta = _residualize_features_against_covars(cov_tr, cov_ev, oth_tr, oth_ev)

    used_dummy = False
    if oth_tr_res.shape[1] == 0:
        oth_tr_res = _make_dummy_feature(pd.DataFrame(index=x_train.index))
        oth_ev_res = _make_dummy_feature(pd.DataFrame(index=x_eval.index))
        used_dummy = True

    return oth_tr_res, oth_ev_res, cov_tr, cov_ev, cov_imp, cov_scl, oth_imp, oth_scl, x_beta, used_dummy


def prepare_split(df: pd.DataFrame, split_seed: int) -> PreparedSplit:
    x_df, feature_names_all = prepare_features(df)
    train_idx, val_idx, test_idx = split_train_val_test(df, split_seed)

    x_train_df = x_df.iloc[train_idx].copy()
    x_val_df = x_df.iloc[val_idx].copy()
    x_test_df = x_df.iloc[test_idx].copy()

    covar_cols, other_cols = split_feature_sets(df=x_df, predictor_cols=list(x_df.columns))
    if FORCE_ALL_FEATURES:
        # In force-all mode, push all non-covariate predictors into the shared branch.
        other_cols = [c for c in x_df.columns if c not in set(covar_cols)]

    x_shared_train_df, x_shared_val_df, x_cov_train_df, x_cov_val_df, cov_imp, cov_scl, shared_imp, shared_scl, x_beta, used_dummy = _prepare_shared_and_covariate_matrices(
        x_train_df, x_val_df, covar_cols, other_cols
    )
    x_shared_train_df2, x_shared_test_df, x_cov_train_df2, x_cov_test_df, _, _, _, _, _, used_dummy_test = _prepare_shared_and_covariate_matrices(
        x_train_df, x_test_df, covar_cols, other_cols
    )

    # keep train matrices from first pass and verify feature names stay aligned
    if list(x_shared_train_df.columns) != list(x_shared_train_df2.columns):
        raise RuntimeError("Shared feature columns mismatch between val and test preprocessing.")
    if list(x_cov_train_df.columns) != list(x_cov_train_df2.columns):
        raise RuntimeError("Covariate columns mismatch between val and test preprocessing.")
    if used_dummy != used_dummy_test:
        raise RuntimeError("Dummy shared feature handling mismatch between val and test preprocessing.")

    prep_bundle = ResidualizationBundle(
        covariate_cols=list(x_cov_train_df.columns),
        shared_feature_cols=list(x_shared_train_df.columns),
        used_dummy_shared_feature=used_dummy,
        cov_imputer=cov_imp,
        cov_scaler=cov_scl,
        shared_imputer=shared_imp,
        shared_scaler=shared_scl,
        x_beta=x_beta,
        y_beta={},
        reg_target_stats={},
    )

    y_train: Dict[str, np.ndarray] = {}
    y_val: Dict[str, np.ndarray] = {}
    y_test: Dict[str, np.ndarray] = {}
    y_cov_val: Dict[str, np.ndarray] = {}
    y_cov_test: Dict[str, np.ndarray] = {}
    m_train: Dict[str, np.ndarray] = {}
    m_val: Dict[str, np.ndarray] = {}
    m_test: Dict[str, np.ndarray] = {}

    train_cov_for_y = x_cov_train_df
    val_cov_for_y = x_cov_val_df
    test_cov_for_y = x_cov_test_df

    for task in ALL_TASKS:
        y_all = pd.to_numeric(df[f"y_{task}"], errors="coerce")
        m_all = pd.to_numeric(df[f"mask_{task}"], errors="coerce").fillna(0).astype(float)

        y_tr_raw = y_all.iloc[train_idx].reset_index(drop=True)
        y_val_raw = y_all.iloc[val_idx].reset_index(drop=True)
        y_te_raw = y_all.iloc[test_idx].reset_index(drop=True)
        m_tr = m_all.iloc[train_idx].to_numpy(dtype=float)
        m_va = m_all.iloc[val_idx].to_numpy(dtype=float)
        m_te = m_all.iloc[test_idx].to_numpy(dtype=float)

        if task in CLS_TASKS:
            y_tr = np.where(np.isfinite(y_tr_raw), y_tr_raw, 0.0)
            y_va = np.where(np.isfinite(y_val_raw), y_val_raw, 0.0)
            y_te = np.where(np.isfinite(y_te_raw), y_te_raw, 0.0)
            y_train[task] = (np.asarray(y_tr, dtype=float) > 0).astype(float)
            y_val[task] = (np.asarray(y_va, dtype=float) > 0).astype(float)
            y_test[task] = (np.asarray(y_te, dtype=float) > 0).astype(float)
            y_cov_val[task] = np.zeros_like(y_val[task], dtype=float)
            y_cov_test[task] = np.zeros_like(y_test[task], dtype=float)
        else:
            ytr_res, yva_res, yva_cov, ybeta = _residualize_continuous_target(
                train_cov_for_y,
                val_cov_for_y,
                y_tr_raw.reset_index(drop=True),
                y_val_raw.reset_index(drop=True),
            )
            ytr_res2, yte_res, yte_cov, ybeta2 = _residualize_continuous_target(
                train_cov_for_y,
                test_cov_for_y,
                y_tr_raw.reset_index(drop=True),
                y_te_raw.reset_index(drop=True),
            )
            if not np.allclose(ybeta, ybeta2, equal_nan=True):
                raise RuntimeError(f"Continuous target residualization beta mismatch for task={task}")
            if not np.allclose(ytr_res, ytr_res2, equal_nan=True):
                raise RuntimeError(f"Continuous training residual mismatch for task={task}")

            obs_train = (m_tr > 0) & np.isfinite(ytr_res)
            mu = float(np.nanmean(ytr_res[obs_train])) if obs_train.any() else 0.0
            sd = float(np.nanstd(ytr_res[obs_train])) if obs_train.any() else 1.0
            if (not np.isfinite(sd)) or sd <= 0:
                sd = 1.0

            prep_bundle.y_beta[task] = np.asarray(ybeta, dtype=float)
            prep_bundle.reg_target_stats[task] = {"mean": mu, "sd": sd}

            y_train[task] = np.nan_to_num((ytr_res - mu) / sd, nan=0.0)
            y_val[task] = np.nan_to_num((yva_res - mu) / sd, nan=0.0)
            y_test[task] = np.nan_to_num((yte_res - mu) / sd, nan=0.0)
            y_cov_val[task] = np.nan_to_num(yva_cov, nan=0.0)
            y_cov_test[task] = np.nan_to_num(yte_cov, nan=0.0)

        m_train[task] = m_tr
        m_val[task] = m_va
        m_test[task] = m_te

    return PreparedSplit(
        x_shared_train=x_shared_train_df.to_numpy(dtype=float),
        x_shared_val=x_shared_val_df.to_numpy(dtype=float),
        x_shared_test=x_shared_test_df.to_numpy(dtype=float),
        x_cov_train=x_cov_train_df.to_numpy(dtype=float),
        x_cov_val=x_cov_val_df.to_numpy(dtype=float),
        x_cov_test=x_cov_test_df.to_numpy(dtype=float),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        y_cov_val=y_cov_val,
        y_cov_test=y_cov_test,
        m_train=m_train,
        m_val=m_val,
        m_test=m_test,
        groups_train=df.iloc[train_idx]["IID"].astype(str).to_numpy(),
        groups_val=df.iloc[val_idx]["IID"].astype(str).to_numpy(),
        groups_test=df.iloc[test_idx]["IID"].astype(str).to_numpy(),
        prep=prep_bundle,
    )


def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = targets * probs + (1.0 - targets) * (1.0 - probs)
    return (1.0 - pt).pow(gamma) * bce


def multitask_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, object], task_weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses = {}
    total: torch.Tensor | None = None
    for task in REG_TASKS:
        mask = batch["m"][task] > 0
        if mask.sum() == 0:
            losses[task] = 0.0
            continue
        pred = outputs[task][mask]
        target = batch["y"][task][mask]
        loss_t = F.mse_loss(pred, target)
        total = loss_t * task_weights[task] if total is None else total + task_weights[task] * loss_t
        losses[task] = float(loss_t.detach().cpu())
    for task in CLS_TASKS:
        mask = batch["m"][task] > 0
        if mask.sum() == 0:
            losses[task] = 0.0
            continue
        logits = outputs[task][mask]
        target = batch["y"][task][mask]
        if CLASS_LOSS == "focal":
            loss_t = focal_bce_with_logits(logits, target, gamma=FOCAL_GAMMA).mean()
        else:
            loss_t = F.binary_cross_entropy_with_logits(logits, target)
        total = loss_t * task_weights[task] if total is None else total + task_weights[task] * loss_t
        losses[task] = float(loss_t.detach().cpu())
    if total is None:
        total = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
    return total, losses


def compute_task_weights(m_train: Dict[str, np.ndarray]) -> Dict[str, float]:
    if TASK_WEIGHTING == "uniform":
        return {task: 1.0 for task in ALL_TASKS}
    obs = {task: float(np.sum(np.asarray(m_train[task]) > 0)) for task in ALL_TASKS}
    inv = {task: 1.0 / max(obs[task], 1.0) for task in ALL_TASKS}
    s = sum(inv.values())
    return {task: inv[task] * len(ALL_TASKS) / max(s, 1e-12) for task in ALL_TASKS}


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    reg_stats: Dict[str, Dict[str, float]],
    y_cov_parts: Dict[str, np.ndarray],
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    model.eval()
    reg_pred = {t: [] for t in REG_TASKS}
    reg_true = {t: [] for t in REG_TASKS}
    cls_prob = {t: [] for t in CLS_TASKS}
    cls_true = {t: [] for t in CLS_TASKS}
    total_loss = 0.0
    n_batches = 0
    weights = {t: 1.0 for t in ALL_TASKS}
    reg_offsets = {t: 0 for t in REG_TASKS}

    with torch.no_grad():
        for batch in loader:
            batch = {
                "x_shared": batch["x_shared"].to(DEVICE),
                "x_cov": batch["x_cov"].to(DEVICE),
                "y": {k: v.to(DEVICE) for k, v in batch["y"].items()},
                "m": {k: v.to(DEVICE) for k, v in batch["m"].items()},
            }
            out = model(batch["x_shared"], batch["x_cov"])
            loss, _ = multitask_loss(out, batch, weights)
            total_loss += float(loss.detach().cpu())
            n_batches += 1

            batch_size = batch["x_shared"].shape[0]
            for t in REG_TASKS:
                mask = batch["m"][t] > 0
                if mask.sum() > 0:
                    mu = reg_stats[t]["mean"]
                    sd = reg_stats[t]["sd"]
                    start = reg_offsets[t]
                    end = start + batch_size
                    cov_part = y_cov_parts[t][start:end]
                    reg_offsets[t] = end
                    cov_obs = cov_part[mask.detach().cpu().numpy()]
                    pred_resid = out[t][mask].detach().cpu().numpy() * sd + mu
                    true_resid = batch["y"][t][mask].detach().cpu().numpy() * sd + mu
                    reg_pred[t].append(pred_resid + cov_obs)
                    reg_true[t].append(true_resid + cov_obs)
            for t in CLS_TASKS:
                mask = batch["m"][t] > 0
                if mask.sum() > 0:
                    cls_prob[t].append(torch.sigmoid(out[t][mask]).detach().cpu().numpy())
                    cls_true[t].append(batch["y"][t][mask].detach().cpu().numpy())

    metrics: Dict[str, Dict[str, float]] = {}
    for t in REG_TASKS:
        if len(reg_true[t]) == 0:
            metrics[t] = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
            continue
        y_true = np.concatenate(reg_true[t])
        y_pred = np.concatenate(reg_pred[t])
        metrics[t] = {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        }
    for t in CLS_TASKS:
        if len(cls_true[t]) == 0:
            metrics[t] = {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan")}
            continue
        y_true = np.concatenate(cls_true[t]).astype(int)
        y_prob = np.concatenate(cls_prob[t])
        metrics[t] = {
            "auroc": safe_auc(y_true, y_prob),
            "auprc": safe_auprc(y_true, y_prob),
            "brier": safe_brier(y_true, y_prob),
        }
    return total_loss / max(n_batches, 1), metrics


def train_one_split(df: pd.DataFrame, split_idx: int) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prep = prepare_split(df, SEED + split_idx)
    weights = compute_task_weights(prep.m_train)

    train_ds = MTLDataset(prep.x_shared_train, prep.x_cov_train, prep.y_train, prep.m_train)
    val_ds = MTLDataset(prep.x_shared_val, prep.x_cov_val, prep.y_val, prep.m_val)
    test_ds = MTLDataset(prep.x_shared_test, prep.x_cov_test, prep.y_test, prep.m_test)

    pin_memory = DEVICE.startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)

    model = SharedBottomMTL(
        shared_input_dim=prep.x_shared_train.shape[1],
        covar_input_dim=prep.x_cov_train.shape[1],
        hidden_dim=HIDDEN_DIM,
        shared_dim=SHARED_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=MIN_DELTA,
        threshold_mode="abs",
        min_lr=1e-6,
    )

    best_state = None
    best_val = math.inf
    best_epoch = -1
    wait = 0
    history_rows = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {
                "x_shared": batch["x_shared"].to(DEVICE),
                "x_cov": batch["x_cov"].to(DEVICE),
                "y": {k: v.to(DEVICE) for k, v in batch["y"].items()},
                "m": {k: v.to(DEVICE) for k, v in batch["m"].items()},
            }
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["x_shared"], batch["x_cov"])
            loss, batch_task_losses = multitask_loss(out, batch, weights)
            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss at split={split_idx}, epoch={epoch}; skipping batch.")
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += float(loss.detach().cpu())
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        val_loss, val_metrics = evaluate_model(model, val_loader, prep.prep.reg_target_stats, prep.y_cov_val)
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history_row = {"split": split_idx, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr}
        for task, mets in val_metrics.items():
            for k, v in mets.items():
                history_row[f"val_{task}_{k}"] = v
        history_rows.append(history_row)

        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if epoch >= MIN_EPOCHS and wait >= PATIENCE:
                print(f"[INFO] Early stopping at split={split_idx}, epoch={epoch}, best_epoch={best_epoch}, best_val={best_val:.6f}")
                break

    if best_state is None:
        raise RuntimeError("Training failed: no best state captured.")

    model.load_state_dict(best_state)
    test_loss, test_metrics = evaluate_model(model, test_loader, prep.prep.reg_target_stats, prep.y_cov_test)

    torch.save(best_state, OUT_DIR / f"split_{split_idx:02d}_best_model.pt")

    pd.DataFrame({
        "shared_feature": prep.prep.shared_feature_cols,
        "is_dummy_shared_feature": [prep.prep.used_dummy_shared_feature] * len(prep.prep.shared_feature_cols),
    }).to_csv(OUT_DIR / f"split_{split_idx:02d}_shared_feature_manifest.csv", index=False)

    pd.DataFrame({
        "covariate": prep.prep.covariate_cols,
    }).to_csv(OUT_DIR / f"split_{split_idx:02d}_covariate_manifest.csv", index=False)

    pd.DataFrame([
        {"task": t, "mean_resid": s["mean"], "sd_resid": s["sd"]} for t, s in prep.prep.reg_target_stats.items()
    ]).to_csv(OUT_DIR / f"split_{split_idx:02d}_regression_target_scaling.csv", index=False)

    preprocess_rows = []
    for task in ALL_TASKS:
        preprocess_rows.append({
            "task": task,
            "task_type": "regression" if task in REG_TASKS else "classification",
            "n_train_observed": int(np.sum(prep.m_train[task] > 0)),
            "n_val_observed": int(np.sum(prep.m_val[task] > 0)),
            "n_test_observed": int(np.sum(prep.m_test[task] > 0)),
            "n_shared_features": int(len(prep.prep.shared_feature_cols)),
            "n_covariates": int(len(prep.prep.covariate_cols)),
            "used_dummy_shared_feature": int(prep.prep.used_dummy_shared_feature),
        })
    pd.DataFrame(preprocess_rows).to_csv(OUT_DIR / f"split_{split_idx:02d}_preprocess_summary.csv", index=False)

    summary = {
        "split": split_idx,
        "n_train_rows": int(prep.x_shared_train.shape[0]),
        "n_val_rows": int(prep.x_shared_val.shape[0]),
        "n_test_rows": int(prep.x_shared_test.shape[0]),
        "n_train_subjects": int(len(np.unique(prep.groups_train))),
        "n_val_subjects": int(len(np.unique(prep.groups_val))),
        "n_test_subjects": int(len(np.unique(prep.groups_test))),
        "n_shared_features": int(len(prep.prep.shared_feature_cols)),
        "n_covariates": int(len(prep.prep.covariate_cols)),
        "used_dummy_shared_feature": int(prep.prep.used_dummy_shared_feature),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "test_loss": float(test_loss),
    }
    for task in ALL_TASKS:
        summary[f"train_obs_{task}"] = int(np.sum(prep.m_train[task] > 0))
        summary[f"val_obs_{task}"] = int(np.sum(prep.m_val[task] > 0))
        summary[f"test_obs_{task}"] = int(np.sum(prep.m_test[task] > 0))
    for task, mets in test_metrics.items():
        for k, v in mets.items():
            summary[f"test_{task}_{k}"] = float(v)

    history_df = pd.DataFrame(history_rows)
    task_weight_df = pd.DataFrame([{"task": k, "weight": v} for k, v in weights.items()])
    preprocess_df = pd.DataFrame(preprocess_rows)
    return summary, history_df, task_weight_df, preprocess_df


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input multitask table not found: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, sep="\t")
    if "IID" not in df.columns:
        raise ValueError("IID column is required.")

    split_rows = []
    history_frames = []
    task_weight_frames = []
    preprocess_frames = []

    for split_idx in range(1, N_SPLITS + 1):
        print(f"[INFO] Starting split {split_idx}/{N_SPLITS}")
        summary, history_df, task_weight_df, preprocess_df = train_one_split(df, split_idx)
        split_rows.append(summary)
        history_frames.append(history_df)
        task_weight_df.insert(0, "split", split_idx)
        preprocess_df.insert(0, "split", split_idx)
        task_weight_frames.append(task_weight_df)
        preprocess_frames.append(preprocess_df)
        history_df.to_csv(OUT_DIR / f"split_{split_idx:02d}_history.csv", index=False)

    metrics = pd.DataFrame(split_rows)
    metrics.to_csv(OUT_DIR / f"multitask_shared_bottom_{FAMILY}_metrics.csv", index=False)
    pd.concat(history_frames, axis=0, ignore_index=True).to_csv(
        OUT_DIR / f"multitask_shared_bottom_{FAMILY}_history.csv", index=False
    )
    pd.concat(task_weight_frames, axis=0, ignore_index=True).to_csv(
        OUT_DIR / f"multitask_shared_bottom_{FAMILY}_task_weights.csv", index=False
    )
    pd.concat(preprocess_frames, axis=0, ignore_index=True).to_csv(
        OUT_DIR / f"multitask_shared_bottom_{FAMILY}_preprocess.csv", index=False
    )

    agg_rows = []
    for col in metrics.columns:
        if col in {"split"}:
            continue
        if pd.api.types.is_numeric_dtype(metrics[col]):
            agg_rows.append({
                "metric": col,
                "mean": float(pd.to_numeric(metrics[col], errors="coerce").mean()),
                "sd": float(pd.to_numeric(metrics[col], errors="coerce").std()),
            })
    pd.DataFrame(agg_rows).to_csv(OUT_DIR / f"multitask_shared_bottom_{FAMILY}_summary.csv", index=False)

    with open(OUT_DIR / f"multitask_shared_bottom_{FAMILY}_settings.json", "w") as f:
        json.dump({
            "input_path": str(INPUT_PATH),
            "family": FAMILY,
            "run_tag": RUN_TAG,
            "output_dir": str(OUT_DIR),
            "seed": SEED,
            "n_splits": N_SPLITS,
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "hidden_dim": HIDDEN_DIM,
            "shared_dim": SHARED_DIM,
            "patience": PATIENCE,
            "min_epochs": MIN_EPOCHS,
            "min_delta": MIN_DELTA,
            "missingness_threshold": MISSINGNESS_THRESHOLD,
            "force_all_features": FORCE_ALL_FEATURES,
            "max_shared_features": MAX_SHARED_FEATURES,
            "class_loss": CLASS_LOSS,
            "focal_gamma": FOCAL_GAMMA,
            "time_as_feature": TIME_AS_FEATURE,
            "task_weighting": TASK_WEIGHTING,
            "device": DEVICE,
            "covariate_strategy": {
                "feature_split": "covariates vs non_covariates using split_feature_sets",
                "shared_input": "scaled residualized non_covariate features",
                "classification_input": "shared trunk output plus scaled covariates",
                "continuous_targets": "training-only covariate residualization, then standardization",
                "classification_targets": "binary observed labels with masks",
            },
        }, f, indent=2)

    print("[INFO] Multitask v3 outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
