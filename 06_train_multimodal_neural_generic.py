#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    raise SystemExit(
        "[ERROR] PyTorch is not available in environment. "
        "Please install torch in my_env first."
    ) from e


ROOT = Path("/gpfs/group/ehlers/mwei/")
PHENO = os.environ.get("PHENO", "externalizing")
TASK_TYPE = os.environ.get("TASK_TYPE", "continuous")

UNIFIED_DIR = ROOT / "results" / PHENO / "unified_inputs"
FORWARD_DIR = ROOT / "results" / PHENO / "forward_inputs"
OUT_DIR = ROOT / "results" / PHENO / "advanced_neural"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TABLE = (
    f"{PHENO}_unified_weighted_forward.tsv"
    if TASK_TYPE == "continuous"
    else f"{PHENO}_unified_weighted.tsv"
)
DEFAULT_TARGET = (
    os.environ.get("NN_TARGET", f"high_{PHENO}_next_q75")
    if TASK_TYPE == "continuous"
    else os.environ.get("NN_TARGET", "binary_event")
)

EXCLUDE_COMMON = {
    "FID", "IID", "src_subject_id", "eventname", "time_year"
}
EXCLUDE_CONT = {
    f"{PHENO}_t", f"{PHENO}_t_next", f"delta_{PHENO}", f"{PHENO}_lag",
    f"delta_{PHENO}_next", "has_next_wave", "future_gap",
    "worsen_event_0p5sd", "worsen_event_1sd", f"high_{PHENO}_next_q75"
}
EXCLUDE_BIN = {
    "time", "event", "binary_event", "baseline_age_year", "baseline_age_month",
    "anchor_offset_month", "anchor_age_month", "observed_by_anchor"
}

def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))

def safe_auprc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_prob))

def safe_corr(y_true, y_pred):
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(y_true[ok], y_pred[ok])[0, 1])

def make_modalities(columns):
    mods = {
        "foundation": [],
        "environment": [],
        "family": [],
        "physical": [],
        "technology": [],
        "neuro": [],
        "brain": [],
        "time": [],
        "other": [],
    }
    for c in columns:
        if c in {"externalizing_prs", "internalizing_youth_prs", "internalizing_combo_prs",
                 "anxiety_prs", "mdd_prs", "neuroticism_prs",
                 "anxiety_prs_z", "mdd_prs_z", "neuroticism_prs_z",
                 "alcohol_prs", "any_prs", "cannabis_prs", "nicotine_prs",
                 "CNT", "CNT2"}:
            mods["foundation"].append(c)
        elif c.startswith("environment_"):
            mods["environment"].append(c)
        elif c.startswith("general_family_context_"):
            mods["family"].append(c)
        elif c.startswith("physical_health_"):
            mods["physical"].append(c)
        elif c.startswith("technology_"):
            mods["technology"].append(c)
        elif c.startswith("neurorecognition_"):
            mods["neuro"].append(c)
        elif c.startswith("brain_") or c.startswith("smri_"):
            mods["brain"].append(c)
        elif c == "time_num":
            mods["time"].append(c)
        else:
            mods["other"].append(c)
    return {k: v for k, v in mods.items() if len(v) > 0}

class MultiModalDataset(Dataset):
    def __init__(self, x_dict, y):
        self.x_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in x_dict.items()}
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        batch = {k: v[idx] for k, v in self.x_dict.items()}
        batch["y"] = self.y[idx]
        return batch

class Branch(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class MultiModalNet(nn.Module):
    def __init__(self, modality_dims, hidden_dim=32, fusion_dim=64, dropout=0.2):
        super().__init__()
        self.branches = nn.ModuleDict({
            name: Branch(dim, hidden_dim, dropout)
            for name, dim in modality_dims.items()
        })
        total_dim = hidden_dim * len(modality_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, x_dict):
        outs = [self.branches[name](x_dict[name]) for name in self.branches]
        fused = torch.cat(outs, dim=1)
        return self.fusion(fused).squeeze(1)

def prepare_data(df, target):
    y = pd.to_numeric(df[target], errors="coerce")
    groups = df["IID"].astype(str)

    exclude = set(EXCLUDE_COMMON)
    if TASK_TYPE == "continuous":
        exclude |= EXCLUDE_CONT
    else:
        exclude |= EXCLUDE_BIN

    feature_cols = [c for c in df.columns if c not in exclude and c != target]
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce").copy()

    ok = y.notna()
    x = x.loc[ok].reset_index(drop=True)
    y = y.loc[ok].reset_index(drop=True)
    groups = groups.loc[ok].reset_index(drop=True)

    x = x.loc[:, x.isna().mean() <= 0.70].copy()
    x = x.loc[:, x.nunique(dropna=True) > 1].copy()

    modalities = make_modalities(x.columns)
    return x, y, groups, modalities

def fit_transform_modalities(x_train, x_val, x_test, modalities):
    out_train, out_val, out_test = {}, {}, {}
    for name, cols in modalities.items():
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        tr = scl.fit_transform(imp.fit_transform(x_train[cols]))
        va = scl.transform(imp.transform(x_val[cols]))
        te = scl.transform(imp.transform(x_test[cols]))
        out_train[name] = tr
        out_val[name] = va
        out_test[name] = te
    return out_train, out_val, out_test

def train_one_split(df, target, split_idx, device, args):
    x, y, groups, modalities = prepare_data(df, target)

    outer = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed + split_idx)
    tr_idx, te_idx = next(outer.split(x, y, groups))

    x_train_all = x.iloc[tr_idx].reset_index(drop=True)
    y_train_all = y.iloc[tr_idx].reset_index(drop=True)
    g_train_all = groups.iloc[tr_idx].reset_index(drop=True)

    x_test = x.iloc[te_idx].reset_index(drop=True)
    y_test = y.iloc[te_idx].reset_index(drop=True)
    g_test = groups.iloc[te_idx].reset_index(drop=True)

    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed + 1000 + split_idx)
    tr2_idx, va_idx = next(inner.split(x_train_all, y_train_all, g_train_all))

    x_train = x_train_all.iloc[tr2_idx].reset_index(drop=True)
    y_train = y_train_all.iloc[tr2_idx].reset_index(drop=True)
    x_val = x_train_all.iloc[va_idx].reset_index(drop=True)
    y_val = y_train_all.iloc[va_idx].reset_index(drop=True)

    xtr, xva, xte = fit_transform_modalities(x_train, x_val, x_test, modalities)

    ds_tr = MultiModalDataset(xtr, y_train.values)
    ds_va = MultiModalDataset(xva, y_val.values)
    ds_te = MultiModalDataset(xte, y_test.values)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False)

    modality_dims = {k: v.shape[1] for k, v in xtr.items()}
    model = MultiModalNet(
        modality_dims=modality_dims,
        hidden_dim=args.hidden_dim,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
    ).to(device)

    if TASK_TYPE == "continuous":
        criterion = nn.MSELoss()
    else:
        pos = max(1, int(y_train.sum()))
        neg = max(1, int(len(y_train) - y_train.sum()))
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_score = -np.inf
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in dl_tr:
            yb = batch["y"].to(device)
            xb = {k: v.to(device) for k, v in batch.items() if k != "y"}

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_pred, val_true = [], []
        with torch.no_grad():
            for batch in dl_va:
                yb = batch["y"].cpu().numpy()
                xb = {k: v.to(device) for k, v in batch.items() if k != "y"}
                out = model(xb).cpu().numpy()
                if TASK_TYPE == "binary":
                    out = 1.0 / (1.0 + np.exp(-out))
                val_pred.append(out)
                val_true.append(yb)

        val_pred = np.concatenate(val_pred)
        val_true = np.concatenate(val_true)

        if TASK_TYPE == "continuous":
            score = -float(np.sqrt(mean_squared_error(val_true, val_pred)))
        else:
            score = safe_auc(val_true.astype(int), val_pred)
            if np.isnan(score):
                score = -np.inf

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    model.eval()

    test_pred, test_true = [], []
    with torch.no_grad():
        for batch in dl_te:
            yb = batch["y"].cpu().numpy()
            xb = {k: v.to(device) for k, v in batch.items() if k != "y"}
            out = model(xb).cpu().numpy()
            if TASK_TYPE == "binary":
                out = 1.0 / (1.0 + np.exp(-out))
            test_pred.append(out)
            test_true.append(yb)

    test_pred = np.concatenate(test_pred)
    test_true = np.concatenate(test_true)

    if TASK_TYPE == "continuous":
        metrics = {
            "split": split_idx,
            "n_train_subjects": int(g_train_all.nunique()),
            "n_test_subjects": int(g_test.nunique()),
            "n_test_rows": int(len(y_test)),
            "rmse": float(np.sqrt(mean_squared_error(test_true, test_pred))),
            "mae": float(mean_absolute_error(test_true, test_pred)),
            "r2": float(r2_score(test_true, test_pred)),
            "corr": safe_corr(test_true.astype(float), test_pred.astype(float)),
        }
    else:
        metrics = {
            "split": split_idx,
            "n_train_subjects": int(g_train_all.nunique()),
            "n_test_subjects": int(g_test.nunique()),
            "n_test_rows": int(len(y_test)),
            "event_rate_test": float(np.mean(test_true)),
            "auroc": safe_auc(test_true.astype(int), test_pred),
            "auprc": safe_auprc(test_true.astype(int), test_pred),
            "brier": float(brier_score_loss(test_true.astype(int), test_pred)),
        }

    pred_df = pd.DataFrame({
        "IID": g_test.values,
        "y_true": test_true,
        "y_pred": test_pred,
        "split": split_idx,
    })

    return metrics, pred_df, modalities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--fusion_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    input_dir = FORWARD_DIR if args.table.endswith("_forward.tsv") else UNIFIED_DIR
    df = pd.read_csv(input_dir / args.table, sep="\t")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_metrics = []
    all_preds = []
    saved_modalities = None

    for split_idx in range(1, args.n_splits + 1):
        metrics, pred_df, modalities = train_one_split(df, args.target, split_idx, device, args)
        all_metrics.append(metrics)
        all_preds.append(pred_df)
        saved_modalities = modalities

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.concat(all_preds, axis=0, ignore_index=True)

    stem = f"{Path(args.table).stem}_{args.target}_multimodal_nn"
    metrics_df.to_csv(OUT_DIR / f"{stem}_metrics.csv", index=False)
    preds_df.to_csv(OUT_DIR / f"{stem}_predictions.csv", index=False)

    summary = {
        "pheno": PHENO,
        "task_type": TASK_TYPE,
        "table": args.table,
        "target": args.target,
        "device": str(device),
        "n_splits": args.n_splits,
        "modalities": saved_modalities,
    }
    if TASK_TYPE == "continuous":
        summary["mean_rmse"] = float(metrics_df["rmse"].mean())
        summary["mean_r2"] = float(metrics_df["r2"].mean())
        summary["mean_corr"] = float(metrics_df["corr"].mean())
    else:
        summary["mean_auroc"] = float(metrics_df["auroc"].mean())
        summary["mean_auprc"] = float(metrics_df["auprc"].mean())
        summary["mean_brier"] = float(metrics_df["brier"].mean())

    with open(OUT_DIR / f"{stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[INFO] Neural outputs written to:", OUT_DIR)
    print("[INFO] Summary:", summary)

if __name__ == "__main__":
    main()
