#!/usr/bin/env python3
"""
Build a shared multi-task table from the per-phenotype unified inputs produced by
00_build_unified_tables.py.

Output
------
/gpfs/group/ehlers/mwei/<family>/multitask_<family>.tsv
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/gpfs/group/ehlers/mwei/")
RESULTS_ROOT = ROOT / "results"
OUT_ROOT = RESULTS_ROOT / "MTL"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

FAMILY = os.environ.get("FAMILY", "weighted").lower()
INCLUDE_TIME = os.environ.get("INCLUDE_TIME", "1") == "1"
DROP_HIGH_MISSING = float(os.environ.get("DROP_HIGH_MISSING", "0.98"))
KEEP_RAW_LABEL_COLUMNS = os.environ.get("KEEP_RAW_LABEL_COLUMNS", "0") == "1"

ID_COLS = ["FID", "IID", "src_subject_id", "eventname", "time_num", "time_year"]
REG_TASKS = ["externalizing", "internalizing"]
BIN_TASKS = ["alcohol", "nicotine", "cannabis", "any"]
ALL_TASKS = REG_TASKS + BIN_TASKS


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    return pd.read_csv(path, sep="\t")


TARGET_SPECS: Dict[str, Dict[str, object]] = {
    "externalizing": {
        "type": "regression",
        "table": RESULTS_ROOT / "externalizing" / "unified_inputs" / f"externalizing_unified_{FAMILY}.tsv",
        "label_col": "externalizing_t",
        "obs_mask_col": None,
    },
    "internalizing": {
        "type": "regression",
        "table": RESULTS_ROOT / "internalizing" / "unified_inputs" / f"internalizing_unified_{FAMILY}.tsv",
        "label_col": "internalizing_t",
        "obs_mask_col": None,
    },
    "alcohol": {
        "type": "classification",
        "table": RESULTS_ROOT / "alcohol" / "unified_inputs" / f"alcohol_unified_{FAMILY}.tsv",
        "label_col": "binary_event",
        "obs_mask_col": "observed_by_anchor",
    },
    "nicotine": {
        "type": "classification",
        "table": RESULTS_ROOT / "nicotine" / "unified_inputs" / f"nicotine_unified_{FAMILY}.tsv",
        "label_col": "binary_event",
        "obs_mask_col": "observed_by_anchor",
    },
    "cannabis": {
        "type": "classification",
        "table": RESULTS_ROOT / "cannabis" / "unified_inputs" / f"cannabis_unified_{FAMILY}.tsv",
        "label_col": "binary_event",
        "obs_mask_col": "observed_by_anchor",
    },
    "any": {
        "type": "classification",
        "table": RESULTS_ROOT / "any" / "unified_inputs" / f"any_unified_{FAMILY}.tsv",
        "label_col": "binary_event",
        "obs_mask_col": "observed_by_anchor",
    },
}


EXCLUDE_FEATURES = {
    # common ids
    "FID", "IID", "src_subject_id", "eventname", "time_num", "time_year",
    # regression labels / helpers
    "externalizing_t", "internalizing_t", "delta_externalizing", "externalizing_lag",
    "delta_internalizing", "internalizing_lag",
    # binary labels / helpers
    "binary_event", "time", "event", "observed_by_anchor",
    "baseline_age_year", "baseline_age_month", "anchor_offset_month", "anchor_age_month",
}


def choose_id_frame(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for task, df in tables.items():
        keep = [c for c in ID_COLS if c in df.columns]
        tmp = df[keep].copy()
        tmp["IID"] = tmp["IID"].astype(str)
        if "eventname" in tmp.columns:
            tmp["eventname"] = tmp["eventname"].astype(str)
        frames.append(tmp)

    base = pd.concat(frames, axis=0, ignore_index=True)
    base = base.drop_duplicates(subset=[c for c in ["IID", "eventname", "time_num"] if c in base.columns])
    base = base.sort_values([c for c in ["IID", "time_num", "eventname"] if c in base.columns]).reset_index(drop=True)
    if "FID" not in base.columns:
        base["FID"] = base["IID"]
    if "src_subject_id" not in base.columns:
        base["src_subject_id"] = base["IID"]
    return base



def task_label_frame(task: str, df: pd.DataFrame) -> pd.DataFrame:
    spec = TARGET_SPECS[task]
    label_col = str(spec["label_col"])
    obs_col = spec["obs_mask_col"]

    cols = [c for c in ["IID", "eventname", "time_num", "time_year"] if c in df.columns]
    out = df[cols].copy()
    out["IID"] = out["IID"].astype(str)
    out[f"y_{task}"] = pd.to_numeric(df[label_col], errors="coerce")

    if spec["type"] == "regression":
        out[f"mask_{task}"] = out[f"y_{task}"].notna().astype(int)
    else:
        if obs_col is not None and obs_col in df.columns:
            out[f"mask_{task}"] = pd.to_numeric(df[obs_col], errors="coerce").fillna(0).astype(int)
        else:
            out[f"mask_{task}"] = out[f"y_{task}"].notna().astype(int)
        out.loc[out[f"mask_{task}"] == 0, f"y_{task}"] = np.nan

    if KEEP_RAW_LABEL_COLUMNS:
        for c in [label_col, obs_col] if obs_col is not None else [label_col]:
            if c in df.columns:
                out[f"raw_{task}_{c}"] = df[c]
    return out



def extract_features(task: str, df: pd.DataFrame) -> pd.DataFrame:
    keep_ids = [c for c in ["IID", "eventname", "time_num"] if c in df.columns]
    feat_cols = [c for c in df.columns if c not in EXCLUDE_FEATURES]
    if not INCLUDE_TIME and "time_num" in feat_cols:
        feat_cols.remove("time_num")

    out = df[keep_ids + feat_cols].copy()
    out["IID"] = out["IID"].astype(str)

    rename_map = {}
    for c in feat_cols:
        # keep covariates / shared identifiers unprefixed if truly shared and not duplicated badly
        if c in {"sex", "age"} or c.startswith("PC") or c.startswith("site_") or c == "time_num":
            rename_map[c] = c
        else:
            rename_map[c] = f"{task}__{c}"
    out = out.rename(columns=rename_map)

    duplicated = out.columns[out.columns.duplicated()].tolist()
    if duplicated:
        out = out.loc[:, ~out.columns.duplicated()].copy()

    return out



def infer_domain(col: str) -> str:
    if col in {"sex", "age", "time_num"} or col.startswith("PC") or col.startswith("site_"):
        return "covariate"
    lowered = col.lower()
    if "__" in col:
        _, raw = col.split("__", 1)
    else:
        raw = col
    rl = raw.lower()
    if rl.endswith("_prs") or "prs" in rl:
        return "prs_foundation"
    if rl.startswith("environment_"):
        return "environment"
    if rl.startswith("general_family_context_"):
        return "general_family_context"
    if rl.startswith("physical_health_"):
        return "physical_health"
    if rl.startswith("technology_"):
        return "technology"
    if rl.startswith("neurorecognition_") or rl.startswith("neurocognition_"):
        return "neuro"
    if rl.startswith("brain_") or rl.startswith("smri_"):
        return "brain"
    return "other"



def main() -> None:
    loaded: Dict[str, pd.DataFrame] = {}
    for task, spec in TARGET_SPECS.items():
        df = read_table(Path(spec["table"]))
        if "IID" not in df.columns:
            raise ValueError(f"IID column missing in {spec['table']}")
        df["IID"] = df["IID"].astype(str)
        if "eventname" in df.columns:
            df["eventname"] = df["eventname"].astype(str)
        loaded[task] = df

    base = choose_id_frame(loaded)

    for task, df in loaded.items():
        ydf = task_label_frame(task, df)
        base = base.merge(ydf, on=[c for c in ["IID", "eventname", "time_num", "time_year"] if c in ydf.columns and c in base.columns], how="left")

    feature_frames: List[pd.DataFrame] = []
    for task, df in loaded.items():
        feature_frames.append(extract_features(task, df))

    out = base.copy()
    for feat_df in feature_frames:
        overlap = [c for c in feat_df.columns if c not in {"IID", "eventname", "time_num"} and c in out.columns]
        if overlap:
            dup_same = []
            for c in overlap:
                merged = out[["IID", "eventname", "time_num", c]].merge(
                    feat_df[["IID", "eventname", "time_num", c]],
                    on=["IID", "eventname", "time_num"],
                    how="inner",
                    suffixes=("_left", "_right"),
                )
                if merged.empty:
                    dup_same.append(c)
                else:
                    left = pd.to_numeric(merged[f"{c}_left"], errors="coerce")
                    right = pd.to_numeric(merged[f"{c}_right"], errors="coerce")
                    both = left.notna() & right.notna()
                    if both.sum() == 0 or np.allclose(left[both], right[both], equal_nan=True):
                        dup_same.append(c)
            feat_df = feat_df.drop(columns=dup_same, errors="ignore")
            overlap = [c for c in feat_df.columns if c not in {"IID", "eventname", "time_num"} and c in out.columns]
            if overlap:
                feat_df = feat_df.rename(columns={c: f"dup__{c}" for c in overlap})
        out = out.merge(feat_df, on=["IID", "eventname", "time_num"], how="left")

    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()

    numeric_exempt = {"FID", "IID", "src_subject_id", "eventname"}
    for c in out.columns:
        if c not in numeric_exempt:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    feature_cols = [
        c for c in out.columns
        if c not in numeric_exempt
        and not c.startswith("y_")
        and not c.startswith("mask_")
        and not c.startswith("raw_")
    ]
    if DROP_HIGH_MISSING < 1.0:
        miss = out[feature_cols].isna().mean()
        drop_cols = miss[miss > DROP_HIGH_MISSING].index.tolist()
        if drop_cols:
            out = out.drop(columns=drop_cols)
            feature_cols = [c for c in feature_cols if c not in drop_cols]

    task_rows = []
    for task in ALL_TASKS:
        task_rows.append({
            "task": task,
            "task_type": TARGET_SPECS[task]["type"],
            "n_observed": int(pd.to_numeric(out[f"mask_{task}"], errors="coerce").fillna(0).sum()),
            "n_nonmissing_y": int(out[f"y_{task}"].notna().sum()),
        })

    manifest = pd.DataFrame({
        "variable": feature_cols,
        "domain": [infer_domain(c) for c in feature_cols],
    })
    manifest["representation_family"] = FAMILY

    out_dir = OUT_ROOT / FAMILY
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out.sort_values([c for c in ["IID", "time_num", "eventname"] if c in out.columns]).reset_index(drop=True)
    out_path = out_dir / f"multitask_{FAMILY}.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    manifest.to_csv(out_dir / f"multitask_{FAMILY}_manifest.csv", index=False)

    summary = {
        "family": FAMILY,
        "include_time": INCLUDE_TIME,
        "n_rows": int(out.shape[0]),
        "n_subjects": int(out["IID"].nunique()),
        "n_features": int(len(feature_cols)),
        "tasks": task_rows,
        "source_tables": {task: str(spec["table"]) for task, spec in TARGET_SPECS.items()},
    }
    with open(out_dir / f"multitask_{FAMILY}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Wrote multi-task table: {out_path}")


if __name__ == "__main__":
    main()
