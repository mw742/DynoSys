#!/usr/bin/env python3
from pathlib import Path
import json
import os

import pandas as pd
import numpy as np

ROOT = Path("/gpfs/group/ehlers/mwei/")

PHENO = os.environ.get("PHENO", "externalizing")
TASK_TYPE = os.environ.get("TASK_TYPE", "continuous")

if TASK_TYPE != "continuous":
    raise SystemExit(f"[ERROR] 03_build_forward_tables_generic.py is only for continuous phenotypes. Got TASK_TYPE={TASK_TYPE}")

INPUT_DIR = ROOT / "results" / PHENO / "unified_inputs"
OUT_DIR = ROOT / "results" / PHENO / "forward_inputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TABLES = [
    f"{PHENO}_unified_pca.tsv",
    f"{PHENO}_unified_weighted.tsv",
    f"{PHENO}_unified_cluster.tsv",
    f"{PHENO}_unified_all.tsv",
]

TARGET = f"{PHENO}_t"

for table in TABLES:
    path = INPUT_DIR / table
    if not path.exists():
        print(f"[WARN] Missing table, skipping: {path}")
        continue

    df = pd.read_csv(path, sep="\t")
    df["IID"] = df["IID"].astype(str)
    df = df.sort_values(["IID", "time_num"]).reset_index(drop=True)

    if TARGET not in df.columns:
        print(f"[WARN] Target {TARGET} not found in {table}, skipping")
        continue

    df[f"{PHENO}_t_next"] = df.groupby("IID")[TARGET].shift(-1)
    df[f"delta_{PHENO}_next"] = df[f"{PHENO}_t_next"] - df[TARGET]
    df["has_next_wave"] = df.groupby("IID")[TARGET].shift(-1).notna().astype(int)
    df["future_gap"] = df.groupby("IID")["time_num"].shift(-1) - df["time_num"]

    out = df.loc[df["has_next_wave"] == 1].copy()

    if out.shape[0] == 0:
        print(f"[WARN] No next-wave rows for {table}, skipping")
        continue

    sd = pd.to_numeric(out[f"delta_{PHENO}_next"], errors="coerce").std(skipna=True)
    if pd.isna(sd) or sd <= 0:
        sd = 1.0

    out["worsen_event_0p5sd"] = (out[f"delta_{PHENO}_next"] >= 0.5 * sd).astype(int)
    out["worsen_event_1sd"] = (out[f"delta_{PHENO}_next"] >= 1.0 * sd).astype(int)
    out[f"high_{PHENO}_next_q75"] = (
        out[f"{PHENO}_t_next"] >= out[f"{PHENO}_t_next"].quantile(0.75)
    ).astype(int)

    stem = table.replace(".tsv", "")
    out.to_csv(OUT_DIR / f"{stem}_forward.tsv", sep="\t", index=False)

    summary = {
        "pheno": PHENO,
        "source_table": table,
        "n_rows": int(out.shape[0]),
        "n_subjects": int(out["IID"].nunique()),
        "mean_future_gap": float(pd.to_numeric(out["future_gap"], errors="coerce").mean()),
        "sd_delta_next": float(sd),
        "worsen_event_0p5sd_rate": float(out["worsen_event_0p5sd"].mean()),
        "worsen_event_1sd_rate": float(out["worsen_event_1sd"].mean()),
        f"high_{PHENO}_next_q75_rate": float(out[f"high_{PHENO}_next_q75"].mean()),
    }
    with open(OUT_DIR / f"{stem}_forward_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

with open(OUT_DIR / "forward_build_settings.json", "w") as f:
    json.dump({
        "pheno": PHENO,
        "task_type": TASK_TYPE,
        "tables": TABLES,
        "target": TARGET
    }, f, indent=2)

print("[INFO] Forward tables written to:", OUT_DIR)
