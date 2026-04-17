#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path("/gpfs/group/ehlers/mwei/")
MULTI = ROOT / "Multi"

PHENO = os.environ.get("PHENO", "externalizing")
TASK_TYPE = os.environ.get("TASK_TYPE", "continuous")  # continuous or binary

OUTDIR = MULTI / "results" / PHENO / "unified_inputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

WAVE_MAP = {
    "baseline_year_1_arm_1": 0,
    "1_year_follow_up_y_arm_1": 1,
    "2_year_follow_up_y_arm_1": 2,
    "3_year_follow_up_y_arm_1": 3,
    "4_year_follow_up_y_arm_1": 4,
}

BRAIN_ANALYSIS_TO_EVENT = {
    "baseline_primary_pc10_site": "baseline_year_1_arm_1",
    "landmark24_primary_pc10_site": "2_year_follow_up_y_arm_1",
    "landmark48_primary_pc10_site": "4_year_follow_up_y_arm_1",
}

CONT_OUTCOME_FILES = {
    "externalizing": ROOT / "OUT" / "externalizing_t_long.tsv",
    "internalizing": ROOT / "OUT" / "internalizing_t_long.tsv",
}

BIN_OUTCOME_FILES = {
    "alcohol": ROOT / "OUT" / "alcohol_survival.tsv",
    "any": ROOT / "OUT" / "any_survival.tsv",
    "cannabis": ROOT / "OUT" / "cannabis_survival.tsv",
    "nicotine": ROOT / "OUT" / "nicotine_survival.tsv",
}

BASELINE_AGE_FILE = ROOT / "core" / "abcd-general" / "get_covariates" / "individual_baseline_ages.csv"
COVARIATE_FILE = Path("/gpfs/group/ehlers/mwei/covariates_final.fixed.tsv")
COVARIATE_COLS = ["sex", "age"] + [f"PC{i}" for i in range(1, 11)]

PRS_MAP = {
    "externalizing": {
        "fam": ROOT / "EXT" / "PRSPC" / "ext_prs_eur" / "geno_subsets" / "imputed_qc_filtered_eur_unrelated.fam",
        "profile": ROOT / "EXT" / "PRSPC" / "ext_prs_eur" / "scores" / "externalizing_PRS_EUR_unrelated.profile",
        "score_col": "externalizing_prs",
    },
    "internalizing": {
        "fam": ROOT / "EXT" / "PRSPC" / "internalizing_prs_eur" / "geno_subsets" / "imputed_qc_filtered_eur_unrelated.fam",
        "profiles": {
            "internalizing_youth_prs": ROOT / "EXT" / "PRSPC" / "internalizing_prs_eur" / "scores" / "internalizing_youth_EUR_unrelated.profile",
            "anxiety_prs": ROOT / "EXT" / "PRSPC" / "internalizing_prs_eur" / "scores" / "anxiety_EUR_unrelated.profile",
            "mdd_prs": ROOT / "EXT" / "PRSPC" / "internalizing_prs_eur" / "scores" / "mdd_EUR_unrelated.profile",
            "neuroticism_prs": ROOT / "EXT" / "PRSPC" / "internalizing_prs_eur" / "scores" / "neuroticism_EUR_unrelated.profile",
        },
    },
    "alcohol": {
        "fam": ROOT / "EXT" / "PRSPC" / "sud_prs_eur" / "geno_subsets" / "imputed_qc_filtered_eur_unrelated.fam",
        "profile": ROOT / "EXT" / "PRSPC" / "sud_prs_eur" / "scores" / "alc_EUR_unrelated.profile",
        "score_col": "alcohol_prs",
    },
    "any": {
        "fam": ROOT / "EXT" / "PRSPC" / "sud_prs_eur" / "geno_subsets" / "imputed_qc_filtered_eur_unrelated.fam",
        "profile": ROOT / "EXT" / "PRSPC" / "sud_prs_eur" / "scores" / "any_EUR_unrelated.profile",
        "score_col": "any_prs",
    },
    "cannabis": {
        "fam": ROOT / "EXT" / "PRSPC" / "sud_prs_eur" / "geno_subsets" / "imputed_qc_filtered_eur_unrelated.fam",
        "profile": ROOT / "EXT" / "PRSPC" / "sud_prs_eur" / "scores" / "canud_EUR_unrelated.profile",
        "score_col": "cannabis_prs",
    },
    "nicotine": {
        "fam": ROOT / "EXT" / "PRSPC" / "sud_prs_eur" / "geno_subsets" / "imputed_qc_filtered_eur_unrelated.fam",
        "profile": ROOT / "EXT" / "PRSPC" / "sud_prs_eur" / "scores" / "nicotine_EUR_unrelated.profile",
        "score_col": "nicotine_prs",
    },
}

DOMAIN_CONFIG = {
    "environment": ROOT / "ENV",
    "general_family_context": ROOT / "ML" / "FAMILY",
    "physical_health": ROOT / "PHY",
    "technology": ROOT / "TECH",
    "neurorecognition": ROOT / "NEU",
}

ID_COLS = ["IID", "eventname", "time_num"]


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def load_fam(path: Path) -> pd.DataFrame:
    fam = pd.read_csv(path, sep=r"\s+", header=None)
    fam.columns = ["FID", "IID"] + [f"V{i}" for i in range(3, fam.shape[1] + 1)]
    fam["FID"] = fam["FID"].astype(str)
    fam["IID"] = fam["IID"].astype(str)
    return fam[["FID", "IID"]].drop_duplicates()


def read_prs_profile(profile_path: Path, score_col: str) -> pd.DataFrame:
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing PRS profile: {profile_path}")

    prs = pd.read_csv(profile_path, sep=r"\s+").copy()
    prs["IID"] = prs["IID"].astype(str)

    if "SCORESUM" in prs.columns:
        prs = prs.rename(columns={"SCORESUM": score_col})
    elif "SCORE" in prs.columns:
        prs = prs.rename(columns={"SCORE": score_col})
    else:
        numeric_cols = [c for c in prs.columns if c not in {"FID", "IID", "PHENO"}]
        if len(numeric_cols) == 0:
            raise ValueError(f"Cannot identify score column in {profile_path}")
        prs = prs.rename(columns={numeric_cols[-1]: score_col})

    keep = [c for c in ["IID", score_col, "CNT", "CNT2"] if c in prs.columns]
    return prs[keep].drop_duplicates(subset=["IID"]).copy()


def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s * 0.0
    return (s - mu) / sd


def load_prs() -> pd.DataFrame:
    cfg = PRS_MAP[PHENO]
    fam = load_fam(cfg["fam"])

    if PHENO == "internalizing":
        youth = read_prs_profile(cfg["profiles"]["internalizing_youth_prs"], "internalizing_youth_prs")
        anx = read_prs_profile(cfg["profiles"]["anxiety_prs"], "anxiety_prs")
        mdd = read_prs_profile(cfg["profiles"]["mdd_prs"], "mdd_prs")
        neuro = read_prs_profile(cfg["profiles"]["neuroticism_prs"], "neuroticism_prs")

        prs = fam.copy()
        prs = prs.merge(youth, on="IID", how="left")
        prs = prs.merge(anx[["IID", "anxiety_prs"]], on="IID", how="left")
        prs = prs.merge(mdd[["IID", "mdd_prs"]], on="IID", how="left")
        prs = prs.merge(neuro[["IID", "neuroticism_prs"]], on="IID", how="left")

        prs["anxiety_prs_z"] = zscore_series(prs["anxiety_prs"])
        prs["mdd_prs_z"] = zscore_series(prs["mdd_prs"])
        prs["neuroticism_prs_z"] = zscore_series(prs["neuroticism_prs"])
        prs["internalizing_combo_prs"] = prs[["anxiety_prs_z", "mdd_prs_z", "neuroticism_prs_z"]].mean(axis=1, skipna=True)

        keep = ["FID", "IID",
                "internalizing_youth_prs", "internalizing_combo_prs",
                "anxiety_prs", "mdd_prs", "neuroticism_prs",
                "anxiety_prs_z", "mdd_prs_z", "neuroticism_prs_z"]
        for c in ["CNT", "CNT2"]:
            if c in prs.columns:
                keep.append(c)
        return prs[keep].drop_duplicates(subset=["IID"]).copy()

    prs = read_prs_profile(cfg["profile"], cfg["score_col"])
    prs = fam.merge(prs, on="IID", how="left")
    return prs.drop_duplicates(subset=["IID"]).copy()


def load_anchor_continuous() -> pd.DataFrame:
    path = CONT_OUTCOME_FILES[PHENO]
    df = pd.read_csv(path, sep="\t").copy()
    df["IID"] = df["IID"].astype(str)
    df["time_num"] = df["eventname"].map(WAVE_MAP)
    df = df.loc[df["time_num"].notna()].copy()
    df["time_num"] = df["time_num"].astype(int)

    target_col = f"{PHENO}_t"
    if target_col not in df.columns:
        raise ValueError(f"Missing target column {target_col} in {path}")

    fam = load_fam(PRS_MAP[PHENO]["fam"])
    df = df.merge(fam, on="IID", how="inner", suffixes=("", "_fam"))

    if "FID" not in df.columns:
        if "FID_fam" in df.columns:
            df["FID"] = df["FID_fam"]
        elif "FID_x" in df.columns and "FID_y" in df.columns:
            df["FID"] = df["FID_x"].fillna(df["FID_y"])
        elif "FID_y" in df.columns:
            df["FID"] = df["FID_y"]
        elif "FID_x" in df.columns:
            df["FID"] = df["FID_x"]

    if "FID" not in df.columns:
        df["FID"] = df["IID"]

    df["FID"] = df["FID"].astype(str)

    drop_tmp = [c for c in ["FID_fam", "FID_x", "FID_y"] if c in df.columns and c != "FID"]
    if drop_tmp:
        df = df.drop(columns=drop_tmp)

    if "src_subject_id" not in df.columns:
        df["src_subject_id"] = df["IID"]
    if "time_year" not in df.columns:
        df["time_year"] = df["time_num"].astype(float)

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.sort_values(["IID", "time_num"]).reset_index(drop=True)

    lag_col = f"{PHENO}_lag"
    delta_col = f"delta_{PHENO}"
    df[lag_col] = df.groupby("IID")[target_col].shift(1)
    df[delta_col] = df[target_col] - df[lag_col]

    keep = ["FID", "IID", "src_subject_id", "eventname", "time_year", "time_num",
            target_col, delta_col, lag_col]
    return df[keep].copy()



def load_baseline_age_months() -> pd.DataFrame:
    age = pd.read_csv(BASELINE_AGE_FILE).copy()
    age["src_subject_id"] = age["src_subject_id"].astype(str)
    age = age.loc[age["eventname"] == "baseline_year_1_arm_1"].copy()
    age["baseline_age_year"] = pd.to_numeric(age["age"], errors="coerce")
    age["baseline_age_month"] = age["baseline_age_year"] * 12.0
    age = age.rename(columns={"src_subject_id": "IID"})
    return age[["IID", "baseline_age_year", "baseline_age_month"]].drop_duplicates(subset=["IID"]).copy()

def load_anchor_binary() -> pd.DataFrame:
    path = BIN_OUTCOME_FILES[PHENO]
    df = pd.read_csv(path, sep="	").copy()
    df["IID"] = df["IID"].astype(str)
    df["FID"] = df["FID"].astype(str)

    required = {"IID", "time", "event"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in {path}: {sorted(miss)}")

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype(int)

    age = load_baseline_age_months()
    subj = df[["FID", "IID", "time", "event"]].drop_duplicates().copy()
    subj = subj.merge(age, on="IID", how="left")

    if subj["baseline_age_month"].isna().any():
        nmiss = int(subj["baseline_age_month"].isna().sum())
        print(f"[WARN] Missing baseline age for {nmiss} subjects; dropping them from binary anchor panel")
        subj = subj.loc[subj["baseline_age_month"].notna()].copy()

    anchors = [
        (0, "baseline_year_1_arm_1"),
        (24, "2_year_follow_up_y_arm_1"),
        (48, "4_year_follow_up_y_arm_1"),
    ]

    grid = []
    for offset_month, eventname in anchors:
        tmp = subj.copy()
        tmp["eventname"] = eventname
        tmp["time_num"] = WAVE_MAP[eventname]
        tmp["anchor_offset_month"] = offset_month
        tmp["anchor_age_month"] = tmp["baseline_age_month"] + offset_month
        tmp["time_year"] = tmp["anchor_age_month"] / 12.0

        # event by anchor
        tmp["binary_event"] = (
            (tmp["event"] == 1) &
            (tmp["time"] <= tmp["anchor_age_month"])
        ).astype(float)

        # observed status:
        # 1 if event already happened by anchor OR subject followed/censored at/after anchor
        tmp["observed_by_anchor"] = (
            ((tmp["event"] == 1) & (tmp["time"] <= tmp["anchor_age_month"])) |
            (tmp["time"] >= tmp["anchor_age_month"])
        ).astype(int)

        # if censored before anchor and no earlier event, mark outcome missing
        tmp.loc[tmp["observed_by_anchor"] == 0, "binary_event"] = np.nan

        grid.append(tmp)

    out = pd.concat(grid, axis=0, ignore_index=True)

    keep = [
        "FID", "IID", "eventname", "time_num", "time_year",
        "baseline_age_year", "baseline_age_month",
        "anchor_offset_month", "anchor_age_month",
        "time", "event", "observed_by_anchor", "binary_event"
    ]
    out["src_subject_id"] = out["IID"]
    keep = ["FID", "IID", "src_subject_id"] + [c for c in keep if c not in {"FID", "IID"}]
    return out[keep].copy()

def anchor_panel() -> pd.DataFrame:
    if TASK_TYPE == "continuous":
        return load_anchor_continuous()
    return load_anchor_binary()


def merge_domain_table(anchor: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = read_table(path).copy()

    if "id" in df.columns and "IID" not in df.columns:
        df = df.rename(columns={"id": "IID"})

    for col in ["IID", "eventname", "time_num"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing required column: {col}")

    df["IID"] = df["IID"].astype(str)

    overlap = [c for c in df.columns if c not in ID_COLS and c in anchor.columns]
    if overlap:
        df = df.drop(columns=overlap)

    return anchor.merge(df, on=["IID", "eventname", "time_num"], how="left")



def load_covariates() -> pd.DataFrame:
    cov = pd.read_csv(COVARIATE_FILE, sep="\t").copy()
    cov["IID"] = cov["IID"].astype(str)
    keep = ["IID"] + [c for c in COVARIATE_COLS if c in cov.columns] + (["site"] if "site" in cov.columns else [])
    cov = cov[keep].drop_duplicates(subset=["IID"]).copy()

    if "site" in cov.columns:
        site_dm = pd.get_dummies(cov["site"], prefix="site", dummy_na=False)
        cov = pd.concat([cov.drop(columns=["site"]), site_dm], axis=1)

    for c in cov.columns:
        if c != "IID":
            cov[c] = pd.to_numeric(cov[c], errors="coerce")

    return cov


def load_domain_family(family: str) -> pd.DataFrame:
    df = anchor_panel().merge(load_prs().drop(columns=["FID"], errors="ignore"), on=["IID"], how="left")
    df = df.merge(load_covariates(), on=["IID"], how="left")

    file_name_map = {
        "environment": f"{PHENO}_environment_{family}_inputs.csv",
        "general_family_context": f"{PHENO}_environment_{family}_inputs.csv",
        "physical_health": f"{PHENO}_environment_{family}_inputs.csv",
        "technology": f"{PHENO}_environment_{family}_inputs.csv",
        "neurorecognition": f"{PHENO}_environment_{family}_inputs.csv",
    }

    for domain_name, domain_root in DOMAIN_CONFIG.items():
        path = domain_root / PHENO / "stage04_final_model_inputs" / file_name_map[domain_name]
        df = merge_domain_table(df, path)

    return df


def load_brain_table(analysis_name: str, brain_file: str) -> pd.DataFrame:
    path = ROOT / "Brain" / PHENO / analysis_name / "stage04_final_model_inputs" / brain_file
    df = pd.read_csv(path).copy()

    if "id" in df.columns and "IID" not in df.columns:
        df = df.rename(columns={"id": "IID"})
    if "IID" not in df.columns:
        raise ValueError(f"Brain file missing IID/id: {path}")

    df["IID"] = df["IID"].astype(str)
    eventname = BRAIN_ANALYSIS_TO_EVENT[analysis_name]
    df["eventname"] = eventname
    df["time_num"] = WAVE_MAP[eventname]

    drop_cols = {"time", "event", "time_from_landmark", "landmark_months"}
    keep = [c for c in df.columns if c in ["IID", "eventname", "time_num"] or c not in drop_cols]
    return df[keep].copy()


def add_brain(df: pd.DataFrame, family: str) -> pd.DataFrame:
    out = df.copy()

    b0 = load_brain_table("baseline_primary_pc10_site", "baseline_primary_pc10_site_brain_raw_stable_inputs.csv")
    out = out.merge(b0, on=["IID", "eventname", "time_num"], how="left")

    b24 = load_brain_table("landmark24_primary_pc10_site", "landmark24_primary_pc10_site_brain_raw_stable_inputs.csv")
    overlap24 = [c for c in b24.columns if c not in ID_COLS and c in out.columns]
    if overlap24:
        b24 = b24.drop(columns=overlap24)
    out = out.merge(b24, on=["IID", "eventname", "time_num"], how="left")

    b48_raw = load_brain_table("landmark48_primary_pc10_site", "landmark48_primary_pc10_site_brain_raw_stable_inputs.csv")
    overlap48r = [c for c in b48_raw.columns if c not in ID_COLS and c in out.columns]
    if overlap48r:
        b48_raw = b48_raw.drop(columns=overlap48r)
    out = out.merge(b48_raw, on=["IID", "eventname", "time_num"], how="left")

    fam_to_file = {
        "pca": "landmark48_primary_pc10_site_brain_pca_inputs.csv",
        "weighted": "landmark48_primary_pc10_site_brain_weighted_inputs.csv",
        "cluster": "landmark48_primary_pc10_site_brain_cluster_inputs.csv",
        "all": "landmark48_primary_pc10_site_brain_all_inputs.csv",
    }

    b48_rep = load_brain_table("landmark48_primary_pc10_site", fam_to_file[family])
    overlap48rep = [c for c in b48_rep.columns if c not in ID_COLS and c in out.columns]
    if overlap48rep:
        b48_rep = b48_rep.drop(columns=overlap48rep)
    out = out.merge(b48_rep, on=["IID", "eventname", "time_num"], how="left")

    return out


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()

    for c in list(out.columns):
        if c in {"FID", "IID", "src_subject_id", "eventname"}:
            continue
        col = out[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        out[c] = pd.to_numeric(col, errors="coerce")

    return out.sort_values(["IID", "time_num"]).reset_index(drop=True)


def predictor_cols(df: pd.DataFrame) -> List[str]:
    exclude = {"FID", "IID", "src_subject_id", "eventname"}
    return [c for c in df.columns if c not in exclude]


def save_manifest(df: pd.DataFrame, family: str):
    rows = []
    for c in predictor_cols(df):
        if c.endswith("_prs") or c in {"CNT", "CNT2"}:
            domain = "prs_foundation"
        elif c.startswith("environment_"):
            domain = "environment"
        elif c.startswith("general_family_context_"):
            domain = "general_family_context"
        elif c.startswith("physical_health_"):
            domain = "physical_health"
        elif c.startswith("technology_"):
            domain = "technology"
        elif c.startswith("neurorecognition_"):
            domain = "neurorecognition"
        elif c.startswith("brain_") or c.startswith("smri_"):
            domain = "brain"
        elif c in {"time", "event", "binary_event", "time_month"}:
            domain = "survival_anchor"
        else:
            domain = "other"
        rows.append({"representation_family": family, "variable": c, "domain": domain})

    pd.DataFrame(rows).to_csv(OUTDIR / f"{PHENO}_unified_{family}_manifest.csv", index=False)


def main():
    settings = {
        "pheno": PHENO,
        "task_type": TASK_TYPE,
        "domain_config": {k: str(v) for k, v in DOMAIN_CONFIG.items()},
        "brain_root": str(ROOT / "Brain" / PHENO),
        "outdir": str(OUTDIR),
    }

    for fam in ["pca", "weighted", "cluster", "all"]:
        df = load_domain_family(fam)
        df = add_brain(df, fam)
        df = finalize(df)

        out_path = OUTDIR / f"{PHENO}_unified_{fam}.tsv"
        df.to_csv(out_path, sep="\t", index=False)
        save_manifest(df, fam)

        summary = {
            "pheno": PHENO,
            "task_type": TASK_TYPE,
            "family": fam,
            "n_rows": int(df.shape[0]),
            "n_subjects": int(df["IID"].nunique()),
            "n_predictors": int(len(predictor_cols(df))),
        }
        with open(OUTDIR / f"{PHENO}_unified_{fam}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    with open(OUTDIR / f"{PHENO}_unified_build_settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    print(f"[INFO] Unified tables written to: {OUTDIR}")


if __name__ == "__main__":
    main()
