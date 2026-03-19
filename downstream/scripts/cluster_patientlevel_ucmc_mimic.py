#!/usr/bin/env python3
"""
Patient-level multivariate clustering for OHCA ICU cohorts (UCMC CLIF + MIMIC-IV).

This script:
1) Builds OHCA+ICU first-encounter cohorts for UCMC (CLIF) and MIMIC-IV (raw CSVs),
   restricted by the same cardiac-arrest ICD prefixes used in this repo.
2) Extracts hourly vitals for the first N hours after each patient's first vital.
3) Builds multivariate patient-time features and runs k-means clustering.
4) Writes cluster assignments, diagnostics, and hourly cluster profiles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ICD_PREFIXES = ["I46.0", "I46.1", "I46.2", "I46.8", "I46.9", "I49.00", "I49.01"]
OHCA_TRUE = {"1", "true", "yes", "y"}
OHCA_FALSE = {"0", "false", "no", "n"}
CORE_VITALS = ["heart_rate", "map", "spo2", "temp_c"]
VITAL_RANGES = {
    "heart_rate": (20.0, 250.0),
    "map": (20.0, 180.0),
    "spo2": (50.0, 100.0),
    "temp_c": (30.0, 43.0),
}
MIMIC_ITEM_TO_VITAL = {
    220045: "heart_rate",
    220052: "map",
    220181: "map",
    220277: "spo2",
    223761: "temp_c",  # Fahrenheit, converted below
    223762: "temp_c",  # Celsius
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster patient-level OHCA multivariate trajectories for UCMC CLIF + MIMIC-IV."
    )
    parser.add_argument(
        "--ucmc-dir",
        type=Path,
        default=Path("/Users/davidbeiser/Library/CloudStorage/Box-Box/RCLIF_data/CLIF_2018_24/2.1.0"),
        help="Path containing UCMC CLIF parquet tables.",
    )
    parser.add_argument(
        "--mimic-dir",
        type=Path,
        default=Path("/Users/davidbeiser/mimic-iv-3.1"),
        help="Path containing raw MIMIC-IV folders (hosp/, icu/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_dir/patient_level_clustering_ucmc_mimic"),
        help="Directory for clustering outputs.",
    )
    parser.add_argument(
        "--max-hours",
        type=int,
        default=72,
        help="Hours after first vital to include.",
    )
    parser.add_argument(
        "--bin-hours",
        type=int,
        default=1,
        help="Time bin width in hours.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="Chunk size when streaming MIMIC chartevents.csv.gz.",
    )
    parser.add_argument(
        "--max-mimic-chunks",
        type=int,
        default=None,
        help="Optional cap on processed MIMIC chartevents chunks (for smoke tests).",
    )
    parser.add_argument(
        "--mimic-vitals-cache",
        type=Path,
        default=None,
        help="Optional parquet cache path for filtered MIMIC vitals long-form rows.",
    )
    parser.add_argument(
        "--min-hours-per-patient",
        type=int,
        default=6,
        help="Minimum distinct hourly bins required for a patient.",
    )
    parser.add_argument(
        "--min-total-measurements",
        type=int,
        default=12,
        help="Minimum non-missing vital values across all bins for a patient.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Fixed number of clusters. If omitted, choose k via Davies-Bouldin over [k-min, k-max].",
    )
    parser.add_argument("--k-min", type=int, default=2, help="Min k for auto-selection.")
    parser.add_argument("--k-max", type=int, default=8, help="Max k for auto-selection.")
    parser.add_argument("--n-init", type=int, default=10, help="K-means random restarts.")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations per k-means run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--cluster-mode",
        choices=["patient", "patient_time"],
        default="patient",
        help="`patient` clusters full trajectories; `patient_time` clusters each patient-hour row.",
    )
    parser.add_argument(
        "--timepoint-impute",
        choices=["last_value", "hourly_mean"],
        default="last_value",
        help="Missing-value strategy for `patient_time` mode.",
    )
    parser.add_argument(
        "--locf-max-gap-hours",
        type=int,
        default=None,
        help="Optional max gap for LOCF in `patient_time` mode; longer gaps are not forward-filled.",
    )
    return parser.parse_args()


def _norm_icd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace(".", "", regex=False).str.strip()


def _prefix_mask(code_series: pd.Series, prefixes: list[str]) -> pd.Series:
    code_norm = _norm_icd(code_series)
    pref_norm = [p.replace(".", "").upper() for p in prefixes]
    keep = np.zeros(len(code_norm), dtype=bool)
    for pref in pref_norm:
        keep |= code_norm.str.startswith(pref).to_numpy()
    return pd.Series(keep, index=code_series.index)


def _apply_plausibility(vitals_long: pd.DataFrame) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []
    for vital, (lo, hi) in VITAL_RANGES.items():
        sub = vitals_long[vitals_long["vital_category"] == vital].copy()
        if sub.empty:
            continue
        sub = sub[(sub["vital_value"] >= lo) & (sub["vital_value"] <= hi)]
        out_parts.append(sub)
    if not out_parts:
        return pd.DataFrame(columns=vitals_long.columns)
    return pd.concat(out_parts, ignore_index=True)


def build_ucmc_cohort(ucmc_dir: Path) -> pd.DataFrame:
    dx = pd.read_parquet(
        ucmc_dir / "clif_hospital_diagnosis.parquet",
        columns=["hospitalization_id", "diagnosis_code", "poa_present"],
    )
    dx = dx[_prefix_mask(dx["diagnosis_code"], ICD_PREFIXES)].copy()
    if dx.empty:
        return pd.DataFrame(columns=["patient_key", "site", "patient_id", "hospitalization_id", "survival_status"])

    poa = dx["poa_present"].astype(str).str.strip().str.lower()
    dx["arrest_type"] = np.where(
        poa.isin(OHCA_TRUE),
        "OHCA",
        np.where(poa.isin(OHCA_FALSE), "IHCA", "Unknown"),
    )

    hosp = pd.read_parquet(
        ucmc_dir / "clif_hospitalization.parquet",
        columns=["patient_id", "hospitalization_id", "admission_dttm", "discharge_category"],
    )
    base = dx.merge(hosp, on="hospitalization_id", how="inner")
    if base.empty:
        return pd.DataFrame(columns=["patient_key", "site", "patient_id", "hospitalization_id", "survival_status"])

    ohca = base[base["arrest_type"].eq("OHCA")].copy()
    if ohca.empty:
        ohca = base[base["arrest_type"].eq("Unknown")].copy()
    if ohca.empty:
        return pd.DataFrame(columns=["patient_key", "site", "patient_id", "hospitalization_id", "survival_status"])

    ohca["admission_dttm"] = pd.to_datetime(ohca["admission_dttm"], errors="coerce")
    ohca["hospitalization_id"] = ohca["hospitalization_id"].astype(str)
    ohca["patient_id"] = ohca["patient_id"].astype(str)
    disc = ohca["discharge_category"].astype(str).str.strip().str.lower()
    ohca["survival_status"] = np.where(disc.eq("expired"), "Non-Survivor", "Survivor")
    ohca = ohca.sort_values(["patient_id", "admission_dttm", "hospitalization_id"])
    first = ohca.drop_duplicates(subset=["patient_id"], keep="first").copy()

    adt = pd.read_parquet(ucmc_dir / "clif_adt.parquet", columns=["hospitalization_id", "location_category"])
    has_icu = (
        adt.assign(location=adt["location_category"].astype(str).str.strip().str.lower())
        .query("location == 'icu'")["hospitalization_id"]
        .astype(str)
        .unique()
    )
    first = first[first["hospitalization_id"].isin(set(has_icu))].copy()
    first["site"] = "ucmc"
    first["patient_key"] = "ucmc:" + first["hospitalization_id"]
    return first[["patient_key", "site", "patient_id", "hospitalization_id", "survival_status"]].reset_index(drop=True)


def build_mimic_cohort(mimic_dir: Path) -> pd.DataFrame:
    dx = pd.read_csv(
        mimic_dir / "hosp" / "diagnoses_icd.csv.gz",
        usecols=["subject_id", "hadm_id", "icd_code"],
        dtype={"subject_id": "string", "hadm_id": "string", "icd_code": "string"},
    )
    dx = dx.dropna(subset=["subject_id", "hadm_id", "icd_code"])
    dx = dx[_prefix_mask(dx["icd_code"], ICD_PREFIXES)].copy()
    if dx.empty:
        return pd.DataFrame(columns=["patient_key", "site", "patient_id", "hospitalization_id", "survival_status"])

    adm = pd.read_csv(
        mimic_dir / "hosp" / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "hospital_expire_flag"],
        dtype={
            "subject_id": "string",
            "hadm_id": "string",
            "admittime": "string",
            "hospital_expire_flag": "Int64",
        },
    )
    base = dx.merge(adm, on=["subject_id", "hadm_id"], how="inner")
    if base.empty:
        return pd.DataFrame(columns=["patient_key", "site", "patient_id", "hospitalization_id", "survival_status"])

    base["admittime"] = pd.to_datetime(base["admittime"], errors="coerce")
    base["survival_status"] = np.where(base["hospital_expire_flag"].fillna(0).astype(int) == 1, "Non-Survivor", "Survivor")
    base = base.sort_values(["subject_id", "admittime", "hadm_id"])
    first = base.drop_duplicates(subset=["subject_id"], keep="first").copy()

    icu = pd.read_csv(
        mimic_dir / "icu" / "icustays.csv.gz",
        usecols=["hadm_id"],
        dtype={"hadm_id": "string"},
    )
    has_icu = set(icu["hadm_id"].dropna().astype(str).unique())
    first = first[first["hadm_id"].astype(str).isin(has_icu)].copy()
    first = first.rename(columns={"subject_id": "patient_id", "hadm_id": "hospitalization_id"})
    first["site"] = "mimic"
    first["patient_key"] = "mimic:" + first["hospitalization_id"].astype(str)
    return first[["patient_key", "site", "patient_id", "hospitalization_id", "survival_status"]].reset_index(drop=True)


def build_hourly_from_long(vitals_long: pd.DataFrame, cohort: pd.DataFrame, max_hours: int, bin_hours: int) -> pd.DataFrame:
    if vitals_long.empty:
        cols = ["patient_key", "site", "patient_id", "hospitalization_id", "survival_status", "hour"] + CORE_VITALS
        return pd.DataFrame(columns=cols)

    vitals_long["recorded_dttm"] = pd.to_datetime(vitals_long["recorded_dttm"], errors="coerce")
    vitals_long["vital_value"] = pd.to_numeric(vitals_long["vital_value"], errors="coerce")
    vitals_long = vitals_long.dropna(subset=["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"])
    if vitals_long.empty:
        cols = ["patient_key", "site", "patient_id", "hospitalization_id", "survival_status", "hour"] + CORE_VITALS
        return pd.DataFrame(columns=cols)

    vitals_long = _apply_plausibility(vitals_long)
    if vitals_long.empty:
        cols = ["patient_key", "site", "patient_id", "hospitalization_id", "survival_status", "hour"] + CORE_VITALS
        return pd.DataFrame(columns=cols)

    time_zero = (
        vitals_long.groupby("hospitalization_id", as_index=False)["recorded_dttm"]
        .min()
        .rename(columns={"recorded_dttm": "time_zero"})
    )
    vitals_long = vitals_long.merge(time_zero, on="hospitalization_id", how="inner")
    vitals_long["hours_from_zero"] = (
        (vitals_long["recorded_dttm"] - vitals_long["time_zero"]).dt.total_seconds() / 3600.0
    )
    vitals_long = vitals_long[
        (vitals_long["hours_from_zero"] >= 0.0) & (vitals_long["hours_from_zero"] < float(max_hours))
    ].copy()
    if vitals_long.empty:
        cols = ["patient_key", "site", "patient_id", "hospitalization_id", "survival_status", "hour"] + CORE_VITALS
        return pd.DataFrame(columns=cols)

    vitals_long["hour"] = (
        np.floor(vitals_long["hours_from_zero"] / float(bin_hours)).astype(int) * int(bin_hours)
    )

    grouped = (
        vitals_long.groupby(["hospitalization_id", "hour", "vital_category"], as_index=False)["vital_value"]
        .median()
    )
    wide = grouped.pivot_table(
        index=["hospitalization_id", "hour"],
        columns="vital_category",
        values="vital_value",
        aggfunc="first",
    ).reset_index()

    for vital in CORE_VITALS:
        if vital not in wide.columns:
            wide[vital] = np.nan

    meta_cols = ["patient_key", "site", "patient_id", "hospitalization_id", "survival_status"]
    wide = wide.merge(cohort[meta_cols], on="hospitalization_id", how="inner")
    out_cols = meta_cols + ["hour"] + CORE_VITALS
    return wide[out_cols].sort_values(["site", "hospitalization_id", "hour"]).reset_index(drop=True)


def load_ucmc_hourly(ucmc_dir: Path, cohort: pd.DataFrame, max_hours: int, bin_hours: int) -> pd.DataFrame:
    if cohort.empty:
        return pd.DataFrame(columns=["patient_key", "site", "patient_id", "hospitalization_id", "survival_status", "hour"] + CORE_VITALS)

    hosp_ids = cohort["hospitalization_id"].astype(str).unique().tolist()
    vit = pd.read_parquet(
        ucmc_dir / "clif_vitals.parquet",
        columns=["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"],
        filters=[
            ("hospitalization_id", "in", hosp_ids),
            ("vital_category", "in", CORE_VITALS),
        ],
    )
    vit["hospitalization_id"] = vit["hospitalization_id"].astype(str)
    vit["vital_category"] = vit["vital_category"].astype(str).str.strip().str.lower()
    return build_hourly_from_long(vit, cohort, max_hours=max_hours, bin_hours=bin_hours)


def load_mimic_hourly(
    mimic_dir: Path,
    cohort: pd.DataFrame,
    max_hours: int,
    bin_hours: int,
    chunk_size: int,
    max_mimic_chunks: int | None,
    mimic_vitals_cache: Path | None,
) -> pd.DataFrame:
    cols = ["patient_key", "site", "patient_id", "hospitalization_id", "survival_status", "hour"] + CORE_VITALS
    if cohort.empty:
        return pd.DataFrame(columns=cols)

    if mimic_vitals_cache is not None and mimic_vitals_cache.exists():
        vitals = pd.read_parquet(mimic_vitals_cache)
        vitals["hospitalization_id"] = vitals["hospitalization_id"].astype(str)
        return build_hourly_from_long(vitals, cohort, max_hours=max_hours, bin_hours=bin_hours)

    hadm_set = set(cohort["hospitalization_id"].astype(str).unique())
    wanted_itemids = sorted(MIMIC_ITEM_TO_VITAL.keys())
    chunks: list[pd.DataFrame] = []

    reader = pd.read_csv(
        mimic_dir / "icu" / "chartevents.csv.gz",
        usecols=["hadm_id", "charttime", "itemid", "valuenum"],
        dtype={"hadm_id": "string", "charttime": "string", "itemid": "Int64", "valuenum": "float64"},
        chunksize=chunk_size,
        low_memory=True,
    )

    for idx, chunk in enumerate(reader, start=1):
        if max_mimic_chunks is not None and idx > max_mimic_chunks:
            break
        chunk = chunk.dropna(subset=["hadm_id", "charttime", "itemid", "valuenum"])
        chunk = chunk[chunk["hadm_id"].astype(str).isin(hadm_set)]
        chunk = chunk[chunk["itemid"].astype(int).isin(wanted_itemids)]
        if chunk.empty:
            if idx % 25 == 0:
                print(f"[mimic] processed {idx:,} chunks...")
            continue

        chunk["hospitalization_id"] = chunk["hadm_id"].astype(str)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk["vital_category"] = chunk["itemid"].map(MIMIC_ITEM_TO_VITAL)
        chunk["recorded_dttm"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["vital_value"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.dropna(subset=["hospitalization_id", "vital_category", "recorded_dttm", "vital_value"])
        if chunk.empty:
            if idx % 25 == 0:
                print(f"[mimic] processed {idx:,} chunks...")
            continue

        fmask = chunk["itemid"].eq(223761)
        chunk.loc[fmask, "vital_value"] = (chunk.loc[fmask, "vital_value"] - 32.0) / 1.8
        chunks.append(chunk[["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"]].copy())

        if idx % 25 == 0:
            print(f"[mimic] processed {idx:,} chunks...")

    if not chunks:
        return pd.DataFrame(columns=cols)

    vitals = pd.concat(chunks, ignore_index=True)
    if mimic_vitals_cache is not None:
        mimic_vitals_cache.parent.mkdir(parents=True, exist_ok=True)
        vitals.to_parquet(mimic_vitals_cache, index=False)
    return build_hourly_from_long(vitals, cohort, max_hours=max_hours, bin_hours=bin_hours)


def filter_patient_coverage(
    hourly: pd.DataFrame, min_hours_per_patient: int, min_total_measurements: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if hourly.empty:
        return hourly, pd.DataFrame(columns=["patient_key", "n_hours", "n_measurements"])

    tmp = hourly.copy()
    tmp["row_nonmissing"] = tmp[CORE_VITALS].notna().sum(axis=1)
    coverage = (
        tmp.groupby("patient_key", as_index=False)
        .agg(
            n_hours=("hour", "nunique"),
            n_measurements=("row_nonmissing", "sum"),
        )
    )
    keep = coverage[
        (coverage["n_hours"] >= int(min_hours_per_patient))
        & (coverage["n_measurements"] >= int(min_total_measurements))
    ]["patient_key"]
    filtered = hourly[hourly["patient_key"].isin(set(keep))].copy()
    return filtered, coverage


def build_timepoint_matrix(
    hourly: pd.DataFrame,
    max_hours: int,
    bin_hours: int,
    impute_strategy: str,
    locf_max_gap_hours: int | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    if hourly.empty:
        return (
            pd.DataFrame(),
            np.empty((0, 0)),
            np.empty((0,)),
            np.empty((0,)),
            {},
            {},
        )

    meta_cols = ["patient_key", "site", "patient_id", "hospitalization_id", "survival_status"]
    patient_meta = hourly[meta_cols].drop_duplicates(subset=["patient_key"]).copy()
    hours = pd.DataFrame({"hour": np.arange(0, int(max_hours), int(bin_hours), dtype=int)})
    patient_meta["_k"] = 1
    hours["_k"] = 1
    grid = patient_meta.merge(hours, on="_k", how="inner").drop(columns="_k")

    long_cols = ["patient_key", "hour"] + CORE_VITALS
    merged = grid.merge(hourly[long_cols], on=["patient_key", "hour"], how="left")
    merged = merged.sort_values(["patient_key", "hour"]).reset_index(drop=True)

    pre_missing = {
        vital: int(merged[vital].isna().sum())
        for vital in CORE_VITALS
    }

    if impute_strategy == "last_value":
        for vital in CORE_VITALS:
            ffilled = merged.groupby("patient_key")[vital].ffill()
            if locf_max_gap_hours is not None:
                obs_hour = merged["hour"].where(merged[vital].notna(), np.nan)
                last_obs_hour = obs_hour.groupby(merged["patient_key"]).ffill()
                gap = merged["hour"] - last_obs_hour
                ffilled = ffilled.where(gap <= int(locf_max_gap_hours))
            merged[vital] = merged[vital].fillna(ffilled)
    elif impute_strategy != "hourly_mean":
        raise ValueError(f"Unknown impute strategy: {impute_strategy}")

    for vital in CORE_VITALS:
        hour_site_mean = merged.groupby(["site", "hour"])[vital].transform("mean")
        merged[vital] = merged[vital].fillna(hour_site_mean)
        global_mean = merged[vital].mean()
        if np.isnan(global_mean):
            global_mean = 0.0
        merged[vital] = merged[vital].fillna(float(global_mean))

    post_missing = {
        vital: int(merged[vital].isna().sum())
        for vital in CORE_VITALS
    }

    x = merged[CORE_VITALS].to_numpy(dtype=float, copy=True)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    xz = (x - mu) / sigma

    out = merged[meta_cols + ["hour"] + CORE_VITALS].copy()
    return out, xz, mu, sigma, pre_missing, post_missing


def build_feature_matrix(hourly: pd.DataFrame, max_hours: int, bin_hours: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if hourly.empty:
        return pd.DataFrame(), np.empty((0, 0)), np.empty((0,)), np.empty((0,))

    hours = np.arange(0, int(max_hours), int(bin_hours), dtype=int)
    long_df = hourly.melt(
        id_vars=["patient_key", "hour"],
        value_vars=CORE_VITALS,
        var_name="vital",
        value_name="value",
    )
    mat = long_df.pivot_table(
        index="patient_key",
        columns=["vital", "hour"],
        values="value",
        aggfunc="first",
    )
    full_cols = pd.MultiIndex.from_product([CORE_VITALS, hours], names=["vital", "hour"])
    mat = mat.reindex(columns=full_cols)

    x = mat.to_numpy(dtype=float, copy=True)
    if x.size == 0:
        return mat, x, np.empty((0,)), np.empty((0,))

    col_medians = np.nanmedian(x, axis=0)
    global_median = np.nanmedian(x)
    if np.isnan(global_median):
        global_median = 0.0
    col_medians = np.where(np.isnan(col_medians), global_median, col_medians)
    nan_rows, nan_cols = np.where(np.isnan(x))
    if len(nan_rows) > 0:
        x[nan_rows, nan_cols] = col_medians[nan_cols]

    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    xz = (x - mu) / sigma
    return mat, xz, mu, sigma


def kmeans_fit(
    x: np.ndarray,
    k: int,
    seed: int = 42,
    n_init: int = 10,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    if k < 2:
        raise ValueError("k must be >= 2.")
    if x.shape[0] < k:
        raise ValueError("k cannot exceed number of patients.")

    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = np.inf

    for init_idx in range(n_init):
        pick = rng.choice(x.shape[0], size=k, replace=False)
        centers = x[pick].copy()
        prev_inertia = np.inf

        for _ in range(max_iter):
            dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = dist2.argmin(axis=1)
            inertia = float(np.take_along_axis(dist2, labels[:, None], axis=1).sum())

            new_centers = centers.copy()
            for c in range(k):
                mask = labels == c
                if mask.any():
                    new_centers[c] = x[mask].mean(axis=0)
                else:
                    farthest = int(np.argmax(np.min(dist2, axis=1)))
                    new_centers[c] = x[farthest]

            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if abs(prev_inertia - inertia) <= tol * max(prev_inertia, 1.0) and shift < tol:
                break
            prev_inertia = inertia

        dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist2.argmin(axis=1)
        inertia = float(np.take_along_axis(dist2, labels[:, None], axis=1).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    if best_labels is None or best_centers is None:
        raise RuntimeError("k-means failed to converge.")
    return best_labels, best_centers, best_inertia


def davies_bouldin_index(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    k = centers.shape[0]
    scat = np.zeros(k, dtype=float)
    for c in range(k):
        sub = x[labels == c]
        if sub.shape[0] == 0:
            return float("inf")
        scat[c] = np.sqrt(((sub - centers[c]) ** 2).sum(axis=1)).mean()

    center_dist = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(center_dist, np.inf)
    ratio = (scat[:, None] + scat[None, :]) / center_dist
    di = ratio.max(axis=1)
    return float(np.mean(di))


def choose_k(
    x: np.ndarray,
    k_fixed: int | None,
    k_min: int,
    k_max: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> tuple[int, np.ndarray, np.ndarray, float, pd.DataFrame]:
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 patients for clustering.")

    if k_fixed is not None:
        labels, centers, inertia = kmeans_fit(x, k_fixed, seed=seed, n_init=n_init, max_iter=max_iter)
        dbi = davies_bouldin_index(x, labels, centers)
        diag = pd.DataFrame(
            [{"k": int(k_fixed), "inertia": float(inertia), "davies_bouldin": float(dbi), "selected": True}]
        )
        return int(k_fixed), labels, centers, inertia, diag

    k_lo = max(2, int(k_min))
    k_hi = min(int(k_max), n - 1)
    if k_hi < k_lo:
        k_hi = k_lo
    rows = []
    best = None
    best_k = None

    for k in range(k_lo, k_hi + 1):
        labels, centers, inertia = kmeans_fit(x, k, seed=seed, n_init=n_init, max_iter=max_iter)
        dbi = davies_bouldin_index(x, labels, centers)
        rows.append({"k": k, "inertia": float(inertia), "davies_bouldin": float(dbi)})
        if best is None or dbi < best["davies_bouldin"]:
            best = {"labels": labels, "centers": centers, "inertia": inertia, "davies_bouldin": dbi}
            best_k = k

    if best is None or best_k is None:
        raise RuntimeError("Failed to select k.")

    diag = pd.DataFrame(rows)
    diag["selected"] = diag["k"].eq(best_k)
    return int(best_k), best["labels"], best["centers"], float(best["inertia"]), diag


def write_outputs_patient(
    output_dir: Path,
    hourly_filtered: pd.DataFrame,
    coverage: pd.DataFrame,
    feature_mat: pd.DataFrame,
    labels: np.ndarray,
    centers: np.ndarray,
    selected_k: int,
    k_diag: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    cluster_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_meta = (
        hourly_filtered.groupby("patient_key", as_index=False)
        .agg(
            site=("site", "first"),
            patient_id=("patient_id", "first"),
            hospitalization_id=("hospitalization_id", "first"),
            survival_status=("survival_status", "first"),
        )
    )
    patient_meta = patient_meta.set_index("patient_key").loc[feature_mat.index].reset_index()
    patient_meta["cluster"] = (labels + 1).astype(int)
    patient_meta.to_csv(output_dir / "patient_cluster_assignments.csv", index=False)

    if not coverage.empty:
        coverage.to_csv(output_dir / "patient_coverage.csv", index=False)

    cluster_sizes = (
        patient_meta.groupby(["cluster", "site", "survival_status"], as_index=False)
        .size()
        .rename(columns={"size": "n_patients"})
        .sort_values(["cluster", "site", "survival_status"])
    )
    cluster_sizes.to_csv(output_dir / "cluster_size_by_site_survival.csv", index=False)

    prof = hourly_filtered.merge(patient_meta[["patient_key", "cluster"]], on="patient_key", how="inner")
    cluster_hourly = (
        prof.groupby(["cluster", "hour"], as_index=False)[CORE_VITALS]
        .mean(numeric_only=True)
        .sort_values(["cluster", "hour"])
    )
    cluster_hourly.to_csv(output_dir / "cluster_hourly_profiles.csv", index=False)

    center_orig = centers * sigma + mu
    flat_names = [f"{vital}_h{int(hour):03d}" for vital, hour in feature_mat.columns.tolist()]
    centers_df = pd.DataFrame(center_orig, columns=flat_names)
    centers_df.insert(0, "cluster", np.arange(1, centers_df.shape[0] + 1))
    centers_df.to_csv(output_dir / "cluster_centroids_feature_space.csv", index=False)

    k_diag.to_csv(output_dir / "k_selection_diagnostics.csv", index=False)

    pca_x = feature_mat.to_numpy(dtype=float)
    pca_centered = pca_x - pca_x.mean(axis=0)
    try:
        u, s, _ = np.linalg.svd(pca_centered, full_matrices=False)
        coords = u[:, :2] * s[:2]
        pca_df = patient_meta.copy()
        pca_df["pc1"] = coords[:, 0]
        pca_df["pc2"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
        pca_df.to_csv(output_dir / "patient_cluster_pca.csv", index=False)
    except Exception:
        pass

    summary_lines = [
        f"cluster_mode: {cluster_mode}",
        f"selected_k: {selected_k}",
        f"n_patients_clustered: {feature_mat.shape[0]}",
        f"n_features_per_patient: {feature_mat.shape[1]}",
        f"clusters_saved: {output_dir}",
    ]
    (output_dir / "run_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def write_outputs_timepoint(
    output_dir: Path,
    timepoint_rows: pd.DataFrame,
    coverage: pd.DataFrame,
    labels: np.ndarray,
    centers: np.ndarray,
    selected_k: int,
    k_diag: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    impute_strategy: str,
    locf_max_gap_hours: int | None,
    pre_missing: dict[str, int],
    post_missing: dict[str, int],
    cluster_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    assign = timepoint_rows.copy()
    assign["cluster"] = (labels + 1).astype(int)
    assign.to_csv(output_dir / "timepoint_cluster_assignments.csv", index=False)

    if not coverage.empty:
        coverage.to_csv(output_dir / "patient_coverage.csv", index=False)

    cluster_sizes = (
        assign.groupby(["cluster", "site", "survival_status"], as_index=False)
        .agg(
            n_patient_hours=("patient_key", "size"),
            n_patients=("patient_key", "nunique"),
        )
        .sort_values(["cluster", "site", "survival_status"])
    )
    cluster_sizes.to_csv(output_dir / "cluster_size_by_site_survival.csv", index=False)

    cluster_hourly = (
        assign.groupby(["cluster", "hour"], as_index=False)[CORE_VITALS]
        .mean(numeric_only=True)
        .sort_values(["cluster", "hour"])
    )
    cluster_hourly.to_csv(output_dir / "cluster_hourly_profiles.csv", index=False)

    center_orig = centers * sigma + mu
    centers_df = pd.DataFrame(center_orig, columns=CORE_VITALS)
    centers_df.insert(0, "cluster", np.arange(1, centers_df.shape[0] + 1))
    centers_df.to_csv(output_dir / "cluster_centroids_feature_space.csv", index=False)

    k_diag.to_csv(output_dir / "k_selection_diagnostics.csv", index=False)

    trans = assign.sort_values(["patient_key", "hour"])[["patient_key", "hour", "cluster"]].copy()
    trans["next_cluster"] = trans.groupby("patient_key")["cluster"].shift(-1)
    trans = trans.dropna(subset=["next_cluster"])
    if not trans.empty:
        trans["next_cluster"] = trans["next_cluster"].astype(int)
        trans_counts = (
            trans.groupby(["cluster", "next_cluster"], as_index=False)
            .size()
            .rename(columns={"size": "n_transitions"})
            .sort_values(["cluster", "next_cluster"])
        )
        trans_counts.to_csv(output_dir / "cluster_transition_counts.csv", index=False)

    summary_lines = [
        f"cluster_mode: {cluster_mode}",
        f"timepoint_impute: {impute_strategy}",
        f"locf_max_gap_hours: {locf_max_gap_hours}",
        f"selected_k: {selected_k}",
        f"n_patient_hours_clustered: {assign.shape[0]}",
        f"n_features_per_row: {len(CORE_VITALS)}",
        "pre_impute_missing_counts: "
        + ", ".join([f"{k}={v}" for k, v in pre_missing.items()]),
        "post_impute_missing_counts: "
        + ", ".join([f"{k}={v}" for k, v in post_missing.items()]),
        f"clusters_saved: {output_dir}",
    ]
    (output_dir / "run_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Building UCMC cohort (OHCA ICD prefix restricted)...")
    ucmc_cohort = build_ucmc_cohort(args.ucmc_dir)
    print(f"      UCMC cohort size: {ucmc_cohort['hospitalization_id'].nunique():,}")

    print("[2/6] Building MIMIC cohort (OHCA ICD prefix restricted)...")
    mimic_cohort = build_mimic_cohort(args.mimic_dir)
    print(f"      MIMIC cohort size: {mimic_cohort['hospitalization_id'].nunique():,}")

    print("[3/6] Extracting hourly UCMC vitals...")
    ucmc_hourly = load_ucmc_hourly(args.ucmc_dir, ucmc_cohort, max_hours=args.max_hours, bin_hours=args.bin_hours)
    print(f"      UCMC hourly rows: {len(ucmc_hourly):,}")

    print("[4/6] Streaming MIMIC chartevents and extracting hourly vitals...")
    mimic_hourly = load_mimic_hourly(
        args.mimic_dir,
        mimic_cohort,
        max_hours=args.max_hours,
        bin_hours=args.bin_hours,
        chunk_size=args.chunk_size,
        max_mimic_chunks=args.max_mimic_chunks,
        mimic_vitals_cache=args.mimic_vitals_cache,
    )
    print(f"      MIMIC hourly rows: {len(mimic_hourly):,}")

    combined = pd.concat([ucmc_hourly, mimic_hourly], ignore_index=True)
    if combined.empty:
        raise RuntimeError("No hourly vitals extracted for clustering.")

    print("[5/6] Applying patient coverage filters and building feature matrix...")
    combined_filt, coverage = filter_patient_coverage(
        combined,
        min_hours_per_patient=args.min_hours_per_patient,
        min_total_measurements=args.min_total_measurements,
    )
    if combined_filt.empty:
        raise RuntimeError("No patients left after coverage filters.")

    feat_mat: pd.DataFrame | None = None
    timepoint_rows: pd.DataFrame | None = None
    pre_missing: dict[str, int] = {}
    post_missing: dict[str, int] = {}

    if args.cluster_mode == "patient":
        feat_mat, xz, mu, sigma = build_feature_matrix(
            combined_filt,
            max_hours=args.max_hours,
            bin_hours=args.bin_hours,
        )
        if xz.shape[0] < 2:
            raise RuntimeError("Need at least 2 patients after preprocessing for clustering.")
        print(f"      Patients for clustering: {xz.shape[0]:,}")
        print(f"      Features per patient: {xz.shape[1]:,}")
    else:
        timepoint_rows, xz, mu, sigma, pre_missing, post_missing = build_timepoint_matrix(
            combined_filt,
            max_hours=args.max_hours,
            bin_hours=args.bin_hours,
            impute_strategy=args.timepoint_impute,
            locf_max_gap_hours=args.locf_max_gap_hours,
        )
        if xz.shape[0] < 2:
            raise RuntimeError("Need at least 2 patient-hour rows after preprocessing for clustering.")
        print(f"      Patient-hour rows for clustering: {xz.shape[0]:,}")
        print(f"      Features per row: {xz.shape[1]:,}")
        print(
            "      Missing before impute: "
            + ", ".join([f"{k}={v:,}" for k, v in pre_missing.items()])
        )
        print(
            "      Missing after impute: "
            + ", ".join([f"{k}={v:,}" for k, v in post_missing.items()])
        )

    print("[6/6] Running k-means clustering...")
    selected_k, labels, centers, inertia, k_diag = choose_k(
        x=xz,
        k_fixed=args.k,
        k_min=args.k_min,
        k_max=args.k_max,
        seed=args.seed,
        n_init=args.n_init,
        max_iter=args.max_iter,
    )
    print(f"      Selected k: {selected_k}")
    print(f"      Inertia: {inertia:.2f}")

    if args.cluster_mode == "patient":
        if feat_mat is None:
            raise RuntimeError("Internal error: patient feature matrix missing.")
        write_outputs_patient(
            output_dir=args.output_dir,
            hourly_filtered=combined_filt,
            coverage=coverage,
            feature_mat=feat_mat,
            labels=labels,
            centers=centers,
            selected_k=selected_k,
            k_diag=k_diag,
            mu=mu,
            sigma=sigma,
            cluster_mode=args.cluster_mode,
        )
    else:
        if timepoint_rows is None:
            raise RuntimeError("Internal error: timepoint matrix missing.")
        write_outputs_timepoint(
            output_dir=args.output_dir,
            timepoint_rows=timepoint_rows,
            coverage=coverage,
            labels=labels,
            centers=centers,
            selected_k=selected_k,
            k_diag=k_diag,
            mu=mu,
            sigma=sigma,
            impute_strategy=args.timepoint_impute,
            locf_max_gap_hours=args.locf_max_gap_hours,
            pre_missing=pre_missing,
            post_missing=post_missing,
            cluster_mode=args.cluster_mode,
        )
    print(f"Done. Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
