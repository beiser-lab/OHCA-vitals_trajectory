#!/usr/bin/env python3
"""
Patient-level guideline concordance (6h epoch medians, first 48h) for selected sites.

This script reproduces the core cohort logic used in the OHCA pipeline:
  1) Cardiac arrest ICD filter from clif_hospital_diagnosis
  2) Restrict to OHCA (poa_present truthy; fallback to Unknown if no OHCA rows)
  3) First encounter per patient by admission_dttm
  4) ICU admitted based on clif_adt.location_category == 'icu'
  5) Time-zero = first vital timestamp from clif_vitals

Concordance targets:
  - Glucose: 70-180 mg/dL
  - MAP: >= 65 mmHg
  - SpO2: 90-98%
  - Temperature: 32.0-37.5 C
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ICD_PREFIXES = ["I46.0", "I46.1", "I46.2", "I46.8", "I46.9", "I49.00", "I49.01"]
GLUCOSE_LAB_CATEGORIES = {"glucose_fingerstick", "glucose_serum", "glucose_mixed_venous"}
OHCA_TRUE = {"1", "true", "yes", "y"}
OHCA_FALSE = {"0", "false", "no", "n"}
CORE_VITALS = {"heart_rate", "temp_c", "spo2", "map"}


@dataclass(frozen=True)
class MetricSpec:
    metric: str
    label: str
    lo: float | None
    hi: float | None
    plausible_lo: float
    plausible_hi: float


METRICS = [
    MetricSpec("glucose", "Glucose 70-180 mg/dL", 70.0, 180.0, 20.0, 1000.0),
    MetricSpec("map", "MAP >= 65 mmHg", 65.0, None, 30.0, 180.0),
    MetricSpec("spo2", "SpO2 90-98%", 90.0, 98.0, 50.0, 100.0),
    MetricSpec("temp_c", "Temp 32.0-36.0 C for >=24h", 32.0, 36.0, 32.0, 42.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patient-level 48h guideline concordance for UCMC + MIMIC.")
    parser.add_argument(
        "--mimic-dir",
        type=Path,
        default=Path("/Users/davidbeiser/Documents/Mimic-2-clif/output/rclif-2.1-mimic-3.1"),
    )
    parser.add_argument(
        "--ucmc-dir",
        type=Path,
        default=Path("/Users/davidbeiser/Library/CloudStorage/Box-Box/RCLIF_data/CLIF_2018_24/2.1.0"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_dir/preliminary_figures"),
    )
    parser.add_argument(
        "--max-hours",
        type=int,
        default=48,
    )
    parser.add_argument(
        "--epoch-hours",
        type=int,
        default=6,
    )
    return parser.parse_args()


def _norm_icd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace(".", "", regex=False).str.strip()


def _truthy_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _in_range(s: pd.Series, lo: float | None, hi: float | None) -> pd.Series:
    out = pd.Series(True, index=s.index)
    if lo is not None:
        out &= s >= lo
    if hi is not None:
        out &= s <= hi
    return out


def _site_read(path: Path, table: str, columns: list[str], filters=None) -> pd.DataFrame:
    return pd.read_parquet(path / f"{table}.parquet", columns=columns, filters=filters)


def build_ohca_icu_first_encounter(data_dir: Path) -> pd.DataFrame:
    dx = _site_read(
        data_dir,
        "clif_hospital_diagnosis",
        ["hospitalization_id", "diagnosis_code", "poa_present"],
    )
    dx["diagnosis_norm"] = _norm_icd(dx["diagnosis_code"])
    prefix_norm = [p.replace(".", "").upper() for p in ICD_PREFIXES]
    keep = np.zeros(len(dx), dtype=bool)
    for pref in prefix_norm:
        keep |= dx["diagnosis_norm"].str.startswith(pref)
    dx = dx[keep].copy()
    if dx.empty:
        return pd.DataFrame(columns=["patient_id", "hospitalization_id", "admission_dttm", "survival_status"])

    poa_norm = _truthy_series(dx["poa_present"])
    dx["arrest_type"] = np.where(
        poa_norm.isin(OHCA_TRUE),
        "OHCA",
        np.where(poa_norm.isin(OHCA_FALSE), "IHCA", "Unknown"),
    )

    hosp = _site_read(
        data_dir,
        "clif_hospitalization",
        ["patient_id", "hospitalization_id", "admission_dttm", "discharge_category"],
    )
    base = dx.merge(hosp, on="hospitalization_id", how="inner")
    if base.empty:
        return pd.DataFrame(columns=["patient_id", "hospitalization_id", "admission_dttm", "survival_status"])

    base["admission_dttm"] = pd.to_datetime(base["admission_dttm"], errors="coerce")
    disc = base["discharge_category"].astype(str).str.strip().str.lower()
    base["survival_status"] = np.where(disc.eq("expired"), "Non-Survivor", "Survivor")

    ohca = base[base["arrest_type"].eq("OHCA")].copy()
    if ohca.empty:
        ohca = base[base["arrest_type"].eq("Unknown")].copy()
        ohca.loc[:, "arrest_type"] = "OHCA"
    if ohca.empty:
        return pd.DataFrame(columns=["patient_id", "hospitalization_id", "admission_dttm", "survival_status"])

    ohca = ohca.sort_values(["patient_id", "admission_dttm", "hospitalization_id"])
    first = ohca.drop_duplicates(subset=["patient_id"], keep="first").copy()

    adt = _site_read(data_dir, "clif_adt", ["hospitalization_id", "location_category"])
    has_icu = (
        adt.assign(loc=adt["location_category"].astype(str).str.strip().str.lower())
        .query("loc == 'icu'")["hospitalization_id"]
        .dropna()
        .astype(str)
        .unique()
    )
    first["hospitalization_id"] = first["hospitalization_id"].astype(str)
    first = first[first["hospitalization_id"].isin(set(has_icu))].copy()
    return first[["patient_id", "hospitalization_id", "admission_dttm", "survival_status"]].reset_index(drop=True)


def get_time_zero_and_vitals(data_dir: Path, cohort: pd.DataFrame, max_hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cohort.empty:
        return pd.DataFrame(columns=["hospitalization_id", "time_zero"]), pd.DataFrame()

    hosp_ids = cohort["hospitalization_id"].astype(str).unique().tolist()
    vit = _site_read(
        data_dir,
        "clif_vitals",
        ["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"],
        filters=[("hospitalization_id", "in", hosp_ids), ("vital_category", "in", list(CORE_VITALS))],
    )
    if vit.empty:
        return pd.DataFrame(columns=["hospitalization_id", "time_zero"]), pd.DataFrame()

    vit["hospitalization_id"] = vit["hospitalization_id"].astype(str)
    vit["recorded_dttm"] = pd.to_datetime(vit["recorded_dttm"], errors="coerce")
    vit["vital_value"] = pd.to_numeric(vit["vital_value"], errors="coerce")
    vit = vit.dropna(subset=["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"])
    if vit.empty:
        return pd.DataFrame(columns=["hospitalization_id", "time_zero"]), pd.DataFrame()

    tz = vit.groupby("hospitalization_id", as_index=False)["recorded_dttm"].min().rename(columns={"recorded_dttm": "time_zero"})
    vit = vit.merge(tz, on="hospitalization_id", how="inner")
    vit["hours"] = (vit["recorded_dttm"] - vit["time_zero"]).dt.total_seconds() / 3600.0
    vit = vit[(vit["hours"] >= 0.0) & (vit["hours"] < float(max_hours))].copy()
    return tz, vit


def get_glucose_rows(data_dir: Path, cohort: pd.DataFrame, time_zero: pd.DataFrame, max_hours: int) -> pd.DataFrame:
    if cohort.empty or time_zero.empty:
        return pd.DataFrame(columns=["hospitalization_id", "hours", "vital_value"])

    hosp_ids = cohort["hospitalization_id"].astype(str).unique().tolist()
    labs = _site_read(
        data_dir,
        "clif_labs",
        [
            "hospitalization_id",
            "lab_category",
            "lab_value_numeric",
            "lab_result_dttm",
            "lab_collect_dttm",
            "lab_order_dttm",
        ],
        filters=[("hospitalization_id", "in", hosp_ids), ("lab_category", "in", list(GLUCOSE_LAB_CATEGORIES))],
    )
    if labs.empty:
        return pd.DataFrame(columns=["hospitalization_id", "hours", "vital_value"])

    labs["hospitalization_id"] = labs["hospitalization_id"].astype(str)
    labs["vital_value"] = pd.to_numeric(labs["lab_value_numeric"], errors="coerce")
    ts = pd.to_datetime(labs["lab_result_dttm"], errors="coerce")
    ts = ts.fillna(pd.to_datetime(labs["lab_collect_dttm"], errors="coerce"))
    ts = ts.fillna(pd.to_datetime(labs["lab_order_dttm"], errors="coerce"))
    labs["recorded_dttm"] = ts
    labs = labs.dropna(subset=["hospitalization_id", "vital_value", "recorded_dttm"])
    if labs.empty:
        return pd.DataFrame(columns=["hospitalization_id", "hours", "vital_value"])

    out = labs[["hospitalization_id", "recorded_dttm", "vital_value"]].merge(time_zero, on="hospitalization_id", how="inner")
    out["hours"] = (out["recorded_dttm"] - out["time_zero"]).dt.total_seconds() / 3600.0
    out = out[(out["hours"] >= 0.0) & (out["hours"] < float(max_hours))].copy()
    return out[["hospitalization_id", "hours", "vital_value"]]


def epoch_median_concordance(rows: pd.DataFrame, spec: MetricSpec, epoch_hours: int) -> dict[str, float]:
    out = {
        "metric": spec.metric,
        "label": spec.label,
        "n_epochs": 0.0,
        "n_concordant_epochs": 0.0,
        "concordance_pct": np.nan,
        "n_patients_with_measurement": 0.0,
    }
    if rows.empty:
        return out

    d = rows.copy()
    d["vital_value"] = pd.to_numeric(d["vital_value"], errors="coerce")
    d = d[d["vital_value"].between(spec.plausible_lo, spec.plausible_hi, inclusive="both")].copy()
    if d.empty:
        return out

    d["epoch_start_hr"] = (np.floor(d["hours"] / float(epoch_hours)) * float(epoch_hours)).astype(int)
    ep = (
        d.groupby(["hospitalization_id", "epoch_start_hr"], as_index=False)["vital_value"]
        .median()
        .rename(columns={"vital_value": "epoch_median"})
    )
    if ep.empty:
        return out

    conc = _in_range(ep["epoch_median"], spec.lo, spec.hi)
    n_epochs = float(len(ep))
    n_conc = float(conc.sum())
    out["n_epochs"] = n_epochs
    out["n_concordant_epochs"] = n_conc
    out["concordance_pct"] = (n_conc / n_epochs * 100.0) if n_epochs > 0 else np.nan
    out["n_patients_with_measurement"] = float(ep["hospitalization_id"].nunique())
    return out


def epoch_median_concordance_by_epoch(
    rows: pd.DataFrame,
    spec: MetricSpec,
    epoch_hours: int,
    max_hours: int,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "label",
                "epoch_start_hr",
                "epoch_end_hr",
                "n_epochs",
                "n_concordant_epochs",
                "concordance_pct",
                "n_patients_with_measurement",
            ]
        )

    d = rows.copy()
    d["vital_value"] = pd.to_numeric(d["vital_value"], errors="coerce")
    d = d[d["vital_value"].between(spec.plausible_lo, spec.plausible_hi, inclusive="both")].copy()
    d = d[(d["hours"] >= 0.0) & (d["hours"] < float(max_hours))].copy()
    if d.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "label",
                "epoch_start_hr",
                "epoch_end_hr",
                "n_epochs",
                "n_concordant_epochs",
                "concordance_pct",
                "n_patients_with_measurement",
            ]
        )

    d["epoch_start_hr"] = (np.floor(d["hours"] / float(epoch_hours)) * float(epoch_hours)).astype(int)
    ep = (
        d.groupby(["hospitalization_id", "epoch_start_hr"], as_index=False)["vital_value"]
        .median()
        .rename(columns={"vital_value": "epoch_median"})
    )
    if ep.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "label",
                "epoch_start_hr",
                "epoch_end_hr",
                "n_epochs",
                "n_concordant_epochs",
                "concordance_pct",
                "n_patients_with_measurement",
            ]
        )

    ep["is_concordant"] = _in_range(ep["epoch_median"], spec.lo, spec.hi).astype(float)
    out = (
        ep.groupby("epoch_start_hr", as_index=False)
        .agg(
            n_epochs=("hospitalization_id", "size"),
            n_concordant_epochs=("is_concordant", "sum"),
            n_patients_with_measurement=("hospitalization_id", "nunique"),
        )
    )
    out["concordance_pct"] = np.where(
        out["n_epochs"] > 0,
        out["n_concordant_epochs"] / out["n_epochs"] * 100.0,
        np.nan,
    )
    out["epoch_end_hr"] = out["epoch_start_hr"] + int(epoch_hours)
    out["metric"] = spec.metric
    out["label"] = spec.label
    return out[
        [
            "metric",
            "label",
            "epoch_start_hr",
            "epoch_end_hr",
            "n_epochs",
            "n_concordant_epochs",
            "concordance_pct",
            "n_patients_with_measurement",
        ]
    ].sort_values("epoch_start_hr")


def _build_temp_rows_from_admission(cohort: pd.DataFrame, vitals: pd.DataFrame, max_hours: int) -> pd.DataFrame:
    if cohort.empty or vitals.empty:
        return pd.DataFrame(columns=["hospitalization_id", "hours", "vital_value"])
    temp = vitals[vitals["vital_category"].astype(str).eq("temp_c")][["hospitalization_id", "recorded_dttm", "vital_value"]].copy()
    if temp.empty:
        return pd.DataFrame(columns=["hospitalization_id", "hours", "vital_value"])
    c = cohort[["hospitalization_id", "admission_dttm"]].copy()
    c["hospitalization_id"] = c["hospitalization_id"].astype(str)
    c["admission_dttm"] = pd.to_datetime(c["admission_dttm"], errors="coerce")
    temp["hospitalization_id"] = temp["hospitalization_id"].astype(str)
    temp = temp.merge(c, on="hospitalization_id", how="inner")
    temp = temp.dropna(subset=["recorded_dttm", "admission_dttm", "vital_value"])
    if temp.empty:
        return pd.DataFrame(columns=["hospitalization_id", "hours", "vital_value"])
    temp["hours"] = (temp["recorded_dttm"] - temp["admission_dttm"]).dt.total_seconds() / 3600.0
    temp = temp[(temp["hours"] >= 0.0) & (temp["hours"] < float(max_hours))].copy()
    return temp[["hospitalization_id", "hours", "vital_value"]]


def _max_consecutive_epochs(epoch_starts_in_range: list[int], step: int) -> int:
    if not epoch_starts_in_range:
        return 0
    vals = sorted(epoch_starts_in_range)
    best = 1
    cur = 1
    for i in range(1, len(vals)):
        if vals[i] - vals[i - 1] == step:
            cur += 1
        else:
            cur = 1
        if cur > best:
            best = cur
    return best


def sustained_temp_concordance(
    rows: pd.DataFrame,
    spec: MetricSpec,
    epoch_hours: int,
    max_hours: int,
    required_hours: int = 24,
) -> dict[str, float]:
    out = {
        "metric": spec.metric,
        "label": spec.label,
        "n_epochs": 0.0,  # for sustained temp, this is patient denominator
        "n_concordant_epochs": 0.0,  # for sustained temp, this is concordant patients
        "concordance_pct": np.nan,
        "n_patients_with_measurement": 0.0,
    }
    if rows.empty:
        return out

    d = rows.copy()
    d["vital_value"] = pd.to_numeric(d["vital_value"], errors="coerce")
    d = d[d["vital_value"].between(spec.plausible_lo, spec.plausible_hi, inclusive="both")].copy()
    d = d[(d["hours"] >= 0.0) & (d["hours"] < float(max_hours))].copy()
    if d.empty:
        return out

    d["epoch_start_hr"] = (np.floor(d["hours"] / float(epoch_hours)) * float(epoch_hours)).astype(int)
    ep = (
        d.groupby(["hospitalization_id", "epoch_start_hr"], as_index=False)["vital_value"]
        .median()
        .rename(columns={"vital_value": "epoch_median"})
    )
    if ep.empty:
        return out

    ep["is_in"] = _in_range(ep["epoch_median"], spec.lo, spec.hi)
    required_epochs = int(np.ceil(float(required_hours) / float(epoch_hours)))
    by_pt = []
    for hid, g in ep.groupby("hospitalization_id", dropna=False):
        in_epochs = g.loc[g["is_in"], "epoch_start_hr"].astype(int).tolist()
        max_run = _max_consecutive_epochs(in_epochs, step=int(epoch_hours))
        by_pt.append((str(hid), max_run >= required_epochs))
    by_pt_df = pd.DataFrame(by_pt, columns=["hospitalization_id", "is_concordant"])
    denom = float(len(by_pt_df))
    conc = float(by_pt_df["is_concordant"].sum())
    out["n_epochs"] = denom
    out["n_concordant_epochs"] = conc
    out["concordance_pct"] = (conc / denom * 100.0) if denom > 0 else np.nan
    out["n_patients_with_measurement"] = denom
    return out


def sustained_temp_concordance_by_epoch(
    rows: pd.DataFrame,
    spec: MetricSpec,
    epoch_hours: int,
    max_hours: int,
    required_hours: int = 24,
) -> pd.DataFrame:
    cols = [
        "metric",
        "label",
        "epoch_start_hr",
        "epoch_end_hr",
        "n_epochs",
        "n_concordant_epochs",
        "concordance_pct",
        "n_patients_with_measurement",
    ]
    if rows.empty:
        return pd.DataFrame(columns=cols)

    d = rows.copy()
    d["vital_value"] = pd.to_numeric(d["vital_value"], errors="coerce")
    d = d[d["vital_value"].between(spec.plausible_lo, spec.plausible_hi, inclusive="both")].copy()
    d = d[(d["hours"] >= 0.0) & (d["hours"] < float(max_hours))].copy()
    if d.empty:
        return pd.DataFrame(columns=cols)

    d["epoch_start_hr"] = (np.floor(d["hours"] / float(epoch_hours)) * float(epoch_hours)).astype(int)
    ep = (
        d.groupby(["hospitalization_id", "epoch_start_hr"], as_index=False)["vital_value"]
        .median()
        .rename(columns={"vital_value": "epoch_median"})
    )
    if ep.empty:
        return pd.DataFrame(columns=cols)

    ep["is_in"] = _in_range(ep["epoch_median"], spec.lo, spec.hi)
    epoch_starts = list(range(0, int(max_hours), int(epoch_hours)))
    piv = ep.pivot_table(index="hospitalization_id", columns="epoch_start_hr", values="is_in", aggfunc="max")
    piv = piv.reindex(columns=epoch_starts, fill_value=False).fillna(False).astype(bool)
    if piv.empty:
        return pd.DataFrame(columns=cols)

    required_epochs = int(np.ceil(float(required_hours) / float(epoch_hours)))
    denom = float(len(piv))
    rows_out = []
    arr = piv.to_numpy(dtype=bool)

    for j, ep_start in enumerate(epoch_starts):
        conc_count = 0
        upto = arr[:, : j + 1]
        for r in upto:
            run = 0
            best = 0
            for v in r:
                if v:
                    run += 1
                    if run > best:
                        best = run
                else:
                    run = 0
            if best >= required_epochs:
                conc_count += 1
        rows_out.append(
            {
                "metric": spec.metric,
                "label": spec.label,
                "epoch_start_hr": int(ep_start),
                "epoch_end_hr": int(ep_start + int(epoch_hours)),
                "n_epochs": denom,  # patient denominator for sustained criterion
                "n_concordant_epochs": float(conc_count),  # concordant patients by epoch
                "concordance_pct": (float(conc_count) / denom * 100.0) if denom > 0 else np.nan,
                "n_patients_with_measurement": denom,
            }
        )
    return pd.DataFrame(rows_out, columns=cols)


def analyze_site(site_label: str, data_dir: Path, max_hours: int, epoch_hours: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort = build_ohca_icu_first_encounter(data_dir)
    cohort["site"] = site_label

    time_zero, vitals = get_time_zero_and_vitals(data_dir, cohort, max_hours=max_hours)
    glucose = get_glucose_rows(data_dir, cohort, time_zero, max_hours=max_hours)
    temp_adm = _build_temp_rows_from_admission(cohort, vitals, max_hours=max_hours)

    metric_rows: list[dict[str, float | str]] = []
    epoch_rows: list[pd.DataFrame] = []
    for spec in METRICS:
        if spec.metric == "glucose":
            rows = glucose
            r = epoch_median_concordance(rows, spec=spec, epoch_hours=epoch_hours)
            e = epoch_median_concordance_by_epoch(rows, spec=spec, epoch_hours=epoch_hours, max_hours=max_hours)
        elif spec.metric == "temp_c":
            rows = temp_adm
            r = sustained_temp_concordance(rows, spec=spec, epoch_hours=epoch_hours, max_hours=max_hours, required_hours=24)
            e = sustained_temp_concordance_by_epoch(rows, spec=spec, epoch_hours=epoch_hours, max_hours=max_hours, required_hours=24)
        else:
            rows = vitals[vitals["vital_category"].astype(str).eq(spec.metric)][["hospitalization_id", "hours", "vital_value"]].copy()
            r = epoch_median_concordance(rows, spec=spec, epoch_hours=epoch_hours)
            e = epoch_median_concordance_by_epoch(rows, spec=spec, epoch_hours=epoch_hours, max_hours=max_hours)
        r["site"] = site_label
        metric_rows.append(r)
        if not e.empty:
            e["site"] = site_label
            epoch_rows.append(e)

    site_metrics = pd.DataFrame(metric_rows)
    site_metrics["n_cohort_encounters"] = float(cohort["hospitalization_id"].nunique())
    site_metrics["n_with_time_zero"] = float(time_zero["hospitalization_id"].nunique())
    site_epochs = (
        pd.concat(epoch_rows, ignore_index=True)
        if epoch_rows
        else pd.DataFrame(
            columns=[
                "site",
                "metric",
                "label",
                "epoch_start_hr",
                "epoch_end_hr",
                "n_epochs",
                "n_concordant_epochs",
                "concordance_pct",
                "n_patients_with_measurement",
            ]
        )
    )
    return cohort, site_metrics, site_epochs


def pooled_from_sites(site_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in METRICS:
        sub = site_metrics[site_metrics["metric"] == spec.metric].copy()
        n_epochs = float(sub["n_epochs"].sum())
        n_conc = float(sub["n_concordant_epochs"].sum())
        pct = (n_conc / n_epochs * 100.0) if n_epochs > 0 else np.nan
        n_cohort = float(sub["n_cohort_encounters"].sum())
        n_tz = float(sub["n_with_time_zero"].sum())
        rows.append(
            {
                "site": "pooled_ucmc_mimic",
                "metric": spec.metric,
                "label": spec.label,
                "n_epochs": n_epochs,
                "n_concordant_epochs": n_conc,
                "concordance_pct": pct,
                "n_patients_with_measurement": float(sub["n_patients_with_measurement"].sum()),
                "n_cohort_encounters": n_cohort,
                "n_with_time_zero": n_tz,
            }
        )
    return pd.DataFrame(rows)


def pooled_epoch_from_sites(site_epochs: pd.DataFrame) -> pd.DataFrame:
    if site_epochs.empty:
        return pd.DataFrame(
            columns=[
                "site",
                "metric",
                "label",
                "epoch_start_hr",
                "epoch_end_hr",
                "n_epochs",
                "n_concordant_epochs",
                "concordance_pct",
                "n_patients_with_measurement",
            ]
        )

    out = (
        site_epochs.groupby(["metric", "label", "epoch_start_hr", "epoch_end_hr"], as_index=False)
        .agg(
            n_epochs=("n_epochs", "sum"),
            n_concordant_epochs=("n_concordant_epochs", "sum"),
            n_patients_with_measurement=("n_patients_with_measurement", "sum"),
        )
    )
    out["concordance_pct"] = np.where(
        out["n_epochs"] > 0,
        out["n_concordant_epochs"] / out["n_epochs"] * 100.0,
        np.nan,
    )
    out["site"] = "pooled_ucmc_mimic"
    return out[
        [
            "site",
            "metric",
            "label",
            "epoch_start_hr",
            "epoch_end_hr",
            "n_epochs",
            "n_concordant_epochs",
            "concordance_pct",
            "n_patients_with_measurement",
        ]
    ].sort_values(["metric", "epoch_start_hr"])


def plot_site_grouped(site_metrics: pd.DataFrame, pooled: pd.DataFrame, output_png: Path, max_hours: int) -> None:
    labels = [m.label for m in METRICS]
    order = [m.metric for m in METRICS]
    sites = ["ucmc", "mimic", "pooled_ucmc_mimic"]
    colors = {
        "ucmc": "#1E88E5",
        "mimic": "#43A047",
        "pooled_ucmc_mimic": "#455A64",
    }

    wide = (
        pd.concat([site_metrics, pooled], ignore_index=True)
        .assign(metric=lambda d: pd.Categorical(d["metric"], categories=order, ordered=True))
        .sort_values(["metric", "site"])
    )

    x = np.arange(len(order))
    w = 0.24
    fig, ax = plt.subplots(figsize=(12, 6.6))

    for i, site in enumerate(sites):
        d = wide[wide["site"] == site].set_index("metric").reindex(order)
        y = d["concordance_pct"].to_numpy(dtype=float)
        xpos = x + (i - 1) * w
        bars = ax.bar(xpos, y, width=w, color=colors[site], alpha=0.9, label=site)
        for bar, (_, row) in zip(bars, d.iterrows()):
            yp = float(row["concordance_pct"]) if np.isfinite(row["concordance_pct"]) else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yp + 1.5,
                f"{yp:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Patient-Epoch Concordance (%)")
    ax.set_title(f"Patient-Level Guideline Concordance (6h Epoch Medians, First {max_hours}h)")
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.8, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right", title="Site")

    ax.text(
        0.01,
        -0.19,
        "Glucose/MAP/SpO2 use measured patient-epochs. Temp uses patient-level sustained target (>=24h in 32-36 C).",
        transform=ax.transAxes,
        ha="left",
        fontsize=9,
        color="#4A4A4A",
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_trends(site_epochs: pd.DataFrame, output_png: Path, max_hours: int, epoch_hours: int) -> None:
    order = [m.metric for m in METRICS]
    label_map = {m.metric: m.label for m in METRICS}
    colors = {"ucmc": "#1E88E5", "mimic": "#43A047"}
    sites = ["ucmc", "mimic"]

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.1), sharex=True, sharey=True)
    axes = axes.flatten()
    x_ticks = np.arange(0, max_hours, epoch_hours)

    for ax, metric in zip(axes, order):
        sub = site_epochs[site_epochs["metric"] == metric].copy()
        for site in sites:
            d = sub[sub["site"] == site].sort_values("epoch_start_hr")
            if d.empty:
                continue
            x = d["epoch_start_hr"].to_numpy(dtype=float)
            y = d["concordance_pct"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=2.1, markersize=4.5, color=colors[site], label=site)

        title = label_map[metric]
        if metric == "temp_c":
            title = "Temp 32.0-36.0 C for >=24h (cumulative)"
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.set_xticks(x_ticks)
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.8, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[2:]:
        ax.set_xlabel("Epoch start (hours)")
    for ax in [axes[0], axes[2]]:
        ax.set_ylabel("Concordance (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, title="Site")

    fig.suptitle(f"Guideline Concordance by 6h Epoch (Patient-Level, First {max_hours}h)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sites = {
        "ucmc": args.ucmc_dir,
        "mimic": args.mimic_dir,
    }

    all_site_metrics = []
    all_site_epochs = []
    cohort_counts = []
    for site_label, data_dir in sites.items():
        cohort, site_metrics, site_epochs = analyze_site(site_label, data_dir, max_hours=args.max_hours, epoch_hours=args.epoch_hours)
        all_site_metrics.append(site_metrics)
        all_site_epochs.append(site_epochs)
        cohort_counts.append(
            {
                "site": site_label,
                "n_cohort_encounters": int(cohort["hospitalization_id"].nunique()),
                "n_survivor": int(cohort["survival_status"].eq("Survivor").sum()),
                "n_non_survivor": int(cohort["survival_status"].eq("Non-Survivor").sum()),
            }
        )

    site_metrics = pd.concat(all_site_metrics, ignore_index=True)
    site_epochs = pd.concat(all_site_epochs, ignore_index=True) if all_site_epochs else pd.DataFrame()
    pooled = pooled_from_sites(site_metrics)
    pooled_epochs = pooled_epoch_from_sites(site_epochs)
    cohort_counts_df = pd.DataFrame(cohort_counts)

    site_csv = args.output_dir / f"figure5b_guideline_concordance_patientlevel_by_site_{args.max_hours}h.csv"
    pooled_csv = args.output_dir / f"figure5b_guideline_concordance_patientlevel_pooled_{args.max_hours}h.csv"
    cohort_csv = args.output_dir / f"figure5b_guideline_concordance_patientlevel_cohort_counts_{args.max_hours}h.csv"
    fig_png = args.output_dir / f"figure5b_guideline_concordance_patientlevel_{args.max_hours}h.png"
    epoch_site_csv = args.output_dir / f"figure5c_guideline_concordance_patientlevel_by_epoch_{args.max_hours}h.csv"
    epoch_pooled_csv = args.output_dir / f"figure5c_guideline_concordance_patientlevel_pooled_by_epoch_{args.max_hours}h.csv"
    epoch_fig_png = args.output_dir / f"figure5c_guideline_concordance_patientlevel_by_epoch_{args.max_hours}h.png"

    site_metrics.to_csv(site_csv, index=False)
    pooled.to_csv(pooled_csv, index=False)
    cohort_counts_df.to_csv(cohort_csv, index=False)
    site_epochs.to_csv(epoch_site_csv, index=False)
    pooled_epochs.to_csv(epoch_pooled_csv, index=False)
    plot_site_grouped(site_metrics, pooled, fig_png, max_hours=args.max_hours)
    plot_epoch_trends(site_epochs, epoch_fig_png, max_hours=args.max_hours, epoch_hours=args.epoch_hours)

    print("\nSaved patient-level guideline concordance outputs:")
    for p in [fig_png, site_csv, pooled_csv, cohort_csv, epoch_fig_png, epoch_site_csv, epoch_pooled_csv]:
        print(f"  - {p}")

    print("\nCohort counts:")
    print(cohort_counts_df.to_string(index=False))

    print("\nSite-level concordance (%):")
    print(
        site_metrics[["site", "metric", "concordance_pct", "n_epochs"]]
        .sort_values(["metric", "site"])
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    print("\nEpoch-level concordance preview (%):")
    prev = (
        site_epochs[["site", "metric", "epoch_start_hr", "concordance_pct", "n_epochs"]]
        .sort_values(["metric", "site", "epoch_start_hr"])
        .head(24)
    )
    print(prev.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
