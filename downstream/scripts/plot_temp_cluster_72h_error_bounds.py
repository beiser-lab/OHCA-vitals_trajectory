#!/usr/bin/env python3
"""
Plot 72h cluster temperature trajectories with uncertainty bands.

Uses the same hourly construction logic as clustering (time-zero at first measurement,
1-hour bins, plausibility filtering via imported helper) and then summarizes
temperature by cluster-hour with 95% confidence intervals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster_patientlevel_ucmc_mimic import build_hourly_from_long


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 72h cluster temperature trajectories with error bounds.")
    parser.add_argument(
        "--cluster-assignments",
        type=Path,
        default=Path("output_dir/patient_level_clustering_ucmc_mimic/patient_cluster_assignments.csv"),
        help="Path to patient_cluster_assignments.csv",
    )
    parser.add_argument(
        "--ucmc-dir",
        type=Path,
        default=Path("/Users/davidbeiser/Library/CloudStorage/Box-Box/RCLIF_data/CLIF_2018_24/2.1.0"),
        help="Path to CLIF parquet directory.",
    )
    parser.add_argument(
        "--mimic-vitals-cache",
        type=Path,
        default=Path("output_dir/patient_level_clustering_ucmc_mimic/mimic_vitals_cache.parquet"),
        help="Path to MIMIC long-form vitals cache parquet from clustering run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_dir/patient_level_clustering_ucmc_mimic/gcs_analysis_72h_final120"),
        help="Output directory for figure and summary CSV.",
    )
    parser.add_argument("--max-hours", type=int, default=72, help="Hours from first vital to include.")
    parser.add_argument("--bin-hours", type=int, default=1, help="Hour bin width.")
    return parser.parse_args()


def _id_to_str(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.str.replace(r"\.0$", "", regex=True)


def _load_ucmc_temp_hourly(ucmc_dir: Path, cohort: pd.DataFrame, max_hours: int, bin_hours: int) -> pd.DataFrame:
    sub = cohort[cohort["site"].eq("ucmc")].copy()
    if sub.empty:
        return pd.DataFrame(columns=["patient_key", "hour", "temp_c"])

    hosp_ids = sub["hospitalization_id"].astype(str).unique().tolist()
    vit = pd.read_parquet(
        ucmc_dir / "clif_vitals.parquet",
        columns=["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"],
        filters=[
            ("hospitalization_id", "in", hosp_ids),
            ("vital_category", "in", ["temp_c"]),
        ],
    )
    vit["hospitalization_id"] = _id_to_str(vit["hospitalization_id"])
    vit["vital_category"] = vit["vital_category"].astype("string").str.strip().str.lower()

    hourly = build_hourly_from_long(vit, sub, max_hours=max_hours, bin_hours=bin_hours)
    return hourly[["patient_key", "hour", "temp_c"]].copy()


def _load_mimic_temp_hourly(mimic_cache: Path, cohort: pd.DataFrame, max_hours: int, bin_hours: int) -> pd.DataFrame:
    sub = cohort[cohort["site"].eq("mimic")].copy()
    if sub.empty:
        return pd.DataFrame(columns=["patient_key", "hour", "temp_c"])
    if not mimic_cache.exists():
        raise FileNotFoundError(f"MIMIC vitals cache not found: {mimic_cache}")

    hadm_ids = set(sub["hospitalization_id"].astype(str).unique())
    vit = pd.read_parquet(mimic_cache)
    vit["hospitalization_id"] = _id_to_str(vit["hospitalization_id"])
    vit["vital_category"] = vit["vital_category"].astype("string").str.strip().str.lower()
    vit = vit[
        vit["hospitalization_id"].isin(hadm_ids)
        & vit["vital_category"].eq("temp_c")
    ].copy()

    hourly = build_hourly_from_long(vit, sub, max_hours=max_hours, bin_hours=bin_hours)
    return hourly[["patient_key", "hour", "temp_c"]].copy()


def _summarize_temp(hourly_with_cluster: pd.DataFrame) -> pd.DataFrame:
    out = (
        hourly_with_cluster.groupby(["cluster", "hour"], as_index=False)["temp_c"]
        .agg(n="count", mean_temp_c="mean", std_temp_c="std")
        .sort_values(["cluster", "hour"])
    )
    out["se_temp_c"] = out["std_temp_c"] / np.sqrt(out["n"].clip(lower=1))
    out["se_temp_c"] = out["se_temp_c"].fillna(0.0)
    out["ci95_lo_temp_c"] = out["mean_temp_c"] - 1.96 * out["se_temp_c"]
    out["ci95_hi_temp_c"] = out["mean_temp_c"] + 1.96 * out["se_temp_c"]
    return out


def _plot_temp_with_ci(summary: pd.DataFrame, out_png: Path, title: str) -> None:
    clusters = sorted(summary["cluster"].dropna().astype(int).unique().tolist())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for i, c in enumerate(clusters):
        sub = summary[summary["cluster"].astype(int) == c].sort_values("hour")
        if sub.empty:
            continue
        color = cmap((i % 10) / 10.0)
        ax.plot(sub["hour"], sub["mean_temp_c"], color=color, lw=2.2, label=f"Cluster {c}")
        ax.fill_between(
            sub["hour"].to_numpy(dtype=float),
            sub["ci95_lo_temp_c"].to_numpy(dtype=float),
            sub["ci95_hi_temp_c"].to_numpy(dtype=float),
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlim(0, 71)
    ax.set_xlabel("Hour from time-zero")
    ax.set_ylabel("Temperature (C)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assign = pd.read_csv(args.cluster_assignments)
    required = {"patient_key", "site", "hospitalization_id", "cluster"}
    miss = required - set(assign.columns)
    if miss:
        raise ValueError(f"Missing required columns in assignments: {sorted(miss)}")

    assign["patient_key"] = assign["patient_key"].astype("string")
    assign["site"] = assign["site"].astype("string").str.strip().str.lower()
    assign["hospitalization_id"] = _id_to_str(assign["hospitalization_id"])
    assign["cluster"] = pd.to_numeric(assign["cluster"], errors="coerce")
    assign = assign.dropna(subset=["patient_key", "site", "hospitalization_id", "cluster"]).copy()
    assign["cluster"] = assign["cluster"].astype(int)

    cohort_cols = [c for c in ["patient_key", "site", "patient_id", "hospitalization_id", "survival_status"] if c in assign.columns]
    cohort = assign[cohort_cols].drop_duplicates(subset=["patient_key"]).copy()

    print("[1/4] Loading UCMC hourly temp...")
    ucmc_hourly = _load_ucmc_temp_hourly(args.ucmc_dir, cohort, max_hours=args.max_hours, bin_hours=args.bin_hours)
    print(f"      UCMC patient-hour rows: {len(ucmc_hourly):,}")

    print("[2/4] Loading MIMIC hourly temp...")
    mimic_hourly = _load_mimic_temp_hourly(args.mimic_vitals_cache, cohort, max_hours=args.max_hours, bin_hours=args.bin_hours)
    print(f"      MIMIC patient-hour rows: {len(mimic_hourly):,}")

    print("[3/4] Summarizing by cluster-hour with 95% CI...")
    hourly = pd.concat([ucmc_hourly, mimic_hourly], ignore_index=True)
    if hourly.empty:
        raise RuntimeError("No hourly temperature rows loaded.")

    linked = hourly.merge(assign[["patient_key", "cluster"]].drop_duplicates(), on="patient_key", how="inner")
    summary = _summarize_temp(linked)
    out_csv = args.output_dir / "temperature_cluster_72h_with_95ci.csv"
    summary.to_csv(out_csv, index=False)

    print("[4/4] Plotting...")
    out_png = args.output_dir / "figure_temperature_cluster_72h_with_95ci.png"
    _plot_temp_with_ci(summary, out_png, "72h Cluster Temperature Trajectories with 95% CI")

    print(f"Done. CSV: {out_csv}")
    print(f"Done. PNG: {out_png}")


if __name__ == "__main__":
    main()
