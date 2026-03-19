#!/usr/bin/env python3
"""
Link UCMC+MIMIC temperature cluster assignments to Glasgow Coma Scale (GCS).

Features:
1) Supports patient-level cluster assignments directly.
2) Supports patient-time assignments by collapsing to dominant patient cluster.
3) Pulls UCMC GCS from CLIF `clif_patient_assessments.parquet`.
4) Pulls MIMIC GCS components from `chartevents.csv.gz` and derives GCS total.
5) Writes coverage-focused outputs and summary figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UCMC_GCS_CATEGORIES = ["gcs_eye", "gcs_motor", "gcs_verbal", "gcs_total"]
MIMIC_GCS_ITEM_TO_COMPONENT = {
    220739: "gcs_eye",
    223901: "gcs_motor",
    223900: "gcs_verbal",
}
COMPONENT_BOUNDS = {
    "gcs_eye": (1.0, 4.0),
    "gcs_motor": (1.0, 6.0),
    "gcs_verbal": (1.0, 5.0),
}
COMPONENT_COLS = ["gcs_eye", "gcs_motor", "gcs_verbal"]
ALL_GCS_COLS = COMPONENT_COLS + ["gcs_total"]
TOTAL_METRIC_COLS = ["gcs_total_initial", "gcs_total_final", "gcs_total_min", "gcs_total_max"]
METRIC_ORDER = ["initial", "final", "min", "max"]
VITAL_LABELS = {
    "heart_rate": "Heart Rate (bpm)",
    "map": "MAP (mmHg)",
    "spo2": "SpO2 (%)",
    "temp_c": "Temperature (C)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze association between UCMC+MIMIC clusters and GCS (focus: gcs_total)."
    )
    parser.add_argument(
        "--cluster-assignments",
        type=Path,
        default=Path("output_dir/patient_level_clustering_ucmc_mimic_24h_full/patient_cluster_assignments.csv"),
        help="Path to patient_cluster_assignments.csv or timepoint_cluster_assignments.csv.",
    )
    parser.add_argument(
        "--ucmc-dir",
        type=Path,
        default=Path("/Users/davidbeiser/Library/CloudStorage/Box-Box/RCLIF_data/CLIF_2018_24/2.1.0"),
        help="Path containing CLIF parquet tables.",
    )
    parser.add_argument(
        "--mimic-dir",
        type=Path,
        default=Path("/Users/davidbeiser/mimic-iv-3.1"),
        help="Path containing MIMIC-IV folders (hosp/, icu/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <cluster-assignments parent>/gcs_analysis).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="Chunk size when streaming MIMIC chartevents.",
    )
    parser.add_argument(
        "--max-mimic-chunks",
        type=int,
        default=None,
        help="Optional cap on processed MIMIC chartevents chunks (for smoke tests).",
    )
    parser.add_argument(
        "--mimic-gcs-cache",
        type=Path,
        default=None,
        help="Optional parquet cache for filtered MIMIC GCS component rows.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="UCMC+MIMIC Clusters vs GCS",
        help="Figure title prefix.",
    )
    parser.add_argument(
        "--cluster-hourly-csv",
        type=Path,
        default=None,
        help="Optional path to cluster_hourly_profiles.csv (default: sibling of cluster assignments).",
    )
    parser.add_argument(
        "--trajectory-vital",
        type=str,
        default="temp_c",
        choices=["heart_rate", "map", "spo2", "temp_c"],
        help="Vital to display for cluster trajectory panel.",
    )
    parser.add_argument(
        "--final-target-hour",
        type=float,
        default=None,
        help="If set, define final gcs_total as value closest to this hour from first gcs_total; fallback remains last recorded.",
    )
    return parser.parse_args()


def _id_to_str(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.str.replace(r"\.0$", "", regex=True)


def _read_assignments(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Cluster assignments file not found: {path}")
    raw = pd.read_csv(path)

    required = {"patient_key", "site", "hospitalization_id", "cluster"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Cluster assignments missing required columns: {sorted(missing)}")

    raw["patient_key"] = raw["patient_key"].astype("string")
    raw["site"] = raw["site"].astype("string").str.strip().str.lower()
    raw["hospitalization_id"] = _id_to_str(raw["hospitalization_id"])
    raw["cluster"] = pd.to_numeric(raw["cluster"], errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["patient_key", "site", "hospitalization_id", "cluster"]).copy()
    raw["cluster"] = raw["cluster"].astype(int)

    meta_cols = [c for c in ["patient_key", "site", "patient_id", "hospitalization_id", "survival_status"] if c in raw.columns]

    if raw["patient_key"].nunique() == len(raw):
        assign = raw[meta_cols + ["cluster"]].copy()
        assign = assign.drop_duplicates(subset=["patient_key"], keep="first")
        return assign.sort_values("patient_key").reset_index(drop=True), "patient_level"

    cluster_counts = (
        raw.groupby(["patient_key", "cluster"], as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
        .sort_values(["patient_key", "n_rows", "cluster"], ascending=[True, False, True])
    )
    dominant = cluster_counts.drop_duplicates(subset=["patient_key"], keep="first")[["patient_key", "cluster", "n_rows"]]
    meta = raw[meta_cols].drop_duplicates(subset=["patient_key"], keep="first")
    assign = meta.merge(dominant, on="patient_key", how="inner")
    assign = assign.drop(columns=["n_rows"])
    return assign.sort_values("patient_key").reset_index(drop=True), "dominant_from_timepoints"


def _empty_site_gcs_frame(hosp_ids: list[str], site: str) -> pd.DataFrame:
    cols = (
        ["site", "hospitalization_id"]
        + ALL_GCS_COLS
        + TOTAL_METRIC_COLS
        + ["gcs_total_n_obs", "gcs_total_source", "gcs_total_final_source"]
    )
    base = pd.DataFrame({"hospitalization_id": pd.Series(hosp_ids, dtype="string")})
    base["site"] = site
    for c in ALL_GCS_COLS + TOTAL_METRIC_COLS:
        base[c] = np.nan
    base["gcs_total_n_obs"] = 0
    base["gcs_total_source"] = "missing"
    base["gcs_total_final_source"] = "missing"
    return base[cols]


def _compute_total_metrics(total_rows: pd.DataFrame, hosp_ids: list[str], final_target_hour: float | None) -> pd.DataFrame:
    base = pd.DataFrame({"hospitalization_id": pd.Series(hosp_ids, dtype="string")})
    if total_rows.empty:
        for c in TOTAL_METRIC_COLS:
            base[c] = np.nan
        base["gcs_total_n_obs"] = 0
        base["gcs_total_final_source"] = "missing"
        return base

    use = total_rows.copy()
    use["hospitalization_id"] = _id_to_str(use["hospitalization_id"])
    use["recorded_dttm"] = pd.to_datetime(use["recorded_dttm"], errors="coerce")
    use["gcs_total"] = pd.to_numeric(use["gcs_total"], errors="coerce")
    use = use.dropna(subset=["hospitalization_id", "recorded_dttm", "gcs_total"])
    if use.empty:
        for c in TOTAL_METRIC_COLS:
            base[c] = np.nan
        base["gcs_total_n_obs"] = 0
        base["gcs_total_final_source"] = "missing"
        return base

    use = use.sort_values(["hospitalization_id", "recorded_dttm"])
    first = use.drop_duplicates(subset=["hospitalization_id"], keep="first")[
        ["hospitalization_id", "recorded_dttm", "gcs_total"]
    ].rename(columns={"gcs_total": "gcs_total_initial", "recorded_dttm": "gcs_total_initial_dttm"})
    last = use.drop_duplicates(subset=["hospitalization_id"], keep="last")[
        ["hospitalization_id", "gcs_total"]
    ].rename(columns={"gcs_total": "gcs_total_final"})
    agg = (
        use.groupby("hospitalization_id", as_index=False)["gcs_total"]
        .agg(gcs_total_min="min", gcs_total_max="max", gcs_total_n_obs="size")
    )

    if final_target_hour is not None:
        use_t = use.merge(
            first[["hospitalization_id", "gcs_total_initial_dttm"]],
            on="hospitalization_id",
            how="left",
        )
        use_t["hours_from_initial"] = (
            (use_t["recorded_dttm"] - use_t["gcs_total_initial_dttm"]).dt.total_seconds() / 3600.0
        )
        use_t["abs_target_diff"] = (use_t["hours_from_initial"] - float(final_target_hour)).abs()
        chosen = (
            use_t.sort_values(
                ["hospitalization_id", "abs_target_diff", "recorded_dttm"],
                ascending=[True, True, True],
            )
            .drop_duplicates(subset=["hospitalization_id"], keep="first")
            [["hospitalization_id", "gcs_total"]]
            .rename(columns={"gcs_total": "gcs_total_final"})
        )
        chosen["used_closest"] = True
        last_fallback = last.rename(columns={"gcs_total_final": "gcs_total_final_fallback"})
        final_pick = chosen.merge(last_fallback, on="hospitalization_id", how="outer")
        final_pick["gcs_total_final"] = final_pick["gcs_total_final"].fillna(final_pick["gcs_total_final_fallback"])
        final_pick["gcs_total_final_source"] = np.where(
            final_pick["used_closest"].fillna(False),
            f"closest_to_{float(final_target_hour):g}h",
            "last_recorded",
        )
        final_pick = final_pick[["hospitalization_id", "gcs_total_final", "gcs_total_final_source"]]
    else:
        final_pick = last.copy()
        final_pick["gcs_total_final_source"] = "last_recorded"

    out = base.merge(first, on="hospitalization_id", how="left")
    out = out.merge(final_pick, on="hospitalization_id", how="left")
    out = out.merge(agg, on="hospitalization_id", how="left")
    if "gcs_total_initial_dttm" in out.columns:
        out = out.drop(columns=["gcs_total_initial_dttm"])
    out["gcs_total_n_obs"] = out["gcs_total_n_obs"].fillna(0).astype(int)
    out["gcs_total_final_source"] = out["gcs_total_final_source"].fillna("missing")
    return out


def load_ucmc_gcs(ucmc_dir: Path, hosp_ids: list[str], final_target_hour: float | None) -> pd.DataFrame:
    cols = (
        ["site", "hospitalization_id"]
        + ALL_GCS_COLS
        + TOTAL_METRIC_COLS
        + ["gcs_total_n_obs", "gcs_total_source", "gcs_total_final_source"]
    )
    hosp_ids = pd.Series(hosp_ids, dtype="string").dropna().astype(str).unique().tolist()
    if not hosp_ids:
        return pd.DataFrame(columns=cols)

    path = ucmc_dir / "clif_patient_assessments.parquet"
    if not path.exists():
        raise FileNotFoundError(f"UCMC assessments table not found: {path}")

    raw = pd.read_parquet(
        path,
        columns=["hospitalization_id", "recorded_dttm", "assessment_category", "numerical_value"],
        filters=[
            ("hospitalization_id", "in", hosp_ids),
            ("assessment_category", "in", UCMC_GCS_CATEGORIES),
        ],
    )
    if raw.empty:
        return _empty_site_gcs_frame(hosp_ids, "ucmc")[cols]

    raw["hospitalization_id"] = _id_to_str(raw["hospitalization_id"])
    raw["recorded_dttm"] = pd.to_datetime(raw["recorded_dttm"], errors="coerce")
    raw["assessment_category"] = raw["assessment_category"].astype("string").str.strip().str.lower()
    raw["numerical_value"] = pd.to_numeric(raw["numerical_value"], errors="coerce")
    raw = raw.dropna(subset=["hospitalization_id", "recorded_dttm", "assessment_category", "numerical_value"])

    base = pd.DataFrame({"hospitalization_id": pd.Series(hosp_ids, dtype="string")})

    for comp in COMPONENT_COLS:
        lo, hi = COMPONENT_BOUNDS[comp]
        sub = raw[raw["assessment_category"].eq(comp)].copy()
        sub = sub[sub["numerical_value"].between(lo, hi)]
        sub = sub.sort_values(["hospitalization_id", "recorded_dttm"])
        if sub.empty:
            base[comp] = np.nan
            continue
        first = sub.drop_duplicates(subset=["hospitalization_id"], keep="first")
        first = first[["hospitalization_id", "numerical_value"]].rename(columns={"numerical_value": comp})
        base = base.merge(first, on="hospitalization_id", how="left")

    comp_rows = raw[raw["assessment_category"].isin(COMPONENT_COLS)].copy()
    comp_wide = (
        comp_rows.groupby(["hospitalization_id", "recorded_dttm", "assessment_category"], as_index=False)["numerical_value"]
        .median()
        .pivot_table(
            index=["hospitalization_id", "recorded_dttm"],
            columns="assessment_category",
            values="numerical_value",
            aggfunc="first",
        )
        .reset_index()
    )
    comp_wide.columns.name = None
    for comp in COMPONENT_COLS:
        if comp not in comp_wide.columns:
            comp_wide[comp] = np.nan
    for comp, (lo, hi) in COMPONENT_BOUNDS.items():
        comp_wide.loc[~comp_wide[comp].between(lo, hi), comp] = np.nan
    comp_wide["gcs_total"] = comp_wide[COMPONENT_COLS].sum(axis=1, min_count=3)
    total_rows = comp_wide.dropna(subset=["gcs_total"])[["hospitalization_id", "recorded_dttm", "gcs_total"]].copy()
    total_metrics = _compute_total_metrics(total_rows, hosp_ids, final_target_hour=final_target_hour)
    base = base.merge(total_metrics, on="hospitalization_id", how="left")

    has_total = base["gcs_total_initial"].notna()
    base["gcs_total_source"] = np.where(has_total, "derived_components", "missing")
    base["gcs_total"] = base["gcs_total_initial"]
    base["site"] = "ucmc"
    return base[cols]


def _parse_numeric_from_text(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _extract_mimic_gcs_rows(
    mimic_dir: Path,
    hadm_ids: list[str],
    chunk_size: int,
    max_mimic_chunks: int | None,
    cache_path: Path | None,
) -> pd.DataFrame:
    cols = ["hospitalization_id", "recorded_dttm", "component", "gcs_value"]
    if not hadm_ids:
        return pd.DataFrame(columns=cols)

    hadm_set = set(pd.Series(hadm_ids, dtype="string").tolist())
    itemids = sorted(MIMIC_GCS_ITEM_TO_COMPONENT.keys())

    if cache_path is not None and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached["hospitalization_id"] = _id_to_str(cached["hospitalization_id"])
        cached = cached[cached["hospitalization_id"].isin(hadm_set)].copy()
        return cached[cols]

    src = mimic_dir / "icu" / "chartevents.csv.gz"
    if not src.exists():
        raise FileNotFoundError(f"MIMIC chartevents not found: {src}")

    out_chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(
        src,
        usecols=["hadm_id", "charttime", "itemid", "valuenum", "value"],
        dtype={
            "hadm_id": "string",
            "charttime": "string",
            "itemid": "Int64",
            "valuenum": "float64",
            "value": "string",
        },
        chunksize=chunk_size,
        low_memory=True,
    )

    for idx, chunk in enumerate(reader, start=1):
        if max_mimic_chunks is not None and idx > max_mimic_chunks:
            break

        chunk = chunk.dropna(subset=["hadm_id", "itemid"])
        chunk = chunk[chunk["itemid"].astype(int).isin(itemids)]
        if chunk.empty:
            if idx % 50 == 0:
                print(f"[mimic-gcs] processed {idx:,} chunks...")
            continue

        chunk = chunk[chunk["hadm_id"].astype("string").isin(hadm_set)]
        if chunk.empty:
            if idx % 50 == 0:
                print(f"[mimic-gcs] processed {idx:,} chunks...")
            continue

        chunk["hospitalization_id"] = _id_to_str(chunk["hadm_id"])
        chunk["component"] = chunk["itemid"].astype(int).map(MIMIC_GCS_ITEM_TO_COMPONENT)
        chunk["recorded_dttm"] = pd.to_datetime(chunk["charttime"], errors="coerce")

        vnum = pd.to_numeric(chunk["valuenum"], errors="coerce")
        parsed = _parse_numeric_from_text(chunk["value"])
        chunk["gcs_value"] = vnum.fillna(parsed)

        chunk = chunk.dropna(subset=["hospitalization_id", "recorded_dttm", "component", "gcs_value"])
        if chunk.empty:
            if idx % 50 == 0:
                print(f"[mimic-gcs] processed {idx:,} chunks...")
            continue

        keep = np.zeros(len(chunk), dtype=bool)
        for comp, (lo, hi) in COMPONENT_BOUNDS.items():
            keep |= (chunk["component"].eq(comp) & chunk["gcs_value"].between(lo, hi)).to_numpy()
        chunk = chunk[keep]
        if chunk.empty:
            if idx % 50 == 0:
                print(f"[mimic-gcs] processed {idx:,} chunks...")
            continue

        out_chunks.append(chunk[cols].copy())

        if idx % 50 == 0:
            print(f"[mimic-gcs] processed {idx:,} chunks... kept {sum(len(c) for c in out_chunks):,} rows")

    if not out_chunks:
        return pd.DataFrame(columns=cols)

    out = pd.concat(out_chunks, ignore_index=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
    return out


def load_mimic_gcs(
    mimic_dir: Path,
    hadm_ids: list[str],
    chunk_size: int,
    max_mimic_chunks: int | None,
    cache_path: Path | None,
    final_target_hour: float | None,
) -> pd.DataFrame:
    cols = (
        ["site", "hospitalization_id"]
        + ALL_GCS_COLS
        + TOTAL_METRIC_COLS
        + ["gcs_total_n_obs", "gcs_total_source", "gcs_total_final_source"]
    )
    hadm_ids = pd.Series(hadm_ids, dtype="string").dropna().astype(str).unique().tolist()
    if not hadm_ids:
        return pd.DataFrame(columns=cols)

    rows = _extract_mimic_gcs_rows(
        mimic_dir=mimic_dir,
        hadm_ids=hadm_ids,
        chunk_size=chunk_size,
        max_mimic_chunks=max_mimic_chunks,
        cache_path=cache_path,
    )
    if rows.empty:
        return _empty_site_gcs_frame(hadm_ids, "mimic")[cols]

    rows = rows.sort_values(["hospitalization_id", "component", "recorded_dttm"])
    first_components = rows.drop_duplicates(subset=["hospitalization_id", "component"], keep="first")
    component_pivot = first_components.pivot(index="hospitalization_id", columns="component", values="gcs_value").reset_index()
    component_pivot.columns.name = None

    wide = (
        rows.groupby(["hospitalization_id", "recorded_dttm", "component"], as_index=False)["gcs_value"]
        .median()
        .pivot_table(
            index=["hospitalization_id", "recorded_dttm"],
            columns="component",
            values="gcs_value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    for comp in COMPONENT_COLS:
        if comp not in wide.columns:
            wide[comp] = np.nan
    for comp, (lo, hi) in COMPONENT_BOUNDS.items():
        wide.loc[~wide[comp].between(lo, hi), comp] = np.nan
    wide["gcs_total"] = wide[COMPONENT_COLS].sum(axis=1, min_count=3)
    total_rows = wide.dropna(subset=["gcs_total"])[["hospitalization_id", "recorded_dttm", "gcs_total"]].copy()

    base = pd.DataFrame({"hospitalization_id": pd.Series(hadm_ids, dtype="string")})
    base = base.merge(component_pivot, on="hospitalization_id", how="left")
    for comp in COMPONENT_COLS:
        if comp not in base.columns:
            base[comp] = np.nan
    total_metrics = _compute_total_metrics(total_rows, hadm_ids, final_target_hour=final_target_hour)
    base = base.merge(total_metrics, on="hospitalization_id", how="left")
    base["gcs_total"] = base["gcs_total_initial"]
    base["gcs_total_source"] = np.where(base["gcs_total_initial"].notna(), "derived_components", "missing")
    base["site"] = "mimic"
    return base[cols]


def _summarize_measure(df: pd.DataFrame, measure: str, group_cols: list[str], label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(group_cols, dropna=False)
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        values = pd.to_numeric(sub[measure], errors="coerce")
        n_patients = int(sub["patient_key"].nunique())
        n_obs = int(values.notna().sum())
        nonnull = values.dropna()
        row = {
            "scope": label,
            "measure": measure,
            "n_patients": n_patients,
            "n_with_measure": n_obs,
            "pct_with_measure": (100.0 * n_obs / n_patients) if n_patients > 0 else np.nan,
            "mean": float(nonnull.mean()) if n_obs > 0 else np.nan,
            "std": float(nonnull.std(ddof=1)) if n_obs > 1 else np.nan,
            "median": float(nonnull.median()) if n_obs > 0 else np.nan,
            "q1": float(nonnull.quantile(0.25)) if n_obs > 0 else np.nan,
            "q3": float(nonnull.quantile(0.75)) if n_obs > 0 else np.nan,
            "min": float(nonnull.min()) if n_obs > 0 else np.nan,
            "max": float(nonnull.max()) if n_obs > 0 else np.nan,
        }
        for k, v in zip(group_cols, key):
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage_table(df: pd.DataFrame, group_cols: list[str], scope: str) -> pd.DataFrame:
    recs: list[dict[str, object]] = []
    grouped = df.groupby(group_cols, dropna=False)
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        n = int(sub["patient_key"].nunique())
        has_total = int(sub["gcs_total_initial"].notna().sum())
        has_all_components = int(sub[COMPONENT_COLS].notna().all(axis=1).sum())
        has_any_component = int(sub[COMPONENT_COLS].notna().any(axis=1).sum())
        n_direct = int(sub["gcs_total_source"].eq("direct_total").sum())
        n_derived = int(sub["gcs_total_source"].eq("derived_components").sum())

        rec = {
            "scope": scope,
            "n_patients": n,
            "n_with_gcs_total": has_total,
            "pct_with_gcs_total": (100.0 * has_total / n) if n > 0 else np.nan,
            "n_with_all_components": has_all_components,
            "pct_with_all_components": (100.0 * has_all_components / n) if n > 0 else np.nan,
            "n_with_any_component": has_any_component,
            "pct_with_any_component": (100.0 * has_any_component / n) if n > 0 else np.nan,
            "n_direct_total": n_direct,
            "n_derived_total": n_derived,
        }
        for k, v in zip(group_cols, key):
            rec[k] = v
        recs.append(rec)
    return pd.DataFrame(recs)


def _plot_gcs_total_pooled(df: pd.DataFrame, out_png: Path, title_prefix: str) -> None:
    sub = df.dropna(subset=["gcs_total_initial"]).copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.text(0.5, 0.5, "No non-missing initial gcs_total values.", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(f"{title_prefix}: Initial GCS Total by Cluster (Pooled)")
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    clusters = sorted(sub["cluster"].astype(int).unique().tolist())
    data = [sub.loc[sub["cluster"].astype(int) == c, "gcs_total_initial"].to_numpy() for c in clusters]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bp = ax.boxplot(data, patch_artist=True, tick_labels=[f"C{c}" for c in clusters], showmeans=True)
    for box in bp["boxes"]:
        box.set(facecolor="#8FBBD9", alpha=0.85)
    for med in bp["medians"]:
        med.set(color="#1F2937", linewidth=2)

    for i, c in enumerate(clusters, start=1):
        n_c = int(sub.loc[sub["cluster"].astype(int) == c, "patient_key"].nunique())
        ax.text(i, ax.get_ylim()[0], f"n={n_c}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Initial GCS Total")
    ax.set_xlabel("Cluster")
    ax.set_title("Pooled")
    ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"{title_prefix}: Initial GCS Total by Cluster")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_gcs_total_by_site(df: pd.DataFrame, out_png: Path, title_prefix: str) -> None:
    sites = sorted(df["site"].astype(str).unique().tolist())
    n_sites = max(1, len(sites))
    fig, axes = plt.subplots(1, n_sites, figsize=(6.5 * n_sites, 5), constrained_layout=True, sharey=True)
    if n_sites == 1:
        axes = [axes]

    for ax, site in zip(axes, sites):
        sub = df[(df["site"].astype(str) == site) & (df["gcs_total_initial"].notna())].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No non-missing initial gcs_total", ha="center", va="center")
            ax.axis("off")
            ax.set_title(site)
            continue

        clusters = sorted(sub["cluster"].astype(int).unique().tolist())
        data = [sub.loc[sub["cluster"].astype(int) == c, "gcs_total_initial"].to_numpy() for c in clusters]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=[f"C{c}" for c in clusters], showmeans=True)
        for box in bp["boxes"]:
            box.set(facecolor="#A8D5BA", alpha=0.85)
        for med in bp["medians"]:
            med.set(color="#1F2937", linewidth=2)
        ax.set_title(site)
        ax.set_xlabel("Cluster")
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Initial GCS Total")
    fig.suptitle(f"{title_prefix}: Initial GCS Total by Cluster, Stratified by Site")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _load_cluster_hourly(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["cluster", "hour"] + list(VITAL_LABELS.keys()))
    out = pd.read_csv(path)
    if out.empty:
        return out
    required = {"cluster", "hour"}
    if not required.issubset(set(out.columns)):
        return pd.DataFrame(columns=["cluster", "hour"] + list(VITAL_LABELS.keys()))
    out["cluster"] = pd.to_numeric(out["cluster"], errors="coerce")
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce")
    out = out.dropna(subset=["cluster", "hour"]).copy()
    out["cluster"] = out["cluster"].astype(int)
    out = out.sort_values(["cluster", "hour"]).reset_index(drop=True)
    return out


def _plot_trajectory_with_gcs_summary(
    cluster_hourly: pd.DataFrame,
    key_metrics_wide: pd.DataFrame,
    out_png: Path,
    title_prefix: str,
    vital: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    ax_traj, ax_gcs = axes

    if cluster_hourly.empty or vital not in cluster_hourly.columns:
        ax_traj.text(0.5, 0.5, "Cluster trajectory data not available.", ha="center", va="center")
        ax_traj.axis("off")
    else:
        clusters = sorted(cluster_hourly["cluster"].dropna().astype(int).unique().tolist())
        cmap = plt.get_cmap("tab10")
        for i, c in enumerate(clusters):
            sub = cluster_hourly[cluster_hourly["cluster"].astype(int) == c].sort_values("hour")
            ax_traj.plot(
                sub["hour"],
                sub[vital],
                color=cmap((i % 10) / 10.0),
                linewidth=2.2,
                label=f"Cluster {c}",
            )
        ax_traj.set_xlabel("Hour from time-zero")
        ax_traj.set_ylabel(VITAL_LABELS.get(vital, vital))
        ax_traj.set_title("Cluster Trajectory")
        ax_traj.grid(True, alpha=0.25)
        ax_traj.legend(loc="best", fontsize=9)

    if key_metrics_wide.empty:
        ax_gcs.text(0.5, 0.5, "No pooled GCS key metrics.", ha="center", va="center")
        ax_gcs.axis("off")
    else:
        plot_df = key_metrics_wide.copy()
        plot_df["cluster"] = pd.to_numeric(plot_df["cluster"], errors="coerce")
        plot_df = plot_df.dropna(subset=["cluster"]).sort_values("cluster")
        plot_df["cluster"] = plot_df["cluster"].astype(int)

        x = np.arange(len(plot_df))
        width = 0.18
        colors = {
            "initial": "#4C78A8",
            "final": "#E45756",
            "min": "#72B7B2",
            "max": "#F2CF5B",
        }

        for i, m in enumerate(METRIC_ORDER):
            vals = pd.to_numeric(plot_df.get(f"median_{m}"), errors="coerce").to_numpy(dtype=float)
            ax_gcs.bar(
                x + (i - 1.5) * width,
                vals,
                width=width,
                color=colors[m],
                label=m.capitalize(),
                alpha=0.9,
            )
            for j, v in enumerate(vals):
                if np.isfinite(v):
                    ax_gcs.text(
                        x[j] + (i - 1.5) * width,
                        v + 0.15,
                        f"{v:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        n_vals = pd.to_numeric(plot_df.get("n_with_measure_initial"), errors="coerce").fillna(0).astype(int).to_list()
        ax_gcs.set_xticks(x)
        ax_gcs.set_xticklabels([f"C{c}\n(n={n})" for c, n in zip(plot_df["cluster"].tolist(), n_vals)])
        ax_gcs.set_ylabel("GCS Total (median)")
        ax_gcs.set_title("Pooled Cluster GCS Total: Initial/Final/Min/Max")
        ax_gcs.grid(True, axis="y", alpha=0.25)
        ax_gcs.legend(loc="best", fontsize=9)

    fig.suptitle(f"{title_prefix}: Cluster Trajectory and Pooled GCS Total Key Metrics")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.cluster_assignments.parent / "gcs_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Reading cluster assignments...")
    assign, assign_mode = _read_assignments(args.cluster_assignments)
    print(f"      Assignment mode: {assign_mode}")
    print(f"      Patients: {assign['patient_key'].nunique():,}")

    ucmc_ids = assign.loc[assign["site"].eq("ucmc"), "hospitalization_id"].astype("string").dropna().unique().tolist()
    mimic_ids = assign.loc[assign["site"].eq("mimic"), "hospitalization_id"].astype("string").dropna().unique().tolist()

    print("[2/5] Loading UCMC GCS...")
    ucmc_gcs = load_ucmc_gcs(args.ucmc_dir, ucmc_ids, final_target_hour=args.final_target_hour)
    print(f"      UCMC admissions requested: {len(ucmc_ids):,}")
    print(f"      UCMC rows with initial gcs_total: {int(ucmc_gcs['gcs_total_initial'].notna().sum()):,}")

    print("[3/5] Loading MIMIC GCS...")
    mimic_cache = args.mimic_gcs_cache
    if mimic_cache is None:
        mimic_cache = output_dir / "mimic_gcs_components_cache.parquet"
    mimic_gcs = load_mimic_gcs(
        mimic_dir=args.mimic_dir,
        hadm_ids=mimic_ids,
        chunk_size=args.chunk_size,
        max_mimic_chunks=args.max_mimic_chunks,
        cache_path=mimic_cache,
        final_target_hour=args.final_target_hour,
    )
    print(f"      MIMIC admissions requested: {len(mimic_ids):,}")
    print(f"      MIMIC rows with initial gcs_total: {int(mimic_gcs['gcs_total_initial'].notna().sum()):,}")

    print("[4/5] Linking GCS to clusters and building summaries...")
    gcs_all = pd.concat([ucmc_gcs, mimic_gcs], ignore_index=True)
    gcs_all["hospitalization_id"] = _id_to_str(gcs_all["hospitalization_id"])
    gcs_all["site"] = gcs_all["site"].astype("string").str.lower()

    linked = assign.merge(gcs_all, on=["site", "hospitalization_id"], how="left")
    for c in ALL_GCS_COLS + TOTAL_METRIC_COLS + ["gcs_total_n_obs"]:
        linked[c] = pd.to_numeric(linked[c], errors="coerce")
    linked["gcs_total_source"] = linked["gcs_total_source"].fillna("missing")
    linked["n_gcs_components_present"] = linked[COMPONENT_COLS].notna().sum(axis=1)

    linked.to_csv(output_dir / "patient_gcs_cluster_linkage.csv", index=False)

    coverage_site_cluster = _coverage_table(linked, ["site", "cluster"], scope="site_cluster")
    coverage_pooled_cluster = _coverage_table(linked.assign(site="pooled"), ["site", "cluster"], scope="pooled_cluster")
    coverage_site = _coverage_table(linked, ["site"], scope="site")
    coverage_overall = _coverage_table(linked.assign(site="overall"), ["site"], scope="overall")
    coverage = pd.concat([coverage_site_cluster, coverage_pooled_cluster, coverage_site, coverage_overall], ignore_index=True)
    coverage = coverage.sort_values(["scope", "site", "cluster"], na_position="last")
    coverage.to_csv(output_dir / "gcs_coverage_by_site_cluster.csv", index=False)

    trajectory_frames = []
    for measure in TOTAL_METRIC_COLS:
        trajectory_frames.append(_summarize_measure(linked.assign(site="pooled"), measure, ["site", "cluster"], label="pooled_cluster"))
        trajectory_frames.append(_summarize_measure(linked, measure, ["site", "cluster"], label="site_cluster"))
        trajectory_frames.append(_summarize_measure(linked, measure, ["site"], label="site"))
        trajectory_frames.append(_summarize_measure(linked.assign(site="overall"), measure, ["site"], label="overall"))
    trajectory_summary = pd.concat(trajectory_frames, ignore_index=True)
    trajectory_summary = trajectory_summary.sort_values(["measure", "scope", "site", "cluster"], na_position="last")
    trajectory_summary.to_csv(output_dir / "gcs_total_trajectory_summary_by_cluster.csv", index=False)
    trajectory_summary[trajectory_summary["measure"].eq("gcs_total_initial")].to_csv(
        output_dir / "gcs_total_summary_by_cluster.csv",
        index=False,
    )
    key = trajectory_summary[trajectory_summary["scope"].eq("pooled_cluster")].copy()
    key = key[
        ["cluster", "measure", "n_with_measure", "mean", "median", "q1", "q3", "min", "max"]
    ].copy()
    key["measure_short"] = key["measure"].astype(str).str.replace("gcs_total_", "", regex=False)
    key_wide = (
        key.pivot_table(
            index="cluster",
            columns="measure_short",
            values=["n_with_measure", "mean", "median", "q1", "q3", "min", "max"],
            aggfunc="first",
        )
        .sort_index()
    )
    key_wide.columns = [f"{stat}_{measure}" for stat, measure in key_wide.columns]
    key_wide = key_wide.reset_index()
    key_wide.to_csv(output_dir / "gcs_total_key_metrics_by_cluster.csv", index=False)

    cluster_hourly_csv = args.cluster_hourly_csv
    if cluster_hourly_csv is None:
        cluster_hourly_csv = args.cluster_assignments.parent / "cluster_hourly_profiles.csv"
    cluster_hourly = _load_cluster_hourly(cluster_hourly_csv)
    _plot_trajectory_with_gcs_summary(
        cluster_hourly=cluster_hourly,
        key_metrics_wide=key_wide,
        out_png=output_dir / "figure_cluster_trajectory_and_gcs_total_key_metrics.png",
        title_prefix=args.title_prefix,
        vital=args.trajectory_vital,
    )

    component_frames = []
    for measure in COMPONENT_COLS:
        component_frames.append(_summarize_measure(linked.assign(site="pooled"), measure, ["site", "cluster"], label="pooled_cluster"))
        component_frames.append(_summarize_measure(linked, measure, ["site", "cluster"], label="site_cluster"))
        component_frames.append(_summarize_measure(linked, measure, ["site"], label="site"))
    component_summary = pd.concat(component_frames, ignore_index=True)
    component_summary = component_summary.sort_values(["measure", "scope", "site", "cluster"], na_position="last")
    component_summary.to_csv(output_dir / "gcs_component_summary_by_cluster.csv", index=False)

    print("[5/5] Writing figures and run summary...")
    _plot_gcs_total_pooled(linked, output_dir / "figure_gcs_total_by_cluster_pooled.png", args.title_prefix)
    _plot_gcs_total_by_site(linked, output_dir / "figure_gcs_total_by_cluster_by_site.png", args.title_prefix)

    overall_n = int(linked["patient_key"].nunique())
    overall_with_initial = int(linked["gcs_total_initial"].notna().sum())
    overall_pct_initial = (100.0 * overall_with_initial / overall_n) if overall_n > 0 else np.nan
    overall_with_final = int(linked["gcs_total_final"].notna().sum())
    overall_with_min = int(linked["gcs_total_min"].notna().sum())
    overall_with_max = int(linked["gcs_total_max"].notna().sum())

    summary_lines = [
        f"assignment_mode: {assign_mode}",
        f"cluster_assignments: {args.cluster_assignments}",
        f"final_target_hour: {args.final_target_hour}",
        f"n_patients: {overall_n}",
        f"n_with_gcs_total_initial: {overall_with_initial}",
        f"pct_with_gcs_total_initial: {overall_pct_initial:.2f}",
        f"n_with_gcs_total_final: {overall_with_final}",
        f"n_with_gcs_total_min: {overall_with_min}",
        f"n_with_gcs_total_max: {overall_with_max}",
    ]

    site_cov = coverage[coverage["scope"].eq("site")].copy()
    for _, r in site_cov.iterrows():
        summary_lines.append(
            f"site={r['site']}: n={int(r['n_patients'])}, gcs_total={int(r['n_with_gcs_total'])} ({float(r['pct_with_gcs_total']):.2f}%)"
        )

    low_coverage = site_cov[site_cov["pct_with_gcs_total"] < 70.0]
    if not low_coverage.empty:
        summary_lines.append("warning: one or more site-level gcs_total coverage rates are <70%; interpret cluster associations cautiously.")

    (output_dir / "run_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
