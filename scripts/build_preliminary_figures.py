#!/usr/bin/env python3
"""
Build pooled preliminary-study figures for AHA CART draft.

Inputs:
  - Site-level pooled outputs in:
    <base_dir>/<site>/Upload_to_Box_without_oral_{24,72}/
      - table1_poolable_{24,72}h.csv
      - hourly_vitals_by_trajectory_survival_{24,72}h.csv

Outputs:
  - 4 composite figure PNGs
  - companion CSV summaries
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re


TRAJ_ORDER = ["Persistent High", "Rapid Decline", "Normothermic", "Hypothermic", "No temp data"]
TRAJ_ORDER_MAIN = ["Persistent High", "Rapid Decline", "Normothermic", "Hypothermic"]
TRAJ_COLORS = {
    "Persistent High": "#C62828",
    "Rapid Decline": "#FB8C00",
    "Normothermic": "#43A047",
    "Hypothermic": "#1565C0",
    "No temp data": "#757575",
}
SURV_COLORS = {"Survivor": "#1E88E5", "Non-Survivor": "#E53935"}
TRAJ_DISPLAY = {
    "Persistent High": "Category A",
    "Rapid Decline": "Category B",
    "Normothermic": "Category C",
    "Hypothermic": "Category D",
    "No temp data": "No Temperature Data",
}
VITAL_LABELS = {
    "heart_rate": "Heart Rate (bpm)",
    "map": "Mean Arterial Pressure (mmHg)",
    "spo2": "SpO2 (%)",
    "temp_c": "Temperature (C)",
    "blood_glucose": "Blood Glucose (mg/dL)",
}
VITAL_ORDER = ["heart_rate", "map", "spo2", "temp_c", "blood_glucose"]
HEATMAP_SHORT_LABELS = {
    "map": "MAP",
    "heart_rate": "HR",
    "spo2": "SpO2",
    "temp_c": "Temp",
    "blood_glucose": "Glucose",
}
SURV_ORDER = ["Survivor", "Non-Survivor"]


@dataclass
class LoadedData:
    table1: pd.DataFrame
    hourly: pd.DataFrame


def configure_tufte_style() -> None:
    """
    Global style with a higher data-ink ratio.
    """
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#4f4f4f",
            "axes.linewidth": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
        }
    )


def style_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4f4f4f")
    ax.spines["bottom"].set_color("#4f4f4f")
    ax.tick_params(colors="#303030")
    if grid_axis in {"x", "y"}:
        ax.grid(axis=grid_axis, color="#D0D0D0", linewidth=0.8, alpha=0.35)
    else:
        ax.grid(False)


def traj_label(raw_name: str) -> str:
    return TRAJ_DISPLAY.get(raw_name, raw_name)


def alphabet_labels(n: int) -> list[str]:
    """
    Excel-style sequence: A..Z, AA..AZ, BA...
    """
    labels = []
    for i in range(n):
        x = i
        chars = []
        while True:
            x, rem = divmod(x, 26)
            chars.append(chr(ord("A") + rem))
            if x == 0:
                break
            x -= 1
        labels.append("".join(reversed(chars)))
    return labels


def available_vitals(df: pd.DataFrame) -> list[str]:
    return [v for v in VITAL_ORDER if f"mean_{v}" in df.columns and f"n_{v}" in df.columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pooled OHCA preliminary figures.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/Users/davidbeiser/Library/CloudStorage/Box-Box/CLIF/Projects/AHA-OHCA"),
        help="Directory containing per-site Upload_to_Box_without_oral_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_dir/preliminary_figures"),
        help="Output directory for composite figures and summary CSVs.",
    )
    parser.add_argument(
        "--window-primary",
        type=int,
        default=72,
        choices=[24, 72],
        help="Primary window for Figure 1 and Figure 3.",
    )
    parser.add_argument(
        "--site-date-ranges-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns: site,start_date,end_date for Figure 1 date-range labels.",
    )
    return parser.parse_args()


def _find_input_files(base_dir: Path, pattern: str) -> list[Path]:
    files = sorted(base_dir.glob(pattern))
    return [f for f in files if f.is_file()]


def load_site_date_ranges(
    base_dir: Path,
    site_names: Iterable[str],
    window_hours: int,
    site_date_ranges_csv: Path | None = None,
) -> dict[str, str]:
    """
    Resolve site case-date ranges for labeling.

    Priority:
      1) Explicit metadata file at <base_dir>/site_date_ranges.csv with columns:
         site,start_date,end_date  (YYYY-MM-DD preferred)
      2) Pattern search in each site's pipeline_log.txt for lines like:
         "Case date range: <start> to <end>" or "Date range: <start> to <end>"
      3) Fallback: "Date range: not provided"
    """
    out = {s: "dates n/a" for s in site_names}

    meta_path = site_date_ranges_csv if site_date_ranges_csv is not None else (base_dir / "site_date_ranges.csv")
    if meta_path.exists():
        try:
            md = pd.read_csv(meta_path)
            cols = {c.lower(): c for c in md.columns}
            if all(k in cols for k in ["site", "start_date", "end_date"]):
                for _, r in md.iterrows():
                    s = str(r[cols["site"]]).strip()
                    if s in out:
                        start = str(r[cols["start_date"]]).strip()
                        end = str(r[cols["end_date"]]).strip()
                        if start and end and start.lower() != "nan" and end.lower() != "nan":
                            out[s] = f"{start} to {end}"
        except Exception:
            pass

    # Best-effort parse from pipeline logs only for unresolved sites.
    patt = re.compile(r"(?:Case date range|Date range)\s*:\s*(.+?)\s+to\s+(.+)", flags=re.IGNORECASE)
    for s in site_names:
        if out[s] != "dates n/a":
            continue
        log_path = base_dir / s / f"Upload_to_Box_without_oral_{window_hours}" / "pipeline_log.txt"
        if not log_path.exists():
            continue
        try:
            for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                m = patt.search(line)
                if m:
                    out[s] = f"{m.group(1).strip()} to {m.group(2).strip()}"
                    break
        except Exception:
            continue
    return out


def load_data(base_dir: Path) -> LoadedData:
    table_paths = _find_input_files(base_dir, "*/Upload_to_Box_without_oral_*/table1_poolable_*h.csv")
    hourly_paths = _find_input_files(base_dir, "*/Upload_to_Box_without_oral_*/hourly_vitals_by_trajectory_survival_*h.csv")
    if not table_paths:
        raise FileNotFoundError(f"No table1_poolable files found in {base_dir}")
    if not hourly_paths:
        raise FileNotFoundError(f"No hourly_vitals files found in {base_dir}")

    table_frames = []
    for p in table_paths:
        df = pd.read_csv(p)
        df["source_path"] = str(p)
        table_frames.append(df)
    table1 = pd.concat(table_frames, ignore_index=True)

    hourly_frames = []
    for p in hourly_paths:
        df = pd.read_csv(p)
        df["source_path"] = str(p)
        hourly_frames.append(df)
    hourly = pd.concat(hourly_frames, ignore_index=True)

    table1["site"] = table1["site"].astype(str).str.strip()
    hourly["site"] = hourly["site"].astype(str).str.strip()
    return LoadedData(table1=table1, hourly=hourly)


def clean_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Soft-clean known artifacts in pooled hourly summaries:
      - impossible temperatures from occasional bad source rows
      - implausible MAP values (primarily hour-0 artifact in one site)
    """
    out = hourly.copy()

    def _mask_outliers(vital: str, lo: float, hi: float) -> None:
        mean_col = f"mean_{vital}"
        sd_col = f"sd_{vital}"
        se_col = f"se_{vital}"
        n_col = f"n_{vital}"
        required = [mean_col, sd_col, se_col, n_col]
        if not all(c in out.columns for c in required):
            return
        bad = out[mean_col].notna() & ((out[mean_col] < lo) | (out[mean_col] > hi))
        out.loc[bad, [mean_col, sd_col, se_col]] = np.nan
        out.loc[bad, n_col] = 0

    _mask_outliers("temp_c", lo=32.0, hi=42.0)
    _mask_outliers("map", lo=30.0, hi=180.0)
    _mask_outliers("spo2", lo=50.0, hi=100.0)
    _mask_outliers("blood_glucose", lo=20.0, hi=1000.0)
    return out


def pooled_mean_se(rows: pd.DataFrame, vital: str) -> pd.Series:
    mean_col = f"mean_{vital}"
    sd_col = f"sd_{vital}"
    n_col = f"n_{vital}"
    sub = rows[[mean_col, sd_col, n_col]].dropna(subset=[mean_col]).copy()
    if sub.empty:
        return pd.Series({"mean": np.nan, "se": np.nan, "n": 0})
    n = sub[n_col].fillna(0).astype(float)
    m = sub[mean_col].astype(float)
    sd = sub[sd_col].fillna(0).astype(float)
    ok = n > 0
    if not ok.any():
        return pd.Series({"mean": np.nan, "se": np.nan, "n": 0})
    n = n[ok]
    m = m[ok]
    sd = sd[ok]
    n_total = float(n.sum())
    mean = float((m * n).sum() / n_total)
    if n_total <= 1:
        return pd.Series({"mean": mean, "se": np.nan, "n": int(n_total)})
    ss_within = float(((n - 1) * (sd**2)).sum())
    ss_between = float((n * (m - mean) ** 2).sum())
    var = (ss_within + ss_between) / max(n_total - 1, 1)
    se = float(np.sqrt(var / n_total))
    return pd.Series({"mean": mean, "se": se, "n": int(n_total)})


def _pivot_table1(df: pd.DataFrame, index_cols: Iterable[str]) -> pd.DataFrame:
    p = df.pivot_table(index=list(index_cols), columns="variable", values="value", aggfunc="first").reset_index()
    p.columns.name = None
    return p


def pooled_survival_counts(table1: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    sub = table1[
        (table1["window_hours"] == window_hours)
        & (table1["group_type"] == "survival")
        & (table1["group"].isin(SURV_ORDER))
        & (table1["variable"] == "n")
    ].copy()
    out = (
        sub.groupby("group", as_index=False)["value"]
        .sum()
        .rename(columns={"group": "survival_status", "value": "baseline_n"})
    )
    out["baseline_n"] = out["baseline_n"].astype(float)
    out = out.sort_values("survival_status")
    return out


def figure1_feasibility(
    table1: pd.DataFrame,
    output_dir: Path,
    window_primary: int,
    base_dir: Path,
    site_date_ranges_csv: Path | None = None,
) -> None:
    sub = table1[table1["window_hours"] == window_primary].copy()

    surv = sub[(sub["group_type"] == "survival") & (sub["group"] == "Overall")]
    surv_p = _pivot_table1(surv[surv["variable"].isin(["n", "mortality_n", "mortality_pct"])], ["site"])
    surv_p = surv_p.sort_values("n", ascending=False)
    surv_p["mortality_pct"] = surv_p["mortality_pct"].astype(float)
    surv_p["site"] = surv_p["site"].astype(str)

    traj = sub[(sub["group_type"] == "trajectory") & (sub["group"].isin(TRAJ_ORDER))]
    traj_p = _pivot_table1(traj[traj["variable"].isin(["n", "mortality_n", "mortality_pct"])], ["site", "group"])
    traj_p["n"] = traj_p["n"].astype(float)
    traj_p["mortality_pct"] = traj_p["mortality_pct"].astype(float)

    total_n = float(surv_p["n"].sum())
    date_ranges = load_site_date_ranges(
        base_dir,
        surv_p["site"].tolist(),
        window_primary,
        site_date_ranges_csv=site_date_ranges_csv,
    )
    surv_p["case_date_range"] = surv_p["site"].map(date_ranges).fillna("dates n/a")
    no_temp_n = float(
        traj_p.loc[traj_p["group"] == "No temp data", "n"].sum()
        if (traj_p["group"] == "No temp data").any()
        else 0.0
    )
    has_temp_n = total_n - no_temp_n
    assigned_n = has_temp_n

    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.2], wspace=0.3)

    # Panel A: pooled attrition/coverage
    ax1 = fig.add_subplot(gs[0, 0])
    flow_labels = ["OHCA ICU cohort", "Has temp (32-44C)", "Trajectory assigned"]
    flow_vals = [total_n, has_temp_n, assigned_n]
    ax1.barh(flow_labels, flow_vals, color=["#1565C0", "#1E88E5", "#64B5F6"])
    for i, v in enumerate(flow_vals):
        ax1.text(v + max(flow_vals) * 0.01, i, f"n={int(v):,}", va="center", fontsize=10)
    if total_n > 0:
        ax1.text(
            has_temp_n * 0.45,
            1,
            f"lost {int(no_temp_n):,} ({(no_temp_n / total_n * 100):.1f}%)",
            color="#C62828",
            fontsize=9,
            va="center",
            ha="center",
        )
    ax1.set_title("A. Pooled Cohort Coverage")
    ax1.set_xlabel("Encounters")
    ax1.set_xlim(0, max(flow_vals) * 1.25)
    style_axis(ax1, grid_axis=None)

    # Panel B: site N and mortality
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(surv_p))
    ax2.bar(x, surv_p["n"], color="#90A4AE", alpha=0.9)
    ax2.set_xticks(x)
    tick_labels = [f"{s}\n{dr}" for s, dr in zip(surv_p["site"], surv_p["case_date_range"])]
    ax2.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax2.set_ylabel("OHCA ICU N")
    ax2.set_title("B. Site Sample Size and Mortality")
    ax2.text(0.02, 0.97, f"Total pooled N = {int(total_n):,}", transform=ax2.transAxes, va="top", ha="left", fontsize=10)
    for xi, n in zip(x, surv_p["n"]):
        ax2.text(xi, n + surv_p["n"].max() * 0.015, f"{int(n):,}", ha="center", va="bottom", fontsize=9)
    style_axis(ax2, grid_axis="y")

    ax2b = ax2.twinx()
    ax2b.plot(x, surv_p["mortality_pct"], color="#D32F2F", marker="o", linewidth=2)
    ax2b.set_ylabel("Mortality (%)", color="#D32F2F")
    ax2b.tick_params(axis="y", colors="#D32F2F")
    ax2b.set_ylim(0, max(80, surv_p["mortality_pct"].max() + 10))
    for xi, p in zip(x, surv_p["mortality_pct"]):
        ax2b.text(xi, p + 1.2, f"{p:.1f}%", color="#D32F2F", ha="center", fontsize=9)
    ax2b.spines["top"].set_visible(False)

    # Panel C: trajectory composition by site (stacked %)
    ax3 = fig.add_subplot(gs[0, 2])
    comp = (
        traj_p.pivot(index="site", columns="group", values="n")
        .reindex(columns=TRAJ_ORDER, fill_value=0)
        .fillna(0)
    )
    comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
    sites = comp_pct.index.tolist()
    xpos = np.arange(len(sites))
    bottom = np.zeros(len(sites))
    for g in TRAJ_ORDER:
        vals = comp_pct[g].values
        ax3.bar(xpos, vals, bottom=bottom, color=TRAJ_COLORS[g], label=traj_label(g))
        bottom += vals
    ax3.set_xticks(xpos)
    ax3.set_xticklabels(sites, rotation=20)
    ax3.set_ylabel("Trajectory Composition (%)")
    ax3.set_ylim(0, 100)
    ax3.set_title("C. Site Trajectory Composition")
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    style_axis(ax3, grid_axis="y")

    fig.suptitle(f"Figure 1. Multi-Site Feasibility and Heterogeneity ({window_primary}h)", fontsize=14.5, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"figure1_feasibility_heterogeneity_{window_primary}h.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    surv_p.to_csv(output_dir / f"figure1_site_summary_{window_primary}h.csv", index=False)
    comp_pct.reset_index().to_csv(output_dir / f"figure1_trajectory_composition_pct_{window_primary}h.csv", index=False)


def figure1b_single_panel_anonymized(
    table1: pd.DataFrame,
    output_dir: Path,
    window_primary: int,
    base_dir: Path,
    site_date_ranges_csv: Path | None = None,
) -> None:
    """
    Single-panel version of Figure 1B with anonymized site labels (A, B, C...).
    MIMIC is kept explicitly labeled because it is a public dataset.
    """
    sub = table1[table1["window_hours"] == window_primary].copy()
    surv = sub[(sub["group_type"] == "survival") & (sub["group"] == "Overall")]
    surv_p = _pivot_table1(surv[surv["variable"].isin(["n", "mortality_n", "mortality_pct"])], ["site"])
    surv_p = surv_p.sort_values("n", ascending=False).reset_index(drop=True)
    surv_p["site"] = surv_p["site"].astype(str)
    surv_p["n"] = surv_p["n"].astype(float)
    surv_p["mortality_n"] = surv_p["mortality_n"].astype(float)
    surv_p["mortality_pct"] = surv_p["mortality_pct"].astype(float)

    date_ranges = load_site_date_ranges(
        base_dir,
        surv_p["site"].tolist(),
        window_primary,
        site_date_ranges_csv=site_date_ranges_csv,
    )
    surv_p["case_date_range"] = surv_p["site"].map(date_ranges).fillna("dates n/a")
    surv_p["site_label"] = alphabet_labels(len(surv_p))
    is_mimic = surv_p["site"].str.contains("mimic", case=False, na=False)
    surv_p.loc[is_mimic, "site_label"] = "MIMIC"
    total_n = int(round(float(surv_p["n"].sum())))

    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    x = np.arange(len(surv_p))
    bars = ax.bar(x, surv_p["n"], color="#90A4AE", alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(surv_p["site_label"])
    xlabel = "Site (anonymized; MIMIC shown)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("OHCA ICU N")
    ax.set_title(f"Figure 1B. Site Sample Size and Mortality ({window_primary}h)")
    ax.text(0.02, 0.97, f"Total pooled N = {total_n:,}", transform=ax.transAxes, va="top", ha="left", fontsize=10)
    for rect, n in zip(bars, surv_p["n"]):
        ax.text(rect.get_x() + rect.get_width() / 2, n + surv_p["n"].max() * 0.015, f"{int(n):,}", ha="center", va="bottom", fontsize=9)
    style_axis(ax, grid_axis="y")

    ax2 = ax.twinx()
    ax2.plot(x, surv_p["mortality_pct"], color="#D32F2F", marker="o", linewidth=2)
    ax2.set_ylabel("Mortality (%)", color="#D32F2F")
    ax2.tick_params(axis="y", colors="#D32F2F")
    ax2.set_ylim(0, max(80, float(surv_p["mortality_pct"].max()) + 10))
    for xi, p in zip(x, surv_p["mortality_pct"]):
        ax2.text(xi, float(p) + 1.2, f"{float(p):.1f}%", color="#D32F2F", ha="center", fontsize=9)
    ax2.spines["top"].set_visible(False)

    # If date ranges are available, include compact legend-like text block.
    has_dates = (surv_p["case_date_range"] != "dates n/a").any()
    if has_dates:
        lines = [f"{row.site_label}: {row.case_date_range}" for row in surv_p.itertuples()]
        ax.text(
            1.02,
            0.98,
            "Case date ranges\n" + "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.8,
            color="#303030",
        )

    fig.tight_layout()
    fig.savefig(output_dir / f"figure1b_site_size_mortality_anonymized_{window_primary}h.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    anonymized = surv_p[["site", "site_label", "n", "mortality_n", "mortality_pct", "case_date_range"]].copy()
    anonymized.to_csv(output_dir / f"figure1b_site_summary_anonymized_{window_primary}h.csv", index=False)


def figure2_survival_vitals(hourly: pd.DataFrame, output_dir: Path, window_hours: int = 24) -> None:
    sub = hourly[hourly["window_hours"] == window_hours].copy()
    sub = sub[sub["hour"].between(0, window_hours)]
    vitals = available_vitals(sub)
    if not vitals:
        raise ValueError("No vital summary columns found in hourly pooled data.")

    pooled_rows = []
    for (hour, status), grp in sub.groupby(["hour", "survival_status"], dropna=False):
        row = {"hour": float(hour), "survival_status": status}
        for vital in vitals:
            s = pooled_mean_se(grp, vital)
            row[f"mean_{vital}"] = s["mean"]
            row[f"se_{vital}"] = s["se"]
            row[f"n_{vital}"] = s["n"]
        pooled_rows.append(row)
    pooled = pd.DataFrame(pooled_rows).sort_values(["survival_status", "hour"])

    ncols = 2 if len(vitals) <= 4 else 3
    nrows = int(np.ceil(len(vitals) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.6 * ncols, 4.7 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, vital in zip(axes, vitals):
        for status in ["Survivor", "Non-Survivor"]:
            d = pooled[pooled["survival_status"] == status].sort_values("hour")
            x = d["hour"].to_numpy()
            y = d[f"mean_{vital}"].to_numpy()
            se = d[f"se_{vital}"].to_numpy()
            ci = 1.96 * se
            if vital == "spo2":
                y = np.clip(y, 0.0, 100.0)
                low = np.clip(y - ci, 0.0, 100.0)
                high = np.clip(y + ci, 0.0, 100.0)
            else:
                low = y - ci
                high = y + ci
            ax.plot(x, y, color=SURV_COLORS[status], linewidth=2.2, label=status)
            ax.fill_between(x, low, high, color=SURV_COLORS[status], alpha=0.16)

        ax.set_title(VITAL_LABELS[vital])
        ax.set_xlim(0, window_hours)
        if vital == "spo2":
            ax.set_ylim(top=100)
        ax.set_xlabel("Hours from first vital")
        ax.set_ylabel(VITAL_LABELS[vital])
        style_axis(ax, grid_axis="y")

        end = pooled[pooled["hour"] == float(window_hours)]
        if not end.empty:
            s = end[end["survival_status"] == "Survivor"][f"mean_{vital}"]
            n = end[end["survival_status"] == "Non-Survivor"][f"mean_{vital}"]
            if not s.empty and not n.empty:
                diff = float(s.iloc[0] - n.iloc[0])
                ax.text(0.98, 0.06, f"Delta@{window_hours}h: {diff:+.2f}", transform=ax.transAxes, ha="right", fontsize=9)
    for ax in axes[len(vitals):]:
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color=SURV_COLORS["Survivor"], lw=2.2, label="Survivor"),
        Line2D([0], [0], color=SURV_COLORS["Non-Survivor"], lw=2.2, label="Non-Survivor"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.suptitle(f"Figure 2. Pooled Physiologic Separation by Survival ({window_hours}h)", fontsize=14.5, y=1.045)
    fig.tight_layout()
    fig.savefig(output_dir / f"figure2_survival_vitals_{window_hours}h.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    pooled.to_csv(output_dir / f"figure2_pooled_survival_curves_{window_hours}h.csv", index=False)

    # Site-level sensitivity at final hour
    site_rows = []
    for site, grp in sub[sub["hour"] == float(window_hours)].groupby("site"):
        row = {"site": site}
        for vital in vitals:
            d_surv = grp[grp["survival_status"] == "Survivor"]
            d_non = grp[grp["survival_status"] == "Non-Survivor"]
            m_surv = pooled_mean_se(d_surv, vital)["mean"]
            m_non = pooled_mean_se(d_non, vital)["mean"]
            row[f"delta_{vital}_surv_minus_non"] = m_surv - m_non
        site_rows.append(row)
    pd.DataFrame(site_rows).to_csv(output_dir / f"figure2_site_deltas_{window_hours}h.csv", index=False)


def figure4_temperature_measurement_density(
    table1: pd.DataFrame,
    hourly: pd.DataFrame,
    output_dir: Path,
    window_hours: int = 72,
) -> None:
    """
    Uses pooled hourly outputs to quantify temperature observation density.
    Note: n_temp_c represents encounter-hour observations (max 1 per patient-hour after dedup),
    not raw bedside measurement counts.
    """
    sub = hourly[(hourly["window_hours"] == window_hours) & (hourly["hour"].between(0, window_hours))].copy()
    surv_n = pooled_survival_counts(table1, window_hours)
    n_map = dict(zip(surv_n["survival_status"], surv_n["baseline_n"]))

    by_hour = (
        sub.groupby(["hour", "survival_status"], as_index=False)["n_temp_c"]
        .sum()
        .rename(columns={"n_temp_c": "encounters_with_temp"})
    )
    by_hour = by_hour[by_hour["survival_status"].isin(SURV_ORDER)].copy()
    by_hour["baseline_n"] = by_hour["survival_status"].map(n_map)
    by_hour["pct_with_temp"] = np.where(
        by_hour["baseline_n"] > 0,
        by_hour["encounters_with_temp"] / by_hour["baseline_n"] * 100,
        np.nan,
    )

    horizons = [12, 24, 72]
    summary_rows = []
    for status in SURV_ORDER:
        s = by_hour[by_hour["survival_status"] == status]
        base_n = float(n_map.get(status, np.nan))
        for h in horizons:
            sh = s[s["hour"] <= h]
            tot_obs = float(sh["encounters_with_temp"].sum())
            hours_count = float(h + 1)  # includes hour 0
            cumulative_obs_per_patient = tot_obs / base_n if base_n > 0 else np.nan
            obs_per_patient_per_hour = tot_obs / (base_n * hours_count) if base_n > 0 and hours_count > 0 else np.nan
            summary_rows.append(
                {
                    "survival_status": status,
                    "horizon_hours": h,
                    "total_hourly_temp_observations": tot_obs,
                    "baseline_n": base_n,
                    "hours_in_window": hours_count,
                    "cumulative_temp_observations_per_patient": cumulative_obs_per_patient,
                    "temp_observations_per_patient_per_hour": obs_per_patient_per_hour,
                    "pct_patient_hours_with_temp": obs_per_patient_per_hour * 100 if np.isfinite(obs_per_patient_per_hour) else np.nan,
                }
            )
    summary = pd.DataFrame(summary_rows)

    # Plot: A) per-hour observation intensity, B) hourly coverage drop-off
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.8), gridspec_kw={"width_ratios": [1.0, 1.5]})

    # A
    x = np.arange(len(horizons))
    width = 0.36
    for i, status in enumerate(SURV_ORDER):
        d = summary[summary["survival_status"] == status].sort_values("horizon_hours")
        xpos = x + (i - 0.5) * width
        yvals = d["temp_observations_per_patient_per_hour"]
        ax1.bar(xpos, yvals, width=width, color=SURV_COLORS[status], alpha=0.9, label=status)
        for xi, val in zip(xpos, yvals):
            ax1.text(xi, val + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"0-{h}h" for h in horizons])
    ax1.set_ylabel("Temp observations per patient per hour")
    ax1.set_ylim(0, max(0.7, float(summary["temp_observations_per_patient_per_hour"].max()) * 1.25))
    ax1.set_title("A. Temperature Observation Intensity")
    style_axis(ax1, grid_axis="y")
    ax1.legend(frameon=False)

    # B
    for status in SURV_ORDER:
        d = by_hour[by_hour["survival_status"] == status].sort_values("hour")
        ax2.plot(d["hour"], d["pct_with_temp"], color=SURV_COLORS[status], linewidth=2.3, label=status)
        valid = d["pct_with_temp"].notna()
        if valid.any():
            x_last = float(d.loc[valid, "hour"].iloc[-1])
            y_last = float(d.loc[valid, "pct_with_temp"].iloc[-1])
            ax2.text(max(0.0, x_last - 9.0), y_last, status, color=SURV_COLORS[status], fontsize=9, va="center")
    ax2.set_xlim(0, window_hours)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Hours from first vital")
    ax2.set_ylabel("% of baseline group with temp observation")
    ax2.set_title("B. Hourly Temp Coverage Drop-off (Pooled)")
    style_axis(ax2, grid_axis="y")
    for h in horizons:
        ax2.axvline(h, color="#9E9E9E", linestyle="--", alpha=0.35, linewidth=1)
    ax2.text(
        0.02,
        0.02,
        "Note: pooled outputs lack time-of-death; drop-off reflects combined death/discharge/missingness.",
        transform=ax2.transAxes,
        fontsize=8.5,
        color="#424242",
        ha="left",
        va="bottom",
    )

    fig.suptitle(f"Figure 4. Temperature Measurement Coverage by Survival ({window_hours}h)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"figure4_temperature_measurement_coverage_{window_hours}h.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    by_hour.to_csv(output_dir / f"figure4_temp_hourly_coverage_{window_hours}h.csv", index=False)
    summary.to_csv(output_dir / f"figure4_temp_observation_summary_{window_hours}h.csv", index=False)


def figure3_trajectory_phenotypes(
    table1: pd.DataFrame,
    hourly: pd.DataFrame,
    output_dir: Path,
    window_primary: int = 72,
) -> None:
    tsub = table1[table1["window_hours"] == window_primary].copy()
    hsub = hourly[hourly["window_hours"] == window_primary].copy()
    hsub = hsub[hsub["trajectory"].isin(TRAJ_ORDER_MAIN)]

    # Panel A: pooled trajectory temperature shapes
    traj_temp_rows = []
    for (hour, traj), grp in hsub.groupby(["hour", "trajectory"]):
        s = pooled_mean_se(grp, "temp_c")
        traj_temp_rows.append(
            {
                "hour": float(hour),
                "trajectory": traj,
                "mean_temp_c": s["mean"],
                "se_temp_c": s["se"],
                "n_temp_c": s["n"],
            }
        )
    traj_temp = pd.DataFrame(traj_temp_rows).sort_values(["trajectory", "hour"])

    # Panel B: trajectory mortality distribution across sites
    traj = tsub[(tsub["group_type"] == "trajectory") & (tsub["group"].isin(TRAJ_ORDER_MAIN))]
    traj_p = _pivot_table1(traj[traj["variable"].isin(["n", "mortality_n", "mortality_pct"])], ["site", "group"])
    traj_p["n"] = traj_p["n"].astype(float)
    traj_p["mortality_n"] = traj_p["mortality_n"].astype(float)
    traj_p["mortality_pct"] = traj_p["mortality_pct"].astype(float)
    pooled = (
        traj_p.groupby("group", as_index=False)
        .agg(n=("n", "sum"), deaths=("mortality_n", "sum"))
        .assign(mortality_pct=lambda d: d["deaths"] / d["n"] * 100)
    )
    pooled["group"] = pd.Categorical(pooled["group"], categories=TRAJ_ORDER_MAIN, ordered=True)
    pooled = pooled.sort_values("group")
    n_lookup = dict(zip(pooled["group"].astype(str), pooled["n"]))

    box_rows = []
    for g in TRAJ_ORDER_MAIN:
        vals = traj_p.loc[traj_p["group"] == g, "mortality_pct"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        q1, median, q3 = np.percentile(vals, [25, 50, 75])
        box_rows.append(
            {
                "trajectory": g,
                "min": float(np.min(vals)),
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "max": float(np.max(vals)),
                "site_count": int(len(vals)),
                "pooled_n": int(n_lookup.get(g, np.nan)),
            }
        )
    box_summary = pd.DataFrame(box_rows)

    # Panel C: trajectory-matched survivor vs non-survivor delta at 24h
    delta_rows = []
    h24 = hsub[hsub["hour"] == 24.0].copy()
    delta_vitals = available_vitals(h24)
    for traj_name, grp in h24.groupby("trajectory"):
        row = {"trajectory": traj_name}
        for vital in delta_vitals:
            surv = pooled_mean_se(grp[grp["survival_status"] == "Survivor"], vital)["mean"]
            non = pooled_mean_se(grp[grp["survival_status"] == "Non-Survivor"], vital)["mean"]
            row[f"delta_{vital}"] = surv - non
        delta_rows.append(row)
    delta = pd.DataFrame(delta_rows)
    delta["trajectory"] = pd.Categorical(delta["trajectory"], categories=TRAJ_ORDER_MAIN, ordered=True)
    delta = delta.sort_values("trajectory")

    fig = plt.figure(figsize=(18, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.1], wspace=0.32)

    # A
    ax1 = fig.add_subplot(gs[0, 0])
    for traj_name in TRAJ_ORDER_MAIN:
        d = traj_temp[traj_temp["trajectory"] == traj_name].sort_values("hour")
        x = d["hour"].to_numpy()
        y = d["mean_temp_c"].to_numpy()
        se = d["se_temp_c"].to_numpy()
        ax1.plot(x, y, color=TRAJ_COLORS[traj_name], linewidth=2.2, label=traj_label(traj_name))
        ax1.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=TRAJ_COLORS[traj_name], alpha=0.15)
    ax1.set_title("A. Pooled Temperature Trajectory Shapes")
    ax1.set_xlabel("Hours from first vital")
    ax1.set_ylabel("Temperature (C)")
    ax1.set_xlim(0, window_primary)
    style_axis(ax1, grid_axis="y")
    ax1.legend(frameon=False, fontsize=9, loc="lower right")

    # B
    ax2 = fig.add_subplot(gs[0, 1])
    box_data = [
        traj_p.loc[traj_p["group"] == g, "mortality_pct"].dropna().to_numpy(dtype=float)
        for g in TRAJ_ORDER_MAIN
    ]
    positions = np.arange(1, len(TRAJ_ORDER_MAIN) + 1)
    bp = ax2.boxplot(
        box_data,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        whis=1.5,
        showfliers=True,
        medianprops={"color": "#111111", "linewidth": 2.0},
        whiskerprops={"color": "#212121", "linewidth": 1.5},
        capprops={"color": "#212121", "linewidth": 1.5},
        boxprops={"color": "#212121", "linewidth": 1.1},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#212121",
            "markersize": 4,
            "alpha": 0.9,
        },
    )
    for patch, g in zip(bp["boxes"], TRAJ_ORDER_MAIN):
        patch.set_facecolor(TRAJ_COLORS[g])
        patch.set_alpha(0.82)

    xtick_labels = [f"{traj_label(g)}\n(n={int(n_lookup.get(g, 0))})" for g in TRAJ_ORDER_MAIN]
    ax2.set_xticks(positions)
    ax2.set_xticklabels(xtick_labels, rotation=18)
    ax2.set_ylabel("Mortality (%)")
    ax2.set_title("B. Site Mortality Distribution by Trajectory (Box Plot)")
    ax2.set_ylim(0, 100)
    style_axis(ax2, grid_axis="y")

    # C
    ax3 = fig.add_subplot(gs[0, 2])
    heat_cols = [f"delta_{v}" for v in delta_vitals if f"delta_{v}" in delta.columns]
    if not heat_cols:
        delta["delta_placeholder"] = np.nan
        heat_cols = ["delta_placeholder"]
    heat_disp = delta[heat_cols].to_numpy(dtype=float)
    if np.isnan(heat_disp).all():
        vmax = 1.0
    else:
        vmax = max(float(np.nanmax(np.abs(heat_disp))), 1.0)
    im = ax3.imshow(heat_disp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax3.set_xticks(range(len(heat_cols)))
    ax3.set_xticklabels([HEATMAP_SHORT_LABELS.get(v.replace("delta_", ""), "No Data") for v in heat_cols])
    ax3.set_yticks(range(len(delta)))
    ax3.set_yticklabels([traj_label(v) for v in delta["trajectory"].astype(str).tolist()])
    ax3.set_title("C. Survivor minus Non-Survivor Delta at 24h\n(within trajectory)")
    for i in range(heat_disp.shape[0]):
        for j in range(heat_disp.shape[1]):
            v = heat_disp[i, j]
            if np.isnan(v):
                txt = "NA"
            else:
                txt = f"{v:+.2f}"
            ax3.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Difference")
    style_axis(ax3, grid_axis=None)

    fig.suptitle(f"Figure 3. Temperature Phenotypes and Outcome Relevance ({window_primary}h)", fontsize=14.5, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"figure3_trajectory_phenotypes_{window_primary}h.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    traj_temp_out = traj_temp.copy()
    traj_temp_out["trajectory_label"] = traj_temp_out["trajectory"].map(traj_label)
    traj_p_out = traj_p.copy()
    traj_p_out["group_label"] = traj_p_out["group"].map(traj_label)
    box_summary_out = box_summary.copy()
    box_summary_out["trajectory_label"] = box_summary_out["trajectory"].map(traj_label)
    delta_out = delta.copy()
    delta_out["trajectory_label"] = delta_out["trajectory"].map(traj_label)

    traj_temp_out.to_csv(output_dir / f"figure3_trajectory_temp_shapes_{window_primary}h.csv", index=False)
    traj_p_out.to_csv(output_dir / f"figure3_trajectory_mortality_by_site_{window_primary}h.csv", index=False)
    box_summary_out.to_csv(output_dir / f"figure3_trajectory_mortality_box_summary_{window_primary}h.csv", index=False)
    delta_out.to_csv(output_dir / "figure3_trajectory_matched_deltas_24h.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_tufte_style()

    loaded = load_data(args.base_dir)
    hourly_clean = clean_hourly(loaded.hourly)

    figure1_feasibility(
        loaded.table1,
        args.output_dir,
        args.window_primary,
        args.base_dir,
        site_date_ranges_csv=args.site_date_ranges_csv,
    )
    figure1b_single_panel_anonymized(
        loaded.table1,
        args.output_dir,
        args.window_primary,
        args.base_dir,
        site_date_ranges_csv=args.site_date_ranges_csv,
    )
    figure2_survival_vitals(hourly_clean, args.output_dir, window_hours=24)
    figure2_survival_vitals(hourly_clean, args.output_dir, window_hours=72)
    figure3_trajectory_phenotypes(loaded.table1, hourly_clean, args.output_dir, args.window_primary)
    figure4_temperature_measurement_density(loaded.table1, hourly_clean, args.output_dir, window_hours=72)

    print("\nSaved preliminary-study figure package:")
    for p in sorted(args.output_dir.glob("figure*.png")):
        print(f"  - {p}")
    print("\nCompanion CSVs:")
    for p in sorted(args.output_dir.glob("figure*.csv")):
        print(f"  - {p}")


if __name__ == "__main__":
    main()
