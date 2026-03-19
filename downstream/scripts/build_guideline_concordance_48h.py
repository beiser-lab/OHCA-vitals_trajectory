#!/usr/bin/env python3
"""
Build pooled 48h guideline-concordance summaries and figure across sites.

Targets:
  - Glucose: 70-180 mg/dL
  - MAP: >= 65 mmHg
  - SpO2: 90-98 %
  - Temperature: 32.0-37.5 C

Notes:
  - Vitals use shared hourly trajectory x survival summaries (site-level pooled rows).
  - Glucose uses shared 6h epoch trajectory x survival summaries when available.
  - Without patient identifiers in shared pooled files, concordance is estimated from
    row-level mean/sd assuming approximate normality within each pooled row.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    metric: str
    label: str
    source: str
    mean_col: str
    sd_col: str
    n_col: str
    lo: float | None
    hi: float | None
    plausible_lo: float
    plausible_hi: float


METRICS = [
    MetricSpec(
        metric="glucose",
        label="Glucose 70-180 mg/dL",
        source="glucose",
        mean_col="mean_blood_glucose",
        sd_col="sd_blood_glucose",
        n_col="n_blood_glucose",
        lo=70.0,
        hi=180.0,
        plausible_lo=20.0,
        plausible_hi=1000.0,
    ),
    MetricSpec(
        metric="map",
        label="MAP >= 65 mmHg",
        source="vitals",
        mean_col="mean_map",
        sd_col="sd_map",
        n_col="n_map",
        lo=65.0,
        hi=None,
        plausible_lo=30.0,
        plausible_hi=180.0,
    ),
    MetricSpec(
        metric="spo2",
        label="SpO2 90-98%",
        source="vitals",
        mean_col="mean_spo2",
        sd_col="sd_spo2",
        n_col="n_spo2",
        lo=90.0,
        hi=98.0,
        plausible_lo=50.0,
        plausible_hi=100.0,
    ),
    MetricSpec(
        metric="temp_c",
        label="Temp 32.0-37.5 C",
        source="vitals",
        mean_col="mean_temp_c",
        sd_col="sd_temp_c",
        n_col="n_temp_c",
        lo=32.0,
        hi=37.5,
        plausible_lo=32.0,
        plausible_hi=42.0,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pooled 48h guideline concordance figure and CSVs.")
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
        help="Output directory for figure and CSV outputs.",
    )
    parser.add_argument(
        "--max-hours",
        type=int,
        default=48,
        help="Analyze data where hour is in [0, max-hours).",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=72,
        choices=[24, 72],
        help="Source window file to read from (typically 72).",
    )
    return parser.parse_args()


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    vec_erf = np.vectorize(erf)
    return 0.5 * (1.0 + vec_erf(x / sqrt(2.0)))


def estimate_row_probability(mean: np.ndarray, sd: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
    out = np.full(mean.shape, np.nan, dtype=float)
    valid = np.isfinite(mean)
    if not valid.any():
        return out

    mean_v = mean[valid]
    sd_v = sd[valid]
    sd_pos = np.isfinite(sd_v) & (sd_v > 1e-12)

    p = np.full(mean_v.shape, np.nan, dtype=float)

    # Normal-approximation rows
    if sd_pos.any():
        mv = mean_v[sd_pos]
        sv = sd_v[sd_pos]
        if lo is None and hi is None:
            p_norm = np.ones(mv.shape, dtype=float)
        elif lo is None:
            p_norm = _norm_cdf((hi - mv) / sv)
        elif hi is None:
            p_norm = 1.0 - _norm_cdf((lo - mv) / sv)
        else:
            p_norm = _norm_cdf((hi - mv) / sv) - _norm_cdf((lo - mv) / sv)
        p[sd_pos] = p_norm

    # Deterministic rows (sd missing/zero): in-range indicator from mean
    if (~sd_pos).any():
        mv = mean_v[~sd_pos]
        if lo is None and hi is None:
            p_det = np.ones(mv.shape, dtype=float)
        elif lo is None:
            p_det = (mv <= hi).astype(float)
        elif hi is None:
            p_det = (mv >= lo).astype(float)
        else:
            p_det = ((mv >= lo) & (mv <= hi)).astype(float)
        p[~sd_pos] = p_det

    out[valid] = np.clip(p, 0.0, 1.0)
    return out


def load_vitals(base_dir: Path, window_hours: int) -> pd.DataFrame:
    files = sorted(base_dir.glob(f"*/Upload_to_Box_without_oral_{window_hours}/hourly_vitals_by_trajectory_survival_{window_hours}h.csv"))
    frames = []
    for p in files:
        df = pd.read_csv(p)
        if "site" not in df.columns:
            df["site"] = p.parent.parent.name
        df["source_file"] = str(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["site"] = out["site"].astype(str).str.strip().str.lower()
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce")
    return out


def load_glucose(base_dir: Path, workspace_dir: Path, window_hours: int) -> pd.DataFrame:
    files = set(base_dir.glob(f"*/Upload_to_Box_without_oral_glucose_{window_hours}/hourly_glucose_by_trajectory_survival_{window_hours}h.csv"))

    # Local fallback patterns used in this workspace for additional uploaded site bundles.
    files.update(workspace_dir.glob(f"Upload_to_Box_without_oral_glucose_{window_hours}/hourly_glucose_by_trajectory_survival_{window_hours}h.csv"))
    files.update(workspace_dir.glob(f"Upload_to_Box_without_oral_glucose_*_{window_hours}/hourly_glucose_by_trajectory_survival_{window_hours}h.csv"))

    frames = []
    for p in sorted(files):
        df = pd.read_csv(p)
        if "site" not in df.columns:
            df["site"] = p.parent.parent.name
        df["source_file"] = str(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates()
    out["site"] = out["site"].astype(str).str.strip().str.lower()
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce")
    return out


def estimate_site_concordance(df: pd.DataFrame, spec: MetricSpec, max_hours: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["site", "metric", "label", "concordance_pct", "concordant_n_est", "denominator_n", "rows_used"])

    d = df.copy()
    d = d[d["hour"].between(0, max_hours - 1, inclusive="both")].copy()
    if d.empty:
        return pd.DataFrame(columns=["site", "metric", "label", "concordance_pct", "concordant_n_est", "denominator_n", "rows_used"])

    for col in [spec.mean_col, spec.sd_col, spec.n_col]:
        if col not in d.columns:
            return pd.DataFrame(columns=["site", "metric", "label", "concordance_pct", "concordant_n_est", "denominator_n", "rows_used"])
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d[d[spec.n_col].fillna(0) > 0].copy()
    d = d[d[spec.mean_col].between(spec.plausible_lo, spec.plausible_hi, inclusive="both")].copy()
    if d.empty:
        return pd.DataFrame(columns=["site", "metric", "label", "concordance_pct", "concordant_n_est", "denominator_n", "rows_used"])

    p = estimate_row_probability(
        mean=d[spec.mean_col].to_numpy(dtype=float),
        sd=d[spec.sd_col].to_numpy(dtype=float),
        lo=spec.lo,
        hi=spec.hi,
    )
    d["p_concordant"] = p
    d = d[np.isfinite(d["p_concordant"])].copy()
    if d.empty:
        return pd.DataFrame(columns=["site", "metric", "label", "concordance_pct", "concordant_n_est", "denominator_n", "rows_used"])

    d["concordant_n_est"] = d[spec.n_col] * d["p_concordant"]
    out = (
        d.groupby("site", as_index=False)
        .agg(
            concordant_n_est=("concordant_n_est", "sum"),
            denominator_n=(spec.n_col, "sum"),
            rows_used=("site", "size"),
        )
    )
    out["concordance_pct"] = np.where(
        out["denominator_n"] > 0,
        out["concordant_n_est"] / out["denominator_n"] * 100.0,
        np.nan,
    )
    out["metric"] = spec.metric
    out["label"] = spec.label
    return out[["site", "metric", "label", "concordance_pct", "concordant_n_est", "denominator_n", "rows_used"]]


def pooled_summary(site_metric: pd.DataFrame, total_sites: int) -> pd.DataFrame:
    if site_metric.empty:
        return pd.DataFrame(columns=["metric", "label", "pooled_concordance_pct", "concordant_n_est", "denominator_n", "n_sites_available", "n_sites_total", "site_coverage_pct", "site_min_pct", "site_max_pct"])

    rows = []
    for metric in [m.metric for m in METRICS]:
        sub = site_metric[site_metric["metric"] == metric].copy()
        if sub.empty:
            rows.append(
                {
                    "metric": metric,
                    "label": next(m.label for m in METRICS if m.metric == metric),
                    "pooled_concordance_pct": np.nan,
                    "concordant_n_est": 0.0,
                    "denominator_n": 0.0,
                    "n_sites_available": 0,
                    "n_sites_total": total_sites,
                    "site_coverage_pct": 0.0,
                    "site_min_pct": np.nan,
                    "site_max_pct": np.nan,
                }
            )
            continue

        concordant = float(sub["concordant_n_est"].sum())
        denom = float(sub["denominator_n"].sum())
        pooled_pct = (concordant / denom * 100.0) if denom > 0 else np.nan
        rows.append(
            {
                "metric": metric,
                "label": sub["label"].iloc[0],
                "pooled_concordance_pct": pooled_pct,
                "concordant_n_est": concordant,
                "denominator_n": denom,
                "n_sites_available": int(sub["site"].nunique()),
                "n_sites_total": total_sites,
                "site_coverage_pct": float(sub["site"].nunique()) / float(total_sites) * 100.0 if total_sites > 0 else np.nan,
                "site_min_pct": float(sub["concordance_pct"].min()),
                "site_max_pct": float(sub["concordance_pct"].max()),
            }
        )
    out = pd.DataFrame(rows)
    order = [m.metric for m in METRICS]
    out["metric"] = pd.Categorical(out["metric"], categories=order, ordered=True)
    return out.sort_values("metric").reset_index(drop=True)


def plot_concordance(overall: pd.DataFrame, site_metric: pd.DataFrame, output_png: Path, max_hours: int) -> None:
    order = [m.metric for m in METRICS]
    labels = [m.label for m in METRICS]
    palette = {
        "glucose": "#455A64",
        "map": "#1E88E5",
        "spo2": "#43A047",
        "temp_c": "#E53935",
    }

    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    x = np.arange(len(order))
    y = [float(overall.loc[overall["metric"] == m, "pooled_concordance_pct"].iloc[0]) for m in order]
    bar_colors = [palette[m] for m in order]
    bars = ax.bar(x, y, color=bar_colors, alpha=0.9, width=0.66, zorder=2)

    rng = np.random.default_rng(7)
    for i, metric in enumerate(order):
        sub = site_metric[site_metric["metric"] == metric].copy()
        if sub.empty:
            continue
        smin = float(sub["concordance_pct"].min())
        smax = float(sub["concordance_pct"].max())
        ax.vlines(i, smin, smax, color="#333333", linewidth=1.0, alpha=0.65, zorder=3)
        jitter = rng.uniform(-0.09, 0.09, size=len(sub))
        ax.scatter(np.full(len(sub), i) + jitter, sub["concordance_pct"], s=14, color="#222222", alpha=0.72, zorder=4)

    for i, (b, metric) in enumerate(zip(bars, order)):
        sub = overall.loc[overall["metric"] == metric].iloc[0]
        yp = float(b.get_height())
        txt_main = "NA" if not np.isfinite(yp) else f"{yp:.1f}%"
        txt_n = f"{int(sub['n_sites_available'])}/{int(sub['n_sites_total'])} sites"
        ax.text(b.get_x() + b.get_width() / 2, (yp if np.isfinite(yp) else 0) + 2.0, txt_main, ha="center", va="bottom", fontsize=10)
        ax.text(b.get_x() + b.get_width() / 2, (yp if np.isfinite(yp) else 0) + 0.25, txt_n, ha="center", va="bottom", fontsize=8.5, color="#444444")

    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Estimated Concordance (%)")
    ax.set_title(f"Guideline Concordance Across Sites (First {max_hours}h)")
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.8, alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.01,
        -0.2,
        "Bars = pooled estimate. Points/range = site-level heterogeneity. "
        "Estimate uses row-level mean/sd from shared pooled files (normal approximation).",
        transform=ax.transAxes,
        fontsize=8.8,
        color="#4A4A4A",
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vitals = load_vitals(args.base_dir, args.window_hours)
    glucose = load_glucose(args.base_dir, Path.cwd(), args.window_hours)

    all_sites = sorted(vitals["site"].astype(str).str.strip().unique().tolist()) if not vitals.empty else []
    total_sites = len(all_sites)

    pieces = []
    for spec in METRICS:
        df = glucose if spec.source == "glucose" else vitals
        res = estimate_site_concordance(df, spec, args.max_hours)
        if not res.empty:
            res["source"] = spec.source
            pieces.append(res)

    site_metric = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["site", "metric", "label", "concordance_pct", "concordant_n_est", "denominator_n", "rows_used", "source"]
    )
    site_metric["site"] = site_metric["site"].astype(str).str.strip().str.lower()

    overall = pooled_summary(site_metric, total_sites=total_sites)

    # Save outputs
    site_metric.to_csv(args.output_dir / f"figure5_guideline_concordance_by_site_{args.max_hours}h.csv", index=False)
    overall.to_csv(args.output_dir / f"figure5_guideline_concordance_overall_{args.max_hours}h.csv", index=False)

    availability_rows = []
    for spec in METRICS:
        have = sorted(site_metric.loc[site_metric["metric"] == spec.metric, "site"].unique().tolist())
        missing = sorted(set(all_sites) - set(have))
        availability_rows.append(
            {
                "metric": spec.metric,
                "label": spec.label,
                "n_sites_available": len(have),
                "n_sites_total": total_sites,
                "sites_available": ";".join(have),
                "sites_missing": ";".join(missing),
            }
        )
    availability = pd.DataFrame(availability_rows)
    availability.to_csv(args.output_dir / f"figure5_guideline_concordance_availability_{args.max_hours}h.csv", index=False)

    plot_concordance(
        overall=overall,
        site_metric=site_metric,
        output_png=args.output_dir / f"figure5_guideline_concordance_{args.max_hours}h.png",
        max_hours=args.max_hours,
    )

    print("\nSaved guideline-concordance outputs:")
    for p in sorted(args.output_dir.glob(f"figure5_guideline_concordance*_{args.max_hours}h.*")):
        print(f"  - {p}")

    print("\nMetric site coverage:")
    for row in availability.itertuples():
        print(f"  - {row.metric}: {row.n_sites_available}/{row.n_sites_total} sites")


if __name__ == "__main__":
    main()
