#!/usr/bin/env python3
"""
Create per-patient clustering figures from patient-level clustering outputs.

Expected input files in --cluster-dir:
  - patient_cluster_assignments.csv
  - cluster_hourly_profiles.csv
  - cluster_size_by_site_survival.csv
  - cluster_centroids_feature_space.csv
  - k_selection_diagnostics.csv
"""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CORE_VITALS = ["heart_rate", "map", "spo2", "temp_c"]
VITAL_LABELS = {
    "heart_rate": "Heart Rate (bpm)",
    "map": "MAP (mmHg)",
    "spo2": "SpO2 (%)",
    "temp_c": "Temperature (C)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-patient cluster figures.")
    parser.add_argument(
        "--cluster-dir",
        type=Path,
        required=True,
        help="Directory containing per-patient clustering outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: <cluster-dir>/figures).",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="UCMC+MIMIC Per-Patient Clustering",
        help="Prefix for figure titles.",
    )
    return parser.parse_args()


def _read_required_csv(base: Path, filename: str) -> pd.DataFrame:
    p = base / filename
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return pd.read_csv(p)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / den
    half = z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def _cluster_colors(clusters: list[int]) -> dict[int, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {c: cmap((i % 10) / 10.0) for i, c in enumerate(clusters)}


def figure1_trajectories(hourly: pd.DataFrame, assign: pd.DataFrame, output_dir: Path, title_prefix: str) -> None:
    clusters = sorted(hourly["cluster"].dropna().astype(int).unique().tolist())
    colors = _cluster_colors(clusters)
    n_by_cluster = assign.groupby("cluster")["patient_key"].nunique().to_dict()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, vital in zip(axes, CORE_VITALS):
        for c in clusters:
            sub = hourly[hourly["cluster"].astype(int) == c].sort_values("hour")
            if sub.empty:
                continue
            n = int(n_by_cluster.get(c, 0))
            ax.plot(
                sub["hour"],
                sub[vital],
                color=colors[c],
                linewidth=2,
                label=f"Cluster {c} (n={n:,})",
            )
        ax.set_title(VITAL_LABELS[vital])
        ax.set_xlabel("Hour from time-zero")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(f"{title_prefix}: Figure 1. Cluster Trajectories", fontsize=14)
    fig.savefig(output_dir / "figure1_cluster_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure2_composition(assign: pd.DataFrame, output_dir: Path, title_prefix: str) -> None:
    clusters = sorted(assign["cluster"].dropna().astype(int).unique().tolist())
    colors = _cluster_colors(clusters)
    sites = sorted(assign["site"].astype(str).unique().tolist())

    site_cluster = (
        assign.groupby(["site", "cluster"], as_index=False)["patient_key"]
        .nunique()
        .rename(columns={"patient_key": "n_patients"})
    )
    surv_cluster = (
        assign.groupby(["cluster", "survival_status"], as_index=False)["patient_key"]
        .nunique()
        .rename(columns={"patient_key": "n_patients"})
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    x = np.arange(len(sites))
    bottom = np.zeros(len(sites), dtype=float)
    for c in clusters:
        y = np.array(
            [
                site_cluster.loc[
                    (site_cluster["site"] == s) & (site_cluster["cluster"].astype(int) == c),
                    "n_patients",
                ].sum()
                for s in sites
            ],
            dtype=float,
        )
        axes[0].bar(x, y, bottom=bottom, color=colors[c], label=f"Cluster {c}")
        bottom += y
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sites, rotation=20, ha="right")
    axes[0].set_ylabel("Patients")
    axes[0].set_title("A. Cluster Composition by Site")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, axis="y", alpha=0.25)

    survival_levels = ["Survivor", "Non-Survivor"]
    x2 = np.arange(len(clusters))
    width = 0.42
    for i, status in enumerate(survival_levels):
        y = np.array(
            [
                surv_cluster.loc[
                    (surv_cluster["cluster"].astype(int) == c)
                    & (surv_cluster["survival_status"].astype(str) == status),
                    "n_patients",
                ].sum()
                for c in clusters
            ],
            dtype=float,
        )
        axes[1].bar(
            x2 + (i - 0.5) * width,
            y,
            width=width,
            label=status,
            color="#1E88E5" if status == "Survivor" else "#E53935",
            alpha=0.85,
        )
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([f"C{c}" for c in clusters])
    axes[1].set_ylabel("Patients")
    axes[1].set_title("B. Survival Counts by Cluster")
    axes[1].legend(loc="best")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"{title_prefix}: Figure 2. Cluster Composition", fontsize=14)
    fig.savefig(output_dir / "figure2_cluster_composition.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    site_cluster.to_csv(output_dir / "figure2_site_cluster_counts.csv", index=False)
    surv_cluster.to_csv(output_dir / "figure2_survival_by_cluster_counts.csv", index=False)


def figure3_mortality(assign: pd.DataFrame, output_dir: Path, title_prefix: str) -> None:
    rows = []
    for c, sub in assign.groupby("cluster"):
        n = int(sub["patient_key"].nunique())
        dead = int(sub[sub["survival_status"].astype(str) == "Non-Survivor"]["patient_key"].nunique())
        p = (dead / n) if n > 0 else np.nan
        lo, hi = _wilson_ci(dead, n)
        rows.append(
            {
                "cluster": int(c),
                "n_patients": n,
                "n_dead": dead,
                "mortality_pct": p * 100.0,
                "ci_lo_pct": lo * 100.0,
                "ci_hi_pct": hi * 100.0,
            }
        )
    out = pd.DataFrame(rows).sort_values("cluster")
    out.to_csv(output_dir / "figure3_mortality_by_cluster.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = out["cluster"].astype(int).to_numpy()
    y = out["mortality_pct"].to_numpy()
    err_lo = (out["mortality_pct"] - out["ci_lo_pct"]).to_numpy()
    err_hi = (out["ci_hi_pct"] - out["mortality_pct"]).to_numpy()
    ax.bar(x, y, color="#4C78A8", alpha=0.9)
    ax.errorbar(x, y, yerr=[err_lo, err_hi], fmt="none", ecolor="#1f2937", capsize=4, lw=1.4)
    for _, r in out.iterrows():
        ax.text(
            int(r["cluster"]),
            float(r["mortality_pct"]) + 1.0,
            f"n={int(r['n_patients'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mortality (%)")
    ax.set_ylim(0, min(100, max(30, float(out["ci_hi_pct"].max()) + 8)))
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title(f"{title_prefix}: Figure 3. Mortality by Cluster")
    fig.savefig(output_dir / "figure3_mortality_by_cluster.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _extract_centroid_tensor(centroids: pd.DataFrame) -> tuple[list[int], np.ndarray]:
    clusters = centroids["cluster"].astype(int).tolist()
    hours = sorted({int(c.split("_h")[-1]) for c in centroids.columns if "_h" in c})
    tensor = np.full((len(clusters), len(CORE_VITALS), len(hours)), np.nan, dtype=float)
    for i, _ in enumerate(clusters):
        for v_idx, vital in enumerate(CORE_VITALS):
            for h_idx, h in enumerate(hours):
                col = f"{vital}_h{h:03d}"
                if col in centroids.columns:
                    tensor[i, v_idx, h_idx] = float(centroids.loc[i, col])
    return clusters, tensor


def figure4_centroid_heatmaps(centroids: pd.DataFrame, output_dir: Path, title_prefix: str) -> None:
    clusters, tensor = _extract_centroid_tensor(centroids.sort_values("cluster").reset_index(drop=True))

    flat = tensor.reshape(-1, tensor.shape[-1])
    z = np.empty_like(tensor)
    for v_idx in range(len(CORE_VITALS)):
        x = tensor[:, v_idx, :].reshape(-1)
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        z[:, v_idx, :] = (tensor[:, v_idx, :] - mu) / sd

    n = len(clusters)
    ncols = min(3, n)
    nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    vmax = np.nanmax(np.abs(z))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 2.0

    long_rows = []
    for i, c in enumerate(clusters):
        ax = axes[i]
        im = ax.imshow(z[i], aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Cluster {c}")
        ax.set_yticks(np.arange(len(CORE_VITALS)))
        ax.set_yticklabels([VITAL_LABELS[v] for v in CORE_VITALS])
        ax.set_xlabel("Hour")
        ax.set_xticks(np.arange(0, z.shape[2], max(1, z.shape[2] // 6)))
        for v_idx, vital in enumerate(CORE_VITALS):
            for h in range(z.shape[2]):
                long_rows.append(
                    {
                        "cluster": int(c),
                        "vital": vital,
                        "hour": int(h),
                        "centroid_value": float(tensor[i, v_idx, h]),
                        "centroid_z": float(z[i, v_idx, h]),
                    }
                )
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    cbar = fig.colorbar(im, ax=axes[: nrows * ncols], fraction=0.022, pad=0.02)
    cbar.set_label("Centroid z-score (within vital)")
    fig.suptitle(f"{title_prefix}: Figure 4. Cluster Centroid Heatmaps", fontsize=14)
    fig.savefig(output_dir / "figure4_centroid_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(long_rows).to_csv(output_dir / "figure4_centroid_heatmaps_long.csv", index=False)


def figure5_k_diagnostics(kdiag: pd.DataFrame, output_dir: Path, title_prefix: str) -> None:
    kdiag = kdiag.sort_values("k").copy()
    kdiag["selected_bool"] = kdiag["selected"].astype(str).str.lower().isin(["true", "1", "yes"])
    kdiag.to_csv(output_dir / "figure5_k_diagnostics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].plot(kdiag["k"], kdiag["inertia"], marker="o", linewidth=2, color="#4C78A8")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("A. Elbow (Inertia)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(kdiag["k"], kdiag["davies_bouldin"], marker="o", linewidth=2, color="#F58518")
    sel = kdiag[kdiag["selected_bool"]]
    if not sel.empty:
        axes[1].scatter(sel["k"], sel["davies_bouldin"], s=80, color="black", zorder=5, label="Selected")
        axes[1].legend(loc="best")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Davies-Bouldin")
    axes[1].set_title("B. Separation (Lower Better)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{title_prefix}: Figure 5. K Selection Diagnostics", fontsize=14)
    fig.savefig(output_dir / "figure5_k_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cluster_dir = args.cluster_dir
    if not cluster_dir.exists():
        raise FileNotFoundError(f"Cluster dir not found: {cluster_dir}")

    output_dir = args.output_dir if args.output_dir is not None else (cluster_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    assign = _read_required_csv(cluster_dir, "patient_cluster_assignments.csv")
    hourly = _read_required_csv(cluster_dir, "cluster_hourly_profiles.csv")
    centroids = _read_required_csv(cluster_dir, "cluster_centroids_feature_space.csv")
    kdiag = _read_required_csv(cluster_dir, "k_selection_diagnostics.csv")

    print("[1/5] Figure 1 trajectories...")
    figure1_trajectories(hourly, assign, output_dir, args.title_prefix)
    print("[2/5] Figure 2 composition...")
    figure2_composition(assign, output_dir, args.title_prefix)
    print("[3/5] Figure 3 mortality...")
    figure3_mortality(assign, output_dir, args.title_prefix)
    print("[4/5] Figure 4 centroid heatmaps...")
    figure4_centroid_heatmaps(centroids, output_dir, args.title_prefix)
    print("[5/5] Figure 5 k diagnostics...")
    figure5_k_diagnostics(kdiag, output_dir, args.title_prefix)

    print(f"Done. Figures written to: {output_dir}")


if __name__ == "__main__":
    main()
