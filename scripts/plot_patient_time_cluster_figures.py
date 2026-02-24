#!/usr/bin/env python3
"""
Auto-generate five figure panels for patient-time clustering outputs.

Inputs expected in clustering output directories:
  - cluster_hourly_profiles.csv
  - timepoint_cluster_assignments.csv
  - cluster_transition_counts.csv
  - cluster_centroids_feature_space.csv
  - k_selection_diagnostics.csv
  - run_summary.txt (optional)

Figures:
  1) Cluster trajectories over clock time (4 vital panels)
  2) State occupancy over time (overall + by site)
  3) Transition dynamics heatmap
  4) Outcome association (dominant vs first-hour cluster mortality)
  5) Imputation sensitivity panel (primary vs sensitivity run)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

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
    parser = argparse.ArgumentParser(description="Build patient-time clustering summary figures.")
    parser.add_argument(
        "--primary-dir",
        type=Path,
        required=True,
        help="Primary clustering output dir (typically last_value run).",
    )
    parser.add_argument(
        "--sensitivity-dir",
        type=Path,
        default=None,
        help="Optional secondary clustering dir (typically hourly_mean run) for figure 5.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir for figures and companion CSVs. Defaults to <primary-dir>/figures.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="Patient-Time Clustering",
        help="Prefix for figure titles.",
    )
    return parser.parse_args()


def _read_required_csv(base_dir: Path, filename: str) -> pd.DataFrame:
    p = base_dir / filename
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return pd.read_csv(p)


def _parse_run_summary(run_summary_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not run_summary_path.exists():
        return out
    for line in run_summary_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _clusters_from_frames(frames: Iterable[pd.DataFrame]) -> list[int]:
    vals: set[int] = set()
    for df in frames:
        for c in ("cluster", "next_cluster"):
            if c in df.columns:
                vals.update(pd.to_numeric(df[c], errors="coerce").dropna().astype(int).tolist())
    return sorted(vals)


def _cluster_color_map(clusters: list[int]) -> dict[int, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {c: cmap((i % 10) / 10.0) for i, c in enumerate(clusters)}


def _transition_prob_matrix(trans_counts: pd.DataFrame, clusters: list[int]) -> pd.DataFrame:
    mat = (
        trans_counts.pivot_table(
            index="cluster",
            columns="next_cluster",
            values="n_transitions",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=clusters, columns=clusters, fill_value=0.0)
        .astype(float)
    )
    row_sum = mat.sum(axis=1).replace(0.0, np.nan)
    prob = mat.div(row_sum, axis=0).fillna(0.0)
    return prob


def figure1_cluster_trajectories(primary_dir: Path, output_dir: Path, title_prefix: str) -> None:
    hourly = _read_required_csv(primary_dir, "cluster_hourly_profiles.csv")
    size = _read_required_csv(primary_dir, "cluster_size_by_site_survival.csv")
    clusters = sorted(hourly["cluster"].dropna().astype(int).unique().tolist())
    colors = _cluster_color_map(clusters)
    n_lookup = size.groupby("cluster")["n_patient_hours"].sum().to_dict()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, vital in zip(axes, CORE_VITALS):
        for c in clusters:
            sub = hourly[hourly["cluster"].astype(int) == c].sort_values("hour")
            if sub.empty or vital not in sub.columns:
                continue
            n_h = int(n_lookup.get(c, 0))
            ax.plot(
                sub["hour"],
                sub[vital],
                color=colors[c],
                linewidth=2.0,
                label=f"Cluster {c} (N-h={n_h:,})",
            )
        ax.set_title(VITAL_LABELS[vital])
        ax.set_xlabel("Hour from time-zero")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(f"{title_prefix}: Figure 1. Cluster Trajectories Over Time", fontsize=14)
    fig.savefig(output_dir / "figure1_cluster_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _occupancy_pct(assign: pd.DataFrame, clusters: list[int]) -> pd.DataFrame:
    counts = (
        assign.groupby(["hour", "cluster"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    wide = (
        counts.pivot_table(index="hour", columns="cluster", values="n", aggfunc="sum", fill_value=0)
        .reindex(columns=clusters, fill_value=0)
        .sort_index()
    )
    pct = wide.div(wide.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return pct


def figure2_state_occupancy(primary_dir: Path, output_dir: Path, title_prefix: str) -> None:
    assign = _read_required_csv(primary_dir, "timepoint_cluster_assignments.csv")
    clusters = sorted(assign["cluster"].dropna().astype(int).unique().tolist())
    colors = _cluster_color_map(clusters)
    sites = sorted(assign["site"].astype(str).unique().tolist())

    nrows = 1 + len(sites)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 2.6 + 2.0 * nrows), sharex=True, constrained_layout=True)
    if nrows == 1:
        axes = [axes]

    overall_pct = _occupancy_pct(assign, clusters)
    x = overall_pct.index.to_numpy()
    ys = [overall_pct[c].to_numpy() for c in clusters]
    axes[0].stackplot(x, ys, colors=[colors[c] for c in clusters], alpha=0.85)
    axes[0].set_ylabel("Overall\nProportion")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, axis="y", alpha=0.25)

    long_parts: list[pd.DataFrame] = []
    ov_long = (
        overall_pct.reset_index()
        .melt(id_vars="hour", var_name="cluster", value_name="occupancy_pct")
        .assign(site="ALL")
    )
    long_parts.append(ov_long)

    for i, site in enumerate(sites, start=1):
        sub = assign[assign["site"].astype(str) == site].copy()
        pct = _occupancy_pct(sub, clusters)
        x_site = pct.index.to_numpy()
        ys_site = [pct[c].to_numpy() for c in clusters]
        axes[i].stackplot(x_site, ys_site, colors=[colors[c] for c in clusters], alpha=0.85)
        axes[i].set_ylabel(f"{site}\nProportion")
        axes[i].set_ylim(0, 1.0)
        axes[i].grid(True, axis="y", alpha=0.25)
        long_parts.append(
            pct.reset_index().melt(id_vars="hour", var_name="cluster", value_name="occupancy_pct").assign(site=site)
        )

    axes[-1].set_xlabel("Hour from time-zero")
    legend_handles = [plt.Line2D([0], [0], color=colors[c], lw=6) for c in clusters]
    fig.legend(legend_handles, [f"Cluster {c}" for c in clusters], loc="upper right", ncol=min(4, len(clusters)))
    fig.suptitle(f"{title_prefix}: Figure 2. Cluster State Occupancy Over Time", fontsize=14)
    fig.savefig(output_dir / "figure2_state_occupancy_over_time.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    occupancy_long = pd.concat(long_parts, ignore_index=True)
    occupancy_long.to_csv(output_dir / "figure2_state_occupancy_over_time.csv", index=False)


def figure3_transition_heatmap(primary_dir: Path, output_dir: Path, title_prefix: str) -> None:
    trans = _read_required_csv(primary_dir, "cluster_transition_counts.csv")
    clusters = _clusters_from_frames([trans])
    if not clusters:
        raise ValueError("No clusters found in transition counts.")
    prob = _transition_prob_matrix(trans, clusters)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(prob.to_numpy(), cmap="Blues", vmin=0.0, vmax=max(0.25, float(prob.to_numpy().max())))
    ax.set_xticks(np.arange(len(clusters)))
    ax.set_yticks(np.arange(len(clusters)))
    ax.set_xticklabels([f"C{c}" for c in clusters])
    ax.set_yticklabels([f"C{c}" for c in clusters])
    ax.set_xlabel("Next cluster")
    ax.set_ylabel("Current cluster")
    ax.set_title("Row-normalized transition probabilities")
    for i, c0 in enumerate(clusters):
        for j, c1 in enumerate(clusters):
            v = float(prob.loc[c0, c1])
            ax.text(j, i, f"{v:.2f}" if v >= 0.01 else ".", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability")
    fig.suptitle(f"{title_prefix}: Figure 3. Transition Dynamics Heatmap", fontsize=14)
    fig.savefig(output_dir / "figure3_transition_dynamics_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    prob_out = prob.copy()
    prob_out.index.name = "cluster"
    prob_out.reset_index().to_csv(output_dir / "figure3_transition_probabilities.csv", index=False)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    den = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2.0 * n)) / den
    half = z * np.sqrt((phat * (1.0 - phat) + (z**2) / (4.0 * n)) / n) / den
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _dominant_cluster(assign: pd.DataFrame) -> pd.DataFrame:
    counts = (
        assign.groupby(["patient_key", "cluster"], as_index=False)
        .size()
        .rename(columns={"size": "n_hours"})
    )
    counts = counts.sort_values(["patient_key", "n_hours", "cluster"], ascending=[True, False, True])
    return counts.drop_duplicates("patient_key", keep="first")[["patient_key", "cluster"]]


def _first_hour_cluster(assign: pd.DataFrame) -> pd.DataFrame:
    ordered = assign.sort_values(["patient_key", "hour", "cluster"])
    return ordered.drop_duplicates("patient_key", keep="first")[["patient_key", "cluster"]]


def _mortality_by_cluster(patient_clusters: pd.DataFrame, patient_meta: pd.DataFrame, label: str) -> pd.DataFrame:
    df = patient_clusters.merge(patient_meta, on="patient_key", how="inner")
    df["is_dead"] = df["survival_status"].astype(str).eq("Non-Survivor").astype(int)
    rows = []
    for c, sub in df.groupby("cluster"):
        n = int(sub["patient_key"].nunique())
        dead = int(sub["is_dead"].sum())
        p = dead / n if n > 0 else np.nan
        lo, hi = _wilson_ci(dead, n)
        rows.append(
            {
                "summary_type": label,
                "cluster": int(c),
                "n_patients": n,
                "n_dead": dead,
                "mortality_pct": p * 100.0,
                "ci_lo_pct": lo * 100.0,
                "ci_hi_pct": hi * 100.0,
            }
        )
    return pd.DataFrame(rows).sort_values("cluster")


def figure4_outcome_association(primary_dir: Path, output_dir: Path, title_prefix: str) -> None:
    assign = _read_required_csv(primary_dir, "timepoint_cluster_assignments.csv")
    patient_meta = assign[["patient_key", "survival_status"]].drop_duplicates("patient_key")

    dom = _dominant_cluster(assign[["patient_key", "cluster", "hour"]].copy())
    first = _first_hour_cluster(assign[["patient_key", "cluster", "hour"]].copy())
    dom_sum = _mortality_by_cluster(dom, patient_meta, label="dominant_cluster")
    first_sum = _mortality_by_cluster(first, patient_meta, label="first_hour_cluster")
    out = pd.concat([dom_sum, first_sum], ignore_index=True)
    out.to_csv(output_dir / "figure4_outcome_association.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True, constrained_layout=True)
    for ax, label, sub in [
        (axes[0], "Dominant Cluster", dom_sum),
        (axes[1], "First-Hour Cluster", first_sum),
    ]:
        clusters = sub["cluster"].astype(int).tolist()
        y = sub["mortality_pct"].to_numpy()
        err_lo = (sub["mortality_pct"] - sub["ci_lo_pct"]).to_numpy()
        err_hi = (sub["ci_hi_pct"] - sub["mortality_pct"]).to_numpy()
        ax.bar(clusters, y, color="#4C78A8", alpha=0.85)
        ax.errorbar(clusters, y, yerr=[err_lo, err_hi], fmt="none", ecolor="#1f2937", capsize=4, lw=1.4)
        for _, r in sub.iterrows():
            ax.text(int(r["cluster"]), float(r["mortality_pct"]) + 1.0, f"n={int(r['n_patients'])}", ha="center", va="bottom", fontsize=9)
        ax.set_title(label)
        ax.set_xlabel("Cluster")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Mortality (%)")
    axes[0].set_ylim(0, min(100, max(30, float(out["ci_hi_pct"].max()) + 8)))
    fig.suptitle(f"{title_prefix}: Figure 4. Outcome Association by Cluster", fontsize=14)
    fig.savefig(output_dir / "figure4_outcome_association.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _infer_sensitivity_dir(primary_dir: Path) -> Path | None:
    name = primary_dir.name
    if "last_value" in name:
        cand = primary_dir.parent / name.replace("last_value", "hourly_mean")
        return cand if cand.exists() else None
    if "hourly_mean" in name:
        cand = primary_dir.parent / name.replace("hourly_mean", "last_value")
        return cand if cand.exists() else None
    return None


def _heatmap(ax: plt.Axes, mat: np.ndarray, xticks: list[str], yticks: list[str], title: str, cmap: str, vmin: float, vmax: float) -> None:
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_xticklabels(xticks, rotation=0)
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels(yticks)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}" if abs(v) >= 0.01 else ".", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def figure5_imputation_sensitivity(
    primary_dir: Path,
    sensitivity_dir: Path | None,
    output_dir: Path,
    title_prefix: str,
) -> None:
    if sensitivity_dir is None or not sensitivity_dir.exists():
        fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
        ax.axis("off")
        ax.text(
            0.01,
            0.70,
            "Figure 5 not available: no sensitivity directory found/provided.",
            fontsize=12,
            ha="left",
            va="center",
        )
        ax.text(
            0.01,
            0.40,
            "Provide --sensitivity-dir (e.g., hourly_mean run) to compare imputation strategies.",
            fontsize=10,
            ha="left",
            va="center",
        )
        fig.savefig(output_dir / "figure5_imputation_sensitivity.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    p_cent = _read_required_csv(primary_dir, "cluster_centroids_feature_space.csv")
    s_cent = _read_required_csv(sensitivity_dir, "cluster_centroids_feature_space.csv")
    p_trans = _read_required_csv(primary_dir, "cluster_transition_counts.csv")
    s_trans = _read_required_csv(sensitivity_dir, "cluster_transition_counts.csv")
    p_k = _read_required_csv(primary_dir, "k_selection_diagnostics.csv")
    s_k = _read_required_csv(sensitivity_dir, "k_selection_diagnostics.csv")
    p_meta = _parse_run_summary(primary_dir / "run_summary.txt")
    s_meta = _parse_run_summary(sensitivity_dir / "run_summary.txt")

    p_impute = p_meta.get("timepoint_impute", primary_dir.name)
    s_impute = s_meta.get("timepoint_impute", sensitivity_dir.name)

    p_cent = p_cent[["cluster"] + CORE_VITALS].copy().sort_values("cluster")
    s_cent = s_cent[["cluster"] + CORE_VITALS].copy().sort_values("cluster")
    p_cent["cluster"] = p_cent["cluster"].astype(int)
    s_cent["cluster"] = s_cent["cluster"].astype(int)
    p_cent_mat = p_cent[CORE_VITALS].to_numpy(dtype=float)
    s_cent_mat = s_cent[CORE_VITALS].to_numpy(dtype=float)

    both = np.vstack([p_cent_mat, s_cent_mat])
    mu = np.nanmean(both, axis=0)
    sd = np.nanstd(both, axis=0)
    sd[sd == 0.0] = 1.0
    p_cent_z = (p_cent_mat - mu) / sd
    s_cent_z = (s_cent_mat - mu) / sd

    p_clusters = _clusters_from_frames([p_trans])
    s_clusters = _clusters_from_frames([s_trans])
    p_prob = _transition_prob_matrix(p_trans, p_clusters)
    s_prob = _transition_prob_matrix(s_trans, s_clusters)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_c1 = fig.add_subplot(gs[0, 0])
    ax_c2 = fig.add_subplot(gs[0, 1])
    ax_k = fig.add_subplot(gs[0, 2])
    ax_t1 = fig.add_subplot(gs[1, 0])
    ax_t2 = fig.add_subplot(gs[1, 1])
    ax_txt = fig.add_subplot(gs[1, 2])

    _heatmap(
        ax_c1,
        p_cent_z,
        xticks=[VITAL_LABELS[v] for v in CORE_VITALS],
        yticks=[f"C{c}" for c in p_cent["cluster"].tolist()],
        title=f"Centroids (z by vital): {p_impute}",
        cmap="coolwarm",
        vmin=-2.5,
        vmax=2.5,
    )
    _heatmap(
        ax_c2,
        s_cent_z,
        xticks=[VITAL_LABELS[v] for v in CORE_VITALS],
        yticks=[f"C{c}" for c in s_cent["cluster"].tolist()],
        title=f"Centroids (z by vital): {s_impute}",
        cmap="coolwarm",
        vmin=-2.5,
        vmax=2.5,
    )

    _heatmap(
        ax_t1,
        p_prob.to_numpy(),
        xticks=[f"C{c}" for c in p_prob.columns.tolist()],
        yticks=[f"C{c}" for c in p_prob.index.tolist()],
        title=f"Transitions: {p_impute}",
        cmap="Blues",
        vmin=0.0,
        vmax=max(0.25, float(p_prob.to_numpy().max())),
    )
    _heatmap(
        ax_t2,
        s_prob.to_numpy(),
        xticks=[f"C{c}" for c in s_prob.columns.tolist()],
        yticks=[f"C{c}" for c in s_prob.index.tolist()],
        title=f"Transitions: {s_impute}",
        cmap="Blues",
        vmin=0.0,
        vmax=max(0.25, float(s_prob.to_numpy().max())),
    )

    p_k = p_k.sort_values("k")
    s_k = s_k.sort_values("k")
    ax_k.plot(p_k["k"], p_k["davies_bouldin"], marker="o", linewidth=2, label=p_impute)
    ax_k.plot(s_k["k"], s_k["davies_bouldin"], marker="o", linewidth=2, label=s_impute)
    if "selected" in p_k.columns and p_k["selected"].astype(str).str.lower().isin(["true", "1"]).any():
        sel = p_k[p_k["selected"].astype(str).str.lower().isin(["true", "1"])].iloc[0]
        ax_k.scatter([sel["k"]], [sel["davies_bouldin"]], s=80, color="black", zorder=5)
    if "selected" in s_k.columns and s_k["selected"].astype(str).str.lower().isin(["true", "1"]).any():
        sel = s_k[s_k["selected"].astype(str).str.lower().isin(["true", "1"])].iloc[0]
        ax_k.scatter([sel["k"]], [sel["davies_bouldin"]], s=80, color="black", zorder=5)
    ax_k.set_title("K-Selection Diagnostics (Davies-Bouldin)")
    ax_k.set_xlabel("k")
    ax_k.set_ylabel("Davies-Bouldin (lower better)")
    ax_k.grid(True, alpha=0.3)
    ax_k.legend()

    p_sel = p_k[p_k["selected"].astype(str).str.lower().isin(["true", "1"])]
    s_sel = s_k[s_k["selected"].astype(str).str.lower().isin(["true", "1"])]
    p_sel_k = int(p_sel.iloc[0]["k"]) if not p_sel.empty else None
    s_sel_k = int(s_sel.iloc[0]["k"]) if not s_sel.empty else None
    p_sel_db = float(p_sel.iloc[0]["davies_bouldin"]) if not p_sel.empty else np.nan
    s_sel_db = float(s_sel.iloc[0]["davies_bouldin"]) if not s_sel.empty else np.nan

    ax_txt.axis("off")
    ax_txt.text(0.01, 0.90, "Sensitivity Summary", fontsize=13, ha="left", va="top")
    ax_txt.text(
        0.01,
        0.70,
        f"{p_impute}: selected k={p_sel_k}, DB={p_sel_db:.3f}\n"
        f"{s_impute}: selected k={s_sel_k}, DB={s_sel_db:.3f}",
        fontsize=11,
        ha="left",
        va="top",
    )
    ax_txt.text(
        0.01,
        0.38,
        "Interpretation:\n"
        "- Lower DB suggests cleaner separation.\n"
        "- Compare centroid/transition stability across imputations.",
        fontsize=10,
        ha="left",
        va="top",
    )

    fig.suptitle(f"{title_prefix}: Figure 5. Imputation Sensitivity", fontsize=14)
    fig.savefig(output_dir / "figure5_imputation_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    cmp = pd.DataFrame(
        [
            {"method": p_impute, "selected_k": p_sel_k, "selected_db": p_sel_db},
            {"method": s_impute, "selected_k": s_sel_k, "selected_db": s_sel_db},
        ]
    )
    cmp.to_csv(output_dir / "figure5_imputation_sensitivity_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    primary_dir = args.primary_dir
    if not primary_dir.exists():
        raise FileNotFoundError(f"Primary directory not found: {primary_dir}")

    sensitivity_dir = args.sensitivity_dir
    if sensitivity_dir is None:
        sensitivity_dir = _infer_sensitivity_dir(primary_dir)

    output_dir = args.output_dir if args.output_dir is not None else (primary_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Figure 1: cluster trajectories...")
    figure1_cluster_trajectories(primary_dir, output_dir, args.title_prefix)
    print("[2/5] Figure 2: state occupancy...")
    figure2_state_occupancy(primary_dir, output_dir, args.title_prefix)
    print("[3/5] Figure 3: transition heatmap...")
    figure3_transition_heatmap(primary_dir, output_dir, args.title_prefix)
    print("[4/5] Figure 4: outcome association...")
    figure4_outcome_association(primary_dir, output_dir, args.title_prefix)
    print("[5/5] Figure 5: imputation sensitivity...")
    figure5_imputation_sensitivity(primary_dir, sensitivity_dir, output_dir, args.title_prefix)
    print(f"Done. Figures written to: {output_dir}")


if __name__ == "__main__":
    main()
