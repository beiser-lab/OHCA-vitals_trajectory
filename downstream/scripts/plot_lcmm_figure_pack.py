#!/usr/bin/env python3
"""
Generate LCMM figure pack:
  1) class trajectories
  2) posterior certainty
  3) class composition
  4) mortality by class
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Plot LCMM figure pack.")
    parser.add_argument(
        "--lcmm-dir",
        type=Path,
        required=True,
        help="Directory containing LCMM outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: <lcmm-dir>/figures).",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="UCMC+MIMIC LCMM",
    )
    return parser.parse_args()


def _read_required_csv(base: Path, name: str) -> pd.DataFrame:
    p = base / name
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return pd.read_csv(p)


def _class_colors(classes: list[int]) -> dict[int, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {c: cmap((i % 10) / 10.0) for i, c in enumerate(classes)}


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    den = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / den
    half = z * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def figure1_trajectories(
    profiles: pd.DataFrame,
    assigns: pd.DataFrame,
    output_dir: Path,
    title_prefix: str,
) -> None:
    classes = sorted(profiles["class"].dropna().astype(int).unique().tolist())
    colors = _class_colors(classes)
    n_by_class = assigns.groupby("class")["patient_key"].nunique().to_dict()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, vital in zip(axes, CORE_VITALS):
        for c in classes:
            sub = profiles[profiles["class"].astype(int) == c].sort_values("hour")
            if sub.empty:
                continue
            ax.plot(
                sub["hour"],
                sub[vital],
                color=colors[c],
                linewidth=2.0,
                label=f"Class {c} (n={int(n_by_class.get(c, 0)):,})",
            )
        ax.set_title(VITAL_LABELS[vital])
        ax.set_xlabel("Hour from time-zero")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(f"{title_prefix}: Figure 1. LCMM Class Trajectories", fontsize=14)
    fig.savefig(output_dir / "figure1_lcmm_class_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    profiles.to_csv(output_dir / "figure1_lcmm_class_trajectories.csv", index=False)


def figure2_posterior_certainty(
    assigns: pd.DataFrame,
    output_dir: Path,
    title_prefix: str,
) -> None:
    classes = sorted(assigns["class"].dropna().astype(int).unique().tolist())
    colors = _class_colors(classes)
    x = assigns["max_posterior_prob"].astype(float).clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].hist(x, bins=30, color="#4C78A8", alpha=0.9, edgecolor="white")
    axes[0].axvline(float(x.median()), color="black", linestyle="--", linewidth=1.5, label=f"Median={x.median():.3f}")
    axes[0].set_xlabel("Max posterior probability")
    axes[0].set_ylabel("Patients")
    axes[0].set_title("A. Overall Assignment Certainty")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="best")

    data = [assigns.loc[assigns["class"].astype(int) == c, "max_posterior_prob"].astype(float).to_numpy() for c in classes]
    bp = axes[1].boxplot(data, patch_artist=True, tick_labels=[f"C{c}" for c in classes], showfliers=False)
    for patch, c in zip(bp["boxes"], classes):
        patch.set_facecolor(colors[c])
        patch.set_alpha(0.7)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Max posterior probability")
    axes[1].set_title("B. Certainty by Assigned Class")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"{title_prefix}: Figure 2. Posterior Certainty", fontsize=14)
    fig.savefig(output_dir / "figure2_lcmm_posterior_certainty.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    q = (
        assigns.groupby("class", as_index=False)["max_posterior_prob"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            q25=lambda s: float(s.quantile(0.25)),
            q75=lambda s: float(s.quantile(0.75)),
            min="min",
            max="max",
        )
        .sort_values("class")
    )
    q.to_csv(output_dir / "figure2_lcmm_posterior_certainty_summary.csv", index=False)


def figure3_class_composition(
    assigns: pd.DataFrame,
    output_dir: Path,
    title_prefix: str,
) -> None:
    classes = sorted(assigns["class"].dropna().astype(int).unique().tolist())
    colors = _class_colors(classes)

    by_site = (
        assigns.groupby(["site", "class"], as_index=False)["patient_key"]
        .nunique()
        .rename(columns={"patient_key": "n_patients"})
    )
    by_surv = (
        assigns.groupby(["survival_status", "class"], as_index=False)["patient_key"]
        .nunique()
        .rename(columns={"patient_key": "n_patients"})
    )
    sites = sorted(by_site["site"].astype(str).unique().tolist())
    survs = ["Survivor", "Non-Survivor"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    x_site = np.arange(len(sites))
    bottom = np.zeros(len(sites), dtype=float)
    for c in classes:
        y = np.array(
            [
                by_site.loc[
                    (by_site["site"] == s) & (by_site["class"].astype(int) == c),
                    "n_patients",
                ].sum()
                for s in sites
            ],
            dtype=float,
        )
        axes[0].bar(x_site, y, bottom=bottom, color=colors[c], label=f"Class {c}")
        bottom += y
    axes[0].set_xticks(x_site)
    axes[0].set_xticklabels(sites)
    axes[0].set_ylabel("Patients")
    axes[0].set_title("A. Composition by Site")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    x_surv = np.arange(len(survs))
    bottom2 = np.zeros(len(survs), dtype=float)
    for c in classes:
        y = np.array(
            [
                by_surv.loc[
                    (by_surv["survival_status"].astype(str) == s)
                    & (by_surv["class"].astype(int) == c),
                    "n_patients",
                ].sum()
                for s in survs
            ],
            dtype=float,
        )
        axes[1].bar(x_surv, y, bottom=bottom2, color=colors[c], label=f"Class {c}")
        bottom2 += y
    axes[1].set_xticks(x_surv)
    axes[1].set_xticklabels(survs)
    axes[1].set_ylabel("Patients")
    axes[1].set_title("B. Composition by Survival Status")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"{title_prefix}: Figure 3. Class Composition", fontsize=14)
    fig.savefig(output_dir / "figure3_lcmm_class_composition.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    by_site.to_csv(output_dir / "figure3_lcmm_class_composition_by_site.csv", index=False)
    by_surv.to_csv(output_dir / "figure3_lcmm_class_composition_by_survival.csv", index=False)


def figure4_mortality_by_class(
    assigns: pd.DataFrame,
    output_dir: Path,
    title_prefix: str,
) -> None:
    rows = []
    for c, sub in assigns.groupby("class"):
        n = int(sub["patient_key"].nunique())
        dead = int(sub[sub["survival_status"].astype(str) == "Non-Survivor"]["patient_key"].nunique())
        p = dead / n if n > 0 else np.nan
        lo, hi = _wilson_ci(dead, n)
        rows.append(
            {
                "class": int(c),
                "n_patients": n,
                "n_dead": dead,
                "mortality_pct": p * 100.0,
                "ci_lo_pct": lo * 100.0,
                "ci_hi_pct": hi * 100.0,
            }
        )
    out = pd.DataFrame(rows).sort_values("class")
    out.to_csv(output_dir / "figure4_lcmm_mortality_by_class.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = out["class"].astype(int).to_numpy()
    y = out["mortality_pct"].to_numpy()
    err_lo = (out["mortality_pct"] - out["ci_lo_pct"]).to_numpy()
    err_hi = (out["ci_hi_pct"] - out["mortality_pct"]).to_numpy()
    ax.bar(x, y, color="#4C78A8", alpha=0.9)
    ax.errorbar(x, y, yerr=[err_lo, err_hi], fmt="none", ecolor="#1f2937", capsize=4, lw=1.5)
    for _, r in out.iterrows():
        ax.text(
            int(r["class"]),
            float(r["mortality_pct"]) + 1.0,
            f"n={int(r['n_patients'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("LCMM Class")
    ax.set_ylabel("Mortality (%)")
    ax.set_ylim(0, min(100, max(30, float(out["ci_hi_pct"].max()) + 8)))
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title(f"{title_prefix}: Figure 4. Mortality by Class")
    fig.savefig(output_dir / "figure4_lcmm_mortality_by_class.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    lcmm_dir = args.lcmm_dir
    if not lcmm_dir.exists():
        raise FileNotFoundError(f"LCMM dir not found: {lcmm_dir}")

    output_dir = args.output_dir if args.output_dir is not None else (lcmm_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    assigns = _read_required_csv(lcmm_dir, "patient_class_assignments.csv")
    profiles = _read_required_csv(lcmm_dir, "lcmm_class_trajectory_profiles.csv")
    _read_required_csv(lcmm_dir, "lcmm_model_selection.csv")  # existence check

    print("[1/4] Figure 1 class trajectories...")
    figure1_trajectories(profiles, assigns, output_dir, args.title_prefix)
    print("[2/4] Figure 2 posterior certainty...")
    figure2_posterior_certainty(assigns, output_dir, args.title_prefix)
    print("[3/4] Figure 3 class composition...")
    figure3_class_composition(assigns, output_dir, args.title_prefix)
    print("[4/4] Figure 4 mortality by class...")
    figure4_mortality_by_class(assigns, output_dir, args.title_prefix)
    print(f"Done. LCMM figure pack written to: {output_dir}")


if __name__ == "__main__":
    main()
