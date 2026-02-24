#!/usr/bin/env python3
"""
Build a pooled demographics table from site-level table1_poolable files.

Inputs:
  <base_dir>/<site>/Upload_to_Box_without_oral_<window>/table1_poolable_<window>h.csv

Outputs:
  - table_demographics_pooled_<window>h.csv (grant-ready wide table)
  - table_demographics_pooled_<window>h.md (markdown rendering)
  - table_demographics_pooled_<window>h_long.csv (numeric companion)
  - table_demographics_pooled_<window>h_note.txt (footnote text)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_ORDER = ["Overall", "Survivor", "Non-Survivor"]
SEX_ORDER = ["male", "female", "unknown"]
RACE_ORDER = [
    "white",
    "black_or_african_american",
    "asian",
    "american_indian_or_alaska_native",
    "native_hawaiian_or_other_pacific_islander",
    "other",
    "unknown",
]
ETHNICITY_ORDER = ["non-hispanic", "hispanic", "other", "unknown"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pooled demographics table from site-level table1_poolable files.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/Users/davidbeiser/Library/CloudStorage/Box-Box/CLIF/Projects/AHA-OHCA"),
        help="Directory containing per-site Upload_to_Box_without_oral_<window> folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_dir/preliminary_figures"),
        help="Output directory for pooled demographics tables.",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=72,
        choices=[24, 72],
        help="Window (hours) to pool.",
    )
    return parser.parse_args()


def title_slug(slug: str) -> str:
    return slug.replace("_", " ").title()


def ordered_union(slugs: list[str], preferred: list[str]) -> list[str]:
    out: list[str] = []
    for s in preferred:
        if s in slugs:
            out.append(s)
    for s in sorted(slugs):
        if s not in out:
            out.append(s)
    return out


def parse_category_slugs(df: pd.DataFrame, prefix: str) -> list[str]:
    patt = re.compile(rf"^{re.escape(prefix)}_(.+)_n$")
    slugs: set[str] = set()
    for v in df["variable"].dropna().astype(str):
        m = patt.match(v)
        if m:
            slugs.add(m.group(1))
    return sorted(slugs)


def fmt_count_pct(count: float, denom: float) -> str:
    if not np.isfinite(denom) or denom <= 0:
        return "NA"
    return f"{int(round(count)):,} ({(count / denom * 100):.1f}%)"


def pooled_mean_sd(rows: pd.DataFrame, mean_var: str, sd_var: str) -> tuple[float, float, int]:
    m = rows.loc[rows["variable"] == mean_var, ["site", "value", "n"]].rename(columns={"value": "mean", "n": "n_mean"})
    s = rows.loc[rows["variable"] == sd_var, ["site", "value", "n"]].rename(columns={"value": "sd", "n": "n_sd"})
    z = m.merge(s, on="site", how="inner")
    z["n"] = pd.to_numeric(z["n_mean"], errors="coerce")
    z["mean"] = pd.to_numeric(z["mean"], errors="coerce")
    z["sd"] = pd.to_numeric(z["sd"], errors="coerce")
    z = z[(z["n"] > 0) & z["mean"].notna()]
    if z.empty:
        return float("nan"), float("nan"), 0

    n = z["n"].astype(float)
    mean = z["mean"].astype(float)
    sd = z["sd"].fillna(0).astype(float)
    n_total = float(n.sum())
    pooled_mean = float((mean * n).sum() / n_total)
    if n_total <= 1:
        return pooled_mean, float("nan"), int(round(n_total))

    ss_within = float(((n - 1) * (sd**2)).sum())
    ss_between = float((n * (mean - pooled_mean) ** 2).sum())
    pooled_var = (ss_within + ss_between) / max(n_total - 1, 1)
    pooled_sd = float(np.sqrt(pooled_var))
    return pooled_mean, pooled_sd, int(round(n_total))


def load_table(base_dir: Path, window_hours: int) -> pd.DataFrame:
    pattern = f"*/Upload_to_Box_without_oral_{window_hours}/table1_poolable_{window_hours}h.csv"
    paths = sorted(base_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {base_dir}/{pattern}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "site" not in df.columns:
            df["site"] = p.parts[-3]
        df["site"] = df["site"].astype(str).str.strip()
        df["source_path"] = str(p)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["window_hours"] = pd.to_numeric(out["window_hours"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["n"] = pd.to_numeric(out["n"], errors="coerce")
    return out


def build_table(df: pd.DataFrame, window_hours: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    keep = (
        (df["window_hours"] == float(window_hours))
        & (df["group_type"] == "survival")
        & (df["group"].isin(GROUP_ORDER))
    )
    sub = df.loc[keep].copy()
    if sub.empty:
        raise ValueError(f"No rows for window={window_hours} and survival groups.")

    site_count = int(sub["site"].nunique())
    sex_slugs = parse_category_slugs(sub, "sex")
    race_slugs = parse_category_slugs(sub, "race")
    ethnicity_slugs = parse_category_slugs(sub, "ethnicity")

    sex_slugs = ordered_union(sex_slugs, SEX_ORDER)
    race_slugs = ordered_union(race_slugs, RACE_ORDER)
    ethnicity_slugs = ordered_union(ethnicity_slugs, ETHNICITY_ORDER)

    overall = sub[sub["group"] == "Overall"].copy()
    overall_n_by_site = overall.loc[overall["variable"] == "n", ["site", "value"]].rename(columns={"value": "n"})
    overall_age_by_site = overall.loc[overall["variable"] == "age_mean", ["site", "value"]].rename(columns={"value": "age_mean"})
    overall_age_cov = overall_n_by_site.merge(overall_age_by_site, on="site", how="left")
    overall_age_cov["has_age"] = overall_age_cov["age_mean"].notna()
    overall_n_total = float(overall_n_by_site["n"].sum())
    overall_age_n = float(overall_age_cov.loc[overall_age_cov["has_age"], "n"].sum())
    overall_age_missing_n = max(overall_n_total - overall_age_n, 0.0)
    overall_age_missing_pct = (overall_age_missing_n / overall_n_total * 100.0) if overall_n_total > 0 else float("nan")
    missing_site_bits = []
    for _, r in overall_age_cov.loc[~overall_age_cov["has_age"]].sort_values("site").iterrows():
        missing_site_bits.append(f"{r['site']} (n={int(round(float(r['n']))):,})")
    if overall_age_missing_n > 0.5:
        site_txt = "; ".join(missing_site_bits) if missing_site_bits else "site-level source files"
        age_note = (
            f"Age was unavailable for {int(round(overall_age_missing_n)):,} of {int(round(overall_n_total)):,} patients "
            f"({overall_age_missing_pct:.1f}%). Age statistics are pooled among available records (n={int(round(overall_age_n)):,}). "
            f"Missing age source: {site_txt}."
        )
    else:
        age_note = f"Age was available for all {int(round(overall_n_total)):,} patients."

    display: dict[str, dict[str, str]] = {g: {} for g in GROUP_ORDER}
    long_rows: list[dict[str, object]] = []
    sex_missing_label = "Sex: Missing/Unspecified, n (%)"
    race_missing_label = "Race: Missing/Unspecified, n (%)"
    ethnicity_missing_label = "Ethnicity: Missing/Unspecified, n (%)"

    display["Overall"]["Sites, n"] = f"{site_count:,}"
    for g in GROUP_ORDER[1:]:
        display[g]["Sites, n"] = ""

    for g in GROUP_ORDER:
        gs = sub[sub["group"] == g].copy()
        n_total = float(gs.loc[gs["variable"] == "n", "value"].sum())
        display[g]["N"] = f"{int(round(n_total)):,}"
        long_rows.append({"group": g, "section": "overall", "characteristic": "N", "count": n_total, "denom": n_total, "pct": 100.0})

        age_mean, age_sd, age_n = pooled_mean_sd(gs, "age_mean", "age_sd")
        if age_n > 0 and np.isfinite(age_mean):
            sd_str = f"{age_sd:.2f}" if np.isfinite(age_sd) else "NA"
            display[g]["Age, mean ± SD"] = f"{age_mean:.2f} ± {sd_str}"
        else:
            display[g]["Age, mean ± SD"] = "NA"
        long_rows.append({"group": g, "section": "age", "characteristic": "age_mean", "count": age_mean, "denom": age_n, "pct": np.nan})
        long_rows.append({"group": g, "section": "age", "characteristic": "age_sd", "count": age_sd, "denom": age_n, "pct": np.nan})
        long_rows.append({"group": g, "section": "age", "characteristic": "age_n", "count": age_n, "denom": n_total, "pct": (age_n / n_total * 100.0) if n_total > 0 else np.nan})

        for slug in sex_slugs:
            var = f"sex_{slug}_n"
            count = float(gs.loc[gs["variable"] == var, "value"].sum())
            label = f"Sex: {title_slug(slug)}, n (%)"
            display[g][label] = fmt_count_pct(count, n_total)
            long_rows.append({"group": g, "section": "sex", "characteristic": slug, "count": count, "denom": n_total, "pct": (count / n_total * 100.0) if n_total > 0 else np.nan})
        sex_counted = float(sum(float(gs.loc[gs["variable"] == f"sex_{slug}_n", "value"].sum()) for slug in sex_slugs))
        sex_missing = max(n_total - sex_counted, 0.0)
        if sex_missing > 0.5:
            display[g][sex_missing_label] = fmt_count_pct(sex_missing, n_total)
            long_rows.append({"group": g, "section": "sex", "characteristic": "missing_unspecified", "count": sex_missing, "denom": n_total, "pct": (sex_missing / n_total * 100.0) if n_total > 0 else np.nan})

        for slug in race_slugs:
            var = f"race_{slug}_n"
            count = float(gs.loc[gs["variable"] == var, "value"].sum())
            label = f"Race: {title_slug(slug)}, n (%)"
            display[g][label] = fmt_count_pct(count, n_total)
            long_rows.append({"group": g, "section": "race", "characteristic": slug, "count": count, "denom": n_total, "pct": (count / n_total * 100.0) if n_total > 0 else np.nan})
        race_counted = float(sum(float(gs.loc[gs["variable"] == f"race_{slug}_n", "value"].sum()) for slug in race_slugs))
        race_missing = max(n_total - race_counted, 0.0)
        if race_missing > 0.5:
            display[g][race_missing_label] = fmt_count_pct(race_missing, n_total)
            long_rows.append({"group": g, "section": "race", "characteristic": "missing_unspecified", "count": race_missing, "denom": n_total, "pct": (race_missing / n_total * 100.0) if n_total > 0 else np.nan})

        for slug in ethnicity_slugs:
            var = f"ethnicity_{slug}_n"
            count = float(gs.loc[gs["variable"] == var, "value"].sum())
            label = f"Ethnicity: {title_slug(slug)}, n (%)"
            display[g][label] = fmt_count_pct(count, n_total)
            long_rows.append({"group": g, "section": "ethnicity", "characteristic": slug, "count": count, "denom": n_total, "pct": (count / n_total * 100.0) if n_total > 0 else np.nan})
        ethnicity_counted = float(sum(float(gs.loc[gs["variable"] == f"ethnicity_{slug}_n", "value"].sum()) for slug in ethnicity_slugs))
        ethnicity_missing = max(n_total - ethnicity_counted, 0.0)
        if ethnicity_missing > 0.5:
            display[g][ethnicity_missing_label] = fmt_count_pct(ethnicity_missing, n_total)
            long_rows.append({"group": g, "section": "ethnicity", "characteristic": "missing_unspecified", "count": ethnicity_missing, "denom": n_total, "pct": (ethnicity_missing / n_total * 100.0) if n_total > 0 else np.nan})

    row_order = ["Sites, n", "N", "Age, mean ± SD"]
    row_order += [f"Sex: {title_slug(s)}, n (%)" for s in sex_slugs]
    if any(sex_missing_label in display[g] for g in GROUP_ORDER):
        row_order.append(sex_missing_label)
    row_order += [f"Race: {title_slug(s)}, n (%)" for s in race_slugs]
    if any(race_missing_label in display[g] for g in GROUP_ORDER):
        row_order.append(race_missing_label)
    row_order += [f"Ethnicity: {title_slug(s)}, n (%)" for s in ethnicity_slugs]
    if any(ethnicity_missing_label in display[g] for g in GROUP_ORDER):
        row_order.append(ethnicity_missing_label)

    table = pd.DataFrame({"Characteristic": row_order})
    for g in GROUP_ORDER:
        table[g] = [display[g].get(r, "") for r in row_order]

    long_df = pd.DataFrame(long_rows)
    return table, long_df, age_note


def write_markdown_table(table: pd.DataFrame, path: Path, note: str | None = None) -> None:
    cols = list(table.columns)
    header = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, rule]
    for _, row in table.iterrows():
        vals = [str(row[c]) if pd.notna(row[c]) else "" for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    if note:
        lines.append("")
        lines.append(f"Note: {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(args.base_dir, args.window_hours)
    table, long_df, age_note = build_table(df, args.window_hours)

    csv_path = args.output_dir / f"table_demographics_pooled_{args.window_hours}h.csv"
    md_path = args.output_dir / f"table_demographics_pooled_{args.window_hours}h.md"
    long_path = args.output_dir / f"table_demographics_pooled_{args.window_hours}h_long.csv"
    note_path = args.output_dir / f"table_demographics_pooled_{args.window_hours}h_note.txt"

    table.to_csv(csv_path, index=False)
    write_markdown_table(table, md_path, note=age_note)
    long_df.to_csv(long_path, index=False)
    note_path.write_text(age_note + "\n", encoding="utf-8")

    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")
    print(f"Saved {long_path}")
    print(f"Saved {note_path}")


if __name__ == "__main__":
    main()
