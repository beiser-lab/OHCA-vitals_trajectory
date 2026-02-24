#!/usr/bin/env python3
"""
Multivariate latent class mixed model (growth-mixture) for UCMC + MIMIC OHCA cohorts.

Model (per class k, vital v):
  y_itv = beta_kv0 + beta_kv1 * t + ... + beta_kvd * t^d + b_iv + e_itv
  b_iv ~ N(0, tau_kv^2), e_itv ~ N(0, sigma_kv^2)

This is a class-specific random-intercept trajectory model estimated via EM with
weighted updates and analytic block likelihoods.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cluster_patientlevel_ucmc_mimic import (  # noqa: E402
    CORE_VITALS,
    build_mimic_cohort,
    build_ucmc_cohort,
    filter_patient_coverage,
    load_mimic_hourly,
    load_ucmc_hourly,
)


EPS = 1e-8


@dataclass
class LCMMFit:
    n_classes: int
    degree: int
    pi: np.ndarray
    beta: np.ndarray  # [k, v, p]
    sigma2: np.ndarray  # [k, v]
    tau2: np.ndarray  # [k, v]
    responsibilities: np.ndarray  # [n, k]
    loglik: float
    n_iter: int
    converged: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit multivariate LCMM/growth-mixture on UCMC+MIMIC OHCA.")
    parser.add_argument(
        "--ucmc-dir",
        type=Path,
        default=Path("/Users/davidbeiser/Library/CloudStorage/Box-Box/RCLIF_data/CLIF_2018_24/2.1.0"),
    )
    parser.add_argument(
        "--mimic-dir",
        type=Path,
        default=Path("/Users/davidbeiser/mimic-iv-3.1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_dir/lcmm_ucmc_mimic"),
    )
    parser.add_argument(
        "--mimic-vitals-cache",
        type=Path,
        default=Path("output_dir/patient_time_clustering_full/mimic_vitals_cache_24h_full.parquet"),
    )
    parser.add_argument("--max-hours", type=int, default=24)
    parser.add_argument("--bin-hours", type=int, default=1)
    parser.add_argument("--min-hours-per-patient", type=int, default=6)
    parser.add_argument("--min-total-measurements", type=int, default=12)
    parser.add_argument("--degree", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--n-classes", type=int, default=None, help="Fixed number of latent classes.")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--n-starts", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--inner-var-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _poly_features(t: np.ndarray, degree: int) -> np.ndarray:
    return np.vstack([t**d for d in range(degree + 1)]).T


def _logsumexp_rowwise(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.max(a, axis=1, keepdims=True)
    z = np.exp(a - m)
    s = np.sum(z, axis=1, keepdims=True)
    lse = np.log(s) + m
    return lse.ravel(), z / s


def _block_sse_sume(block: dict[str, Any], beta: np.ndarray) -> tuple[float, float]:
    x_ty = block["x_ty"]
    x_tx = block["x_tx"]
    sum_y = float(block["sum_y"])
    y_ty = float(block["y_ty"])
    sse = y_ty - 2.0 * float(np.dot(beta, x_ty)) + float(beta @ x_tx @ beta)
    sume = sum_y - float(np.dot(block["x_t1"], beta))
    return max(sse, 0.0), sume


def _block_loglik(block: dict[str, Any], beta: np.ndarray, sigma2: float, tau2: float) -> float:
    m = int(block["m"])
    if m == 0:
        return 0.0
    sigma2 = max(float(sigma2), EPS)
    tau2 = max(float(tau2), 0.0)
    sse, sume = _block_sse_sume(block, beta)
    if tau2 <= 0.0:
        quad = sse / sigma2
        logdet = m * math.log(sigma2)
    else:
        quad = (sse / sigma2) - (tau2 / (sigma2 * (sigma2 + m * tau2))) * (sume**2)
        logdet = (m - 1) * math.log(sigma2) + math.log(sigma2 + m * tau2)
    return -0.5 * (m * math.log(2.0 * math.pi) + logdet + quad)


def _update_beta_gls(
    blocks: list[dict[str, np.ndarray] | None],
    weights: np.ndarray,
    beta_prev: np.ndarray,
    sigma2: float,
    tau2: float,
) -> np.ndarray:
    p = len(beta_prev)
    sxx = np.zeros((p, p), dtype=float)
    sxy = np.zeros(p, dtype=float)
    sigma2 = max(float(sigma2), EPS)
    tau2 = max(float(tau2), 0.0)
    for i, block in enumerate(blocks):
        if block is None:
            continue
        w = float(weights[i])
        if w <= 0.0:
            continue
        m = int(block["m"])
        xtx = block["x_tx"]
        xt1 = block["x_t1"]
        xty = block["x_ty"]
        sumy = float(block["sum_y"])
        a = 1.0 / sigma2
        b = (tau2 / (sigma2 * (sigma2 + m * tau2))) if tau2 > 0.0 else 0.0
        sxx += w * (a * xtx - b * np.outer(xt1, xt1))
        sxy += w * (a * xty - b * xt1 * sumy)
    ridge = 1e-6 * np.eye(p)
    try:
        beta = np.linalg.solve(sxx + ridge, sxy)
    except np.linalg.LinAlgError:
        beta = beta_prev.copy()
    return beta


def _update_variances_moments(
    blocks: list[dict[str, np.ndarray] | None],
    weights: np.ndarray,
    beta: np.ndarray,
    sigma2_prev: float,
    tau2_prev: float,
) -> tuple[float, float]:
    num_sigma = 0.0
    den_sigma = 0.0
    wsum = 0.0
    wmean_num = 0.0
    means: list[tuple[float, float, int]] = []

    mse_num = 0.0
    mse_den = 0.0

    for i, block in enumerate(blocks):
        if block is None:
            continue
        w = float(weights[i])
        if w <= 0.0:
            continue
        m = int(block["m"])
        sse, sume = _block_sse_sume(block, beta)
        mse_num += w * sse
        mse_den += w * m
        if m > 1:
            em = sume / m
            sse_w = max(sse - (sume**2) / m, 0.0)
            num_sigma += w * sse_w
            den_sigma += w * (m - 1)
            means.append((w, em, m))
            wsum += w
            wmean_num += w * em
        else:
            em = sume
            means.append((w, em, m))
            wsum += w
            wmean_num += w * em

    if den_sigma > 0:
        sigma2 = max(num_sigma / den_sigma, EPS)
    elif mse_den > 0:
        sigma2 = max(mse_num / mse_den, EPS)
    else:
        sigma2 = max(float(sigma2_prev), EPS)

    if wsum <= 0.0 or not means:
        return sigma2, max(float(tau2_prev), 0.0)

    mu = wmean_num / wsum
    var_between = sum(w * ((m - mu) ** 2) for w, m, _ in means) / wsum
    mean_inv_m = sum(w * (1.0 / max(mi, 1)) for w, _, mi in means) / wsum
    tau2 = max(0.0, var_between - sigma2 * mean_inv_m)
    return sigma2, tau2


def _initialize_params(
    n_subjects: int,
    n_classes: int,
    n_vitals: int,
    p: int,
    rng: np.random.Generator,
    global_beta: np.ndarray,
    global_sigma2: np.ndarray,
    global_tau2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = rng.integers(0, n_classes, size=n_subjects)
    resp = np.zeros((n_subjects, n_classes), dtype=float)
    resp[np.arange(n_subjects), z] = 1.0
    pi = np.clip(resp.mean(axis=0), EPS, 1.0)
    pi = pi / pi.sum()

    beta = np.zeros((n_classes, n_vitals, p), dtype=float)
    sigma2 = np.zeros((n_classes, n_vitals), dtype=float)
    tau2 = np.zeros((n_classes, n_vitals), dtype=float)
    for k in range(n_classes):
        for v in range(n_vitals):
            beta[k, v] = global_beta[v] + rng.normal(scale=0.05, size=p)
            sigma2[k, v] = max(global_sigma2[v] * rng.uniform(0.8, 1.2), EPS)
            tau2[k, v] = max(global_tau2[v] * rng.uniform(0.8, 1.2), 0.0)
    return pi, beta, sigma2, tau2


def _subject_loglik_matrix(
    subject_blocks: dict[str, Any],
    pi: np.ndarray,
    beta: np.ndarray,
    sigma2: np.ndarray,
    tau2: np.ndarray,
) -> np.ndarray:
    n = subject_blocks["n_subjects"]
    kmax = len(pi)
    out = np.zeros((n, kmax), dtype=float)
    for i in range(n):
        for k in range(kmax):
            ll = math.log(max(pi[k], EPS))
            for v_idx, vital in enumerate(CORE_VITALS):
                block = subject_blocks["blocks"][vital][i]
                if block is None:
                    continue
                ll += _block_loglik(
                    block,
                    beta[k, v_idx],
                    sigma2[k, v_idx],
                    tau2[k, v_idx],
                )
            out[i, k] = ll
    return out


def fit_lcmm(
    subject_blocks: dict[str, Any],
    n_classes: int,
    degree: int,
    n_starts: int,
    max_iter: int,
    tol: float,
    inner_var_iters: int,
    seed: int,
) -> LCMMFit:
    n = subject_blocks["n_subjects"]
    n_vitals = len(CORE_VITALS)
    p = degree + 1
    global_beta = subject_blocks["global_beta"]
    global_sigma2 = subject_blocks["global_sigma2"]
    global_tau2 = subject_blocks["global_tau2"]

    best: LCMMFit | None = None
    rng_master = np.random.default_rng(seed)

    for s in range(n_starts):
        rng = np.random.default_rng(int(rng_master.integers(1, 10_000_000)))
        pi, beta, sigma2, tau2 = _initialize_params(
            n_subjects=n,
            n_classes=n_classes,
            n_vitals=n_vitals,
            p=p,
            rng=rng,
            global_beta=global_beta,
            global_sigma2=global_sigma2,
            global_tau2=global_tau2,
        )

        prev_ll = -np.inf
        converged = False
        resp = np.zeros((n, n_classes), dtype=float)

        for it in range(1, max_iter + 1):
            ll_mat = _subject_loglik_matrix(subject_blocks, pi, beta, sigma2, tau2)
            ll_row, resp = _logsumexp_rowwise(ll_mat)
            ll = float(np.sum(ll_row))

            # M-step: mixture weights
            nk = resp.sum(axis=0)
            pi = np.clip(nk / max(n, 1), EPS, 1.0)
            pi = pi / pi.sum()

            # M-step: class-vital parameters
            for k in range(n_classes):
                w = resp[:, k]
                for v_idx, vital in enumerate(CORE_VITALS):
                    blocks = subject_blocks["blocks"][vital]
                    b = beta[k, v_idx].copy()
                    s2 = float(sigma2[k, v_idx])
                    t2 = float(tau2[k, v_idx])
                    for _ in range(max(1, inner_var_iters)):
                        b = _update_beta_gls(blocks, w, b, s2, t2)
                        s2, t2 = _update_variances_moments(blocks, w, b, s2, t2)
                    beta[k, v_idx] = b
                    sigma2[k, v_idx] = max(s2, EPS)
                    tau2[k, v_idx] = max(t2, 0.0)

            rel = abs(ll - prev_ll) / max(1.0, abs(prev_ll)) if np.isfinite(prev_ll) else np.inf
            if np.isfinite(prev_ll) and rel < tol:
                converged = True
                prev_ll = ll
                break
            prev_ll = ll

        fit = LCMMFit(
            n_classes=n_classes,
            degree=degree,
            pi=pi.copy(),
            beta=beta.copy(),
            sigma2=sigma2.copy(),
            tau2=tau2.copy(),
            responsibilities=resp.copy(),
            loglik=float(prev_ll),
            n_iter=it,
            converged=converged,
        )
        if best is None or fit.loglik > best.loglik:
            best = fit

    if best is None:
        raise RuntimeError("LCMM fitting failed.")
    return best


def _n_params(n_classes: int, n_vitals: int, degree: int) -> int:
    p = degree + 1
    per_class = n_vitals * (p + 2)  # betas + sigma2 + tau2
    return (n_classes - 1) + n_classes * per_class


def _build_subject_blocks(hourly: pd.DataFrame, max_hours: int, degree: int) -> dict[str, Any]:
    patients = hourly["patient_key"].drop_duplicates().astype(str).tolist()
    p_to_idx = {p: i for i, p in enumerate(patients)}

    blocks: dict[str, list[dict[str, np.ndarray] | None]] = {v: [None] * len(patients) for v in CORE_VITALS}
    global_beta = np.zeros((len(CORE_VITALS), degree + 1), dtype=float)
    global_sigma2 = np.ones(len(CORE_VITALS), dtype=float)
    global_tau2 = np.full(len(CORE_VITALS), 0.1, dtype=float)

    for v_idx, vital in enumerate(CORE_VITALS):
        sub = hourly[["patient_key", "hour", vital]].dropna().copy()
        if sub.empty:
            continue
        sub["patient_key"] = sub["patient_key"].astype(str)
        sub["t"] = sub["hour"].astype(float) / float(max(max_hours - 1, 1))
        sub = sub.sort_values(["patient_key", "hour"])

        # Global initialization by pooled OLS
        X_all = _poly_features(sub["t"].to_numpy(dtype=float), degree)
        y_all = sub[vital].to_numpy(dtype=float)
        try:
            b = np.linalg.lstsq(X_all, y_all, rcond=None)[0]
        except np.linalg.LinAlgError:
            b = np.zeros(degree + 1, dtype=float)
        global_beta[v_idx] = b

        # Rough pooled variance initialization
        e_all = y_all - X_all @ b
        global_sigma2[v_idx] = max(float(np.nanvar(e_all)), EPS)
        means = sub.groupby("patient_key")[vital].mean()
        global_tau2[v_idx] = max(float(np.nanvar(means.to_numpy(dtype=float))), 0.0)

        for pk, g in sub.groupby("patient_key", sort=False):
            i = p_to_idx.get(str(pk))
            if i is None:
                continue
            t = g["t"].to_numpy(dtype=float)
            y = g[vital].to_numpy(dtype=float)
            X = _poly_features(t, degree)
            blocks[vital][i] = {
                "m": int(len(y)),
                "x_tx": X.T @ X,
                "x_t1": np.sum(X, axis=0),
                "x_ty": X.T @ y,
                "sum_y": float(np.sum(y)),
                "y_ty": float(np.dot(y, y)),
            }

    return {
        "patients": patients,
        "n_subjects": len(patients),
        "blocks": blocks,
        "global_beta": global_beta,
        "global_sigma2": global_sigma2,
        "global_tau2": global_tau2,
    }


def _predict_profiles(fit: LCMMFit, max_hours: int) -> pd.DataFrame:
    rows = []
    tvec = np.arange(0, int(max_hours), dtype=float) / float(max(max_hours - 1, 1))
    for k in range(fit.n_classes):
        for h_idx, t in enumerate(tvec):
            row = {"class": k + 1, "hour": int(h_idx)}
            for v_idx, vital in enumerate(CORE_VITALS):
                b = fit.beta[k, v_idx]
                x = np.array([t**d for d in range(fit.degree + 1)], dtype=float)
                row[vital] = float(np.dot(x, b))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["class", "hour"]).reset_index(drop=True)


def _write_outputs(
    output_dir: Path,
    fit: LCMMFit,
    model_rows: list[dict[str, Any]],
    selected_k: int,
    hourly: pd.DataFrame,
    subject_blocks: dict[str, Any],
    max_hours: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_df = pd.DataFrame(model_rows).sort_values("k")
    model_df["selected"] = model_df["k"].eq(selected_k)
    model_df.to_csv(output_dir / "lcmm_model_selection.csv", index=False)

    patients = pd.DataFrame({"patient_key": subject_blocks["patients"]})
    meta = (
        hourly.groupby("patient_key", as_index=False)
        .agg(
            site=("site", "first"),
            patient_id=("patient_id", "first"),
            hospitalization_id=("hospitalization_id", "first"),
            survival_status=("survival_status", "first"),
        )
    )
    patients = patients.merge(meta, on="patient_key", how="left")

    resp = fit.responsibilities
    resp_df = pd.DataFrame(resp, columns=[f"class_{k+1}_prob" for k in range(fit.n_classes)])
    assign = patients.copy()
    assign["class"] = (np.argmax(resp, axis=1) + 1).astype(int)
    assign["max_posterior_prob"] = np.max(resp, axis=1)
    assign = pd.concat([assign, resp_df], axis=1)
    assign.to_csv(output_dir / "patient_class_assignments.csv", index=False)

    params_rows = []
    for k in range(fit.n_classes):
        for v_idx, vital in enumerate(CORE_VITALS):
            row = {
                "class": k + 1,
                "vital": vital,
                "sigma2": float(fit.sigma2[k, v_idx]),
                "tau2": float(fit.tau2[k, v_idx]),
            }
            for d in range(fit.degree + 1):
                row[f"beta_{d}"] = float(fit.beta[k, v_idx, d])
            params_rows.append(row)
    pd.DataFrame(params_rows).to_csv(output_dir / "lcmm_class_parameters.csv", index=False)

    profiles = _predict_profiles(fit, max_hours=max_hours)
    profiles.to_csv(output_dir / "lcmm_class_trajectory_profiles.csv", index=False)

    size_site = (
        assign.groupby(["class", "site", "survival_status"], as_index=False)
        .size()
        .rename(columns={"size": "n_patients"})
        .sort_values(["class", "site", "survival_status"])
    )
    size_site.to_csv(output_dir / "lcmm_class_size_by_site_survival.csv", index=False)

    summary = [
        "model: multivariate latent class mixed model (random-intercept growth-mixture)",
        f"selected_k: {selected_k}",
        f"degree: {fit.degree}",
        f"loglik: {fit.loglik:.6f}",
        f"n_subjects_clustered: {len(subject_blocks['patients'])}",
        f"n_vitals: {len(CORE_VITALS)}",
        f"max_hours: {max_hours}",
        f"converged: {fit.converged}",
        f"n_iter: {fit.n_iter}",
    ]
    (output_dir / "run_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Building UCMC OHCA ICU cohort...")
    ucmc_cohort = build_ucmc_cohort(args.ucmc_dir)
    print(f"      UCMC cohort size: {ucmc_cohort['hospitalization_id'].nunique():,}")

    print("[2/7] Building MIMIC OHCA ICU cohort...")
    mimic_cohort = build_mimic_cohort(args.mimic_dir)
    print(f"      MIMIC cohort size: {mimic_cohort['hospitalization_id'].nunique():,}")

    print("[3/7] Loading hourly UCMC vitals...")
    ucmc_hourly = load_ucmc_hourly(args.ucmc_dir, ucmc_cohort, max_hours=args.max_hours, bin_hours=args.bin_hours)
    print(f"      UCMC hourly rows: {len(ucmc_hourly):,}")

    print("[4/7] Loading hourly MIMIC vitals (cache-enabled)...")
    mimic_hourly = load_mimic_hourly(
        args.mimic_dir,
        mimic_cohort,
        max_hours=args.max_hours,
        bin_hours=args.bin_hours,
        chunk_size=2_000_000,
        max_mimic_chunks=None,
        mimic_vitals_cache=args.mimic_vitals_cache,
    )
    print(f"      MIMIC hourly rows: {len(mimic_hourly):,}")

    combined = pd.concat([ucmc_hourly, mimic_hourly], ignore_index=True)
    if combined.empty:
        raise RuntimeError("No hourly rows available for LCMM.")

    print("[5/7] Applying patient coverage filters...")
    combined_filt, coverage = filter_patient_coverage(
        combined,
        min_hours_per_patient=args.min_hours_per_patient,
        min_total_measurements=args.min_total_measurements,
    )
    coverage.to_csv(args.output_dir / "patient_coverage.csv", index=False)
    if combined_filt.empty:
        raise RuntimeError("No patients left after coverage filters.")
    print(f"      Patients before filter: {combined['patient_key'].nunique():,}")
    print(f"      Patients after filter:  {combined_filt['patient_key'].nunique():,}")

    print("[6/7] Building longitudinal blocks...")
    subject_blocks = _build_subject_blocks(combined_filt, max_hours=args.max_hours, degree=args.degree)
    n_subjects = subject_blocks["n_subjects"]
    if n_subjects < 20:
        raise RuntimeError(f"Too few subjects for LCMM: {n_subjects}")
    print(f"      Subjects for LCMM: {n_subjects:,}")

    print("[7/7] Fitting LCMM across class counts...")
    if args.n_classes is not None:
        k_values = [int(args.n_classes)]
    else:
        k_values = list(range(int(args.k_min), int(args.k_max) + 1))
    k_values = [k for k in k_values if 2 <= k < n_subjects]
    if not k_values:
        raise ValueError("No valid class counts to fit.")

    model_rows: list[dict[str, Any]] = []
    fits: dict[int, LCMMFit] = {}
    for k in k_values:
        print(f"      Fitting k={k} ...")
        fit = fit_lcmm(
            subject_blocks=subject_blocks,
            n_classes=k,
            degree=args.degree,
            n_starts=args.n_starts,
            max_iter=args.max_iter,
            tol=args.tol,
            inner_var_iters=args.inner_var_iters,
            seed=args.seed + k,
        )
        fits[k] = fit
        p = _n_params(k, len(CORE_VITALS), args.degree)
        bic = -2.0 * fit.loglik + p * math.log(max(n_subjects, 1))
        aic = -2.0 * fit.loglik + 2.0 * p
        model_rows.append(
            {
                "k": k,
                "loglik": fit.loglik,
                "aic": aic,
                "bic": bic,
                "n_params": p,
                "converged": fit.converged,
                "n_iter": fit.n_iter,
            }
        )
        print(f"        loglik={fit.loglik:.2f}, BIC={bic:.2f}, converged={fit.converged}, iter={fit.n_iter}")

    model_df = pd.DataFrame(model_rows)
    best_idx = model_df["bic"].idxmin()
    best_k = int(model_df.loc[best_idx, "k"])
    best_fit = fits[best_k]
    print(f"      Selected k by BIC: {best_k}")

    _write_outputs(
        output_dir=args.output_dir,
        fit=best_fit,
        model_rows=model_rows,
        selected_k=best_k,
        hourly=combined_filt,
        subject_blocks=subject_blocks,
        max_hours=args.max_hours,
    )
    print(f"Done. LCMM outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
