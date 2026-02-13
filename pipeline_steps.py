"""
pipeline_steps.py — All pipeline step functions.
Import and call from notebook.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_helpers import (
    read_table, build_icd_filter, map_description, vitals_filter_sql,
    block_plot_params, hourly_plot_params,
    VITAL_SIGNS, TRAJ_COLORS, SURV_COLORS, TRAJ_ORDER,
    VITALS_INFO, VITALS_HOURLY_INFO,
)


# =============================================================
# STEP 2: BUILD COHORT
# =============================================================
def step2_build_cohort(config, con, log):
    """Build full cardiac arrest cohort from hospital_diagnosis. Window-independent."""
    log.info("\n" + "=" * 60)
    log.info("  STEP 2: BUILD COHORT FROM clif_hospital_diagnosis")
    log.info("=" * 60)

    hosp_dx_table = read_table(config, "clif_hospital_diagnosis")
    hosp_table = read_table(config, "clif_hospitalization")
    icd_filter = build_icd_filter("diagnosis_code")

    cohort_v2 = con.execute(f"""
        SELECT DISTINCT
            h.patient_id, d.hospitalization_id, d.diagnosis_code,
            d.diagnosis_primary, d.poa_present,
            CASE WHEN d.poa_present = 1 THEN 'OHCA'
                 WHEN d.poa_present = 0 THEN 'IHCA' ELSE 'Unknown' END AS arrest_type,
            hc.discharge_category,
            CASE WHEN hc.discharge_category = 'Expired' THEN 'Non-Survivor'
                 ELSE 'Survivor' END AS survival_status
        FROM {hosp_dx_table} d
        INNER JOIN {hosp_table} hc ON d.hospitalization_id = hc.hospitalization_id
        INNER JOIN (SELECT DISTINCT patient_id, hospitalization_id FROM {hosp_table}) h
            ON d.hospitalization_id = h.hospitalization_id
        WHERE {icd_filter}
    """).fetchdf()
    cohort_v2["icd_description"] = cohort_v2["diagnosis_code"].apply(map_description)

    enc_per_pt = cohort_v2.groupby("patient_id")["hospitalization_id"].nunique()
    log.info(f"\n  FULL COHORT: {cohort_v2['patient_id'].nunique():,} patients, "
             f"{cohort_v2['hospitalization_id'].nunique():,} encounters")
    log.info(f"  Enc/patient: mean={enc_per_pt.mean():.2f}, median={enc_per_pt.median():.1f}")

    log.info(f"\n  {'Type':<8s} {'Status':<15s} {'Patients':>10s} {'Encounters':>12s}")
    log.info(f"  {'-'*8} {'-'*15} {'-'*10} {'-'*12}")
    for atype in ["OHCA", "IHCA"]:
        for status in ["Survivor", "Non-Survivor"]:
            sub = cohort_v2[(cohort_v2["arrest_type"] == atype) & (cohort_v2["survival_status"] == status)]
            if len(sub) > 0:
                log.info(f"  {atype:<8s} {status:<15s} {sub['patient_id'].nunique():>10,} {sub['hospitalization_id'].nunique():>12,}")
        sub = cohort_v2[cohort_v2["arrest_type"] == atype]
        t = sub["hospitalization_id"].nunique()
        d = sub[sub["survival_status"] == "Non-Survivor"]["hospitalization_id"].nunique()
        log.info(f"  {atype} mortality: {d/t*100:.1f}% ({d:,}/{t:,})")

    log.info(f"\n  ICD code breakdown:")
    icd_counts = cohort_v2.groupby(["diagnosis_code", "icd_description"])["hospitalization_id"].nunique().reset_index()
    icd_counts.columns = ["code", "description", "encounters"]
    for _, r in icd_counts.sort_values("encounters", ascending=False).iterrows():
        log.info(f"    {r['code']:<10s} {r['encounters']:>6,}  {r['description']}")

    cohort_v2.to_csv(config["intermediate_dir"] / "cohort_v2_hospital_diagnosis.csv", index=False)
    cohort_v2.to_parquet(config["intermediate_dir"] / "cohort_v2_hospital_diagnosis.parquet", index=False)
    con.register("cohort_v2_df", cohort_v2)
    log.info(f"  [OK] Saved cohort_v2")
    log.info("=" * 60)
    return cohort_v2


# =============================================================
# STEP 3: FILTER OHCA → FIRST ENCOUNTER → ICU
# =============================================================
def step3_filter_ohca_icu(config, con, log, cohort_v2):
    """Filter to OHCA, first encounter, ICU-admitted. Window-independent."""
    log.info("\n" + "=" * 60)
    log.info("  STEP 3: FILTER OHCA → FIRST ENCOUNTER → ICU ADMITTED")
    log.info("=" * 60)

    hosp_table = read_table(config, "clif_hospitalization")
    adt_table = read_table(config, "clif_adt")

    ohca_all = cohort_v2[cohort_v2["arrest_type"] == "OHCA"].copy()
    log.info(f"\n  3a. OHCA: {ohca_all['patient_id'].nunique():,} patients, {ohca_all['hospitalization_id'].nunique():,} encounters")

    con.register("ohca_all_df", ohca_all)
    cohort_ohca_first = con.execute(f"""
        WITH ranked AS (
            SELECT c.*, h.admission_dttm,
                ROW_NUMBER() OVER (PARTITION BY c.patient_id ORDER BY h.admission_dttm ASC) AS rn
            FROM ohca_all_df c
            INNER JOIN {hosp_table} h ON c.hospitalization_id = h.hospitalization_id
            WHERE c.arrest_type = 'OHCA'
        )
        SELECT * FROM ranked WHERE rn = 1
    """).fetchdf()
    log.info(f"  3b. First encounter: {cohort_ohca_first['patient_id'].nunique():,} patients, "
             f"{cohort_ohca_first['hospitalization_id'].nunique():,} encounters")

    con.register("ohca_first_df", cohort_ohca_first)

    # Admission path breakdown
    admission_paths = con.execute(f"""
        WITH patient_locations AS (
            SELECT a.hospitalization_id,
                MAX(CASE WHEN a.location_category='ed' THEN 1 ELSE 0 END) AS has_ed,
                MAX(CASE WHEN a.location_category='icu' THEN 1 ELSE 0 END) AS has_icu,
                MAX(CASE WHEN a.location_category='ward' THEN 1 ELSE 0 END) AS has_ward,
                MAX(CASE WHEN a.location_category='stepdown' THEN 1 ELSE 0 END) AS has_stepdown
            FROM {adt_table} a
            WHERE a.hospitalization_id IN (SELECT hospitalization_id FROM ohca_first_df)
            GROUP BY a.hospitalization_id
        )
        SELECT CASE
                WHEN has_ed=1 AND has_icu=0 AND has_ward=0 AND has_stepdown=0 THEN 'ED only'
                WHEN has_icu=1 THEN 'ICU admitted'
                WHEN has_ward=1 AND has_icu=0 THEN 'Ward only (no ICU)'
                ELSE 'Other'
            END AS admission_path, COUNT(*) AS encounters
        FROM patient_locations GROUP BY admission_path ORDER BY encounters DESC
    """).fetchdf()
    log.info(f"\n  3c. Admission path breakdown:")
    for _, r in admission_paths.iterrows():
        pct = r["encounters"] / cohort_ohca_first["hospitalization_id"].nunique() * 100
        log.info(f"      {r['admission_path']:<25s}: {r['encounters']:>6,} ({pct:.1f}%)")

    # Mortality by path
    mort_by_path = con.execute(f"""
        WITH patient_locations AS (
            SELECT a.hospitalization_id,
                MAX(CASE WHEN a.location_category='icu' THEN 1 ELSE 0 END) AS has_icu,
                MAX(CASE WHEN a.location_category='ward' THEN 1 ELSE 0 END) AS has_ward,
                MAX(CASE WHEN a.location_category='ed' THEN 1 ELSE 0 END) AS has_ed,
                MAX(CASE WHEN a.location_category='stepdown' THEN 1 ELSE 0 END) AS has_stepdown
            FROM {adt_table} a
            WHERE a.hospitalization_id IN (SELECT hospitalization_id FROM ohca_first_df)
            GROUP BY a.hospitalization_id
        ),
        paths AS (
            SELECT hospitalization_id,
                CASE WHEN has_ed=1 AND has_icu=0 AND has_ward=0 AND has_stepdown=0 THEN 'ED only'
                     WHEN has_icu=1 THEN 'ICU admitted'
                     WHEN has_ward=1 AND has_icu=0 THEN 'Ward only'
                     ELSE 'Other' END AS admission_path
            FROM patient_locations
        )
        SELECT p.admission_path, c.survival_status, COUNT(*) AS n
        FROM paths p INNER JOIN ohca_first_df c ON p.hospitalization_id=c.hospitalization_id
        GROUP BY p.admission_path, c.survival_status ORDER BY p.admission_path
    """).fetchdf()
    log.info(f"\n  Mortality by admission path:")
    log.info(f"  {'Path':<25s} {'Survivor':>10s} {'Non-Surv':>10s} {'Total':>8s} {'Mort%':>8s}")
    log.info(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for path in mort_by_path["admission_path"].unique():
        sub = mort_by_path[mort_by_path["admission_path"] == path]
        s = sub[sub["survival_status"] == "Survivor"]["n"].sum()
        d = sub[sub["survival_status"] == "Non-Survivor"]["n"].sum()
        t = s + d
        log.info(f"  {path:<25s} {s:>10,} {d:>10,} {t:>8,} {d/t*100 if t else 0:>7.1f}%")

    # Filter to ICU
    icu_ids = con.execute(f"""
        SELECT DISTINCT hospitalization_id FROM {adt_table}
        WHERE location_category='icu'
            AND hospitalization_id IN (SELECT hospitalization_id FROM ohca_first_df)
    """).fetchdf()

    cohort_ohca_icu = cohort_ohca_first[
        cohort_ohca_first["hospitalization_id"].isin(icu_ids["hospitalization_id"])
    ].copy()

    removed = cohort_ohca_first[~cohort_ohca_first["hospitalization_id"].isin(icu_ids["hospitalization_id"])]
    log.info(f"\n  ICU-admitted only:")
    log.info(f"      Patients  : {cohort_ohca_icu['patient_id'].nunique():,}")
    log.info(f"      Encounters: {cohort_ohca_icu['hospitalization_id'].nunique():,}")
    log.info(f"      Removed   : {removed['hospitalization_id'].nunique():,}")

    s = cohort_ohca_icu[cohort_ohca_icu["survival_status"] == "Survivor"]["patient_id"].nunique()
    ns = cohort_ohca_icu[cohort_ohca_icu["survival_status"] == "Non-Survivor"]["patient_id"].nunique()
    total = cohort_ohca_icu["patient_id"].nunique()
    log.info(f"\n  FINAL OHCA COHORT: Survivor={s:,}, Non-Survivor={ns:,}, Mortality={ns/total*100:.1f}%")

    # ICU type breakdown
    icu_types = con.execute(f"""
        SELECT a.location_type, COUNT(DISTINCT a.hospitalization_id) AS encounters
        FROM {adt_table} a
        WHERE a.hospitalization_id IN (SELECT hospitalization_id FROM ohca_first_df)
            AND a.location_category='icu'
        GROUP BY a.location_type ORDER BY encounters DESC
    """).fetchdf()
    log.info(f"\n  ICU type breakdown:")
    for _, r in icu_types.iterrows():
        log.info(f"      {str(r['location_type']):25s} : {r['encounters']:>6,}")

    dates = con.execute(f"""
        SELECT MIN(h.admission_dttm), MAX(h.admission_dttm)
        FROM ohca_first_df c INNER JOIN {hosp_table} h ON c.hospitalization_id=h.hospitalization_id
        WHERE c.hospitalization_id IN (SELECT hospitalization_id FROM icu_ids)
    """).fetchone()
    con.register("icu_ids", icu_ids)
    log.info(f"\n  Date range: {dates[0]} to {dates[1]}")

    cohort_ohca_icu.to_csv(config["intermediate_dir"] / "cohort_ohca_icu.csv", index=False)
    cohort_ohca_icu.to_parquet(config["intermediate_dir"] / "cohort_ohca_icu.parquet", index=False)
    con.register("ohca_icu_df", cohort_ohca_icu)
    log.info(f"  [OK] Saved cohort_ohca_icu")
    log.info("=" * 60)
    return cohort_ohca_icu


# =============================================================
# STEP 4a: RAW VITALS + TEMPERATURE
# =============================================================
def step4a_extract_vitals(config, con, log):
    """Extract raw vitals and temperature for the window. Window-dependent."""
    WH = config["window_hours"]
    log.info("\n" + "=" * 60)
    log.info(f"  STEP 4a: RAW VITALS (0-{WH}h)")
    log.info("=" * 60)

    vitals_table = read_table(config, "clif_vitals")
    vf = vitals_filter_sql()

    # --- All vitals ---
    raw_vitals = con.execute(f"""
        WITH first_vital AS (
            SELECT v.hospitalization_id, MIN(v.recorded_dttm) AS first_vital_dttm
            FROM {vitals_table} v
            WHERE v.hospitalization_id IN (SELECT hospitalization_id FROM ohca_icu_df)
                AND v.vital_category IN ({vf})
            GROUP BY v.hospitalization_id
        ),
        cohort AS (
            SELECT DISTINCT c.patient_id, c.hospitalization_id, c.survival_status, c.arrest_type,
                fv.first_vital_dttm AS time_zero
            FROM ohca_icu_df c
            INNER JOIN first_vital fv ON c.hospitalization_id=fv.hospitalization_id
            WHERE fv.first_vital_dttm IS NOT NULL
        )
        SELECT c.patient_id, c.hospitalization_id, c.survival_status, c.arrest_type,
            v.vital_category, CAST(v.vital_value AS DOUBLE) AS vital_value,
            v.recorded_dttm, c.time_zero,
            ROUND(EXTRACT(EPOCH FROM (v.recorded_dttm - c.time_zero))/3600, 2) AS hours_from_first_vital,
            v.meas_site_name
        FROM cohort c
        INNER JOIN {vitals_table} v ON c.hospitalization_id=v.hospitalization_id
        WHERE v.vital_category IN ({vf})
            AND v.recorded_dttm >= c.time_zero
            AND v.recorded_dttm < c.time_zero + INTERVAL {WH} HOUR
        ORDER BY c.hospitalization_id, v.recorded_dttm
    """).fetchdf()

    log.info(f"\n  All vitals: {len(raw_vitals):,} rows, {raw_vitals['hospitalization_id'].nunique():,} encounters")
    for vital in VITAL_SIGNS:
        sub = raw_vitals[raw_vitals["vital_category"] == vital]
        log.info(f"    {vital:15s}: {len(sub):>10,} meas, {sub['hospitalization_id'].nunique():>6,} enc")
    for status in ["Survivor", "Non-Survivor"]:
        sub = raw_vitals[raw_vitals["survival_status"] == status]
        log.info(f"    {status:15s}: {sub['hospitalization_id'].nunique():>6,} enc, {len(sub):>10,} meas")

    raw_vitals.to_parquet(config["intermediate_dir"] / f"raw_vitals_ohca_icu_{WH}h.parquet", index=False)
    log.info(f"  [OK] Saved raw_vitals_{WH}h")

    # --- Temperature ---
    log.info(f"\n  --- Temperature (time 0 = first temp) ---")
    raw_temp = con.execute(f"""
        WITH first_temp AS (
            SELECT v.hospitalization_id, MIN(v.recorded_dttm) AS first_temp_dttm
            FROM {vitals_table} v
            WHERE v.hospitalization_id IN (SELECT hospitalization_id FROM ohca_icu_df)
                AND v.vital_category='temp_c' AND CAST(v.vital_value AS DOUBLE) BETWEEN 32 AND 44
            GROUP BY v.hospitalization_id
        ),
        first_vital AS (
            SELECT v.hospitalization_id, MIN(v.recorded_dttm) AS first_vital_dttm
            FROM {vitals_table} v
            WHERE v.hospitalization_id IN (SELECT hospitalization_id FROM ohca_icu_df)
                AND v.vital_category IN ({vf})
            GROUP BY v.hospitalization_id
        ),
        cohort AS (
            SELECT DISTINCT c.patient_id, c.hospitalization_id, c.survival_status, c.arrest_type,
                ft.first_temp_dttm AS time_zero_temp, fv.first_vital_dttm AS time_zero_vital
            FROM ohca_icu_df c
            INNER JOIN first_temp ft ON c.hospitalization_id=ft.hospitalization_id
            LEFT JOIN first_vital fv ON c.hospitalization_id=fv.hospitalization_id
        )
        SELECT c.patient_id, c.hospitalization_id, c.survival_status, c.arrest_type,
            CAST(v.vital_value AS DOUBLE) AS temperature,
            v.recorded_dttm, c.time_zero_temp, c.time_zero_vital,
            ROUND(EXTRACT(EPOCH FROM (v.recorded_dttm - c.time_zero_temp))/3600, 2) AS hours_from_first_temp,
            ROUND(EXTRACT(EPOCH FROM (v.recorded_dttm - c.time_zero_vital))/3600, 2) AS hours_from_first_vital,
            v.meas_site_name
        FROM cohort c
        INNER JOIN {vitals_table} v ON c.hospitalization_id=v.hospitalization_id
        WHERE v.vital_category='temp_c'
            AND CAST(v.vital_value AS DOUBLE) BETWEEN 32 AND 44
            AND v.recorded_dttm >= c.time_zero_temp
            AND v.recorded_dttm < c.time_zero_temp + INTERVAL {WH} HOUR
        ORDER BY c.hospitalization_id, v.recorded_dttm
    """).fetchdf()

    log.info(f"  Temp: {len(raw_temp):,} rows, {raw_temp['hospitalization_id'].nunique():,} encounters")
    log.info(f"\n  Temperature by site:")
    log.info(f"  {'Site':<20s} {'Enc':>12s} {'Meas':>14s}")
    log.info(f"  {'-'*20} {'-'*12} {'-'*14}")
    for site, grp in raw_temp.groupby("meas_site_name"):
        log.info(f"  {str(site):<20s} {grp['hospitalization_id'].nunique():>12,} {len(grp):>14,}")
    for status in ["Survivor", "Non-Survivor"]:
        sub = raw_temp[raw_temp["survival_status"] == status]
        log.info(f"    {status:15s}: {sub['hospitalization_id'].nunique():>6,} enc, {len(sub):>10,} meas")

    raw_temp.to_parquet(config["intermediate_dir"] / f"raw_temp_ohca_icu_{WH}h.parquet", index=False)
    log.info(f"  [OK] Saved raw_temp_{WH}h")
    log.info("=" * 60)
    return raw_vitals, raw_temp


# =============================================================
# STEP 4b: BLOCK VITALS
# =============================================================
def step4b_block_vitals(config, con, log):
    """Aggregate vitals into N-hour blocks. Window-dependent."""
    WH = config["window_hours"]
    BS = config["block_size"]
    NB = config["n_blocks"]
    log.info("\n" + "=" * 60)
    log.info(f"  STEP 4b: {BS}-HOUR BLOCK VITALS (0-{WH}h)")
    log.info("=" * 60)

    vitals_table = read_table(config, "clif_vitals")
    vf = vitals_filter_sql()

    block_vitals = con.execute(f"""
        WITH first_vital AS (
            SELECT v.hospitalization_id, MIN(v.recorded_dttm) AS first_vital_dttm
            FROM {vitals_table} v
            WHERE v.hospitalization_id IN (SELECT hospitalization_id FROM ohca_icu_df)
                AND v.vital_category IN ({vf})
            GROUP BY v.hospitalization_id
        ),
        cohort AS (
            SELECT DISTINCT c.patient_id, c.hospitalization_id, c.survival_status,
                c.arrest_type, c.poa_present, fv.first_vital_dttm AS time_zero
            FROM ohca_icu_df c
            INNER JOIN first_vital fv ON c.hospitalization_id=fv.hospitalization_id
            WHERE fv.first_vital_dttm IS NOT NULL
        ),
        blocks AS (SELECT unnest(generate_series(0, {NB - 1})) AS block_num),
        vitals_binned AS (
            SELECT c.patient_id, c.hospitalization_id, c.survival_status,
                c.arrest_type, c.poa_present, b.block_num, b.block_num*{BS} AS block_start_hr,
                v.vital_category, CAST(v.vital_value AS DOUBLE) AS vital_value
            FROM cohort c CROSS JOIN blocks b
            INNER JOIN {vitals_table} v ON c.hospitalization_id=v.hospitalization_id
                AND v.recorded_dttm >= c.time_zero + INTERVAL (b.block_num*{BS}) HOUR
                AND v.recorded_dttm < c.time_zero + INTERVAL ((b.block_num+1)*{BS}) HOUR
            WHERE v.vital_category IN ({vf})
        )
        SELECT patient_id, hospitalization_id, survival_status, arrest_type, poa_present,
            block_num, block_start_hr,
            AVG(CASE WHEN vital_category='heart_rate' THEN vital_value END) AS mean_heart_rate,
            AVG(CASE WHEN vital_category='temp_c' THEN vital_value END) AS mean_temp_c,
            AVG(CASE WHEN vital_category='spo2' THEN vital_value END) AS mean_spo2,
            AVG(CASE WHEN vital_category='map' THEN vital_value END) AS mean_map,
            COUNT(CASE WHEN vital_category='heart_rate' THEN 1 END) AS n_heart_rate,
            COUNT(CASE WHEN vital_category='temp_c' THEN 1 END) AS n_temp_c,
            COUNT(CASE WHEN vital_category='spo2' THEN 1 END) AS n_spo2,
            COUNT(CASE WHEN vital_category='map' THEN 1 END) AS n_map
        FROM vitals_binned
        GROUP BY patient_id, hospitalization_id, survival_status, arrest_type, poa_present, block_num, block_start_hr
        ORDER BY survival_status, patient_id, block_num
    """).fetchdf()

    log.info(f"\n  {NB} blocks of {BS}h (0-{WH}h)")
    log.info(f"  Patients  : {block_vitals['patient_id'].nunique():,}")
    log.info(f"  Encounters: {block_vitals['hospitalization_id'].nunique():,}")
    log.info(f"  Rows      : {len(block_vitals):,}")

    log.info(f"\n  By survival:")
    for status in ["Survivor", "Non-Survivor"]:
        sub = block_vitals[block_vitals["survival_status"] == status]
        log.info(f"    {status:<15s} {sub['patient_id'].nunique():>10,} patients, {sub['hospitalization_id'].nunique():>12,} enc")

    log.info(f"\n  Coverage per block:")
    log.info(f"  {'Block':<8s} {'HR':>6s} {'Temp':>6s} {'SpO2':>6s} {'MAP':>6s}")
    log.info(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for b in range(NB):
        sub = block_vitals[block_vitals["block_num"] == b]
        log.info(f"  {b*BS}-{(b+1)*BS}h   {(sub['n_heart_rate']>0).sum():>6,} {(sub['n_temp_c']>0).sum():>6,} {(sub['n_spo2']>0).sum():>6,} {(sub['n_map']>0).sum():>6,}")

    for vital, col in [("heart_rate","n_heart_rate"),("temp_c","n_temp_c"),("spo2","n_spo2"),("map","n_map")]:
        log.info(f"  {vital:15s}: avg {block_vitals[col].mean():.1f} meas per {BS}h block")

    block_vitals.to_csv(config["intermediate_dir"] / f"block_vitals_ohca_icu_{BS}h_{WH}h.csv", index=False)
    block_vitals.to_parquet(config["intermediate_dir"] / f"block_vitals_ohca_icu_{BS}h_{WH}h.parquet", index=False)
    log.info(f"  [OK] Saved block_vitals_{BS}h_{WH}h")
    log.info("=" * 60)
    return block_vitals


# =============================================================
# STEP 5: PLOTS — BLOCKED + HOURLY
# =============================================================
def step5_vitals_plots(config, log, block_vitals, raw_vitals):
    """Generate blocked and hourly vital sign plots. Window-dependent."""
    WH = config["window_hours"]
    BS = config["block_size"]
    log.info("\n" + "=" * 60)
    log.info(f"  STEP 5: PLOTS (0-{WH}h)")
    log.info("=" * 60)

    tick_pos, tick_labels, bxlim = block_plot_params(config)
    hxlim, hxticks = hourly_plot_params(config)

    # --- 2x2 Block plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, info) in zip(axes.flatten(), VITALS_INFO.items()):
        for status, color in SURV_COLORS.items():
            subset = block_vitals[block_vitals["survival_status"] == status]
            grouped = subset.groupby("block_start_hr")[col].agg(["mean","std","count"])
            grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
            mid = grouped.index + BS/2
            n = subset["hospitalization_id"].nunique()
            ax.plot(mid, grouped["mean"], color=color, linewidth=2, marker="o", markersize=4,
                    label=f"{status} (n={n:,})")
            ax.fill_between(mid, grouped["mean"]-1.96*grouped["se"],
                            grouped["mean"]+1.96*grouped["se"], color=color, alpha=0.15)
        ax.set_xlabel("Hours from First Vital"); ax.set_ylabel(info["ylabel"]); ax.set_title(info["title"])
        ax.set_xlim(*bxlim); ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=6, rotation=45, ha="right")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f"OHCA (ICU): Survivors vs Non-Survivors [{BS}h blocks, 0-{WH}h]",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(config["upload_dir"] / f"ohca_icu_vitals_blocked_{WH}h.png", dpi=150, bbox_inches="tight")
    log.info(f"  [OK] ohca_icu_vitals_blocked_{WH}h.png"); plt.close()

    # --- Individual block plots ---
    for col, info in VITALS_INFO.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        for status, color in SURV_COLORS.items():
            subset = block_vitals[block_vitals["survival_status"] == status]
            grouped = subset.groupby("block_start_hr")[col].agg(["mean","std","count"])
            grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
            mid = grouped.index + BS/2
            n = subset["hospitalization_id"].nunique()
            ax.plot(mid, grouped["mean"], color=color, linewidth=2, marker="o", markersize=5,
                    label=f"{status} (n={n:,})")
            ax.fill_between(mid, grouped["mean"]-1.96*grouped["se"],
                            grouped["mean"]+1.96*grouped["se"], color=color, alpha=0.15)
        ax.set_xlabel("Hours from First Vital"); ax.set_ylabel(info["ylabel"])
        ax.set_title(f"OHCA (ICU): {info['title']} [0-{WH}h]")
        ax.set_xlim(*bxlim); ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        fname = f"ohca_icu_{col.replace('mean_','')}_blocked_{WH}h.png"
        fig.savefig(config["upload_dir"] / fname, dpi=150, bbox_inches="tight")
        log.info(f"  [OK] {fname}"); plt.close()

    # --- 2x2 Hourly plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (vital, info) in zip(axes.flatten(), VITALS_HOURLY_INFO.items()):
        vd = raw_vitals[raw_vitals["vital_category"] == vital].copy()
        vd["hour"] = vd["hours_from_first_vital"].round(0).astype(int)
        vd = vd[(vd["hour"] >= 0) & (vd["hour"] <= WH)]
        for status, color in SURV_COLORS.items():
            sub = vd[vd["survival_status"] == status]
            grouped = sub.groupby("hour")["vital_value"].agg(["mean","std","count"])
            grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
            n = sub["hospitalization_id"].nunique()
            ax.plot(grouped.index, grouped["mean"], color=color, linewidth=1.5,
                    label=f"{status} (n={n:,})")
            ax.fill_between(grouped.index, grouped["mean"]-1.96*grouped["se"],
                            grouped["mean"]+1.96*grouped["se"], color=color, alpha=0.15)
        ax.set_xlabel("Hours from First Vital"); ax.set_ylabel(info["ylabel"]); ax.set_title(info["title"])
        ax.set_xlim(*hxlim); ax.set_xticks(hxticks); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f"OHCA (ICU): Survivors vs Non-Survivors [Hourly, 0-{WH}h]",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(config["upload_dir"] / f"ohca_icu_vitals_hourly_{WH}h.png", dpi=150, bbox_inches="tight")
    log.info(f"  [OK] ohca_icu_vitals_hourly_{WH}h.png"); plt.close()

    # --- Individual hourly plots ---
    for vital, info in VITALS_HOURLY_INFO.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        vd = raw_vitals[raw_vitals["vital_category"] == vital].copy()
        vd["hour"] = vd["hours_from_first_vital"].round(0).astype(int)
        vd = vd[(vd["hour"] >= 0) & (vd["hour"] <= WH)]
        for status, color in SURV_COLORS.items():
            sub = vd[vd["survival_status"] == status]
            grouped = sub.groupby("hour")["vital_value"].agg(["mean","std","count"])
            grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
            n = sub["hospitalization_id"].nunique()
            ax.plot(grouped.index, grouped["mean"], color=color, linewidth=1.5,
                    label=f"{status} (n={n:,})")
            ax.fill_between(grouped.index, grouped["mean"]-1.96*grouped["se"],
                            grouped["mean"]+1.96*grouped["se"], color=color, alpha=0.15)
        ax.set_xlabel("Hours from First Vital"); ax.set_ylabel(info["ylabel"])
        ax.set_title(f"OHCA (ICU): {info['title']} — Hourly [0-{WH}h]")
        ax.set_xlim(*hxlim); ax.set_xticks(hxticks); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = f"ohca_icu_{vital}_hourly_{WH}h.png"
        fig.savefig(config["upload_dir"] / fname, dpi=150, bbox_inches="tight")
        log.info(f"  [OK] {fname}"); plt.close()

    log.info("=" * 60)


# =============================================================
# STEP 6: SAVE FOR R
# =============================================================
def step6_save_for_r(config, log, cohort_ohca_icu, raw_vitals, raw_temp, block_vitals):
    WH = config["window_hours"]
    BS = config["block_size"]
    log.info("\n" + "=" * 60)
    log.info(f"  STEP 6: SAVE FOR R ({WH}h)")
    log.info("=" * 60)
    cohort_ohca_icu.to_parquet(config["intermediate_dir"] / "cohort_ohca_icu.parquet", index=False)
    log.info(f"  [OK] cohort_ohca_icu.parquet — {cohort_ohca_icu['hospitalization_id'].nunique():,} enc")
    raw_vitals.to_parquet(config["intermediate_dir"] / f"raw_vitals_ohca_icu_{WH}h.parquet", index=False)
    log.info(f"  [OK] raw_vitals_{WH}h.parquet — {raw_vitals['hospitalization_id'].nunique():,} enc")
    raw_temp.to_parquet(config["intermediate_dir"] / f"raw_temp_ohca_icu_{WH}h.parquet", index=False)
    log.info(f"  [OK] raw_temp_{WH}h.parquet — {raw_temp['hospitalization_id'].nunique():,} enc")
    block_vitals.to_parquet(config["intermediate_dir"] / f"block_vitals_ohca_icu_{BS}h_{WH}h.parquet", index=False)
    log.info(f"  [OK] block_vitals_{BS}h_{WH}h.parquet")
    log.info("=" * 60)


# =============================================================
# STEP 7: TRAJECTORY ASSIGNMENT
# =============================================================
def step7_trajectory_assignment(config, log, cohort_ohca_icu, raw_temp):
    WH = config["window_hours"]
    log.info("\n" + "=" * 60)
    log.info(f"  STEP 7: TEMPERATURE TRAJECTORY ASSIGNMENT (0-{WH}h)")
    log.info("=" * 60)

    n0 = cohort_ohca_icu["hospitalization_id"].nunique()
    n1 = raw_temp["hospitalization_id"].nunique()
    log.info(f"\n  OHCA ICU cohort: {n0:,}")
    log.info(f"  Has temp_c (32-44): {n1:,} (lost {n0-n1:,})")

    hourly_counts = raw_temp.copy()
    hourly_counts["adm"] = hourly_counts["hours_from_first_temp"].round(0).astype(int)
    hourly_counts = hourly_counts[(hourly_counts["adm"] >= 0) & (hourly_counts["adm"] < WH)]
    density = hourly_counts.groupby("hospitalization_id").agg(
        n_measurements=("temperature","count"), n_hours=("adm","nunique")).reset_index()

    n2 = density["hospitalization_id"].nunique()
    log.info(f"  Within 0-{WH}h: {n2:,} (lost {n1-n2:,})")
    log.info(f"  Median meas/patient: {density['n_measurements'].median():.0f}")
    log.info(f"  Median hours: {density['n_hours'].median():.0f}")
    log.info(f"  1 measurement: {(density['n_measurements']==1).sum():,}")
    log.info(f"  <3 hours: {(density['n_hours']<3).sum():,}")

    # Build vitals_temp
    vitals_temp = raw_temp[["hospitalization_id","temperature","recorded_dttm","hours_from_first_temp"]].copy()
    vitals_temp["adm"] = vitals_temp["hours_from_first_temp"].round(0).astype(int)
    vitals_temp = vitals_temp[(vitals_temp["adm"] >= 0) & (vitals_temp["adm"] < WH)]

    n_before = vitals_temp["hospitalization_id"].nunique()
    vitals_temp = (vitals_temp.sort_values(["hospitalization_id","adm","recorded_dttm"])
                   .groupby(["hospitalization_id","adm"]).first().reset_index())
    n_after = vitals_temp["hospitalization_id"].nunique()
    log.info(f"\n  Before dedup: {n_before:,} enc, {len(vitals_temp):,} rows → After: {n_after:,} enc")

    vitals_temp["temp_raw"] = vitals_temp["temperature"]
    temp_mean = vitals_temp["temperature"].mean()
    temp_std = vitals_temp["temperature"].std()
    vitals_temp["temperature"] = (vitals_temp["temperature"] - temp_mean) / temp_std
    log.info(f"  Z-score: mean={temp_mean:.2f}C, SD={temp_std:.2f}C")

    # Trajectory curves
    vitals_temp["traj1"] = -0.89548 - 0.00298*vitals_temp["adm"] + 0.00010*(vitals_temp["adm"]**2)
    vitals_temp["traj2"] = -0.00667 + 0.00050*vitals_temp["adm"] - 0.00001*(vitals_temp["adm"]**2)
    vitals_temp["traj3"] =  1.35157 - 0.06946*vitals_temp["adm"] + 0.00065*(vitals_temp["adm"]**2)
    vitals_temp["traj4"] =  1.22203 - 0.00590*vitals_temp["adm"] - 0.00007*(vitals_temp["adm"]**2)

    for i in range(1, 5):
        vitals_temp[f"err{i}"] = (vitals_temp["temperature"] - vitals_temp[f"traj{i}"])**2

    traj_assignment = vitals_temp.groupby("hospitalization_id")[["err1","err2","err3","err4"]].sum().reset_index()
    traj_labels = {1:"Hypothermic", 2:"Normothermic", 3:"Rapid Decline", 4:"Persistent High"}
    traj_assignment["model"] = traj_assignment[["err1","err2","err3","err4"]].idxmin(axis=1)
    traj_assignment["model"] = traj_assignment["model"].map({"err1":1,"err2":2,"err3":3,"err4":4})
    traj_assignment["trajectory"] = traj_assignment["model"].map(traj_labels)
    traj_assignment = traj_assignment.merge(
        cohort_ohca_icu[["hospitalization_id","survival_status"]].drop_duplicates(), on="hospitalization_id")

    n_traj = traj_assignment["hospitalization_id"].nunique()
    log.info(f"\n  Trajectory assigned: {n_traj:,} (lost {n_after-n_traj:,})")

    log.info(f"\n  {'Trajectory':<20s} {'Survivor':>10s} {'Non-Surv':>10s} {'Total':>8s} {'Mort%':>8s}")
    log.info(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    total_all = 0
    for tn in ["Hypothermic","Normothermic","Rapid Decline","Persistent High"]:
        sub = traj_assignment[traj_assignment["trajectory"] == tn]
        s = (sub["survival_status"]=="Survivor").sum()
        ns = (sub["survival_status"]=="Non-Survivor").sum()
        t = len(sub); total_all += t
        log.info(f"  {tn:<20s} {s:>10,} {ns:>10,} {t:>8,} {ns/t*100 if t else 0:>7.1f}%")
    log.info(f"  {'TOTAL':<20s} {'':>10s} {'':>10s} {total_all:>8,}")

    log.info(f"\n  === DROP SUMMARY ===")
    log.info(f"  {'OHCA ICU cohort':<40s} {n0:>8,}")
    log.info(f"  {'Has temp_c (32-44C)':<40s} {n1:>8,} (lost {n0-n1:,})")
    log.info(f"  {f'Within 0-{WH}h':<40s} {n2:>8,} (lost {n1-n2:,})")
    log.info(f"  {'After hourly dedup':<40s} {n_after:>8,} (lost {n2-n_after:,})")
    log.info(f"  {'Trajectory assigned':<40s} {n_traj:>8,} (lost {n_after-n_traj:,})")

    traj_assignment.to_csv(config["intermediate_dir"] / f"ohca_icu_traj_assignment_{WH}h.csv", index=False)
    log.info(f"  [OK] Saved traj_assignment_{WH}h")
    log.info("=" * 60)
    return traj_assignment, vitals_temp


# =============================================================
# STEP 7b: HOURLY VITALS TABLE BY TRAJECTORY × SURVIVAL
# =============================================================
def step7b_hourly_vitals_table(config, con, log, traj_assignment):
    WH = config["window_hours"]
    log.info("\n" + "=" * 60)
    log.info(f"  STEP 7b: HOURLY VITALS TABLE (0-{WH}h)")
    log.info("=" * 60)

    vitals_table = read_table(config, "clif_vitals")
    vf = vitals_filter_sql()

    raw_vitals_wh = con.execute(f"""
        WITH first_vital AS (
            SELECT v.hospitalization_id, MIN(v.recorded_dttm) AS first_vital_dttm
            FROM {vitals_table} v
            WHERE v.hospitalization_id IN (SELECT hospitalization_id FROM ohca_icu_df)
                AND v.vital_category IN ({vf})
            GROUP BY v.hospitalization_id
        ),
        cohort AS (
            SELECT DISTINCT c.patient_id, c.hospitalization_id, c.survival_status,
                fv.first_vital_dttm AS time_zero
            FROM ohca_icu_df c
            INNER JOIN first_vital fv ON c.hospitalization_id=fv.hospitalization_id
        )
        SELECT c.hospitalization_id, c.survival_status,
            v.vital_category, CAST(v.vital_value AS DOUBLE) AS vital_value,
            v.recorded_dttm, c.time_zero,
            ROUND(EXTRACT(EPOCH FROM (v.recorded_dttm - c.time_zero))/3600) AS hour
        FROM cohort c
        INNER JOIN {vitals_table} v ON c.hospitalization_id=v.hospitalization_id
        WHERE v.vital_category IN ({vf})
            AND v.recorded_dttm >= c.time_zero
            AND EXTRACT(EPOCH FROM (v.recorded_dttm - c.time_zero))/3600 < {WH}
        ORDER BY c.hospitalization_id, v.recorded_dttm
    """).fetchdf()

    log.info(f"  Raw: {len(raw_vitals_wh):,} rows, {raw_vitals_wh['hospitalization_id'].nunique():,} enc")

    rv = (raw_vitals_wh.sort_values(["hospitalization_id","vital_category","hour","recorded_dttm"])
          .groupby(["hospitalization_id","vital_category","hour"]).first().reset_index())
    rv = rv.merge(traj_assignment[["hospitalization_id","trajectory"]], on="hospitalization_id", how="inner")
    rv["trajectory"] = rv["trajectory"].fillna("Unassigned")

    log.info(f"  After dedup+join: {len(rv):,} rows")

    hourly_long = rv.groupby(["hour","trajectory","survival_status","vital_category"]).agg(
        mean_value=("vital_value","mean"), sd_value=("vital_value","std"),
        n=("hospitalization_id","nunique")).reset_index()
    hourly_long["se"] = hourly_long["sd_value"] / np.sqrt(hourly_long["n"])

    hourly_wide = hourly_long.pivot_table(
        index=["hour","trajectory","survival_status"], columns="vital_category",
        values=["mean_value","sd_value","se","n"], aggfunc="first").reset_index()
    hourly_wide.columns = [f"{v}_{s}" if s else v for v, s in hourly_wide.columns]

    rename_map = {}
    for vital in VITAL_SIGNS:
        rename_map[f"mean_value_{vital}"] = f"mean_{vital}"
        rename_map[f"sd_value_{vital}"] = f"sd_{vital}"
        rename_map[f"se_{vital}"] = f"se_{vital}"
        rename_map[f"n_{vital}"] = f"n_{vital}"
    hourly_wide = hourly_wide.rename(columns=rename_map)
    hourly_wide = hourly_wide.sort_values(["trajectory","survival_status","hour"]).reset_index(drop=True)

    log.info(f"  Final: {len(hourly_wide):,} rows, cols={list(hourly_wide.columns)}")

    hourly_wide.to_csv(config["upload_dir"] / f"hourly_vitals_by_trajectory_survival_{WH}h.csv", index=False)
    hourly_wide.to_parquet(config["upload_dir"] / f"hourly_vitals_by_trajectory_survival_{WH}h.parquet", index=False)
    log.info(f"  [OK] Saved hourly_vitals_{WH}h → upload_dir")
    log.info("=" * 60)
    return hourly_wide


# =============================================================
# STEP 8: TRAJECTORY PLOTS
# =============================================================
def step8_trajectory_plots(config, log, vitals_temp, traj_assignment):
    WH = config["window_hours"]
    log.info("\n" + "=" * 60)
    log.info(f"  STEP 8: TRAJECTORY PLOTS (0-{WH}h)")
    log.info("=" * 60)

    vitals_by_traj = vitals_temp.merge(
        traj_assignment[["hospitalization_id","trajectory"]], on="hospitalization_id")

    # Reference curves
    fig, ax = plt.subplots(figsize=(10, 6))
    hours = np.arange(0, WH+1)
    trajs = {
        "Hypothermic":    -0.89548 - 0.00298*hours + 0.00010*(hours**2),
        "Normothermic":   -0.00667 + 0.00050*hours - 0.00001*(hours**2),
        "Rapid Decline":   1.35157 - 0.06946*hours + 0.00065*(hours**2),
        "Persistent High": 1.22203 - 0.00590*hours - 0.00007*(hours**2),
    }
    for name, curve in trajs.items():
        ax.plot(hours, curve, linewidth=2, label=name, color=TRAJ_COLORS[name])
    ax.set_xlabel("Hours from First Temperature"); ax.set_ylabel("Temperature (z-score)")
    ax.set_title(f"Reference Trajectory Models (0-{WH}h)"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(config["upload_dir"] / f"ohca_traj_reference_{WH}h.png", dpi=150, bbox_inches="tight")
    log.info(f"  [OK] ohca_traj_reference_{WH}h.png"); plt.close()

    # Actual temp (C)
    fig, ax = plt.subplots(figsize=(10, 6))
    for tn, color in TRAJ_COLORS.items():
        sub = vitals_by_traj[vitals_by_traj["trajectory"] == tn]
        grouped = sub.groupby("adm")["temp_raw"].agg(["mean","std","count"])
        grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
        n = sub["hospitalization_id"].nunique()
        ax.plot(grouped.index, grouped["mean"], linewidth=1.5, color=color, label=f"{tn} (n={n:,})")
        ax.fill_between(grouped.index, grouped["mean"]-1.96*grouped["se"],
                        grouped["mean"]+1.96*grouped["se"], color=color, alpha=0.15)
    ax.set_xlabel("Hours from First Temperature"); ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"OHCA (ICU): Actual Temperature by Trajectory [0-{WH}h]")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(config["upload_dir"] / f"ohca_icu_temp_by_trajectory_{WH}h.png", dpi=150, bbox_inches="tight")
    log.info(f"  [OK] ohca_icu_temp_by_trajectory_{WH}h.png"); plt.close()

    # Actual temp (z-score)
    fig, ax = plt.subplots(figsize=(10, 6))
    for tn, color in TRAJ_COLORS.items():
        sub = vitals_by_traj[vitals_by_traj["trajectory"] == tn]
        grouped = sub.groupby("adm")["temperature"].agg(["mean","std","count"])
        grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
        ax.plot(grouped.index, grouped["mean"], linewidth=1.5, color=color, label=tn)
    ax.set_xlabel("Hours from First Temperature"); ax.set_ylabel("Temperature (z-score)")
    ax.set_title(f"OHCA (ICU): Temperature by Trajectory (z-score) [0-{WH}h]")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(config["upload_dir"] / f"ohca_icu_temp_by_trajectory_zscore_{WH}h.png", dpi=150, bbox_inches="tight")
    log.info(f"  [OK] ohca_icu_temp_by_trajectory_zscore_{WH}h.png"); plt.close()
    log.info("=" * 60)
    return vitals_by_traj


# =============================================================
# STEP 9: TRAJECTORY × SURVIVAL PLOTS + MORTALITY BAR
# =============================================================
def step9_traj_survival_plots(config, log, vitals_by_traj, cohort_ohca_icu, traj_assignment):
    WH = config["window_hours"]
    log.info("\n" + "=" * 60)
    log.info(f"  STEP 9: TRAJECTORY × SURVIVAL PLOTS (0-{WH}h)")
    log.info("=" * 60)

    vitals_traj_surv = vitals_by_traj.merge(
        cohort_ohca_icu[["hospitalization_id","survival_status"]].drop_duplicates(),
        on="hospitalization_id")

    # Faceted
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, tn in enumerate(TRAJ_ORDER):
        ax = axes[idx//2][idx%2]
        for status, color in SURV_COLORS.items():
            sub = vitals_traj_surv[(vitals_traj_surv["trajectory"]==tn)&(vitals_traj_surv["survival_status"]==status)]
            if len(sub) == 0: continue
            grouped = sub.groupby("adm")["temp_raw"].agg(["mean","std","count"])
            grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
            n = sub["hospitalization_id"].nunique()
            ax.plot(grouped.index, grouped["mean"], linewidth=1.5, color=color, label=f"{status} (n={n:,})")
            ax.fill_between(grouped.index, grouped["mean"]-1.96*grouped["se"],
                            grouped["mean"]+1.96*grouped["se"], color=color, alpha=0.15)
        ax.set_xlabel("Hours from First Temperature"); ax.set_ylabel("Temperature (°C)")
        ax.set_title(tn); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f"OHCA (ICU): Survivor vs Non-Survivor by Trajectory [0-{WH}h]",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(config["upload_dir"] / f"ohca_icu_temp_traj_survival_facet_{WH}h.png", dpi=150, bbox_inches="tight")
    log.info(f"  [OK] facet_{WH}h.png"); plt.close()

    # Overall
    fig, ax = plt.subplots(figsize=(10, 6))
    for status, color in SURV_COLORS.items():
        sub = vitals_traj_surv[vitals_traj_surv["survival_status"]==status]
        grouped = sub.groupby("adm")["temp_raw"].agg(["mean","std","count"])
        grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
        n = sub["hospitalization_id"].nunique()
        ax.plot(grouped.index, grouped["mean"], linewidth=1.5, color=color, label=f"{status} (n={n:,})")
        ax.fill_between(grouped.index, grouped["mean"]-1.96*grouped["se"],
                        grouped["mean"]+1.96*grouped["se"], color=color, alpha=0.15)
    ax.set_xlabel("Hours from First Temperature"); ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"OHCA (ICU): Temperature — Survivors vs Non-Survivors [0-{WH}h]")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(config["upload_dir"] / f"ohca_icu_temp_survival_overall_{WH}h.png", dpi=150, bbox_inches="tight")
    log.info(f"  [OK] survival_overall_{WH}h.png"); plt.close()

    # Mortality bar
    fig, ax = plt.subplots(figsize=(8, 5))
    mort_rates, totals = [], []
    for t in TRAJ_ORDER:
        sub = traj_assignment[traj_assignment["trajectory"]==t]
        total = len(sub); died = (sub["survival_status"]=="Non-Survivor").sum()
        mort_rates.append(died/total*100 if total else 0); totals.append(total)
    bars = ax.bar(TRAJ_ORDER, mort_rates, color=[TRAJ_COLORS[t] for t in TRAJ_ORDER])
    for bar, rate, n in zip(bars, mort_rates, totals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{rate:.1f}%\n(n={n:,})", ha="center", fontsize=9)
    ax.set_ylabel("Mortality (%)"); ax.set_title(f"OHCA (ICU): Mortality by Trajectory [0-{WH}h]")
    ax.set_ylim(0, max(mort_rates)+15); ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15, ha="right"); fig.tight_layout()
    fig.savefig(config["upload_dir"] / f"ohca_icu_mortality_by_trajectory_{WH}h.png", dpi=150, bbox_inches="tight")
    log.info(f"  [OK] mortality_bar_{WH}h.png"); plt.close()

    # Count bar
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, t in enumerate(TRAJ_ORDER):
        sub = traj_assignment[traj_assignment["trajectory"]==t]
        s = (sub["survival_status"]=="Survivor").sum(); ns = (sub["survival_status"]=="Non-Survivor").sum()
        ax.bar(i-0.15, s, 0.3, color=SURV_COLORS["Survivor"], label="Survivor" if i==0 else "")
        ax.bar(i+0.15, ns, 0.3, color=SURV_COLORS["Non-Survivor"], label="Non-Survivor" if i==0 else "")
    ax.set_xticks(range(len(TRAJ_ORDER))); ax.set_xticklabels(TRAJ_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Number of Patients"); ax.set_title(f"OHCA (ICU): Trajectory by Survival [0-{WH}h]")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(config["upload_dir"] / f"ohca_icu_traj_counts_{WH}h.png", dpi=150, bbox_inches="tight")
    log.info(f"  [OK] traj_counts_{WH}h.png"); plt.close()
    log.info("=" * 60)


# =============================================================
# STEP 10: COHORT COMPARISON
# =============================================================
def step10_cohort_comparison(config, log, cohort_ohca_icu, cohort_v2):
    WH = config["window_hours"]
    log.info("\n" + "=" * 60)
    log.info(f"  STEP 10: COHORT COMPARISON ({WH}h)")
    log.info("=" * 60)

    icu_total = cohort_ohca_icu["hospitalization_id"].nunique()
    icu_died = cohort_ohca_icu[cohort_ohca_icu["survival_status"]=="Non-Survivor"]["hospitalization_id"].nunique()
    icu_mort = icu_died/icu_total*100

    new_total = cohort_v2["hospitalization_id"].nunique()
    new_died = cohort_v2[cohort_v2["survival_status"]=="Non-Survivor"]["hospitalization_id"].nunique()
    new_mort = new_died/new_total*100

    log.info(f"\n  {'Cohort':<50s} {'N':>8s} {'Died':>8s} {'Mort%':>8s}")
    log.info(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*8}")
    log.info(f"  {'hospital_diagnosis (all CA)':<50s} {new_total:>8,} {new_died:>8,} {new_mort:>7.1f}%")
    log.info(f"  {'OHCA (ICU-admitted)':<50s} {icu_total:>8,} {icu_died:>8,} {icu_mort:>7.1f}%")

    fig, ax = plt.subplots(figsize=(8, 6))
    cohorts = ["All CA\n(hospital_diagnosis)", "OHCA\n(ICU-admitted)"]
    morts = [new_mort, icu_mort]; ns = [new_total, icu_total]
    bars = ax.bar(cohorts, morts, color=["#9E9E9E","#2196F3"], width=0.5)
    for bar, rate, n in zip(bars, morts, ns):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{rate:.1f}%\n(n={n:,})", ha="center", fontsize=10)
    ax.set_ylabel("Mortality (%)"); ax.set_title("Mortality Rate by Cohort Definition")
    ax.set_ylim(0, max(morts)+15); ax.grid(True, alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(config["upload_dir"] / f"mortality_cohort_comparison_{WH}h.png", dpi=150, bbox_inches="tight")
    log.info(f"  [OK] mortality_cohort_comparison_{WH}h.png"); plt.close()
    log.info("=" * 60)


# =============================================================
# STEP 11: TABLE 1
# =============================================================
def step11_table1(config, con, log, traj_assignment):
    WH = config["window_hours"]
    log.info("\n" + "=" * 60)
    log.info(f"  STEP 11: TABLE 1 ({WH}h)")
    log.info("=" * 60)

    patient_table = read_table(config, "clif_patient")
    hosp_table = read_table(config, "clif_hospitalization")
    adt_table = read_table(config, "clif_adt")

    table1 = con.execute(f"""
        WITH hosp_info AS (
            SELECT h.hospitalization_id, h.patient_id, h.admission_dttm, h.discharge_dttm,
                ROUND(EXTRACT(EPOCH FROM (h.discharge_dttm-h.admission_dttm))/3600/24, 1) AS los_days
            FROM {hosp_table} h WHERE h.hospitalization_id IN (SELECT hospitalization_id FROM ohca_icu_df)
        ),
        icu_los AS (
            SELECT a.hospitalization_id,
                ROUND(SUM(EXTRACT(EPOCH FROM (a.out_dttm-a.in_dttm)))/3600/24, 1) AS icu_los_days
            FROM {adt_table} a WHERE a.hospitalization_id IN (SELECT hospitalization_id FROM ohca_icu_df)
                AND a.location_category='icu' GROUP BY a.hospitalization_id
        )
        SELECT c.hospitalization_id, c.patient_id, c.survival_status, c.arrest_type,
            p.race_category, p.ethnicity_category, p.sex_category,
            EXTRACT(YEAR FROM h.admission_dttm)-EXTRACT(YEAR FROM p.birth_date) AS age,
            h.los_days, i.icu_los_days
        FROM ohca_icu_df c
        INNER JOIN {patient_table} p ON c.patient_id=p.patient_id
        INNER JOIN hosp_info h ON c.hospitalization_id=h.hospitalization_id
        LEFT JOIN icu_los i ON c.hospitalization_id=i.hospitalization_id
    """).fetchdf()

    table1 = table1.merge(traj_assignment[["hospitalization_id","trajectory"]], on="hospitalization_id", how="left")
    table1["trajectory"] = table1["trajectory"].fillna("No temp data")

    log.info(f"  Rows: {len(table1):,}, Encounters: {table1['hospitalization_id'].nunique():,}")

    # Demographics
    log.info(f"\n  Age: {table1['age'].mean():.1f}±{table1['age'].std():.1f}, "
             f"median {table1['age'].median():.0f} (IQR {table1['age'].quantile(0.25):.0f}-{table1['age'].quantile(0.75):.0f})")
    for col_name, col in [("Sex","sex_category"),("Race","race_category"),("Ethnicity","ethnicity_category")]:
        log.info(f"\n  {col_name}:")
        for val, grp in table1.groupby(col):
            log.info(f"    {str(val):<30s}: {len(grp):>6,} ({len(grp)/len(table1)*100:.1f}%)")
    log.info(f"\n  Hosp LOS: median {table1['los_days'].median():.1f} (IQR {table1['los_days'].quantile(0.25):.1f}-{table1['los_days'].quantile(0.75):.1f})")
    log.info(f"  ICU LOS: median {table1['icu_los_days'].median():.1f} (IQR {table1['icu_los_days'].quantile(0.25):.1f}-{table1['icu_los_days'].quantile(0.75):.1f})")

    for status in ["Survivor","Non-Survivor"]:
        sub = table1[table1["survival_status"]==status]
        log.info(f"\n  {status} (n={len(sub):,}): Age {sub['age'].mean():.1f}±{sub['age'].std():.1f}, "
                 f"Male {(sub['sex_category']=='Male').sum():,} ({(sub['sex_category']=='Male').sum()/len(sub)*100:.1f}%), "
                 f"Hosp LOS {sub['los_days'].median():.1f}, ICU LOS {sub['icu_los_days'].median():.1f}")

    for traj in ["Hypothermic","Normothermic","Rapid Decline","Persistent High","No temp data"]:
        sub = table1[table1["trajectory"]==traj]
        if len(sub) == 0: continue
        mort = (sub["survival_status"]=="Non-Survivor").sum()/len(sub)*100
        log.info(f"\n  {traj} (n={len(sub):,}, mort={mort:.1f}%): Age {sub['age'].mean():.1f}±{sub['age'].std():.1f}, "
                 f"Male {(sub['sex_category']=='Male').sum()/len(sub)*100:.0f}%")

    table1.to_csv(config["intermediate_dir"] / f"table1_ohca_icu_{WH}h.csv", index=False)
    table1.to_parquet(config["intermediate_dir"] / f"table1_ohca_icu_{WH}h.parquet", index=False)
    log.info(f"  [OK] Saved table1_{WH}h")
    log.info("=" * 60)
    return table1


# =============================================================
# TABLE 1 OUTPUTS (text, poolable CSV, summary CSV)
# =============================================================
def save_table1_outputs(config, log, table1):
    print(config)
    WH = config["window_hours"]
    site = config["site_name"]

    n_surv = len(table1[table1["survival_status"]=="Survivor"])
    n_nonsurv = len(table1[table1["survival_status"]=="Non-Survivor"])

    # --- Formatted text ---
    with open(config["upload_dir"] / f"table1_summary_{WH}h.txt", "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write(f"  TABLE 1: OHCA ICU COHORT ({WH}h window)\n")
        f.write("="*70 + "\n\n")
        header = f"  {'Variable':<35s} {'Overall':>12s} {'Survivor':>12s} {'Non-Surv':>12s}\n"
        f.write(header)
        f.write(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12}\n")
        for var_name, func in [
            ("N", lambda s: f"{len(s):,}"),
            ("Age, mean ± SD", lambda s: f"{s['age'].mean():.1f} ± {s['age'].std():.1f}"),
            ("Age, median (IQR)", lambda s: f"{s['age'].median():.0f} ({s['age'].quantile(0.25):.0f}-{s['age'].quantile(0.75):.0f})"),
            ("Male, n (%)", lambda s: f"{(s['sex_category']=='Male').sum():,} ({(s['sex_category']=='Male').sum()/len(s)*100:.1f}%)"),
            ("Hospital LOS, median (IQR)", lambda s: f"{s['los_days'].median():.1f} ({s['los_days'].quantile(0.25):.1f}-{s['los_days'].quantile(0.75):.1f})"),
            ("ICU LOS, median (IQR)", lambda s: f"{s['icu_los_days'].median():.1f} ({s['icu_los_days'].quantile(0.25):.1f}-{s['icu_los_days'].quantile(0.75):.1f})"),
            ("Mortality, n (%)", lambda s: f"{(s['survival_status']=='Non-Survivor').sum():,} ({(s['survival_status']=='Non-Survivor').sum()/len(s)*100:.1f}%)"),
        ]:
            f.write(f"  {var_name:<35s} {func(table1):>12s} "
                    f"{func(table1[table1['survival_status']=='Survivor']):>12s} "
                    f"{func(table1[table1['survival_status']=='Non-Survivor']):>12s}\n")
        f.write(f"\n  Race:\n")
        for race in sorted(table1["race_category"].dropna().unique()):
            ca = (table1["race_category"]==race).sum()
            cs = (table1[table1["survival_status"]=="Survivor"]["race_category"]==race).sum()
            cn = (table1[table1["survival_status"]=="Non-Survivor"]["race_category"]==race).sum()
            f.write(f"    {race:<33s} {ca:>5,} ({ca/len(table1)*100:4.1f}%) "
                    f"{cs:>5,} ({cs/max(n_surv,1)*100:4.1f}%) {cn:>5,} ({cn/max(n_nonsurv,1)*100:4.1f}%)\n")
        f.write(f"\n  Ethnicity:\n")
        for eth in sorted(table1["ethnicity_category"].dropna().unique()):
            ca = (table1["ethnicity_category"]==eth).sum()
            cs = (table1[table1["survival_status"]=="Survivor"]["ethnicity_category"]==eth).sum()
            cn = (table1[table1["survival_status"]=="Non-Survivor"]["ethnicity_category"]==eth).sum()
            f.write(f"    {eth:<33s} {ca:>5,} ({ca/len(table1)*100:4.1f}%) "
                    f"{cs:>5,} ({cs/max(n_surv,1)*100:4.1f}%) {cn:>5,} ({cn/max(n_nonsurv,1)*100:4.1f}%)\n")
        f.write(f"\n\n{'='*70}\n  BY TEMPERATURE TRAJECTORY\n{'='*70}\n\n")
        f.write(f"  {'Variable':<25s} {'Hypothermic':>12s} {'Normothermic':>13s} {'Rapid Decl':>12s} {'Persist Hi':>12s}\n")
        f.write(f"  {'-'*25} {'-'*12} {'-'*13} {'-'*12} {'-'*12}\n")
        for var_name, func in [
            ("N", lambda s: f"{len(s):,}"),
            ("Age, mean ± SD", lambda s: f"{s['age'].mean():.1f}±{s['age'].std():.1f}"),
            ("Male, n (%)", lambda s: f"{(s['sex_category']=='Male').sum()} ({(s['sex_category']=='Male').sum()/max(len(s),1)*100:.0f}%)"),
            ("Hosp LOS, median", lambda s: f"{s['los_days'].median():.1f}"),
            ("ICU LOS, median", lambda s: f"{s['icu_los_days'].median():.1f}"),
            ("Mortality, n (%)", lambda s: f"{(s['survival_status']=='Non-Survivor').sum()} ({(s['survival_status']=='Non-Survivor').sum()/max(len(s),1)*100:.1f}%)"),
        ]:
            vals = [func(table1[table1["trajectory"]==t]) for t in ["Hypothermic","Normothermic","Rapid Decline","Persistent High"]]
            f.write(f"  {var_name:<25s} {vals[0]:>12s} {vals[1]:>13s} {vals[2]:>12s} {vals[3]:>12s}\n")
    log.info(f"  [OK] table1_summary_{WH}h.txt")

    # --- Poolable CSV ---
    rows = []
    for group_type, group_name, sub in [
        ("survival","Overall",table1),
        ("survival","Survivor",table1[table1["survival_status"]=="Survivor"]),
        ("survival","Non-Survivor",table1[table1["survival_status"]=="Non-Survivor"]),
        ("trajectory","Hypothermic",table1[table1["trajectory"]=="Hypothermic"]),
        ("trajectory","Normothermic",table1[table1["trajectory"]=="Normothermic"]),
        ("trajectory","Rapid Decline",table1[table1["trajectory"]=="Rapid Decline"]),
        ("trajectory","Persistent High",table1[table1["trajectory"]=="Persistent High"]),
        ("trajectory","No temp data",table1[table1["trajectory"]=="No temp data"]),
    ]:
        n = len(sub)
        if n == 0: continue
        base = {"site":site, "window_hours":WH, "group_type":group_type, "group":group_name}
        rows.append({**base, "variable":"n", "value":n, "n":n})
        for var, val in [("age_mean",round(sub["age"].mean(),2)),("age_sd",round(sub["age"].std(),2)),
                         ("age_median",round(sub["age"].median(),1)),("age_q25",round(sub["age"].quantile(0.25),1)),
                         ("age_q75",round(sub["age"].quantile(0.75),1))]:
            rows.append({**base, "variable":var, "value":val, "n":n})
        for sex in sub["sex_category"].dropna().unique():
            cnt = (sub["sex_category"]==sex).sum()
            rows.append({**base, "variable":f"sex_{sex.lower()}_n", "value":cnt, "n":n})
            rows.append({**base, "variable":f"sex_{sex.lower()}_pct", "value":round(cnt/n*100,1), "n":n})
        for race in sub["race_category"].dropna().unique():
            cnt = (sub["race_category"]==race).sum()
            rc = race.lower().replace(" ","_").replace("/","_")
            rows.append({**base, "variable":f"race_{rc}_n", "value":cnt, "n":n})
            rows.append({**base, "variable":f"race_{rc}_pct", "value":round(cnt/n*100,1), "n":n})
        for eth in sub["ethnicity_category"].dropna().unique():
            cnt = (sub["ethnicity_category"]==eth).sum()
            ec = eth.lower().replace(" ","_").replace("/","_")
            rows.append({**base, "variable":f"ethnicity_{ec}_n", "value":cnt, "n":n})
            rows.append({**base, "variable":f"ethnicity_{ec}_pct", "value":round(cnt/n*100,1), "n":n})
        for var, val in [("hosp_los_mean",round(sub["los_days"].mean(),2)),("hosp_los_sd",round(sub["los_days"].std(),2)),
                         ("hosp_los_median",round(sub["los_days"].median(),1)),
                         ("hosp_los_q25",round(sub["los_days"].quantile(0.25),1)),
                         ("hosp_los_q75",round(sub["los_days"].quantile(0.75),1))]:
            rows.append({**base, "variable":var, "value":val, "n":n})
        icu_sub = sub.dropna(subset=["icu_los_days"])
        ni = len(icu_sub)
        for var, val in [("icu_los_mean",round(icu_sub["icu_los_days"].mean(),2)),("icu_los_sd",round(icu_sub["icu_los_days"].std(),2)),
                         ("icu_los_median",round(icu_sub["icu_los_days"].median(),1)),
                         ("icu_los_q25",round(icu_sub["icu_los_days"].quantile(0.25),1)),
                         ("icu_los_q75",round(icu_sub["icu_los_days"].quantile(0.75),1))]:
            rows.append({**base, "variable":var, "value":val, "n":ni})
        died = (sub["survival_status"]=="Non-Survivor").sum()
        rows.append({**base, "variable":"mortality_n", "value":died, "n":n})
        rows.append({**base, "variable":"mortality_pct", "value":round(died/n*100,1), "n":n})

    pd.DataFrame(rows).to_csv(config["upload_dir"] / f"table1_poolable_{WH}h.csv", index=False)
    log.info(f"  [OK] table1_poolable_{WH}h.csv ({len(rows)} rows)")

    # --- Summary CSV ---
    srows = []
    for status in ["Overall","Survivor","Non-Survivor"]:
        sub = table1 if status=="Overall" else table1[table1["survival_status"]==status]
        n = len(sub)
        srows.append({"Group":status, "Variable":"N", "Value":f"{n:,}"})
        srows.append({"Group":status, "Variable":"Age, mean ± SD", "Value":f"{sub['age'].mean():.1f} ± {sub['age'].std():.1f}"})
        srows.append({"Group":status, "Variable":"Age, median (IQR)", "Value":f"{sub['age'].median():.0f} ({sub['age'].quantile(0.25):.0f}-{sub['age'].quantile(0.75):.0f})"})
        srows.append({"Group":status, "Variable":"Male, n (%)", "Value":f"{(sub['sex_category']=='Male').sum():,} ({(sub['sex_category']=='Male').sum()/n*100:.1f}%)"})
        for race in sorted(sub["race_category"].dropna().unique()):
            cnt = (sub["race_category"]==race).sum()
            srows.append({"Group":status, "Variable":f"Race: {race}", "Value":f"{cnt:,} ({cnt/n*100:.1f}%)"})
        for eth in sorted(sub["ethnicity_category"].dropna().unique()):
            cnt = (sub["ethnicity_category"]==eth).sum()
            srows.append({"Group":status, "Variable":f"Ethnicity: {eth}", "Value":f"{cnt:,} ({cnt/n*100:.1f}%)"})
        srows.append({"Group":status, "Variable":"Hospital LOS, median (IQR)", "Value":f"{sub['los_days'].median():.1f} ({sub['los_days'].quantile(0.25):.1f}-{sub['los_days'].quantile(0.75):.1f})"})
        srows.append({"Group":status, "Variable":"ICU LOS, median (IQR)", "Value":f"{sub['icu_los_days'].median():.1f} ({sub['icu_los_days'].quantile(0.25):.1f}-{sub['icu_los_days'].quantile(0.75):.1f})"})
        srows.append({"Group":status, "Variable":"Mortality, n (%)", "Value":f"{(sub['survival_status']=='Non-Survivor').sum():,} ({(sub['survival_status']=='Non-Survivor').sum()/n*100:.1f}%)"})
    for traj in ["Hypothermic","Normothermic","Rapid Decline","Persistent High","No temp data"]:
        sub = table1[table1["trajectory"]==traj]
        if len(sub)==0: continue
        n = len(sub); mort = (sub["survival_status"]=="Non-Survivor").sum()
        srows.append({"Group":traj, "Variable":"N", "Value":f"{n:,}"})
        srows.append({"Group":traj, "Variable":"Age, mean ± SD", "Value":f"{sub['age'].mean():.1f} ± {sub['age'].std():.1f}"})
        srows.append({"Group":traj, "Variable":"Age, median (IQR)", "Value":f"{sub['age'].median():.0f} ({sub['age'].quantile(0.25):.0f}-{sub['age'].quantile(0.75):.0f})"})
        srows.append({"Group":traj, "Variable":"Male, n (%)", "Value":f"{(sub['sex_category']=='Male').sum():,} ({(sub['sex_category']=='Male').sum()/n*100:.1f}%)"})
        srows.append({"Group":traj, "Variable":"Hospital LOS, median (IQR)", "Value":f"{sub['los_days'].median():.1f} ({sub['los_days'].quantile(0.25):.1f}-{sub['los_days'].quantile(0.75):.1f})"})
        srows.append({"Group":traj, "Variable":"ICU LOS, median (IQR)", "Value":f"{sub['icu_los_days'].median():.1f} ({sub['icu_los_days'].quantile(0.25):.1f}-{sub['icu_los_days'].quantile(0.75):.1f})"})
        srows.append({"Group":traj, "Variable":"Mortality, n (%)", "Value":f"{mort:,} ({mort/n*100:.1f}%)"})
    pd.DataFrame(srows).to_csv(config["upload_dir"] / f"table1_summary_{WH}h.csv", index=False)
    log.info(f"  [OK] table1_summary_{WH}h.csv")