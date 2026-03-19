import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # OHCA Pipeline — Dual Window (24h + 72h)

    Runs the full OHCA analysis pipeline for both 24h and 72h time windows.
    """)
    return


@app.cell
def _():
    import duckdb
    from pipeline_helpers import make_config, Logger
    from pipeline_steps import (
        step2_build_cohort,
        step3_filter_ohca_icu,
        step4a_extract_vitals,
        step4b_block_vitals,
        step5_vitals_plots,
        step6_save_for_r,
        step7_trajectory_assignment,
        step7b_hourly_vitals_table,
        step8_trajectory_plots,
        step9_traj_survival_plots,
        step10_cohort_comparison,
        step11_table1,
        save_table1_outputs,
    )

    return (
        Logger,
        duckdb,
        make_config,
        save_table1_outputs,
        step10_cohort_comparison,
        step11_table1,
        step2_build_cohort,
        step3_filter_ohca_icu,
        step4a_extract_vitals,
        step4b_block_vitals,
        step5_vitals_plots,
        step6_save_for_r,
        step7_trajectory_assignment,
        step7b_hourly_vitals_table,
        step8_trajectory_plots,
        step9_traj_survival_plots,
    )


@app.cell
def _(Logger, duckdb, make_config):
    con = duckdb.connect()
    config_init = make_config("config.json", window_hours=24)
    log_init = Logger(config_init["upload_dir"] / "pipeline_log.txt")
    return con, config_init, log_init


@app.cell
def _(con, config_init, log_init, step2_build_cohort):
    cohort_v2 = step2_build_cohort(config_init, con, log_init)
    return (cohort_v2,)


@app.cell
def _(cohort_v2, con, config_init, log_init, step3_filter_ohca_icu):
    cohort_ohca_icu = step3_filter_ohca_icu(config_init, con, log_init, cohort_v2)
    return (cohort_ohca_icu,)


@app.cell
def _(
    Logger,
    cohort_ohca_icu,
    cohort_v2,
    con,
    make_config,
    save_table1_outputs,
    step10_cohort_comparison,
    step11_table1,
    step4a_extract_vitals,
    step4b_block_vitals,
    step5_vitals_plots,
    step6_save_for_r,
    step7_trajectory_assignment,
    step7b_hourly_vitals_table,
    step8_trajectory_plots,
    step9_traj_survival_plots,
):
    for window_hours in [24, 72]:
        config = make_config("config.json", window_hours=window_hours)
        log = Logger(config["upload_dir"] / "pipeline_log.txt")

        log.info(f"\n{'#' * 60}")
        log.info(f"  WINDOW: {window_hours}h")
        log.info(f"{'#' * 60}")

        raw_vitals, raw_temp = step4a_extract_vitals(config, con, log)
        block_vitals = step4b_block_vitals(config, con, log)
        step5_vitals_plots(config, log, block_vitals, raw_vitals)
        step6_save_for_r(config, log, cohort_ohca_icu, raw_vitals, raw_temp, block_vitals)
        traj_assignment, vitals_temp = step7_trajectory_assignment(config, log, cohort_ohca_icu, raw_temp)
        _hourly_wide = step7b_hourly_vitals_table(config, con, log, traj_assignment)
        vitals_by_traj = step8_trajectory_plots(config, log, vitals_temp, traj_assignment)
        step9_traj_survival_plots(config, log, vitals_by_traj, cohort_ohca_icu, traj_assignment)
        step10_cohort_comparison(config, log, cohort_ohca_icu, cohort_v2)
        table1 = step11_table1(config, con, log, traj_assignment)
        save_table1_outputs(config, log, table1)

        log.info(f"\n  WINDOW {window_hours}h COMPLETE ✓")

    print("\n" + "=" * 60)
    print("  ALL WINDOWS COMPLETE")
    print("=" * 60)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
