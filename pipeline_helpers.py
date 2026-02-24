"""
pipeline_helpers.py — Config, logger, constants, small utilities.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import duckdb
import warnings
warnings.filterwarnings("ignore")

# =============================================================
# CONSTANTS
# =============================================================
ICD_PREFIXES = ["I46.0", "I46.1", "I46.2", "I46.8", "I46.9", "I49.00", "I49.01"]
ICD_DESCRIPTIONS = {
    "I46.0": "Cardiac arrest with successful resuscitation",
    "I46.1": "Sudden cardiac death, so described",
    "I46.2": "Cardiac arrest due to underlying cardiac condition",
    "I46.8": "Cardiac arrest due to other underlying condition",
    "I46.9": "Cardiac arrest, cause unspecified",
    "I49.00": "Ventricular fibrillation",
    "I49.01": "Ventricular flutter",
}
CLIF_VITAL_SIGNS = ["heart_rate", "temp_c", "spo2", "map"]
BLOOD_GLUCOSE_LAB_CATEGORIES = ["glucose_fingerstick", "glucose_serum", "glucose_mixed_venous"]
BLOOD_LACTATE_LAB_CATEGORIES = ["lactate"]
VITAL_SIGNS = [*CLIF_VITAL_SIGNS]
BLOCK_SIZE = 4
GLUCOSE_EPOCH_HOURS = 6
LACTATE_EPOCH_HOURS = 6

TRAJ_COLORS = {
    "Hypothermic": "#1565C0", "Normothermic": "#43A047",
    "Rapid Decline": "#FF8F00", "Persistent High": "#C62828",
}
SURV_COLORS = {"Survivor": "#2196F3", "Non-Survivor": "#E53935"}
TRAJ_ORDER = ["Persistent High", "Rapid Decline", "Normothermic", "Hypothermic"]
GLUCOSE_TRAJ_LETTERS = {
    "Persistent High": "A",
    "Rapid Decline": "B",
    "Normothermic": "C",
    "Hypothermic": "D",
}

VITALS_INFO = {
    "mean_heart_rate": {"title": "Heart Rate (bpm)", "ylabel": "Heart Rate (bpm)"},
    "mean_temp_c":     {"title": "Temperature (°C)", "ylabel": "Temperature (°C)"},
    "mean_map":        {"title": "Mean Arterial Pressure (mmHg)", "ylabel": "MAP (mmHg)"},
    "mean_spo2":       {"title": "Oxygen Saturation (%)", "ylabel": "SpO2 (%)"},
}
VITALS_HOURLY_INFO = {
    "heart_rate": {"title": "Heart Rate (bpm)", "ylabel": "Heart Rate (bpm)"},
    "temp_c":     {"title": "Temperature (°C)", "ylabel": "Temperature (°C)"},
    "map":        {"title": "Mean Arterial Pressure (mmHg)", "ylabel": "MAP (mmHg)"},
    "spo2":       {"title": "Oxygen Saturation (%)", "ylabel": "SpO2 (%)"},
}


# =============================================================
# LOGGER
# =============================================================
class Logger:
    def __init__(self, path):
        self.path = path
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"Pipeline log — {datetime.now()}\n{'='*60}\n")

    def info(self, msg=""):
        print(msg)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# =============================================================
# CONFIG
# =============================================================
def make_config(config_path="config.json", window_hours=72):
    """Build config dict for a given window."""
    with open(config_path, "r") as f:
        raw = json.load(f)
    data_dir = Path(raw["data_directory"])
    file_format = raw.get("file_format", "parquet")
    site_name, site_source = resolve_site_name(
        data_dir=data_dir,
        file_format=file_format,
        configured_site_name=raw.get("site_name"),
    )
    config = {
        "data_dir": data_dir,
        "output_dir": Path(raw["output_directory"]),
        "intermediate_dir": Path(raw["output_directory"]) / f"intermediate_without_oral_{window_hours}",
        "upload_dir": Path(f"Upload_to_Box_without_oral_{window_hours}"),
        "glucose_upload_dir": Path(f"Upload_to_Box_without_oral_glucose_{window_hours}"),
        "lactate_upload_dir": Path(f"Upload_to_Box_without_oral_lactate_{window_hours}"),
        "timezone": raw.get("timezone", "US/Eastern"),
        "file_format": file_format,
        "site_name": site_name,
        "site_name_source": site_source,
        "window_hours": window_hours,
        "block_size": BLOCK_SIZE,
        "n_blocks": window_hours // BLOCK_SIZE,
        "glucose_epoch_hours": GLUCOSE_EPOCH_HOURS,
        "lactate_epoch_hours": LACTATE_EPOCH_HOURS,
    }
    config["intermediate_dir"].mkdir(parents=True, exist_ok=True)
    config["upload_dir"].mkdir(parents=True, exist_ok=True)
    config["glucose_upload_dir"].mkdir(parents=True, exist_ok=True)
    config["lactate_upload_dir"].mkdir(parents=True, exist_ok=True)
    return config


# =============================================================
# SMALL HELPERS
# =============================================================
def read_table(config, table_name):
    ext = config["file_format"]
    path = config["data_dir"] / f"{table_name}.{ext}"
    if ext == "parquet":
        return f"read_parquet('{path}')"
    elif ext == "csv":
        return f"read_csv_auto('{path}')"
    else:
        raise ValueError(f"Unknown format: {ext}")


def resolve_site_name(data_dir, file_format, configured_site_name=None):
    """
    Resolve site identifier for poolable outputs.
    Priority:
      1) Explicit config site_name when not empty/auto.
      2) Auto-detect from clif_adt.hospital_id.
      3) Fallback to Unknown.
    """
    explicit = (str(configured_site_name).strip() if configured_site_name is not None else "")
    if explicit and explicit.lower() not in {"auto", "from_data"}:
        return explicit, "config.site_name"

    try:
        ext = file_format
        adt_path = Path(data_dir) / f"clif_adt.{ext}"
        if not adt_path.exists():
            return "Unknown", "fallback.no_clif_adt"

        if ext == "parquet":
            qry = f"""
                SELECT DISTINCT CAST(hospital_id AS VARCHAR) AS hospital_id
                FROM read_parquet('{adt_path}')
                WHERE hospital_id IS NOT NULL
                ORDER BY 1
            """
        elif ext == "csv":
            qry = f"""
                SELECT DISTINCT CAST(hospital_id AS VARCHAR) AS hospital_id
                FROM read_csv_auto('{adt_path}')
                WHERE hospital_id IS NOT NULL
                ORDER BY 1
            """
        else:
            return "Unknown", f"fallback.unknown_format.{ext}"

        ids = duckdb.sql(qry).fetchdf()["hospital_id"].dropna().astype(str).tolist()
        ids = [v.strip() for v in ids if v.strip()]
        if len(ids) == 1:
            return ids[0], "clif_adt.hospital_id"
        if len(ids) > 1:
            return "MULTISITE", "clif_adt.hospital_id.multiple"
        return "Unknown", "fallback.empty_hospital_id"
    except Exception:
        return "Unknown", "fallback.exception"


def build_icd_filter(column="diagnosis_code"):
    # Match dotted and undotted ICD formats by normalizing away periods.
    normalized_column = f"REPLACE(UPPER(CAST({column} AS VARCHAR)), '.', '')"
    conditions = " OR ".join(
        f"{normalized_column} LIKE '{code.replace('.', '')}%'"
        for code in ICD_PREFIXES
    )
    return f"({conditions})"


def map_description(code):
    code_str = str(code)
    norm_code = code_str.replace(".", "").upper()
    for prefix, desc in ICD_DESCRIPTIONS.items():
        if norm_code.startswith(prefix.replace(".", "").upper()):
            return desc
    return "Unknown"


def vitals_filter_sql():
    return ", ".join(f"'{v}'" for v in CLIF_VITAL_SIGNS)


def glucose_labs_filter_sql():
    return ", ".join(f"'{v}'" for v in BLOOD_GLUCOSE_LAB_CATEGORIES)


def lactate_labs_filter_sql():
    return ", ".join(f"'{v}'" for v in BLOOD_LACTATE_LAB_CATEGORIES)


def block_plot_params(config):
    n_blocks = config["n_blocks"]
    bs = config["block_size"]
    tick_pos = [b * bs + bs / 2 for b in range(n_blocks)]
    tick_labels = [f"{b*bs}-{(b+1)*bs}" for b in range(n_blocks)]
    xlim = (0.5, config["window_hours"] + 0.5)
    return tick_pos, tick_labels, xlim


def hourly_plot_params(config):
    wh = config["window_hours"]
    step = 2 if wh <= 24 else 4
    return (0, wh), range(0, wh + 1, step)


def glucose_traj_letter(trajectory_name):
    return GLUCOSE_TRAJ_LETTERS.get(trajectory_name, "U")


def glucose_traj_label(trajectory_name):
    return f"Category {glucose_traj_letter(trajectory_name)}"
