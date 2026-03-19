"""
pipeline_helpers.py — Config, logger, constants, small utilities.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
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
VITAL_SIGNS = ["heart_rate", "temp_c", "spo2", "map"]
BLOCK_SIZE = 4

TRAJ_COLORS = {
    "Hypothermic": "#1565C0", "Normothermic": "#43A047",
    "Rapid Decline": "#FF8F00", "Persistent High": "#C62828",
}
SURV_COLORS = {"Survivor": "#2196F3", "Non-Survivor": "#E53935"}
TRAJ_ORDER = ["Persistent High", "Rapid Decline", "Normothermic", "Hypothermic"]

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
    config = {
        "data_dir": Path(raw["data_directory"]),
        "output_dir": Path(raw["output_directory"]),
        "intermediate_dir": Path(raw["output_directory"]) / f"intermediate_without_oral_{window_hours}",
        "upload_dir": Path(f"Upload_to_Box_without_oral_{window_hours}"),
        "timezone": raw.get("timezone", "US/Eastern"),
        "file_format": raw.get("file_format", "parquet"),
        "site_name": raw.get("site_name", "Unknown"),
        "window_hours": window_hours,
        "block_size": BLOCK_SIZE,
        "n_blocks": window_hours // BLOCK_SIZE,
    }
    config["intermediate_dir"].mkdir(parents=True, exist_ok=True)
    config["upload_dir"].mkdir(parents=True, exist_ok=True)
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


def build_icd_filter(column="diagnosis_code"):
    """Build SQL filter for cardiac arrest ICD codes. Handles with/without dots, any case."""
    conditions = []
    for code in ICD_PREFIXES:
        code_upper = code.upper()
        code_nodot = code_upper.replace(".", "")
        # Match "I46.2%" and "I462%" etc.
        conditions.append(f"UPPER({column}) LIKE '{code_upper}%'")
        conditions.append(f"UPPER(REPLACE({column}, '.', '')) LIKE '{code_nodot}%'")
    return f"({' OR '.join(conditions)})"


def map_description(code):
    code_str = str(code).upper()
    code_nodot = code_str.replace(".", "")
    for prefix, desc in ICD_DESCRIPTIONS.items():
        prefix_upper = prefix.upper()
        prefix_nodot = prefix_upper.replace(".", "")
        if code_str.startswith(prefix_upper) or code_nodot.startswith(prefix_nodot):
            return desc
    return "Unknown"


def vitals_filter_sql():
    """Returns SQL fragment for vital_category filter, case-insensitive via LOWER()."""
    return ", ".join(f"'{v}'" for v in VITAL_SIGNS)


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


# Column values that should be normalized to Title Case after any query
_TITLE_CASE_COLS = ["sex_category", "race_category", "ethnicity_category", "discharge_category"]

def normalize_categories(df):
    """Normalize known categorical columns to Title Case for cross-site consistency."""
    for col in _TITLE_CASE_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df.loc[df[col].isin(["Nan", "None", ""]), col] = None
    return df