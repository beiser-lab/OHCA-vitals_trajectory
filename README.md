# OHCA Vitals Trajectory Pipeline

Vitals trajectories in out-of-hospital cardiac arrest (OHCA) patients admitted to the ICU, with a parallel blood glucose analysis package from CLIF labs.

## File Structure

```
OHCA-vitals_trajectory/
├── pyproject.toml              # Dependencies
├── config.json                 # Site-specific config
├── pipeline_helpers.py         # Constants, logger, config builder
├── pipeline_steps.py           # All pipeline step functions
├── run_pipeline.py             # Marimo notebook (entry point)
└── README.md
```

## Setup

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Sync dependencies

```bash
uv sync
```

### 3. Create your `config.json`

```json
{
    "data_directory": "path/to/your/clif/data",
    "output_directory": "path/to/output",
    "file_format": "parquet",
    "site_name": "auto",
    "timezone": "US/Eastern"
}
```

| Field | Description |
|-------|-------------|
| `data_directory` | Path to folder containing CLIF tables (`clif_hospitalization.parquet`, `clif_vitals.parquet`, `clif_labs.parquet`, etc.) |
| `output_directory` | Path for intermediate output files |
| `file_format` | `parquet` or `csv` |
| `site_name` | Optional site label override for poolable outputs. Use `"auto"` (recommended) to infer from `clif_adt.hospital_id`. |
| `timezone` | Timezone for timestamps (default: `US/Eastern`) |

## Run the pipeline

```bash
uv run marimo edit run_pipeline.py
```

This opens the notebook in your browser. Cells run top to bottom:

1. **Cohort building** — window-independent, runs once
2. **Vitals extraction, plots, trajectory assignment, Table 1** — loops over 24h and 72h windows

## Outputs

Results are saved to three folders per window:

- `Upload_to_Box_without_oral_{24,72}/` (primary vitals + temperature trajectory outputs)
- `Upload_to_Box_without_oral_glucose_{24,72}/` (parallel blood glucose outputs)
- `Upload_to_Box_without_oral_lactate_{24,72}/` (parallel blood lactate outputs)

Primary folder includes:

- All figures (`.png`)
- Hourly vitals by trajectory × survival (`.csv`, `.parquet`)
- Table 1 poolable (`.csv`) — long-format for multi-site aggregation
- Table 1 summary (`.txt`, `.csv`)
- Pipeline log (`.txt`)

Glucose folder includes:

- Blood glucose blocked + epoch-smoothed plots (`.png`)
- Blood glucose by category/survival plots (`.png`, categories `A-D`)
- 6-hour epoch blood glucose by trajectory × survival (`.csv`, `.parquet`)
- Glucose pipeline log (`.txt`)

Lactate folder includes:

- Blood lactate blocked + epoch-smoothed plots (`.png`)
- Blood lactate by trajectory/survival plots (`.png`)
- 6-hour epoch blood lactate by trajectory × survival (`.csv`, `.parquet`)
- Lactate pipeline log (`.txt`)

## Upload to Box

After the pipeline completes, upload the output folders to the shared Box folder:

1. Navigate to the shared Box folder: **OHCA Vitals Trajectory → Site Results**
2. Create a folder with your site name (e.g., `Emory/`)
3. Upload all six output folders into it:
   - `Upload_to_Box_without_oral_24/`
   - `Upload_to_Box_without_oral_72/`
   - `Upload_to_Box_without_oral_glucose_24/`
   - `Upload_to_Box_without_oral_glucose_72/`
   - `Upload_to_Box_without_oral_lactate_24/`
   - `Upload_to_Box_without_oral_lactate_72/`

> **Note:** Do NOT upload the `intermediate_without_oral_{24,72}/` folders — those contain raw patient-level data and are for local use only.
