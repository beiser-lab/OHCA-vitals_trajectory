# OHCA Vitals Trajectory Pipeline
Vitals trajectories in out-of-hospital cardiac arrest (OHCA) patients admitted to the ICU. 
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
    "site_name": "YourSite",
    "timezone": "US/Eastern"
}
```

| Field | Description |
|-------|-------------|
| `data_directory` | Path to folder containing CLIF tables (`clif_hospitalization.parquet`, `clif_vitals.parquet`, etc.) |
| `output_directory` | Path for intermediate output files |
| `file_format` | `parquet` or `csv` |
| `site_name` | Your site name (e.g., `Emory`, `UCSF`) — used in poolable Table 1 |
| `timezone` | Timezone for timestamps (default: `US/Eastern`) |

## Run the pipeline

```bash
uv run marimo edit run_pipeline.py
```

This opens the notebook in your browser. Cells run top to bottom:

1. **Cohort building** — window-independent, runs once
2. **Vitals extraction, plots, trajectory assignment, Table 1** — loops over 24h and 72h windows

## Outputs

Results are saved to `Upload_to_Box_without_oral_{24,72}/`:

- All figures (`.png`)
- Hourly vitals by trajectory × survival (`.csv`, `.parquet`)
- Table 1 poolable (`.csv`) — long-format for multi-site aggregation
- Table 1 summary (`.txt`, `.csv`)
- Pipeline log (`.txt`)

