# OHCA Vitals Trajectory Pipeline

Vitals trajectories in out-of-hospital cardiac arrest (OHCA) patients admitted to the ICU.

This repository is the canonical cross-site temperature analysis pipeline for the OHCA project.
Downstream manuscript-specific analyses, pooled figure builders, and exploratory extensions
should live in a separate fork or downstream repo.

## File Structure

```
OHCA-vitals_trajectory/
├── pyproject.toml              # Dependencies
├── config.json                 # Local run config (template tracked here)
├── config.example.json         # Example config for new sites
├── pipeline_helpers.py         # Constants, logger, config builder
├── pipeline_steps.py           # All pipeline step functions
├── run_pipeline.py             # Marimo notebook (entry point)
├── OHCA Multi-Site Consolidation.ipynb
├── downstream/                 # Non-canonical extensions and manuscript work
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

### 3. Configure `config.json`

Start from `config.example.json` and edit `config.json` for your local site environment.

```json
{
    "data_directory": "path/to/your/clif/data",
    "output_directory": "output_dir",
    "file_format": "parquet",
    "site_name": "YourSite",
    "timezone": "US/Eastern"
}
```

| Field | Description |
|-------|-------------|
| `data_directory` | Path to folder containing CLIF tables (`clif_hospitalization.parquet`, `clif_vitals.parquet`, etc.) |
| `output_directory` | Path for intermediate output files (for example `output_dir`) |
| `file_format` | `parquet` or `csv` |
| `site_name` | Your site name (e.g., `Emory`, `UCSF`) — used in poolable Table 1 |
| `timezone` | Timezone for timestamps (default: `US/Eastern`) |

Local sites may customize `config.json`, but site-specific config values should not be treated as shared analysis logic.

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

## Upload to Box

After the pipeline completes, upload both output folders to the shared Box folder:

1. Navigate to the shared Box folder: **OHCA Vitals Trajectory → Site Results**
2. Create a folder with your site name (e.g., `Emory/`)
3. Upload the two output folders into it:
   - `Upload_to_Box_without_oral_24/`
   - `Upload_to_Box_without_oral_72/`

> **Note:** Do NOT upload the `intermediate_without_oral_{24,72}/` folders — those contain raw patient-level data and are for local use only.

## Downstream Work

If you are adding pooled-result summaries, manuscript figures, or exploratory methods,
prefer a downstream fork/repo so this repository remains the shared cross-site source of truth.
This branch also stages legacy downstream materials under `downstream/` to keep the canonical
root focused on the shared pipeline. See `DOWNSTREAM_WORK.md`.
