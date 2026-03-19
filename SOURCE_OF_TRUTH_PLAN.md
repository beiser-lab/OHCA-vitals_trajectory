# Source Of Truth Plan

## Goal

Establish the shared cross-site temperature analysis pipeline as the canonical upstream repo, and keep manuscript-specific, pooled-summary, or exploratory work in a downstream fork/repo.

## Recommended Canonical Repo

Use the existing git-backed repo at:

`/Users/davidbeiser/Documents/ OHCA-vitals_trajectory`

Reason:
- it already tracks `origin` at `Emory-Bhavani-Lab/OHCA-vitals_trajectory`
- it has commit history and can serve as the shared upstream
- the Downloads copy is a snapshot, not a git repo

## Current State

### Cross-site pipeline snapshot

Path:

`/Users/davidbeiser/Downloads/OHCA-vitals_trajectory-main`

Characteristics:
- leaner, temperature-focused cross-site pipeline
- not a git repository
- includes `OHCA Multi-Site Consolidation.ipynb`

### Current git-backed repo

Path:

`/Users/davidbeiser/Documents/ OHCA-vitals_trajectory`

Characteristics:
- connected to GitHub upstream
- includes added glucose/lactate pipeline work
- includes pooled-figure and exploratory scripts under `scripts/`
- currently has local modifications and untracked analysis files

## File Inventory

### Shared files that diverge

These should be reviewed explicitly during migration:
- `README.md`
- `pipeline_helpers.py`
- `pipeline_steps.py`
- `run_pipeline.py`
- `Grant-data-collection.ipynb`
- `config.json`
- `.gitignore`

### Files only in the Downloads snapshot

These likely belong in the canonical cross-site repo:
- `OHCA Multi-Site Consolidation.ipynb`

### Files only in the current git-backed repo

These are good downstream/fork candidates unless intentionally promoted:
- `scripts/`
- `auto-site-from-clif.patch`
- `erc_temperature_trajectory_abstract_draft.md`
- local `output_dir/`
- experimental cluster/GCS tooling
- pooled figure builders and manuscript-specific utilities

## What Should Define The Canonical Upstream

The canonical upstream should contain:
- the shared cross-site temperature pipeline
- reproducible setup instructions
- only generally useful notebooks and scripts
- no manuscript drafts
- no local outputs
- no site-specific config values

Concretely, the upstream baseline should come from the Downloads snapshot for:
- `README.md`
- `pipeline_helpers.py`
- `pipeline_steps.py`
- `run_pipeline.py`
- `Grant-data-collection.ipynb`
- `OHCA Multi-Site Consolidation.ipynb`

Keep from the current git-backed repo only when the change is broadly useful to all sites. The strongest example already identified is the broader output ignore rules in `.gitignore`.

## Recommended Downstream Scope

Move or preserve in a downstream fork/repo:
- glucose and lactate pipeline extensions
- pooled-result aggregation and figure-pack scripts
- cluster and GCS analyses
- abstract/manuscript drafting files
- one-off local patches

Suggested downstream repo name:

`OHCA-vitals_trajectory-downstream`

or

`OHCA-vitals_trajectory-extensions`

## Config Hygiene

Both current copies track a real `config.json`, which should not be the long-term pattern for the canonical repo.

Recommended target state:
- track `config.example.json`
- ignore `config.json`
- keep `README.md` instructions pointed at copying the example into a local config

## Migration Sequence

1. Preserve the current git-backed repo state on a safety branch.
2. Create a new branch for canonicalization.
3. Copy the shared pipeline files from the Downloads snapshot into that branch.
4. Re-apply only generic improvements that belong upstream.
5. Add `OHCA Multi-Site Consolidation.ipynb` to the canonical repo.
6. Remove or relocate downstream-only files from the canonical branch.
7. Replace tracked config guidance with `config.example.json` plus `config.json` ignore rules.
8. Add a short README scope statement naming this repo as the canonical cross-site pipeline.
9. Tag the first cleaned baseline for collaborators.

## Practical Branch Plan

In the git-backed repo:

- safety snapshot: `codex/pre-canonical-cleanup`
- working branch: `codex/canonicalize-cross-site`

If downstream work should remain easily accessible, create a separate repo from the current mixed state before cleanup.

## Immediate Next Edits

When ready to implement the migration in the git-backed repo:
- import the Downloads versions of the main pipeline files
- add `OHCA Multi-Site Consolidation.ipynb`
- add `config.example.json`
- update `.gitignore` to ignore local config and outputs
- add a README note such as: "This repository is the canonical cross-site OHCA temperature analysis pipeline. Downstream manuscript and exploratory analyses should live in a separate fork/repo."

## Notes

- The current git-backed repo is ahead of `origin/main` by one commit and also has local uncommitted changes, so cleanup should happen on a fresh branch rather than directly on `main`.
- The Downloads snapshot should be treated as the desired content baseline, not as the long-term working location.
