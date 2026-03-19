# Downstream Work

This repository is intended to remain the shared cross-site OHCA temperature pipeline.

Legacy downstream materials in this branch have been moved under:

- `downstream/scripts/`
- `downstream/abstracts/`
- `downstream/patches/`

Examples of work that should usually live in a downstream fork or companion repo:
- pooled figure packs for abstracts or manuscripts
- manuscript drafts and submission materials
- site-aggregated summary notebooks
- exploratory clustering or phenotype extensions
- glucose, lactate, GCS, or other analysis branches not required for the shared temperature pipeline
- one-off helper scripts for a single project milestone

Good reasons to promote a downstream change back upstream:
- it is needed by multiple sites
- it improves the shared temperature pipeline itself
- it reduces configuration burden across sites
- it fixes a true bug in the canonical workflow

When in doubt:
- keep the cross-site pipeline here
- keep manuscript- or project-specific extensions downstream
