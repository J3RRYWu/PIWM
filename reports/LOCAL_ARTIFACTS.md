# Local experiment artifacts

The repository contains source code, manuscript revisions, protocols, aggregate metrics and audit reports. Git push is NOT a full backup of the experiments.

- Raw/prepared DonkeyCar recordings live outside this repository.
- `checkpoints/` and all model weight files remain local under the existing ignore rules.
- Generated NumPy arrays under `reports/` remain local. Their size and SHA256 fingerprints at the pre-training push are recorded in `local_array_manifest.json`.
- Page rasterizations under `reports/manuscript_audit/render/` are local QA previews.

Frozen runners verify source, input weights and data fingerprints. Reproduction therefore requires the original local data and checkpoint artifacts in addition to the Git checkout. The manifest is an inventory, not an archive. No local arrays or weights were deleted.

The following matched current-state/history experiment is a new exploratory protocol and will preserve all existing studies.
