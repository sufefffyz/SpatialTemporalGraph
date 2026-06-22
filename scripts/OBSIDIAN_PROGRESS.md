# Obsidian Progress Logging

This script appends project progress updates to a Markdown note inside an Obsidian vault.

## File

- `scripts/obsidian_progress.py`

## What it does

- Creates a note if it does not already exist.
- Appends a timestamped progress entry.
- Optionally captures the current git branch, commit, and worktree status.
- Works with plain Python 3 and does not require extra packages.

## Recommended note path

By default the script writes to:

```text
Projects/SpatialTemporalGraph/Progress.md
```

inside the vault root.

## Usage

Run from the `SpatialTemporalGraph` repository root:

```bash
python3 scripts/obsidian_progress.py \
  --vault "/absolute/path/to/Obsidian Vault" \
  --summary "Updated dataset loading and scaler handling." \
  --file BasicTS/stgraph_ext/dataset.py \
  --file BasicTS/stgraph_ext/scaler.py \
  --next-step "Run the stgraph_ext tests."
```

You can also set the vault path once in your shell:

```bash
export OBSIDIAN_VAULT_PATH="/absolute/path/to/Obsidian Vault"
python3 scripts/obsidian_progress.py \
  --summary "Finished a notebook cleanup." \
  --file BasicTS/scripts/data_visualization/distribution_visualization_stgraph.ipynb
```

## Notes

- Use `--note` if you want a different note path inside the vault.
- Use `--skip-git` if you do not want branch and worktree information in the entry.
- Use `--dry-run` to preview the generated entry without writing anything.
