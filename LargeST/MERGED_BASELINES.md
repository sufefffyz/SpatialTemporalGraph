## Imported Baselines

This repository now contains two upstream source snapshots imported without their
original `.git` history:

- `../BiST`
- `../PatchSTG`

To keep the current `LargeST` codebase stable while still making these methods
available under the same umbrella, the merge strategy is:

- `BiST` is partially integrated into `LargeST` as a native experiment:
  - `experiments/bist/`
  - `src/models/bist.py`
  - `src/engines/bist_engine.py`
  - `src/utils/bist_args.py`
  - `src/utils/bist_dataloader.py`
- `PatchSTG` is vendored as a self-contained experiment snapshot under:
  - `experiments/patchstg/`

The original source snapshots are also preserved at the project root so the
upstream layout remains inspectable:

- `/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BiST`
- `/Users/richardo/Desktop/STproject/SpatialTemporalGraph/PatchSTG`

This avoids importing upstream commit/contributor history while keeping a
practical merged layout for future adaptation.
