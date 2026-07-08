# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Analysis and figure-generation code for the TIGRESS-CR simulations (cosmic-ray MHD, star-forming ISM boxes run in Athena++). It is a **research/plotting toolkit**, not a simulation code — the raw HDF5/history/zprof outputs live outside the repo (default: `/scratch/gpfs/changgoo/tigress_classic/` on Princeton `stellar`) and are consumed here.

## Dependency: pyathena

Everything hinges on [`pyathena`](https://github.com/jeonggyukim/pyathena). The known-good pin is commit `87af62bcc25b7822d31b8e199e88d243d31f11b5`. Follow the README to create the `pyathena` conda env, then `pip install .` inside it (optionally checking out that commit first). Activate `pyathena` before running anything here.

## Running / common commands

There is no build or test suite. Typical workflows:

- Regenerate paper figures (from `python/`):
  ```sh
  cd python
  python paperI.py                           # all figures
  python paperI.py 3                         # only figure group 3 (snapshots)
  python paperII.py --group=s28              # regenerate s28 figures
  python paperII.py --group=all --num=1      # only block 1 (history) for every group
  ```
  `paperI.py` and `paperII.py` are the shell entry points. Group choices for
  `--group`: `s28 | s29 | vAi | fc | beta | all`. Block numbers 1–11 map to
  the figure sequence in `paperII.py::draw_figures`.

- Post-process z-profiles in parallel (per-snapshot netCDF cache under `<basedir>/zprof_postproc/`):
  ```sh
  mpirun -n <N> python python/process_zprof.py <sim_basedir>
  ```

- Sync generated figures to the public web host:
  ```sh
  ./update_figures.sh        # rsyncs *_figures/ dirs into /tigress/changgoo/public_html/TIGRESS-CR/
  ```

- Lint / format (matches the pre-commit config, `ruff` v0.14.9):
  ```sh
  ruff check --fix .
  ruff format .
  ```
  Ignored rules (see `.ruff.toml`): F841, E741, E402, E731, F401. Do **not** try to "clean up" unused imports or module-level imports below other statements — they're intentional.

## Architecture

### The `LoadSimTIGRESSPP` god-class (mixin composition)

`python/load_sim_tigresspp.py` defines the single object that everything downstream instantiates:

```python
class LoadSimTIGRESSPP(LoadSim, Hst, Timing, Zprof, SliceProj, PDF, PostProcessingZprof):
    ...
```

Each mixin is a sibling file in `python/` and contributes one slice of behavior:

- `hst.Hst` — history file reader (`read_hst`), unit conversions.
- `zprof.Zprof` — reads `.zprof` CSVs into an xarray `Dataset` split by phase (`CNM/UNM/WNM/WHIM/HIM`) and optionally by `vz_dir`. Also owns `load_zprof_postproc[_one]` which caches per-snapshot netCDFs under `savdir/zprof_postproc/`.
- `slc_prj.SliceProj` — slice/projection extraction from HDF5, cached via pyathena decorators.
- `pdf.PDF` — 2D joint PDF builders.
- `timing.Timing` — `.task_time.txt` parsing.
- `postproc_zprof.PostProcessingZprof` — builds the on-the-fly z-profile from a full snapshot (used by `Zprof.load_zprof_postproc_one`). Contains B-field rotation utilities (`get_b_angle`, `rotate_vector`) used to project CR fluxes/stresses onto the field-aligned frame.

Constructor side effects (all in `__init__`):
- `check_configure_options()` inspects `self.par` and sets `self.options` (bools like `mhd`, `newcool`, `cosmic_ray`, `wind`, `xray`, `feedback_scalars`). Downstream code branches on `self.options[...]` — always check there rather than re-parsing `par`.
- Loads cooling/heating tables (`cool_ftn.runtime.csv`), popsynth (`pop_synth.runtime.csv`), and external gravity (`extgrav.runtime.csv`) from `basedir` if present.
- Registers derived fields via `fields.add_fields` and then customizes cmap/norm per-field in `update_derived_fields()`. `self.dfi` is the derived-field-info dict — plotters read `dfi[f]["imshow_args"]`, `["label_name"]`, `["label_unit"]` etc.
- Builds `self.cpp_to_cc` mapping Athena++ primitive names (`rho`, `press`, `vel1`, `Bcc1`, `rHI`, ...) to pyathena "classic" names (`density`, `pressure`, `velocity1`, `cell_centered_B1`, `xHI`, ...). `get_data(..., load_derived=True)` performs this rename and then evaluates derived fields.

`LoadSimTIGRESSPPAll` (same file) manages many sims: pass a `{model_key: basedir}` dict, then `sa.set_model(key)` lazily constructs (and caches in `simdict`) the corresponding `LoadSimTIGRESSPP`.

### Data flow

```
Athena++ HDF5 / hst / zprof   →   LoadSimTIGRESSPP.{get_data, read_hst, load_zprof, get_slice, ...}
                                        │
                                        ├── xarray Datasets with derived fields
                                        │
                                        ▼
                       plotting_scripts.py  ─────  plot_slices.py
                                        │              │
                                        ▼              ▼
                                   *_figures/*.pdf, *.png
```

Nearly every plotting function in `plotting_scripts.py` (~4400 lines) takes `simgroup` (a `{group_name: {model_key: LoadSimTIGRESSPP}}` dict) and a `group` string. The two globals to know: `plotting_scripts.setup(outdir, model_name_dict, model_color_dict)` must be called before drawing so figures land in the right dir with consistent colors/labels; after that `ps.fig_outdir`, `ps.model_name`, `ps.model_color` are read by every plot function. The `paper.mplstyle` in `python/` is auto-applied on import of `plotting_scripts`.

### Derived CR/MHD fields

`fields.set_derived_fields_user(par)` in `python/fields.py` adds the CR-specific derived fields the base pyathena registry doesn't know about — `Vtotz` (advection + streaming), `0-Veff3` (effective flux velocity, uses `par["cr"]["vmax"]`), `kappa_para` (parallel diffusion coefficient from `0-Sigma_diff1`). Add new CR fields here rather than sprinkling them across plotting scripts.

### Model naming conventions

Model keys are the sim directory basenames (e.g. `crmhd-8pc-b1-diode-lngrad_out-sigma_selfc-Vmax2-rst`). Human-readable names, colors, and edge colors are dictionaries — `paperI.py` defines them inline, `cr_zprof.py` holds a bigger registry (`model_name`, `model_color`, `model_default`, `model_beta`, `model_sigma`, ...). When adding a new sim, extend those dicts rather than hard-coding strings in a plot function.

## Notebooks

`notebooks/` is the exploratory surface — most are per-paper (`paperII_figures_*.ipynb`) or per-investigation (`inspect_badcells.ipynb`, `inspect_gradP.ipynb`, ...). Committed notebooks contain outputs; the `end-of-file-fixer` and `trailing-whitespace` pre-commit hooks will touch them on commit. Untracked `Untitled*.ipynb` are throwaway.

## Figure output directories

The many `*_figures/` directories at the repo root are outputs, not sources. They are `.gitignore`d for `*.pdf` and `*.png` but the directories themselves are tracked. Which script writes where is set by the `outdir` argument in each script's `setup()` call (e.g. `paperI.py` writes to `../figures-new`; the appendix pass writes to `../fig_bcs`).
