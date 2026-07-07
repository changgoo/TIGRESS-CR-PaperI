# paperII figure generation: notebook → CLI refactor

Date: 2026-07-07

## Goal

Move the model-setup + `plotting_scripts` orchestration currently duplicated
across `notebooks/paperII_figures_{s28,s29,vAi,fcr,b}.ipynb` into
`python/paperII.py` so the paper-II figures can be regenerated from the shell:

```
python paperII.py --group=s28
python paperII.py --group=s28 --num=4
python paperII.py --group=all
```

The notebooks stay on disk (they remain the exploratory surface); after the
refactor a notebook is a two-liner:

```python
from paperII import run_group
run_group("s28")
```

## Non-goals

- Modifying `plotting_scripts.py`, `cr_zprof.py`, `load_sim_tigresspp.py`, or
  any other module. This is purely an orchestration refactor of `paperII.py`.
- Changing the existing `paperII.load(model_dict, gr, tslice, colors, verbose)`
  function signature — it stays and is called from `run_group`.
- Reproducing every exploratory cell from the notebooks (see "dropped cells"
  below).

## Architecture

### `paperII.py` becomes three-layered

1. **Group config** — a module-level `GROUPS` dict, one entry per paper-II
   figure group. Each entry is a plain dict; no per-group hook functions.
2. **`run_group(group, num="all")`** — reads the config, builds `model_dict`
   from `cr_data_load`, calls the existing `load()`, filters `simgroup[group]`,
   applies color overrides, loads `zp_pp` for CR sims, calls `ps.setup(...)`,
   dispatches to `draw_figures`.
3. **`draw_figures(simgroup, group, cfg, num="all")`** — numbered
   `if num == "all" or num == N:` blocks mirroring the `paperI.py`
   convention. Each block corresponds to one figure or a tight cluster
   (e.g. pressure_t + pressure_z + vertical_equilibrium_t all in block 2).

### Group config schema

```python
GROUPS = {
    "<name>": dict(
        patterns=[<glob>, <glob>, ...],   # concatenated via cr_data_load
        colors=[<matplotlib color>, ...], # passed to load(..., colors=)
        tslice=slice(<start>, <stop>),    # analysis window in Myr
        tmin=<int>, tmax=<int>,           # plot_history x-limits
        filter=lambda m: <bool>,          # selects simgroup[group] from ["all"]
        color_overrides={<model_key>: <color>, ...},  # optional
    ),
    ...
}
```

Concrete values (transcribed from the current notebooks):

| group | patterns                                              | colors                                                       | tslice        | tmin | tmax | filter               |
|-------|-------------------------------------------------------|--------------------------------------------------------------|---------------|------|------|----------------------|
| s28   | `*b1-*Vmax2-rst`, `*sigma28*`                         | `#E77500`, `xkcd:pink`, `xkcd:orchid`, `xkcd:coral`          | `(270, 500)`  | 200  | 500  | `m != "crmhd"`       |
| s29   | `*b1-*Vmax2-rst`, `*sigma29*`                         | `#E77500`, `xkcd:aqua`, `xkcd:azure`, `xkcd:indigo`          | `(270, 500)`  | 200  | 500  | `m != "crmhd"`       |
| vAi   | `*b1-*Vmax2-rst`, `*va1`                              | `#E77500`, `xkcd:teal`, `xkcd:coral`, `xkcd:indigo`          | `(270, 500)`  | 200  | 500  | `m != "crmhd"`       |
| fc    | `*b1-*Vmax2-rst`, `*-fc*`                             | `#E77500`, `xkcd:crimson`, `xkcd:magenta`, `xkcd:blue`       | `(200, 500)`  | 0    | 600  | `"fc" in m`          |
| beta  | `*b1-*Vmax2-rst`, `*b0.1-*Vmax2`, `*b10-*Vmax2-rst2`  | `#E77500`, `xkcd:olive`, `xkcd:gold`                         | `(200, 500)`  | 0    | 700  | `m != "crmhd"`       |

Only `s28` has color_overrides:
`{"sigma28-nost": "xkcd:pink", "sigma28-vAtot": "xkcd:orchid"}`.

### `draw_figures` block layout

Numbered blocks the CLI's `--num` flag targets:

| num | Block contents                                                                                                        |
|-----|-----------------------------------------------------------------------------------------------------------------------|
| 1   | `plot_history(simgroup, "all", tmin=cfg["tmin"], tmax=cfg["tmax"])`                                                   |
| 2   | `plot_pressure_t(..., "all", zslice=slice(-50,50), tmax=500)`, `plot_pressure_z(..., "all")`, `plot_vertical_equilibrium_t(..., group, tmin=cfg["tmin"], zmax=1000)` |
| 3   | `plot_area_mass_fraction_z(simgroup, group)`                                                                          |
| 4   | `plot_cr_velocity_z(simgroup, group, both=True)`, `plot_cr_velocity_z_all(simgroup, "all")`                            |
| 5   | `plot_kappa_z(simgroup, "all")` (default phases) + `plot_kappa_z(..., phases=[["CNM","UNM"], "WNM", ["WHIM","HIM"]])` |
| 6   | Loop over `simgroup[group].items()`: `plot_gainloss_z_each(s, m, phases=[["CNM","UNM","WNM"], "WHIM", "HIM"])`        |
| 7   | `plot_flux_tz(simgroup, group)` with `axes[0].set_xlim(cfg["tmin"], cfg["tmax"])`                                     |
| 8   | `plot_flux_z(simgroup, "all", vz_dir=1, both=True)`                                                                   |
| 9   | `plot_loading_z_merged(simgroup, "all", vz_dir=None, both=True)`                                                      |
| 10  | `plot_momentum_transfer_z(simgroup, "all", show_option=1, zmin=0, zref=1000)`                                         |
| 11  | Wind & joint PDFs: for each `m,s in simgroup[group]`: `load_windpdf(s, both=True)`; then `plot_jointpdf(simgroup, group)`, `plot_jointpdf(simgroup, group, flux="eflux")`, `plot_voutpdf(simgroup, group)` |

Blocks 1–10 come from the s28/s29/vAi/fc/beta notebooks (which agree on the
sequence). Block 11 runs unconditionally per user's instruction ("just run
them without assuming they will fail").

`num="all"` runs every block; `num=<int>` runs just that block.

### CLI

`argparse` with two flags:

- `--group`: one of `s28 | s29 | vAi | fc | beta | all`. Required.
- `--num`: figure block number, integer, or `all`. Default `all`.

`--group=all` iterates `for g in GROUPS: run_group(g, num=num)`.

The existing broken `if __name__ == "__main__": load()` block is removed and
replaced with the argparse dispatcher.

## Cells dropped from the notebooks

Explicitly not carried over to the CLI:

- `beta`: the manual `b10` history merge that reads a second sim
  `crmhd-8pc-b10-mhdbc_diode-crbc_lngrad_out-sigma_selfc-Vmax2` and calls
  `s.merge_hst(hold=h0)`. This is a data-provenance fixup, not a figure step;
  it stays in the notebook. **Per user decision (option a).**
- `fc`: exploratory boxplots of `hst["nfofc"]`, `hst["nbad_d"]`, `hst["nbad_p"]`,
  and the `load_hdf5(num=21, outid=7)` inspection cell. These are QA plots,
  not paper figures.
- `s28`: the `print(m, hasattr(s, "zp_pp_ph"))` debug line after
  `load_zprof_postproc`. `zp_pp` loading itself is kept but unified to
  cosmic-ray sims only (`if s.options["cosmic_ray"]`), matching the other
  notebooks.

## Consequences for downstream files

- Notebooks continue to work with `from paperII import load`. They can also
  now `from paperII import run_group` to drive figure generation from one
  cell.
- The `python/` module set is unchanged. No new files. `paperII.py` grows
  by ~150–200 lines.

## Testing / verification

No automated test suite exists in this repo. Verification is manual:

1. `python paperII.py --group=s28 --num=1` produces `<repo>/s28_figures/…`
   files that match the current notebook output (spot-check one figure).
2. `python paperII.py --group=all` runs all 5 groups to completion without
   exception.
3. Notebooks still import `load` and `run_group` without breakage.

## Rollout

Single commit that:
- Replaces the current `paperII.py` (keeping the existing `load` function
  intact) with the new orchestration layer.
- Adds a note to `CLAUDE.md`'s "Running / common commands" section pointing
  at `python paperII.py --group=<name>`.
