# paperII CLI refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the 5 `paperII_figures_{s28,s29,vAi,fcr,b}.ipynb` notebook workflows into a CLI in `python/paperII.py` so figures can be regenerated with `python paperII.py --group=<name>`.

**Architecture:** One module-level `GROUPS` config dict (patterns/colors/tslice/tmin/tmax/filter/color_overrides per group). `run_group(group, num="all")` builds `model_dict` via `cr_data_load`, calls the existing `load()`, filters `simgroup[group]`, applies color overrides, loads `zp_pp` for CR sims, calls `ps.setup(...)`, dispatches to `draw_figures(simgroup, group, cfg, num)`. `draw_figures` is a series of numbered `if num == "all" or num == N` blocks mirroring the notebook cell sequence and the `paperI.py` convention. An `argparse` `__main__` block exposes `--group` and `--num`.

**Tech Stack:** Python 3, argparse, pyathena (LoadSim), `plotting_scripts` module in this repo.

## Global Constraints

- The existing `python/paperII.py::load(model_dict, gr, tslice, colors, verbose)` function is **kept as-is** — the refactor only adds new callers on top.
- Do NOT remove existing module-level imports of `plot_snapshot_comp`, `plot_slices_cr`, `LogNorm`, `cmr`, `np`, `LoadSimTIGRESSPP` from `paperII.py`, even if unused after the refactor. `.ruff.toml` ignores F401 in this repo; other files may still `from paperII import ...` those names.
- Preserve `basedir = "/scratch/gpfs/changgoo/tigress_classic/"` as a module-level constant in `paperII.py` (notebooks read it via `paperII.basedir`).
- Figure output directory pattern must stay `f"../{group}_figures"` because `python/paperII.py` is run from `python/` (matches all notebooks and `paperI.py`).
- The 5 groups and their configs are exactly what the spec table lists — do not "clean up" glob patterns or color choices.
- `--num` accepts either an integer 1–11 or the string `"all"`.
- Group `all` iterates `for g in GROUPS: run_group(g, num=num)`.
- **No test suite exists in this repo.** Verification is manual per Task 2's checklist. Do NOT add pytest or a tests directory.

---

### Task 1: Rewrite `python/paperII.py` as a CLI driver

**Files:**
- Modify: `python/paperII.py` (full replacement of the existing 95-line file, but the `load()` function body is preserved verbatim)

**Interfaces:**
- Consumes:
  - `cr_zprof.cr_data_load(basedir, pattern) -> dict[str, str]`
  - `cr_zprof.load_windpdf(s, both=True)`
  - `plotting_scripts.setup(outdir, model_name, model_color)`
  - `plotting_scripts.plot_history(simgroup, group, tmin=..., tmax=...)`
  - `plotting_scripts.plot_pressure_t(simgroup, group, zslice=..., tmax=...)`
  - `plotting_scripts.plot_pressure_z(simgroup, group)`
  - `plotting_scripts.plot_vertical_equilibrium_t(simgroup, group, tmin=..., zmax=...)`
  - `plotting_scripts.plot_area_mass_fraction_z(simgroup, group)`
  - `plotting_scripts.plot_cr_velocity_z(simgroup, group, both=True)`
  - `plotting_scripts.plot_cr_velocity_z_all(simgroup, group)`
  - `plotting_scripts.plot_kappa_z(simgroup, group, phases=...)`
  - `plotting_scripts.plot_gainloss_z_each(s, m, phases=...)`
  - `plotting_scripts.plot_flux_tz(simgroup, group)` — returns a Figure whose `axes[0].set_xlim(a, b)` we call
  - `plotting_scripts.plot_flux_z(simgroup, group, vz_dir=1, both=True)`
  - `plotting_scripts.plot_loading_z_merged(simgroup, group, vz_dir=None, both=True)`
  - `plotting_scripts.plot_momentum_transfer_z(simgroup, group, show_option=1, zmin=0, zref=1000)`
  - `plotting_scripts.plot_jointpdf(simgroup, group, flux=...)`
  - `plotting_scripts.plot_voutpdf(simgroup, group)`
  - The `load()` function already in `paperII.py` — signature and body unchanged.
- Produces:
  - `paperII.basedir: str`
  - `paperII.GROUPS: dict[str, dict]`
  - `paperII.run_group(group: str, num: str | int = "all") -> None`
  - `paperII.draw_figures(simgroup: dict, group: str, cfg: dict, num: str | int = "all") -> None`
  - `paperII.load(...)` — unchanged, still importable.

- [ ] **Step 1: Read the current `paperII.py` in full**

Run: `cat python/paperII.py`

Purpose: capture the exact body of the existing `load(model_dict, gr, tslice=slice(200,500), colors=None, verbose=True)` function so it can be pasted back verbatim. It is roughly 60 lines and contains model renaming logic that must not change.

- [ ] **Step 2: Replace `python/paperII.py` with the new file**

```python
#!/usr/bin/env python3
"""Paper-II figure generation.

CLI:
    python paperII.py --group=s28
    python paperII.py --group=s28 --num=4
    python paperII.py --group=all

Notebook use:
    from paperII import run_group
    run_group("s28")
"""
import argparse
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmasher as cmr

from cr_zprof import cr_data_load, load_group, load_windpdf
from load_sim_tigresspp import LoadSimTIGRESSPP

import plotting_scripts as ps
from plot_slices import plot_snapshot_comp, plot_slices_cr


basedir = "/scratch/gpfs/changgoo/tigress_classic/"


GROUPS = {
    "s28": dict(
        patterns=["*b1-*Vmax2-rst", "*sigma28*"],
        colors=["#E77500", "xkcd:pink", "xkcd:orchid", "xkcd:coral"],
        tslice=slice(270, 500),
        tmin=200,
        tmax=500,
        filter=lambda m: m != "crmhd",
        color_overrides={
            "sigma28-nost": "xkcd:pink",
            "sigma28-vAtot": "xkcd:orchid",
        },
    ),
    "s29": dict(
        patterns=["*b1-*Vmax2-rst", "*sigma29*"],
        colors=["#E77500", "xkcd:aqua", "xkcd:azure", "xkcd:indigo"],
        tslice=slice(270, 500),
        tmin=200,
        tmax=500,
        filter=lambda m: m != "crmhd",
    ),
    "vAi": dict(
        patterns=["*b1-*Vmax2-rst", "*va1"],
        colors=["#E77500", "xkcd:teal", "xkcd:coral", "xkcd:indigo"],
        tslice=slice(270, 500),
        tmin=200,
        tmax=500,
        filter=lambda m: m != "crmhd",
    ),
    "fc": dict(
        patterns=["*b1-*Vmax2-rst", "*-fc*"],
        colors=["#E77500", "xkcd:crimson", "xkcd:magenta", "xkcd:blue"],
        tslice=slice(200, 500),
        tmin=0,
        tmax=600,
        filter=lambda m: "fc" in m,
    ),
    "beta": dict(
        patterns=["*b1-*Vmax2-rst", "*b0.1-*Vmax2", "*b10-*Vmax2-rst2"],
        colors=["#E77500", "xkcd:olive", "xkcd:gold"],
        tslice=slice(200, 500),
        tmin=0,
        tmax=700,
        filter=lambda m: m != "crmhd",
    ),
}


def load(model_dict, gr, tslice=slice(200, 500), colors=None, verbose=True):
    simgroup = dict()
    model_name = dict()
    model_color = dict()

    simgroup[gr] = dict()

    # categorize models
    mlist = []
    for k, d in model_dict.items():
        sim = LoadSimTIGRESSPP(d, verbose=verbose)
        par = sim.par
        base_split = sim.basename.split("-")
        model = []
        if par["cr"]["self_consistent_flag"] == 0:
            sigma_exp = int(-np.log10(par["cr"]["sigma"]))
            sigma = f"sigma{sigma_exp}"
            model.append(sigma)
            if par["cr"]["valfven_flag"] == 1:
                model.append("vAi")
            elif par["cr"]["valfven_flag"] == 0:
                model.append("vAtot")
            elif par["cr"]["valfven_flag"] == -1:
                model.append("nost")
                if par["cr"]["vs_flag"] == 1:
                    print("no streaming with invalud valfven_flag")
            if base_split[0] != "crmhd":
                model.append(base_split[0].split("_")[1])
        else:
            model.append(base_split[0])
            if par["problem"]["beta0"] != 1:
                model.append(f"b{par['problem']['beta0']}")
            if par["cr"]["vmax"] != 2.0e9:
                model.append(f"Vmax{int(par['cr']['vmax'] / 1.0e9)}")
        # if "rst" in base_split[-1]:
        #     model.append(base_split[-1])
        if "fc" in base_split[-1]:
            model.append(base_split[-1])
        if "noperp" in base_split[-1]:
            model.append(base_split[-1])
        newkey = "-".join(model)

        mlist.append(newkey)
        model_name[newkey] = newkey.replace("sigma", "σ").replace("-vAi", "")

        print(f"Renamed {k} --> {newkey}: {model_name[newkey]}")
        simgroup[gr][newkey] = sim

    # load data/ assign colors
    for group in simgroup:
        load_group(simgroup, group)
        for i, (m, s) in enumerate(simgroup[group].items()):
            if isinstance(tslice, dict):
                s.tslice_Myr = tslice[m]
            else:
                s.tslice_Myr = tslice
            s.tslice = slice(s.tslice_Myr.start / s.u.Myr, s.tslice_Myr.stop / s.u.Myr)

            # load_windpdf(s, both=True)
            # zp_pp = s.load_zprof_postproc()

            if colors is None:
                model_color[m] = f"C{i}"
            else:
                model_color[m] = colors[i]

            print(
                f"Loaded {m} in group {group} with name {model_name[m]} and color {model_color[m]}"
            )

    return simgroup, model_name, model_color


def run_group(group, num="all"):
    """Build simgroup, filter, set up plotting_scripts, then draw figures."""
    if group not in GROUPS:
        raise ValueError(f"unknown group {group!r}; choices: {list(GROUPS)}")
    cfg = GROUPS[group]

    model_dict = {}
    for pat in cfg["patterns"]:
        model_dict.update(cr_data_load(basedir, pat))

    simgroup, model_name, model_color = load(
        model_dict,
        "all",
        tslice=cfg["tslice"],
        colors=cfg["colors"],
        verbose=False,
    )

    simgroup[group] = {
        m: s for m, s in simgroup["all"].items() if cfg["filter"](m)
    }

    for m, c in cfg.get("color_overrides", {}).items():
        model_color[m] = c

    for m, s in simgroup["all"].items():
        if s.options["cosmic_ray"]:
            s.load_zprof_postproc()

    ps.setup(f"../{group}_figures", model_name, model_color)

    draw_figures(simgroup, group, cfg, num=num)


def draw_figures(simgroup, group, cfg, num="all"):
    """Numbered figure blocks. `num='all'` runs every block; int runs one."""
    tmin, tmax = cfg["tmin"], cfg["tmax"]

    if num == "all" or num == 1:
        ps.plot_history(simgroup, "all", tmin=tmin, tmax=tmax)

    if num == "all" or num == 2:
        ps.plot_pressure_t(simgroup, "all", zslice=slice(-50, 50), tmax=500)
        ps.plot_pressure_z(simgroup, "all")
        ps.plot_vertical_equilibrium_t(simgroup, group, tmin=tmin, zmax=1000)

    if num == "all" or num == 3:
        ps.plot_area_mass_fraction_z(simgroup, group)

    if num == "all" or num == 4:
        ps.plot_cr_velocity_z(simgroup, group, both=True)
        ps.plot_cr_velocity_z_all(simgroup, "all")

    if num == "all" or num == 5:
        ps.plot_kappa_z(simgroup, "all")
        ps.plot_kappa_z(
            simgroup,
            "all",
            phases=[["CNM", "UNM"], "WNM", ["WHIM", "HIM"]],
        )

    if num == "all" or num == 6:
        for m, s in simgroup[group].items():
            ps.plot_gainloss_z_each(
                s,
                m,
                phases=[["CNM", "UNM", "WNM"], "WHIM", "HIM"],
            )

    if num == "all" or num == 7:
        f = ps.plot_flux_tz(simgroup, group)
        f.axes[0].set_xlim(tmin, tmax)

    if num == "all" or num == 8:
        ps.plot_flux_z(simgroup, "all", vz_dir=1, both=True)

    if num == "all" or num == 9:
        ps.plot_loading_z_merged(simgroup, "all", vz_dir=None, both=True)

    if num == "all" or num == 10:
        ps.plot_momentum_transfer_z(
            simgroup, "all", show_option=1, zmin=0, zref=1000
        )

    if num == "all" or num == 11:
        for m, s in simgroup[group].items():
            load_windpdf(s, both=True)
        ps.plot_jointpdf(simgroup, group)
        ps.plot_jointpdf(simgroup, group, flux="eflux")
        ps.plot_voutpdf(simgroup, group)


def _parse_num(val):
    if val == "all":
        return "all"
    return int(val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        required=True,
        choices=list(GROUPS) + ["all"],
        help="figure group to generate",
    )
    parser.add_argument(
        "--num",
        default="all",
        help='figure block number (1-11) or "all"',
    )
    args = parser.parse_args()
    num = _parse_num(args.num)

    if args.group == "all":
        for g in GROUPS:
            run_group(g, num=num)
    else:
        run_group(args.group, num=num)
```

**Important:** the `load()` body above must match the existing `paperII.load()` byte-for-byte. Before writing, compare against Step 1's `cat` output — if they differ (e.g., someone edited `load()` since this plan was written), keep the version on disk and re-embed it here. Do NOT paraphrase or "clean up" its internals. It renames models by inspecting `par["cr"]["self_consistent_flag"]`, `par["cr"]["valfven_flag"]`, `par["cr"]["vs_flag"]`, `par["problem"]["beta0"]`, `par["cr"]["vmax"]`, and the `basename.split("-")` last segment for `fc` / `noperp`.

- [ ] **Step 3: Smoke check — argparse parses**

Run (any shell, no data required, just needs the imports to resolve — so use the pyathena env):
```bash
cd python
python paperII.py --help
```
Expected: usage message listing `--group {s28,s29,vAi,fc,beta,all}` and `--num`. No traceback.

- [ ] **Step 4: Smoke check — import surface intact**

Run:
```bash
cd python
python -c "import paperII; print(paperII.basedir); print(list(paperII.GROUPS)); print(callable(paperII.load), callable(paperII.run_group), callable(paperII.draw_figures))"
```
Expected output:
```
/scratch/gpfs/changgoo/tigress_classic/
['s28', 's29', 'vAi', 'fc', 'beta']
True True True
```
Any `ImportError` or `AttributeError` here is a real failure and must be fixed before commit.

- [ ] **Step 5: End-to-end check on one group, one figure block**

(This requires access to the simulation data on `/scratch/gpfs/changgoo/tigress_classic/` and is the user's manual step — if you are executing this plan without data, skip and note in commit message. Do NOT modify the code in an attempt to make it runnable without data.)

Run:
```bash
cd python
python paperII.py --group=s28 --num=1
```
Expected: `../s28_figures/` contains at least one new `.pdf` or `.png` (the history figure). No exception.

- [ ] **Step 6: Commit**

```bash
git add python/paperII.py
git commit -m "$(cat <<'EOF'
Refactor paperII.py into a CLI driver for figure groups

Adds GROUPS config, run_group(group), draw_figures(simgroup, group, cfg, num)
and an argparse entry point supporting

    python paperII.py --group=<s28|s29|vAi|fc|beta|all> [--num=<N|all>]

The existing load() function is preserved unchanged. Notebooks continue to
work via `from paperII import load` and can now also import run_group.
EOF
)"
```

---

### Task 2: Document the new CLI in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (the "Running / common commands" section)

**Interfaces:** none — documentation only.

- [ ] **Step 1: Read the current "Running / common commands" section**

Run: `sed -n '/^## Running/,/^## Architecture/p' CLAUDE.md`

Locate the paperI bullet:
```
- Regenerate paper figures (from `python/`):
  ```sh
  cd python
  python paperI.py           # all figures
  python paperI.py 3         # only figure group 3 (snapshots)
  python paperII.py          # invoked via wrapper functions from notebooks
  ```
```

- [ ] **Step 2: Replace the `paperII.py` bullet line and add a `--group` example**

Edit the existing block so the `python paperII.py` line becomes:
```
  python paperII.py --group=s28              # regenerate s28 figures
  python paperII.py --group=all --num=1      # only block 1 (history) for every group
```

Full replacement block after edit:
```
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
```

- [ ] **Step 3: Verify the edit**

Run: `grep -n 'paperII.py' CLAUDE.md`
Expected: three lines mentioning `paperII.py` (two in the code block, one in the descriptive sentence). No stale references to "wrapper functions from notebooks".

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "Document paperII CLI in CLAUDE.md"
```

---

## Post-implementation manual verification (owner runs on stellar)

After both tasks are committed, from the `pyathena` conda env with data mounted:

1. `cd python && python paperII.py --group=s28 --num=1` — spot check the history figure in `../s28_figures/`.
2. `python paperII.py --group=all` — run all 5 groups to completion.
3. Open one of the paperII notebooks and confirm `from paperII import load, run_group` still works.

If any group fails on block 11 because a sim lacks wind PDF data, that is a real bug in the sim/data — not a plan defect. Per the design decision, block 11 runs unconditionally.
