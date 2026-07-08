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
PATH = osp.dirname(osp.abspath(__file__))

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

    ps.setup(f"{PATH}/../paperII_figures/{group}_figures", model_name, model_color)

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

    if num == "all" or num == 12:
        for m, s in simgroup["all"].items():
            if s.options["cosmic_ray"]:
                ps.plot_crgain_cumsum(s, m)


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
