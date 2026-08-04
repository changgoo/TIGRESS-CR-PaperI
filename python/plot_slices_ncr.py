#!/usr/bin/env python3
"""Summary slice and projection plots for TIGRESS-NCR MHD/CRMHD runs.

The Tigris HDF5 frontend exposes chemistry scalars as ``xHI``, ``xH2``, and
``xe`` through :class:`LoadSimTIGRESSPP`.  Radiation bins currently retain
their Tigris names (``Er_rayt0/1/2``), so this module adds plotting-time
aliases for LyC, LW, and PE radiation while also accepting the older
TIGRESS-NCR field names.
"""

import argparse
import glob
import os
import os.path as osp

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as ac
import astropy.units as au
from matplotlib.colors import LogNorm
from mpi4py import MPI
from pyathena.plt_tools.make_movie import make_movie
from pyathena.plt_tools.plt_starpar import scatter_sp

from load_sim_tigresspp import LoadSimTIGRESSPP


ERAD_PE_ISRF = 7.613e-14
ERAD_LW_ISRF = 1.335e-14

RADIATION_ALIASES = {
    # Tigris ray-tracing frequency bins: 18.0, 12.2, and 9.0 eV.
    "LyC": ("Er_rayt0", "rad_energy_density_PH"),
    "LW": ("Er_rayt1", "rad_energy_density_LW"),
    "PE": ("Er_rayt2", "rad_energy_density_PE"),
}

NCR_MHD_FIELDS = (
    "nH",
    "nH2",
    "nHI",
    "nHII",
    "xe",
    "T",
    "pok",
    "vmag",
    "Bmag",
    "Zgas",
    "rret",
    "Erad_PE",
    "Erad_LW",
    "Erad_FUV",
    "Erad_LyC",
)

NCR_CR_FIELDS = ("sigma_para", "VAi_mag", "Vcr_mag", "pok_cr")

NCR_PROJECTION_FIELDS = (
    "Sigma",
    "Sigma_H2",
    "Sigma_HI",
    "Sigma_HII",
    "EM",
    "teflux",
    "keflux",
)

SUMMARY_PROJECTION_TITLES = {
    "Sigma": r"$\Sigma_{\rm gas}$",
    "Sigma_H2": r"$\Sigma_{\rm H_2}$",
    "Sigma_HI": r"$\Sigma_{\rm H\,I}$",
    "Sigma_HII": r"$\Sigma_{\rm H\,II}$",
}


def _first_available(data, aliases):
    for name in aliases:
        if name in data:
            return data[name]
    raise KeyError(f"none of the radiation aliases are available: {aliases}")


def _radiation_energy(*bands):
    """Return a derived-field function converting selected bands to cgs."""

    def derive(data, units):
        total = sum(_first_available(data, RADIATION_ALIASES[band]) for band in bands)
        return total * units.energy_density.cgs.value

    return derive


def _chi_fuv(data, units):
    energy = _radiation_energy("LW", "PE")(data, units)
    return energy / (ERAD_LW_ISRF + ERAD_PE_ISRF)


def _density_to_nh(units):
    """Convert code density to hydrogen-nucleus number density."""
    return (units.density / (units.muH * units.mH / au.cm**3)).cgs.value


def _n_h(data, units):
    return data["density"] * _density_to_nh(units)


def _n_h2(data, units):
    return _n_h(data, units) * data["xH2"]


def _n_hi(data, units):
    return _n_h(data, units) * data["xHI"]


def _n_hii(data, units):
    return _n_h(data, units) * (1.0 - data["xHI"] - 2.0 * data["xH2"])


def _temperature_ncr(data, units):
    """Gas temperature using the live NCR H2 and electron abundances."""
    particles_per_h = 1.1 + data["xe"] - data["xH2"]
    pressure_over_kb = data["pressure"] * (units.energy_density / ac.k_B).cgs.value
    return pressure_over_kb / (_n_h(data, units) * particles_per_h)


def _field_info(func, label_name, label_unit, vmin, vmax, cmap="viridis"):
    label = f"{label_name}\\;{label_unit}" if label_unit else label_name
    return {
        "func": func,
        "label": label,
        "label_name": label_name,
        "label_unit": label_unit,
        "imshow_args": {"cmap": cmap, "norm": LogNorm(vmin, vmax)},
    }


def get_ncr_dfi(sim):
    """Return the simulation field registry plus current NCR radiation fields."""
    dfi = dict(sim.dfi)
    dfi.update(
        {
            "nH": _field_info(
                _n_h,
                r"$n_{\rm H}$",
                r"$[\mathrm{cm^{-3}}]$",
                1e-4,
                1e4,
                cmap="cmr.rainforest",
            ),
            "nH2": _field_info(
                _n_h2,
                r"$n_{\rm H_2}$",
                r"$[\mathrm{cm^{-3}}]$",
                1e-4,
                1e4,
                cmap="cmr.rainforest",
            ),
            "nHI": _field_info(
                _n_hi,
                r"$n_{\rm H\,I}$",
                r"$[\mathrm{cm^{-3}}]$",
                1e-4,
                1e4,
                cmap="cmr.rainforest",
            ),
            "nHII": _field_info(
                _n_hii,
                r"$n_{\rm H\,II}$",
                r"$[\mathrm{cm^{-3}}]$",
                1e-4,
                1e4,
                cmap="cmr.rainforest",
            ),
            "T": _field_info(
                _temperature_ncr,
                r"$T$",
                r"$[\mathrm{K}]$",
                1e1,
                1e7,
                cmap="RdYlBu_r",
            ),
            "Erad_LyC": _field_info(
                _radiation_energy("LyC"),
                r"$\mathcal{E}_{\rm LyC}$",
                r"$[\mathrm{erg\,cm^{-3}}]$",
                1e-18,
                5e-11,
            ),
            "Erad_LW": _field_info(
                _radiation_energy("LW"),
                r"$\mathcal{E}_{\rm LW}$",
                r"$[\mathrm{erg\,cm^{-3}}]$",
                1e-16,
                5e-11,
            ),
            "Erad_PE": _field_info(
                _radiation_energy("PE"),
                r"$\mathcal{E}_{\rm PE}$",
                r"$[\mathrm{erg\,cm^{-3}}]$",
                1e-16,
                5e-11,
            ),
            "Erad_FUV": _field_info(
                _radiation_energy("LW", "PE"),
                r"$\mathcal{E}_{\rm FUV}$",
                r"$[\mathrm{erg\,cm^{-3}}]$",
                5e-16,
                5e-11,
            ),
            "chi_FUV": _field_info(
                _chi_fuv,
                r"$\chi_{\rm FUV}$",
                "",
                1e-4,
                1e4,
            ),
        }
    )
    return dfi


def default_slice_fields(sim):
    """Choose NCR fields for either an MHD or CRMHD simulation."""
    fields = list(NCR_MHD_FIELDS)
    if sim.options.get("cosmic_ray", False):
        fields.extend(NCR_CR_FIELDS)
    return fields


def default_projection_fields(sim):
    """Choose NCR projection fields, adding CR energy flux when available."""
    fields = list(NCR_PROJECTION_FIELDS)
    if sim.options.get("cosmic_ray", False):
        fields.append("creflux")
    return fields


def _field_data(sim, slc, field, dfi):
    info = dfi[field]
    if "func" in info:
        return info["func"](slc, sim.u), info
    return slc[field], info


def _plot_slice(
    sim,
    slc,
    field,
    dfi,
    horizontal,
    vertical,
    vector_components,
    *,
    ax=None,
    kpc=False,
    vec=None,
    stream_kwargs=None,
):
    ax = plt.gca() if ax is None else ax
    try:
        data, info = _field_data(sim, slc, field, dfi)
    except KeyError:
        return None

    scale = 1e-3 if kpc else 1.0
    image = ax.pcolormesh(
        data[horizontal] * scale,
        data[vertical] * scale,
        data,
        cmap=info["imshow_args"]["cmap"],
        norm=info["imshow_args"]["norm"],
        shading="auto",
    )
    if vec is not None:
        first, second = vector_components
        try:
            vx = slc[f"{vec}{first}"].data
            vy = slc[f"{vec}{second}"].data
        except KeyError:
            pass
        else:
            ax.streamplot(
                slc[horizontal].data * scale,
                slc[vertical].data * scale,
                vx,
                vy,
                **({"color": "k"} if stream_kwargs is None else stream_kwargs),
            )

    axis_index = {"x": 0, "y": 1, "z": 2}
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(
        sim.domain["le"][axis_index[horizontal]] * scale,
        sim.domain["re"][axis_index[horizontal]] * scale,
    )
    ax.set_ylim(
        sim.domain["le"][axis_index[vertical]] * scale,
        sim.domain["re"][axis_index[vertical]] * scale,
    )
    return image


def plot_slice_xy(
    sim, slc, field, dfi, *, ax=None, kpc=False, vec=None, stream_kwargs=None
):
    """Draw one face-on NCR slice."""
    return _plot_slice(
        sim,
        slc,
        field,
        dfi,
        "x",
        "y",
        (1, 2),
        ax=ax,
        kpc=kpc,
        vec=vec,
        stream_kwargs=stream_kwargs,
    )


def plot_slice_xz(
    sim, slc, field, dfi, *, ax=None, kpc=False, vec=None, stream_kwargs=None
):
    """Draw one edge-on NCR slice."""
    return _plot_slice(
        sim,
        slc,
        field,
        dfi,
        "x",
        "z",
        (1, 3),
        ax=ax,
        kpc=kpc,
        vec=vec,
        stream_kwargs=stream_kwargs,
    )


def _vector_fields(fields, enabled=True):
    vectors = {field: None for field in fields}
    if not enabled:
        return vectors
    for field, vector in {
        "vmag": "velocity",
        "Bmag": "cell_centered_B",
        "VAi_mag": "0-Vs",
        "Vcr_mag": "0-Fc",
        "pok_mag": "cell_centered_B",
    }.items():
        if field in vectors:
            vectors[field] = vector
    return vectors


def plot_slices_ncr(
    sim,
    num,
    *,
    fields=None,
    kpc=False,
    vectors=True,
    savefig=True,
    output_dir=None,
    force_override=False,
):
    """Plot edge-on and face-on NCR fields for an MHD or CRMHD snapshot."""
    fields = default_slice_fields(sim) if fields is None else list(fields)
    dfi = get_ncr_dfi(sim)
    unknown = [field for field in fields if field not in dfi]
    if unknown:
        raise KeyError(f"fields are missing from the derived-field registry: {unknown}")

    slc_xy = sim.get_slice(
        num,
        "allslc.z",
        slc_kwargs={"z": 0, "method": "nearest"},
        force_override=force_override,
    )
    slc_xz = sim.get_slice(
        num,
        "allslc.y",
        slc_kwargs={"y": 0, "method": "nearest"},
        force_override=force_override,
    )
    xz_ratio = sim.domain["Lx"][2] / sim.domain["Lx"][1]
    figure, axes = plt.subplots(
        2,
        len(fields),
        sharex="col",
        sharey="row",
        figsize=(1.6 * len(fields), 1.6 * (1.0 + xz_ratio)),
        gridspec_kw={
            "height_ratios": [xz_ratio, 1.0],
            "wspace": 0.0,
            "hspace": 0.0,
        },
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(2, len(fields))
    vector_fields = _vector_fields(fields, enabled=vectors)

    for row, (slc, vertical, plotter) in enumerate(
        ((slc_xz, "z", plot_slice_xz), (slc_xy, "y", plot_slice_xy))
    ):
        for column, field in enumerate(fields):
            ax = axes[row, column]
            vector = vector_fields[field]
            density = 0.5 if vector == "0-Vs" else 1.0
            image = plotter(
                sim,
                slc,
                field,
                dfi,
                ax=ax,
                kpc=kpc,
                vec=vector,
                stream_kwargs={
                    "color": "silver" if field in {"Bmag", "pok_mag"} else "k",
                    "density": density if row else (density, 3.0 * density),
                    "linewidth": 0.5,
                    "arrowsize": 0.7,
                },
            )
            if image is not None and row == 0:
                colorbar = figure.colorbar(
                    image,
                    ax=ax,
                    orientation="horizontal",
                    location="top",
                    pad=0.01,
                    shrink=0.8,
                    aspect=10,
                )
                label = dfi[field]["label_name"]
                unit = dfi[field].get("label_unit", "")
                colorbar.set_label(f"{label}\n{unit}" if unit else label)
            if row == 0:
                ax.axhline(0.0, color="k", linestyle=":", linewidth=0.7)
            ax.axis("off")

    time_myr = slc_xy.attrs["time"] * sim.u.Myr
    axes[0, 0].annotate(
        f"t={time_myr:.2f} Myr",
        (0.05, 0.98),
        ha="left",
        va="top",
        xycoords="axes fraction",
        fontsize="large",
        bbox={"boxstyle": "round,pad=0.2", "fc": "w", "ec": "k", "lw": 1},
    )
    if savefig:
        output_dir = output_dir or osp.join(sim.savdir, "ncr_slices")
        os.makedirs(output_dir, exist_ok=True)
        path = osp.join(output_dir, f"{sim.basename}_{num:05d}.png")
        figure.savefig(path, dpi=200, bbox_inches="tight")
    return figure


def plot_projections_ncr(
    sim,
    num,
    *,
    fields=None,
    savefig=True,
    output_dir=None,
    force_override=False,
):
    """Plot edge-on NCR surface densities and vertical energy fluxes."""
    fields = default_projection_fields(sim) if fields is None else list(fields)
    projection = sim.get_prj(
        num,
        "y",
        prefix="prj.y",
        force_override=force_override,
    )
    plot_args, labels = sim.set_prj_dfi()
    xz_ratio = sim.domain["Lx"][2] / sim.domain["Lx"][1]
    figure, axes = plt.subplots(
        1,
        len(fields),
        figsize=(1.8 * len(fields), 1.8 * xz_ratio),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, field in zip(axes, fields):
        ax.axis("off")
        if field not in projection or field not in plot_args:
            continue
        data = projection[field].sel(phase="whole")
        image = ax.pcolormesh(data.x, data.z, data, shading="auto", **plot_args[field])
        figure.colorbar(
            image,
            ax=ax,
            orientation="horizontal",
            location="top",
            pad=0.01,
            shrink=0.8,
            aspect=10,
            label=labels[field],
        )
        ax.set_aspect("equal", adjustable="box")

    time_myr = projection.attrs["time"] * sim.u.Myr
    axes[0].annotate(
        f"t={time_myr:.2f} Myr",
        (0.05, 0.98),
        ha="left",
        va="top",
        xycoords="axes fraction",
        fontsize="large",
        bbox={"boxstyle": "round,pad=0.2", "fc": "w", "ec": "k", "lw": 1},
    )
    if savefig:
        output_dir = output_dir or osp.join(sim.savdir, "ncr_projections")
        os.makedirs(output_dir, exist_ok=True)
        path = osp.join(output_dir, f"{sim.basename}_{num:05d}.png")
        figure.savefig(path, dpi=200, bbox_inches="tight")
    return figure


def _summary_cbar_value(value, *, logarithmic):
    """Format a compact colorbar endpoint for an inset summary panel."""
    if logarithmic and value > 0.0:
        exponent = np.log10(value)
        if np.isclose(exponent, round(exponent)):
            return rf"$10^{{{int(round(exponent))}}}$"
    if value == 0.0:
        return "$0$"
    if abs(value) < 1.0e-2 or abs(value) >= 1.0e3:
        exponent = int(np.floor(np.log10(abs(value))))
        coefficient = value / 10.0**exponent
        if np.isclose(abs(coefficient), 1.0):
            sign = "-" if coefficient < 0.0 else ""
            return rf"${sign}10^{{{exponent}}}$"
        return rf"${coefficient:.1f}\!\times\!10^{{{exponent}}}$"
    return f"${value:g}$"


def _add_summary_colorbar(
    figure,
    ax,
    image,
    title,
    *,
    height=0.07,
    title_size=10,
    endpoint_size=8,
):
    """Add a high-contrast inset colorbar without crowded automatic ticks."""
    cax = ax.inset_axes([0.08, 0.86, 0.84, height])
    colorbar = figure.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.set_ticks([])
    colorbar.outline.set_edgecolor("white")
    colorbar.outline.set_linewidth(0.9)

    effects = [path_effects.withStroke(linewidth=2.0, foreground="black")]
    text_options = {
        "va": "center",
        "color": "white",
        "transform": cax.transAxes,
        "path_effects": effects,
        "clip_on": False,
    }
    logarithmic = isinstance(image.norm, LogNorm)
    cax.text(
        0.02,
        0.5,
        _summary_cbar_value(image.norm.vmin, logarithmic=logarithmic),
        ha="left",
        fontsize=endpoint_size,
        **text_options,
    )
    cax.text(
        0.5,
        0.5,
        title,
        ha="center",
        fontsize=title_size,
        **text_options,
    )
    cax.text(
        0.98,
        0.5,
        _summary_cbar_value(image.norm.vmax, logarithmic=logarithmic),
        ha="right",
        fontsize=endpoint_size,
        **text_options,
    )
    return colorbar


def _summary_panel(
    figure,
    ax,
    data,
    *,
    cmap,
    norm,
    title,
    colorbar=True,
    colorbar_height=0.07,
    title_size=10,
    endpoint_size=8,
):
    """Draw one axis-off panel in the C-version summary-plot style."""
    array = np.ma.masked_less_equal(data, 0.0) if isinstance(norm, LogNorm) else data
    horizontal, vertical = data.dims[-1], data.dims[-2]
    image = ax.pcolormesh(
        data[horizontal],
        data[vertical],
        array,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if colorbar:
        _add_summary_colorbar(
            figure,
            ax,
            image,
            title,
            height=colorbar_height,
            title_size=title_size,
            endpoint_size=endpoint_size,
        )
    else:
        ax.text(
            0.5,
            0.93,
            title,
            color="white",
            fontsize=10,
            ha="center",
            va="top",
            transform=ax.transAxes,
            path_effects=[path_effects.withStroke(linewidth=2.0, foreground="black")],
        )
    return image


def _summary_figure_size(sim, edge_field_count):
    edge_aspect = sim.domain["Lx"][0] / sim.domain["Lx"][2]
    edge_width = min(14.0, max(4.1, 4.0 * edge_field_count * edge_aspect))
    width_ratios = [1.0, 1.0, 1.0, edge_width]
    target_ratio = sum(width_ratios) / 4.0
    height = 11.0
    width = min(34.0, max(18.0, height * target_ratio))
    if width == 34.0:
        height = max(7.0, width / target_ratio)
    return width_ratios, (width, height)


def plot_snapshot_ncr(
    sim,
    num,
    *,
    parnum=None,
    savefig=True,
    output_dir=None,
    force_override=False,
):
    """Plot the four-column TIGRESS-NCR summary used by the C-version code.

    The columns contain face-on species projections, face-on species slices,
    face-on thermal/radiation slices, and a strip of edge-on diagnostics.
    CRMHD runs add CR transport and pressure panels to the edge-on strip.
    """
    dfi = get_ncr_dfi(sim)
    if parnum is None:
        parnum = num
    particles = sim.load_parbin(parnum)
    cluster_age = particles["age"] * sim.u.Myr
    clusters = particles[(particles["mass"] > 0.0) & (cluster_age < 40.0)]

    projection_xy = sim.get_prj(
        num,
        "z",
        prefix="prj.z",
        force_override=force_override,
    )
    slice_xy = sim.get_slice(
        num,
        "allslc.z",
        slc_kwargs={"z": 0, "method": "nearest"},
        force_override=force_override,
    )
    slice_xz = sim.get_slice(
        num,
        "allslc.y",
        slc_kwargs={"y": 0, "method": "nearest"},
        force_override=force_override,
    )
    projection_args, _ = sim.set_prj_dfi()

    projection_fields = ("Sigma", "Sigma_H2", "Sigma_HI", "Sigma_HII")
    species_fields = ("nH", "nH2", "nHI", "nHII")
    thermal_fields = ("T", "pok", "Erad_PE", "Erad_LyC")
    edge_fields = ["nH", "T", "vz", "pok", "Bmag", "Erad_PE", "Erad_LyC"]
    if sim.options.get("cosmic_ray", False):
        edge_fields.extend(NCR_CR_FIELDS)

    width_ratios, figsize = _summary_figure_size(sim, len(edge_fields))
    figure = plt.figure(figsize=figsize, constrained_layout=True)
    outer = figure.add_gridspec(4, 4, width_ratios=width_ratios)
    axes = {
        "projection": [],
        "slice_xy_species": [],
        "slice_xy_thermal": [],
        "slice_xz": [],
    }

    for row, field in enumerate(projection_fields):
        ax = figure.add_subplot(outer[row, 0])
        data = projection_xy[field].sel(phase="whole")
        args = projection_args[field]
        _summary_panel(
            figure,
            ax,
            data,
            cmap=args["cmap"],
            norm=args["norm"],
            title=SUMMARY_PROJECTION_TITLES[field],
            colorbar=row == 0,
        )
        if row == 0 and not clusters.empty:
            first_particle_collection = len(ax.collections)
            scatter_sp(
                clusters,
                ax,
                "z",
                kind="prj",
                runaway=False,
                agemax=40.0,
                agemax_sn=40.0,
                norm_factor=5.0,
                edgecolors="white",
                linewidths=0.4,
            )
            for collection in ax.collections[first_particle_collection:]:
                collection.set_zorder(5)
        axes["projection"].append(ax)

    for row, field in enumerate(species_fields):
        ax = figure.add_subplot(outer[row, 1])
        data, info = _field_data(sim, slice_xy, field, dfi)
        _summary_panel(
            figure,
            ax,
            data,
            cmap=info["imshow_args"]["cmap"],
            norm=info["imshow_args"]["norm"],
            title=info["label_name"],
            colorbar=row == 0,
        )
        axes["slice_xy_species"].append(ax)

    for row, field in enumerate(thermal_fields):
        ax = figure.add_subplot(outer[row, 2])
        data, info = _field_data(sim, slice_xy, field, dfi)
        _summary_panel(
            figure,
            ax,
            data,
            cmap=info["imshow_args"]["cmap"],
            norm=info["imshow_args"]["norm"],
            title=info["label_name"],
        )
        axes["slice_xy_thermal"].append(ax)

    edge_grid = outer[:, 3].subgridspec(1, len(edge_fields))
    for column, field in enumerate(edge_fields):
        ax = figure.add_subplot(edge_grid[0, column])
        data, info = _field_data(sim, slice_xz, field, dfi)
        _summary_panel(
            figure,
            ax,
            data,
            cmap=info["imshow_args"]["cmap"],
            norm=info["imshow_args"]["norm"],
            title=info["label_name"],
            colorbar_height=0.055,
            title_size=9,
            endpoint_size=7,
        )
        axes["slice_xz"].append(ax)

    time_myr = slice_xy.attrs["time"] * sim.u.Myr
    figure.suptitle(
        f"{sim.basename} output {num:05d}  t={time_myr:.2f} Myr",
        fontsize=12,
    )
    if savefig:
        output_dir = output_dir or osp.join(sim.savdir, "snapshot_ncr")
        os.makedirs(output_dir, exist_ok=True)
        path = osp.join(output_dir, f"snapshot.{num:05d}.png")
        figure.savefig(path, dpi=200, bbox_inches="tight")
    return figure, axes


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TIGRESS-NCR MHD/CRMHD summary plots"
    )
    parser.add_argument("basedir", help="simulation output directory")
    parser.add_argument(
        "nums",
        nargs="*",
        type=int,
        help="snapshot numbers (default: all primary HDF5 snapshots)",
    )
    parser.add_argument("--savdir", help="cache and figure directory")
    parser.add_argument("--force", action="store_true", help="rebuild cached products")
    parser.add_argument(
        "--projections",
        action="store_true",
        help="also generate the standalone NCR projection figures",
    )
    parser.add_argument(
        "--no-movies",
        action="store_true",
        help="skip MP4 creation after all frames are generated",
    )
    parser.add_argument("--fps", type=int, default=15, help="movie frame rate")
    return parser.parse_args()


def make_movies_ncr(sim, *, include_projections=False, fps=15):
    """Create MP4 movies from the available NCR frame directories."""
    movies_dir = osp.join(sim.savdir, "movies")
    os.makedirs(movies_dir, exist_ok=True)
    products = [
        ("snapshot_ncr", f"{sim.basename}_snapshot_ncr.mp4"),
        ("ncr_slices", f"{sim.basename}_ncr_slices.mp4"),
    ]
    if include_projections:
        products.append(("ncr_projections", f"{sim.basename}_ncr_projections.mp4"))

    outputs = []
    for frame_dir, movie_name in products:
        frame_glob = osp.join(sim.savdir, frame_dir, "*.png")
        if not glob.glob(frame_glob):
            continue
        movie_path = osp.join(movies_dir, movie_name)
        if not make_movie(frame_glob, movie_path, fps_in=fps, fps_out=fps):
            raise RuntimeError(f"failed to create movie: {movie_path}")
        outputs.append(movie_path)
    return outputs


def main():
    args = _parse_args()
    comm = MPI.COMM_WORLD
    sim = LoadSimTIGRESSPP(
        args.basedir,
        savdir=args.savdir,
        verbose=comm.rank == 0,
    )
    nums = list(args.nums or sim.nums) if comm.rank == 0 else None
    nums = comm.bcast(nums, root=0)
    rank_nums = nums[comm.rank :: comm.size]
    print(f"rank {comm.rank}/{comm.size}: {rank_nums}", flush=True)

    for num in rank_nums:
        figure, _ = plot_snapshot_ncr(
            sim,
            num,
            force_override=args.force,
        )
        plt.close(figure)
        figure = plot_slices_ncr(
            sim,
            num,
            force_override=args.force,
        )
        plt.close(figure)
        if args.projections:
            figure = plot_projections_ncr(
                sim,
                num,
                force_override=args.force,
            )
            plt.close(figure)
    comm.barrier()
    if comm.rank == 0 and not args.no_movies:
        make_movies_ncr(
            sim,
            include_projections=args.projections,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
