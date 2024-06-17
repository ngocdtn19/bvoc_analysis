import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib as mpl
import xarray as xr
import pandas as pd
import pickle

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from mypath import *
from mk import *


colors_dict = {
    "co2": "#8da0cb",
    "co2f": "#762a83",
    "co2fi": "#9970ab",
    "lulcc": "#b3de69",
    "clim": "#fb8072",
    "all": "#66c2a5",
    "tas": "#e31a1c",
    "rsds": "#fee08b",
    "pr": "#386cb0",
}

linestyles_dict = {
    "co2": "-",
    "co2f": "-",
    "co2fi": "-",
    "lulcc": "-",
    "clim": "-.",
    "all": "--",
    "tas": "-",
    "rsds": "-",
    "pr": "-",
}

map_colors_dict = {
    "co2": "#beaed4",
    "co2f": "#beaed4",
    "co2fi": "#beaed4",
    "co$_2$": "#beaed4",
    "lulcc": "#b3de69",
    "clim": "#fb9a99",
    "tas": "#e31a1c",
    "rsds": "#ffff99",
    "pr": "#386cb0",
    "nan": "lightgrey",
}

model_orders = [
    "VISIT(G1997)",
    "CESM2-WACCM(G2012)",
    "NorESM2-LM(G2012)",
    "UKESM1-0-LL(P2011)",
    "GFDL-ESM4(G2006)",
    "GISS-E2.1-G(G1995)",
]

title_sz = 16
legend_sz = 14
unit_sz = 12


# Plot cross-validation
def plt_cross_val_metrics():
    cmip6_models = model_orders[1:]
    l_score = []
    for m in cmip6_models:
        cv_metric_file = f"./cv/{m}.pkl"
        with open(cv_metric_file, "rb") as f:
            cv_metrics = pickle.load(f)
            df = pd.DataFrame(cv_metrics)
            df["model_names"] = m
            l_score.append(df)
    concat_df = pd.concat(l_score, ignore_index=True)
    l_score_names = ["r2", "rmse", "mae"]
    ylabels = [" ", "[$gC  m^{-2}  yr^{-1}$]", "[$gC  m^{-2}  yr^{-1}$]"]
    titles = ["R$^{2}$", "RMSE", "MAE"]
    rows = 1
    cols = 3
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3 * cols, 4.5 * rows),
        layout="constrained",
    )
    for i, (s, ylabel, t) in enumerate(zip(l_score_names, ylabels, titles)):
        c = i % cols
        ax = axes[c]
        barplot = sns.barplot(
            concat_df,
            x="model_names",
            y=s,
            ax=ax,
            palette=sns.color_palette("Set3"),
            width=0.65,
        )
        ax.set_xticklabels(cmip6_models, rotation=90)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(t)


# Plot Fig. 1b - regional contribution for present-day (2000-2014)
def plt_regional_contri(emiisop):
    l_roi = LIST_REGION
    model_names = list(emiisop.multi_models.keys())

    l_y = []
    for roi in l_roi:
        l_y.append(
            np.array(
                [
                    emiisop.multi_models[name]
                    .regional_rate[roi]
                    .sel(year=slice(2000, 2014))
                    .mean()
                    .item()
                    for name in model_names
                ]
            )
        )
    l_y = np.array(l_y)
    df = pd.DataFrame({roi: val for roi, val in zip(l_roi, l_y)}, index=model_names)
    ax = df.plot.bar(
        stacked=True,
        color=ROI_COLORS,
        rot=45,
    )
    for x, y in enumerate(df.sum(axis=1)):
        ax.annotate(np.round(y, decimals=0), (x, y + 5), ha="center")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="right",
        ncol=1,
        bbox_to_anchor=(0.5, 0.5, 0.7, 0.1),
        borderaxespad=0.0,
    )
    ax.set_ylabel(
        VIZ_OPT[emiisop.var_name]["line_bar_unit"], fontsize=unit_sz, fontweight="bold"
    )


# Plot Fig. 2 - spatial distribution of the mean annual totals of isoprene emission in the present day (2000-2014)
def plt_glob_present_map(emiisop, cmap="YlGn"):
    list_models = [
        "VISIT-S3(G1997)",
        "CESM2-WACCM(G2012)",
        "UKESM1-0-LL(P2011)",
        "NorESM2-LM(G2012)",
        "GFDL-ESM4(G2006)",
        "GISS-E2.1-G(G1995)",
    ]
    vmin, vmax = 0, 40
    cmap = mpl.colormaps.get_cmap(cmap)
    cmap.set_under("snow")

    rows = 3
    cols = 2
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 2 * rows),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )
    for i, m in enumerate(list_models):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        ax.coastlines()
        data = (
            emiisop.multi_models[m]
            .annual_per_area_unit.sel(year=slice(2000, 2014))
            .mean("year")
        ) * emiisop.multi_models[m].ds_mask["mask"]
        if "VISIT" in m:
            data = data.sel(lat=slice(82.75, -55.25))
        else:
            data = data.sel(lat=slice(-55.25, 82.75))
        data.plot.pcolormesh(
            ax=ax,
            cmap=cmap,
            levels=21,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
        )
        ax.set_title(m)
    bounds = np.arange(vmin, vmax + vmax * 0.05, vmax * 0.05)
    norm = mpl.colors.BoundaryNorm(bounds, mpl.cm.plasma.N, extend="both")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, location="bottom")
    cbar.set_label("[$gC  m^{-2}  yr^{-1}$]", size=unit_sz)


# Plot Fig.5 - Spatial distribution of isoprene emission trends from 1850 to 2014


def cal_org_trends_map(var_obj, var_name, model_name):
    file_mk_org = os.path.join(
        DATA_DIR, "processed_data/mk_org", f"{model_name}_{var_name}.nc"
    )
    if not os.path.exists(file_mk_org):
        annual_ds = var_obj.multi_models[model_name].annual_per_area_unit

        y = xr.DataArray(
            np.arange(len(annual_ds["year"])) + 1,
            dims="year",
            coords={"year": annual_ds["year"]},
        )
        slope = mk.kendall_correlation(annual_ds, y, "year")
        slope.to_netcdf(file_mk_org)
    else:
        slope = xr.open_dataset(file_mk_org)
    return slope


def plt_glob_trends_map_emiisop(emiisop, cmap="bwr"):
    list_models = [
        "VISIT-S3(G1997)",
        "CESM2-WACCM(G2012)",
        "UKESM1-0-LL(P2011)",
        "NorESM2-LM(G2012)",
        "GFDL-ESM4(G2006)",
        "GISS-E2.1-G(G1995)",
    ]

    cmap = mpl.colormaps.get_cmap(cmap)
    vmin, vmax = -50, 50

    rows = 3
    cols = 2
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 2 * rows),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )
    for i, m in enumerate(list_models):
        # calculate mk trends
        slope_ds = cal_org_trends_map(emiisop, "emiisop", m)

        r = i // cols
        c = i % cols
        ax = axes[r, c]
        ax.coastlines()
        data = slope_ds[list(slope_ds.keys())[0]] * 1e3
        if "VISIT" in m:
            data = data.sel(lat=slice(82.75, -55.25))
        else:
            data = data.sel(lat=slice(-55.25, 82.75))
        data.plot.pcolormesh(
            ax=ax,
            cmap=cmap,
            levels=21,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
        )
        ax.set_title(m)
    bounds = np.arange(vmin, vmax + vmax * 0.1, vmax * 0.1)
    norm = mpl.colors.BoundaryNorm(bounds, mpl.cm.plasma.N, extend="both")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, location="bottom")
    cbar.set_label("[$mgC  m^{-2}  yr^{-2}$]", size=unit_sz)


# Plot Fig. 6 & Fig. 8
def plt_glob_rate(models, visit, mode="main"):
    models["VISIT(G1997)"] = visit
    rows = 2
    cols = 3
    fig, axes = plt.subplots(
        rows,
        cols,
        sharex=True,
        sharey=True,
        figsize=(5 * cols, 3.5 * rows),
        layout="constrained",
    )

    for i, m in enumerate(model_orders):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        axbox = ax.get_position()

        if mode == "main":
            df = models[m].main_rates_ts
            legend_elements = [
                Line2D([0], [0], color="#762a83", lw=2.5, label="co$_2f$"),
                Line2D([0], [0], color="#9970ab", lw=2.5, label="co$_2fi$"),
                Line2D([0], [0], color="#b3de69", lw=2.5, label="lulcc"),
                Line2D([0], [0], color="#fb8072", ls="-.", lw=2.5, label="clim"),
                Line2D([0], [0], color="#66c2a5", ls="--", lw=2.5, label="all"),
            ]
        else:
            df = models[m].clim_rates_ts
        pred_fields = df.columns
        for f in pred_fields:
            obj = df[f]
            x, y = obj.index, obj.values
            ax.plot(
                x,
                y,
                label=f,
                linewidth=2.5,
                color=colors_dict[f],
                ls=linestyles_dict[f],
            )
        ax.set_ylim([-160, 110])
        ax.set_title(m, fontsize=title_sz)
        if r in [0, 1]:
            axes[r, 0].set_ylabel(
                "Isoprene emission changes [$TgC  yr^{-1}$]", fontsize=unit_sz
            )

    if mode == "main":
        fig.legend(
            handles=legend_elements,
            ncol=5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            fontsize=legend_sz,
        )
    else:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            fontsize=legend_sz,
        )
    # if fig_name:
    #     path_ = f"../figures/{fig_name}.tiff"
    #     fig.savefig(
    #         path_,
    #         format="tiff",
    #         dpi=300,
    #         bbox_inches="tight",
    #     )


# Plot Fig. 7 & Fig. 9
def plt_glob_rate_drivers(models, visit, mode="main"):
    models["VISIT(G1997)"] = visit
    rows = 2
    cols = 3
    fig, axes = plt.subplots(
        rows,
        cols,
        sharey=True,
        figsize=(3 * cols, 4.5 * rows),
        layout="constrained",
    )
    for i, m in enumerate(model_orders):
        r = i // cols
        c = i % cols
        ax = axes[r, c]

        if mode == "main":
            df = models[m].main_df_rate
        else:
            df = models[m].clim_df_rate
        list_drivers = df["driver"].values
        color_list = [colors_dict[d] for d in list_drivers]
        barplot = sns.barplot(
            df,
            x="driver",
            y="slope",
            ax=ax,
            palette=sns.color_palette(color_list),
        )
        for p, sig in zip(barplot.patches, df["sig"]):
            if sig == True:
                h = p.get_height()
                add_h = 0.1
                if mode == "clim":
                    add_h = 0.025
                h = h if h > 0 else h - add_h
                barplot.text(p.get_x() + p.get_width() / 2.0, h, "*", ha="center")
                print(h)
        ax.set_title(m, fontsize=title_sz)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if r in [0, 1]:
            axes[r, 0].set_ylabel(
                "Isoprene emission trends [$TgC  yr^{-2}$]", fontsize=unit_sz
            )
        ax.set_ylim(-1, 1)
        if mode == "clim":
            ax.set_ylim(-0.25, 0.25)


# Plot Fig. 10 & Fig. 12
def plt_contri_map(models, visit, mode="main"):
    models["VISIT(G1997)"] = visit
    cmap = "bwr"
    vmin, vmax = -15, 15
    fig = plt.figure(layout="constrained", figsize=(3.5 * 4, 3.5 * 4))
    subfigs = fig.subfigures(6, 1, hspace=0.1)
    if mode == "main":
        for n, m in enumerate(model_orders):
            pred_fields = models[m].list_main_driver
            axis = subfigs[n].subplots(
                1, 3, subplot_kw=dict(projection=ccrs.PlateCarree())
            )
            subfigs[n].suptitle(m, fontsize=title_sz)
            for i, f in enumerate(pred_fields):
                ax = axis[i]
                if len(pred_fields) < 3:
                    ax = axis[i + 1]
                ax.coastlines()
                data = models[m].contribution_mk[f] * 1e3
                data.plot.pcolormesh(
                    ax=ax,
                    cmap=cmap,
                    levels=11,
                    vmin=vmin,
                    vmax=vmax,
                    extend="both",
                    add_colorbar=False,
                )
            if n in [4, 5]:
                axis[0].axis("off")
                axis[0].axis("off")
                axis[0].axis("off")
                axis[0].axis("off")

        bounds = [i for i in range(vmin, vmax + 3, 3)]
        norm = mpl.colors.BoundaryNorm(bounds, mpl.cm.plasma.N, extend="both")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        axins = inset_axes(
            axis[1],
            width="100%",  # width: 5% of parent_bbox width
            height="20%",  # height: 50%
            loc="center",
            bbox_to_anchor=(0, -1, 1, 1),
            bbox_transform=axis[1].transAxes,
            borderpad=0,
        )
        # if placing color in the left side
        # axins = inset_axes(
        # axis[1],
        # width="100%",  # width: 5% of parent_bbox width
        # height="20%",  # height: 50%
        # loc="upper left",
        # bbox_to_anchor=(-1.025, 1, 1, 1),
        # bbox_transform=axis[1].transAxes,
        # borderpad=0,
        # )
        subfigs[-1].text(0.15, -0.1, "(a) CO$_2$", fontsize=title_sz, fontweight="bold")
        subfigs[-1].text(0.45, -0.1, "(b) LULCC", fontsize=title_sz, fontweight="bold")
        subfigs[-1].text(0.8, -0.1, "(c) Clim", fontsize=title_sz, fontweight="bold")
    else:
        vmin, vmax = -2.5, 2.5
        for n, m in enumerate(model_orders):
            pred_fields = models[m].clim_predictors
            axis = subfigs[n].subplots(
                1, 3, subplot_kw=dict(projection=ccrs.PlateCarree())
            )
            subfigs[n].suptitle(m, fontsize=title_sz)
            for i, f in enumerate(pred_fields):
                ax = axis[i]
                ax.coastlines()
                data = models[m].contribution_mk[f] * 1e3
                data.plot.pcolormesh(
                    ax=ax,
                    cmap=cmap,
                    levels=11,
                    vmin=vmin,
                    vmax=vmax,
                    extend="both",
                    add_colorbar=False,
                )
        bounds = np.arange(vmin, vmax + 0.5, 0.5)
        norm = mpl.colors.BoundaryNorm(bounds, mpl.cm.plasma.N, extend="both")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        axins = inset_axes(
            axis[1],
            width="100%",  # width: 5% of parent_bbox width
            height="20%",  # height: 50%
            loc="center",
            bbox_to_anchor=(0, -1, 1, 1),
            bbox_transform=axis[1].transAxes,
            borderpad=0,
        )
        subfigs[-1].text(
            0.09, -0.1, "(a) Temperature", fontsize=title_sz, fontweight="bold"
        )
        subfigs[-1].text(
            0.39, -0.1, "(b) Shortwave radiation", fontsize=title_sz, fontweight="bold"
        )
        subfigs[-1].text(
            0.76, -0.1, "(c) Precipitation", fontsize=title_sz, fontweight="bold"
        )

    cbar = fig.colorbar(sm, cax=axins, shrink=1, orientation="horizontal")
    cbar.set_label("[$mgC  m^{-2}  yr^{-2}$]", size=unit_sz)

    # path_ = f"figs/IAV.jpg"
    # fig.savefig(path_, format="jpg", dpi=900, bbox_inches="tight")


# Plot Fig. 11 & Fig. 13
def plt_max_impact_map(models, visit, mode="main"):
    models["VISIT(G1997)"] = visit
    list_models = [
        "VISIT(G1997)",
        "CESM2-WACCM(G2012)",
        "UKESM1-0-LL(P2011)",
        "NorESM2-LM(G2012)",
        "GFDL-ESM4(G2006)",
        "GISS-E2.1-G(G1995)",
    ]
    rows = 3
    cols = 2
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 2 * rows),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )

    for i, m in enumerate(list_models):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        ax.coastlines()
        if mode == "main":
            drivers = models[m].list_main_driver
            data = models[m].ctb_main_map
            cmap = mpl.colors.ListedColormap(
                [map_colors_dict[d] for d in models[m].list_main_driver + ["nan"]]
            )
        else:
            drivers = models[m].clim_predictors
            data = models[m].ctb_clim_map
            cmap = mpl.colors.ListedColormap(
                [map_colors_dict[d] for d in models[m].clim_predictors + ["nan"]]
            )
        data = data.fillna(len(drivers))

        # interpolate visit map to the model coords
        visit_land = xr.open_dataset(
            "/mnt/dg3/ngoc/cmip6_bvoc_als/data/axl/mask/mask_fx_VISIT-S3(G1997)_historical_r1i1p1f1_gn.nc"
        )
        if m == "VISIT(G1997)":
            land_mask = visit_land.where(visit_land.mask != np.nan, 1)
        else:
            visit_land.coords["lon"] = (
                visit_land.coords["lon"] % 360
            )  # if interpolate from visit to gfdl
            visit_land = visit_land.sortby(visit_land.lon)
            visit_land = visit_land.rio.set_spatial_dims("lat", "lon", inplace=True)

            interp_lat = data.lat.values
            interp_lon = data.lon.values
            land_mask = visit_land.interp(
                lat=interp_lat, lon=interp_lon, method="linear"
            )
            land_mask = land_mask.where(land_mask.mask != np.nan, 1)

        data = data * land_mask["mask"]
        data["driver"].plot(
            cmap=cmap,
            vmin=0,
            vmax=len(drivers) + 1,
            ax=ax,
            add_colorbar=False,
        )
        ax.set_title(m, fontsize=title_sz)
    axins = inset_axes(
        axes[2, 0],
        width="100%",  # width: 5% of parent_bbox width
        height="20%",  # height: 50%
        loc="center",
        bbox_to_anchor=(0.5, -0.75, 1, 1),
        bbox_transform=axes[2, 0].transAxes,
        borderpad=0,
    )
    if mode == "main":
        list_dom_drivers = ["co$_2$", "lulcc", "clim", "nan"]
        sup_cmap = mpl.colors.ListedColormap(
            [map_colors_dict[d] for d in list_dom_drivers]
        )
    elif mode == "clim":
        list_dom_drivers = ["tas", "rsds", "pr", "nan"]
        sup_cmap = mpl.colors.ListedColormap(
            [map_colors_dict[d] for d in list_dom_drivers]
        )

    center = [0.5 * (i * 2 + 1) for i in range(len(list_dom_drivers))]
    bounds = [i for i in range(len(list_dom_drivers) + 1)]
    norm = mpl.colors.BoundaryNorm(bounds, sup_cmap.N)
    sm = plt.cm.ScalarMappable(cmap=sup_cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm, cax=axins, shrink=1, ticks=center, orientation="horizontal", pad=0.05
    )
    cbar.set_ticklabels(list_dom_drivers, size=11)
    cbar.set_label("Dominant driver", size=unit_sz, weight="bold")


# Plot Fig. 14
def plt_inter_model_spreads(cmap="OrRd"):
    cmip6_files = sorted(glob.glob(os.path.join(RES_DIR, "mk", "*.nc")))
    visit_files = sorted(glob.glob(os.path.join(VISIT_DIR, "mk_1x1.25", "*.nc")))
    all_files = cmip6_files + visit_files
    list_var = ["co2f", "lulcc", "clim", "tas", "rsds", "pr"]
    list_index = [
        "(a) CO$_2$",
        "(b) LULCC",
        "(c) Clim",
        "(d) Temperature",
        "(e) Shortwave radiation",
        "(f) Precipitation",
    ]

    multi_models = {}

    for v in list_var:
        l_var = []
        for i, f in enumerate(all_files):
            if f"_{v}" in f:
                ds = xr.open_dataset(f)
                l_var.append(ds.rename({list(ds.keys())[0]: i}))
                print(f)
        print(v, len(l_var))
        if v == "co2":
            assert len(l_var) == 4
        multi_models[v] = (xr.merge(l_var)).to_array().std("variable")

    rows = 3
    cols = 2
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 2 * rows),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )

    vmin, vmax = 0, [25, 5]
    for i, (v, t) in enumerate(zip(list_var, list_index)):
        r = i % rows
        c = i // rows
        ax = axes[r, c]
        ax.coastlines()
        data = multi_models[v] * 1e3
        data = data.sel(lat=slice(-55.25, 82.75))
        data.plot.pcolormesh(
            ax=ax,
            cmap=cmap,
            levels=11,
            vmin=vmin,
            vmax=vmax[c],
            # extend="both",
            # cbar_kwargs={
            #     "label": "[$mgC  m^{-2}  yr^{-2}$]",
            #     "orientation": "horizontal",
            #     "pad": 0.05,
            # },
            add_colorbar=False,
        )
        ax.set_title(t)
    for i in [0, 1]:
        bounds = np.arange(vmin, vmax[i] + vmax[i] * 0.1, vmax[i] * 0.1)
        norm = mpl.colors.BoundaryNorm(bounds, mpl.cm.plasma.N, extend="both")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[:, i], shrink=0.7, location="bottom")
        cbar.set_label("[$mgC  m^{-2}  yr^{-2}$]", size=unit_sz)


# sup plt RF validation
def sup_plt_glob_rate(models):
    rows = 2
    cols = 3
    fig, axes = plt.subplots(
        rows,
        cols,
        sharey=True,
        figsize=(5 * cols, 3.5 * rows),
        layout="constrained",
    )

    for i, m in enumerate([m for m in models.keys() if "VISIT" not in m]):
        j, k = i // cols, i % cols
        ax = axes[j, k]
        pred_fields = ["reg", "emiisop"]
        r, p = pearsonr(models[m].sim_rate["reg"], models[m].sim_rate["emiisop"])
        rmse = mean_squared_error(
            models[m].sim_rate["reg"], models[m].sim_rate["emiisop"], squared=False
        )
        colors_list = ["#80b1d3", "#fb8072"]
        colors_dict = {
            m_name: c for m_name, c in zip(pred_fields, colors_list[: len(pred_fields)])
        }
        lss = ["-", "--", "-", "-."]
        ls_dict = {m_name: c for m_name, c in zip(pred_fields, lss[: len(pred_fields)])}
        for f in pred_fields:
            obj = models[m].sim_rate[f]
            x, y = obj.year, obj.values
            ax.plot(
                x,
                y,
                label=f,
                linewidth=2.5,
                ls=ls_dict[f],
                color=colors_dict[f],
            )
        ax.set_title(m, fontsize=title_sz)
        ax.set_ylim([350, 650])
        ax.annotate(
            f"r = {np.round(r, decimals=3)}\nrmse = {np.round(rmse, decimals=1)}",
            (0.8, 1.03),
            xycoords="axes fraction",
        )
        if j in [0, 1]:
            axes[j, 0].set_ylabel("[$TgC  yr^{-1}$]", fontsize=unit_sz)
    fig.delaxes(axes[-1][-1])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        fontsize=legend_sz,
    )
