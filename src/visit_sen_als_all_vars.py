# %%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import cartopy.crs as ccrs
import pymannkendall as pymk


from const import *
from mypath import *
from mk import *
from senAls_mulLinear import cal_actual_rate
from senAls_utils import prep_area
from utils import *
from copy import deepcopy


def cal_mk(ds, var_name):
    ds_var = ds[var_name]
    x = ds_var
    y = xr.DataArray(
        np.arange(len(ds_var["year"])) + 1,
        dims="year",
        coords={"year": ds_var["year"]},
    )
    slope = kendall_correlation(x, y, "year")
    return slope


class VisitSenAls:
    base_dir = VISIT_DIR
    list_sim = [
        "co2f",
        "co2f_clim",
        "co2f_clim_lulcc",
        "co2f_lulcc_rsds_pr",
        "co2f_lulcc_tas_pr",
        "co2f_lulcc_tas_rsds",
    ]
    clim_predictors = ["tas", "rsds", "pr"]
    list_pred = clim_predictors + ["clim"]

    def __init__(self, var_name="emiisop") -> None:
        self.var_name = var_name

        self.files = {
            var: os.path.join(self.base_dir, f"VISIT-SH{i}_{self.var_name}.nc")
            for i, var in enumerate(self.list_sim, 1)
        }

        self.ds_vars = {}
        self.mask = None
        self.org_mk = {}

        self.contribution_mk = {}
        self.clim_df_rate = pd.DataFrame()

        self.load_data()
        # self.cal_mask()
        self.cal_driver_mk()
        self.cal_glob_change_ts()
        self.plt_glob_rate_drivers()
        self.plt_glob_driver_pixel()
        self.plt_glob_rate()

    def load_data(self):
        for v in self.list_sim:
            self.ds_vars[v] = xr.open_dataset(self.files[v])

        clim = xr.Dataset({})
        # luc = xr.Dataset({})
        tas = xr.Dataset({})
        rsds = xr.Dataset({})
        pr = xr.Dataset({})

        # calculate the impacts of clim, tas, rsds, pr on isoprene emisison (varied driver - fixed driver case)
        clim[self.var_name] = (
            self.ds_vars["co2f_clim"][self.var_name]
            - self.ds_vars["co2f"][self.var_name]
        )

        # luc[self.var_name] = (
        #     self.ds_vars["co2_met_luc"][self.var_name]
        #     - self.ds_vars["co2_met"][self.var_name]
        # )

        tas[self.var_name] = (
            self.ds_vars["co2f_clim_lulcc"][self.var_name]
            - self.ds_vars["co2f_lulcc_rsds_pr"][self.var_name]
        )
        rsds[self.var_name] = (
            self.ds_vars["co2f_clim_lulcc"][self.var_name]
            - self.ds_vars["co2f_lulcc_tas_pr"][self.var_name]
        )
        pr[self.var_name] = (
            self.ds_vars["co2f_clim_lulcc"][self.var_name]
            - self.ds_vars["co2f_lulcc_tas_rsds"][self.var_name]
        )
        # clim[self.var_name] = (
        #     tas[self.var_name] + rsds[self.var_name] + pr[self.var_name]
        # )

        self.ds_vars["clim"] = clim
        # self.ds_vars["luc"] = luc
        self.ds_vars["tas"] = tas
        self.ds_vars["rsds"] = rsds
        self.ds_vars["pr"] = pr

    def cal_glob_change_ts(self):
        self.rates_ts = []
        slope_ts = []
        sig_ts = []
        for var in self.list_pred:
            ds = self.ds_vars[var][self.var_name]
            glob_rate_ts, glob_change_ts = cal_actual_rate(ds, "VISIT", mode="ts")
            self.rates_ts.append(glob_rate_ts)
            trend_test = pymk.original_test(glob_rate_ts.values, alpha=0.05)
            slope_ts.append(trend_test.slope)
            sig_ts.append(trend_test.h)
        self.clim_rates_ts = pd.DataFrame(
            {var: rate for var, rate in zip(self.list_pred, self.rates_ts)},
            index=[i for i in range(1850, 2015)],
        )
        self.clim_df_rate["driver"] = self.list_pred
        self.clim_df_rate["slope"] = slope_ts
        self.clim_df_rate["sig"] = sig_ts

    # def cal_mask(self):
    #     mask = self.ds_vars["co2_met_luc"][self.var_name].mean("year").values
    #     mask[mask > 0] = 1
    #     mask[mask <= 0] = np.nan
    #     self.mask = mask

    def cal_driver_mk(self):
        for v in self.list_pred:
            print(v)
            file_path_org = os.path.join(self.base_dir, "mk_0.5x0.5", f"{v}.nc")
            file_path_interp = os.path.join(self.base_dir, "mk_1x1.25", f"{v}.nc")
            if os.path.exists(file_path_org):
                ds = xr.open_dataset(file_path_org)
                ds = ds.rename(name_dict={list(ds.keys())[0]: v})
                self.org_mk[v] = ds[v]
            else:
                mk_ds = xr.Dataset({})
                mk_ds[v] = cal_mk(self.ds_vars[v], self.var_name)
                mk_ds.to_netcdf(file_path_org)
                self.org_mk[v] = mk_ds[v]
            # interpolate to 1*1.25 degree for plotting std map
            if not os.path.exists(file_path_interp):
                ds = deepcopy(self.org_mk[v])
                ds["lon"] = ds["lon"] % 360
                ds = ds.sortby(ds.lon)
                interpolate(ds).to_netcdf(file_path_interp)

            self.contribution_mk[v] = self.org_mk[v].sel(lat=slice(82.75, -55.25))
        self.ctb_clim_map, self.mk_impact_area_df = self.cal_drivers_pixel_level(
            self.contribution_mk
        )

    def cal_drivers_pixel_level(self, ds_pixel):
        clim = ds_pixel["clim"].values
        # luc = ds_pixel["luc"].values
        # co2 = ds_pixel["co2"].values
        tas = ds_pixel["tas"].values
        rsds = ds_pixel["rsds"].values
        pr = ds_pixel["pr"].values

        # co2 met luc
        # cml = ds_pixel["co2_met_luc"].values

        # stacked = np.stack((tas * cml, rsds * cml, pr * cml), axis=-1)
        # cml = ds_pixel["co2_met_luc"].values

        # make mask without nan values
        valid_mask = np.zeros(clim.shape)
        for arr in ds_pixel.values():
            arr = np.nan_to_num(arr.values)
            arr[arr != 0] = 1
            valid_mask += arr
        valid_mask[valid_mask < 1] = np.nan
        valid_mask[valid_mask > 0] = 1

        stacked = np.stack((tas, rsds, pr), axis=-1)
        stacked = np.nan_to_num(stacked)
        abs_stacked = np.absolute(stacked)
        max_stacked = np.nanargmax(abs_stacked, axis=-1)

        driver_arr = max_stacked * valid_mask
        coords = {
            "lon": ds_pixel["tas"].lon.values,
            "lat": ds_pixel["tas"].lat.values,
        }

        driver_abs_pixel = xr.Dataset(
            {"driver": (("lat", "lon"), driver_arr)}, coords=coords
        )
        # cal dominant impact by area
        # --- aggregate the impact by area ---
        processed_area = prep_area(driver_abs_pixel["driver"], "VISIT") * valid_mask
        processed_area = processed_area.to_dataset().assign(
            driver=driver_abs_pixel["driver"]
        )
        area_dict = {}
        total_a = 0
        for v, f in enumerate(self.clim_predictors):
            area = processed_area.where(processed_area["driver"] == v, drop=True)
            area = area["areacella"].sum(["lat", "lon"]).item()
            area_dict[f] = area
            total_a += area

        impact_by_area = [area_dict[f] * 100 / total_a for f in self.clim_predictors]
        impact_area_df = pd.DataFrame()
        impact_area_df["percentage"] = impact_by_area
        impact_area_df["driver"] = self.clim_predictors
        return driver_abs_pixel, impact_area_df

    def plt_glob_rate_drivers(self):
        fig, ax = plt.subplots(1, 1, figsize=(3, 4), layout="constrained")
        barplot = sns.barplot(
            self.clim_df_rate,
            x="driver",
            y="slope",
            ax=ax,
            palette=sns.color_palette(["#e31a1c", "#fee08b", "#386cb0", "#fb8072"]),
        )
        for p, sig in zip(barplot.patches, self.clim_df_rate["sig"]):
            if sig == True:
                h = p.get_height()
                add_h = 0.1
                h = h if h > 0 else h - add_h
                barplot.text(p.get_x() + p.get_width() / 2.0, h, "*", ha="center")
        ax.set_title("VISIT(G1997)")
        ax.set_xlabel(" ")
        ax.set_ylabel("Isoprene emission trends [$TgC yr^{-2}$]")
        ax.set_ylim(-0.25, 0.25)

    def plt_glob_driver_pixel(self):
        visit_land = xr.open_dataset(
            "/mnt/dg3/ngoc/cmip6_bvoc_als/data/axl/mask/mask_fx_VISIT-S3(G1997)_historical_r1i1p1f1_gn.nc"
        )
        land_mask = visit_land.where(visit_land.mask != np.nan, 1)
        abs_driver = self.ctb_clim_map
        abs_driver = abs_driver.fillna(len(self.clim_predictors))
        abs_driver = abs_driver * land_mask["mask"]
        abs_driver = abs_driver.sel(lat=slice(82.75, -55.25))

        fig = plt.figure(figsize=(3.75, 5))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines()
        title = f"VISIT(G1997)"  # - Dominant driver of trends in {self.var_name}"
        cmap = matplotlib.colors.ListedColormap(
            ["#e31a1c", "#ffff99", "#386cb0", "lightgrey"]
        )
        center = [0.5 * (i * 2 + 1) for i in range(len(self.clim_predictors) + 1)]
        cax = abs_driver["driver"].plot(
            cmap=cmap,
            vmin=0,
            vmax=len(self.clim_predictors) + 1,
            ax=ax,
            add_colorbar=False,
        )
        cbar = fig.colorbar(
            cax,
            ticks=center,
            orientation="horizontal",
            pad=0.05,
        )
        cbar.ax.set_xticklabels(self.clim_predictors + ["nan"], size=11)
        cbar.set_label(label="Dominant driver", size=9, weight="bold")
        plt.title(title, fontsize=11)

    def plt_contri_map(self, vmin=-2.5, vmax=2.5):
        i = 0
        for f in self.clim_predictors:
            i = i + 1
            fig = plt.figure(1 + i, figsize=(3.75, 5))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.coastlines()
            title = (
                f"VISIT(G1997)"  # - Contribution of {f} to the {self.var_name} trends"
            )
            data = self.contribution_mk[f] * 1e3
            data.plot.pcolormesh(
                ax=ax,
                cmap="bwr",
                levels=11,
                vmin=vmin,
                vmax=vmax,
                extend="both",
                cbar_kwargs={
                    "label": "[$mgC m^{-2} yr^{-2}$]",
                    "orientation": "horizontal",
                    "pad": 0.05,
                },
            )
            plt.title(title, fontsize=11)

    def plt_glob_rate(self):
        pred_fields = self.clim_predictors + ["clim"]
        colors_list = ["#e31a1c", "#fee08b", "#386cb0", "#fb8072"]
        colors_dict = {
            m_name: c for m_name, c in zip(pred_fields, colors_list[: len(pred_fields)])
        }
        lss = ["-", "-", "-", "--"]
        ls_dict = {m_name: c for m_name, c in zip(pred_fields, lss[: len(pred_fields)])}
        # lines = df.plot.line()
        fig, ax = plt.subplots(figsize=(5.5, 3.75), layout="constrained")
        axbox = ax.get_position()
        for v in pred_fields:
            obj = self.clim_rates_ts[v]
            x, y = obj.index, obj.values
            ax.plot(x, y, label=v, lw=2.5, color=colors_dict[v], ls=ls_dict[v])
        # ax.set_xlabel("Year", fontweight="bold", fontsize=14)
        ax.set_ylabel(
            "Isoprene emission changes [$TgC  yr^{-1}$]",
            # fontweight="bold",
            # fontsize=14,
        )
        ax.set_ylim([-65, 55])
        ax.set_title(f"VISIT(G1997)")  # - Drivers of Annual Trend of {self.var_name}")
        ax.legend(
            loc="center",
            ncol=len(pred_fields),
            bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.25],
        )


# %%
