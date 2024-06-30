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
        "control",
        "co2f",
        "co2f_clim",
        "co2f_clim_lulcc",
        "co2f_lulcc_rsds_pr",
        "co2f_lulcc_tas_pr",
        "co2f_lulcc_tas_rsds",
    ]
    list_main_driver = ["co2f", "lulcc", "clim"]
    clim_predictors = ["tas", "rsds", "pr"]

    def __init__(self, var_name="emiisop") -> None:
        self.var_name = var_name

        self.files = {
            var: os.path.join(self.base_dir, f"VISIT-SH{i}_{self.var_name}.nc")
            for i, var in enumerate(self.list_sim, 0)
        }

        self.ds_vars = {}
        self.org_mk = {}

        self.contribution_mk = {}
        self.main_df_rate = pd.DataFrame()

        self.load_data()
        self.contribution_mk_cal()

        self.main_rates_ts, self.main_df_rate = self.cal_glob_change_ts("main")
        self.clim_rates_ts, self.clim_df_rate = self.cal_glob_change_ts("clim")

        self.ctb_main_map, self.mk_impact_area_df = self.max_impact_cal("main")
        self.ctb_clim_map, self.mk_impact_area_df = self.max_impact_cal("clim")

    def load_data(self):
        for v in self.list_sim:
            self.ds_vars[v] = xr.open_dataset(self.files[v])

        co2f = xr.Dataset({})
        clim = xr.Dataset({})
        lulcc = xr.Dataset({})
        cml = xr.Dataset({})

        tas = xr.Dataset({})
        rsds = xr.Dataset({})
        pr = xr.Dataset({})

        co2f[self.var_name] = (
            self.ds_vars["co2f"][self.var_name] - self.ds_vars["control"][self.var_name]
        )

        clim[self.var_name] = (
            self.ds_vars["co2f_clim"][self.var_name]
            - self.ds_vars["co2f"][self.var_name]
        )

        lulcc[self.var_name] = (
            self.ds_vars["co2f_clim_lulcc"][self.var_name]
            - self.ds_vars["co2f_clim"][self.var_name]
        )

        cml[self.var_name] = (
            self.ds_vars["co2f_clim_lulcc"][self.var_name]
            - self.ds_vars["control"][self.var_name]
        )

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

        self.ds_vars["clim"] = clim
        self.ds_vars["lulcc"] = lulcc
        self.ds_vars["co2f"] = co2f
        self.ds_vars["all"] = cml

        self.ds_vars["tas"] = tas
        self.ds_vars["rsds"] = rsds
        self.ds_vars["pr"] = pr

    def get_list_driver(self, mode):
        list_driver = self.list_main_driver + ["all"]
        if mode != "main":
            list_driver = self.clim_predictors + ["clim"]

        return list_driver

    def cal_glob_change_ts(self, mode):

        list_driver = self.get_list_driver(mode)

        driver_df_rate = pd.DataFrame()
        rate_ts = []
        slope_ts = []
        sig_ts = []
        for var in list_driver:
            ds = self.ds_vars[var][self.var_name]
            glob_rate_ts, _ = cal_actual_rate(ds, "VISIT", mode="ts")
            rate_ts.append(glob_rate_ts)
            trend_test = pymk.original_test(glob_rate_ts.values, alpha=0.05)
            slope_ts.append(trend_test.slope)
            sig_ts.append(trend_test.h)
        driver_rates_ts = pd.DataFrame(
            {var: rate for var, rate in zip(list_driver, rate_ts)},
            index=[i for i in range(1850, 2015)],
        )
        driver_df_rate["driver"] = list_driver
        driver_df_rate["slope"] = slope_ts
        driver_df_rate["sig"] = sig_ts

        return driver_rates_ts, driver_df_rate

    def contribution_mk_cal(self):
        list_driver = self.list_main_driver + self.clim_predictors
        for v in list_driver:
            print(v)
            file_path_org = os.path.join(self.base_dir, "mk_0.5x0.5", f"{v}.nc")
            file_path_interp = os.path.join(
                self.base_dir, "mk_1x1.25", f"VISIT(G1997)_{v}.nc"
            )
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

    def max_impact_cal(self, mode):

        list_driver = self.list_main_driver if mode == "main" else self.clim_predictors

        # make mask without nan values
        valid_mask = np.zeros(self.contribution_mk[list_driver[0]].shape)
        for arr in self.contribution_mk.values():
            arr = np.nan_to_num(arr.values)
            arr[arr != 0] = 1
            valid_mask += arr
        valid_mask[valid_mask < 1] = np.nan
        valid_mask[valid_mask > 0] = 1

        stacked = np.stack(tuple(self.contribution_mk[v] for v in list_driver), axis=-1)
        stacked = np.nan_to_num(stacked)
        abs_stacked = np.absolute(stacked)
        max_stacked = np.nanargmax(abs_stacked, axis=-1)

        driver_arr = max_stacked * valid_mask
        coords = {
            "lon": self.contribution_mk[list_driver[0]].lon.values,
            "lat": self.contribution_mk[list_driver[0]].lat.values,
        }

        driver_abs_pixel = xr.Dataset(
            {"driver": (("lat", "lon"), driver_arr)}, coords=coords
        )

        # --- aggregate the impact by area ---
        processed_area = prep_area(driver_abs_pixel["driver"], "VISIT") * valid_mask
        processed_area = processed_area.to_dataset().assign(
            driver=driver_abs_pixel["driver"]
        )
        area_dict = {}
        total_a = 0
        for v, f in enumerate(list_driver):
            area = processed_area.where(processed_area["driver"] == v, drop=True)
            area = area["areacella"].sum(["lat", "lon"]).item()
            area_dict[f] = area
            total_a += area

        impact_by_area = [area_dict[f] * 100 / total_a for f in list_driver]
        impact_area_df = pd.DataFrame()
        impact_area_df["percentage"] = impact_by_area
        impact_area_df["driver"] = list_driver

        return driver_abs_pixel, impact_area_df


# %%
visit = VisitSenAls()
