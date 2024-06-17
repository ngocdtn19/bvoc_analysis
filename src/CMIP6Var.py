# %%
import xarray as xr
import calendar

# import rioxarray
import numpy as np
import matplotlib
import pandas as pd
import copy
import seaborn as sns
import regionmask
import geopandas as gpd

sns.set()

from utils import *
from const import *
from mypath import *


class CMIP6Var:
    crs = "EPSG:4326"

    def __init__(self, model_name, org_ds_var, var_name):
        self.model_name = model_name
        self.org_ds_var = org_ds_var
        self.var_name = var_name

        self.years = []

        self.ds_area = []
        self.ds_sftlf = []
        self.ds_mask = []
        self.total_land_area = None

        self.nodays_m = []

        self.monthly_ds = None
        self.monthly_per_area_unit = None

        self.annual_ds = None
        self.annual_per_area_unit = None

        self.weighted_monthly_ds = None
        self.weighted_annual_ds = None

        self.global_rate = []
        self.global_rate_anml = []

        self.seasonal_ds = []
        self.seasonal_per_area_unit = []
        self.global_seasonal_rate = {}
        self.global_seasonal_rate_anml = {}

        # self.roi_area = {}
        self.regional_ds = {}
        self.regional_rate = {}
        self.regional_mean_1850_2014 = {}
        self.regional_rate_anml = {}
        self.regional_ss_rate = {}
        self.regional_ss_rate_anml = {}

        self.cal_years()
        self.cal_nodays_m()

        self.get_ds_area()
        self.get_ds_sftlf()
        self.get_ds_mask()
        self.cal_glob_land_area()

        self.cal_monthly_ds()
        self.cal_monthly_per_area_unit()

        self.cal_annual_ds()
        self.cal_annual_per_area_unit()

        self.cal_seasonal_ds()
        self.cal_seasonal_per_area_unit()
        self.cal_global_seasonal_rate()  # cal global seasonal mean

        self.cal_glob_rate()  # cal global annual totals
        self.cal_reg_ann_rate()  # cal regional annual totals
        self.cal_reg_ss_rate()  # cal regional seasonal mean

    def cal_years(self):
        self.years = list(
            set(t.year for t in self.org_ds_var[DIM_TIME].values)
        )  # check again

    def get_ds_area(self):
        for f in AREA_LIST:
            # only compare 5 first characters of the model name bc we have n VISIT model test an
            # len ("VISIT")  = 5
            if self.model_name[:5] in f:
                self.ds_area = xr.open_dataset(f)

    def get_ds_sftlf(self):
        for f in SFLTF_LIST:
            if self.model_name[:5] in f:
                self.ds_sftlf = xr.open_dataset(f)

    def get_ds_mask(self):
        for f in MASK_LIST:
            if self.model_name[:5] in f:
                self.ds_mask = xr.open_dataset(f)

    def cal_glob_land_area(self):
        reindex_ds_lf = self.ds_sftlf[VAR_SFTLF].reindex_like(
            self.ds_area[VAR_AREA], method="nearest", tolerance=0.01
        )
        land_area = self.ds_area[VAR_AREA] * self.ds_mask[VAR_MASK]
        self.glob_land_area = land_area.sum().item()

    def cal_glob_rate(self):
        self.global_rate = self.annual_ds.sum(dim=[DIM_LAT, DIM_LON])
        self.global_rate_anml = self.global_rate - self.global_rate.mean(skipna=True)

    def cal_seasonal_ds(self):
        ds = self.monthly_ds
        self.seasonal_ds = ds.resample(time="QS-DEC").mean(skipna=True)

    def cal_global_seasonal_rate(self):  # unit: Tg/month - for plot trend
        ds = self.seasonal_ds
        self.global_seasonal_rate = ds.groupby("time").sum(dim=[DIM_LAT, DIM_LON])
        self.global_seasonal_rate_anml = self.global_seasonal_rate.groupby(
            "time.month"
        ) - self.global_seasonal_rate.groupby("time.month").mean(skipna=True)

    def cal_seasonal_per_area_unit(self):
        ds = self.monthly_per_area_unit
        self.seasonal_per_area_unit = ds.resample(time="QS-DEC").mean(skipna=True)

    def clip_2_roi_ds(self):
        ds = copy.deepcopy(self.monthly_ds)
        ds = ds.rio.write_crs("epsg:4326", inplace=True)

        ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180
        ds = ds.sortby(ds.lon)
        ds = ds.rio.set_spatial_dims("lon", "lat", inplace=True)
        subset = {}
        for roi in LIST_REGION:
            # subset[roi] = ds.rio.clip_box(
            #     minx=ROI_DICT[roi]["min_lon"],
            #     miny=ROI_DICT[roi]["min_lat"],
            #     maxx=ROI_DICT[roi]["max_lon"],
            #     maxy=ROI_DICT[roi]["max_lat"],
            #     crs=self.crs,
            # )
            subset[roi] = clip_region_mask(ds, roi)
        return subset

    def clip_2_roi_area(self):
        ds_area = copy.deepcopy(self.ds_area[VAR_AREA])
        ds_area = ds_area.rio.write_crs("epsg:4326", inplace=True)

        ds_area.coords["lon"] = (ds_area.coords["lon"] + 180) % 360 - 180
        ds_area = ds_area.sortby(ds_area.lon)
        ds_area = ds_area.rio.set_spatial_dims("lon", "lat", inplace=True)
        subset = {}
        for roi in LIST_REGION:
            # subset[roi] = ds_area.rio.clip_box(
            #     minx=ROI_DICT[roi]["min_lon"],
            #     miny=ROI_DICT[roi]["min_lat"],
            #     maxx=ROI_DICT[roi]["max_lon"],
            #     maxy=ROI_DICT[roi]["max_lat"],
            #     # crs=self.crs,
            # ).sum(dim=[DIM_LAT, DIM_LON], skipna=True)
            subset[roi] = clip_region_mask(ds_area, roi)
        return subset

    def resample(self):
        pass

    def cal_nodays_m(self):
        # this will be overwritten by child class
        return

    def cal_monthly_ds(self):
        # this will be overwritten by child class
        return

    def cal_monthly_per_area_unit(self):
        # this will be overwritten by child class
        return

    def cal_annual_ds(self):
        # this will be overwritten by child class
        return

    def cal_annual_per_area_unit(self):
        # this will be overwritten by child class
        return

    def cal_reg_ann_rate(self):
        # this will be overwritten by child class
        return

    def cal_reg_ss_rate(self):
        # this will be overwritten by child class
        return


class Cmip6TimeSum(CMIP6Var):
    n_month = 12
    noday_360 = [30] * n_month
    noday_noleap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)
        # self.cal_y10_rate()

    def cal_nodays_m(self):
        calendar = self.org_ds_var.time.dt.calendar

        l = int(len(self.org_ds_var.time) / self.n_month)
        if calendar == "360_day":
            nodays_m = np.array(self.noday_360 * l)
        else:
            nodays_m = np.array(self.noday_noleap * l)

        self.nodays_m = nodays_m

    def cal_monthly_per_area_unit(self):
        reindex_ds_lf = (
            self.ds_sftlf[VAR_SFTLF].reindex_like(
                self.org_ds_var[self.var_name], method="nearest", tolerance=0.01
            )
            * 1e-2
            if self.model_name == "UKESM1-0-LL"
            else 1
        )
        self.monthly_per_area_unit = (
            reindex_ds_lf
            * KG_2_G
            * ISOP_2_C
            * DAY_RATE
            * self.org_ds_var[self.var_name].transpose(..., "time")
            * self.nodays_m
        )

    def cal_annual_per_area_unit(self):
        ds = self.monthly_per_area_unit
        self.annual_per_area_unit = ds.groupby(ds.time.dt.year).sum("time", skipna=True)

    def cal_monthly_ds(self):
        reindex_ds_lf = (
            self.ds_sftlf[VAR_SFTLF].reindex_like(
                self.ds_area[VAR_AREA], method="nearest", tolerance=0.01
            )
            * 1e-2
            if self.model_name == "UKESM1-0-LL"
            else 1
        )
        reindex_ds_var = self.org_ds_var[self.var_name].reindex_like(
            self.ds_area[VAR_AREA], method="nearest", tolerance=0.01
        )
        self.monthly_ds = (
            reindex_ds_lf
            * self.ds_area[VAR_AREA]
            * DAY_RATE
            * KG_2_TG
            * ISOP_2_C
            * reindex_ds_var
            * self.nodays_m
        )

    def cal_annual_ds(self):
        ds = self.monthly_ds
        self.annual_ds = ds.groupby(ds.time.dt.year).sum(skipna=True)

    # def cal_y10_rate(self):
    #     y10 = []
    #     y10_rate = []
    #     interval = 10
    #     l = int(len(self.years) / interval) - 1

    #     for i in range(0, l):
    #         y10.append(self.years[interval * i])
    #         y10_rate.append(
    #             np.mean(self.global_rate[interval * i : interval * (i + 1)]).item()
    #         )

    #     y10.append(self.years[(l) * interval])
    #     y10_rate.append(np.mean(self.global_rate[interval * (l - 1) :]).item())

    #     self.y10_rate = {"years": y10, "y10_rate": y10_rate}

    def cal_reg_ann_rate(self):
        clipped_ds = self.clip_2_roi_ds()
        for roi in clipped_ds:
            mon_ds_roi = clipped_ds[roi]
            ann_ds_roi = mon_ds_roi.groupby(mon_ds_roi.time.dt.year).sum(skipna=True)

            self.regional_ds[roi] = ann_ds_roi
            self.regional_rate[roi] = ann_ds_roi.sum(dim=[DIM_LAT, DIM_LON])
            self.regional_mean_1850_2014[roi] = self.regional_rate[roi].mean(
                skipna=True
            )
            self.regional_rate_anml[roi] = (
                self.regional_rate[roi] - self.regional_mean_1850_2014[roi]
            )

    def cal_reg_ss_rate(self):
        clipped_ds = self.clip_2_roi_ds()
        for roi in clipped_ds:
            mon_ds_roi = clipped_ds[roi]
            ss_ds_roi = mon_ds_roi.resample(time="QS-DEC").mean(skipna=True)
            self.regional_ss_rate[roi] = (
                ss_ds_roi.groupby("time")
                .sum(dim=[DIM_LAT, DIM_LON])
                .reset_coords(["spatial_ref"], drop=True)
            )
            self.regional_ss_rate_anml[roi] = self.regional_ss_rate[roi].groupby(
                "time.month"
            ) - self.regional_ss_rate[roi].groupby("time.month").mean(skipna=True)
            self.regional_ss_rate_anml[roi] = self.regional_ss_rate_anml[
                roi
            ].reset_coords(["month"], drop=True)


class BVOC(Cmip6TimeSum):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_monthly_per_area_unit(self):
        self.monthly_per_area_unit = (
            KG_2_G
            * DAY_RATE
            * self.org_ds_var[self.var_name].transpose(..., "time")
            * self.nodays_m
        )

    def cal_monthly_ds(self):
        reindex_ds_var = self.org_ds_var[self.var_name].reindex_like(
            self.ds_area[VAR_AREA], method="nearest", tolerance=0.01
        )
        self.monthly_ds = (
            self.ds_area[VAR_AREA] * DAY_RATE * KG_2_TG * reindex_ds_var * self.nodays_m
        )


class PP(Cmip6TimeSum):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_monthly_per_area_unit(self):
        self.monthly_per_area_unit = (
            KG_2_G
            * DAY_RATE
            * self.org_ds_var[self.var_name].transpose(..., "time")
            * self.nodays_m
        )

    def cal_monthly_ds(self):
        reindex_ds_var = self.org_ds_var[self.var_name].reindex_like(
            self.ds_area[VAR_AREA], method="nearest", tolerance=0.01
        )
        self.monthly_ds = (
            self.ds_area[VAR_AREA] * DAY_RATE * KG_2_PG * reindex_ds_var * self.nodays_m
        )


class PR(CMIP6Var):
    kg_2_mm = 86400  # 1 kg/m2/s = 86400 mm/day.

    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_monthly_ds(self):
        reindex_ds_var = self.org_ds_var[self.var_name].reindex_like(
            self.ds_area[VAR_AREA], method="nearest", tolerance=0.01
        )
        # total_land_area = self.ds_area[VAR_AREA].sum().item()

        self.monthly_ds = (
            reindex_ds_var
            * self.kg_2_mm
            * self.ds_area[VAR_AREA]
            * self.ds_mask[VAR_MASK]
            / self.glob_land_area
        )

    def cal_annual_ds(self):
        ds = self.monthly_ds
        self.annual_ds = ds.groupby(ds.time.dt.year).mean(skipna=True)

    def cal_monthly_per_area_unit(self):
        # reindex_ds_var = self.org_ds_var[self.var_name].reindex_like(
        #     self.ds_area[VAR_AREA], method="nearest", tolerance=0.01
        # )
        self.monthly_per_area_unit = self.org_ds_var[self.var_name] * self.kg_2_mm

    def cal_annual_per_area_unit(self):
        ds = self.monthly_per_area_unit
        self.annual_per_area_unit = ds.groupby(ds.time.dt.year).mean(
            "time", skipna=True
        )

    def cal_reg_ann_rate(self):
        # total_area = self.ds_area[VAR_AREA].sum().item()
        clipped_ds = self.clip_2_roi_ds()
        clipped_area = self.clip_2_roi_area()
        for roi in clipped_ds:
            roi_area = clipped_area[roi].sum(dim=[DIM_LAT, DIM_LON])
            mon_ds_roi = clipped_ds[roi] * self.glob_land_area / roi_area
            ann_ds_roi = mon_ds_roi.groupby(mon_ds_roi.time.dt.year).mean(skipna=True)

            self.regional_ds[roi] = ann_ds_roi
            self.regional_rate[roi] = ann_ds_roi.sum(
                dim=[DIM_LAT, DIM_LON], skipna=True
            )
            self.regional_mean_1850_2014[roi] = self.regional_rate[roi].mean(
                skipna=True
            )
            self.regional_rate_anml[roi] = (
                self.regional_rate[roi] - self.regional_mean_1850_2014[roi]
            )

    def cal_reg_ss_rate(self):
        # total_area = self.ds_area[VAR_AREA].sum().item()
        clipped_ds = self.clip_2_roi_ds()
        clipped_area = self.clip_2_roi_area()
        for roi in clipped_ds:
            roi_area = clipped_area[roi].sum(dim=[DIM_LAT, DIM_LON])
            mon_ds_roi = clipped_ds[roi] * self.glob_land_area / roi_area
            ss_ds_roi = mon_ds_roi.resample(time="QS-DEC").mean(skipna=True)
            self.regional_ss_rate[roi] = (
                ss_ds_roi.groupby("time")
                .sum(dim=[DIM_LAT, DIM_LON])
                .reset_coords(["spatial_ref"], drop=True)
            )
            self.regional_ss_rate_anml[roi] = self.regional_ss_rate[roi].groupby(
                "time.month"
            ) - self.regional_ss_rate[roi].groupby("time.month").mean(skipna=True)
            self.regional_ss_rate_anml[roi] = self.regional_ss_rate_anml[
                roi
            ].reset_coords(["month"], drop=True)


class TAS(CMIP6Var):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_monthly_ds(self):
        # total_area = self.ds_area[VAR_AREA].sum().item()

        reindex_ds_var = self.org_ds_var[self.var_name].reindex_like(
            self.ds_area[VAR_AREA], method="nearest", tolerance=0.01
        )
        self.monthly_ds = (
            (reindex_ds_var - K_2_C)
            * self.ds_area[VAR_AREA]
            * self.ds_mask[VAR_MASK]
            / self.glob_land_area
        )

    def cal_monthly_per_area_unit(self):
        self.monthly_per_area_unit = (
            copy.deepcopy(self.org_ds_var[self.var_name]) - K_2_C
        )

    def cal_annual_ds(self):
        ds = self.monthly_ds[:]
        self.annual_ds = ds.groupby(ds.time.dt.year).mean("time", skipna=True)

    def cal_annual_per_area_unit(self):
        ds = self.monthly_per_area_unit
        self.annual_per_area_unit = ds.groupby(ds.time.dt.year).mean(
            "time", skipna=True
        )

    def cal_reg_ann_rate(self):
        # total_area = self.ds_area[VAR_AREA].sum().item()
        clipped_ds = self.clip_2_roi_ds()
        clipped_area = self.clip_2_roi_area()
        for roi in clipped_ds:
            roi_area = clipped_area[roi].sum(dim=[DIM_LAT, DIM_LON])
            mon_ds_roi = clipped_ds[roi] * self.glob_land_area / roi_area
            ann_ds_roi = mon_ds_roi.groupby(mon_ds_roi.time.dt.year).mean(skipna=True)

            self.regional_ds[roi] = ann_ds_roi
            self.regional_rate[roi] = ann_ds_roi.sum(
                dim=[DIM_LAT, DIM_LON], skipna=True
            )
            self.regional_mean_1850_2014[roi] = self.regional_rate[roi].mean(
                skipna=True
            )
            self.regional_rate_anml[roi] = (
                self.regional_rate[roi] - self.regional_mean_1850_2014[roi]
            )

    def cal_reg_ss_rate(self):
        # total_area = self.ds_area[VAR_AREA].sum().item()
        clipped_ds = self.clip_2_roi_ds()
        clipped_area = self.clip_2_roi_area()
        for roi in clipped_ds:
            roi_area = clipped_area[roi].sum(dim=[DIM_LAT, DIM_LON])
            mon_ds_roi = clipped_ds[roi] * self.glob_land_area / roi_area
            ss_ds_roi = mon_ds_roi.resample(time="QS-DEC").mean(skipna=True)
            self.regional_ss_rate[roi] = (
                ss_ds_roi.groupby("time")
                .sum(dim=[DIM_LAT, DIM_LON])
                .reset_coords(["spatial_ref"], drop=True)
            )
            self.regional_ss_rate_anml[roi] = self.regional_ss_rate[roi].groupby(
                "time.month"
            ) - self.regional_ss_rate[roi].groupby("time.month").mean(skipna=True)
            self.regional_ss_rate_anml[roi] = self.regional_ss_rate_anml[
                roi
            ].reset_coords(["month"], drop=True)


class RSDS(TAS):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_monthly_ds(self):
        # total_area = self.ds_area[VAR_AREA].sum().item()

        reindex_ds_var = self.org_ds_var[self.var_name].reindex_like(
            self.ds_area[VAR_AREA], method="nearest", tolerance=0.01
        )
        self.monthly_ds = (
            reindex_ds_var
            * self.ds_area[VAR_AREA]
            * self.ds_mask[VAR_MASK]
            / self.glob_land_area
        )

    def cal_monthly_per_area_unit(self):
        self.monthly_per_area_unit = copy.deepcopy(self.org_ds_var[self.var_name])


class EMIOA(Cmip6TimeSum):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_monthly_ds(self):
        reindex_ds_var = self.org_ds_var[self.var_name].reindex_like(
            self.ds_area[VAR_AREA], method="nearest", tolerance=0.01
        )
        self.monthly_ds = (
            self.ds_area[VAR_AREA] * DAY_RATE * KG_2_TG * reindex_ds_var * self.nodays_m
        )

    def cal_annual_ds(self):
        ds = self.monthly_ds
        self.annual_ds = ds.groupby(ds.time.dt.year).sum()

    def cal_monthly_per_area_unit(self):
        self.monthly_per_area_unit = (
            KG_2_G
            * DAY_RATE
            * self.org_ds_var[self.var_name].transpose(..., "time")
            * self.nodays_m
        )


class CHEPSOA(EMIOA):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)


class EMIPOA(EMIOA):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)


class MRSO(CMIP6Var):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_monthly_ds(self):
        # self.monthly_ds = self.org_ds_var[self.var_name].where(
        #     self.org_ds_var[self.var_name] > 0
        # )
        self.monthly_ds = self.org_ds_var[self.var_name]

    def cal_monthly_per_area_unit(self):
        # self.monthly_per_area_unit = self.org_ds_var[self.var_name].where(
        #     self.org_ds_var[self.var_name] > 0
        # )
        self.monthly_per_area_unit = self.org_ds_var[self.var_name]

    def cal_annual_ds(self):
        ds = self.monthly_ds[:]
        self.annual_ds = ds.groupby(ds.time.dt.year).mean("time", skipna=True)

    def cal_annual_per_area_unit(self):
        ds = self.monthly_per_area_unit
        self.annual_per_area_unit = ds.groupby(ds.time.dt.year).mean(
            "time", skipna=True
        )

    def cal_glob_rate(self):
        self.global_rate = self.annual_ds.mean(dim=[DIM_LAT, DIM_LON])
        self.global_rate_anml = self.global_rate - self.global_rate.mean(skipna=True)

    def cal_global_seasonal_rate(self):
        ds = self.seasonal_ds
        self.global_seasonal_rate = ds.groupby("time").mean(dim=[DIM_LAT, DIM_LON])
        self.global_seasonal_rate_anml = self.global_seasonal_rate.groupby(
            "time.month"
        ) - self.global_seasonal_rate.groupby("time.month").mean(skipna=True)

    def cal_reg_ann_rate(self):
        clipped_ds = self.clip_2_roi_ds()
        for roi in clipped_ds:
            mon_ds_roi = clipped_ds[roi]
            self.regional_ds[roi] = mon_ds_roi.groupby(mon_ds_roi.time.dt.year).mean(
                skipna=True
            )
            self.regional_rate[roi] = self.regional_ds[roi].mean(dim=[DIM_LAT, DIM_LON])
            self.regional_mean_1850_2014[roi] = self.regional_rate[roi].mean(
                skipna=True
            )
            self.regional_rate_anml[roi] = (
                self.regional_rate[roi] - self.regional_mean_1850_2014[roi]
            )

    def cal_reg_ss_rate(self):
        clipped_ds = self.clip_2_roi_ds()
        for roi in clipped_ds:
            mon_ds_roi = clipped_ds[roi]
            ss_ds_roi = mon_ds_roi.resample(time="QS-DEC").mean(skipna=True)
            self.regional_ss_rate[roi] = (
                ss_ds_roi.groupby("time")
                .mean(dim=[DIM_LAT, DIM_LON])
                .reset_coords(["spatial_ref"], drop=True)
            )
            self.regional_ss_rate_anml[roi] = self.regional_ss_rate[roi].groupby(
                "time.month"
            ) - self.regional_ss_rate[roi].groupby("time.month").mean(skipna=True)
            self.regional_ss_rate_anml[roi] = self.regional_ss_rate_anml[
                roi
            ].reset_coords(["month"], drop=True)


class MRSOS(MRSO):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)


class LAI(MRSO):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)


class CO2mass(CMIP6Var):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_glob_rate(self):
        # unit conversion from mass (kg) to ppm
        ds = self.org_ds_var[self.var_name] * 28.966 / (44 * 5.148e18) * 1e6
        self.global_rate = ds.groupby(ds.time.dt.year).mean(skipna=True)
        self.global_rate_anml = self.global_rate - self.global_rate.mean(skipna=True)

    def clip_2_roi_ds(self):
        return

    def cal_reg_ann_rate(self):
        return

    def cal_seasonal_ds(self):
        ds = self.org_ds_var[self.var_name] * 28.966 / (44 * 5.148e18) * 1e6
        self.seasonal_ds = ds.resample(time="QS-DEC").mean(skipna=True)

    def cal_global_seasonal_rate(self):
        ds = self.seasonal_ds
        self.global_seasonal_rate = ds
        self.global_seasonal_rate_anml = self.global_seasonal_rate.groupby(
            "time.month"
        ) - self.global_seasonal_rate.groupby("time.month").mean(skipna=True)

    def cal_annual_per_area_unit(self):
        return

    def cal_seasonal_per_area_unit(self):
        return


class CO2s(RSDS):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)


# %%


# unit function for clipping to region mask
