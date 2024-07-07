# %%
import xarray as xr
import cftime
import numpy as np
import copy

from datetime import datetime, timedelta
from CMIP6Var import CMIP6Var

from const import *
from mypath import *


def year_2_cft(dcm_year):
    year = int(dcm_year)
    rem = dcm_year - year

    base = datetime(year, 1, 1)
    dt = base + timedelta(
        seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem
    )

    cft = cftime.datetime(dt.year, dt.month, dt.day)

    return cft


def visit_t2cft(visit_nc, var_name, m_name="VISIT_ORG"):
    org_visit_ds = xr.open_dataset(visit_nc, decode_times=False)
    cft = [year_2_cft(org_time) for org_time in org_visit_ds.time.values]
    org_visit_ds.coords["time"] = cft

    if var_name == "emiisop":
        if "org" in m_name.lower():
            org_visit_ds = org_visit_ds.rename({"Isprn": var_name})
        else:
            org_visit_ds = org_visit_ds.rename({"isopr": var_name})

    org_visit_ds = org_visit_ds.where(
        org_visit_ds[var_name].sel(time=slice("1850-01", "2014-12"))
    )
    org_visit_ds = org_visit_ds.where(org_visit_ds[var_name] != -9999.0)
    org_visit_ds = org_visit_ds.where(org_visit_ds[var_name] != -99999.0)
    return org_visit_ds


def grid_area(lat1, lat2, lon1, lon2):
    E_RAD = 6378137.0
    # m, GRS-80(revised)
    E_FLAT = 298.257
    PI = 3.1415926
    E_EXC = math.sqrt(2.0 / E_FLAT - 1.0 / (E_FLAT * E_FLAT))

    if lat1 > 90.0:
        lat1 = 90.0
    if lat2 < -90.0:
        lat2 = -90.0

    m_lat = (lat1 + lat2) / 2.0 * PI / 180.0

    aa1 = 1.0 - E_EXC * E_EXC * math.sin(m_lat) * math.sin(m_lat)
    l_lat = (
        PI
        / 180.0
        * E_RAD
        * (1.0 - E_EXC * E_EXC)
        / math.pow(aa1, 1.5)
        * math.fabs(lat1 - lat2)
    )

    aa2 = 1.0 - E_EXC * E_EXC * math.sin(lat1 * PI / 180.0) * math.sin(
        lat1 * PI / 180.0
    )
    l_lon1 = (
        PI
        / 180.0
        * E_RAD
        * math.cos(lat1 * PI / 180.0)
        / math.sqrt(aa2)
        * math.fabs(lon1 - lon2)
    )
    aa3 = 1.0 - E_EXC * E_EXC * math.sin(lat2 * PI / 180.0) * math.sin(
        lat2 * PI / 180.0
    )
    l_lon2 = (
        PI
        / 180.0
        * E_RAD
        * math.cos(lat2 * PI / 180.0)
        / math.sqrt(aa3)
        * math.fabs(lon1 - lon2)
    )

    area = (l_lon1 + l_lon2) * l_lat / 2.0

    return area


def cal_ds_area(visit_nc="../data/VISIT/visit_20160105_BVOCisprn.nc"):
    ds = xr.open_dataset(visit_nc, decode_times=False)
    nlat = ds.lat.values.reshape(-1)
    nlon = ds.lon.values.reshape(-1)
    garea = []
    ds_area = {}

    final_arr = []
    for g in range(0, len(nlat)):
        glat = 89.75 - 0.5 * g
        garea.append(grid_area(glat + 0.25, glat - 0.25, 0.0, 0.5))
    arr = np.array(garea)

    _ = [final_arr.append([i] * len(nlon)) for i in arr]
    data = np.array(final_arr)
    ds_area = xr.Dataset(
        data_vars=dict(areacella=(["lat", "lon"], data)),
        coords=dict(
            lat=nlat,
            lon=nlon,
        ),
    )

    return ds_area


# def save_2_nc(org_visit_ds):
#     var_name = "emiisop"
#     m_name = "VISIT"
#     org_visit_ds.to_netcdf(f"../data/var/{var_name}/{var_name}_AERmon_{m_name}_historical_r1i1p1f1_gn_190101-201512.nc")


class CMIP6Visit(CMIP6Var):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)
        # self.cal_y10_rate()

    # def get_ds_sftlf(self):
    #     return

    def cal_monthly_per_area_unit(self):
        self.monthly_per_area_unit = self.org_ds_var[self.var_name] * MG_2_G

    def cal_annual_per_area_unit(self):
        ds = self.monthly_per_area_unit
        self.annual_per_area_unit = ds.groupby(ds.time.dt.year).sum(skipna=True)

    def cal_monthly_ds(self):
        reindex_ds_var = self.org_ds_var[self.var_name].reindex_like(
            self.ds_area[VAR_AREA], method="nearest", tolerance=0.01
        )
        self.monthly_ds = self.ds_area[VAR_AREA] * reindex_ds_var * MG_2_TG

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


class VisitTAS(CMIP6Var):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_monthly_ds(self):
        # self.monthly_ds = copy.deepcopy(self.org_ds_var[self.var_name])
        # recheck global estimate based on grid area - 04/03/2024
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

    def cal_annual_ds(self):
        # ws = np.cos(np.deg2rad(self.org_ds_var.lat))
        # ds = self.monthly_ds[:]
        # ds = ds.groupby(ds.time.dt.year).mean("time")
        # self.annual_ds = ds.weighted(ws)
        # recheck global estimate based on grid area - 04/03/2024
        ds = self.monthly_ds[:]
        self.annual_ds = ds.groupby(ds.time.dt.year).mean("time", skipna=True)

    def cal_annual_per_area_unit(self):
        ds = self.monthly_per_area_unit
        self.annual_per_area_unit = ds.groupby(ds.time.dt.year).mean(
            "time", skipna=True
        )

    # def cal_global_seasonal_rate(self):
    #     ds = self.seasonal_ds
    #     ws = np.cos(np.deg2rad(self.org_ds_var.lat))
    #     self.global_seasonal_rate = ds.weighted(ws).mean(dim=[DIM_LAT, DIM_LON])
    #     self.global_seasonal_rate_anml = (
    #         ds.weighted(ws).mean(dim=[DIM_LAT, DIM_LON]).groupby("time.month")
    #         - ds.weighted(ws).mean(dim=[DIM_LAT, DIM_LON]).groupby("time.month").mean()
    #     )

    # def cal_glob_rate(self):
    #     self.global_rate = self.annual_ds.sum(dim=[DIM_LAT, DIM_LON], skipna=True)
    #     self.global_rate_anml = self.global_rate - self.global_rate.mean(skipna=True)

    def cal_reg_ann_rate(self):
        # clipped_ds = self.clip_2_roi_ds()
        # for roi in clipped_ds:
        #     mon_ds_roi = clipped_ds[roi]
        #     ann_ds_roi = mon_ds_roi.groupby(mon_ds_roi.time.dt.year).mean(skipna=True)
        #     ws = np.cos(np.deg2rad(ann_ds_roi.lat))
        #     ann_ds_roi = ann_ds_roi.weighted(ws)

        #     self.regional_ds[roi] = ann_ds_roi
        #     self.regional_rate[roi] = ann_ds_roi.mean(
        #         dim=[DIM_LAT, DIM_LON], skipna=True
        #     )
        #     self.regional_mean_1850_2014[roi] = self.regional_rate[roi].mean(
        #         skipna=True
        #     )
        #     self.regional_rate_anml[roi] = (
        #         self.regional_rate[roi] - self.regional_mean_1850_2014[roi]
        #     )
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
        # clipped_ds = self.clip_2_roi_ds()
        # for roi in clipped_ds:
        #     mon_ds_roi = clipped_ds[roi]
        #     ss_ds_roi = mon_ds_roi.resample(time="QS-DEC").mean(skipna=True)
        #     ws = np.cos(np.deg2rad(ss_ds_roi.lat))

        #     self.regional_ss_rate[roi] = ss_ds_roi.weighted(ws).mean(
        #         dim=[DIM_LAT, DIM_LON]
        #     )
        #     self.regional_ss_rate_anml[roi] = self.regional_ss_rate[roi].groupby(
        #         "time.month"
        #     ) - self.regional_ss_rate[roi].groupby("time.month").mean(skipna=True)

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


class VisitRSDS(VisitTAS):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)


class VisitMRSO(VisitTAS):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_monthly_ds(self):
        self.monthly_ds = self.org_ds_var[self.var_name].where(
            self.org_ds_var[self.var_name] > 0
        )


class VisitMRSOS(VisitTAS):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_monthly_ds(self):
        self.monthly_ds = self.org_ds_var[self.var_name].where(
            self.org_ds_var[self.var_name] > 0
        )

    def cal_monthly_per_area_unit(self):
        self.monthly_per_area_unit = self.monthly_ds


class VisitLAI(VisitTAS):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)


class VisitPR(VisitTAS):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_nodays_m(self):
        self.nodays_m = self.org_ds_var.time.dt.days_in_month

    def cal_monthly_ds(self):
        monthly_ds = self.org_ds_var[self.var_name] / self.nodays_m
        self.monthly_ds = (
            monthly_ds
            * self.ds_area[VAR_AREA]
            * self.ds_mask[VAR_MASK]
            / self.glob_land_area
        )

    def cal_monthly_per_area_unit(self):
        self.monthly_per_area_unit = self.org_ds_var[self.var_name] / self.nodays_m


class VisitGPP(CMIP6Visit):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_monthly_per_area_unit(self):
        self.monthly_per_area_unit = (
            self.org_ds_var[self.var_name] * 1e2
        )  # convert unit from Megagram per ha to gram per m2

    def cal_monthly_ds(self):
        reindex_ds_var = self.org_ds_var[self.var_name].reindex_like(
            self.ds_area[VAR_AREA], method="nearest", tolerance=0.01
        )
        self.monthly_ds = (
            self.ds_area[VAR_AREA] * reindex_ds_var * 1e-13
        )  # convert unit to Petagram


class VisitBVOC(CMIP6Visit):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)


class VisitCO2s(VisitTAS):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)


# %%
