# %%
import xarray as xr
import calendar

# import rioxarray
import numpy as np
import matplotlib as mpl
import pandas as pd
import copy
import seaborn as sns
import regionmask
import geopandas as gpd


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

        self.global_rate = []
        self.global_rate_anml = []

        self.regional_ds = {}
        self.regional_rate = {}

        self.cal_nodays_m()

        self.get_ds_area()
        self.get_ds_sftlf()
        self.get_ds_mask()
        self.cal_glob_land_area()

        self.cal_monthly_ds()
        self.cal_monthly_per_area_unit()

        self.cal_annual_ds()
        self.cal_annual_per_area_unit()

        self.cal_glob_rate()  # cal global annual totals
        self.cal_reg_ann_rate()  # cal regional annual totals

    def get_ds_area(self):
        for f in AREA_LIST:
            # only compare 5 first characters of the model name bc we have n VISIT model test an
            # len ("VISIT")  = 5
            if self.model_name[:5] in f:
                self.ds_area = xr.open_dataset(f)

    def get_ds_sftlf(self):
        """
        Get the grid land fraction of each model
        """
        for f in SFLTF_LIST:
            if self.model_name[:5] in f:
                self.ds_sftlf = xr.open_dataset(f)

    def get_ds_mask(self):
        """
        Get the land mask of each model
        """
        for f in MASK_LIST:
            if self.model_name[:5] in f:
                self.ds_mask = xr.open_dataset(f)

    def cal_glob_land_area(self):
        land_area = self.ds_area[VAR_AREA] * self.ds_mask[VAR_MASK]
        self.glob_land_area = land_area.sum().item()

    def cal_glob_rate(self):
        self.global_rate = self.annual_ds.sum(dim=[DIM_LAT, DIM_LON])
        self.global_rate_anml = self.global_rate - self.global_rate.mean(skipna=True)

    def clip_2_roi_ds(self):
        """
        Clip the monthly dataset of a variable for the SREX regions
        """
        ds = copy.deepcopy(self.monthly_ds)
        ds = ds.rio.write_crs("epsg:4326", inplace=True)

        ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180
        ds = ds.sortby(ds.lon)
        ds = ds.rio.set_spatial_dims("lon", "lat", inplace=True)
        subset = {}
        for roi in LIST_REGION:
            subset[roi] = clip_region_mask(ds, roi)
        return subset

    def clip_2_roi_area(self):
        """
        Clip the grid area dataset for the SREX regions
        """
        ds_area = copy.deepcopy(self.ds_area[VAR_AREA])
        ds_area = ds_area.rio.write_crs("epsg:4326", inplace=True)

        ds_area.coords["lon"] = (ds_area.coords["lon"] + 180) % 360 - 180
        ds_area = ds_area.sortby(ds_area.lon)
        ds_area = ds_area.rio.set_spatial_dims("lon", "lat", inplace=True)
        subset = {}
        for roi in LIST_REGION:
            subset[roi] = clip_region_mask(ds_area, roi)
        return subset

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


class EMIISOP(CMIP6Var):
    """
    Preprocessing isoprene emissions for each model (1 variable - 1 model)
    """

    n_month = 12
    noday_360 = [30] * n_month
    noday_noleap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_nodays_m(self):
        """
        Calculate the number of days in each month according to the calendar defined by each model
        """
        calendar = self.org_ds_var.time.dt.calendar

        l = int(len(self.org_ds_var.time) / self.n_month)
        if calendar == "360_day":
            nodays_m = np.array(self.noday_360 * l)
        else:
            nodays_m = np.array(self.noday_noleap * l)

        self.nodays_m = nodays_m

    def cal_monthly_per_area_unit(self):
        """
        Unit conversion from original data (kg m⁻² s⁻¹) to (gC m⁻² month⁻¹)
        """
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
        """
        Calculate the annual per area unit dataset (gC m⁻² yr⁻¹) from monthly per area unit dataset (gC m⁻² month⁻¹)
        """
        ds = self.monthly_per_area_unit
        self.annual_per_area_unit = ds.groupby(ds.time.dt.year).sum("time", skipna=True)

    def cal_monthly_ds(self):
        """
        Unit conversion from original data (kg m⁻² s⁻¹) to (TgC month⁻¹)
        """
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
        """
        Calculate the annual dataset of emission (TgC yr⁻¹) or GPP (PgC yr⁻¹) from monthly dataset (TgC month⁻¹)
        """
        ds = self.monthly_ds
        self.annual_ds = ds.groupby(ds.time.dt.year).sum(skipna=True)

    def cal_reg_ann_rate(self):
        """
        Calculate the regional annual rate of emission (TgC yr⁻¹) or GPP (PgC yr⁻¹) from clipped (regional) monthly data
        """
        clipped_ds = self.clip_2_roi_ds()
        for roi in clipped_ds:
            mon_ds_roi = clipped_ds[roi]
            ann_ds_roi = mon_ds_roi.groupby(mon_ds_roi.time.dt.year).sum(skipna=True)

            self.regional_ds[roi] = ann_ds_roi
            self.regional_rate[roi] = ann_ds_roi.sum(dim=[DIM_LAT, DIM_LON])


class PP(EMIISOP):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_monthly_per_area_unit(self):
        """
        Unit conversion from original data (kgC m⁻² s⁻¹) to (gC m⁻² month⁻¹)
        """
        self.monthly_per_area_unit = (
            KG_2_G
            * DAY_RATE
            * self.org_ds_var[self.var_name].transpose(..., "time")
            * self.nodays_m
        )

    def cal_monthly_ds(self):
        """
        Unit conversion from original data (kgC m⁻² s⁻¹) to (PgC month⁻¹)
        """
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
        """
        Calculate the area-weighted monthly ds of precipitation (mm day⁻¹) from original data (kg m⁻² s⁻¹)
        This data will be used to calculate annual ds and global rate
        """
        reindex_ds_var = self.org_ds_var[self.var_name].reindex_like(
            self.ds_area[VAR_AREA], method="nearest", tolerance=0.01
        )
        self.monthly_ds = (
            reindex_ds_var
            * self.kg_2_mm
            * self.ds_area[VAR_AREA]
            * self.ds_mask[VAR_MASK]
            / self.glob_land_area
        )

    def cal_annual_ds(self):
        """
        Calculate the annual mean of precipitation (mm day⁻¹) from monthly ds (mm day⁻¹)
        """
        ds = self.monthly_ds
        self.annual_ds = ds.groupby(ds.time.dt.year).mean(skipna=True)

    def cal_monthly_per_area_unit(self):
        """
        Unit conversion from orginal data (kg m⁻² s⁻¹) to ((mm day⁻¹) (not weighted by areas)
        """
        self.monthly_per_area_unit = self.org_ds_var[self.var_name] * self.kg_2_mm

    def cal_annual_per_area_unit(self):
        """
        Calculate the annual per area unit of precipitation (mm day⁻¹) from monthly per area unit ds
        Area weighting is not required for this calculation, as shown in the map
        """
        ds = self.monthly_per_area_unit
        self.annual_per_area_unit = ds.groupby(ds.time.dt.year).mean(
            "time", skipna=True
        )

    def cal_reg_ann_rate(self):
        """
        Calculate the regional annual rate of precipitation (mm day⁻¹) from clipped (regional) monthly data
        """
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


class TAS(CMIP6Var):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_monthly_ds(self):
        """
        Calculates the area-weighted monthly mean temperature (°C) from orginal data (Kelvin)
        """
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
        """
        Unit conversion from orginal data (Kelvin) to degrees Celsius (°C) (not weighted by areas)
        """
        self.monthly_per_area_unit = (
            copy.deepcopy(self.org_ds_var[self.var_name]) - K_2_C
        )

    def cal_annual_ds(self):
        """
        Calculate the annual mean of temperature from monthly ds (°C)
        """
        ds = self.monthly_ds[:]
        self.annual_ds = ds.groupby(ds.time.dt.year).mean("time", skipna=True)

    def cal_annual_per_area_unit(self):
        """
        Calculate the annual per area unit of temperature (°C) from monthly per area unit ds
        Area weighting is not required for this calculation, as shown in the map
        """
        ds = self.monthly_per_area_unit
        self.annual_per_area_unit = ds.groupby(ds.time.dt.year).mean(
            "time", skipna=True
        )

    def cal_reg_ann_rate(self):
        """
        Calculate the regional annual rate (°C) from clipped (regional) monthly data
        """
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


class RSDS(TAS):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)

    def cal_monthly_ds(self):
        """
        Calculates the area-weighted monthly mean radiation (W m⁻²) from original data (W m⁻²)
        """
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
        """
        Calculate the monthly mean radiation (not weighted by areas) (W m⁻²)
        """
        self.monthly_per_area_unit = copy.deepcopy(self.org_ds_var[self.var_name])


class LAI(RSDS):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)


class CO2s(RSDS):
    def __init__(self, model_name, org_ds_var, var_name):
        super().__init__(model_name, org_ds_var, var_name)


# %%


# unit function for clipping to region mask
