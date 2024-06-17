#%%
from utils import *
from const import *
from mypath import *

import xarray as xr
import calendar
import rioxarray
import numpy as np
import matplotlib
import pandas as pd
import copy
import glob
import os
import pymannkendall as mk


class Model:
    def __init__(self, model_name):

        self.mod_name = model_name
        self.vars = {}

        self.axl = {}
        self.obj_type = None

        self.extract_var_by_model_name(model_name)

    
    def extract_var_by_model_name(self, model_name):
        var_list = os.listdir(VAR_DIR)

        for var in var_list[1:2]:
            all_files = glob.glob(os.path.join(VAR_DIR, var, "*.nc"))
            print(var)
            l_var = []
            for f in all_files:

                if model_name in f:
                    l_var.append(xr.open_dataset(f))
            self.vars[var] = xr.concat(l_var, dim=DIM_TIME)
    

var_name = "emiisop"    

def get_model_name(data_dir=AXL_DIR):
    axl_list = os.listdir(data_dir)
    all_files = glob.glob(os.path.join(data_dir, axl_list[0], "*.nc"))
    model_names = [f.split("\\")[-1].split("_")[2] for f in all_files]
    return model_names


def cal_nodays_m (ds):
    n_month = 12
    noday_360 = [30] * n_month
    noday_noleap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    calendar = ds.time.dt.calendar

    l = int(len(ds.time) / n_month)
    if calendar == "360_day":
        nodays_m = np.array(noday_360 * l)
    else: nodays_m = np.array(noday_noleap * l)

    return nodays_m


def cal_monthly_per_area_unit(ds):   #unit: g/m2/month - for plot global map

    monthly_per_area_unit = (
            KG_2_G
            * ISOP_2_C
            * DAY_RATE
            * ds[var_name].transpose(..., "time")
            * cal_nodays_m (ds)
        )
    return monthly_per_area_unit


def cal_annual_per_area_unit(monthly_per_area_unit): #unit: g/m2/year - for plot global map
    ds = monthly_per_area_unit
    annual_per_area_unit = ds.groupby(ds.time.dt.year).sum()
    return annual_per_area_unit




def cal_monthly_ds(ds, model_name):                  #unit: Tg/month - for plot trend
    for f in SFLTF_LIST: ds_sftlf = xr.open_dataset(f) if model_name in f else 1
    for f in AREA_LIST: ds_area = xr.open_dataset(f) if model_name in f else 1
    
    reindex_ds_lf = (ds_sftlf[VAR_SFTLF].reindex_like(ds[var_name], method="nearest", tolerance=0.01)* 1e-2)
    reindex_ds_area = ds_area[VAR_AREA].reindex_like(ds[var_name], method="nearest", tolerance=0.01)
    monthly_ds = (
        ds[var_name].transpose(..., "time")
        * reindex_ds_lf
        * reindex_ds_area
        * DAY_RATE
        * KG_2_TG
        * ISOP_2_C
        * cal_nodays_m (ds)
    )
    return monthly_ds

def cal_seasonal_ds(monthly_ds):
    seasonal_ds = monthly_ds.resample(time='QS-DEC').mean()
    return seasonal_ds

def cal_global_seasonal_rate(seasonal_ds):               #unit: Tg/month - for plot trend
    global_seasonal_rate = seasonal_ds.groupby('time').sum(dim=[DIM_LAT, DIM_LON])
    return global_seasonal_rate

def cal_annual_ds(monthly_ds):                #unit: Tg/year - for plot trend
    ds = monthly_ds
    annual_ds = ds.groupby(ds.time.dt.year).sum()
    return annual_ds

def cal_global_annual_rate(annual_ds):               #unit: Tg/year - for plot trend
    global_annual_rate = annual_ds.sum(dim=[DIM_LAT, DIM_LON])
    return global_annual_rate


def plot_global_annual_trend(annual_ds, mode="annual"):
    global_rate = cal_global_annual_rate(annual_ds)
    global_rate_anml = global_rate - global_rate.mean()
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    axbox = ax.get_position()
    if mode == "annual":
        x, y = global_rate.year, global_rate
    elif mode == "anomaly":
        x, y = global_rate_anml.year, global_rate_anml
    ax.plot(x, y, label=var_name, linewidth=1.25, marker="o", ms=2.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("(TgC/year)")          
    # plt.ylim([275, 280])
    ax.set_title(f"Annual Global Trend of {var_name}")
    ax.legend(
        loc="center",
        ncol=5,
        bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.25],
    )

def plot_global_seasonal_trend(seasonal_ds, mode="abs"):
    global_rate = cal_global_seasonal_rate(seasonal_ds)
    global_rate_anml = global_rate.groupby('time.month') - global_rate.groupby('time.month').mean()
    
    colors = {3: "grey", 6: "lightgreen", 9: "green", 12: "purple"}
    seasons = {3: "spring", 6: "summer", 9: "fall", 12: "winter"}

    f, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    axbox = ax.get_position()
    if mode == "abs":
        rate = global_rate
    elif mode == "anomaly":
        rate = global_rate_anml
    for month, arr in rate.groupby('time.month'):
        arr.plot(ax=ax,
                color="grey",
                marker="o",
                markerfacecolor=colors[month],
                markeredgecolor=colors[month],
                label=seasons[month])

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set(title="Seasonal Change in Total Emission Over Time (TgC/month)")
    plt.show()



"""
Trend Mann Kendall test for annual total per pixel

"""
def k_cor(x,y, pthres = 0.05, direction = True):
    """
    Uses the pymannkendall module to calculate a Kendall correlation test
    :x vector: Input pixel vector to run tests on
    :y vector: The date input vector
    :pthres: Significance of the underlying test
 
    """

    # Check NA values
    co = np.count_nonzero(~np.isnan(x))
    if co < 4: # If fewer than 4 observations return -9999
        return -9999
    # Run the kendalltau test
    trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(x)

    # Criterium to return results in case of Significance
    return slope if p < pthres else 0  

# The function we are going to use for applying our kendal test per pixel
def kendall_correlation(x,y,dim='year'):
    # x = Pixel value, y = a vector containing the date, dim == dimension
    return xr.apply_ufunc(
        k_cor, x , y,
        input_core_dims=[[dim], [dim]],
        vectorize=True, # !Important!
        output_dtypes=[float]
        )
"""
###excutation
x = xr.DataArray(np.arange(len(annual_ds['year']))+1, dims='year',
                 coords={'year': annual_ds['year']})  
r = kendall_correlation(annual_ds, x,'year')
"""
def plot_s_kendall(s, cmap="RdBu_r"):
    fig = plt.figure(1, figsize=(30, 13))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(
        xlocs=range(-180, 180, 40),
        ylocs=range(-80, 81, 20),
        draw_labels=True,
        linewidth=1,
        edgecolor="dimgrey",
    )
    my_cmap = matplotlib.cm.get_cmap(cmap)

    data = s
    title = "Annual Trends of isoprene emission from 1850-2014 using the Mann-Kendall method"
    data.plot.pcolormesh(
        ax=ax,
        cmap=my_cmap,
        cbar_kwargs={"label": "TgC/year"},
    )
    plt.title(title, fontsize=18)

def get_var_ds(model_name):               #unit: kg/m2/month
    model_name = "CESM2-WACCM"
    var_files = glob.glob(os.path.join("../data/var/", "loadoa", "*.nc"))
    model_files = [f for f in var_files if model_name in f]
    l_model_ds = []
    for f in model_files:
        l_model_ds.append(xr.open_dataset(f).resample(time='M').sum(skipna=True))
        var_ds = xr.concat(l_model_ds, dim="time")
    return var_ds

var_area = xr.open_dataset("../data/axl/areacella/areacella_fx_CESM2-WACCM_historical_r1i1p1f1_gn.nc")

def cal_mon_ds(ds):                     #unit: Tg/month
    reindex_ds_var = ds["loadoa"].reindex_like(var_area["areacella"], method="nearest", tolerance=0.01)
    mon_ds = reindex_ds_var* var_area["areacella"]*1e-9
    return mon_ds

def cal_annual_ds(mon_ds):
    ds = mon_ds 
    annual_ds = ds.groupby(ds.time.dt.year).sum(skipna=True)
    return annual_ds

def cal_glob_rate(annual_ds):
    global_rate = annual_ds.sum(dim=[DIM_LAT, DIM_LON])
    global_rate_anml = global_rate - global_rate.mean(skipna=True)
    return global_rate, global_rate_anml



# %%
