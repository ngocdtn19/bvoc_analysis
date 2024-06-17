# %%
import xarray as xr
import xskillscore as xskill
import copy
import matplotlib


from utils import *
from const import *
from mypath import *

from visit_preprocess import visit_t2cft


DICT_MODEL_NAMES = get_list_model_by_name(sorted(ISOP_LIST))

TOPDOWN_DS = xr.open_dataset(CONCAT_TOPDOWN_PATH)

AVAIL_TIME = slice("2005-01", "2017-12")


def cal_nodays_m(ds):
    n_month = 12
    noday_360 = [30] * n_month
    noday_noleap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    calendar = ds.time.dt.calendar

    l = int(len(ds.time) / n_month)
    if calendar == "360_day":
        nodays_m = np.array(noday_360 * l)
    else:
        nodays_m = np.array(noday_noleap * l)

    return nodays_m


def preprocess_truth_model_ds(model_name, var_name="emiisop"):
    topdown_ds = copy.deepcopy(TOPDOWN_DS)

    # merge and convert to gC/m2/month
    list_nc_files = DICT_MODEL_NAMES[model_name]
    if "VISIT" not in model_name:
        model_ds = merge_by_model(list_nc_files)
        nodays_m = cal_nodays_m(model_ds)
        model_ds[var_name] = KG_2_G * ISOP_2_C * DAY_RATE * model_ds[var_name]
        model_ds[var_name] = model_ds[var_name].transpose(..., "time") * nodays_m
    else:
        model_ds = visit_t2cft(list_nc_files[0], var_name, model_name)
        model_ds[var_name] = model_ds[var_name] * MG_2_G

    sliced_model_ds = model_ds.sel(time=AVAIL_TIME)

    # fix coords
    sliced_model_ds = copy.deepcopy(sliced_model_ds[var_name])
    sliced_model_ds = sliced_model_ds.rio.write_crs("epsg:4326", inplace=True)
    sliced_model_ds.coords["lon"] = (sliced_model_ds.coords["lon"] + 180) % 360 - 180
    sliced_model_ds = sliced_model_ds.sortby(sliced_model_ds.lon)
    sliced_model_ds = sliced_model_ds.rio.set_spatial_dims("lon", "lat", inplace=True)

    interpolated_ds = interpolate(sliced_model_ds, mode="visit")

    topdown_ds["time"] = interpolated_ds.time

    # convert to gC/m2/month unit for topdown OMIv1
    # topdown_ds = topdown_ds.where(topdown_ds[var_name] != -99.0)
    # topdown_ds[var_name] = (
    #     topdown_ds[var_name] / topdown_ds.Grid_area * KG_2_G * ISOP_2_C
    # )
    td_nodays_m = cal_nodays_m(topdown_ds)
    topdown_ds[var_name] = topdown_ds[var_name] = (
        KG_2_G * ISOP_2_C * DAY_RATE * topdown_ds[var_name]
    )
    topdown_ds[var_name] = topdown_ds[var_name].transpose(..., "time") * td_nodays_m

    sliced_topdown_ds = copy.deepcopy(topdown_ds[var_name])
    sliced_topdown_ds = sliced_topdown_ds.rio.write_crs("epsg:4326", inplace=True)
    sliced_topdown_ds.coords["lon"] = (
        sliced_topdown_ds.coords["lon"] + 180
    ) % 360 - 180
    sliced_topdown_ds = sliced_topdown_ds.sortby(sliced_model_ds.lon)
    sliced_topdown_ds = sliced_topdown_ds.rio.set_spatial_dims(
        "lon", "lat", inplace=True
    )

    inter_td_ds = interpolate(sliced_topdown_ds, mode="visit")

    inter_td_ds = inter_td_ds.groupby(inter_td_ds.time.dt.year).sum("time", skipna=True)
    interpolated_ds = interpolated_ds.groupby(interpolated_ds.time.dt.year).sum(
        "time", skipna=True
    )

    return inter_td_ds, interpolated_ds


def val_single_score(truth_ds, model_ds, score):
    test_score = score(truth_ds, model_ds, dim="year", skipna=True)
    test_score_mean = np.format_float_positional(
        test_score.mean(dim=["lat", "lon"], skipna=True).values, precision=3
    )
    return test_score, test_score_mean


def plot_map(test_score, i):
    fig = plt.figure(i, figsize=(8, 6))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    # ax.gridlines(
    #     xlocs=range(-180, 180, 40),
    #     ylocs=range(-80, 81, 20),
    #     draw_labels=True,
    #     linewidth=1,
    #     edgecolor="dimgrey",
    # )
    my_cmap = matplotlib.cm.get_cmap("Spectral_r")
    my_cmap.set_under("#e0f3f8")

    test_score.plot.pcolormesh(
        ax=ax,
        cmap=my_cmap,
        levels=21,
        vmin=0.25,
        vmax=5,
        extend="both",
        cbar_kwargs={
            "label": "[$gC m^{-2} year^{-1}$]",
            "orientation": "horizontal",
            "pad": 0.05,
        },
    )


def main():
    dict_score = {
        # "r": xskill.pearson_r,
        # "pearson_r_p_value": xskill.pearson_r_p_value,
        # "mae": xskill.mae,
        "rmse": xskill.rmse,
        # "me": xskill.me,
    }

    for i, model_name in enumerate(DICT_MODEL_NAMES.keys()):
        print(model_name)
        truth_ds, model_ds = preprocess_truth_model_ds(model_name)

        for j, score_name in enumerate(dict_score.keys()):
            test_score, test_score_mean = val_single_score(
                truth_ds, model_ds, dict_score[score_name]
            )
            plot_map(test_score, i * len(DICT_MODEL_NAMES.keys()) + j + 1)
            plt.title(f"{model_name} - {score_name}", fontsize=14)
            plt.annotate(
                f"Mean {score_name} = {test_score_mean}",
                xy=(120, 95),
                fontsize=12,
            )
            # plt.savefig(
            #     os.path.join("../fig/validate", f"{model_name}-{score_name}.png")
            # )


# %%
