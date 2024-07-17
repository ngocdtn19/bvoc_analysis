from CMIP6Model import *


def prep_area(ds, model_name):
    base_dir = f"{DATA_DIR}/original/axl/areacella"
    fname = (
        "areacella_fx_GFDL-ESM4_historical_r1i1p1f1_gr1.nc"
        if "VISIT" not in model_name
        else "areacella_fx_VISIT_historical_r1i1p1f1_gn.nc"
    )
    ds_area = xr.open_dataset(os.path.join(base_dir, fname))

    reindex_ds_area = ds_area["areacella"].reindex_like(
        ds, method="nearest", tolerance=0.01
    )
    return reindex_ds_area


def cal_actual_rate(ds, model_name, mode="diff"):
    reindex_ds_area = prep_area(ds, model_name)

    global_map = ds * reindex_ds_area * 1e-12

    global_rate = global_map.sum(dim=["lat", "lon"])
    global_rate = global_rate.item() if mode == "diff" else global_rate

    return global_rate, global_map


def prep_to_clip_reg(ds):
    ds = ds.rio.write_crs("epsg:4326", inplace=True)

    ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180

    ds = ds.sortby(ds.lon)
    ds = ds.rio.set_spatial_dims("lon", "lat", inplace=True)
    return ds


def wrap_long(data):
    # print("Original shape -", data.shape)
    lon = data.coords["lon"]
    lon_idx = data.dims.index("lon")
    wrap_data, wrap_lon = add_cyclic_point(data.values, coord=lon, axis=lon_idx)
    # print("New shape -", wrap_data.shape)
    return wrap_data, wrap_lon
