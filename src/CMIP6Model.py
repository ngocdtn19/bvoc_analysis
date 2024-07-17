# %%

from CMIP6Var import *
from const import *
from mypath import *
from scipy import stats
from visit_preprocess import *
from cartopy.util import add_cyclic_point
import mk
import xskillscore as xs

sns.set_style("ticks")


class ModelVar:
    def __init__(self):
        self.var_obj_dict = {
            "emiisop": EMIISOP,
            "gpp": PP,
            "pr": PR,
            "tas": TAS,
            "rsds": RSDS,
            "lai": LAI,
            "co2s": CO2s,
        }
        self.var_obj_visit_dict = {
            "emiisop": VisitEMIISOP,
            "tas": VisitTAS,
            "rsds": VisitRSDS,
            "pr": VisitPR,
            "gpp": VisitGPP,
            "lai": VisitLAI,
            "co2s": VisitCO2s,
        }


class Model(ModelVar):
    """
    Extract data for each model, consisting of multiple variables (1 model - n variables)
        List models:
            VISIT-S3
            CMIP6 models:
                CESM2-WACCM(G2012)
                NorESM2-LM(G2012)
                GFDL-ESM4(G2006)
                GISS-E2-1-G(G1995)
                UKESM1-0-LL(P2011)
        List variables:
            emiisop: isoprene emission
            tas: temperature
            rsds: shortwave radiation
            pr: precipitation
    """

    def __init__(self, model_name="UKESM1-0-LL") -> None:
        super().__init__()

        self.model_name = model_name
        self.var_names = list()
        self.var_objs = {}

        self.extract_vars()

    def get_var_ds(self, var_name):
        var_files = sorted(glob.glob(os.path.join(VAR_DIR, var_name, "*.nc")))
        model_files = [f for f in var_files if self.model_name in f]
        l_model_ds = []
        for f in model_files:
            var_ds = (
                visit_t2cft(f, var_name, m_name=self.model_name)
                if "VISIT" in self.model_name
                else (xr.load_dataset(f))
            )
            l_model_ds.append(var_ds)

        return xr.concat(l_model_ds, dim=DIM_TIME)

    def extract_vars(self):
        all_var_files = list()

        for dp, dn, fn in os.walk(VAR_DIR):
            all_var_files += [os.path.join(dp, file) for file in fn]

        # remove files without containing model_name
        all_var_files = [f for f in all_var_files if self.model_name in f]
        self.var_names = sorted(
            list(set([f.split("/")[-1].split("_")[0] for f in all_var_files]))
        )

        for v_name in self.var_names:
            print(v_name)
            if "VISIT" in self.model_name:
                self.var_objs[v_name] = self.var_obj_visit_dict[v_name](
                    self.model_name, self.get_var_ds(v_name), v_name
                )
            else:
                self.var_objs[v_name] = self.var_obj_dict[v_name](
                    self.model_name, self.get_var_ds(v_name), v_name
                )


class Var(ModelVar):
    """
    Intermodel comparison of variables from historical simulations among selected models (1 var - n models)
        List models:
            VISIT-S3
            CMIP6 models:
                CESM2-WACCM(G2012)
                NorESM2-LM(G2012)
                GFDL-ESM4(G2006)
                GISS-E2-1-G(G1995)
                UKESM1-0-LL(P2011)
        List variables:
            emiisop: isoprene emission
            tas: temperature
            rsds: shortwave radiation
            pr: precipitation
    """

    processed_dir = os.path.join(DATA_DIR, "processed_org_data")

    def __init__(self, var_name):
        super().__init__()
        self.var_name = var_name
        self.obj_type = None
        self.multi_models = {}

        self.get_obj_type()
        self.get_multi_models()

    def get_obj_type(self):
        self.obj_type = (
            self.var_obj_dict[self.var_name]
            if self.var_name in self.var_obj_dict.keys()
            else None
        )
        self.obj_type_visit = (
            self.var_obj_visit_dict[self.var_name]
            if self.var_name in self.var_obj_visit_dict.keys()
            else None
        )

    def get_model_name(self, path):
        var_name = {
            "emiisop": "AERmon",
            "gpp": "Lmon",
            "pr": "Amon",
            "tas": "Amon",
            "rsds": "Amon",
            "lai": "Lmon",
            "co2s": "Emon",
        }
        return (
            path.split("\\")[-1]
            .split(var_name[self.var_name])[-1]
            .split("historical")[0]
            .replace("_", "")
        )

    def get_multi_models(self):
        """
        Load and concatenate data for each model, then create a dictionary of timeseries xr.Dataset for multiple models.
        """
        all_files = sorted(glob.glob(os.path.join(VAR_DIR, self.var_name, "*.nc")))
        model_names = sorted(list(set([self.get_model_name(f) for f in all_files])))

        multi_models = {}
        for m_name in model_names:
            print(m_name)
            if "VISIT" not in m_name:
                l_model = []
                for f in all_files:
                    if m_name in f:
                        l_model.append(xr.load_dataset(f))

                multi_models[m_name] = self.obj_type(
                    m_name, xr.concat(l_model, dim=DIM_TIME), self.var_name
                )
            else:
                for f in all_files:
                    if m_name in f:
                        multi_models[m_name] = self.obj_type_visit(
                            m_name, visit_t2cft(f, self.var_name, m_name), self.var_name
                        )

        self.multi_models = multi_models

    def plot_regional_map(self):
        """Regional map visual examination"""
        rois = LIST_REGION
        l_m_name = list(self.multi_models.keys())
        ds = self.multi_models[l_m_name[0]]
        for i, r in enumerate(rois):
            fig = plt.figure(1 + i, figsize=(30, 13))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.gridlines(
                xlocs=range(-180, 180, 40),
                ylocs=range(-80, 81, 20),
                draw_labels=True,
                linewidth=1,
                edgecolor="dimgrey",
            )
            data = ds.regional_ds[r].sel(year=2014)
            data.plot.pcolormesh(
                ax=ax,
                cmap="tab20c_r",
                levels=8,
                cbar_kwargs={"label": VIZ_OPT[self.var_name]["map_unit"]},
            )
            plt.title(f"{l_m_name[0]} - {r} ", fontsize=18)

    def save_2_nc(self):
        model_names = list(self.multi_models.keys())
        for name in model_names:
            annual_data = self.multi_models[name].annual_per_area_unit
            if self.var_name in ["emiotherbvocs", "emibvoc", "gpp", "npp"]:
                annual_ds = xr.Dataset(
                    data_vars=dict(
                        var_name=(["lat", "lon", "year"], annual_data.values)
                    ),
                    coords=dict(
                        lat=annual_data.lat, lon=annual_data.lon, year=annual_data.year
                    ),
                )
            else:
                annual_ds = xr.Dataset(
                    data_vars=dict(
                        var_name=(["year", "lat", "lon"], annual_data.values)
                    ),
                    coords=dict(
                        lat=annual_data.lat, lon=annual_data.lon, year=annual_data.year
                    ),
                )
            annual_ds = annual_ds.rename({"var_name": self.var_name})
            annual_ds.to_netcdf(
                os.path.join(
                    self.processed_dir,
                    "annual_per_area_unit",
                    f"{name}_{self.var_name}.nc",
                )
            )


class Land:
    """
    Analysis of land data for each CMIP6 model, consisting of multiple PFT types (1 model - n PFT types)
        List CMIP6 models:
            CESM2-WACCM
            GFDL-ESM4
            GISS-E2-1-G
            UKESM1-0-LL
        List PFT types:
            treeFrac: fraction of tree (%)
            grassFrac: fraction of grass (%)
            shrubFrac/pastureFrac:
                fraction of shrub (%) for CESM2-WACCM and GISS-E2-1-G,
                fraction of pasture (%) for GFDL-ESM4 and UKESM1-0-LL
            cropFrac: fraction of crop (%)
    """

    processed_dir = os.path.join(DATA_DIR, "processed_org_data")

    def __init__(self, model_name="UKESM1-0-LL", mon_type="Lmon") -> None:
        self.model_name = model_name
        self.mon_type = mon_type

        self.org_cell_objs = {}
        self.area_weighted_cell_obj = {}
        self.roi_ltype = {}
        self.roi_area = {}

        self.ds_area = None
        self.ds_sftlf = None

        self.get_ds_area()
        self.extract_merge_land_type()
        self.cal_area_weighted_cell()
        self.clip_2_roi()

    def get_ds_area(self):
        ds_area = [xr.load_dataset(f) for f in AREA_LIST if self.model_name in f][0]
        ds_sftlf = [xr.load_dataset(f) for f in SFLTF_LIST if self.model_name in f][0]

        self.ds_sftlf = ds_sftlf[VAR_SFTLF].reindex_like(
            ds_area, method="nearest", tolerance=0.01
        )
        self.ds_area = self.ds_sftlf * ds_area * 1e-2

    def extract_merge_land_type(self):
        all_nc = sorted(
            glob.glob(os.path.join(LAND_DIR, self.model_name, self.mon_type, "*.nc"))
        )
        self.land_type = list(
            set([f.split("/")[-1].split("_")[0] for f in all_nc if self.mon_type in f])
        )
        (
            self.land_type.remove("fracLut")
            if "fracLut" in self.land_type
            else self.land_type
        )
        for ltype in self.land_type:
            ltype_ds = []
            list_nc = [f for f in all_nc if self.mon_type in f and ltype in f]

            for nc in list_nc:
                ltype_ds.append(xr.load_dataset(nc))

            self.org_cell_objs[ltype] = xr.concat(ltype_ds, dim=DIM_TIME)

    def cal_area_weighted_cell(self):
        ds_area = copy.deepcopy(self.ds_area[VAR_AREA])
        for ltype in self.land_type:
            reindex_ltype = self.org_cell_objs[ltype][ltype].reindex_like(
                self.ds_area, method="nearest", tolerance=0.01
            )

            ds = reindex_ltype * ds_area * 1e-2
            ds = ds.rio.write_crs("epsg:4326", inplace=True)
            ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180
            ds = ds.sortby(ds.lon)
            ds = ds.rio.set_spatial_dims("lon", "lat", inplace=True)
            self.area_weighted_cell_obj[ltype] = ds

    def clip_2_roi(self, boundary_dict={}):
        land_types = sorted(list(self.org_cell_objs.keys()))

        for i, roi in enumerate(
            LIST_REGION_LAND
        ):  # update with region mask ar6.land/serex, change interested LIST_REGION to all regions if use plot_global_annual_trend
            self.roi_ltype[roi] = {}

            ds_area = copy.deepcopy(self.ds_area[VAR_AREA])
            ds_area = ds_area.rio.write_crs("epsg:4326", inplace=True)
            ds_area.coords["lon"] = (ds_area.coords["lon"] + 180) % 360 - 180
            ds_area = ds_area.sortby(ds_area.lon)
            ds_area = ds_area.rio.set_spatial_dims("lon", "lat", inplace=True)

            self.roi_area[roi] = clip_region_mask(ds_area, roi).sum(["lat", "lon"])

            for ltype in land_types:
                ds = copy.deepcopy(self.area_weighted_cell_obj[ltype])
                ds = ds.rio.set_spatial_dims("lon", "lat", inplace=True)
                self.roi_ltype[roi][ltype] = (
                    (clip_region_mask(ds, roi).sum(["lat", "lon"]))
                    / self.roi_area[roi]
                    * 1e2
                )

    def plot_mk(self, sy, ey, ltype, cmap="RdBu_r"):
        ds = self.area_weighted_cell_obj[ltype]
        annual_ds = (ds.sel(time=ds.time.dt.month.isin([12]))) * 1e-6
        annual_ds = annual_ds.sel(
            time=annual_ds.time.dt.year.isin([i for i in range(sy, ey + 1)])
        )

        x = xr.DataArray(
            np.arange(len(annual_ds["time"])) + 1,
            dims="time",
            coords={"time": annual_ds["time"]},
        )
        s = mk.kendall_correlation(annual_ds, x, "time")

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
        my_cmap = mpl.cm.get_cmap(cmap)

        data = s
        title = f"Annual Trends of {ltype} from {sy}-{ey} using the Mann-Kendall method"
        data.plot.pcolormesh(
            ax=ax,
            cmap=my_cmap,
            cbar_kwargs={"label": "$km^{2}/year$"},
        )
        plt.title(title, fontsize=18)
        # plt.savefig(
        #     os.path.join("../fig/", self.model_name, f"mk-{ltype}-{sy}-{ey}.png")
        # )

    def save_2_nc(self):
        ds_sftlf = copy.deepcopy(self.ds_sftlf)
        for ltype in self.land_type:
            reindex_ltype = self.org_cell_objs[ltype][ltype].reindex_like(
                self.ds_sftlf, method="nearest", tolerance=0.01
            )
            data = reindex_ltype * ds_sftlf * 1e-2
            data = self.org_cell_objs[ltype][ltype]

            data = data.sel(time=data.time.dt.month.isin([12]))
            lat = data.lat.values
            lon = data.lon.values
            year = data.time.dt.year.values
            ds = xr.Dataset(
                data_vars=dict(var_name=(["year", "lat", "lon"], data.values)),
                coords={"lat": lat, "lon": lon, "year": year},
            )
            ds = ds.rename({"var_name": ltype})
            ds.to_netcdf(
                os.path.join(
                    self.processed_dir,
                    "land",
                    f"{self.model_name}_{ltype}.nc",
                )
            )


# a = Var("tas")
# %%
