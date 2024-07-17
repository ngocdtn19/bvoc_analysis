# %%
import pickle
import copy
import numpy as np

from mk import *

from senAls_utils import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error as mse,
    mean_absolute_error as mae,
    r2_score,
)
from scipy.stats import pearsonr
import pymannkendall as pymk


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


class RegSingleModel:
    """
    Analysis of emulated sensitivity experiments on isoprene emissions for CMIP6 models:
        CESM2-WACCM(G2012)
        NorESM2-LM(G2012)
        GFDL-ESM4(G2006)
        GISS-E2-1-G(G1995)
        UKESM1-0-LL(P2011)
    """

    def __init__(
        self,
        model_name="CESM2-WACCM(G2012)",
        start_year=1850,
        end_year=2014,
        cross_val=False,
    ) -> None:
        self.target_name = "emiisop"
        self.model_name = model_name

        self.start_year = start_year
        self.end_year = end_year
        self.cross_val = cross_val

        self.clim_predictors = ["tas", "rsds", "pr"]
        self.other_predictors = ["treeFrac"]

        self.org_ds = None
        self.contribution_mk = {}
        self.ds_mk_pixel = {}
        # self.ds_mk_rate = pd.DataFrame()

        self.get_predictors_simulation()
        self.read_data()
        self.train_RF()
        if not self.cross_val:
            self.extr_mask()
            # if len(glob.glob(os.path.join(f"{CMIP6_SENSALS_DIR}/mk", f"{self.model_name}*"))) == 0:
            self.run_simulation()
            self.sensitivity_cal()

            self.contribution_mk_cal()
            # max impact calculation
            self.ctb_main_map = self.max_impact_cal(mode="main")
            self.ctb_clim_map = self.max_impact_cal(mode="clim")
            # cal glob_change_ts_cal - bar plot
            self.main_rates_ts, self.main_df_rate = self.glob_change_ts_cal(mode="main")
            self.clim_rates_ts, self.clim_df_rate = self.glob_change_ts_cal(mode="clim")
            # cal glob_rate_ts - line plot
            self.sim_rate = self.glob_rate_ts_cal()
            # cal impact_by_area - bar plot
            self.main_impact_by_area = self.glob_ctb_area_cal(mode="main")
            self.clim_impact_by_area = self.glob_ctb_area_cal(mode="clim")

    def get_predictors_simulation(self):
        if self.model_name in [
            "CESM2-WACCM(G2012)",
            "NorESM2-LM(G2012)",
            "UKESM1-0-LL(P2011)",
        ]:
            self.predictors = self.clim_predictors + ["co2s"] + self.other_predictors
            self.list_sim_id = [
                "control",
                "co2",
                "co2_clim",
                "co2_clim_luc",
                "co2_luc_rsds_pr",
                "co2_luc_tas_pr",
                "co2_luc_tas_rsds",
            ]
            self.list_driver = ["co2fi", "lulcc", "clim", "all", "tas", "rsds", "pr"]
        else:
            self.predictors = self.clim_predictors + self.other_predictors
            self.list_sim_id = [
                "control",
                "clim",
                "clim_luc",
                "luc_rsds_pr",
                "luc_tas_pr",
                "luc_tas_rsds",
            ]
            self.list_driver = ["lulcc", "clim", "all", "tas", "rsds", "pr"]

    def read_data(self):
        iters = self.predictors + [self.target_name]
        dss = {}
        for p in iters:
            f = os.path.join(
                CMIP6_SENSALS_DIR, "input_data", f"{self.model_name}_{p}.nc"
            )
            ds = xr.open_dataset(f)[p].sel(year=slice(self.start_year, self.end_year))
            # ds = ds.sel(year=slice(1901, 2014)) if "VISIT" in self.model_name else ds
            dss[p] = (("lat", "lon", "year"), ds.transpose("lat", "lon", "year").data)
        lat = ds.lat.values
        lon = ds.lon.values
        year = ds.year.values
        self.org_ds = xr.Dataset(dss, coords={"lat": lat, "lon": lon, "year": year})

    def train_RF(self):
        model_folder = os.path.join(
            "models",
            "without_year",
            f"{self.start_year}_{self.end_year}",
            "_".join(self.predictors),
        )
        model_path = os.path.join(model_folder, f"{self.model_name}.sav")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        ts_data = self.org_ds.copy()
        ts_data = ts_data.where(ts_data[self.target_name] != 0, drop=True)
        ts_data = ts_data.to_dataframe().reset_index()
        self.ts_data_df = ts_data

        # extract X, y for training and cross-validation
        ts_data = ts_data[ts_data[self.target_name].notna()]
        print("number of samples:", len(ts_data))
        ts_data = ts_data.fillna(0)

        X = ts_data.drop([self.target_name, "year"], axis=1).values
        y = ts_data[self.target_name].values.reshape(-1, 1)

        if not os.path.exists(model_path):
            rf = RandomForestRegressor().fit(X, y)
            self.model = rf
            pickle.dump(self.model, open(model_path, "wb"))
        else:
            self.model = pickle.load(open(model_path, "rb"))

        if self.cross_val:

            cv_metric_file = f"./cv/{self.model_name}.pkl"
            if not os.path.exists(cv_metric_file):
                years = ts_data["year"].unique()
                n_fold = 3
                kf = KFold(n_splits=n_fold, shuffle=False)
                year_fold = kf.split(years)

                self.cv_metrics = {
                    "corr": [],
                    "r2": [],
                    "rmse": [],
                    "mae": [],
                }
                for i, (train_index, val_index) in enumerate(year_fold):
                    print(f"---- Fold: {i} ----")
                    train_year = years[train_index]
                    val_year = years[val_index]
                    print(f"Train years: {train_year}")
                    print(f"Val years: {val_year}")

                    df_train = ts_data[ts_data["year"].isin(train_year)]
                    df_val = ts_data[ts_data["year"].isin(val_year)]

                    X_train = df_train.drop([self.target_name, "year"], axis=1).values
                    y_train = df_train[self.target_name].values.reshape(-1, 1)

                    X_val = df_val.drop([self.target_name, "year"], axis=1).values
                    y_val = df_val[self.target_name].values.reshape(-1, 1)

                    rf = RandomForestRegressor().fit(X_train, y_train)
                    y_pred = rf.predict(X_val)

                    df = pd.DataFrame()
                    df["val"] = y_val.reshape(-1)
                    df["pred"] = y_pred.reshape(-1)

                    corr_fold = df["val"].corr(df["pred"])
                    rmse = mse(y_val, y_pred, squared=False)
                    mae_err = mae(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    print(f"Pearson R: {corr_fold}")
                    print(f"RMSE: {rmse}")
                    print(f"MAE: {mae_err}")
                    print(f"R2: {r2}")

                    self.cv_metrics["corr"].append(corr_fold)
                    self.cv_metrics["rmse"].append(rmse)
                    self.cv_metrics["mae"].append(mae_err)
                    self.cv_metrics["r2"].append(r2)

                with open(cv_metric_file, "wb") as f:
                    pickle.dump(self.cv_metrics, f)
            else:
                with open(cv_metric_file, "rb") as f:
                    self.cv_metrics = pickle.load(f)

    def extr_mask(self):
        mask = self.org_ds.where(self.org_ds[self.target_name] != 0)
        mask = mask[self.target_name].mean("year").values
        mask[mask > 0] = 1
        self.mask = mask

    def run_simulation(self):
        def get_fixed_predictors(df, sim_id, year):
            # get fixed cols
            if sim_id == "control":
                cols = self.predictors
            elif sim_id == "co2":
                cols = ["treeFrac"] + self.clim_predictors
            elif sim_id == "co2_clim" or sim_id == "clim":
                cols = ["treeFrac"]
            elif "clim_luc" in sim_id:
                cols = []
            elif "luc_rsds_pr" in sim_id:
                cols = ["tas"]
            elif "luc_tas_pr" in sim_id:
                cols = ["rsds"]
            elif "luc_tas_rsds" in sim_id:
                cols = ["pr"]

            keys = ["lat", "lon"]
            if len(cols) > 0:
                df_cols = df[df["year"] == year][cols + keys]
                df_no_cols = df.drop(cols, axis=1)
                df_fixed_cols = df_no_cols.merge(df_cols, how="left", on=keys).fillna(0)
            else:
                return df
            return df_fixed_cols

        org_ts_data = self.ts_data_df.copy()
        # filter no na values
        org_ts_data = org_ts_data[org_ts_data[self.target_name].notna()].fillna(0)
        org_X = org_ts_data.drop([self.target_name, "year"], axis=1).values

        self.org_y = org_ts_data[self.target_name].values

        self.regressed_data = self.model.predict(org_X)
        print("R:", pearsonr(self.org_y, self.regressed_data))

        df_sim = copy.deepcopy(org_ts_data)
        df_sim["reg"] = self.regressed_data
        df_sim["emiisop"] = self.org_y
        for sid in self.list_sim_id:
            fixed_var_df = org_ts_data.drop(self.target_name, axis=1)
            # get fixed predictors
            fixed_predictors = get_fixed_predictors(fixed_var_df, sid, self.start_year)
            fixed_predictors = fixed_predictors[list(self.ts_data_df.columns[:-1])]
            fixed_predictors = fixed_predictors.drop("year", axis=1)
            # make prediction
            df_sim[sid] = self.model.predict(fixed_predictors.values)

        keys = ["lat", "lon", "year"]
        self.df_sim = df_sim[keys + ["reg"] + ["emiisop"] + self.list_sim_id]

    def sensitivity_cal(self):
        keys = ["lat", "lon", "year"]
        sen_df = pd.DataFrame()
        sen_df[keys] = self.df_sim[keys]

        if "co2" in self.df_sim.columns:
            sen_df["co2fi"] = self.df_sim["co2"] - self.df_sim["control"]
            sen_df["lulcc"] = self.df_sim["co2_clim_luc"] - self.df_sim["co2_clim"]
            sen_df["clim"] = self.df_sim["co2_clim"] - self.df_sim["co2"]
            sen_df["all"] = self.df_sim["co2_clim_luc"] - self.df_sim["control"]
            sen_df["tas"] = self.df_sim["co2_clim_luc"] - self.df_sim["co2_luc_rsds_pr"]
            sen_df["rsds"] = self.df_sim["co2_clim_luc"] - self.df_sim["co2_luc_tas_pr"]
            sen_df["pr"] = self.df_sim["co2_clim_luc"] - self.df_sim["co2_luc_tas_rsds"]
        else:
            sen_df["lulcc"] = self.df_sim["clim_luc"] - self.df_sim["clim"]
            sen_df["clim"] = self.df_sim["clim"] - self.df_sim["control"]
            sen_df["all"] = self.df_sim["clim_luc"] - self.df_sim["control"]
            sen_df["tas"] = self.df_sim["clim_luc"] - self.df_sim["luc_rsds_pr"]
            sen_df["rsds"] = self.df_sim["clim_luc"] - self.df_sim["luc_tas_pr"]
            sen_df["pr"] = self.df_sim["clim_luc"] - self.df_sim["luc_tas_rsds"]

        self.sensitivity_ds = sen_df.groupby(keys).mean().to_xarray()

    def contribution_mk_cal(self):
        # mk_rates = []
        for v in self.list_driver:
            # print(f"start mk {v}")
            mk_dir = f"{CMIP6_SENSALS_DIR}/contribution_mk"
            if not os.path.exists(mk_dir):
                os.mkdir(mk_dir)
            file_path = os.path.join(mk_dir, f"{self.model_name}_{v}.nc")
            if os.path.exists(file_path):
                ds = xr.open_dataset(file_path)
                ds = ds.rename(name_dict={list(ds.keys())[0]: v})
                self.contribution_mk[v] = ds[v]
            else:
                mk_ds = xr.Dataset({})
                mk_ds[v] = cal_mk(self.sensitivity_ds, v)
                mk_ds.to_netcdf(file_path)
                self.contribution_mk[v] = mk_ds[v]
            # print(f"end mk {v}")
            # _, glob_change = cal_actual_rate(self.contribution_mk[v], self.model_name)
            # mk_rates.append(glob_rate)
            self.ds_mk_pixel[v] = self.contribution_mk[v]
        # self.ds_mk_rate["driver"] = self.list_driver
        # self.ds_mk_rate["rate"] = mk_rates

    def max_impact_cal(self, mode="main"):
        if mode == "main":
            if "co2fi" in self.list_driver:
                list_driver = ["co2fi", "lulcc", "clim"]
            else:
                list_driver = ["lulcc", "clim"]
            self.list_main_driver = list_driver
        elif mode == "clim":
            list_driver = self.clim_predictors

        # cal mask without nan values
        valid_mask = np.zeros(self.ds_mk_pixel[list_driver[0]].shape)
        for v in list_driver:
            arr = np.nan_to_num(self.ds_mk_pixel[v].values)
            arr[arr != 0] = 1
            valid_mask += arr
        valid_mask[valid_mask < 1] = np.nan
        valid_mask[valid_mask > 0] = 1
        self.valid_mask = valid_mask

        # cal max impact of driver
        stacked = np.stack(tuple(self.ds_mk_pixel[v] for v in list_driver), axis=-1)
        abs_stacked = np.absolute(np.nan_to_num(stacked))
        max_stacked = np.nanargmax(abs_stacked, axis=-1) * self.valid_mask

        coords = {
            "lon": self.ds_mk_pixel[list_driver[0]].lon.values,
            "lat": self.ds_mk_pixel[list_driver[0]].lat.values,
        }

        return xr.Dataset({"driver": (("lat", "lon"), max_stacked)}, coords=coords)

    def glob_rate_ts_cal(self):
        keys = ["lat", "lon", "year"]
        sim_ds = self.df_sim.groupby(keys).mean().to_xarray()
        sim_rate = xr.Dataset({})
        list_id = ["reg", "emiisop"]
        for sid in list_id:
            sim_rate[sid], _ = cal_actual_rate(sim_ds[sid], self.model_name, mode="ts")
        return sim_rate

    def glob_change_ts_cal(self, mode="main"):
        rates_ts = []
        slope_ts = []
        sig_ts = []
        df_trend_rate = pd.DataFrame()
        if mode == "main":
            list_pred = self.list_main_driver + ["all"]
        elif mode == "clim":
            list_pred = self.clim_predictors + ["clim"]
        for v in list_pred:
            glob_rate_ts, _ = cal_actual_rate(
                self.sensitivity_ds[v], self.model_name, mode="ts"
            )
            rates_ts.append(glob_rate_ts)

            trend_test = pymk.original_test(glob_rate_ts.values, alpha=0.05)
            slope_ts.append(trend_test.slope)
            sig_ts.append(trend_test.h)
        df_trend_rate["driver"] = list_pred
        df_trend_rate["slope"] = slope_ts
        df_trend_rate["sig"] = sig_ts
        return (
            pd.DataFrame(
                {var: rate for var, rate in zip(list_pred, rates_ts)},
                index=[i for i in range(1850, 2015)],
            ),
            df_trend_rate,
        )

    def glob_ctb_area_cal(self, mode="main"):
        if mode == "main":
            data = self.ctb_main_map
            list_pred = self.list_main_driver
        elif mode == "clim":
            data = self.ctb_clim_map
            list_pred = self.clim_predictors

        # --- aggregate the impact by area ---
        processed_area = (
            prep_area(data["driver"], self.model_name).to_dataset() * self.valid_mask
        )
        processed_area = processed_area.assign(driver=data["driver"])
        area_dict = {}
        total_a = 0
        for v, f in enumerate(list_pred):
            area = processed_area.where(processed_area["driver"] == v, drop=True)
            area = area["areacella"].sum(["lat", "lon"]).item()
            area_dict[f] = area
            total_a += area

        impact_by_area = [area_dict[f] * 100 / total_a for f in list_pred]
        impact_area_df = pd.DataFrame()
        impact_area_df["percentage"] = impact_by_area
        impact_area_df["driver"] = list_pred
        print(impact_area_df)
        return impact_area_df
        # self.impact_area_df.plot.bar(x="driver", y="percentage")


# %%
