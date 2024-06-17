# %%
import pickle
import copy
import numpy as np

from mk import *
from senAls_mulLinear import *
from senAls_ML import *
from sklearn.ensemble import RandomForestRegressor


class RegSingleModel:
    def __init__(self, model_name="VISIT", start_year=2005, end_year=2014) -> None:
        self.target_name = "emiisop"
        self.model_name = model_name

        self.start_year = start_year
        self.end_year = end_year

        self.clim_predictors = ["tas", "rsds", "pr"]
        self.other_predictors = ["lai"]

        self.org_ds = None

        self.get_predictors()
        self.read_data()
        self.train_RF()
        self.extr_mask()
        self.sensitivity_cal()
        self.contribution_cal()

    def get_predictors(self):
        if self.model_name in ["CESM2-WACCM", "NorESM2-LM", "UKESM1-0-LL"]:
            self.predictors = self.clim_predictors + ["co2s"] + self.other_predictors
        else:
            self.predictors = self.clim_predictors + self.other_predictors

    def read_data(self):
        iters = self.predictors + [self.target_name]
        dss = {}
        for p in iters:
            f = os.path.join(RES_DIR, f"{self.model_name}_{p}.nc")
            ds = xr.open_dataset(f)[p].sel(year=slice(self.start_year, self.end_year))
            # ds = ds.sel(year=slice(1901, 2014)) if "VISIT" in self.model_name else ds
            dss[p] = (("lat", "lon", "year"), ds.transpose("lat", "lon", "year").data)
        lat = ds.lat.values
        lon = ds.lon.values
        year = ds.year.values
        self.org_ds = xr.Dataset(dss, coords={"lat": lat, "lon": lon, "year": year})

    def train_RF(self):
        model_folder = os.path.join(
            "models", f"{self.start_year}_{self.end_year}", "_".join(self.predictors)
        )
        model_path = os.path.join(model_folder, f"{self.model_name}.sav")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        ts_data = self.org_ds.copy()
        ts_data = ts_data.where(ts_data[self.target_name] != 0, drop=True)
        ts_data = ts_data.to_dataframe().reset_index()
        self.ts_data_df = ts_data

        if not os.path.exists(model_path):
            ts_data = ts_data[ts_data[self.target_name].notna()]
            print("number of samples:", len(ts_data))
            ts_data = ts_data.fillna(0)

            X = ts_data.drop(self.target_name, axis=1).values
            y = ts_data[self.target_name].values.reshape(-1, 1)
            rf = RandomForestRegressor().fit(X, y)
            self.model = rf
            pickle.dump(self.model, open(model_path, "wb"))
        else:
            self.model = pickle.load(open(model_path, "rb"))

    def extr_mask(self):
        mask = self.org_ds.where(self.org_ds[self.target_name] != 0, drop=True)
        mask = mask[self.target_name].isel(year=0).values
        mask[mask > 0] = 1
        self.mask = mask

    def sensitivity_cal(self):
        def get_fixed_predictors(df, p_col, year):
            keys = ["lat", "lon"]
            df_y_fix = df[df["year"] == year][[p_col] + keys]

            df_no_pcol = df.drop(p_col, axis=1)
            df_fixed_pcol = df_no_pcol.merge(df_y_fix, how="left", on=keys).fillna(0)
            return df_fixed_pcol

        org_ts_data = self.ts_data_df.copy()
        # filter no na values
        org_ts_data = org_ts_data[org_ts_data[self.target_name].notna()].fillna(0)
        org_X = org_ts_data.drop(self.target_name, axis=1).values

        self.org_y = org_ts_data[self.target_name].values

        self.regressed_data = self.model.predict(org_X)
        print("R:", pearsonr(self.org_y, self.regressed_data))

        df_est = copy.deepcopy(org_ts_data)
        df_est["reg"] = self.regressed_data
        for p in self.predictors:
            fixed_var_df = org_ts_data.drop(self.target_name, axis=1)
            # get fixed predictors
            fixed_predictors = get_fixed_predictors(fixed_var_df, p, self.start_year)
            fixed_predictors = fixed_predictors[list(self.ts_data_df.columns[:-1])]

            est = self.model.predict(fixed_predictors.values)
            assert len(fixed_var_df) == len(est)
            df_est[f"{p}_fixed"] = est

        keys = ["lat", "lon", "year"]
        df_est = df_est[keys + ["reg"] + [f"{p}_fixed" for p in self.predictors]]

        self.sensitivity_ds = (
            self.ts_data_df.merge(df_est, how="left", on=keys)
            .groupby(keys)
            .mean()
            .to_xarray()
        )

    def contribution_cal(self):
        def recal_ctb_map(sensitivity_ds, fields, xi, mask, clim_only=False):
            list_ctb_map = []  # contribution map
            ctb_dict = {}
            n = len(fields)
            # cal mk time series mk slope
            ds_mk_slope = sensitivity_ds.copy()
            for f in fields:
                ds_mk_slope[f] = kendall_correlation(ds_mk_slope[f], xi, "year") * mask
            # cal contribution
            for f in fields:
                Ei = ds_mk_slope[f]  # cal Ei
                # cal Ek
                k_fields = [k for k in fields if k != f]
                Ek = ds_mk_slope[k_fields].to_array().sum("variable")
                # cal contribution
                ctb = (Ek - (n - 2) * Ei) / (n - 1)
                list_ctb_map.append(np.absolute(ctb.values))
                ctb_dict[f] = ctb
            # --- calculate max_ctb_map - map ---
            if clim_only:  # only for climate vars (temp, prep, ...)
                list_ctb_map = list_ctb_map[: len(self.clim_predictors)]
            # stack n x ctb map (X x Y) to (X x Y x n)
            stacked_ctb = np.stack((i for i in list_ctb_map), axis=-1)
            # find the max values of the n dim and return the index (axis = -1 means the last dim of (X x Y x n))
            max_ctb_map = np.argmax(np.nan_to_num(stacked_ctb), axis=-1)
            return max_ctb_map, ds_mk_slope, ctb_dict

        def recal_ctb_rate(sensitivity_ds, org_fields):
            glob_rate_est = pd.DataFrame()
            # org_fields = list(sensitivity_ds.data_vars)
            fields = [self.target_name, "reg"] + org_fields
            for f in fields:
                ds_cp = sensitivity_ds.copy()
                glob_rate_est[f], _ = cal_actual_rate(ds_cp[f], model_name, mode="ts")

            glob_rate_est["year"] = sensitivity_ds.year.values
            return glob_rate_est.set_index("year")

        def recal_ctb_area():
            # --- aggregate the impact by area ---
            processed_area = (
                prep_area(self.max_impact["max_impact"], self.model_name).to_dataset()
                * self.mask
            )
            processed_area = processed_area.assign(
                max_impact=self.max_impact["max_impact"]
            )
            area_dict = {}
            total_a = 0
            for v, f in enumerate(self.predictors):
                area = processed_area.where(
                    processed_area["max_impact"] == v, drop=True
                )
                area = area["areacella"].sum(["lat", "lon"]).item()
                area_dict[f] = area
                total_a += area

            impact_by_area = [area_dict[f] * 100 / total_a for f in self.predictors]
            self.impact_area_df = pd.DataFrame()
            self.impact_area_df["percentage"] = impact_by_area
            self.impact_area_df["vars"] = self.predictors
            self.impact_area_df.plot.bar(x="vars", y="percentage")

        # cal y, xi for mk slope calculation
        y = self.sensitivity_ds["year"]
        xi = xr.DataArray(np.arange(len(y)) + 1, dims="year", coords={"year": y})

        # cal mean of sensitivity ds by climate variables
        clim_fixed_predictors = [f"{f}_fixed" for f in self.clim_predictors]
        self.sensitivity_ds["clim_fixed"] = (
            self.sensitivity_ds[clim_fixed_predictors].to_array().mean("variable")
        )

        # add clim to the predictors to make analysis
        self.main_predictors = [
            f for f in self.predictors if f not in self.clim_predictors
        ] + ["clim"]

        # full climate vars (temp, pre, ...)
        full_pred_fields = [f"{p}_fixed" for p in self.predictors]
        # aggreate climate vars (temp, pre, ...) to clim
        main_pred_fields = [f"{p}_fixed" for p in self.main_predictors]
        all_pred_fields = full_pred_fields + ["clim_fixed"]

        # calculate the most impactful contribution map
        self.ctb_main_map, self.slope_main_dict, self.ctb_main_dict = recal_ctb_map(
            self.sensitivity_ds, main_pred_fields, xi, self.mask
        )
        self.ctb_clim_map, self.slope_clim_dict, self.ctb_clim_dict = recal_ctb_map(
            self.sensitivity_ds, full_pred_fields, xi, self.mask, clim_only=True
        )
        # create the most impactful contribution dataset
        self.impact_map = xr.Dataset(
            {
                "ctb_main": (("lat", "lon"), self.ctb_main_map * self.mask),
                "ctb_clim": (("lat", "lon"), self.ctb_clim_map * self.mask),
            },
            coords={
                "lon": self.sensitivity_ds[self.target_name].lon.values,
                "lat": self.sensitivity_ds[self.target_name].lat.values,
            },
        )
        # cal global rate by each variables
        self.glob_rate_est_clim = recal_ctb_rate(self.sensitivity_ds, full_pred_fields)
        self.glob_rate_est_main = recal_ctb_rate(self.sensitivity_ds, main_pred_fields)
        self.glob_rate_est_all = recal_ctb_rate(self.sensitivity_ds, all_pred_fields)

        # plot max_impact - map
        # self.max_impact["max_impact"].plot(
        #     levels=5,
        #     cmap=matplotlib.colors.ListedColormap(
        #         matplotlib.colormaps["Accent"].colors[: (len(self.predictors))]
        #     ),
        # )

    def plt_impact_rate(self):
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        axbox = ax.get_position()
        pred_fields = [f"{p}_fixed" for p in self.predictors]
        colors_list = ["#7fc97f", "#fb8072", "#fdc086", "#386cb0", "#beaed4"]
        colors_dict = {
            m_name: c for m_name, c in zip(pred_fields, colors_list[: len(pred_fields)])
        }
        for f in pred_fields:
            obj = self.impact_rate[f]
            x, y = obj.index, obj.values
            ax.plot(
                x,
                y,
                label=f,
                linewidth=1.75,
                marker="o",
                ms=4,
                color=colors_dict[f],
                markerfacecolor="white",
                markeredgecolor=colors_dict[f],
            )
        ax.set_xlabel("Year")
        ax.set_ylabel(VIZ_OPT[self.target_name]["line_bar_unit"])
        # plt.ylim([250, 650])
        ax.set_title(
            f"{self.model_name} - Drivers of Annual Trend of {self.target_name}"
        )
        ax.legend(
            loc="center",
            ncol=len(pred_fields),
            bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.25],
        )

    def plt_max_impact_map(self, mode="main"):
        fig = plt.figure(figsize=(12, 9))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(
            xlocs=range(-180, 180, 40),
            ylocs=range(-80, 81, 20),
            draw_labels=True,
            linewidth=1,
            edgecolor="dimgrey",
        )
        if mode == "main":
            predictors = self.main_predictors

            cmap = matplotlib.colors.ListedColormap(["#beaed4", "#7fc97f", "#f0027f"])
            data = self.impact_map["ctb_main"]
        else:
            predictors = self.clim_predictors
            cmap = matplotlib.colors.ListedColormap(["#fb8072", "#fdc086", "#386cb0"])
            data = self.impact_map["ctb_clim"]
        center = [0.5 * (i * 2 + 1) for i in range(len(predictors))]
        cax = data.plot(
            cmap=cmap,
            vmin=0,
            vmax=len(predictors),
            ax=ax,
            add_colorbar=False,
        )
        cbar = fig.colorbar(
            cax,
            ticks=center,
            orientation="horizontal",
            pad=0.05,
        )
        cbar.ax.set_xticklabels(predictors, size=18)
        title = f"{self.model_name} - Drivers of changes in {self.target_name}"
        cbar.set_label(label="Dominant driver", size=18, weight="bold")
        plt.title(title, fontsize=18)

    def plt_contri_map(self, mode="main"):
        if mode == "main":
            pred_fields = self.main_predictors
            ds = self.ctb_main_dict
        else:
            pred_fields = self.main_predictors
            ds = self.ctb_clim_dict
        i = 0
        for f in pred_fields:
            i = i + 1
            fig = plt.figure(1 + i, figsize=(12, 9))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.gridlines(
                xlocs=range(-180, 180, 40),
                ylocs=range(-80, 81, 20),
                draw_labels=True,
                linewidth=1,
                edgecolor="dimgrey",
            )
            title = f"{self.model_name} - Contribution of {f} to the {self.target_name} trends"
            data = ds[f]
            data.plot.pcolormesh(
                ax=ax,
                cmap="PiYG_r",
                levels=21,
                vmin=-0.005,
                vmax=0.005,
                extend="both",
                cbar_kwargs={
                    "label": VIZ_OPT[self.target_name]["map_unit"],
                    "orientation": "horizontal",
                    "pad": 0.05,
                },
            )
            plt.title(title, fontsize=18)


# %%
model_name = "CESM2-WACCM"
a = RegSingleModel(model_name, start_year=2005, end_year=2014)

# %%
import pickle
import copy
import numpy as np

from mk import *
from senAls_mulLinear import *
from senAls_ML import *
from sklearn.ensemble import RandomForestRegressor


class RegSingleModel:
    def __init__(self, model_name="VISIT", start_year=2005, end_year=2014) -> None:
        self.target_name = "emiisop"
        self.model_name = model_name

        self.start_year = start_year
        self.end_year = end_year

        self.clim_predictors = ["tas", "rsds", "pr"]
        self.other_predictors = ["lai"]

        self.org_ds = None

        self.get_predictors()
        self.read_data()
        self.train_RF()
        self.extr_mask()
        self.sensitivity_cal()
        self.contribution_cal()

    def get_predictors(self):
        if self.model_name in ["CESM2-WACCM", "NorESM2-LM", "UKESM1-0-LL"]:
            self.predictors = self.clim_predictors + ["co2s"] + self.other_predictors
        else:
            self.predictors = self.clim_predictors + self.other_predictors

    def read_data(self):
        iters = self.predictors + [self.target_name]
        dss = {}
        for p in iters:
            f = os.path.join(RES_DIR, f"{self.model_name}_{p}.nc")
            ds = xr.open_dataset(f)[p].sel(year=slice(self.start_year, self.end_year))
            # ds = ds.sel(year=slice(1901, 2014)) if "VISIT" in self.model_name else ds
            dss[p] = (("lat", "lon", "year"), ds.transpose("lat", "lon", "year").data)
        lat = ds.lat.values
        lon = ds.lon.values
        year = ds.year.values
        self.org_ds = xr.Dataset(dss, coords={"lat": lat, "lon": lon, "year": year})

    def train_RF(self):
        model_folder = os.path.join(
            "models", f"{self.start_year}_{self.end_year}", "_".join(self.predictors)
        )
        model_path = os.path.join(model_folder, f"{self.model_name}.sav")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        ts_data = self.org_ds.copy()
        ts_data = ts_data.where(ts_data[self.target_name] != 0, drop=True)
        ts_data = ts_data.to_dataframe().reset_index()
        self.ts_data_df = ts_data

        if not os.path.exists(model_path):
            ts_data = ts_data[ts_data[self.target_name].notna()]
            print("number of samples:", len(ts_data))
            ts_data = ts_data.fillna(0)

            X = ts_data.drop(self.target_name, axis=1).values
            y = ts_data[self.target_name].values.reshape(-1, 1)
            rf = RandomForestRegressor().fit(X, y)
            self.model = rf
            pickle.dump(self.model, open(model_path, "wb"))
        else:
            self.model = pickle.load(open(model_path, "rb"))

    def extr_mask(self):
        mask = self.org_ds.where(self.org_ds[self.target_name] != 0, drop=True)
        mask = mask[self.target_name].isel(year=0).values
        mask[mask > 0] = 1
        self.mask = mask

    def sensitivity_cal(self):
        def get_fixed_predictors(df, p_col, year):
            keys = ["lat", "lon"]
            df_y_fix = df[df["year"] == year][[p_col] + keys]

            df_no_pcol = df.drop(p_col, axis=1)
            df_fixed_pcol = df_no_pcol.merge(df_y_fix, how="left", on=keys).fillna(0)
            return df_fixed_pcol

        org_ts_data = self.ts_data_df.copy()
        # filter no na values
        org_ts_data = org_ts_data[org_ts_data[self.target_name].notna()].fillna(0)
        org_X = org_ts_data.drop(self.target_name, axis=1).values

        self.org_y = org_ts_data[self.target_name].values

        self.regressed_data = self.model.predict(org_X)
        print("R:", pearsonr(self.org_y, self.regressed_data))

        df_est = copy.deepcopy(org_ts_data)
        df_est["reg"] = self.regressed_data
        for p in self.predictors:
            fixed_var_df = org_ts_data.drop(self.target_name, axis=1)
            # get fixed predictors
            fixed_predictors = get_fixed_predictors(fixed_var_df, p, self.start_year)
            fixed_predictors = fixed_predictors[list(self.ts_data_df.columns[:-1])]

            est = self.model.predict(fixed_predictors.values)
            assert len(fixed_var_df) == len(est)
            df_est[f"{p}_fixed"] = est

        keys = ["lat", "lon", "year"]
        df_est = df_est[keys + ["reg"] + [f"{p}_fixed" for p in self.predictors]]

        self.sensitivity_ds = (
            self.ts_data_df.merge(df_est, how="left", on=keys)
            .groupby(keys)
            .mean()
            .to_xarray()
        )

    def contribution_cal(self):
        def recal_ctb_map(sensitivity_ds, fields, xi, mask, clim_only=False):
            list_ctb_map = []  # contribution map
            ctb_dict = {}
            n = len(fields)
            # cal mk time series mk slope
            ds_mk_slope = sensitivity_ds.copy()
            for f in fields:
                ds_mk_slope[f] = kendall_correlation(ds_mk_slope[f], xi, "year") * mask
            # cal contribution
            for f in fields:
                Ei = ds_mk_slope[f]  # cal Ei
                # cal Ek
                k_fields = [k for k in fields if k != f]
                Ek = ds_mk_slope[k_fields].to_array().sum("variable")
                # cal contribution
                ctb = (Ek - (n - 2) * Ei) / (n - 1)
                list_ctb_map.append(np.absolute(ctb.values))
                ctb_dict[f] = ctb
            # --- calculate max_ctb_map - map ---
            if clim_only:  # only for climate vars (temp, prep, ...)
                list_ctb_map = list_ctb_map[: len(self.clim_predictors)]
            # stack n x ctb map (X x Y) to (X x Y x n)
            stacked_ctb = np.stack((i for i in list_ctb_map), axis=-1)
            # find the max values of the n dim and return the index (axis = -1 means the last dim of (X x Y x n))
            max_ctb_map = np.argmax(np.nan_to_num(stacked_ctb), axis=-1)
            return max_ctb_map, ds_mk_slope, ctb_dict

        def recal_ctb_rate(sensitivity_ds, org_fields):
            glob_rate_est = pd.DataFrame()
            # org_fields = list(sensitivity_ds.data_vars)
            fields = [self.target_name, "reg"] + org_fields
            for f in fields:
                ds_cp = sensitivity_ds.copy()
                glob_rate_est[f], _ = cal_actual_rate(ds_cp[f], model_name, mode="ts")

            glob_rate_est["year"] = sensitivity_ds.year.values
            return glob_rate_est.set_index("year")

        def recal_ctb_area():
            # --- aggregate the impact by area ---
            processed_area = (
                prep_area(self.max_impact["max_impact"], self.model_name).to_dataset()
                * self.mask
            )
            processed_area = processed_area.assign(
                max_impact=self.max_impact["max_impact"]
            )
            area_dict = {}
            total_a = 0
            for v, f in enumerate(self.predictors):
                area = processed_area.where(
                    processed_area["max_impact"] == v, drop=True
                )
                area = area["areacella"].sum(["lat", "lon"]).item()
                area_dict[f] = area
                total_a += area

            impact_by_area = [area_dict[f] * 100 / total_a for f in self.predictors]
            self.impact_area_df = pd.DataFrame()
            self.impact_area_df["percentage"] = impact_by_area
            self.impact_area_df["vars"] = self.predictors
            self.impact_area_df.plot.bar(x="vars", y="percentage")

        # cal y, xi for mk slope calculation
        y = self.sensitivity_ds["year"]
        xi = xr.DataArray(np.arange(len(y)) + 1, dims="year", coords={"year": y})

        # cal mean of sensitivity ds by climate variables
        clim_fixed_predictors = [f"{f}_fixed" for f in self.clim_predictors]
        self.sensitivity_ds["clim_fixed"] = (
            self.sensitivity_ds[clim_fixed_predictors].to_array().mean("variable")
        )

        # add clim to the predictors to make analysis
        self.main_predictors = [
            f for f in self.predictors if f not in self.clim_predictors
        ] + ["clim"]

        # full climate vars (temp, pre, ...)
        full_pred_fields = [f"{p}_fixed" for p in self.predictors]
        # aggreate climate vars (temp, pre, ...) to clim
        main_pred_fields = [f"{p}_fixed" for p in self.main_predictors]
        all_pred_fields = full_pred_fields + ["clim_fixed"]

        # calculate the most impactful contribution map
        self.ctb_main_map, self.slope_main_dict, self.ctb_main_dict = recal_ctb_map(
            self.sensitivity_ds, main_pred_fields, xi, self.mask
        )
        self.ctb_clim_map, self.slope_clim_dict, self.ctb_clim_dict = recal_ctb_map(
            self.sensitivity_ds, full_pred_fields, xi, self.mask, clim_only=True
        )
        # create the most impactful contribution dataset
        self.impact_map = xr.Dataset(
            {
                "ctb_main": (("lat", "lon"), self.ctb_main_map * self.mask),
                "ctb_clim": (("lat", "lon"), self.ctb_clim_map * self.mask),
            },
            coords={
                "lon": self.sensitivity_ds[self.target_name].lon.values,
                "lat": self.sensitivity_ds[self.target_name].lat.values,
            },
        )
        # cal global rate by each variables
        self.glob_rate_est_clim = recal_ctb_rate(self.sensitivity_ds, full_pred_fields)
        self.glob_rate_est_main = recal_ctb_rate(self.sensitivity_ds, main_pred_fields)
        self.glob_rate_est_all = recal_ctb_rate(self.sensitivity_ds, all_pred_fields)

        # plot max_impact - map
        # self.max_impact["max_impact"].plot(
        #     levels=5,
        #     cmap=matplotlib.colors.ListedColormap(
        #         matplotlib.colormaps["Accent"].colors[: (len(self.predictors))]
        #     ),
        # )

    def plt_impact_rate(self):
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        axbox = ax.get_position()
        pred_fields = [f"{p}_fixed" for p in self.predictors]
        colors_list = ["#7fc97f", "#fb8072", "#fdc086", "#386cb0", "#beaed4"]
        colors_dict = {
            m_name: c for m_name, c in zip(pred_fields, colors_list[: len(pred_fields)])
        }
        for f in pred_fields:
            obj = self.impact_rate[f]
            x, y = obj.index, obj.values
            ax.plot(
                x,
                y,
                label=f,
                linewidth=1.75,
                marker="o",
                ms=4,
                color=colors_dict[f],
                markerfacecolor="white",
                markeredgecolor=colors_dict[f],
            )
        ax.set_xlabel("Year")
        ax.set_ylabel(VIZ_OPT[self.target_name]["line_bar_unit"])
        # plt.ylim([250, 650])
        ax.set_title(
            f"{self.model_name} - Drivers of Annual Trend of {self.target_name}"
        )
        ax.legend(
            loc="center",
            ncol=len(pred_fields),
            bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.25],
        )

    def plt_max_impact_map(self, mode="main"):
        fig = plt.figure(figsize=(12, 9))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(
            xlocs=range(-180, 180, 40),
            ylocs=range(-80, 81, 20),
            draw_labels=True,
            linewidth=1,
            edgecolor="dimgrey",
        )
        if mode == "main":
            predictors = self.main_predictors

            cmap = matplotlib.colors.ListedColormap(["#beaed4", "#7fc97f", "#f0027f"])
            data = self.impact_map["ctb_main"]
        else:
            predictors = self.clim_predictors
            cmap = matplotlib.colors.ListedColormap(["#fb8072", "#fdc086", "#386cb0"])
            data = self.impact_map["ctb_clim"]
        center = [0.5 * (i * 2 + 1) for i in range(len(predictors))]
        cax = data.plot(
            cmap=cmap,
            vmin=0,
            vmax=len(predictors),
            ax=ax,
            add_colorbar=False,
        )
        cbar = fig.colorbar(
            cax,
            ticks=center,
            orientation="horizontal",
            pad=0.05,
        )
        cbar.ax.set_xticklabels(predictors, size=18)
        title = f"{self.model_name} - Drivers of changes in {self.target_name}"
        cbar.set_label(label="Dominant driver", size=18, weight="bold")
        plt.title(title, fontsize=18)

    def plt_contri_map(self, mode="main"):
        if mode == "main":
            pred_fields = [f"{p}_fixed" for p in self.main_predictors]
            ds = self.ctb_main_dict
        else:
            pred_fields = [f"{f}_fixed" for f in self.clim_predictors]
            ds = self.ctb_clim_dict
        i = 0
        for f in pred_fields:
            i = i + 1
            fig = plt.figure(1 + i, figsize=(12, 9))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.gridlines(
                xlocs=range(-180, 180, 40),
                ylocs=range(-80, 81, 20),
                draw_labels=True,
                linewidth=1,
                edgecolor="dimgrey",
            )
            title = f"{self.model_name} - Contribution of {f} to the {self.target_name} trends"
            data = ds[f]
            data.plot.pcolormesh(
                ax=ax,
                cmap="PiYG_r",
                levels=21,
                vmin=-0.005,
                vmax=0.005,
                extend="both",
                cbar_kwargs={
                    "label": VIZ_OPT[self.target_name]["map_unit"],
                    "orientation": "horizontal",
                    "pad": 0.05,
                },
            )
            plt.title(title, fontsize=18)


# %%
model_name = "CESM2-WACCM"
a = RegSingleModel(model_name, start_year=2005, end_year=2014)

# %%
a.plt_contri_map()

# %%
a.plt_contri_map(mode="clim")

# %%
a.plt_max_impact_map()

# %%
a.plt_max_impact_map(mode="clim")

# %%
a.sensitivity_ds

# %%
glob_rate_ctb = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb["year"] = a.sensitivity_ds.year.values
glob_rate_ctb.set_index("year")

# %%
glob_rate_ctb.plot(x="year", cmap="Accent")

# %%
model_name = "CESM2-WACCM"
a = RegSingleModel(model_name, start_year=1980, end_year=2005)

# %%
a.plt_contri_map()

# %%
a.plt_contri_map(mode="clim")

# %%
glob_rate_ctb = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb["year"] = a.sensitivity_ds.year.values
glob_rate_ctb.set_index("year")

# %%
glob_rate_ctb.plot(x="year", cmap="Accent")

# %%
a.plt_max_impact_map()

# %%
a.plt_max_impact_map(mode="clim")

# %%
model_name = "CESM2-WACCM"
a = RegSingleModel(model_name, start_year=1850, end_year=2014)

# %%
a.glob_rate_est_all.plot()

# %%
glob_rate_ctb = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb["year"] = a.sensitivity_ds.year.values
glob_rate_ctb.set_index("year")

# %%
glob_rate_ctb.plot(x="year", cmap="Accent")

# %%
a.plt_contri_map()

# %%
a.plt_max_impact_map()

# %%
a.plt_max_impact_map(mode="clim")

# %%
a.plt_contri_map(mode="clim")

# %%
import pickle
import copy
import numpy as np

from mk import *
from senAls_mulLinear import *
from senAls_ML import *
from sklearn.ensemble import RandomForestRegressor


class RegSingleModel:
    def __init__(self, model_name="VISIT", start_year=2005, end_year=2014) -> None:
        self.target_name = "emiisop"
        self.model_name = model_name

        self.start_year = start_year
        self.end_year = end_year

        self.clim_predictors = ["tas", "rsds", "pr"]
        self.other_predictors = ["lai"]

        self.org_ds = None

        self.get_predictors()
        self.read_data()
        self.train_RF()
        self.extr_mask()
        self.sensitivity_cal()
        self.contribution_cal()

    def get_predictors(self):
        if self.model_name in ["CESM2-WACCM", "NorESM2-LM", "UKESM1-0-LL"]:
            self.predictors = self.clim_predictors + ["co2s"] + self.other_predictors
        else:
            self.predictors = self.clim_predictors + self.other_predictors

    def read_data(self):
        iters = self.predictors + [self.target_name]
        dss = {}
        for p in iters:
            f = os.path.join(RES_DIR, f"{self.model_name}_{p}.nc")
            ds = xr.open_dataset(f)[p].sel(year=slice(self.start_year, self.end_year))
            # ds = ds.sel(year=slice(1901, 2014)) if "VISIT" in self.model_name else ds
            dss[p] = (("lat", "lon", "year"), ds.transpose("lat", "lon", "year").data)
        lat = ds.lat.values
        lon = ds.lon.values
        year = ds.year.values
        self.org_ds = xr.Dataset(dss, coords={"lat": lat, "lon": lon, "year": year})

    def train_RF(self):
        model_folder = os.path.join(
            "models", f"{self.start_year}_{self.end_year}", "_".join(self.predictors)
        )
        model_path = os.path.join(model_folder, f"{self.model_name}.sav")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        ts_data = self.org_ds.copy()
        ts_data = ts_data.where(ts_data[self.target_name] != 0, drop=True)
        ts_data = ts_data.to_dataframe().reset_index()
        self.ts_data_df = ts_data

        if not os.path.exists(model_path):
            ts_data = ts_data[ts_data[self.target_name].notna()]
            print("number of samples:", len(ts_data))
            ts_data = ts_data.fillna(0)

            X = ts_data.drop(self.target_name, axis=1).values
            y = ts_data[self.target_name].values.reshape(-1, 1)
            rf = RandomForestRegressor().fit(X, y)
            self.model = rf
            pickle.dump(self.model, open(model_path, "wb"))
        else:
            self.model = pickle.load(open(model_path, "rb"))

    def extr_mask(self):
        mask = self.org_ds.where(self.org_ds[self.target_name] != 0, drop=True)
        mask = mask[self.target_name].isel(year=0).values
        mask[mask > 0] = 1
        self.mask = mask

    def sensitivity_cal(self):
        def get_fixed_predictors(df, p_col, year):
            keys = ["lat", "lon"]
            df_y_fix = df[df["year"] == year][[p_col] + keys]

            df_no_pcol = df.drop(p_col, axis=1)
            df_fixed_pcol = df_no_pcol.merge(df_y_fix, how="left", on=keys).fillna(0)
            return df_fixed_pcol

        org_ts_data = self.ts_data_df.copy()
        # filter no na values
        org_ts_data = org_ts_data[org_ts_data[self.target_name].notna()].fillna(0)
        org_X = org_ts_data.drop(self.target_name, axis=1).values

        self.org_y = org_ts_data[self.target_name].values

        self.regressed_data = self.model.predict(org_X)
        print("R:", pearsonr(self.org_y, self.regressed_data))

        df_est = copy.deepcopy(org_ts_data)
        df_est["reg"] = self.regressed_data
        for p in self.predictors:
            fixed_var_df = org_ts_data.drop(self.target_name, axis=1)
            # get fixed predictors
            fixed_predictors = get_fixed_predictors(fixed_var_df, p, self.start_year)
            fixed_predictors = fixed_predictors[list(self.ts_data_df.columns[:-1])]

            est = self.model.predict(fixed_predictors.values)
            assert len(fixed_var_df) == len(est)
            df_est[f"{p}_fixed"] = est

        keys = ["lat", "lon", "year"]
        df_est = df_est[keys + ["reg"] + [f"{p}_fixed" for p in self.predictors]]

        self.sensitivity_ds = (
            self.ts_data_df.merge(df_est, how="left", on=keys)
            .groupby(keys)
            .mean()
            .to_xarray()
        )

    def contribution_cal(self):
        def recal_ctb_map(sensitivity_ds, fields, xi, mask, clim_only=False):
            list_ctb_map = []  # contribution map
            ctb_dict = {}
            n = len(fields)
            # cal mk time series mk slope
            ds_mk_slope = sensitivity_ds.copy()
            for f in fields:
                ds_mk_slope[f] = kendall_correlation(ds_mk_slope[f], xi, "year") * mask
            # cal contribution
            for f in fields:
                Ei = ds_mk_slope[f]  # cal Ei
                # cal Ek
                k_fields = [k for k in fields if k != f]
                Ek = ds_mk_slope[k_fields].to_array().sum("variable")
                # cal contribution
                ctb = (Ek - (n - 2) * Ei) / (n - 1)
                list_ctb_map.append(np.absolute(ctb.values))
                ctb_dict[f] = ctb
            # --- calculate max_ctb_map - map ---
            if clim_only:  # only for climate vars (temp, prep, ...)
                list_ctb_map = list_ctb_map[: len(self.clim_predictors)]
            # stack n x ctb map (X x Y) to (X x Y x n)
            stacked_ctb = np.stack((i for i in list_ctb_map), axis=-1)
            # find the max values of the n dim and return the index (axis = -1 means the last dim of (X x Y x n))
            max_ctb_map = np.argmax(np.nan_to_num(stacked_ctb), axis=-1)
            return max_ctb_map, ds_mk_slope, ctb_dict

        def recal_ctb_rate(sensitivity_ds, org_fields):
            glob_rate_est = pd.DataFrame()
            # org_fields = list(sensitivity_ds.data_vars)
            fields = [self.target_name, "reg"] + org_fields
            for f in fields:
                ds_cp = sensitivity_ds.copy()
                glob_rate_est[f], _ = cal_actual_rate(ds_cp[f], model_name, mode="ts")

            glob_rate_est["year"] = sensitivity_ds.year.values
            return glob_rate_est.set_index("year")

        def recal_ctb_area():
            # --- aggregate the impact by area ---
            processed_area = (
                prep_area(self.max_impact["max_impact"], self.model_name).to_dataset()
                * self.mask
            )
            processed_area = processed_area.assign(
                max_impact=self.max_impact["max_impact"]
            )
            area_dict = {}
            total_a = 0
            for v, f in enumerate(self.predictors):
                area = processed_area.where(
                    processed_area["max_impact"] == v, drop=True
                )
                area = area["areacella"].sum(["lat", "lon"]).item()
                area_dict[f] = area
                total_a += area

            impact_by_area = [area_dict[f] * 100 / total_a for f in self.predictors]
            self.impact_area_df = pd.DataFrame()
            self.impact_area_df["percentage"] = impact_by_area
            self.impact_area_df["vars"] = self.predictors
            self.impact_area_df.plot.bar(x="vars", y="percentage")

        # cal y, xi for mk slope calculation
        y = self.sensitivity_ds["year"]
        xi = xr.DataArray(np.arange(len(y)) + 1, dims="year", coords={"year": y})

        # cal mean of sensitivity ds by climate variables
        clim_fixed_predictors = [f"{f}_fixed" for f in self.clim_predictors]
        self.sensitivity_ds["clim_fixed"] = (
            self.sensitivity_ds[clim_fixed_predictors].to_array().mean("variable")
        )

        # add clim to the predictors to make analysis
        self.main_predictors = [
            f for f in self.predictors if f not in self.clim_predictors
        ] + ["clim"]

        # full climate vars (temp, pre, ...)
        full_pred_fields = [f"{p}_fixed" for p in self.predictors]
        # aggreate climate vars (temp, pre, ...) to clim
        main_pred_fields = [f"{p}_fixed" for p in self.main_predictors]
        all_pred_fields = full_pred_fields + ["clim_fixed"]

        # calculate the most impactful contribution map
        self.ctb_main_map, self.slope_main_dict, self.ctb_main_dict = recal_ctb_map(
            self.sensitivity_ds, main_pred_fields, xi, self.mask
        )
        self.ctb_clim_map, self.slope_clim_dict, self.ctb_clim_dict = recal_ctb_map(
            self.sensitivity_ds, full_pred_fields, xi, self.mask, clim_only=True
        )
        # create the most impactful contribution dataset
        self.impact_map = xr.Dataset(
            {
                "ctb_main": (("lat", "lon"), self.ctb_main_map * self.mask),
                "ctb_clim": (("lat", "lon"), self.ctb_clim_map * self.mask),
            },
            coords={
                "lon": self.sensitivity_ds[self.target_name].lon.values,
                "lat": self.sensitivity_ds[self.target_name].lat.values,
            },
        )
        # cal global rate by each variables
        self.glob_rate_est_clim = recal_ctb_rate(self.sensitivity_ds, full_pred_fields)
        self.glob_rate_est_main = recal_ctb_rate(self.sensitivity_ds, main_pred_fields)
        self.glob_rate_est_all = recal_ctb_rate(self.sensitivity_ds, all_pred_fields)

        # plot max_impact - map
        # self.max_impact["max_impact"].plot(
        #     levels=5,
        #     cmap=matplotlib.colors.ListedColormap(
        #         matplotlib.colormaps["Accent"].colors[: (len(self.predictors))]
        #     ),
        # )

    def plt_impact_rate(self):
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        axbox = ax.get_position()
        pred_fields = [f"{p}_fixed" for p in self.predictors]
        colors_list = ["#7fc97f", "#fb8072", "#fdc086", "#386cb0", "#beaed4"]
        colors_dict = {
            m_name: c for m_name, c in zip(pred_fields, colors_list[: len(pred_fields)])
        }
        for f in pred_fields:
            obj = self.impact_rate[f]
            x, y = obj.index, obj.values
            ax.plot(
                x,
                y,
                label=f,
                linewidth=1.75,
                marker="o",
                ms=4,
                color=colors_dict[f],
                markerfacecolor="white",
                markeredgecolor=colors_dict[f],
            )
        ax.set_xlabel("Year")
        ax.set_ylabel(VIZ_OPT[self.target_name]["line_bar_unit"])
        # plt.ylim([250, 650])
        ax.set_title(
            f"{self.model_name} - Drivers of Annual Trend of {self.target_name}"
        )
        ax.legend(
            loc="center",
            ncol=len(pred_fields),
            bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.25],
        )

    def plt_max_impact_map(self, mode="main"):
        fig = plt.figure(figsize=(12, 9))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(
            xlocs=range(-180, 180, 40),
            ylocs=range(-80, 81, 20),
            draw_labels=True,
            linewidth=1,
            edgecolor="dimgrey",
        )
        if mode == "main":
            predictors = self.main_predictors

            cmap = matplotlib.colors.ListedColormap(["#beaed4", "#7fc97f", "#f0027f"])
            data = self.impact_map["ctb_main"]
        else:
            predictors = self.clim_predictors
            cmap = matplotlib.colors.ListedColormap(["#fb8072", "#fdc086", "#386cb0"])
            data = self.impact_map["ctb_clim"]
        center = [0.5 * (i * 2 + 1) for i in range(len(predictors))]
        cax = data.plot(
            cmap=cmap,
            vmin=0,
            vmax=len(predictors),
            ax=ax,
            add_colorbar=False,
        )
        cbar = fig.colorbar(
            cax,
            ticks=center,
            orientation="horizontal",
            pad=0.05,
        )
        cbar.ax.set_xticklabels(predictors, size=18)
        title = f"{self.model_name} - Drivers of changes in {self.target_name}"
        cbar.set_label(label="Dominant driver", size=18, weight="bold")
        plt.title(title, fontsize=18)

    def plt_contri_map(self, mode="main"):
        if mode == "main":
            pred_fields = [f"{p}_fixed" for p in self.main_predictors]
            ds = self.ctb_main_dict
        else:
            pred_fields = [f"{f}_fixed" for f in self.clim_predictors]
            ds = self.ctb_clim_dict
        i = 0
        for f in pred_fields:
            i = i + 1
            fig = plt.figure(1 + i, figsize=(12, 9))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.gridlines(
                xlocs=range(-180, 180, 40),
                ylocs=range(-80, 81, 20),
                draw_labels=True,
                linewidth=1,
                edgecolor="dimgrey",
            )
            title = f"{self.model_name} - Contribution of {f} to the {self.target_name} trends"
            data = ds[f]
            data.plot.pcolormesh(
                ax=ax,
                cmap="PiYG_r",
                levels=21,
                vmin=-0.005,
                vmax=0.005,
                extend="both",
                cbar_kwargs={
                    "label": VIZ_OPT[self.target_name]["map_unit"],
                    "orientation": "horizontal",
                    "pad": 0.05,
                },
            )
            plt.title(title, fontsize=18)


# %%
model_name = "CESM2-WACCM"
a = RegSingleModel(model_name, start_year=1850, end_year=2014)

# %%
a.glob_rate_est_main.plot()

# %%
glob_rate_ctb = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb["year"] = a.sensitivity_ds.year.values
glob_rate_ctb.set_index("year")
glob_rate_ctb.plot(x="year", cmap="Accent")

# %%
glob_rate_ctb_main = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.main_predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb_main[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb_main["year"] = a.sensitivity_ds.year.values
glob_rate_ctb_main.set_index("year")
glob_rate_ctb_main.plot(x="year", cmap="Accent")

# %%
a.slope_main_dict

# %%
model_name = ["GFDL-ESM4", "GISS-E2-1-G", "NorESM2-LM", "UKESM1-0-LL"]
for m in model_name:
    a = RegSingleModel(m, start_year=1850, end_year=2014)

# %%
import pickle
import copy
import numpy as np

from mk import *
from senAls_mulLinear import *
from senAls_ML import *
from sklearn.ensemble import RandomForestRegressor


class RegSingleModel:
    def __init__(self, model_name="VISIT", start_year=2005, end_year=2014) -> None:
        self.target_name = "emiisop"
        self.model_name = model_name

        self.start_year = start_year
        self.end_year = end_year

        self.clim_predictors = ["tas", "rsds", "pr"]
        self.other_predictors = ["lai"]

        self.org_ds = None

        self.get_predictors()
        self.read_data()
        self.train_RF()
        self.extr_mask()
        self.sensitivity_cal()
        self.contribution_cal()

    def get_predictors(self):
        if self.model_name in ["CESM2-WACCM", "NorESM2-LM", "UKESM1-0-LL"]:
            self.predictors = self.clim_predictors + ["co2s"] + self.other_predictors
        else:
            self.predictors = self.clim_predictors + self.other_predictors

    def read_data(self):
        iters = self.predictors + [self.target_name]
        dss = {}
        for p in iters:
            f = os.path.join(RES_DIR, f"{self.model_name}_{p}.nc")
            ds = xr.open_dataset(f)[p].sel(year=slice(self.start_year, self.end_year))
            # ds = ds.sel(year=slice(1901, 2014)) if "VISIT" in self.model_name else ds
            dss[p] = (("lat", "lon", "year"), ds.transpose("lat", "lon", "year").data)
        lat = ds.lat.values
        lon = ds.lon.values
        year = ds.year.values
        self.org_ds = xr.Dataset(dss, coords={"lat": lat, "lon": lon, "year": year})

    def train_RF(self):
        model_folder = os.path.join(
            "models", f"{self.start_year}_{self.end_year}", "_".join(self.predictors)
        )
        model_path = os.path.join(model_folder, f"{self.model_name}.sav")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        ts_data = self.org_ds.copy()
        ts_data = ts_data.where(ts_data[self.target_name] != 0, drop=True)
        ts_data = ts_data.to_dataframe().reset_index()
        self.ts_data_df = ts_data

        if not os.path.exists(model_path):
            ts_data = ts_data[ts_data[self.target_name].notna()]
            print("number of samples:", len(ts_data))
            ts_data = ts_data.fillna(0)

            X = ts_data.drop(self.target_name, axis=1).values
            y = ts_data[self.target_name].values.reshape(-1, 1)
            rf = RandomForestRegressor().fit(X, y)
            self.model = rf
            pickle.dump(self.model, open(model_path, "wb"))
        else:
            self.model = pickle.load(open(model_path, "rb"))

    def extr_mask(self):
        mask = self.org_ds.where(self.org_ds[self.target_name] != 0, drop=True)
        mask = mask[self.target_name].isel(year=0).values
        mask[mask > 0] = 1
        self.mask = mask

    def sensitivity_cal(self):
        def get_fixed_predictors(df, p_col, year):
            keys = ["lat", "lon"]
            df_y_fix = df[df["year"] == year][[p_col] + keys]

            df_no_pcol = df.drop(p_col, axis=1)
            df_fixed_pcol = df_no_pcol.merge(df_y_fix, how="left", on=keys).fillna(0)
            return df_fixed_pcol

        org_ts_data = self.ts_data_df.copy()
        # filter no na values
        org_ts_data = org_ts_data[org_ts_data[self.target_name].notna()].fillna(0)
        org_X = org_ts_data.drop(self.target_name, axis=1).values

        self.org_y = org_ts_data[self.target_name].values

        self.regressed_data = self.model.predict(org_X)
        print("R:", pearsonr(self.org_y, self.regressed_data))

        df_est = copy.deepcopy(org_ts_data)
        df_est["reg"] = self.regressed_data
        for p in self.predictors:
            fixed_var_df = org_ts_data.drop(self.target_name, axis=1)
            # get fixed predictors
            fixed_predictors = get_fixed_predictors(fixed_var_df, p, self.start_year)
            fixed_predictors = fixed_predictors[list(self.ts_data_df.columns[:-1])]

            est = self.model.predict(fixed_predictors.values)
            assert len(fixed_var_df) == len(est)
            df_est[f"{p}_fixed"] = est

        keys = ["lat", "lon", "year"]
        df_est = df_est[keys + ["reg"] + [f"{p}_fixed" for p in self.predictors]]

        self.sensitivity_ds = (
            self.ts_data_df.merge(df_est, how="left", on=keys)
            .groupby(keys)
            .mean()
            .to_xarray()
        )

    def contribution_cal(self):
        def recal_ctb_map(sensitivity_ds, fields, xi, mask, clim_only=False):
            list_ctb_map = []  # contribution map
            ctb_dict = {}
            n = len(fields)
            # cal mk time series mk slope
            ds_mk_slope = sensitivity_ds.copy()
            for f in fields:
                ds_mk_slope[f] = kendall_correlation(ds_mk_slope[f], xi, "year") * mask
            # cal contribution
            for f in fields:
                Ei = ds_mk_slope[f]  # cal Ei
                # cal Ek
                k_fields = [k for k in fields if k != f]
                Ek = ds_mk_slope[k_fields].to_array().sum("variable")
                # cal contribution
                ctb = (Ek - (n - 2) * Ei) / (n - 1)
                list_ctb_map.append(np.absolute(ctb.values))
                ctb_dict[f] = ctb
            # --- calculate max_ctb_map - map ---
            if clim_only:  # only for climate vars (temp, prep, ...)
                list_ctb_map = list_ctb_map[: len(self.clim_predictors)]
            # stack n x ctb map (X x Y) to (X x Y x n)
            stacked_ctb = np.stack((i for i in list_ctb_map), axis=-1)
            # find the max values of the n dim and return the index (axis = -1 means the last dim of (X x Y x n))
            max_ctb_map = np.argmax(np.nan_to_num(stacked_ctb), axis=-1)
            return max_ctb_map, ds_mk_slope, ctb_dict

        def recal_ctb_rate(sensitivity_ds, org_fields):
            glob_rate_est = pd.DataFrame()
            # org_fields = list(sensitivity_ds.data_vars)
            fields = [self.target_name, "reg"] + org_fields
            for f in fields:
                ds_cp = sensitivity_ds.copy()
                glob_rate_est[f], _ = cal_actual_rate(ds_cp[f], model_name, mode="ts")

            glob_rate_est["year"] = sensitivity_ds.year.values
            return glob_rate_est.set_index("year")

        def recal_ctb_area():
            # --- aggregate the impact by area ---
            processed_area = (
                prep_area(self.max_impact["max_impact"], self.model_name).to_dataset()
                * self.mask
            )
            processed_area = processed_area.assign(
                max_impact=self.max_impact["max_impact"]
            )
            area_dict = {}
            total_a = 0
            for v, f in enumerate(self.predictors):
                area = processed_area.where(
                    processed_area["max_impact"] == v, drop=True
                )
                area = area["areacella"].sum(["lat", "lon"]).item()
                area_dict[f] = area
                total_a += area

            impact_by_area = [area_dict[f] * 100 / total_a for f in self.predictors]
            self.impact_area_df = pd.DataFrame()
            self.impact_area_df["percentage"] = impact_by_area
            self.impact_area_df["vars"] = self.predictors
            self.impact_area_df.plot.bar(x="vars", y="percentage")

        # cal y, xi for mk slope calculation
        y = self.sensitivity_ds["year"]
        xi = xr.DataArray(np.arange(len(y)) + 1, dims="year", coords={"year": y})

        # cal mean of sensitivity ds by climate variables
        clim_fixed_predictors = [f"{f}_fixed" for f in self.clim_predictors]
        self.sensitivity_ds["clim_fixed"] = (
            self.sensitivity_ds[clim_fixed_predictors].to_array().mean("variable")
        )

        # add clim to the predictors to make analysis
        self.main_predictors = [
            f for f in self.predictors if f not in self.clim_predictors
        ] + ["clim"]

        # full climate vars (temp, pre, ...)
        full_pred_fields = [f"{p}_fixed" for p in self.predictors]
        # aggreate climate vars (temp, pre, ...) to clim
        main_pred_fields = [f"{p}_fixed" for p in self.main_predictors]
        all_pred_fields = full_pred_fields + ["clim_fixed"]

        # calculate the most impactful contribution map
        self.ctb_main_map, self.slope_main_dict, self.ctb_main_dict = recal_ctb_map(
            self.sensitivity_ds, main_pred_fields, xi, self.mask
        )
        self.ctb_clim_map, self.slope_clim_dict, self.ctb_clim_dict = recal_ctb_map(
            self.sensitivity_ds, full_pred_fields, xi, self.mask, clim_only=True
        )
        # create the most impactful contribution dataset
        self.impact_map = xr.Dataset(
            {
                "ctb_main": (("lat", "lon"), self.ctb_main_map * self.mask),
                "ctb_clim": (("lat", "lon"), self.ctb_clim_map * self.mask),
            },
            coords={
                "lon": self.sensitivity_ds[self.target_name].lon.values,
                "lat": self.sensitivity_ds[self.target_name].lat.values,
            },
        )
        # cal global rate by each variables
        self.glob_rate_est_clim = recal_ctb_rate(self.sensitivity_ds, full_pred_fields)
        self.glob_rate_est_main = recal_ctb_rate(self.sensitivity_ds, main_pred_fields)
        self.glob_rate_est_all = recal_ctb_rate(self.sensitivity_ds, all_pred_fields)

        # plot max_impact - map
        # self.max_impact["max_impact"].plot(
        #     levels=5,
        #     cmap=matplotlib.colors.ListedColormap(
        #         matplotlib.colormaps["Accent"].colors[: (len(self.predictors))]
        #     ),
        # )

    def plt_impact_rate(self):
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        axbox = ax.get_position()
        pred_fields = [f"{p}_fixed" for p in self.predictors]
        colors_list = ["#7fc97f", "#fb8072", "#fdc086", "#386cb0", "#beaed4"]
        colors_dict = {
            m_name: c for m_name, c in zip(pred_fields, colors_list[: len(pred_fields)])
        }
        for f in pred_fields:
            obj = self.impact_rate[f]
            x, y = obj.index, obj.values
            ax.plot(
                x,
                y,
                label=f,
                linewidth=1.75,
                marker="o",
                ms=4,
                color=colors_dict[f],
                markerfacecolor="white",
                markeredgecolor=colors_dict[f],
            )
        ax.set_xlabel("Year")
        ax.set_ylabel(VIZ_OPT[self.target_name]["line_bar_unit"])
        # plt.ylim([250, 650])
        ax.set_title(
            f"{self.model_name} - Drivers of Annual Trend of {self.target_name}"
        )
        ax.legend(
            loc="center",
            ncol=len(pred_fields),
            bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.25],
        )

    def plt_max_impact_map(self, mode="main"):
        fig = plt.figure(figsize=(12, 9))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(
            xlocs=range(-180, 180, 40),
            ylocs=range(-80, 81, 20),
            draw_labels=True,
            linewidth=1,
            edgecolor="dimgrey",
        )
        if mode == "main":
            predictors = self.main_predictors

            cmap = matplotlib.colors.ListedColormap(["#beaed4", "#7fc97f", "#f0027f"])
            data = self.impact_map["ctb_main"]
        else:
            predictors = self.clim_predictors
            cmap = matplotlib.colors.ListedColormap(["#fb8072", "#fdc086", "#386cb0"])
            data = self.impact_map["ctb_clim"]
        center = [0.5 * (i * 2 + 1) for i in range(len(predictors))]
        cax = data.plot(
            cmap=cmap,
            vmin=0,
            vmax=len(predictors),
            ax=ax,
            add_colorbar=False,
        )
        cbar = fig.colorbar(
            cax,
            ticks=center,
            orientation="horizontal",
            pad=0.05,
        )
        cbar.ax.set_xticklabels(predictors, size=18)
        title = f"{self.model_name} - Drivers of changes in {self.target_name}"
        cbar.set_label(label="Dominant driver", size=18, weight="bold")
        plt.title(title, fontsize=18)

    def plt_contri_map(self, mode="main"):
        if mode == "main":
            pred_fields = [f"{p}_fixed" for p in self.main_predictors]
            ds = self.ctb_main_dict
        else:
            pred_fields = [f"{f}_fixed" for f in self.clim_predictors]
            ds = self.ctb_clim_dict
        i = 0
        for f in pred_fields:
            i = i + 1
            fig = plt.figure(1 + i, figsize=(12, 9))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.gridlines(
                xlocs=range(-180, 180, 40),
                ylocs=range(-80, 81, 20),
                draw_labels=True,
                linewidth=1,
                edgecolor="dimgrey",
            )
            title = f"{self.model_name} - Contribution of {f} to the {self.target_name} trends"
            data = ds[f]
            data.plot.pcolormesh(
                ax=ax,
                cmap="PiYG_r",
                levels=21,
                vmin=-0.005,
                vmax=0.005,
                extend="both",
                cbar_kwargs={
                    "label": VIZ_OPT[self.target_name]["map_unit"],
                    "orientation": "horizontal",
                    "pad": 0.05,
                },
            )
            plt.title(title, fontsize=18)


# %%
model_name = "GFDL-ESM4"
a = RegSingleModel(model_name, start_year=1850, end_year=2014)

# %%
pred_fields = [f"{p}_fixed" for p in a.main_predictors]
ds = a.ctb_main_dict
i = 0
for f in pred_fields:
    i = i + 1
    fig = plt.figure(1 + i, figsize=(12, 9))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(
        xlocs=range(-180, 180, 40),
        ylocs=range(-80, 81, 20),
        draw_labels=True,
        linewidth=1,
        edgecolor="dimgrey",
    )
    title = f"{a.model_name} - Contribution of {f} to the {a.target_name} trends"
    data = ds[f]
    data.plot.pcolormesh(
        ax=ax,
        cmap="PiYG_r",
        levels=11,
        vmin=-0.01,
        vmax=0.01,
        extend="both",
        cbar_kwargs={
            "label": VIZ_OPT[a.target_name]["map_unit"],
            "orientation": "horizontal",
            "pad": 0.05,
        },
    )
    plt.title(title, fontsize=18)

# %%
fig = plt.figure(figsize=(12, 9))
ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines(
    xlocs=range(-180, 180, 40),
    ylocs=range(-80, 81, 20),
    draw_labels=True,
    linewidth=1,
    edgecolor="dimgrey",
)
predictors = a.main_predictors

cmap = matplotlib.colors.ListedColormap(["#7fc97f", "#f0027f"])
data = a.impact_map["ctb_main"]
center = [0.5 * (i * 2 + 1) for i in range(len(predictors))]
cax = data.plot(
    cmap=cmap,
    vmin=0,
    vmax=len(predictors),
    ax=ax,
    add_colorbar=False,
)
cbar = fig.colorbar(
    cax,
    ticks=center,
    orientation="horizontal",
    pad=0.05,
)
cbar.ax.set_xticklabels(predictors, size=18)
title = f"{a.model_name} - Drivers of changes in {a.target_name}"
cbar.set_label(label="Dominant driver", size=18, weight="bold")
plt.title(title, fontsize=18)

# %%
glob_rate_ctb_main = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.main_predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb_main[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb_main["year"] = a.sensitivity_ds.year.values
glob_rate_ctb_main.set_index("year")
glob_rate_ctb_main.plot(x="year", cmap="Accent")

# %%

glob_rate_ctb = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb["year"] = a.sensitivity_ds.year.values
glob_rate_ctb.set_index("year")
glob_rate_ctb.plot(x="year")

# %%
a.plt_max_impact_map(mode="clim")

# %%
pred_fields = [f"{f}_fixed" for f in a.clim_predictors]
ds = a.ctb_clim_dict
i = 0
for f in pred_fields:
    i = i + 1
    fig = plt.figure(1 + i, figsize=(12, 9))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(
        xlocs=range(-180, 180, 40),
        ylocs=range(-80, 81, 20),
        draw_labels=True,
        linewidth=1,
        edgecolor="dimgrey",
    )
    title = f"{a.model_name} - Contribution of {f} to the {a.target_name} trends"
    data = ds[f]
    data.plot.pcolormesh(
        ax=ax,
        cmap="PiYG_r",
        levels=11,
        vmin=-0.003,
        vmax=0.003,
        extend="both",
        cbar_kwargs={
            "label": VIZ_OPT[a.target_name]["map_unit"],
            "orientation": "horizontal",
            "pad": 0.05,
        },
    )
    plt.title(title, fontsize=18)

# %%
pred_fields = [f"{f}_fixed" for f in a.clim_predictors]
ds = a.ctb_clim_dict
i = 0
for f in pred_fields:
    i = i + 1
    fig = plt.figure(1 + i, figsize=(12, 9))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(
        xlocs=range(-180, 180, 40),
        ylocs=range(-80, 81, 20),
        draw_labels=True,
        linewidth=1,
        edgecolor="dimgrey",
    )
    title = f"{a.model_name} - Contribution of {f} to the {a.target_name} trends"
    data = ds[f]
    data.plot.pcolormesh(
        ax=ax,
        cmap="PiYG_r",
        levels=11,
        vmin=-0.0025,
        vmax=0.0025,
        extend="both",
        cbar_kwargs={
            "label": VIZ_OPT[a.target_name]["map_unit"],
            "orientation": "horizontal",
            "pad": 0.05,
        },
    )
    plt.title(title, fontsize=18)

# %%
model_name = "GISS-E2-1-G"
a = RegSingleModel(model_name, start_year=1850, end_year=2014)

# %%
glob_rate_ctb_main = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.main_predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb_main[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb_main["year"] = a.sensitivity_ds.year.values
glob_rate_ctb_main.set_index("year")
glob_rate_ctb_main.plot(x="year", cmap="Accent")

# %%
glob_rate_ctb = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb["year"] = a.sensitivity_ds.year.values
glob_rate_ctb.set_index("year")
glob_rate_ctb.plot(x="year")

# %%
a.glob_rate_est_main.plot()

# %%
fig = plt.figure(figsize=(12, 9))
ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines(
    xlocs=range(-180, 180, 40),
    ylocs=range(-80, 81, 20),
    draw_labels=True,
    linewidth=1,
    edgecolor="dimgrey",
)
predictors = a.main_predictors

cmap = matplotlib.colors.ListedColormap(["#7fc97f", "#f0027f"])
data = a.impact_map["ctb_main"]
center = [0.5 * (i * 2 + 1) for i in range(len(predictors))]
cax = data.plot(
    cmap=cmap,
    vmin=0,
    vmax=len(predictors),
    ax=ax,
    add_colorbar=False,
)
cbar = fig.colorbar(
    cax,
    ticks=center,
    orientation="horizontal",
    pad=0.05,
)
cbar.ax.set_xticklabels(predictors, size=18)
title = f"{a.model_name} - Drivers of changes in {a.target_name}"
cbar.set_label(label="Dominant driver", size=18, weight="bold")
plt.title(title, fontsize=18)

# %%
a.plt_max_impact_map(mode="clim")

# %%
pred_fields = [f"{f}_fixed" for f in a.clim_predictors]
ds = a.ctb_clim_dict
i = 0
for f in pred_fields:
    i = i + 1
    fig = plt.figure(1 + i, figsize=(12, 9))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(
        xlocs=range(-180, 180, 40),
        ylocs=range(-80, 81, 20),
        draw_labels=True,
        linewidth=1,
        edgecolor="dimgrey",
    )
    title = f"{a.model_name} - Contribution of {f} to the {a.target_name} trends"
    data = ds[f]
    data.plot.pcolormesh(
        ax=ax,
        cmap="PiYG_r",
        levels=11,
        vmin=-0.0025,
        vmax=0.0025,
        extend="both",
        cbar_kwargs={
            "label": VIZ_OPT[a.target_name]["map_unit"],
            "orientation": "horizontal",
            "pad": 0.05,
        },
    )
    plt.title(title, fontsize=18)

# %%
pred_fields = [f"{f}_fixed" for f in a.main_predictors]
ds = a.ctb_main_dict
i = 0
for f in pred_fields:
    i = i + 1
    fig = plt.figure(1 + i, figsize=(12, 9))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(
        xlocs=range(-180, 180, 40),
        ylocs=range(-80, 81, 20),
        draw_labels=True,
        linewidth=1,
        edgecolor="dimgrey",
    )
    title = f"{a.model_name} - Contribution of {f} to the {a.target_name} trends"
    data = ds[f]
    data.plot.pcolormesh(
        ax=ax,
        cmap="PiYG_r",
        levels=11,
        vmin=-0.01,
        vmax=0.01,
        extend="both",
        cbar_kwargs={
            "label": VIZ_OPT[a.target_name]["map_unit"],
            "orientation": "horizontal",
            "pad": 0.05,
        },
    )
    plt.title(title, fontsize=18)

# %%
model_name = "UKESM1-0-LL"
a = RegSingleModel(model_name, start_year=1850, end_year=2014)

# %%

pred_fields = [f"{p}_fixed" for p in a.main_predictors]
ds = a.ctb_main_dict
i = 0
for f in pred_fields:
    i = i + 1
    fig = plt.figure(1 + i, figsize=(12, 9))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(
        xlocs=range(-180, 180, 40),
        ylocs=range(-80, 81, 20),
        draw_labels=True,
        linewidth=1,
        edgecolor="dimgrey",
    )
    title = f"{a.model_name} - Contribution of {f} to the {a.target_name} trends"
    data = ds[f]
    data.plot.pcolormesh(
        ax=ax,
        cmap="PiYG_r",
        levels=11,
        vmin=-0.025,
        vmax=0.025,
        extend="both",
        cbar_kwargs={
            "label": VIZ_OPT[a.target_name]["map_unit"],
            "orientation": "horizontal",
            "pad": 0.05,
        },
    )
    plt.title(title, fontsize=18)

# %%

pred_fields = [f"{f}_fixed" for f in a.clim_predictors]
ds = a.ctb_clim_dict
i = 0
for f in pred_fields:
    i = i + 1
    fig = plt.figure(1 + i, figsize=(12, 9))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(
        xlocs=range(-180, 180, 40),
        ylocs=range(-80, 81, 20),
        draw_labels=True,
        linewidth=1,
        edgecolor="dimgrey",
    )
    title = f"{a.model_name} - Contribution of {f} to the {a.target_name} trends"
    data = ds[f]
    data.plot.pcolormesh(
        ax=ax,
        cmap="PiYG_r",
        levels=11,
        vmin=-0.025,
        vmax=0.025,
        extend="both",
        cbar_kwargs={
            "label": VIZ_OPT[a.target_name]["map_unit"],
            "orientation": "horizontal",
            "pad": 0.05,
        },
    )
    plt.title(title, fontsize=18)

# %%

pred_fields = [f"{f}_fixed" for f in a.clim_predictors]
ds = a.ctb_clim_dict
i = 0
for f in pred_fields:
    i = i + 1
    fig = plt.figure(1 + i, figsize=(12, 9))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(
        xlocs=range(-180, 180, 40),
        ylocs=range(-80, 81, 20),
        draw_labels=True,
        linewidth=1,
        edgecolor="dimgrey",
    )
    title = f"{a.model_name} - Contribution of {f} to the {a.target_name} trends"
    data = ds[f]
    data.plot.pcolormesh(
        ax=ax,
        cmap="PiYG_r",
        levels=11,
        vmin=-0.01,
        vmax=0.01,
        extend="both",
        cbar_kwargs={
            "label": VIZ_OPT[a.target_name]["map_unit"],
            "orientation": "horizontal",
            "pad": 0.05,
        },
    )
    plt.title(title, fontsize=18)

# %%
a.plt_max_impact_map()

# %%
a.plt_max_impact_map(mode="clim")

# %%
model_name = "CESM2-WACCM"
a = RegSingleModel(model_name, start_year=1850, end_year=2014)

# %%
a.glob_rate_est_main.plot(cmap="Accent", ylim=[-30, 70])

# %%
a.glob_rate_est_main.plot(cmap="Accent", ylim=[-30, 70])

# %%
glob_rate_ctb_main = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.main_predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb_main[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb_main["year"] = a.sensitivity_ds.year.values
glob_rate_ctb_main.set_index("year")
glob_rate_ctb_main.plot(x="year", cmap="Accent", ylim=[-30, 70])


# %%
glob_rate_ctb = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb["year"] = a.sensitivity_ds.year.values
glob_rate_ctb.set_index("year")
glob_rate_ctb.plot(x="year", ylim=[-30, 70], cmap="Accent")

# %%

pred_fields = [f"{p}_fixed" for p in a.main_predictors]
ds = a.ctb_main_dict
i = 0
for f in pred_fields:
    i = i + 1
    fig = plt.figure(1 + i, figsize=(12, 9))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(
        xlocs=range(-180, 180, 40),
        ylocs=range(-80, 81, 20),
        draw_labels=True,
        linewidth=1,
        edgecolor="dimgrey",
    )
    title = f"{a.model_name} - Contribution of {f} to the {a.target_name} trends"
    data = ds[f]
    data.plot.pcolormesh(
        ax=ax,
        cmap="PiYG_r",
        levels=11,
        vmin=-0.025,
        vmax=0.025,
        extend="both",
        cbar_kwargs={
            "label": VIZ_OPT[a.target_name]["map_unit"],
            "orientation": "horizontal",
            "pad": 0.05,
        },
    )
    plt.title(title, fontsize=18)

# %%

pred_fields = [f"{f}_fixed" for f in a.clim_predictors]
ds = a.ctb_clim_dict
i = 0
for f in pred_fields:
    i = i + 1
    fig = plt.figure(1 + i, figsize=(12, 9))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(
        xlocs=range(-180, 180, 40),
        ylocs=range(-80, 81, 20),
        draw_labels=True,
        linewidth=1,
        edgecolor="dimgrey",
    )
    title = f"{a.model_name} - Contribution of {f} to the {a.target_name} trends"
    data = ds[f]
    data.plot.pcolormesh(
        ax=ax,
        cmap="PiYG_r",
        levels=11,
        vmin=-0.01,
        vmax=0.01,
        extend="both",
        cbar_kwargs={
            "label": VIZ_OPT[a.target_name]["map_unit"],
            "orientation": "horizontal",
            "pad": 0.05,
        },
    )
    plt.title(title, fontsize=18)

# %%
model_name = "NorESM2-LM"
a = RegSingleModel(model_name, start_year=1850, end_year=2014)

# %%
glob_rate_ctb = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb["year"] = a.sensitivity_ds.year.values
glob_rate_ctb.set_index("year")
glob_rate_ctb.plot(x="year", ylim=[-30, 70], cmap="Accent")

# %%
glob_rate_ctb_main = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.main_predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb_main[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb_main["year"] = a.sensitivity_ds.year.values
glob_rate_ctb_main.set_index("year")
glob_rate_ctb_main.plot(x="year", cmap="Accent", ylim=[-30, 70])


# %%
model_name = "UKESM1-0-LL"
a = RegSingleModel(model_name, start_year=1850, end_year=2014)

# %%
glob_rate_ctb = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb["year"] = a.sensitivity_ds.year.values
glob_rate_ctb.set_index("year")
glob_rate_ctb.plot(x="year", ylim=[-30, 70], cmap="Accent")

# %%
glob_rate_ctb_main = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.main_predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb_main[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb_main["year"] = a.sensitivity_ds.year.values
glob_rate_ctb_main.set_index("year")
glob_rate_ctb_main.plot(x="year", cmap="Accent", ylim=[-30, 70])

# %%
model_name = "GISS-E2-1-G"
a = RegSingleModel(model_name, start_year=1850, end_year=2014)

# %%
glob_rate_ctb_main = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.main_predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb_main[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb_main["year"] = a.sensitivity_ds.year.values
glob_rate_ctb_main.set_index("year")
glob_rate_ctb_main.plot(
    x="year",
    cmap=matplotlib.colors.ListedColormap(["#386cb0", "#666666"]),
    ylim=[-30, 70],
)

# %%
glob_rate_ctb = pd.DataFrame()
fields = [f"{p}_fixed" for p in a.predictors]
for f in fields:
    ds_cp = a.sensitivity_ds.copy()
    glob_rate_ctb[f], _ = cal_actual_rate(
        ds_cp["reg"] - ds_cp[f], model_name, mode="ts"
    )

glob_rate_ctb["year"] = a.sensitivity_ds.year.values
glob_rate_ctb.set_index("year")
glob_rate_ctb.plot(
    x="year",
    ylim=[-60, 70],
    cmap=matplotlib.colors.ListedColormap(["#7fc97f", "#fdc086", "#386cb0", "#666666"]),
)

# %%
a.glob_rate_est_clim.plot()
