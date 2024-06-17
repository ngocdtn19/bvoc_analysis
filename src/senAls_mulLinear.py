# %%

# 3rd party modules
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# project modules
from senAls_utils import *

sns.set_style("ticks")
FIG_INDEX = 0


class RegSingleModel:
    def __init__(
        self,
        model_name="VISIT",
    ) -> None:
        self.target_name = "emiisop"
        self.model_name = model_name

        self.org_ds = None

        self.get_predictors()
        self.read_data()
        self.ml_als()
        # self.cal_glob_change_ts()
        self.cal_change_past_hist_con()
        self.cal_glob_past_hist_con()
        self.cal_reg_past_hist_con()

        # self.plt_map_change()
        # self.plt_global_change_con()
        # self.plt_regional_change_con()

    def get_predictors(self):
        base_predictors = ["tas", "rsds", "pr"]

        if self.model_name in ["CESM2-WACCM", "NorESM2-LM", "UKESM1-0-LL"]:
            base_predictors.append("co2s")

        self.predictor_names = base_predictors

    def read_data(self):
        iters = self.predictor_names + [self.target_name]
        dss = {}
        for p in iters:
            f = os.path.join(RES_DIR, f"{self.model_name}_{p}.nc")
            ds = xr.open_dataset(f)[p]
            # ds = ds.sel(year=slice(1901, 2014)) if "VISIT" in self.model_name else ds
            dss[p] = (("lat", "lon", "year"), ds.transpose("lat", "lon", "year").data)
        lat = ds.lat.values
        lon = ds.lon.values
        year = ds.year.values
        self.org_ds = xr.Dataset(dss, coords={"lat": lat, "lon": lon, "year": year})

    def regression(self, data):
        preds_dict = {}
        for pn in self.predictor_names:
            preds_dict[pn] = data[pn].values

        X = np.column_stack((preds_dict[pn] for pn in self.predictor_names))

        y = data[self.target_name]
        X = np.nan_to_num(X)
        y = np.nan_to_num(y).reshape(-1)
        est = LinearRegression()
        est.fit(X, y)

        return xr.DataArray(np.append(est.coef_, est.intercept_))

    def ml_als(self):
        self.preds = {}

        stacked = self.org_ds.stack(allpoints=["lat", "lon"])
        coefs = stacked.groupby("allpoints").map(self.regression)
        self.coefs = coefs.unstack("allpoints")

        # cal predictors
        for i, pn in enumerate(self.predictor_names):
            p_val = self.coefs.isel(dim_0=i).values
            self.preds[pn] = self.org_ds[pn].values * p_val.reshape(p_val.shape + (1,))

        # cal intercept
        intercept = self.coefs.isel(dim_0=-1).values
        self.preds["intercept"] = intercept.reshape(intercept.shape + (1,)) * np.ones(
            self.org_ds["emiisop"].values.shape
        )

        ts_pred = 0
        for pn in self.preds.keys():
            ts_pred += self.preds[pn]

        self.ts_pred = xr.Dataset(
            {"ts_pred": (("lat", "lon", "year"), ts_pred)},
            coords={
                "lon": self.org_ds.lon.values,
                "lat": self.org_ds.lat.values,
                "year": self.org_ds.year.values,
            },
        )

    def cal_glob_change_ts(self):
        self.pred_glob_rate, self.pred_glob = cal_actual_rate(
            self.ts_pred["ts_pred"], self.model_name, mode="ts"
        )
        self.truth_glob_rate, self.truth_glob = cal_actual_rate(
            self.org_ds[self.target_name], self.model_name, mode="ts"
        )
        r, p = pearsonr(self.truth_glob_rate.values, self.pred_glob_rate.values)
        mse = mean_squared_error(
            self.truth_glob_rate.values, self.pred_glob_rate.values
        )
        rmse = math.sqrt(mse)

        df = pd.DataFrame(
            {
                "regression": self.pred_glob_rate.values,
                "model": self.truth_glob_rate.values,
            },
            index=[i for i in range(1850, 2015)],
        )
        # lines = df.plot.line()
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        sns.lineplot(
            df,
            ax=ax,
            palette=["#f0027f", "#bf5b17"],
            lw=1.75,
            marker="o",
            ms=5,
            markeredgecolor="w",
        )
        ax.set_xlabel("Year", fontweight="bold", fontsize=14)
        ax.set_ylabel("[TgC]", fontweight="bold", fontsize=14)
        ax.set_title(f"Annual Global Trend of {self.target_name} - {self.model_name}")
        fig.text(
            0.88,
            0.95,
            f"r = {np.round(r, decimals=3)}\nrmse = {np.round(rmse, decimals=1)}",
            fontsize=12,
        )

    def cal_change_past_hist_con(self):
        self.past_hist = {}
        for pn in self.predictor_names:
            self.past_hist[pn] = self.preds[pn][:, :, -25:-1].mean(axis=2) - self.preds[
                pn
            ][:, :, 0:25].mean(axis=2)

        est_stacked = np.stack(
            (self.past_hist[pn] for pn in self.predictor_names),
            axis=-1,
        )
        abslt_est = np.absolute(est_stacked)
        max_contrb = np.argmax(abslt_est, axis=2)

        total_con = 0
        for pn in self.predictor_names:
            total_con += self.past_hist[pn]

        self.past_hist["total_con"] = total_con

        abs_con = xr.Dataset(
            {"abs_con": (("lat", "lon"), max_contrb)},
            coords={"lon": self.org_ds.lon.values, "lat": self.org_ds.lat.values},
        )
        real_con = xr.Dataset(
            {pn: (("lat", "lon"), self.past_hist[pn]) for pn in self.past_hist.keys()},
            coords={"lon": self.org_ds.lon.values, "lat": self.org_ds.lat.values},
        )

        self.proj_abs_con = abs_con
        self.proj_real_con = real_con

    def cal_glob_past_hist_con(self):
        target_change = self.org_ds[self.target_name].isel(year=slice(-25, -1)).mean(
            "year"
        ) - self.org_ds[self.target_name].isel(year=slice(0, 25)).mean("year")
        target_rate, self.target_change = cal_actual_rate(
            target_change, self.model_name
        )

        df = pd.DataFrame()
        des = []
        rates = []
        prj = 0

        self.pred_change = {}
        for p in self.predictor_names:
            des.append(p)

            rate, self.pred_change[p] = cal_actual_rate(
                self.proj_real_con[p], self.model_name
            )
            prj += rate
            rates.append(rate)

        des.append("regression")
        des.append("model")
        rates.append(prj)
        rates.append(target_rate)
        df["des"] = des
        df["rates"] = rates

        self.global_change_con = df

    def cal_reg_past_hist_con(self):
        df = pd.DataFrame()
        rates = []
        des = []
        scale = []
        for roi in LIST_REGION:
            regression = 0
            target_change = prep_to_clip_reg(self.target_change)
            target_reg_rate = (
                clip_region_mask(target_change, roi).sum(dim=["lat", "lon"]).item()
            )
            for p in self.predictor_names:
                des.append(p)

                pred_change = self.pred_change[p]
                pred_change = prep_to_clip_reg(pred_change)

                rate = clip_region_mask(pred_change, roi).sum(dim=["lat", "lon"]).item()
                rates.append(rate)

                regression += rate

                scale.append(roi)

            des.append("regression")
            rates.append(regression)
            scale.append(roi)

            des.append("model")
            rates.append(target_reg_rate)
            scale.append(roi)

        df["scale"] = scale
        df["des"] = des
        df["rates"] = rates
        self.regional_change_con = df

    def plt_map_change(self):
        global FIG_INDEX
        # rs, cs = 1, 2
        # w, h = 15 * cs, 12 * rs
        # fig, ax = plt.subplots(rs, cs, figsize=(w, h))

        # # self.proj_abs_con["abs_con"].plot(ax=ax[0], levels=5, vmin=0, vmax=4)
        # # self.proj_real_con["total_con"].plot(ax=ax[1], extend="both", cmap="coolwarm")
        # self.proj_abs_con["abs_con"].plot(ax=ax[0])

        # self.proj_real_con["total_con"].plot(
        #     ax=ax[1],
        #     extend="both",
        #     cmap="coolwarm",
        # )
        # ax[0].set_title(f"{self.model_name} - Drivers of change in {self.target_name}")
        # ax[1].set_title(
        #     f"{self.model_name} - Changes in {self.target_name} between PI and PD"
        # )
        mask = self.target_change.values
        mask[mask != 0] = 1
        mask[mask == 0] = np.nan

        mode = ["drivers", "total_change"]
        for m in mode:
            FIG_INDEX += 1
            fig = plt.figure(1 + FIG_INDEX, figsize=(12, 9))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            # regionmask.defined_regions.srex.plot(
            #     ax=ax,
            #     projection=ccrs.PlateCarree(),
            #     regions=["SEA", "EAS", "AMZ", "SSA"],
            #     add_label=True,
            #     label="abbrev",
            # )
            ax.coastlines()
            ax.gridlines(
                xlocs=range(-180, 180, 40),
                ylocs=range(-80, 81, 20),
                draw_labels=True,
                linewidth=1,
                edgecolor="dimgrey",
            )
            if m == "drivers":
                data = self.proj_abs_con["abs_con"] * mask
                wrap_data, wrap_lon = wrap_long(data)

                title = f"{self.model_name} - Drivers of changes in {self.target_name}"
                center = [0.5 * (i * 2 + 1) for i in range(len(self.predictor_names))]
                # center = [0.5, 1.5, 2.5, 3.5, 4.5]
                # center = [0.5, 1.5, 2.5]

                cax = ax.pcolormesh(
                    wrap_lon,
                    data.lat,
                    wrap_data,
                    cmap=matplotlib.colors.ListedColormap(
                        matplotlib.colormaps["Accent"].colors[
                            : (len(self.predictor_names))
                        ]
                    ),
                    vmin=0,
                    vmax=len(self.predictor_names),
                )

                cbar = fig.colorbar(
                    cax,
                    ticks=center,
                    orientation="horizontal",
                    pad=0.05,
                )
                cbar.ax.set_xticklabels(self.predictor_names, size=18)
                # cbar.ax.set_xticklabels(["tas", "rsds", "pr"], size=18)
                cbar.set_label(label="Dominant driver", size=18, weight="bold")

            if m == "total_change":
                data = self.proj_real_con["total_con"] * mask
                wrap_data, wrap_lon = wrap_long(data)

                title = f"{self.model_name} - Changes in {self.target_name} between PI and PD"
                cax = ax.pcolormesh(
                    wrap_lon,
                    data.lat,
                    wrap_data,
                    cmap="coolwarm",
                    # levels=11,
                    vmin=-12,
                    vmax=12,
                )
                cbar = fig.colorbar(
                    cax,
                    extend="both",
                    orientation="horizontal",
                    pad=0.05,
                )
                cbar.set_label(label="[$gC/m^{2}/year$]", size=18, weight="bold")
            plt.title(title, fontsize=18, weight="bold")

    def plt_global_change_con(self):
        rs, cs = 1, 1
        w, h = 6 * cs, 6 * rs
        fig, axes = plt.subplots(rs, cs, figsize=(w, h))
        sns.barplot(
            self.global_change_con,
            x="des",
            y="rates",
            ax=axes,
            palette=sns.color_palette("Accent", len(self.global_change_con)),
            # palette=["#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#f0027f", "#bf5b17"],
        ).set(title=f"{self.model_name}")
        axes.set_xlabel(" ")
        axes.set_ylabel("Isoprene emission [TgC]")
        axes.set_ylim([-120, 75])

    def plt_regional_change_con(self):
        sns.barplot(self.regional_change_con, x="scale", y="rates", hue="des")


class MultiModelReg:
    def __init__(self) -> None:
        self.model_names = [
            "CESM2-WACCM",
            "GFDL-ESM4",
            "GISS-E2-1-G",
            "NorESM2-LM",
            "UKESM1-0-LL",
        ]
        self.reg_ds = []
        self.glb_ds = []

        self.extr_data()

        self.plt_reg_change_con()

    def extr_data(self):
        global FIG_INDEX
        for mn in self.model_names:
            print(mn)
            reg_single_model = RegSingleModel(model_name=mn)
            FIG_INDEX += 1
            reg_df = reg_single_model.regional_change_con
            reg_df["model"] = [mn] * len(reg_df)

            self.reg_ds.append(reg_df)

        self.reg_ds = pd.concat(self.reg_ds, ignore_index=True)

    def plt_reg_change_con(self):
        rs, cs = 2, 2
        w, h = 10 * cs, 10 * rs
        fig, axes = plt.subplots(rs, cs, figsize=(w, h))
        for i, roi in enumerate(LIST_REGION):
            df = self.reg_ds[self.reg_ds["scale"] == roi]
            ax = axes[i // 2, i % 2]
            sns.barplot(
                df,
                x="model",
                y="rates",
                hue="des",
                ax=ax,
                # palette=sns.color_palette("Accent", 7),
                palette=[
                    "#beaed4",
                    "#fdc086",
                    "#ffff99",
                    "#386cb0",
                    "#f0027f",
                    "#bf5b17",
                ],
            )
            ax.set_title(roi)
            ax.set_ylabel("Isoprene emission [TgC]")


# %%
