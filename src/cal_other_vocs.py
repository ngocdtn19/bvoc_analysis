# %%
from CMIP6Model import Var


def cal_other_bvocs():
    isop = Var("emiisop")
    bvoc = Var("emibvoc")

    model_names = list(isop.multi_models.keys())

    for name in model_names:
        print(name)
        var_name = "emiotherbvocs"
        reindex_ds_bvoc = (
            bvoc.multi_models[name]
            .org_ds_var["emibvoc"]
            .reindex_like(
                isop.multi_models[name].org_ds_var["emiisop"],
                method="nearest",
                tolerance=0.01,
            )
        )
        other_bvocs = reindex_ds_bvoc - isop.multi_models[name].org_ds_var["emiisop"]
        other_bvocs = other_bvocs.rename(var_name)
        other_bvocs.to_netcdf(
            f"../data/var/{var_name}/{var_name}_AERmon_{name}_historical_r1i1p1f1_gn_185001-201412.nc"
        )

def cal_emipoa():
    oa = Var("emioa")
    soa = Var("chepsoa")

    model_names = list(oa.multi_models.keys())

    for name in model_names:
        print(name)
        var_name = "emipoa"
        reindex_ds_oa = (
            oa.multi_models[name]
            .org_ds_var["emioa"]
            .reindex_like(
                soa.multi_models[name].org_ds_var["chepsoa"],
                method="nearest",
                tolerance=0.01,
            )
        )
        emipoa = reindex_ds_oa - soa.multi_models[name].org_ds_var["chepsoa"]
        emipoa = emipoa.rename(var_name)
        emipoa.to_netcdf(
            f"../data/var/{var_name}/{var_name}_AERmon_{name}_historical_r1i1p1f1_gn_185001-201412.nc"
        )

# %%
