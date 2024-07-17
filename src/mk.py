import pymannkendall as pymk
import xarray as xr
import numpy as np


def k_cor(x, y, pthres=0.05, direction=True):
    """Uses the pymannkendall module to calculate a Kendall correlation test
    Param:
        x vector: Input pixel vector to run tests on
        y vector: The date input vector
        pthres: Significance of the underlying test
    Return:
        xarray.Dataset: slope
    """

    # Check NA values
    co = np.count_nonzero(~np.isnan(x))
    if co < 4:  # If fewer than 4 observations return -9999
        return "nan"
    # Run the kendalltau test
    trend, h, p, z, Tau, s, var_s, slope, intercept = pymk.original_test(x)

    # Criterium to return results in case of Significance
    return slope if p < pthres else "nan"


# The function we are going to use for applying our kendal test per pixel
def kendall_correlation(x, y, dim="year"):
    # x = Pixel value, y = a vector containing the date, dim == dimension
    return xr.apply_ufunc(
        k_cor,
        x,
        y,
        input_core_dims=[[dim], [dim]],
        vectorize=True,  # !Important!
        output_dtypes=[float],
    )
