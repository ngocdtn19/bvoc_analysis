import glob
import os
from const import *

"""
Directory structure
{data_dir}/data 
    /original
        /var
        /land
        /axl
    /processed_org_data 
        /annual_per_area_unit
        /land
        /mk_trends_map
    /sensitivity_als
        /CMIP6
            /input_data # resampled data from annual_per_area_unit
            /contribution_mk 
        /VISIT
            /input_data
            /contribution_mk
                /mk_1x1.25
                /mk_0.5x0.5
    /gfdl_esm4_latlon
    /visit_latlon
"""

DATA_SERVER = f"/mnt/dg3/ngoc/cmip6_bvoc_als/data/"
DATA_LOCAL = "../data/"
CMIP6_SENSALS_DIR = f"{DATA_SERVER}/sensitivity_als/CMIP6/"
VISIT_SENSALS_DIR = f"{DATA_SERVER}/sensitivity_als/VISIT/"


DATA_DIR = DATA_LOCAL
if os.path.exists(DATA_SERVER):
    DATA_DIR = DATA_SERVER

VAR_DIR = os.path.join(DATA_DIR, "original/var")
LAND_DIR = os.path.join(DATA_DIR, "original/land")
AXL_DIR = os.path.join(DATA_DIR, "original/axl")

LIST_ATTR = [attr.split("\\")[-1] for attr in glob.glob(os.path.join(VAR_DIR, "*"))]

ISOP_LIST = glob.glob(os.path.join(VAR_DIR, "emiisop", "*.nc"))
BVOC_LIST = glob.glob(os.path.join(VAR_DIR, "emibvoc", "*.nc"))

AREA_LIST = glob.glob(os.path.join(AXL_DIR, VAR_AREA, "*.nc"))
SFLTF_LIST = glob.glob(os.path.join(AXL_DIR, VAR_SFTLF, "*.nc"))
MASK_LIST = glob.glob(os.path.join(AXL_DIR, VAR_MASK, "*.nc"))


VISIT_LAT_FILE = os.path.join(DATA_DIR, "visit_latlon", "visit_lat.npy")
VISIT_LONG_FILE = os.path.join(DATA_DIR, "visit_latlon", "visit_long.npy")
GFDL_LAT_FILE = os.path.join(DATA_DIR, "gfdl_esm4_latlon", "lat.npy")
GFDL_LONG_FILE = os.path.join(DATA_DIR, "gfdl_esm4_latlon", "lon.npy")
