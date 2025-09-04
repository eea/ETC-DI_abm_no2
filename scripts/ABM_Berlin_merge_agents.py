# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 09:56:01 2025

@author: jetschny
"""
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

# Ensure pyproj is installed
try:
    import pyproj
except ImportError:
    # This might require a restart of the kernel in some environments
    print("pyproj not found, attempting to install...")
    # !pip install pyproj # Uncomment if running in an environment where pip install works directly
    import pyproj
finally:
    # Verify pyproj version if needed for specific functionalities
    # print(f"pyproj version: {pyproj.__version__}")
    pass

# from shapely.geometry import Point
from pyproj import Transformer

import matsim
import pandas as pd
# import sys
import geopandas as gpd
# from geocube.api.core import make_geocube
from shapely.geometry import Point
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image

# from collections import defaultdict

# base_folder = "Z:/Environment and Health/Air Quality/city_data_raw/abm_no2/"
base_folder = "Z:/Environment and Health/Air Quality/abm_no2/"
# network_path = "berlin_data_matsim/berlin-v6.4.output_network.xml.gz"

agent_home_path = "berlin_output/agents_berlin_citylim_home.csv"
agent_work_path = "berlin_output/agents_berlin_citylim_work.csv"


NO2_annual_raster_Berlin = "berlin_data_aq/NO2_2023_epsg3035_clipBerlin.tif"
# berlin_outline = "berlin_data_shapes/Berlin_Bezirke_dissolve.gpkg"

home_df = pd.read_csv(base_folder+agent_home_path)
work_df = pd.read_csv(base_folder+agent_work_path)

# Count unique agent ids before merge
home_count = home_df["agent_id"].nunique()
work_count = work_df["agent_id"].nunique()

# Merge on agent id
merged_df = pd.merge(home_df, work_df, on="agent_id", how="inner")

# Count after merge
merged_count = merged_df["agent_id"].nunique()

# Calculate how many were removed
removed_home = home_count - merged_count
removed_work = work_count - merged_count

print(f"Agents with home location only (removed): {removed_home}")
print(f"Agents with work location only (removed): {removed_work}")
print(f"Agents kept (with both home & work): {merged_count}")