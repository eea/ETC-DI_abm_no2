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
# from pyproj import Transformer

import matsim
# import pandas as pd
# import sys
# import geopandas as gpd
# from geocube.api.core import make_geocube
# from shapely.geometry import Point
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image

# from collections import defaultdict

# base_folder = "Z:/Environment and Health/Air Quality/city_data_raw/abm_no2/"
base_folder = "Z:/Environment and Health/Air Quality/abm_no2/"
# network_path = "berlin_data_matsim/berlin-v6.4.output_network.xml.gz"

network_path = 'berlin_data_matsim/berlin-v6.4.output_network.xml.gz'
events_path = 'berlin_data_matsim/berlin-v6.4.output_events.xml.gz'
plans_path = 'berlin_data_matsim/berlin-v6.4.output_plans.xml.gz'

# NO2_annual_raster_Berlin = "berlin_data_aq/NO2_2023_epsg3035_clipBerlin.tif"
# berlin_outline = "berlin_data_shapes/Berlin_Bezirke_dissolve.gpkg"


berlin_home_locations = []
berlin_work_locations = []
processed_all_agents = set()

nets_generator = matsim.read_network(base_folder + network_path)

activity_types = set()

for person, plan in nets_generator:
    for item in plan:
        if item.tag == 'activity':
            activity_type = item.attrib.get('type')
            if activity_type:
                activity_types.add(activity_type)

print("Unique activity types found in the network file:")
for activity_type in sorted(activity_types):
    print(f" - {activity_type}")
