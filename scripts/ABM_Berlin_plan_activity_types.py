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
# network_path = 'berlin_data_matsim/test_network.xml'
# events_path = 'berlin_data_matsim/test_events.xml'
# plans_path = 'berlin_data_matsim/test_plans.xml'



berlin_home_locations = []
berlin_work_locations = []
processed_all_agents = set()
n_people = 1000  # scan first 1000 persons; adjust as you like

# plans_generator = matsim.plan_reader(base_folder + plans_path, selected_plans_only = True)
plans_generator = matsim.plan_reader(base_folder+plans_path,selected_plans_only=False)  # set False to see all plans

activity_types = set()
# activities = set()

# for person, plan in plans_generator:
#     for item in plan:
#         45
#         if item.tag == 'activity':
#             activity_type = item.attrib.get('type')
#             if activity_type:
#                 activity_types.add(activity_type)

# print("Unique activity types found in the plans file:")
# for activity_type in sorted(activity_types):
#     print(f" - {activity_type}")

for i, (person, plan) in enumerate(plans_generator, 1):
    for item in plan:
        if getattr(item, "tag", None) == "activity":
            t = item.attrib.get("type")
            if t:
                activity_types.add(t)
    if i >= n_people:
        break

print("Unique activity types (sample):")
for t in sorted(activity_types):
    print(" -", t)
print("Total types in sample:", len(activity_types))