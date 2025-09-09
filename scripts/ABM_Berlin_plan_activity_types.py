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

plans_generator = matsim.plan_reader(base_folder + plans_path, selected_plans_only = True)

activity_types = set()
# activities = set()

for person, plan in plans_generator:
    for item in plan:
        45
        if item.tag == 'activity':
            activity_type = item.attrib.get('type')
            if activity_type:
                activity_types.add(activity_type)

print("Unique activity types found in the plans file:")
for activity_type in sorted(activity_types):
    print(f" - {activity_type}")


# plans = matsim.plan_reader('output_plans.xml.gz', selected_plans_only = True)

# Each plan is returned as a tuple with its owning person (for now, is this ok?)
# - The name of the element is in its .tag (e.g. 'plan', 'leg', 'route', 'attributes')
# - An element's attributes are accessed using .attrib['attrib-name']
# - Use the element's .text field to get data outside of attributes (e.g. a route's list of links)
# - Every element can be iterated on to get its children (e.g. the plan's activities and legs)
# - Emits person even if that person has no plans

# for person, plan in plans_generator:

#     # do stuff with this plan, e.g.
#     work_activities = filter(
#         lambda e: e.tag == 'activity' and e.attrib['type'] == 'w',
#         plan)

#     print('person', person.attrib['id'], 'selected plan w/', len(list(work_activities)), 'work-act')
#     # activities.append(num_activities)

# person 1 selected plan w/ 2 work-act
# person 10 selected plan w/ 1 work-act
# person 100 selected plan w/ 1 work-act
# ...