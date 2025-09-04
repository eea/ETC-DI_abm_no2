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

import matsim
import pandas as pd
# import sys
# import geopandas as gpd
# from geocube.api.core import make_geocube

import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image

from collections import defaultdict

# base_folder = "Z:/Environment and Health/Air Quality/city_data_raw/abm_no2/"
base_folder = "Z:/Environment and Health/Air Quality/abm_no2/"
network_path = "berlin_data_matsim/berlin-v6.4.output_network.xml.gz"

network_path = 'berlin_data_matsim/berlin-v6.4.output_network.xml.gz'
events_path = 'berlin_data_matsim/berlin-v6.4.output_events.xml.gz'
plans_path = 'berlin_data_matsim/berlin-v6.4.output_plans.xml.gz'

NO2_annual_raster_Berlin = "berlin_data_aq/NO2_2023_epsg3035_clipBerlin.tif"
berlin_outline = "berlin_data_shapes/Berlin_Bezirke_dissolve.gpkg"


# -------------------------------------------------------------------
# 1. NETWORK: Read a MATSim network:
net = matsim.read_network(base_folder+network_path)

net.nodes
# Dataframe output:
#           x        y node_id
# 0  -20000.0      0.0       1
# 1  -15000.0      0.0       2
# 2    -865.0   5925.0       3
# ...

net.links
# Dataframe output:
#      length  capacity  freespeed  ...  link_id from_node to_node
# 0   10000.0   36000.0      27.78  ...        1         1       2
# 1   10000.0    3600.0      27.78  ...        2         2       3
# 2   10000.0    3600.0      27.78  ...        3         2       4
# ...

# Extra: create a Geopandas dataframe with LINESTRINGS for our network
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

geo = net.as_geo()
geo.plot()    

events = matsim.event_reader(base_folder+events_path, types='entered link,left link')

# defaultdict creates a blank dict entry on first reference; similar to {} but more friendly
link_counts = defaultdict(int)

for event in events:
    if event['type'] == 'entered link':
        link_counts[event['link']] += 1

# convert our link_counts dict to a pandas dataframe,
# with 'link_id' column as the index and 'count' column with value:
link_counts = pd.DataFrame.from_dict(link_counts, orient='index', columns=['count']).rename_axis('link_id')

# attach counts to our Geopandas network from above
volumes = geo.merge(link_counts, on='link_id')
volumes.plot(column='count', figsize=(10,10), cmap='Wistia') #cmap is colormap