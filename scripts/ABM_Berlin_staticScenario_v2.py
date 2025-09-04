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

network_path = 'berlin_data_matsim/berlin-v6.4.output_network.xml.gz'
events_path = 'berlin_data_matsim/berlin-v6.4.output_events.xml.gz'
plans_path = 'berlin_data_matsim/berlin-v6.4.output_plans.xml.gz'

NO2_annual_raster_Berlin = "berlin_data_aq/NO2_2023_epsg3035_clipBerlin.tif"
berlin_outline = "berlin_data_shapes/Berlin_Bezirke_dissolve.gpkg"

# prompt: from the plans_path file can you extract how many people (Agents) are in the scenario?

# Stream through a MATSim plans file.
# The matsim.plan_reader returns a generator
# plans_generator = matsim.plan_reader(base_folder+plans_path)

# # To count unique persons, iterate through the generator and collect person IDs
# person_ids = set() # Using a set automatically handles uniqueness

# for person, plan in plans_generator:
#     # The person information is in the first element of the tuple
#     # The person object has attributes like 'id'
#     person_id = person.attrib.get('id')
#     if person_id: # Ensure the person id is not None or empty
#         person_ids.add(person_id)

# # Count the number of unique agents (persons)
# num_agents = len(person_ids)


# Berlin boundary prep

# Define the assumed CRS of the MATSim coordinates
matsim_crs = 'EPSG:25832'

# Load the Berlin boundary GeoPackage and ensure a valid CRS
berlin_outline_gdf = gpd.read_file(base_folder+berlin_outline)

# --- CRS Handling for Berlin Outline ---
# Target CRS for intersection/containment checks. ETRS89-LAEA (EPSG:3035) is good for Europe-wide
# analyses and should also be suitable for your raster data.
target_crs_for_geometry_check = 'EPSG:3035'

# Check if the GeoDataFrame has a CRS
if berlin_outline_gdf.crs is None:
    print(f"Warning: Berlin outline GeoDataFrame has no CRS. Assuming {target_crs_for_geometry_check} based on previous context.")
    berlin_outline_gdf = berlin_outline_gdf.to_crs(target_crs_for_geometry_check) # Use to_crs even for initial set to ensure consistency
else:
    print(f"Original Berlin outline CRS: {berlin_outline_gdf.crs}")
    if berlin_outline_gdf.crs != target_crs_for_geometry_check:
        berlin_outline_gdf = berlin_outline_gdf.to_crs(target_crs_for_geometry_check)
        print(f"Reprojected Berlin outline CRS: {berlin_outline_gdf.crs}")

# Assuming berlin_outline_gdf contains a single polygon or a MultiPolygon representing Berlin
# Dissolve if necessary and get the single geometry for easier checking
berlin_geometry = berlin_outline_gdf.dissolve().geometry.iloc[0]

transformer = Transformer.from_crs(matsim_crs, berlin_outline_gdf.crs, always_xy=True)

berlin_home_locations = []
berlin_work_locations = []
processed_all_agents = set()

plans_generator = matsim.plan_reader(base_folder + plans_path)

for person, plan in plans_generator:
    agent_id = person.attrib.get('id')

    if not agent_id or agent_id in processed_all_agents:
        continue

    processed_all_agents.add(agent_id)

    for item in plan:
        if item.tag == 'activity':
            activity_type = item.attrib.get('type', '').lower()
            # print(f"Activity name : {activity_type} with agent_id  {agent_id}")
            if 'home' in activity_type:
                try:
                    home_x = float(item.attrib['x'])
                    home_y = float(item.attrib['y'])

                    # Fast transform
                    home_x_proj, home_y_proj = transformer.transform(home_x, home_y)
                    home_reprojected_point = Point(home_x_proj, home_y_proj)

                    if berlin_geometry.contains(home_reprojected_point):
                       berlin_home_locations.append({
                            'agent_id': agent_id,
                            'x_home_matsim': home_x,
                            'y_home_matsim': home_y,
                            'home_geometry': home_reprojected_point
                        })
                       break  # Done with this agent
                except Exception as e:
                    print(f"Error for agent {agent_id}: {e}")
                break  # Whether success or fail, don’t keep checking activities
                
            if 'work' in activity_type:
                try:
                    work_x = float(item.attrib['x'])
                    work_y = float(item.attrib['y'])

                    # Fast transform
                    work_x_proj, work_y_proj = transformer.transform(work_x, work_y)
                    work_reprojected_point = Point(work_x_proj, work_y_proj)

                    if berlin_geometry.contains(work_reprojected_point):
                       berlin_work_locations.append({
                            'agent_id': agent_id,
                            'x_work_matsim': work_x,
                            'y_work_matsim': work_y,
                            'work_geometry': work_reprojected_point
                        })
                       break  # Done with this agent
                except Exception as e:
                    print(f"Error for agent {agent_id}: {e}")
                break  # Whether success or fail, don’t keep checking activities
                
print(f"The Matsim pouints coordinates were reprojected to the {berlin_outline_gdf.crs}")
print(f"Total number of agents in the scenario: {len(processed_all_agents)}")
print(f"Number of agents with home locations within Berlin : {len(berlin_home_locations)}")
print(f"Number of agents with work locations within Berlin : {len(berlin_work_locations)}")

# activity_types = set()

# for person, plan in plans_generator:
#     for item in plan:
#         if item.tag == 'activity':
#             activity_type = item.attrib.get('type')
#             if activity_type:
#                 activity_types.add(activity_type)

# print("Unique activity types found in the plans file:")
# for activity_type in sorted(activity_types):
#     print(f" - {activity_type}")

# prompt: export the agent_homes_df to csv
agents_in_berlin_output_path = "berlin_output/agents_berlin_citylim_home.csv"

# Convert the list of dictionaries to a pandas DataFrame
berlin_home_locations_df = pd.DataFrame(berlin_home_locations)

# Now call .to_csv() on the DataFrame
berlin_home_locations_df.to_csv(base_folder+agents_in_berlin_output_path, index=False)
print(f"DataFrame exported to csv: {base_folder+agents_in_berlin_output_path}")

# prompt: export the agent_homes_df to csv
agents_in_berlin_output_path = "berlin_output/agents_berlin_citylim_work.csv"

# Convert the list of dictionaries to a pandas DataFrame
berlin_work_locations_df = pd.DataFrame(berlin_work_locations)

# Now call .to_csv() on the DataFrame
berlin_work_locations_df.to_csv(base_folder+agents_in_berlin_output_path, index=False)
print(f"DataFrame exported to csv: {base_folder+agents_in_berlin_output_path}")