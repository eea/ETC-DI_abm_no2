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

import matsim
# import pandas as pd
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

all_eligible_home_locations = []
processed_all_agents = set() # To ensure we only process each agent once

print("Collecting all eligible home locations within Berlin for spatial selection...")
# Re-initialize the plans generator to ensure we read from the beginning
plans_generator = matsim.plan_reader(base_folder+plans_path)

for person, plan in plans_generator:
    agent_id = person.attrib.get('id')

    if agent_id and agent_id not in processed_all_agents:
        processed_all_agents.add(agent_id) # Mark as processed globally for this iteration

        home_activity_found = False
        for item in plan:
            if item.tag == 'activity':
                activity_type = item.attrib.get('type')
                if activity_type and 'home' in activity_type.lower():
                    home_x_str = item.attrib.get('x')
                    home_y_str = item.attrib.get('y')

                    if home_x_str is not None and home_y_str is not None:
                        try:
                            home_x = float(home_x_str)
                            home_y = float(home_y_str)

                            # Create GeoSeries for the MATSim point
                            matsim_point_gs = gpd.GeoSeries([Point(home_x, home_y)], crs=matsim_crs)

                            # Reproject the point to the Berlin outline's CRS
                            reprojected_point_gs = matsim_point_gs.to_crs(berlin_outline_gdf.crs)
                            reprojected_point = reprojected_point_gs.iloc[0]

                            if berlin_geometry.contains(reprojected_point):
                                all_eligible_home_locations.append({
                                    'agent_id': agent_id,
                                    'x_matsim': home_x, # Original MATSim X
                                    'y_matsim': home_y, # Original MATSim Y
                                    'geometry': reprojected_point # Reprojected geometry for spatial operations
                                })
                                # Found home within Berlin, move to the next agent
                                home_activity_found = True
                                break # Stop searching activities for this person
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Could not parse coordinates for agent {agent_id} home activity: {e}")
                        except Exception as e:
                            print(f"Error during CRS transformation for agent {agent_id}: {e}")
            if home_activity_found: # If a home activity was found for this person, no need to check other activities
                break
print(f"The Matsim pouints coordinates were reprojected to the {berlin_outline_gdf.crs}")
print(f"Total number of agents in the scenario: {len(processed_all_agents)}")
print(f"Number of agents with home locations within Berlin : {len(all_eligible_home_locations)}")
