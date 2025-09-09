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

# import matsim
import pandas as pd
import rasterio
# import sys
import geopandas as gpd
# import gzip
# import xml.etree.ElementTree as ET
# from geocube.api.core import make_geocube
# from shapely.geometry import Point
# import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image
# from pyproj import Transformer

# from collections import defaultdict

# base_folder = "Z:/Environment and Health/Air Quality/city_data_raw/abm_no2/"
base_folder = "Z:/Environment and Health/Air Quality/abm_no2/"
# network_path = "berlin_data_matsim/berlin-v6.4.output_network.xml.gz"

agent_home_path              ="berlin_output/agent_commute.csv"
# agent_home_path              ="berlin_output/agent_home_location.csv"
# agent_commute_start_path     ="berlin_output/agent_commute_start_location.csv"
# agent_commute_end_path       ="berlin_output/agent_commute_end_location.csv"

NO2_annual_raster_Berlin = "berlin_data_aq/NO2_2023_epsg3035.tif"
# berlin_outline = "berlin_data_shapes/Berlin_Bezirke_dissolve.gpkg"

print("Loading data")
commute_df = pd.read_csv(base_folder+agent_home_path)
# home_df = pd.read_csv(base_folder+agent_home_path)
# commute_start_df = pd.read_csv(base_folder+agent_commute_start_path)
# commute_end_df = pd.read_csv(base_folder+agent_commute_end_path)


home_df = commute_df[["x_home_matsim","y_home_matsim"]].rename(columns={'x_home_matsim': 'lon','y_home_matsim': 'lat'})
comstart_df = commute_df[["from_x","from_y"]].rename(columns={'from_x': 'lon','from_y': 'lat'})
comend_df = commute_df[["to_x","to_y"]].rename(columns={'to_x': 'lon','to_y': 'lat'})

# home_df = pd.DataFrame(np.vstack(commute_df["x_home_matsim"], commute_df["y_home_matsim"]), columns=["lon","lat"])

print("Ceating geo-dataframes")
#convert from data frame to geo-datafram
home_gdf = gpd.GeoDataFrame(
        commute_df,
        geometry=gpd.points_from_xy(home_df["lon"], home_df["lat"]),
        crs="EPSG:3035",
    )
comstart_gdf = gpd.GeoDataFrame(
        comstart_df,
        geometry=gpd.points_from_xy(comstart_df["lon"], comstart_df["lat"]),
        crs="EPSG:3035",
    )
comend_gdf = gpd.GeoDataFrame(
        comend_df,
        geometry=gpd.points_from_xy(comend_df["lon"], comend_df["lat"]),
        crs="EPSG:3035",
    )

print("Preparing coordinates")
# Prepare (x, y) tuples in raster CRS
home_coords = [(geom.x, geom.y) for geom in home_gdf.geometry]
comstart_coords = [(geom.x, geom.y) for geom in comstart_gdf.geometry]
comend_coords = [(geom.x, geom.y) for geom in comend_gdf.geometry]

print("Extracting NO2 values for coordinates ")

# Load the NO2 raster
with rasterio.open(base_folder+NO2_annual_raster_Berlin) as src:
    # Reproject points into raster CRS if needed
    if src.crs is None:
        raise ValueError("Raster has no CRS. Please define or provide a georeferenced raster.")
    # pts = gdf.to_crs(src.crs)

    
    # Sample raster (nearest-neighbor)
    # For multi-band rasters, choose band via indexes=[band]
    # Note: rasterio.sample returns an iterator of arrays
    home_samples = list(src.sample(home_coords))
    comstart_samples = list(src.sample(comstart_coords))
    comend_samples = list(src.sample(comend_coords))

    # Flatten and handle nodata
    nodata = src.nodata
    home_vals = []
    comstart_vals = []
    comend_vals = []
    
    for s in home_samples:
        val = s.item() if hasattr(s, "item") else float(s[0])  # scalar
        if nodata is not None and val == nodata:
            home_vals.append(np.nan)
        else:
            # mask values outside raster bounds also come back as fill; rasterio.sample
            # returns the dataset's nodata for out-of-bounds in many cases; we cover with nan above.
            home_vals.append(val)
            
    for s in comstart_samples:
        val = s.item() if hasattr(s, "item") else float(s[0])  # scalar
        if nodata is not None and val == nodata:
            comstart_vals.append(np.nan)
        else:
            # mask values outside raster bounds also come back as fill; rasterio.sample
            # returns the dataset's nodata for out-of-bounds in many cases; we cover with nan above.
            comstart_vals.append(val)
            
    for s in comend_samples:
        val = s.item() if hasattr(s, "item") else float(s[0])  # scalar
        if nodata is not None and val == nodata:
            comend_vals.append(np.nan)
        else:
            # mask values outside raster bounds also come back as fill; rasterio.sample
            # returns the dataset's nodata for out-of-bounds in many cases; we cover with nan above.
            comend_vals.append(val)

print("Creating final dataframes")
# attach to original dataframe (preserves original order)
# home_df=home_df.drop(columns=['x_home_matsim', 'y_home_matsim'])
# home_df=home_df.rename(columns={"lon": "lon_home", "lat": "lat_home"})
# home_df["lon_comstart"]=comstart_df["lon"]
# home_df["lat_comstart"]=comstart_df["lat"]
# home_df["lon_comend"]=comend_df["lon"]
# home_df["lat_comend"]=comend_df["lat"]


commute_df["NO2_home"] = home_vals
commute_df["NO2_comstart"] = comstart_vals
commute_df["NO2_comend"] = comend_vals
 
 
commute_df["Dist_commute_m"] = np.sqrt((commute_df["from_x"] - commute_df["to_x"])**2 + (commute_df["from_y"] - commute_df["to_y"])**2)
# a = np.sqrt((home_df["lon_comstart"] - home_df["lon_home"])**2 + (home_df["lat_comstart"] - home_df["lat_home"])**2)
# b = np.sqrt((home_df["lon_comend"] - home_df["lon_home"])**2 + (home_df["lat_comend"] - home_df["lat_home"])**2)

commute_df.head()

export_df = commute_df[["agent_id","link_id","length","x_home_matsim","y_home_matsim","from_x","from_y","to_x", "to_y","NO2_home","NO2_comstart","NO2_comend"]]

print("Exporting dataframe as csv")
export_df.to_csv(base_folder+ "berlin_output/agent_locations_no2.csv", index=False)
print("... all finished")