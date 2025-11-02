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
from pyproj import Transformer

# from collections import defaultdict

# base_folder = "Z:/Environment and Health/Air Quality/city_data_raw/abm_no2/"
base_folder = "Z:/Environment and Health/Air Quality/abm_no2/"
# network_path = "berlin_data_matsim/berlin-v6.4.output_network.xml.gz"

# agent_home_path              ="berlin_output/planned_commutes.csv"
agent_home_path              ="berlin_output/commute_summary.csv"
# agent_commute_start_path     ="berlin_output/agent_commute_start_location.csv"
# agent_commute_end_path       ="berlin_output/agent_commute_end_location.csv"

NO2_annual_raster_Berlin = "berlin_data_aq/NO2_2023_epsg3035.tif"
berlin_outline = "berlin_data_shapes/Berlin_Bezirke_dissolve.gpkg"

print("Loading data")
commute_df = pd.read_csv(base_folder+agent_home_path)
# home_df = pd.read_csv(base_folder+agent_home_path)
# commute_start_df = pd.read_csv(base_folder+agent_commute_start_path)
# commute_end_df = pd.read_csv(base_folder+agent_commute_end_path)

berlin_outline_gdf = gpd.read_file(base_folder+berlin_outline)
target_crs_for_geometry_check = 'EPSG:3035'
berlin_outline_gdf = berlin_outline_gdf.to_crs(target_crs_for_geometry_check) # Use to_crs even for initial set to ensure consistency
berlin_geometry = berlin_outline_gdf.dissolve().geometry.iloc[0]

# home_df = pd.DataFrame(np.vstack(commute_df["x_home_matsim"], commute_df["y_home_matsim"]), columns=["lon","lat"])
to_aq_crs = Transformer.from_crs("epsg:25832", "epsg:3035", always_xy=True)

home_lon, home_lat = to_aq_crs.transform(commute_df["home_x"], commute_df["home_y"])
work_lon, work_lat = to_aq_crs.transform(commute_df["work_x"], commute_df["work_y"])

commute_df["home_lon"],commute_df["home_lat"] = home_lon, home_lat
commute_df["work_lon"],commute_df["work_lat"] = work_lon, work_lat


commute_gdf = gpd.GeoDataFrame(
        commute_df,
        geometry=gpd.points_from_xy(commute_df["home_lon"], commute_df["home_lat"]),
        crs="EPSG:3035",
    )
# Clip points by the outline (keeps only points inside the outline)
commute_df = gpd.clip(commute_gdf, berlin_outline_gdf)

home_df = commute_df[["home_lon","home_lat"]].rename(columns={'home_lon': 'lon','home_lat': 'lat'})
work_df = commute_df[["work_lon","work_lat"]].rename(columns={'work_lon': 'lon','work_lat': 'lat'})


print("Ceating geo-dataframes")
#convert from data frame to geo-datafram
home_gdf = gpd.GeoDataFrame(
        home_df,
        geometry=gpd.points_from_xy(home_df["lon"], home_df["lat"]),
        crs="EPSG:3035",
    )
work_gdf = gpd.GeoDataFrame(
        work_df,
        geometry=gpd.points_from_xy(work_df["lon"], work_df["lat"]),
        crs="EPSG:3035",
    )

    
print("Preparing coordinates")
# Prepare (x, y) tuples in raster CRS
home_coords = [(geom.x, geom.y) for geom in home_gdf.geometry]
work_coords = [(geom.x, geom.y) for geom in work_gdf.geometry]


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
    work_samples = list(src.sample(work_coords))

    # Flatten and handle nodata
    nodata = src.nodata
    home_vals = []
    work_vals = []
    comend_vals = []
    
    for s in home_samples:
        val = s.item() if hasattr(s, "item") else float(s[0])  # scalar
        if nodata is not None and val == nodata:
            home_vals.append(np.nan)
        else:
            # mask values outside raster bounds also come back as fill; rasterio.sample
            # returns the dataset's nodata for out-of-bounds in many cases; we cover with nan above.
            home_vals.append(val)
            
    for s in work_samples:
        val = s.item() if hasattr(s, "item") else float(s[0])  # scalar
        if nodata is not None and val == nodata:
            work_vals.append(np.nan)
        else:
            # mask values outside raster bounds also come back as fill; rasterio.sample
            # returns the dataset's nodata for out-of-bounds in many cases; we cover with nan above.
            work_vals.append(val)
            
   
print("Creating final dataframes")
# attach to original dataframe (preserves original order)
# home_df=home_df.drop(columns=['x_home_matsim', 'y_home_matsim'])
# home_df=home_df.rename(columns={"lon": "lon_home", "lat": "lat_home"})
# home_df["lon_comstart"]=work_df["lon"]
# home_df["lat_comstart"]=work_df["lat"]
# home_df["lon_comend"]=comend_df["lon"]
# home_df["lat_comend"]=comend_df["lat"]


commute_df["NO2_home"] = home_vals
commute_df["NO2_work"] = work_vals
commute_df["NO2_workday"] = commute_df["NO2_work"]*1/3+commute_df["NO2_home"]*2/3
commute_df["deviation"]=commute_df["NO2_home"]/(commute_df["NO2_work"]*1/3+commute_df["NO2_home"]*2/3)*100-100

  
commute_df["Dist_commute_m"] = np.sqrt((commute_df["home_lon"] - commute_df["work_lon"])**2 + (commute_df["work_lat"] - commute_df["work_lat"])**2)
# a = np.sqrt((home_df["lon_comstart"] - home_df["lon_home"])**2 + (home_df["lat_comstart"] - home_df["lat_home"])**2)
# b = np.sqrt((home_df["lon_comend"] - home_df["lon_home"])**2 + (home_df["lat_comend"] - home_df["lat_home"])**2)

commute_df.head()

# export_df = commute_df[["agent_id","link_id","length","x_home_matsim","y_home_matsim","from_x","from_y","to_x", "to_y","NO2_home","NO2_comstart","NO2_comend"]]

print("Exporting dataframe as csv")
commute_df.to_csv(base_folder+ "berlin_output/agent_locations_no2_v3.csv", index=False)

# Take a random sample every 10 rows
commute_samples_df = commute_df.sample(n=1000, random_state=42)
commute_samples_df.to_csv(base_folder+ "berlin_output/agent_locations_no2_v3_subset.csv", index=False)
print("... all finished")