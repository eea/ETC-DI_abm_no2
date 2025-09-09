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
import pandas as pd
# import sys
# import geopandas as gpd
# import gzip
# import xml.etree.ElementTree as ET
# from geocube.api.core import make_geocube
# from shapely.geometry import Point
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
from pyproj import Transformer

# from collections import defaultdict

# base_folder = "Z:/Environment and Health/Air Quality/city_data_raw/abm_no2/"
base_folder = "Z:/Environment and Health/Air Quality/abm_no2/"
# network_path = "berlin_data_matsim/berlin-v6.4.output_network.xml.gz"

agent_home_path = "berlin_output/agents_berlin_citylim_home.csv"
network_path = "berlin_output/network_berlin.csv"
network_xml =  "berlin_data_matsim/berlin-v6.4.output_network.xml.gz"

NO2_annual_raster_Berlin = "berlin_data_aq/NO2_2023_epsg3035_clipBerlin.tif"
# berlin_outline = "berlin_data_shapes/Berlin_Bezirke_dissolve.gpkg"


def normalize_series(s: pd.Series) -> pd.Series:
    # normalize to string, strip, lowercase; collapse internal whitespace
    s = s.astype("string").fillna("").str.strip().str.lower()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s

def build_token_version(s: pd.Series) -> pd.Series:
    # helpful if IDs sometimes have delimiters; keep alnum + delimiters collapsed
    # e.g., "link-123_a" -> "link-123_a"
    return s.str.replace(r"[^a-z0-9_\-:/\.]", "", regex=True)


home_df = pd.read_csv(base_folder+agent_home_path)
network_df = pd.read_csv(base_folder+network_path)

net = matsim.read_network(base_folder+network_xml)   # returns a network object

# Count unique agent ids before merge
home_count = home_df["link_id"].nunique()
network_count = network_df["link_id"].nunique()

# Build node dataframe
nodes = pd.DataFrame(net.nodes)

# Merge on agent id
# merged_df = pd.merge(home_df, work_df, on="agent_id", how="inner")

# Count after merge
# merged_count = merged_df["agent_id"].nunique()

# Calculate how many were removed
# removed_home = home_count - merged_count
# removed_work = work_count - merged_count

print(f"Number of link-ids accociated to agents : {home_count}")
print(f"Number of link-ids in network  : {network_count}")
# print(f"Agents kept (with both home & work): {merged_count}")

sa_raw = normalize_series(home_df["link_id"])
sb_raw = normalize_series(network_df["link_id"])

# You can try both raw and lightly tokenized/cleaned versions
sa = build_token_version(sa_raw)
sb = build_token_version(sb_raw)

ua = pd.Series(sa.unique(), name="a")
ub = pd.Series(sb.unique(), name="b")

print(f"Unique IDs in network: {len(ua):,}, B: {len(ub):,}")

# ----------------------
# 1) Exact matches
# ----------------------
exact = pd.Index(ua).intersection(pd.Index(ub))
print(f"Exact matches: {len(exact):,}")
# pd.DataFrame({"id": sorted(exact)}).to_csv(OUT / "link_exact_matches.csv", index=False)

# Merge on link_id
merged = home_df.merge(network_df, on="link_id", how="inner", suffixes=("_home", "_network"))

print(merged.head())
print(f"Number of matching rows: {len(merged)}")


# Join twice: once for from_node, once for to_node
merged = merged.merge(nodes.rename(columns={"node_id":"from_node","x":"from_x","y":"from_y"}),
                    on="from_node", how="left")
merged = merged.merge(nodes.rename(columns={"node_id":"to_node","x":"to_x","y":"to_y","length":"length"}),
                    on="to_node", how="left")

merged_home=merged[["agent_id","x_home_matsim","y_home_matsim"]].copy()
merged_commute_start=merged[["agent_id","from_x","from_y"]].copy()
merged_commute_end=merged[["agent_id","to_x","to_y"]].copy()

to_aq_crs = Transformer.from_crs("epsg:25832", "epsg:3035", always_xy=True)

# returns lon, lat
home_lon, home_lat = to_aq_crs.transform(merged_home["x_home_matsim"], merged_home["y_home_matsim"])
from_lon, from_lat = to_aq_crs.transform(merged_commute_start["from_x"], merged_commute_start["from_y"])
to_lon, to_lat = to_aq_crs.transform(merged_commute_end["to_x"], merged_commute_end["to_y"])

merged_home["lon"],merged_home["lat"] = home_lon, home_lat
merged_commute_start["lon"],merged_commute_start["lat"] = from_lon, from_lat
merged_commute_end["lon"],merged_commute_end["lat"] = to_lon, to_lat


merged.to_csv(base_folder+ "berlin_output/agent_commute.csv", index=False)
# merged_home.to_csv(base_folder+ "berlin_output/agent_home_location.csv", index=False)
# merged_commute_start.to_csv(base_folder+ "berlin_output/agent_commute_start_location.csv", index=False)
# merged_commute_end.to_csv(base_folder+ "berlin_output/agent_commute_end_location.csv", index=False)