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
import seaborn as sns
import matplotlib.pyplot as plt
# import geopandas as gpd
# import gzip
# import xml.etree.ElementTree as ET
# from geocube.api.core import make_geocube
# from shapely.geometry import Point
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
from pyproj import Transformer
plt.close('all')


# from collections import defaultdict


base_folder = "Z:/Environment and Health/Air Quality/abm_no2/"

agent_info_path = "berlin_output/agent_locations_no2_v2.csv"

agent_df = pd.read_csv(base_folder+agent_info_path)


# Create figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

sns.histplot(agent_df["NO2_home"], bins=12, kde=True, color="skyblue", edgecolor="black", ax=axes[0])

# Labels
axes[0].set_title("Home", fontsize=14, fontweight="bold")
axes[0].set_xlabel("NO2 concentration", fontsize=12)
axes[0].set_ylabel("Frequency", fontsize=12)
axes[0].set_ylim(0, 30000)  

sns.histplot(agent_df["NO2_work"], bins=12, kde=True, color="skyblue", edgecolor="black", ax=axes[1])

# Labels
axes[1].set_title("Work", fontsize=14, fontweight="bold")
axes[1].set_xlabel("NO2 concentration", fontsize=12)
axes[1].set_ylabel("Frequency", fontsize=12)
axes[1].set_ylim(0, 30000)  

sns.histplot(agent_df["NO2_home"]-agent_df["NO2_work"], bins=12, kde=True, color="skyblue", edgecolor="black", ax=axes[2])

# Labels
axes[2].set_title("Differences home-work", fontsize=14, fontweight="bold")
axes[2].set_xlabel("NO2 concentration", fontsize=12)
axes[2].set_ylabel("Frequency", fontsize=12)
axes[2].set_ylim(0, 30000)  

# Create figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

sns.histplot(agent_df["NO2_home"], bins=12, kde=True, color="skyblue", edgecolor="black", ax=axes[0])

# Labels
axes[0].set_title("Home", fontsize=14, fontweight="bold")
axes[0].set_xlabel("NO2 concentration", fontsize=12)
axes[0].set_ylabel("Frequency", fontsize=12)
axes[0].set_ylim(0, 30000)  

sns.histplot(agent_df["NO2_work"]*1/3+agent_df["NO2_home"]*2/3, bins=12, kde=True, color="skyblue", edgecolor="black", ax=axes[1])

# Labels
axes[1].set_title("Work-day (8h)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("NO2 concentration", fontsize=12)
axes[1].set_ylabel("Frequency", fontsize=12)
axes[1].set_ylim(0, 30000)  

sns.histplot(agent_df["NO2_home"]-(agent_df["NO2_work"]*1/3+agent_df["NO2_home"]*2/3), bins=12, kde=True, color="skyblue", edgecolor="black", ax=axes[2])

# Labels
axes[2].set_title("Differences home-workday", fontsize=14, fontweight="bold")
axes[2].set_xlabel("NO2 concentration", fontsize=12)
axes[2].set_ylabel("Frequency", fontsize=12)
axes[2].set_ylim(0, 30000)  


# plt.figure(figsize=(8, 5))
# plt.scatter(agent_df["NO2_home"]-agent_df["NO2_work"],agent_df["Dist_commute_m"], color="teal", edgecolor="black", alpha=0.7)
# plt.title("NO2 concentration vs. Commute Distance")
# plt.xlabel("NO2 concentration")
# plt.ylabel("Home-Work Distance (m)")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.show()
