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
# from collections import defaultdict
from pyproj import Transformer
plt.close('all')

plot_switch=False

base_folder         = "Z:/Environment and Health/Air Quality/abm_no2/"

agent_info_path     = "berlin_output/agent_locations_no2_v2.csv"
agent_info_path2     = "berlin_output/commute_summary.csv"

df1=pd.read_csv(base_folder+agent_info_path)
df2=pd.read_csv(base_folder+agent_info_path2)

# Step 1: Harmonize the IDs in df1
df1['person_id'] = df1['person_id'].apply(
    lambda x: x if str(x).startswith('bb_') else str(x).replace('berlin_', 'bb_', 1)
)
df2['agent_id'] = df2['agent_id'].apply(
    lambda x: x if str(x).startswith('bb_') else str(x).replace('berlin_', 'bb_', 1)
)


overlap = pd.merge(df1[['person_id']], df2[['agent_id']],
                   left_on='person_id', right_on='agent_id', how='inner')

# Unique IDs
unique_in_df1 = df1[~df1['person_id'].isin(df2['agent_id'])]
unique_in_df2 = df2[~df2['agent_id'].isin(df1['person_id'])]

print(f"Number of overlapping IDs: {len(overlap)}")
print(f"Number of unique IDs in df1: {len(unique_in_df1)}")
print(f"Number of unique IDs in df2: {len(unique_in_df2)}")