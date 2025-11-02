# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 13:23:47 2025

@author: jetschny
"""
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


# example_usage.py
# from matsim_commutes import (
#     extract_planned_commutes,
#     extract_realized_commutes,
#     save_csv,
#     inspect_plans_schema,
#     inspect_events_schema,
# )
# import os, gzip, xml.etree.ElementTree as ET
# from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.close('all')

# base_folder = "Z:/Environment and Health/Air Quality/city_data_raw/abm_no2/"
base_folder = "Z:/Environment and Health/Air Quality/abm_no2/"
# network_path = "berlin_data_matsim/berlin-v6.4.output_network.xml.gz"

network_path = 'berlin_data_matsim/berlin-v6.4.output_network.xml.gz'
events_path = 'berlin_data_matsim/berlin-v6.4.output_events.xml.gz'
plans_path = 'berlin_data_matsim/berlin-v6.4.output_plans.xml.gz'

agent_home_path = "berlin_output/agent_socio-economic_stats.csv"

agent_df=pd.read_csv(base_folder+agent_home_path)

# Create figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)

sns.histplot(agent_df["age"],stat='percent', bins=10, color="skyblue", edgecolor="black", ax=axes[0][0])
# Labels
# axes[0][0].set_title("Age", fontsize=14, fontweight="bold")
axes[0][0].set_xlabel("Age", fontsize=12)
axes[0][0].set_ylabel("Percentage", fontsize=12)
axes[0][0].set_ylim(0, 30)  

sns.histplot(agent_df["employed"],stat='percent', bins=2,color="skyblue", edgecolor="black", ax=axes[0][1])
# Labels
# axes[0][0].set_title("Age", fontsize=14, fontweight="bold")
axes[0][1].set_xlabel("Employment", fontsize=12)
axes[0][1].set_ylabel("Percentage", fontsize=12)
axes[0][1].set_ylim(0, 60)  

sns.histplot(agent_df["income"],stat='percent', bins=10, color="skyblue", edgecolor="black", ax=axes[0][2])
# Labels
# axes[0][0].set_title("Age", fontsize=14, fontweight="bold")
axes[0][2].set_xlabel("Income", fontsize=12)
axes[0][2].set_ylabel("Percentage", fontsize=12)
axes[0][2].set_ylim(0, 30)  

sns.histplot(agent_df["sex"],stat='percent', color="skyblue", edgecolor="black", ax=axes[1][0])
# Labels
# axes[0][0].set_title("Age", fontsize=14, fontweight="bold")
axes[1][0].set_xlabel("Sex", fontsize=12)
axes[1][0].set_ylabel("Percentage", fontsize=12)
axes[1][0].set_ylim(0, 60)  

sns.histplot(agent_df["household_size"],stat='percent', color="skyblue", edgecolor="black", ax=axes[1][1])
# Labels
# axes[0][0].set_title("Age", fontsize=14, fontweight="bold")
axes[1][1].set_xlabel("Household size", fontsize=12)
axes[1][1].set_ylabel("Percentage", fontsize=12)
axes[1][1].set_ylim(0, 30)  

sns.histplot(agent_df["carAvail"],stat='percent',color="skyblue", edgecolor="black", ax=axes[1][2])
# Labels
# axes[0][0].set_title("Age", fontsize=14, fontweight="bold")
axes[1][2].set_xlabel("Car availibility", fontsize=12)
axes[1][2].set_ylabel("Percentage", fontsize=12)
axes[1][2].set_ylim(0, 100)  

agent_df_export=agent_df[["person_id", "age", "economic_status", "employed", "employment", "household_size", "income", "sex"]].copy()
agent_df_export.to_csv(base_folder+ "berlin_output/agent_socio-economic_stats_short.csv", index=False)