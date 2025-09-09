# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 13:23:47 2025

@author: jetschny
"""

# example_usage.py
from matsim_commutes import (
    extract_planned_commutes,
    extract_realized_commutes,
    save_csv,
    inspect_plans_schema,
    inspect_events_schema,
)

# base_folder = "Z:/Environment and Health/Air Quality/city_data_raw/abm_no2/"
base_folder = "Z:/Environment and Health/Air Quality/abm_no2/"
# network_path = "berlin_data_matsim/berlin-v6.4.output_network.xml.gz"

network_path = 'berlin_data_matsim/berlin-v6.4.output_network.xml.gz'
events_path = 'berlin_data_matsim/berlin-v6.4.output_events.xml.gz'
plans_path = 'berlin_data_matsim/berlin-v6.4.output_plans.xml.gz'

# Peek at schema to confirm tag/attr variants
print(inspect_plans_schema(base_folder+plans_path, max_people=3))
print(inspect_events_schema(base_folder+events_path, max_events=5000))


planned = extract_planned_commutes(base_folder+plans_path, network_path=base_folder+network_path, all_occurrences=False)
realized = extract_realized_commutes(base_folder+events_path, plans_path=base_folder+plans_path, network_path=base_folder+network_path)

print("planned rows:", len(planned))
print("realized rows:", len(realized))

save_csv(planned, "planned_commutes.csv")
save_csv(realized, "realized_commutes.csv")


# print("Extract Planned commutes from the selected plan per person")
# # 1) Planned commutes from the selected plan per person
# planned = extract_planned_commutes(
#     plans_path=base_folder+plans_path,
#     network_path=base_folder+network_path,   # optional, for link→(x,y)
#     all_occurrences=False                    # True to capture all home→work in a day
# )
# print("Exxport to csv ")

# save_csv(planned, base_folder+"berlin_output/planned_commutes.csv")
# print("Extract realized commutes paired from events (home act end → work act start)")

# # 2) Realized commutes paired from events (home act end → work act start)
# realized = extract_realized_commutes(
#     events_path=base_folder+events_path,
#     plans_path=base_folder+plans_path,       # optional, for coords if present in plan
#     network_path=base_folder+network_path,   # optional, fills coords via link midpoints
#     all_occurrences=False
# )
# print("Exxport to csv ")

# save_csv(realized, base_folder+"berlin_output/realized_commutes.csv")
# print(f"Planned rows: {len(planned)} | Realized rows: {len(realized)}")



