# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 15:04:12 2025

@author: jetschny
"""

import gzip


# ---------- CONFIG ----------
base_folder = "Z:/Environment and Health/Air Quality/abm_no2/berlin_data_matsim/"
network_path = 'berlin_data_matsim/berlin-v6.4.output_network.xml.gz'
events_path  = 'berlin_data_matsim/berlin-v6.4.output_events.xml.gz'
plans_path   = 'berlin_data_matsim/berlin-v6.4.output_plans.xml.gz'


src1 = base_folder+network_path
src2 = base_folder+events_path
src3 = base_folder+plans_path

dst1 = base_folder+"test_network.xml"
dst2 = base_folder+"test_events.xml"
dst3 = base_folder+"test_plans.xml"

with gzip.open(src1, "rt", encoding="utf-8", errors="ignore") as fin, open(dst1, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        fout.write(line)
        if i >= 10000:  # first 10,000 lines
            break

with gzip.open(src2, "rt", encoding="utf-8", errors="ignore") as fin, open(dst2, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        fout.write(line)
        if i >= 10000:  # first 10,000 lines
            break
        
with gzip.open(src3, "rt", encoding="utf-8", errors="ignore") as fin, open(dst3, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        fout.write(line)
        if i >= 10000:  # first 10,000 lines
            break


print("✅ Created:", dst1)
print("✅ Created:", dst2)
print("✅ Created:", dst3)
