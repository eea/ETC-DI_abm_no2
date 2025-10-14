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
import os, gzip, xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd


# base_folder = "Z:/Environment and Health/Air Quality/city_data_raw/abm_no2/"
base_folder = "Z:/Environment and Health/Air Quality/abm_no2/"
# network_path = "berlin_data_matsim/berlin-v6.4.output_network.xml.gz"

network_path = 'berlin_data_matsim/berlin-v6.4.output_network.xml.gz'
events_path = 'berlin_data_matsim/berlin-v6.4.output_events.xml.gz'
plans_path = 'berlin_data_matsim/berlin-v6.4.output_plans.xml.gz'


JAVA_TYPE_COERCERS = {
    "java.lang.Integer": lambda v: int(v),
    "java.lang.Double":  lambda v: float(v),
    "java.lang.Boolean": lambda v: v.strip().lower() == "true",
    "java.lang.Long":    lambda v: int(v),
    "java.lang.Float":   lambda v: float(v),
    "java.lang.String":  lambda v: v,
}

def _open_maybe_gz(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")

def _iter_person_attributes_from_plans(plans_path):
    """Yield (person_id, attrs_dict) from plans.xml(.gz), streaming."""
    with _open_maybe_gz(plans_path) as f:
        ctx = ET.iterparse(f, events=("start", "end"))
        _, root = next(ctx)  # get root
        person_id = None
        attrs = None
        inside_attrs = False

        for event, elem in ctx:
            tag = elem.tag
            if event == "start":
                if tag == "person":
                    person_id = elem.attrib.get("id")
                    attrs = {}
                elif tag == "attributes" and person_id is not None:
                    inside_attrs = True
                elif tag == "attribute" and inside_attrs and person_id is not None:
                    name = elem.attrib.get("name")
                    jcls = elem.attrib.get("class", "java.lang.String")
                    # value may be text inside element; will read on 'end'
            elif event == "end":
                if tag == "attribute" and inside_attrs and person_id is not None:
                    name = elem.attrib.get("name")
                    jcls = elem.attrib.get("class", "java.lang.String")
                    val_txt = (elem.text or "").strip()
                    if name:
                        coerce = JAVA_TYPE_COERCERS.get(jcls, JAVA_TYPE_COERCERS["java.lang.String"])
                        try:
                            attrs[name] = coerce(val_txt) if val_txt != "" else None
                        except Exception:
                            attrs[name] = val_txt  # keep raw if coercion fails
                    elem.clear()
                elif tag == "attributes":
                    inside_attrs = False
                    elem.clear()
                elif tag == "person":
                    # We’re done with this person
                    yield person_id, attrs or {}
                    person_id, attrs = None, None
                    root.clear()
    # end with
    return

def _iter_person_attributes_from_personAttributes(person_attrs_path):
    """Yield (person_id, attrs_dict) from personAttributes.xml(.gz)."""
    with _open_maybe_gz(person_attrs_path) as f:
        ctx = ET.iterparse(f, events=("start", "end"))
        _, root = next(ctx)
        person_id, attrs = None, None
        for event, elem in ctx:
            tag = elem.tag
            if event == "start":
                if tag == "person":
                    person_id = elem.attrib.get("id")
                    attrs = {}
            elif event == "end":
                if tag == "attribute" and person_id is not None:
                    name = elem.attrib.get("name")
                    jcls = elem.attrib.get("class", "java.lang.String")
                    val_txt = (elem.text or "").strip()
                    if name:
                        coerce = JAVA_TYPE_COERCERS.get(jcls, JAVA_TYPE_COERCERS["java.lang.String"])
                        try:
                            attrs[name] = coerce(val_txt) if val_txt != "" else None
                        except Exception:
                            attrs[name] = val_txt
                    elem.clear()
                elif tag == "person":
                    yield person_id, attrs or {}
                    person_id, attrs = None, None
                    root.clear()
    return

def build_agents_dataframe(base_folder, plans_path, fallback_person_attrs_path=None, sample=None):
    """
    Returns a pandas DataFrame with one row per person and columns for attributes.
    If `sample` is not None, limit to that many persons (for quick tests).
    """
    plans_file = os.path.join(base_folder, plans_path)
    # Try to find a sibling personAttributes file if not provided
    if fallback_person_attrs_path is None:
        candidate = plans_file.replace("plans", "personAttributes").replace("output_plans", "output_personAttributes")
        fallback_person_attrs_path = candidate if os.path.exists(candidate) else None

    rows = []
    n = 0

    # Prefer attributes embedded in plans
    try_sources = []
    try_sources.append(("plans", plans_file))
    if fallback_person_attrs_path:
        try_sources.append(("personAttrs", fallback_person_attrs_path))

    found_any = False
    for src_kind, path in try_sources:
        if not os.path.exists(path):
            continue
        it = _iter_person_attributes_from_plans(path) if src_kind == "plans" else _iter_person_attributes_from_personAttributes(path)
        for pid, attrs in it:
            found_any = True
            row = {"person_id": pid}
            row.update(attrs)
            rows.append(row)
            n += 1
            if sample and n >= sample:
                break
        if found_any:
            break  # stop after first successful source

    if not found_any:
        raise FileNotFoundError("No person attributes found in plans or personAttributes file.")

    df = pd.DataFrame(rows).drop_duplicates(subset=["person_id"])
    return df

# Peek at schema to confirm tag/attr variants
print(inspect_plans_schema(base_folder+plans_path, max_people=3))
print(inspect_events_schema(base_folder+events_path, max_events=5000))

# Build a tidy table of agents and their socio-economic attributes
agents = build_agents_dataframe(base_folder, plans_path)    # add sample=5000 for a quick preview

print("Columns found:", list(agents.columns))
print(agents.head())

# planned = extract_planned_commutes(base_folder+plans_path, network_path=base_folder+network_path, all_occurrences=False)
# realized = extract_realized_commutes(base_folder+events_path, plans_path=base_folder+plans_path, network_path=base_folder+network_path)

# print("planned rows:", len(planned))
# print("realized rows:", len(realized))

# save_csv(planned, "planned_commutes.csv")
# save_csv(realized, "realized_commutes.csv")

