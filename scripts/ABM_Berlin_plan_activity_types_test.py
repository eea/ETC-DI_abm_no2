import gzip
import xml.etree.ElementTree as ET

path = r"Z:\Environment and Health\Air Quality\abm_no2\berlin_data_matsim\berlin-v6.4.output_plans.xml.gz"

def local(tag):
    return tag.split('}', 1)[-1]

with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
    ctx = ET.iterparse(f, events=("end",))
    act_types = set()
    count_persons = 0
    for ev, el in ctx:
        if local(el.tag) == "person":
            count_persons += 1
        elif local(el.tag) in ("act", "activity"):
            t = el.attrib.get("type")
            if t:
                act_types.add(t)
        if count_persons >= 5000:
            break
    del ctx

print("Example activity types:", act_types)
print("Persons seen:", count_persons)

# if any("home" in a["type"].lower() for a in acts):
#     if any("work" in a["type"].lower() for a in acts):
#         print("Found person with both home and work:", current_person)
#         break