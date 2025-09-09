
# matsim_commute_extractor.py
# Extract home→work commute locations from a MATSim plans file.
# Usage:
#   python matsim_commute_extractor.py --plans plans.xml.gz --out commutes.csv
# Optional (to map linkIds to coordinates if activities specify link rather than x/y):
#   python matsim_commute_extractor.py --plans plans.xml.gz --network network.xml.gz --out commutes.csv
#
# Notes:
# - We read only the selected plan (selected="yes") per person.
# - We detect 'home' and 'work' by robust matching:
#     * case-insensitive exact match 'home'/'work'
#     * prefix before an underscore: e.g., 'home_8_to_16' -> 'home'
#     * first token before ':' (common in some pipelines): 'home:zone123' -> 'home'
# - We define the commute as the FIRST leg that goes from a home activity to a subsequent work activity.
#   (Adjust with --all if you want ALL home→work occurrences in a day.)
# - Outputs CSV columns:
#     person_id, home_x, home_y, home_link, work_x, work_y, work_link, dep_time_s, arr_time_s, mode
#
# Requires: Python 3.8+, no external libs.

import argparse
import csv
import gzip
import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple, Iterable

def open_maybe_gzip(path: str):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'rt', encoding='utf-8')

def parse_time_to_seconds(t: Optional[str]) -> Optional[int]:
    """Parse MATSim time strings (e.g., '08:30:00' or seconds) into seconds. Return None if missing."""
    if t is None or t == '':
        return None
    # If plain integer seconds
    if t.isdigit():
        return int(t)
    # HH:MM:SS (may exceed 24h in some scenarios)
    parts = t.split(':')
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    # Fallback: try float seconds
    try:
        return int(float(t))
    except ValueError:
        return None

def normalize_type(t: Optional[str]) -> str:
    if not t:
        return ''
    t_low = t.lower()
    # common decorations: 'home_8_to_16', 'home:zone123'
    if '_' in t_low:
        t_low = t_low.split('_', 1)[0]
    if ':' in t_low:
        t_low = t_low.split(':', 1)[0]
    return t_low.strip()

def is_home(act_type: str) -> bool:
    t = normalize_type(act_type)
    return t in {'home', 'h'}  # add more variants here if needed

def is_work(act_type: str) -> bool:
    t = normalize_type(act_type)
    return t in {'work', 'w', 'office', 'job'}  # extend as needed

def build_network_link_xy(network_path: Optional[str]) -> Dict[str, Tuple[float, float]]:
    """Parse network once to map linkId -> (x, y) using from/to node coordinates or link coords if present."""
    if not network_path:
        return {}
    link_xy: Dict[str, Tuple[float, float]] = {}
    with open_maybe_gzip(network_path) as f:
        context = ET.iterparse(f, events=('start', 'end'))
        nodes: Dict[str, Tuple[float,float]] = {}
        in_nodes = False
        in_links = False
        for ev, elem in context:
            tag = elem.tag.split('}')[-1]
            if ev == 'start' and tag == 'nodes':
                in_nodes = True
            elif ev == 'end' and tag == 'nodes':
                in_nodes = False
            elif ev == 'start' and tag == 'links':
                in_links = True
            elif ev == 'end' and tag == 'links':
                in_links = False
            elif ev == 'end' and in_nodes and tag == 'node':
                nid = elem.get('id')
                x = elem.get('x'); y = elem.get('y')
                if nid and x and y:
                    try:
                        nodes[nid] = (float(x), float(y))
                    except ValueError:
                        pass
                elem.clear()
            elif ev == 'end' and in_links and tag == 'link':
                lid = elem.get('id'); from_id = elem.get('from'); to_id = elem.get('to')
                # Prefer center as average of from/to if available
                if lid and from_id in nodes and to_id in nodes:
                    fx, fy = nodes[from_id]; tx, ty = nodes[to_id]
                    link_xy[lid] = ((fx + tx)/2.0, (fy + ty)/2.0)
                elem.clear()
            # free memory occasionally
            if ev == 'end' and tag in {'node','link'}:
                elem.clear()
    return link_xy

def activities_and_legs(plan_elem) -> Iterable[Tuple[str, dict]]:
    """Yield a sequence of ('act', attrs_dict) and ('leg', attrs_dict) in order for a plan element."""
    for child in plan_elem:
        tag = child.tag.split('}')[-1]
        if tag == 'act':
            yield ('act', child.attrib.copy())
        elif tag == 'leg':
            leg = child.attrib.copy()
            # Try to read dep_time from <route> start/end if not set; MATSim leg may not carry times.
            yield ('leg', leg)

def best_xy(attrs: dict, link_xy: Dict[str, Tuple[float,float]]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    x = attrs.get('x'); y = attrs.get('y'); link = attrs.get('link')
    if x and y:
        try:
            return float(x), float(y), link
        except ValueError:
            pass
    if link and link in link_xy:
        lx, ly = link_xy[link]
        return lx, ly, link
    return None, None, link

def extract_commute_from_plan(plan_elem, link_xy: Dict[str, Tuple[float,float]], all_occurrences: bool=False):
    """Return list of commute dicts for the FIRST (or all) home→work pair with the leg between them."""
    seq = list(activities_and_legs(plan_elem))
    results = []
    # Find patterns: act(home) -> leg -> act(work)
    for i in range(len(seq)-2):
        k1, a1 = seq[i]
        k2, lg = seq[i+1]
        k3, a3 = seq[i+2]
        if k1=='act' and k2=='leg' and k3=='act' and is_home(a1.get('type','')) and is_work(a3.get('type','')):
            hx, hy, hlink = best_xy(a1, link_xy)
            wx, wy, wlink = best_xy(a3, link_xy)
            dep = parse_time_to_seconds(a1.get('end_time') or a1.get('endTime'))
            arr = parse_time_to_seconds(a3.get('start_time') or a3.get('startTime'))
            mode = lg.get('mode')
            results.append({
                'home_x': hx, 'home_y': hy, 'home_link': hlink,
                'work_x': wx, 'work_y': wy, 'work_link': wlink,
                'dep_time_s': dep, 'arr_time_s': arr, 'mode': mode
            })
            if not all_occurrences:
                break
    return results

def iter_selected_plans(plans_path: str):
    with open_maybe_gzip(plans_path) as f:
        context = ET.iterparse(f, events=('start','end'))
        person_id = None
        current_plan = None
        selected = False
        for ev, elem in context:
            tag = elem.tag.split('}')[-1]
            if ev=='start' and tag=='person':
                person_id = elem.get('id')
            elif ev=='end' and tag=='person':
                person_id = None
                elem.clear()
            elif ev=='start' and tag=='plan':
                selected = (elem.get('selected','no').lower() in {'yes','true','1'})
                current_plan = elem
            elif ev=='end' and tag=='plan':
                if selected and person_id is not None:
                    yield person_id, current_plan
                # Clear to save memory
                if current_plan is not None:
                    current_plan.clear()
                current_plan = None
                selected = False
            # free memory
            if ev=='end' and tag in {'act','leg'}:
                elem.clear()

def main():
    ap = argparse.ArgumentParser(description='Extract home→work commute locations from MATSim plans.')
    ap.add_argument('--plans', required=True, help='Path to plans.xml or plans.xml.gz')
    ap.add_argument('--network', required=False, help='Path to network.xml or network.xml.gz (optional)')
    ap.add_argument('--out', required=True, help='Output CSV path')
    ap.add_argument('--all', action='store_true', help='Extract ALL home→work occurrences (default: first only)')
    args = ap.parse_args()

    link_xy = build_network_link_xy(args.network) if args.network else {}

    with open(args.out, 'w', newline='', encoding='utf-8') as outf:
        writer = csv.DictWriter(outf, fieldnames=['person_id','home_x','home_y','home_link','work_x','work_y','work_link','dep_time_s','arr_time_s','mode'])
        writer.writeheader()
        count = 0
        for person_id, plan_elem in iter_selected_plans(args.plans):
            commutes = extract_commute_from_plan(plan_elem, link_xy, all_occurrences=args.all)
            for c in commutes:
                c_row = {'person_id': person_id}
                c_row.update(c)
                writer.writerow(c_row)
                count += 1
    print(f'Wrote {count} commute rows to {args.out}')

if __name__ == '__main__':
    main()
