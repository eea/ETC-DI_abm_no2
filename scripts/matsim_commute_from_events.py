
# matsim_commute_from_events.py
# Extract realized home→work commute timings from MATSim events.
#
# Usage:
#   python matsim_commute_from_events.py --events events.xml.gz --out realized_commutes.csv
#
# Optional enrichment for coordinates:
#   --plans plans.xml.gz      # to get home/work x/y or link per person (selected plan)
#   --network network.xml.gz  # to map linkId->(x,y) if only link is known
#   --all                     # capture all home→work pairs (default: first pair only)
#
# Output columns:
#   person_id, dep_time_s, arr_time_s, home_x, home_y, home_link, work_x, work_y, work_link
#
# Notes:
# - We scan events in chronological order and pair actend(type=home) with the next actstart(type=work).
# - If multiple home/work cycles exist, use --all to capture all pairs.
# - If --plans is provided, we read the selected plan once to cache home/work coordinates per person.
# - If --network is provided, we map link IDs to approximate (x,y) via midpoints of from/to nodes.

import argparse
import gzip
import csv
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, Optional, Iterable

def open_maybe_gzip(path: str):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'rt', encoding='utf-8')

def parse_time_to_seconds(t: str) -> Optional[int]:
    if t is None or t == '':
        return None
    if t.isdigit():
        return int(t)
    parts = t.split(':')
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        h, m, s = map(int, parts)
        return h*3600 + m*60 + s
    try:
        return int(float(t))
    except ValueError:
        return None

def normalize_type(t: Optional[str]) -> str:
    if not t:
        return ''
    t = t.lower()
    if '_' in t:
        t = t.split('_', 1)[0]
    if ':' in t:
        t = t.split(':', 1)[0]
    return t.strip()

def is_home(t: Optional[str]) -> bool:
    t = normalize_type(t)
    return t in {'home','h'}

def is_work(t: Optional[str]) -> bool:
    t = normalize_type(t)
    return t in {'work','w','office','job'}

def build_network_link_xy(network_path: Optional[str]) -> Dict[str, Tuple[float, float]]:
    if not network_path:
        return {}
    link_xy = {}
    nodes = {}
    with open_maybe_gzip(network_path) as f:
        context = ET.iterparse(f, events=('start','end'))
        in_nodes = False; in_links = False
        for ev, elem in context:
            tag = elem.tag.split('}')[-1]
            if ev=='start' and tag=='nodes':
                in_nodes = True
            elif ev=='end' and tag=='nodes':
                in_nodes = False
            elif ev=='start' and tag=='links':
                in_links = True
            elif ev=='end' and tag=='links':
                in_links = False
            elif ev=='end' and in_nodes and tag=='node':
                nid = elem.get('id'); x = elem.get('x'); y = elem.get('y')
                if nid and x and y:
                    try:
                        nodes[nid] = (float(x), float(y))
                    except ValueError:
                        pass
                elem.clear()
            elif ev=='end' and in_links and tag=='link':
                lid = elem.get('id'); fid = elem.get('from'); tid = elem.get('to')
                if lid and fid in nodes and tid in nodes:
                    fx, fy = nodes[fid]; tx, ty = nodes[tid]
                    link_xy[lid] = ((fx+tx)/2.0, (fy+ty)/2.0)
                elem.clear()
            if ev=='end' and tag in {'node','link'}:
                elem.clear()
    return link_xy

def read_home_work_from_plans(plans_path: str) -> Dict[str, Dict[str, Tuple[Optional[float], Optional[float], Optional[str]]]]:
    """Return {person_id: {'home': (x,y,link), 'work': (x,y,link)}} using selected plan."""
    def best_xy(attrs):
        x = attrs.get('x'); y = attrs.get('y'); link = attrs.get('link')
        try:
            if x is not None and y is not None:
                return float(x), float(y), link
        except ValueError:
            pass
        return None, None, link

    result: Dict[str, Dict[str, Tuple[Optional[float], Optional[float], Optional[str]]]] = {}
    with open_maybe_gzip(plans_path) as f:
        context = ET.iterparse(f, events=('start','end'))
        person_id = None; in_selected = False
        for ev, elem in context:
            tag = elem.tag.split('}')[-1]
            if ev=='start' and tag=='person':
                person_id = elem.get('id')
                result.setdefault(person_id, {})
            elif ev=='end' and tag=='person':
                person_id = None; elem.clear()
            elif ev=='start' and tag=='plan':
                sel = elem.get('selected','no').lower() in {'yes','true','1'}
                in_selected = sel
            elif ev=='end' and tag=='plan':
                in_selected = False; elem.clear()
            elif ev=='end' and tag=='act' and in_selected and person_id is not None:
                a_type = elem.get('type','')
                if is_home(a_type) and 'home' not in result[person_id]:
                    result[person_id]['home'] = best_xy(elem.attrib)
                elif is_work(a_type) and 'work' not in result[person_id]:
                    result[person_id]['work'] = best_xy(elem.attrib)
                elem.clear()
            if ev=='end' and tag in {'leg'}:
                elem.clear()
    return result

def main():
    ap = argparse.ArgumentParser(description='Extract realized home→work commutes from MATSim events.')
    ap.add_argument('--events', required=True, help='events.xml or events.xml.gz')
    ap.add_argument('--out', required=True, help='Output CSV path')
    ap.add_argument('--plans', required=False, help='plans.xml(.gz) to enrich with coordinates')
    ap.add_argument('--network', required=False, help='network.xml(.gz) to map link->x,y')
    ap.add_argument('--all', action='store_true', help='Capture all home→work pairs (default: first only)')
    args = ap.parse_args()

    link_xy = build_network_link_xy(args.network) if args.network else {}
    home_work_cache = read_home_work_from_plans(args.plans) if args.plans else {}

    # State per person for pairing home→work
    last_home_dep: Dict[str, Tuple[Optional[int], Optional[str]]] = {}
    results = []

    with open_maybe_gzip(args.events) as f:
        context = ET.iterparse(f, events=('end',))
        for ev, elem in context:
            tag = elem.tag.split('}')[-1]
            if tag != 'event':
                elem.clear()
                continue
            etype = elem.get('type', '').lower()
            person = elem.get('person') or elem.get('agent')
            time_s = parse_time_to_seconds(elem.get('time'))
            actType = elem.get('actType') or elem.get('acttype')  # case variants
            link = elem.get('link') or elem.get('linkId')

            if etype == 'actend' and person and is_home(actType):
                # Mark a candidate home departure time (end of home activity)
                last_home_dep[person] = (time_s, link)

            elif etype == 'actstart' and person and is_work(actType):
                # Pair with the last home end if present
                if person in last_home_dep:
                    dep_time_s, home_link = last_home_dep[person]
                    arr_time_s = time_s
                    # Prepare coordinates from cache or link map
                    hx = hy = wx = wy = None
                    hw_links = {'home_link': None, 'work_link': None}
                    # from plans cache first
                    if person in home_work_cache:
                        hx, hy, hw_links['home_link'] = home_work_cache[person].get('home', (None,None,None))
                        wx, wy, hw_links['work_link'] = home_work_cache[person].get('work', (None,None,None))
                    # fallback to event links if missing
                    if hw_links['home_link'] is None:
                        hw_links['home_link'] = home_link
                    if hw_links['work_link'] is None:
                        hw_links['work_link'] = link
                    # if xy missing but we have link and network map
                    if (hx is None or hy is None) and hw_links['home_link'] and hw_links['home_link'] in link_xy:
                        hx, hy = link_xy[hw_links['home_link']]
                    if (wx is None or wy is None) and hw_links['work_link'] and hw_links['work_link'] in link_xy:
                        wx, wy = link_xy[hw_links['work_link']]

                    results.append({
                        'person_id': person,
                        'dep_time_s': dep_time_s,
                        'arr_time_s': arr_time_s,
                        'home_x': hx, 'home_y': hy, 'home_link': hw_links['home_link'],
                        'work_x': wx, 'work_y': wy, 'work_link': hw_links['work_link'],
                    })
                    if not args.all:
                        # reset so we don't add more for this person
                        last_home_dep.pop(person, None)
                    else:
                        # keep looking for more cycles; clear this one to find the next
                        last_home_dep.pop(person, None)
            elem.clear()

    # Write output
    with open(args.out, 'w', newline='', encoding='utf-8') as outf:
        writer = csv.DictWriter(outf, fieldnames=['person_id','dep_time_s','arr_time_s','home_x','home_y','home_link','work_x','work_y','work_link'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f'Wrote {len(results)} realized commute rows to {args.out}')

if __name__ == '__main__':
    main()
