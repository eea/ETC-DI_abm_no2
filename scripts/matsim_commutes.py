"""
matsim_commutes.py
Robust, IDE-friendly utilities to extract home→work commutes from MATSim plans + events.

Features:
  - Supports both <act> and <activity> in plans.
  - Flexible 'selected' handling: if missing, treats the first plan as selected.
  - Events: supports 'ActivityEnd'/'ActivityStart' (camel case) and 'actend'/'actstart' (lower case).
  - Attribute variants handled: link/linkId, actType/acttype, start_time/startTime, end_time/endTime.
  - Includes inspectors: inspect_plans_schema(), inspect_events_schema()

Public API:
  extract_planned_commutes(plans_path, network_path=None, all_occurrences=False) -> List[dict]
  extract_realized_commutes(events_path, plans_path=None, network_path=None, all_occurrences=False) -> List[dict]
  save_csv(rows, out_path, fieldnames=None)
  inspect_plans_schema(plans_path, max_people=5) -> dict
  inspect_events_schema(events_path, max_events=10000) -> dict
"""

from typing import Dict, Iterable, List, Optional, Tuple, Set
import gzip
import xml.etree.ElementTree as ET
import csv

# ---------- helpers ----------

def open_maybe_gzip(path: str):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'rt', encoding='utf-8')

def parse_time_to_seconds(t: Optional[str]) -> Optional[int]:
    if t is None or t == '':
        return None
    if t.isdigit():
        return int(t)
    parts = t.split(':')
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    try:
        return int(float(t))
    except ValueError:
        return None

def local_tag(elem) -> str:
    return elem.tag.split('}')[-1]

def normalize_type(t: Optional[str]) -> str:
    if not t:
        return ''
    t_low = t.lower()
    for sep in ['_', ':', '.']:
        if sep in t_low:
            t_low = t_low.split(sep, 1)[0]
    return t_low.strip()

HOME_ALIASES: Set[str] = {'home', 'h'}
WORK_ALIASES: Set[str] = {'work', 'w', 'office', 'job'}
STAGE_ACTIVITY_ALIASES: Set[str] = {
    'pt interaction',
    'car interaction',
    'bike interaction',
    'walk interaction',
    'transfer',
    'access',
    'egress'
}

def is_home(act_type: Optional[str]) -> bool:
    return normalize_type(act_type) in HOME_ALIASES

def is_work(act_type: Optional[str]) -> bool:
    return normalize_type(act_type) in WORK_ALIASES

# ---------- network parsing ----------

def build_network_link_xy(network_path: Optional[str]) -> Dict[str, Tuple[float, float]]:
    if not network_path:
        return {}
    link_xy: Dict[str, Tuple[float, float]] = {}
    nodes: Dict[str, Tuple[float, float]] = {}
    with open_maybe_gzip(network_path) as f:
        context = ET.iterparse(f, events=('start', 'end'))
        in_nodes = False
        in_links = False
        for ev, elem in context:
            tag = local_tag(elem)
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
                if lid and from_id in nodes and to_id in nodes:
                    fx, fy = nodes[from_id]; tx, ty = nodes[to_id]
                    link_xy[lid] = ((fx + tx)/2.0, (fy + ty)/2.0)
                elem.clear()
            if ev == 'end' and tag in {'node', 'link'}:
                elem.clear()
    return link_xy

# ---------- plans parsing ----------

def activities_and_legs(plan_elem) -> Iterable[Tuple[str, dict]]:
    for child in plan_elem:
        tag = local_tag(child)
        if tag in ('act', 'activity'):
            yield ('act', child.attrib.copy())
        elif tag == 'leg':
            yield ('leg', child.attrib.copy())

def best_xy(attrs: dict, link_xy: Dict[str, Tuple[float, float]]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    x = attrs.get('x'); y = attrs.get('y'); link = attrs.get('link') or attrs.get('linkId')
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
    """
    Find home→work by allowing any number of intermediate legs and *stage* activities.
    Pattern: act(home) → [legs/acts(stage)]* → act(work)
    Record the departure time from the home act's end_time and arrival time at the work act's start_time.
    """
    seq = list(activities_and_legs(plan_elem))
    results = []
    i = 0
    n = len(seq)
    while i < n:
        kind, a = seq[i]
        if kind == 'act' and is_home(a.get('type', '')):
            # Found a home act; remember its attrs (for dep time & coords)
            home_attrs = a
            # Scan forward to the next *work* act, skipping stage activities
            j = i + 1
            while j < n:
                k2, obj2 = seq[j]
                if k2 == 'act':
                    # If it's a primary "work" act → we have a commute
                    if is_work(obj2.get('type', '')):
                        hx, hy, hlink = best_xy(home_attrs, link_xy)
                        wx, wy, wlink = best_xy(obj2, link_xy)
                        dep = parse_time_to_seconds(home_attrs.get('end_time') or home_attrs.get('endTime'))
                        arr = parse_time_to_seconds(obj2.get('start_time') or obj2.get('startTime'))
                        # We *try* to grab the mode from the first leg after home, if present
                        mode = None
                        # look ahead from i+1 to find the first leg
                        for k3 in range(i+1, min(j, n)):
                            if seq[k3][0] == 'leg':
                                mode = seq[k3][1].get('mode')
                                break
                        results.append({
                            'home_x': hx, 'home_y': hy, 'home_link': hlink,
                            'work_x': wx, 'work_y': wy, 'work_link': wlink,
                            'dep_time_s': dep, 'arr_time_s': arr, 'mode': mode
                        })
                        if not all_occurrences:
                            return results
                        # continue scanning after this work act
                        i = j
                        break
                    # If it's a stage activity → skip and keep scanning
                    tnorm = normalize_type(obj2.get('type', ''))
                    if tnorm not in STAGE_ACTIVITY_ALIASES:
                        # Non-stage and not work → stop this attempt (likely a different day-pattern)
                        break
                j += 1
        i += 1
    return results


def iter_selected_plans(plans_path: str):
    """Yield (person_id, selected_plan_elem). If no plan marked selected, take first plan."""
    with open_maybe_gzip(plans_path) as f:
        context = ET.iterparse(f, events=('start','end'))
        person_id = None
        in_person = False
        emitted_for_person = False
        first_plan_elem = None
        for ev, elem in context:
            tag = local_tag(elem)
            if ev=='start' and tag=='person':
                person_id = elem.get('id')
                in_person = True
                emitted_for_person = False
                first_plan_elem = None
            elif ev=='end' and tag=='person':
                if not emitted_for_person and first_plan_elem is not None and person_id is not None:
                    yield person_id, first_plan_elem
                person_id = None
                in_person = False
                emitted_for_person = False
                first_plan_elem = None
                elem.clear()
            elif ev=='start' and tag=='plan':
                selected_attr = (elem.get('selected') or '').lower()
                selected = selected_attr in {'yes','true','1'}
                if first_plan_elem is None:
                    first_plan_elem = elem
                if selected and in_person and person_id is not None and not emitted_for_person:
                    yield person_id, elem
                    emitted_for_person = True
            elif ev=='end' and tag=='plan':
                if elem is not first_plan_elem:
                    elem.clear()
            if ev=='end' and tag in {'act','activity','leg'}:
                elem.clear()

# ---------- public API: planned ----------

def extract_planned_commutes(plans_path: str, network_path: Optional[str]=None, all_occurrences: bool=False) -> List[dict]:
    link_xy = build_network_link_xy(network_path) if network_path else {}
    out: List[dict] = []
    for person_id, plan_elem in iter_selected_plans(plans_path):
        commutes = extract_commute_from_plan(plan_elem, link_xy, all_occurrences=all_occurrences)
        for c in commutes:
            row = {'person_id': person_id}
            row.update(c)
            out.append(row)
        plan_elem.clear()
    return out

# ---------- public API: realized (events) ----------

def read_home_work_from_plans(plans_path: str) -> Dict[str, Dict[str, Tuple[Optional[float], Optional[float], Optional[str]]]]:
    def best_xy_local(attrs):
        x = attrs.get('x'); y = attrs.get('y'); link = attrs.get('link') or attrs.get('linkId')
        try:
            if x is not None and y is not None:
                return float(x), float(y), link
        except ValueError:
            pass
        return None, None, link
    
    result: Dict[str, Dict[str, Tuple[Optional[float], Optional[float], Optional[str]]]] = {}
    for pid, plan in iter_selected_plans(plans_path):
        result.setdefault(pid, {})
        for kind, attrs in activities_and_legs(plan):
            if kind != 'act':
                continue
            a_type = attrs.get('type','')
            if is_home(a_type) and 'home' not in result[pid]:
                result[pid]['home'] = best_xy_local(attrs)
            elif is_work(a_type) and 'work' not in result[pid]:
                result[pid]['work'] = best_xy_local(attrs)
        plan.clear()
    return result

def extract_realized_commutes(events_path: str, plans_path: Optional[str]=None, network_path: Optional[str]=None, all_occurrences: bool=False) -> List[dict]:
    link_xy = build_network_link_xy(network_path) if network_path else {}
    home_work_cache = read_home_work_from_plans(plans_path) if plans_path else {}
    last_home_dep: Dict[str, Tuple[Optional[int], Optional[str]]] = {}
    results: List[dict] = []

    with open_maybe_gzip(events_path) as f:
        context = ET.iterparse(f, events=('end',))
        for ev, elem in context:
            if local_tag(elem) != 'event':
                elem.clear()
                continue
            etype = (elem.get('type') or '')
            person = elem.get('person') or elem.get('agent')
            time_s = parse_time_to_seconds(elem.get('time'))
            actType = elem.get('actType') or elem.get('acttype')
            link = elem.get('link') or elem.get('linkId')

            if person is None:
                elem.clear()
                continue

            if etype.lower() in {'actend','activityend'} and is_home(actType):
                last_home_dep[person] = (time_s, link)

            elif etype.lower() in {'actstart','activitystart'} and is_work(actType):
                if person in last_home_dep:
                    dep_time_s, home_link = last_home_dep[person]
                    arr_time_s = time_s
                    hx = hy = wx = wy = None
                    home_link_out = None
                    work_link_out = None
                    if person in home_work_cache:
                        hx, hy, home_link_out = home_work_cache[person].get('home', (None,None,None))
                        wx, wy, work_link_out = home_work_cache[person].get('work', (None,None,None))
                    if home_link_out is None:
                        home_link_out = home_link
                    if work_link_out is None:
                        work_link_out = link
                    if (hx is None or hy is None) and home_link_out and home_link_out in link_xy:
                        hx, hy = link_xy[home_link_out]
                    if (wx is None or wy is None) and work_link_out and work_link_out in link_xy:
                        wx, wy = link_xy[work_link_out]

                    results.append({
                        'person_id': person,
                        'dep_time_s': dep_time_s,
                        'arr_time_s': arr_time_s,
                        'home_x': hx, 'home_y': hy, 'home_link': home_link_out,
                        'work_x': wx, 'work_y': wy, 'work_link': work_link_out,
                    })
                    last_home_dep.pop(person, None)
            elem.clear()
    return results

# ---------- inspectors ----------

def inspect_plans_schema(plans_path: str, max_people: int = 5) -> dict:
    stats = {'activity_tags': set(), 'has_selected': False, 'people_sample': 0, 'plans_per_person': [], 'sample_types': set()}
    with open_maybe_gzip(plans_path) as f:
        context = ET.iterparse(f, events=('start','end'))
        in_person = False
        plans_count = 0
        for ev, elem in context:
            tag = local_tag(elem)
            if ev=='start' and tag=='person':
                in_person = True
                plans_count = 0
            elif ev=='end' and tag=='person':
                in_person = False
                stats['plans_per_person'].append(plans_count)
                stats['people_sample'] += 1
                elem.clear()
                if stats['people_sample'] >= max_people:
                    break
            elif ev=='start' and tag=='plan' and in_person:
                sel = (elem.get('selected') or '').lower()
                if sel in {'yes','true','1'}:
                    stats['has_selected'] = True
                plans_count += 1
            elif ev=='end' and tag in {'act','activity'}:
                stats['activity_tags'].add(tag)
                t = elem.get('type')
                if t:
                    stats['sample_types'].add(t)
                elem.clear()
            elif ev=='end' and tag=='leg':
                elem.clear()
    stats['activity_tags'] = sorted(stats['activity_tags'])
    stats['sample_types'] = sorted(stats['sample_types'])
    return stats

def inspect_events_schema(events_path: str, max_events: int = 10000) -> dict:
    stats = {'event_types': {}, 'has_person': False, 'has_actType': False, 'has_link': False, 'scanned': 0}
    with open_maybe_gzip(events_path) as f:
        context = ET.iterparse(f, events=('end',))
        for ev, elem in context:
            if local_tag(elem) != 'event':
                elem.clear()
                continue
            etype = elem.get('type') or ''
            stats['event_types'][etype] = stats['event_types'].get(etype, 0) + 1
            if (elem.get('person') or elem.get('agent')) is not None:
                stats['has_person'] = True
            if (elem.get('actType') or elem.get('acttype')) is not None:
                stats['has_actType'] = True
            if (elem.get('link') or elem.get('linkId')) is not None:
                stats['has_link'] = True
            stats['scanned'] += 1
            elem.clear()
            if stats['scanned'] >= max_events:
                break
    return stats

# ---------- CSV writer ----------

def save_csv(rows: List[dict], out_path: str, fieldnames: Optional[List[str]]=None) -> None:
    if not rows:
        if fieldnames:
            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        else:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write('')
        return
    if fieldnames is None:
        keys = list(rows[0].keys())
        for r in rows[1:]:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

if __name__ == '__main__':
    print('This module is intended to be imported and called from your IDE.')
    print('Helpers: extract_planned_commutes, extract_realized_commutes, save_csv,')
    print('          inspect_plans_schema, inspect_events_schema')
