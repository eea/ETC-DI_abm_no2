"""
matsim_commutes.py
Robust, IDE-friendly utilities to extract home→work commutes from MATSim plans + events.

Features:
  - Supports both <act> and <activity> in plans.
  - Flexible 'selected' handling: if missing, treats the first plan as selected.
  - Events: supports 'ActivityEnd'/'ActivityStart' (camel case) and 'actend'/'actstart' (lower case).
  - Attribute variants handled: link/linkId, actType/acttype, start_time/startTime, end_time/endTime.
  - Includes inspectors: inspect_plans_schema(), inspect_events_schema()
  - Progress output to stderr (% of file, MB read, item count, elapsed, ETA);
    switch off with progress=False on the public functions.

Public API:
  extract_planned_commutes(plans_path, network_path=None, all_occurrences=False) -> List[dict]
  extract_realized_commutes(events_path, plans_path=None, network_path=None, all_occurrences=False) -> List[dict]
  save_csv(rows, out_path, fieldnames=None)
  inspect_plans_schema(plans_path, max_people=5) -> dict
  inspect_events_schema(events_path, max_events=10000) -> dict
  diagnose_plans(plans_path) -> dict   # explains why persons have no commute
"""

from typing import Dict, Iterable, List, Optional, Tuple, Set
import gzip
import os
import sys
import time
import xml.etree.ElementTree as ET
import csv

# ---------- helpers ----------

def open_maybe_gzip(path: str):
    """Open as binary stream (iterparse handles the XML encoding itself).
    Binary mode also lets us read the position in the (compressed) file for progress."""
    if path.endswith('.gz'):
        return gzip.open(path, 'rb')
    return open(path, 'rb')


class Progress:
    """Lightweight progress reporter based on bytes read from the file on disk.

    For .gz files the position in the *compressed* file is used, so the percentage
    and ETA are correct without knowing the uncompressed size.
    Prints at most every `interval` seconds to stderr, e.g.:
      [plans] 37.2%  1.21/3.25 GB  persons: 19,431  elapsed 4m12s  ETA 7m05s
    """

    def __init__(self, path: str, stream, label: str, unit: str = 'items',
                 enabled: bool = True, interval: float = 5.0, check_every: int = 2000):
        self.enabled = enabled
        self.label = label
        self.unit = unit
        self.interval = interval
        self.check_every = check_every
        self.raw = getattr(stream, 'fileobj', None) or stream   # underlying file for gzip
        try:
            self.total = os.path.getsize(path)
        except OSError:
            self.total = 0
        self.count = 0
        self.t0 = time.monotonic()
        self.t_last = self.t0

    @staticmethod
    def _fmt_t(s: float) -> str:
        s = int(s)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f'{h}h{m:02d}m{sec:02d}s' if h else f'{m}m{sec:02d}s'

    @staticmethod
    def _fmt_b(b: float) -> str:
        for u in ('B', 'KB', 'MB', 'GB'):
            if b < 1024 or u == 'GB':
                return f'{b:.2f} {u}' if u != 'B' else f'{int(b)} B'
            b /= 1024
        return f'{b:.2f} GB'

    def _print(self, final: bool = False) -> None:
        now = time.monotonic()
        elapsed = now - self.t0
        try:
            pos = self.raw.tell()
        except (OSError, ValueError, AttributeError):
            pos = 0
        if final:
            pos = self.total or pos
        msg = f'[{self.label}] '
        if self.total:
            frac = min(pos / self.total, 1.0)
            msg += f'{frac*100:5.1f}%  {self._fmt_b(pos)}/{self._fmt_b(self.total)}  '
            eta = (elapsed / frac - elapsed) if frac > 0 else 0
        else:
            eta = 0
        msg += f'{self.unit}: {self.count:,}  elapsed {self._fmt_t(elapsed)}'
        if final:
            msg += '  done'
        elif eta:
            msg += f'  ETA {self._fmt_t(eta)}'
        print(msg, file=sys.stderr, flush=True)
        self.t_last = now

    def tick(self, n: int = 1) -> None:
        """Call once per processed item (person, event, link ...). Cheap."""
        self.count += n
        if self.enabled and self.count % self.check_every == 0:
            if time.monotonic() - self.t_last >= self.interval:
                self._print()

    def done(self) -> None:
        if self.enabled:
            self._print(final=True)

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

def is_stage(act_type: Optional[str]) -> bool:
    """MATSim stage/interaction activities (routing helpers, not real activities).
    Catches every '... interaction' variant (pt, car, drt, ride, bike, freight, 'car_interaction', ...)."""
    if not act_type:
        return False
    low = act_type.lower().strip()
    return low.endswith('interaction') or low in STAGE_ACTIVITY_ALIASES or normalize_type(low) in STAGE_ACTIVITY_ALIASES

# ---------- network parsing ----------

def build_network_link_xy(network_path: Optional[str], progress: bool = True) -> Dict[str, Tuple[float, float]]:
    if not network_path:
        return {}
    link_xy: Dict[str, Tuple[float, float]] = {}
    nodes: Dict[str, Tuple[float, float]] = {}
    with open_maybe_gzip(network_path) as f:
        prog = Progress(network_path, f, 'network', unit='nodes+links', enabled=progress)
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
                prog.tick()
        prog.done()
    return link_xy

# ---------- plans parsing ----------

def activities_and_legs(plan_elem) -> Iterable[Tuple[str, dict]]:
    """Yield ('act', attrs) / ('leg', attrs). For legs, two extra keys are added from child elements:
       '_routingMode'    <attributes><attribute name="routingMode">pt</attribute></attributes>
       '_route_distance' <route distance="1234.5" .../>
    """
    for child in plan_elem:
        tag = local_tag(child)
        if tag in ('act', 'activity'):
            yield ('act', child.attrib.copy())
        elif tag == 'leg':
            attrs = child.attrib.copy()
            for sub in child:
                st = local_tag(sub)
                if st == 'attributes':
                    for a in sub:
                        if local_tag(a) == 'attribute' and a.get('name') == 'routingMode':
                            attrs['_routingMode'] = (a.text or '').strip()
                elif st == 'route':
                    d = sub.get('distance')
                    if d not in (None, '', 'NaN'):
                        attrs['_route_distance'] = d
                    rt = sub.get('trav_time') or sub.get('travTime')
                    if rt not in (None, '', 'undefined'):
                        attrs['_route_trav_time'] = rt
            if '_routingMode' not in attrs and child.get('routingMode'):
                attrs['_routingMode'] = child.get('routingMode')
            yield ('leg', attrs)

WALK_MODES: Set[str] = {'walk', 'non_network_walk', 'transit_walk', 'access_walk', 'egress_walk'}

def _leg_time(leg: dict, key_snake: str, key_camel: str) -> Optional[int]:
    v = parse_time_to_seconds(leg.get(key_snake) or leg.get(key_camel))
    if v is None and key_snake == 'trav_time':
        v = parse_time_to_seconds(leg.get('_route_trav_time'))   # MATSim often stores it on <route>
    return v

def summarize_trip(legs: List[dict], dep_fallback: Optional[int]) -> dict:
    """Main mode, leg modes, arrival time, distance for the legs between home and work."""
    leg_modes = [l.get('mode') or '' for l in legs]
    # main mode: routingMode of the last leg (the trip that arrives at work);
    # fallback: first non-walk leg mode, else 'walk'
    mode = None
    for l in reversed(legs):
        if l.get('_routingMode'):
            mode = l['_routingMode']
            break
    if mode is None:
        non_walk = [m for m in leg_modes if m and m not in WALK_MODES]
        mode = non_walk[-1] if non_walk else (leg_modes[-1] if leg_modes else None)
    # departure: first leg dep_time (fallback when home has no end_time)
    dep_leg = _leg_time(legs[0], 'dep_time', 'depTime') if legs else None
    # arrival: last leg dep_time + trav_time
    arr = None
    if legs:
        d = _leg_time(legs[-1], 'dep_time', 'depTime')
        t = _leg_time(legs[-1], 'trav_time', 'travTime')
        if d is not None and t is not None:
            arr = d + t
    if arr is None and legs:
        # fallback: departure + sum of all leg travel times (stops/stage acts assumed 0 s)
        start = dep_fallback if dep_fallback is not None else dep_leg
        tts = [_leg_time(l, 'trav_time', 'travTime') for l in legs]
        if start is not None and all(x is not None for x in tts):
            arr = start + sum(tts)
    dist = 0.0; dist_ok = False
    for l in legs:
        try:
            dist += float(l['_route_distance']); dist_ok = True
        except (KeyError, ValueError):
            pass
    return {'mode': mode, 'leg_modes': '-'.join(leg_modes), 'arr': arr, 'dep_leg': dep_leg,
            'route_distance_m': round(dist, 1) if dist_ok else None}

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

def extract_commute_from_plan(plan_elem, link_xy: Dict[str, Tuple[float,float]], all_occurrences: bool=False,
                              allow_intermediate: bool=False):
    """
    Find home→work by allowing any number of intermediate legs and *stage* activities.
    Pattern: act(home) → [legs/acts(stage)]* → act(work)
    Record the departure time from the home act's end_time and arrival time at the work act's start_time.

    allow_intermediate=False: only stage activities ('pt interaction' etc.) may lie between home and work.
    allow_intermediate=True : any non-home activities may lie in between (home -> shop/education -> work
                              is counted as a commute; the commute then runs from the last home act).
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
                        legs = [seq[k3][1] for k3 in range(i + 1, j) if seq[k3][0] == 'leg']
                        stops = [seq[k3][1].get('type', '') for k3 in range(i + 1, j)
                                 if seq[k3][0] == 'act' and not is_stage(seq[k3][1].get('type', ''))]
                        dep = parse_time_to_seconds(home_attrs.get('end_time') or home_attrs.get('endTime'))
                        trip = summarize_trip(legs, dep)
                        if dep is None:
                            dep = trip['dep_leg']
                        arr = parse_time_to_seconds(obj2.get('start_time') or obj2.get('startTime'))
                        if arr is None:
                            arr = trip['arr']
                        results.append({
                            'home_x': hx, 'home_y': hy, 'home_link': hlink,
                            'work_x': wx, 'work_y': wy, 'work_link': wlink,
                            'dep_time_s': dep, 'arr_time_s': arr,
                            'travel_time_s': (arr - dep) if (arr is not None and dep is not None) else None,
                            'mode': trip['mode'],
                            'leg_modes': trip['leg_modes'],
                            'route_distance_m': trip['route_distance_m'],
                            'stops': '|'.join(stops),
                        })
                        if not all_occurrences:
                            return results
                        # continue scanning after this work act
                        i = j
                        break
                    t2 = obj2.get('type', '')
                    if is_home(t2):
                        # back home before reaching work → restart from this home act
                        break
                    if is_stage(t2):
                        pass   # routing helper → keep scanning
                    elif not allow_intermediate:
                        # Non-stage and not work → stop this attempt (likely a different day-pattern)
                        break
                j += 1
        i += 1
    return results


def iter_selected_plans(plans_path: str, progress: bool = True, with_person: bool = False):
    """Yield (person_id, selected_plan_elem). If no plan is marked selected, take the first plan.

    Plans are only yielded at the 'end' event of <person>, i.e. once the whole person
    subtree (all plans with all acts/legs) has been parsed completely. Child elements
    are NOT cleared individually (Element.clear() also wipes attributes); instead the
    whole person is cleared after the consumer is done with it.
    """
    with open_maybe_gzip(plans_path) as f:
        prog = Progress(plans_path, f, 'plans', unit='persons', enabled=progress)
        context = ET.iterparse(f, events=('start', 'end'))
        root = None
        person_id = None
        selected_plan = None
        first_plan = None
        for ev, elem in context:
            tag = local_tag(elem)
            if ev == 'start':
                if root is None:
                    root = elem
                if tag == 'person':
                    person_id = elem.get('id')
                    person_elem = elem
                    selected_plan = None
                    first_plan = None
                continue
            # ev == 'end': element and all its children are complete
            if tag == 'plan' and person_id is not None:
                if first_plan is None:
                    first_plan = elem
                if selected_plan is None and (elem.get('selected') or '').lower() in {'yes', 'true', '1'}:
                    selected_plan = elem
            elif tag == 'person':
                plan = selected_plan if selected_plan is not None else first_plan
                if plan is not None and person_id is not None:
                    if with_person:
                        yield person_id, plan, person_elem
                    else:
                        yield person_id, plan
                elif with_person and person_id is not None:
                    yield person_id, None, person_elem
                person_id = None
                selected_plan = None
                first_plan = None
                elem.clear()
                if root is not None:
                    root.clear()   # drop references to already processed persons
                prog.tick()
        prog.done()

# ---------- public API: planned ----------

def extract_planned_commutes(plans_path: str, network_path: Optional[str]=None, all_occurrences: bool=False,
                             progress: bool = True, allow_intermediate: bool = False) -> List[dict]:
    link_xy = build_network_link_xy(network_path, progress=progress) if network_path else {}
    out: List[dict] = []
    for person_id, plan_elem in iter_selected_plans(plans_path, progress=progress):
        commutes = extract_commute_from_plan(plan_elem, link_xy, all_occurrences=all_occurrences,
                                             allow_intermediate=allow_intermediate)
        for c in commutes:
            row = {'person_id': person_id}
            row.update(c)
            out.append(row)
        plan_elem.clear()
    return out

# ---------- public API: realized (events) ----------

def read_home_work_from_plans(plans_path: str, progress: bool = True) -> Dict[str, Dict[str, Tuple[Optional[float], Optional[float], Optional[str]]]]:
    def best_xy_local(attrs):
        x = attrs.get('x'); y = attrs.get('y'); link = attrs.get('link') or attrs.get('linkId')
        try:
            if x is not None and y is not None:
                return float(x), float(y), link
        except ValueError:
            pass
        return None, None, link
    
    result: Dict[str, Dict[str, Tuple[Optional[float], Optional[float], Optional[str]]]] = {}
    for pid, plan in iter_selected_plans(plans_path, progress=progress):
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

def extract_realized_commutes(events_path: str, plans_path: Optional[str]=None, network_path: Optional[str]=None,
                              all_occurrences: bool=False, progress: bool = True) -> List[dict]:
    link_xy = build_network_link_xy(network_path, progress=progress) if network_path else {}
    home_work_cache = read_home_work_from_plans(plans_path, progress=progress) if plans_path else {}
    last_home_dep: Dict[str, Tuple[Optional[int], Optional[str]]] = {}
    results: List[dict] = []

    with open_maybe_gzip(events_path) as f:
        prog = Progress(events_path, f, 'events', unit='events', enabled=progress, check_every=20000)
        context = ET.iterparse(f, events=('start', 'end'))
        root = None
        for ev, elem in context:
            if ev == 'start':
                if root is None:
                    root = elem
                continue
            if local_tag(elem) != 'event':
                elem.clear()
                continue
            prog.tick()
            if prog.count % 100000 == 0 and root is not None:
                root.clear()   # drop the (already cleared) event shells kept by the root
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
        prog.done()
    return results

# ---------- diagnostics: why does a person have no commute? ----------

def peek_xml(path: str, max_elems: int = 200) -> dict:
    """Return the root tag and the first few distinct element tags of an XML file (quick sanity check)."""
    from collections import Counter
    tags = Counter()
    root = None
    with open_maybe_gzip(path) as f:
        for i, (ev, elem) in enumerate(ET.iterparse(f, events=('start',))):
            t = local_tag(elem)
            if root is None:
                root = t
            tags[t] += 1
            if i >= max_elems:
                break
    guess = {'population': 'plans', 'events': 'events', 'network': 'network',
             'facilities': 'facilities', 'vehicleDefinitions': 'vehicles',
             'transitSchedule': 'transit schedule', 'counts': 'counts'}.get(root, 'unknown')
    return {'root': root, 'looks_like': guess, 'first_tags': dict(tags.most_common())}

def _person_attributes(person_elem) -> Dict[str, str]:
    """Read <attributes><attribute name=..>value</attribute></attributes> directly under <person>."""
    out: Dict[str, str] = {}
    if person_elem is None:
        return out
    for child in person_elem:
        if local_tag(child) == 'attributes':
            for a in child:
                if local_tag(a) == 'attribute' and a.get('name'):
                    out[a.get('name')] = (a.text or '').strip()
    return out

def diagnose_plans(plans_path: str, progress: bool = True, top: int = 20, verbose: bool = True) -> dict:
    """Classify every person in the plans file and explain why no commute is found.

    Categories (each person in exactly one):
      commute_strict        home -> [stage acts] -> work found (what extract_planned_commutes finds)
      commute_only_loose    home -> other act(s) -> work, e.g. home->education->work (found with allow_intermediate=True)
      work_without_home_before  has work, but no home act before it (e.g. plan starts at work/other)
      no_work_act           selected plan has no work-type activity at all (students, retirees, ...)
      no_home_act           has work but no home-type activity anywhere
      no_plan               person without any plan
    Also reports: subpopulations, unrecognised activity types, and which activity types break the
    home->work chain.
    """
    from collections import Counter
    cats = Counter()
    subpop = Counter()
    subpop_cat: Dict[str, Counter] = {}
    act_types = Counter()          # raw activity types (persons having it)
    breakers = Counter()           # activity type between home and work in 'loose' cases
    chains = Counter()             # example act-type chains of persons without work
    for pid, plan, person in iter_selected_plans(plans_path, progress=progress, with_person=True):
        attrs = _person_attributes(person)
        sp = attrs.get('subpopulation', '(none)')
        subpop[sp] += 1
        if plan is None:
            cat = 'no_plan'
        else:
            acts = [a.get('type', '') for k, a in activities_and_legs(plan) if k == 'act']
            for t in set(acts):
                act_types[t] += 1
            real = [t for t in acts if not is_stage(t)]
            has_w = any(is_work(t) for t in real)
            has_h = any(is_home(t) for t in real)
            if extract_commute_from_plan(plan, {}, allow_intermediate=False):
                cat = 'commute_strict'
            elif extract_commute_from_plan(plan, {}, allow_intermediate=True):
                cat = 'commute_only_loose'
                # record the activities between a home act and the next work act
                # (same pairing as extract_commute_from_plan; work may also occur before the first home)
                last_home = None
                for i, t in enumerate(real):
                    if is_home(t):
                        last_home = i
                    elif is_work(t) and last_home is not None:
                        for t2 in real[last_home + 1:i]:
                            breakers[t2] += 1
                        break
            elif not has_w:
                cat = 'no_work_act'
                chains[' > '.join(real[:6]) + (' ...' if len(real) > 6 else '')] += 1
            elif not has_h:
                cat = 'no_home_act'
            else:
                cat = 'work_without_home_before'
        cats[cat] += 1
        subpop_cat.setdefault(sp, Counter())[cat] += 1

    total = sum(cats.values())
    if total == 0:
        info = peek_xml(plans_path)
        print(f"\nNo <person> elements found in {plans_path}.\n"
              f"  root element: <{info['root']}>  -> this looks like a {info['looks_like']} file, not a plans file.\n"
              f"  first tags:   {info['first_tags']}\n"
              f"  Pass the population file instead, e.g. '<run>.output_plans.xml.gz' "
              f"(root element <population>).", file=sys.stderr)
        return {'persons': 0, 'file_check': info}
    unrecognised = {t: n for t, n in act_types.items()
                    if not (is_home(t) or is_work(t) or is_stage(t))}
    report = {
        'persons': total,
        'categories': dict(cats.most_common()),
        'subpopulations': dict(subpop.most_common()),
        'categories_by_subpopulation': {k: dict(v.most_common()) for k, v in subpop_cat.items()},
        'activity_types': dict(act_types.most_common()),
        'non_home_work_stage_types': dict(sorted(unrecognised.items(), key=lambda kv: -kv[1])[:top]),
        'chain_breakers_between_home_and_work': dict(breakers.most_common(top)),
        'top_chains_without_work': dict(chains.most_common(top)),
    }
    if verbose:
        def pct(n): return f'{n:>9,}  ({100*n/total:5.1f}%)' if total else str(n)
        print(f'\nPersons: {total:,}')
        print('\nCategories:')
        for k, v in cats.most_common():
            print(f'  {k:28s}{pct(v)}')
        print('\nSubpopulations:')
        for k, v in subpop.most_common():
            print(f'  {k:28s}{pct(v)}   ' + ', '.join(f'{c}={n:,}' for c, n in subpop_cat[k].most_common()))
        print('\nActivity types (persons having it):')
        for k, v in act_types.most_common(top):
            flag = 'home' if is_home(k) else 'work' if is_work(k) else 'stage' if is_stage(k) else ''
            print(f'  {k:28s}{v:>9,}  {flag}')
        if breakers:
            print('\nActivities between home and work (cases found only with allow_intermediate=True):')
            for k, v in breakers.most_common(top):
                print(f'  {k:28s}{v:>9,}')
        if chains:
            print('\nMost common day chains of persons WITHOUT a work activity:')
            for k, v in chains.most_common(10):
                print(f'  {v:>9,}  {k}')
    return report

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
