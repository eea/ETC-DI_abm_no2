# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 16:22:43 2025

@author: jetschny
"""

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

import os, gzip, xml.etree.ElementTree as ET
from collections import defaultdict, deque
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

plt.close('all')


# ---------- CONFIG ----------
base_folder = "Z:/Environment and Health/Air Quality/abm_no2/"
network_path = 'berlin_data_matsim/berlin-v6.4.output_network.xml.gz'
events_path  = 'berlin_data_matsim/berlin-v6.4.output_events.xml.gz'
plans_path   = 'berlin_data_matsim/berlin-v6.4.output_plans.xml.gz'
output_folder = "berlin_output/"

N_AGENTS   = 20       # sample size
DT_SECONDS = 10       # interpolation step
RANDOM_SEED = 42

plot_figures=False

# ---------- HELPERS ----------
def _open(path):
    return gzip.open(os.path.join(base_folder, path), "rb") if path.endswith(".gz") else open(os.path.join(base_folder, path), "rb")

def _iterparse(path, tags=None):
    def _local(tag): 
        return tag.split('}', 1)[-1]  # strip "{namespace}"
    with _open(path) as f:
        ctx = ET.iterparse(f, events=("start", "end"))
        _, root = next(ctx)
        tags_set = set(tags) if tags else None
        for ev, el in ctx:
            local = _local(el.tag)
            if tags_set is None or local in tags_set:
                el.tag = local
                yield ev, el
            if ev == "end":
                el.clear()  # free memory safely (works with ElementTree)
        root.clear()

# ---------- 1) NETWORK: nodes + links ----------
def load_network_nodes_links(network_path):
    nodes = {}
    links = {}
    # First pass: nodes
    for ev, el in _iterparse(network_path, tags={"node","link"}):
        if ev=="end" and el.tag=="node":
            nid = el.attrib["id"]
            x = float(el.attrib["x"]); y = float(el.attrib["y"])
            nodes[nid] = (x, y)
        elif ev=="end" and el.tag=="link":
            lid = el.attrib["id"]
            from_id = el.attrib["from"]; to_id = el.attrib["to"]
            length = float(el.attrib.get("length", "0"))
            freespeed = float(el.attrib.get("freespeed", "0"))
            links[lid] = {
                "from": from_id,
                "to": to_id,
                "length": length,
                "freespeed": freespeed
            }
    # Attach coords
    for lid, d in links.items():
        fx, fy = nodes[d["from"]]
        tx, ty = nodes[d["to"]]
        d["from_xy"] = (fx, fy)
        d["to_xy"] = (tx, ty)
    return nodes, links

t0 = time.time()
nodes, links = load_network_nodes_links(network_path)
print(f"Network loaded in {time.time()-t0:.1f} s")
print("Network:", len(nodes), "nodes;", len(links), "links")

# ---------- 2) PLANS: home/work coords + legs sequence metadata ----------
def coerce_time(t):
    if t is None: return None
    if t.endswith(":"):
        t = t[:-1]
    if ":" in t:
        # "HH:MM:SS" or "DD:HH:MM:SS" sometimes; we reduce to seconds
        parts = t.split(":")
        parts = [float(p) for p in parts]
        # If DD:HH:MM:SS:
        if len(parts)==4:
            dd, hh, mm, ss = parts
            return dd*24*3600 + hh*3600 + mm*60 + ss
        elif len(parts)==3:
            hh, mm, ss = parts
            return hh*3600 + mm*60 + ss
        elif len(parts)==2:
            hh, mm = parts
            return hh*3600 + mm*60
    try:
        return float(t)
    except:
        return None

def extract_home_work_and_trip_templates(plans_path, sample_n=N_AGENTS, rng_seed=RANDOM_SEED):
    """
    Returns:
      agents_info: dict person_id -> {"home":(x,y), "work":(x,y)}
      commute_templates: dict person_id -> {"outbound": {"dep":sec,"arr":sec},
                                            "return":   {"dep":sec,"arr":sec},
                                            "modes": [mode sequence of selected legs]}
    """
    rng = np.random.default_rng(rng_seed)

    # --- helper filters ---
    def is_home_type(t):
        t = (t or "").lower()
        return "home" in t

    def is_work_type(t):
        t = (t or "").lower()
        return any(w in t for w in ("work", "business", "edu"))
    
    # First pass: collect candidates with both home & work
    home_work = {}
    legs_meta = {}

    current_person = None
    acts = []
    legs = []

    for ev, el in _iterparse(plans_path, tags={"person","plan","act","activity","leg"}):
       
        if ev == "start" and el.tag == "person":
            current_person = el.attrib["id"]
            acts, legs = [], []
            skip_person = False  # flag to skip unselected plans
            
        if ev == "start" and el.tag == "plan":
            # If plan is not selected, skip parsing acts/legs for this person
            if el.attrib.get("selected") not in ("yes", "true", "1"):
                skip_person = True
            else:
                skip_person = False
            
        elif ev == "end" and el.tag in ("act", "activity") and not skip_person and current_person is not None:
            # act attrs: type, x,y, link, end_time, start_time
            typ = el.attrib.get("type","")
            x = el.attrib.get("x"); y = el.attrib.get("y"); link = el.attrib.get("link")
            end_t = coerce_time(el.attrib.get("end_time"))
            start_t = coerce_time(el.attrib.get("start_time"))
            if (x is None or y is None) and link is not None and link in links:
                # use link's from-node as proxy
                x, y = links[link]["from_xy"]
            act = {"type":typ, "x": float(x) if x else None, "y": float(y) if y else None,
                   "link":link, "end":end_t, "start":start_t}
            acts.append(act)
            
        elif ev=="end" and el.tag=="leg" and current_person is not None:
            mode = el.attrib.get("mode","")
            dep_t = coerce_time(el.attrib.get("dep_time"))
            trav_t = coerce_time(el.attrib.get("trav_time"))
            legs.append({"mode":mode, "dep":dep_t, "trav":trav_t})
            
        elif ev=="end" and el.tag=="person":
            # finalize
            # Find first 'home' and any 'work'
            # home_candidates = [a for a in acts if a["type"].startswith("home") and a["x"] is not None]
            # work_candidates = [a for a in acts if a["type"].startswith("work") and a["x"] is not None]
            home_candidates = [a for a in acts if is_home_type(a["type"])]
            work_candidates = [a for a in acts if is_work_type(a["type"])]
            if home_candidates and work_candidates:
                home = home_candidates[0]
                # take the first work location encountered
                work = work_candidates[0]
                home_work[current_person] = {"home":(home["x"], home["y"]), "work":(work["x"], work["y"])}
                # Build a rough leg time index by alternating acts and legs (typical MATSim plan structure)
                # Identify the first leg that starts from a home act and goes to a work act (outbound),
                # and the last leg that goes from work to home (return).
                # Simple heuristic: use end/start times on acts to match legs timing.
                # Construct a timeline: Act0 (end) -> Leg0 (dep=Act0.end or leg.dep) -> Act1 (start/ end) ...
                timeline = []
                L = min(len(legs), len(acts)-1)
                for i in range(L):
                    a0, lg, a1 = acts[i], legs[i], acts[i+1]
                    dep = lg["dep"] if lg["dep"] is not None else a0["end"]
                    arr = (dep + lg["trav"]) if (dep is not None and lg["trav"] is not None) else a1["start"]
                    timeline.append({"from_act":a0, "leg":lg, "to_act":a1, "dep":dep, "arr":arr})
                # outbound
                out = next((t for t in timeline if is_home_type(t["from_act"]["type"])
                            and is_work_type(t["to_act"]["type"])), None)
                
                # return (take last occurrence)
                ret_candidates = [t for t in timeline
                          if is_work_type(t["from_act"]["type"])
                          and is_home_type(t["to_act"]["type"])]
               
                ret = ret_candidates[-1] if ret_candidates else None
                commute = {}
                if out:
                    dep = out["dep"] or 7*3600
                    arr = out["arr"] or 8*3600
                    commute["outbound"] = {"dep": dep, "arr": arr, "mode": out["leg"]["mode"]}

                if ret:
                    dep = ret["dep"] or 17*3600
                    arr = ret["arr"] or 18*3600
                    commute["return"] = {"dep": dep, "arr": arr, "mode": ret["leg"]["mode"]}
                if commute:
                    legs_meta[current_person] = commute
            # reset
            current_person, acts, legs = None, [], []
            

    
    
    print("Persons parsed (with any acts):", len(home_work))
    print("Commute candidates (with timing):", len(legs_meta))
    if len(home_work) == 0:
        print("⚠️  No home/work pairs found — check activity naming or file content.")
    # sample persons that have both directions if possible, else at least outbound
    # eligible = [pid for pid in legs_meta.keys()]
    eligible = list(legs_meta.keys())  # keep everyone, no sampling
    # if len(eligible) > sample_n:
        # eligible = list(rng.choice(eligible, size=sample_n, replace=False))
    # agents_info = {pid: home_work[pid] for pid in eligible if pid in home_work}
    # Keep commute templates for chosen
    # commute_templates = {pid: legs_meta[pid] for pid in eligible if pid in legs_meta}

    agents_info = {pid: home_work[pid] for pid in eligible if pid in home_work}
    commute_templates = {pid: legs_meta[pid] for pid in eligible if pid in legs_meta}

    return agents_info, commute_templates

t0 = time.time()
agents_info, commute_templates = extract_home_work_and_trip_templates(plans_path)
print(f"Plans parsed in {time.time()-t0:.1f} s")
print(f"Selected {len(agents_info)} agents with home/work; {len(commute_templates)} with commute timing.")

def build_commute_summary_df(agents_info, commute_templates):
    """
    Combine home/work coordinates and commute timing info into a flat DataFrame.
    Columns:
      person_id, home_x, home_y, work_x, work_y,
      dep_home, arr_work, dep_work, arr_home, mode
    """
    records = []

    for pid, info in agents_info.items():
        home_x, home_y = info.get("home", (None, None))
        work_x, work_y = info.get("work", (None, None))

        commute = commute_templates.get(pid, {})
        out = commute.get("outbound", {})
        ret = commute.get("return", {})

        rec = {
            "agent_id": pid,
            "home_x": home_x,
            "home_y": home_y,
            "work_x": work_x,
            "work_y": work_y,
            "departure_home": out.get("dep"),
            "arrival_work": out.get("arr"),
            "departure_work": ret.get("dep"),
            "arrival_home": ret.get("arr"),
            "transport_mode": out.get("mode") or ret.get("mode"),
        }
        records.append(rec)

    df_summary = pd.DataFrame.from_records(records, columns=[
        "agent_id",
        "home_x", "home_y",
        "work_x", "work_y",
        "departure_home", "arrival_work",
        "departure_work", "arrival_home",
        "transport_mode",
    ])

    return df_summary

t0 = time.time()
commute_summary_df = build_commute_summary_df(agents_info, commute_templates)
print(f"Commute summary built in {time.time()-t0:.1f} s")

def seconds_to_hhmm(s):
    if pd.isna(s): return ""
    h = int(s // 3600); m = int((s % 3600) // 60)
    return f"{h:02d}:{m:02d}"
for col in ["departure_home", "arrival_work", "departure_work", "arrival_home"]:
    commute_summary_df[col] = commute_summary_df[col].apply(seconds_to_hhmm)

print(commute_summary_df.head())
commute_summary_df.to_csv(os.path.join(base_folder, output_folder,"commute_summary.csv"), index=False)
print("Save commute summary to file :", base_folder + output_folder + "commute_summary.csv")


# ---------- 3) EVENTS: extract per-link trajectories for chosen agents ----------
def build_link_sequences_for_agents(events_path, person_ids):
    """
    Returns: dict person_id -> list of (time, event_type, link_id, vehicle_id)
    We only store LinkEnter/LinkLeave for precision; for teleported legs we keep PersonDeparture/Arrival.
    """
    out = defaultdict(list)
    keep = set(person_ids)
    for ev, el in _iterparse(events_path, tags={"event"}):
        if ev != "end": 
            continue
        typ = el.attrib.get("type", "")
        t   = float(el.attrib.get("time", "nan"))
        pid = el.attrib.get("person")
        if pid is None:
            # PT vehicle events etc. We sometimes need vehicle->person mapping via PersonEntersVehicle
            pid = None
        if typ in ("LinkEnter", "linkEnter", "entered link"):
            lid = el.attrib.get("link")
            vid = el.attrib.get("vehicle")
            # priority: person (walk/bike) OR infer via PersonEntersVehicle mapping (not implemented for brevity)
            # Most car trips have person=null here; but we’ll rely on departure/arrival windows to filter later.
            out["__link_events__"].append((t, typ, lid, vid))
        elif typ in ("LinkLeave", "linkLeave", "left link"):
            lid = el.attrib.get("link"); vid = el.attrib.get("vehicle")
            out["__link_events__"].append((t, typ, lid, vid))
        elif typ in ("PersonDeparture","person departure"):
            if pid in keep:
                out[pid].append((t, "dep", el.attrib.get("link"), el.attrib.get("vehicle"), el.attrib.get("legMode")))
        elif typ in ("PersonArrival","person arrival"):
            if pid in keep:
                out[pid].append((t, "arr", el.attrib.get("link"), el.attrib.get("vehicle"), el.attrib.get("legMode")))
        elif typ in ("PersonEntersVehicle",):
            # Map vehicle->person (single-occupant heuristic)
            if pid in keep:
                vid = el.attrib.get("vehicle")
                out.setdefault("__veh2pers__", {}).setdefault(vid, set()).add(pid)
        elif typ in ("PersonLeavesVehicle",):
            pass
    return out

events_raw = build_link_sequences_for_agents(events_path, list(agents_info.keys()))

veh2pers = events_raw.get("__veh2pers__", {})
link_events = sorted(events_raw.get("__link_events__", []))  # global sequence of all link events

# Build a per-person list of commute legs based on templates, then pick relevant link events in [dep,arr]
def reconstruct_commute_trajectories(agents_info, commute_templates, link_events, veh2pers, dt=DT_SECONDS):
    """
    Returns a DataFrame with columns:
      person_id, direction, t, x, y, mode, link_id (optional)
    """
    rows = []
    for pid, hw in agents_info.items():
        home_xy = hw["home"]; work_xy = hw["work"]
        templ = commute_templates.get(pid, {})
        for direction in ("outbound","return"):
            if direction not in templ: 
                continue
            dep = templ[direction]["dep"]; arr = templ[direction]["arr"]; mode = templ[direction]["mode"]
            if dep is None or arr is None or arr <= dep:
                continue

            # Window of interest
            # Collect link events overlapping [dep, arr] and try to identify vehicle used by this person
            # Heuristic: if veh2pers has vehicles used by pid (entered around dep..arr), restrict to those
            candidate_vids = {vid for vid, ppl in veh2pers.items() if pid in ppl} if veh2pers else set()

            segs = []
            last_enter = None
            last_lid = None
            last_t = None
            used_vids = candidate_vids or None

            for t, typ, lid, vid in link_events:
                if t < dep - 60:    # a small buffer
                    continue
                if t > arr + 60:
                    break
                if used_vids is not None and vid not in used_vids:
                    continue
                if typ.lower().startswith("linkenter"):
                    last_enter = (t, lid)
                elif typ.lower().startswith("linkleave"):
                    if last_enter and last_enter[1]==lid:
                        t_enter = max(dep, last_enter[0])
                        t_leave = min(arr, t)
                        if t_leave > t_enter:
                            segs.append((lid, t_enter, t_leave))
                        last_enter = None

            # If we found no link sequence (e.g., teleported leg), fall back to straight segment home->work or reverse
            if not segs:
                if direction=="outbound":
                    start_xy, end_xy = home_xy, work_xy
                else:
                    start_xy, end_xy = work_xy, home_xy
                # straight interpolation
                times = np.arange(dep, arr+1e-9, dt, dtype=float)
                for t in times:
                    alpha = (t - dep) / max(arr - dep, 1e-6)
                    x = start_xy[0]*(1-alpha) + end_xy[0]*alpha
                    y = start_xy[1]*(1-alpha) + end_xy[1]*alpha
                    rows.append((pid, direction, t, x, y, mode, None))
                continue

            # Build time-stepped trajectory along links (linear from 'from' to 'to' node)
            for lid, t0, t1 in segs:
                link = links.get(lid)
                if not link: 
                    continue
                fx, fy = link["from_xy"]; tx, ty = link["to_xy"]
                times = np.arange(t0, t1+1e-9, dt, dtype=float)
                for k, t in enumerate(times):
                    # 0..1 along the link
                    alpha = 0.0 if len(times)==1 else (t - t0) / max(t1 - t0, 1e-6)
                    x = fx*(1-alpha) + tx*alpha
                    y = fy*(1-alpha) + ty*alpha
                    rows.append((pid, direction, t, x, y, mode, lid))

    df = pd.DataFrame(rows, columns=["person_id","direction","t","x","y","mode","link_id"])
    return df

t0 = time.time()
traj = reconstruct_commute_trajectories(agents_info, commute_templates, link_events, veh2pers, dt=DT_SECONDS)
print(f"Trajectories reconstructed in {time.time()-t0:.1f} s")

print("Trajectory rows:", len(traj))
print(traj.head())

# ---------- 4) Quick summaries ----------
# Per-agent vectors (as requested): for each person, (x,y,t) sorted by time
agent_vectors = {pid: df[["x","y","t"]].sort_values("t").to_numpy()
                 for pid, df in traj.groupby("person_id")}

# Save for GIS / post-processing if desired
# out_csv = os.path.join(base_folder, outpout_folder"commute_trajectories_sample10.csv")
traj.to_csv(base_folder + output_folder + "commute_trajectories.csv", index=False)
print("Save trajectories to file :", base_folder + output_folder + "commute_trajectories.csv")

if plot_figures:
    # ---------- 5) Visualizations ----------
    # (A) Static map of trajectories colored by agent
    fig, ax = plt.subplots(figsize=(8,8))
    # plot a light network backdrop (sample a subset to keep it fast)
    skipl = max(1, int(len(links)/5000))  # thin if huge
    for i, (lid, lk) in enumerate(links.items()):
        if (i % skipl)==0:
            fx, fy = lk["from_xy"]; tx, ty = lk["to_xy"]
            ax.plot([fx, tx], [fy, ty], linewidth=0.3, alpha=0.2)
    
    # plot trajectories
    for pid, df in traj.groupby("person_id"):
        ax.plot(df["x"], df["y"], linewidth=1.5, alpha=0.8, label=str(pid))
    ax.set_title("Sampled Commute Trajectories (home↔work)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=6, ncol=2)
    plt.tight_layout()
    plt.show()
    
    # (B) Timeline chart for one agent (position vs. time)
    example_pid = next(iter(agents_info.keys())) if agents_info else None
    if example_pid:
        dfp = traj[traj["person_id"]==example_pid].sort_values("t")
        # distance along path (approx) for a clear line
        # compute progressive distance
        d = [0.0]
        prev = None
        for row in dfp.itertuples():
            if prev is None:
                prev = (row.x, row.y)
                continue
            dx = row.x - prev[0]; dy = row.y - prev[1]
            d.append(d[-1] + math.hypot(dx,dy))
            prev = (row.x, row.y)
        dfp = dfp.assign(s=np.array(d))
        fig2, ax2 = plt.subplots(figsize=(9,3))
        ax2.plot(dfp["t"]/3600.0, dfp["s"])
        ax2.set_xlabel("time [h]"); ax2.set_ylabel("cumulative distance [m-ish]")
        ax2.set_title(f"Commute movement timeline — person {example_pid}")
        plt.tight_layout()
        plt.show()
        
end_time = time.time()
print(f"\nTotal runtime: {(end_time - start_time)/60:.2f} minutes")

