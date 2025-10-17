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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_folder = "Z:/Environment and Health/Air Quality/abm_no2/"
network_path = 'berlin_data_matsim/berlin-v6.4.output_network.xml.gz'
plans_path   = 'berlin_data_matsim/berlin-v6.4.output_plans.xml.gz'
N_AGENTS     = 10
DT_SECONDS   = 15   # coarser = faster

def _open(path):
    path = os.path.join(base_folder, path)
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")

def load_network(network_path):
    nodes, links = {}, {}
    with _open(network_path) as f:
        ctx = ET.iterparse(f, events=("end",))
        for ev, el in ctx:
            if el.tag == "node":
                nodes[el.attrib["id"]] = (float(el.attrib["x"]), float(el.attrib["y"]))
                el.clear()
            elif el.tag == "link":
                lid = el.attrib["id"]; fr = el.attrib["from"]; to = el.attrib["to"]
                length = float(el.attrib.get("length","0"))
                links[lid] = {"from":fr,"to":to,"length":length}
                el.clear()
    # attach coords
    for lid, d in links.items():
        d["from_xy"] = nodes[d["from"]]; d["to_xy"] = nodes[d["to"]]
    return nodes, links

nodes, links = load_network(network_path)

def to_seconds(t):
    if t is None: return None
    s = str(t)
    if ":" in s:
        parts = [float(x) for x in s.split(":")]
        if len(parts)==3:
            h,m,s = parts; return h*3600+m*60+s
        if len(parts)==4:
            d,h,m,s = parts; return d*86400+h*3600+m*60+s
    try: return float(s)
    except: return None

def sample_agents_and_commutes(plans_path, n=N_AGENTS):
    """
    Returns:
      chosen: dict pid -> {"home":(x,y), "work":(x,y)}
      commutes: dict pid -> {"outbound": leg_info, "return": leg_info}
      leg_info: {"dep":sec,"trav":sec,"mode":str,"route_links":[...]}
    """
    rng = np.random.default_rng(42)
    candidates = []
    with _open(plans_path) as f:
        ctx = ET.iterparse(f, events=("start","end"))
        _, root = next(ctx)
        pid = None; acts=[]; legs=[]
        capture_route = False; last_leg = None
        for ev, el in ctx:
            if ev=="start" and el.tag=="person":
                pid = el.attrib["id"]; acts=[]; legs=[]
            elif ev=="end" and el.tag=="act" and pid:
                typ = el.attrib.get("type","")
                x = el.attrib.get("x"); y = el.attrib.get("y"); link = el.attrib.get("link")
                # if act has only link, use link's from-node coordinates
                if (x is None or y is None) and link in links:
                    x,y = links[link]["from_xy"]
                acts.append({"type":typ,
                             "x": float(x) if x else None,
                             "y": float(y) if y else None})
                el.clear()
            elif ev=="start" and el.tag=="leg" and pid:
                dep = to_seconds(el.attrib.get("dep_time"))
                trav = to_seconds(el.attrib.get("trav_time"))
                mode = el.attrib.get("mode","")
                last_leg = {"dep":dep,"trav":trav,"mode":mode,"route_links":[]}
            elif ev=="end" and el.tag=="route" and pid and last_leg is not None:
                # route text is space-separated link ids for type="links"
                txt = (el.text or "").strip()
                if txt:
                    last_leg["route_links"] = txt.split()
                el.clear()
            elif ev=="end" and el.tag=="leg" and pid and last_leg is not None:
                legs.append(last_leg); last_leg = None; el.clear()
            elif ev=="end" and el.tag=="person" and pid:
                # find first home and first work act
                homes = [a for a in acts if a["type"].startswith("home") and a["x"] is not None]
                works = [a for a in acts if a["type"].startswith("work") and a["x"] is not None]
                if homes and works and legs:
                    # build act-leg-act timeline
                    L = min(len(legs), len(acts)-1)
                    timeline = []
                    for i in range(L):
                        timeline.append((acts[i], legs[i], acts[i+1]))
                    # pick home->work earliest and work->home latest
                    out = next(((a0,lg,a1) for (a0,lg,a1) in timeline
                                if a0["type"].startswith("home") and a1["type"].startswith("work")
                                and lg["dep"] is not None and lg["trav"] is not None), None)
                    rets = [t for t in timeline if t[0]["type"].startswith("work") and t[2]["type"].startswith("home")
                            and t[1]["dep"] is not None and t[1]["trav"] is not None]
                    ret = rets[-1] if rets else None
                    if out or ret:
                        candidates.append((
                            pid,
                            {"home":(homes[0]["x"],homes[0]["y"]),
                             "work":(works[0]["x"],works[0]["y"])},
                            {"outbound": {"dep": out[1]["dep"], "trav": out[1]["trav"],
                                          "mode": out[1]["mode"], "route_links": out[1]["route_links"]} if out else None,
                             "return":   {"dep": ret[1]["dep"], "trav": ret[1]["trav"],
                                          "mode": ret[1]["mode"], "route_links": ret[1]["route_links"]} if ret else None}
                        ))
                pid=None; acts=[]; legs=[]; root.clear()
    # sample
    if len(candidates) > n:
        idx = rng.choice(len(candidates), size=n, replace=False)
        chosen_pairs = [candidates[i] for i in idx]
    else:
        chosen_pairs = candidates
    chosen = {pid: info for pid, info, _ in chosen_pairs}
    commutes = {pid: cm for pid, _, cm in chosen_pairs}
    return chosen, commutes

agents_info, commutes = sample_agents_and_commutes(plans_path, N_AGENTS)

def build_traj_from_routes(agents_info, commutes, dt=DT_SECONDS):
    rows = []
    for pid, places in agents_info.items():
        for direction in ("outbound","return"):
            leg = commutes.get(pid, {}).get(direction)
            if not leg or leg["dep"] is None or leg["trav"] is None:
                continue
            dep = leg["dep"]; arr = dep + leg["trav"]
            links_seq = leg.get("route_links") or []
            # if no explicit route, straight-line fallback between home/work
            if not links_seq:
                start_xy = places["home"] if direction=="outbound" else places["work"]
                end_xy   = places["work"] if direction=="outbound" else places["home"]
                times = np.arange(dep, arr+1e-9, dt)
                for t in times:
                    a = (t-dep)/max(arr-dep,1e-6)
                    x = start_xy[0]*(1-a) + end_xy[0]*a
                    y = start_xy[1]*(1-a) + end_xy[1]*a
                    rows.append((pid, direction, t, x, y, leg["mode"], None))
                continue
            # compute cumulative distances along the route
            segs = []
            total = 0.0
            for lid in links_seq:
                lk = links.get(lid)
                if not lk: 
                    continue
                L = lk["length"] if lk["length"] > 0 else 1.0
                segs.append((lid, L, lk["from_xy"], lk["to_xy"]))
                total += L
            if total <= 0:
                continue
            # distribute time proportional to segment length
            t0 = dep
            for lid, L, (fx,fy), (tx,ty) in segs:
                dt_seg = leg["trav"] * (L/total)
                t1 = t0 + dt_seg
                times = np.arange(t0, t1+1e-9, dt)
                for t in times:
                    a = (t-t0)/max(dt_seg,1e-6)
                    x = fx*(1-a) + tx*a
                    y = fy*(1-a) + ty*a
                    rows.append((pid, direction, t, x, y, leg["mode"], lid))
                t0 = t1
    df = pd.DataFrame(rows, columns=["person_id","direction","t","x","y","mode","link_id"])
    return df

traj = build_traj_from_routes(agents_info, commutes, dt=DT_SECONDS)
print(f"Built {len(traj):,} trajectory points for {traj['person_id'].nunique()} agents.")


# Save for GIS / post-processing if desired
out_csv = os.path.join(base_folder, "commute_trajectories_sample10.csv")
traj.to_csv(out_csv, index=False)
print("Saved:", out_csv)

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
