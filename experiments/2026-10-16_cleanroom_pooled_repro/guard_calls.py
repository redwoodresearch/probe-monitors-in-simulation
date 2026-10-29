"""Why do some Sol / own_all_rs C calls have no p2 command unit in the same run? (0d / 09, 10-17; a1; the follow-up to cunits_by_tool's
guard: H_B 208, H_A 208, H_sim 346, run1 8, run2 10, sim 5.) Built from cunits_by_tool.py rev 3 (reviewed): the same folders, the same
shared staged file per folder, the same unit -> staged-row mapping.

For every Sol (and own_all_rs) call not among p2's calls of the same run, from the staged rows of that call (kinds call_end and command
only; output_end rows carry another turn's irid): the tool (call_end row name), the call's command parameter NAME (reader.command_param: e.g. cmd / command /
file_text / new_str / path; None when the value is missing or blank), whether a `command` row exists, whether it has no readout, and
whether p2's score file has a unit at that command row's readout. Revision 2 after review round 1 (review/guard_rounds.md).
Counts per set x arm group and pooled, with Sol C units and the rate per 1,000 Sol units; Sol vs own_all_rs agreement; a seeded sample
(seed 20261017) of <= 10 run keys. Keys, tool / parameter names, booleans and counts only; no text, no score.

  CR_CFG=<pinned input folder> uv run --no-project --python 3.12 --with modal==1.6.1 modal run guard_calls.py
"""

import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

import modal

SCORES = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
ACTS = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"
POOLED = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
CAT = "xfer/fpr507prep/records/results/alltasks_CRD_FULL_r2_allorigins_pooledworld_catalogue_e12_2026-10-16_193738_catalogue.json"
CAT_SHA = "15ee837f09b13533e5c1038bcfe8665e8f7804445b9276111104770c473241c8"
FULL_SHA = "d6923a75aa8f914cd3b2e92d8403f125dd095219997aebea8b1d6e4e2b67f77c"
IN = {"full": "INPUTS_FULL_2026-10-16_110447.json", "v2": "INPUTS_FINAL_V2_FROZEN_2026-10-16_044120.json",
      "e12": "E12_BLOCK_2026-10-16_092322.json", "excl": "exclusions_2026-10-16_102159.json"}
F = {"p2": "p2/fibs__command.npz", "sol": "sola_uncut_reselect/sol_a_uncut__cmd_A.npz",
     "oars": "own_all_rs_uncut/own_all_rs_uncut_L48__C.npz"}
WANT = {"p2": "command", "sol": "call_end", "oars": "call_end"}
SETS = ["S_old", "S_new", "S_newmains", "S_paper493"]
ROLE_GROUP = {"B": "H_B", "A": "H_A", "S": "H_sim"}
TOOL = re.compile(r"^[a-z_]{1,40}$")

app = modal.App("probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
if modal.is_local():
    image = image.add_local_dir(os.environ["CR_CFG"], "/cfg")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def jl(name):
    return json.loads(Path("/cfg", IN[name]).read_bytes())


def base(k):
    return str(k).split("#")[0]


def namespaces(e12, v2):
    """Canonical worlds from the registries (as cleanroom_reduce.py, addenda 1-2)."""
    ns = {"ep2": {f: d["world_e2"] for f, d in e12["folders"].items()}, "arm": {}, "root": {}}
    for d in v2["extensions"]["S_paper493"]["folders"]:
        ns["arm"][d["arm"]] = d["world"]
    for d in v2["extensions"]["S_paper493"]["simh"]:
        ns["root"][d["root"]] = d["world"]
    for name, d in e12["fill_in"].items():
        assert ns["root"].setdefault(d["root"], d["world"]) == d["world"], name
    return ns


def one_group(g: dict) -> dict:
    """One C folder: map units (as cunits_by_tool rev 3); per run, Sol C units and the guard calls' flags."""
    import numpy as np

    def rows(e):
        b = Path("/s", e["path"]).read_bytes()
        assert sha(b) == e["sha256"], e["path"]
        z = np.load(io.BytesIO(b), allow_pickle=False)
        out = {}
        for k, rk, tok in zip(z["key"].tolist(), z["run_key"].tolist(), z["token"].tolist()):
            w, cell = str(k).split("|", 1)
            out.setdefault(str(rk), {"cell": cell, "world": w, "toks": []})["toks"].append(int(tok))
        return out

    def staged(files, want):
        got = {}
        for s, p in files.items():
            h = hashlib.sha256()
            with Path("/a", p).open("rb") as fb:
                for blk in iter(lambda: fb.read(1 << 22), b""):
                    h.update(blk)
            assert h.hexdigest() == s, f"staged sha {p}"
            with Path("/a", p).open() as fh:
                for line in fh:
                    it = json.loads(line)
                    k = base(it.get("key"))
                    if k in want:
                        assert k not in got, f"a run staged twice in {p}"
                        got[k] = [{q: c.get(q) for q in ("name", "kind", "readout", "irid", "call_index", "param", "n")}
                                  for c in it["calls"]]
        return got

    def nm(v):
        return str(v) if TOOL.fullmatch(str(v)) else "unlisted"

    r = {x: rows(g[x]) for x in F}
    st_all = staged(g["staged"], set().union(*[{base(rk) for rk in r[x]} for x in F]))
    out = {"runs": {}, "key_conflicts": 0, "folder": g["folder"]}
    for rk in sorted(set().union(*[set(r[x]) for x in F])):
        infos = {x: r[x][rk] for x in F if rk in r[x]}
        cells = {i["cell"] for i in infos.values()}
        worlds = {g["relabel"] or i["world"] for i in infos.values()}
        if len(cells) != 1 or len(worlds) != 1:
            out["key_conflicts"] += 1
            continue
        key = f"{worlds.pop()}|{cells.pop()}"
        calls = st_all.get(base(rk))
        res = {"key": key, "sol_units": 0, "unmapped": {x: 0 for x in F}, "no_staged": calls is None, "guard": {}}
        if calls is None:
            out["runs"][rk] = res
            continue
        at, byu = {}, {}
        for c in calls:
            if c["readout"] is not None:
                at.setdefault((int(c["readout"]), c["kind"]), []).append(c)
            if c["kind"] in ("call_end", "command"):
                byu.setdefault((c["irid"], c["call_index"]), []).append(c)
        callset = {x: set() for x in F}
        for x, info in infos.items():
            for tok in info["toks"]:
                cs = at.get((tok, WANT[x]), [])
                if len(cs) != 1:
                    res["unmapped"][x] += 1
                    continue
                callset[x].add((cs[0]["irid"], cs[0]["call_index"]))
                res["sol_units"] += int(x == "sol")
        p2toks = set(infos["p2"]["toks"]) if "p2" in infos else set()
        for x in ("sol", "oars"):
            if "p2" not in infos or x not in infos:
                res["guard"][x] = None
                continue
            det = []
            for u in sorted(callset[x] - callset["p2"], key=str):
                rs = byu.get(u, [])
                cmd = [c for c in rs if c["kind"] == "command"]
                ce = [c for c in rs if c["kind"] == "call_end"]
                prm = (ce or cmd)[0]["param"] if (ce or cmd) else None  # reader.command_param: the parameter NAME, None if empty
                det.append({"tool": nm(ce[0]["name"]) if ce else None,
                            "command_param": None if prm is None else nm(prm), "command_param_empty_or_missing": prm is None,
                            "has_command_row": bool(cmd), "command_readout_none": bool(cmd) and all(c["readout"] is None for c in cmd),
                            "p2_unit_at_command_readout": any(c["readout"] in p2toks for c in cmd if c["readout"] is not None)})
            res["guard"][x] = det
        out["runs"][rk] = res
    return out


@app.function(image=image, cpu=8, memory=16384, timeout=2400, retries=0,
              volumes={"/s": modal.Volume.from_name(SCORES, version=2).read_only(),
                       "/a": modal.Volume.from_name(ACTS, version=2).read_only(),
                       "/p": modal.Volume.from_name(POOLED, version=2).read_only()})
def run() -> dict:
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np

    assert sha(Path("/cfg", IN["full"]).read_bytes()) == FULL_SHA, "INPUTS_FULL sha"
    full, v2, e12, ex = jl("full"), jl("v2"), jl("e12"), jl("excl")
    assert sha(Path("/cfg", IN["v2"]).read_bytes()) == full["base"]["sha256"], "INPUTS_FINAL_V2 sha"
    assert sha(Path("/cfg", IN["e12"]).read_bytes()) == full["e12"]["sha256"], "E12 sha"
    assert sha(Path("/cfg", IN["excl"]).read_bytes()) == full["exclusions"]["sha256"], "exclusions sha"
    cb = Path("/p", CAT).read_bytes()
    assert sha(cb) == CAT_SHA, "catalogue sha"
    cat = json.loads(cb)["runs"]
    cohort = {c: s for s in SETS for c in ex["final_cohort_keys"][s]["cells"]}
    ns = namespaces(e12, v2)
    # ---- every C folder with all three files: its namespace relabel and each reader's own staged files
    groups, skipped_folders, listed_not_used = {}, [], {}
    for k, o in full["overrides"].items():
        if not k.endswith("|C"):
            continue
        ob = Path("/cfg", Path(o["path"]).name).read_bytes()
        assert sha(ob) == o["sha256"], k
        om = json.loads(ob)
        setname = k.split("|")[0].replace("+25f", "")
        ents = [(f, d, "arm", None) for f, d in om["arms"].items()]
        simh = {}
        for fk, lst in om["simh"].items():
            for e in lst:
                simh.setdefault(e["root"], {})[fk] = e
        ents += [(rt, d, "simh", rt) for rt, d in simh.items()]
        for folder, d, kind, root in ents:
            if not all(F[x] in d for x in F):
                continue
            p = d[F["p2"]]["path"]
            if p in groups:
                continue
            if kind == "arm" and setname == "pools_ep2":
                if folder not in ns["ep2"]:
                    skipped_folders.append(folder)  # not admitted (addendum 1)
                    continue
                relabel = ns["ep2"][folder]
            elif kind == "arm" and setname == "S_paper493":
                relabel = ns["arm"][folder]
            elif kind == "simh":
                relabel = ns["root"].get(root)
            else:
                relabel = None
            # the folder's staged file = the one staged sha that all three readers list (review round 2); its path from p2's
            # manifest by folder name, cross-checked by sha; every other listed sha is "listed, not used" and reported
            lists = {x: set(d[F[x]].get("staged_shas") or []) for x in F}
            shared = set.intersection(*lists.values())
            assert len(shared) == 1, f"{folder}: {len(shared)} staged shas shared by all three readers"
            s_shared = shared.pop()
            mb = Path("/s", d[F["p2"]]["manifest"]["path"]).read_bytes()
            assert sha(mb) == d[F["p2"]]["manifest"]["sha256"], "p2 manifest sha"
            st_ = [s for s in json.loads(mb)["dump"]["staged"] if Path(s["folder"].rstrip("/")).name == Path(p).parent.name]
            assert len(st_) == 1 and st_[0]["sha"] == s_shared, f"{folder}: p2 manifest's staged entry vs the shared sha"
            assert Path("/a", st_[0]["path"]).exists(), f"staged file missing on the acts Volume: {st_[0]['path']}"
            extra = {x: sorted(v - {s_shared}) for x, v in lists.items() if v - {s_shared}}
            if extra:
                listed_not_used[folder] = extra
            groups[p] = {x: {"path": d[F[x]]["path"], "sha256": d[F[x]]["sha256"]} for x in F} | {
                "staged": {s_shared: st_[0]["path"]}, "relabel": relabel, "folder": folder}
    with ProcessPoolExecutor(8) as pool:
        parts = list(pool.map(one_group, [groups[p] for p in sorted(groups)]))
    # ---- assign each run (canonical key, once) to arm groups and sets; tally guard calls by class
    agg, seen, skipped, dup_keys, sol_vs_oars_differ, guard_runs = {}, {}, 0, 0, 0, []
    key_conflicts = sum(pt["key_conflicts"] for pt in parts)
    for pt in parts:
        for rk, res in pt["runs"].items():
            key = res["key"]
            if key in seen:
                dup_keys += 1
                assert seen[key] == rk, f"one canonical key, two physical runs: {key}"
                continue
            seen[key] = rk
            w, cell = key.split("|", 1)
            dest = []
            if key in cat:
                for role in cat[key]["roles"]:
                    dest += [(ROLE_GROUP[role], s) for s in cat[key]["sets"]] + [(ROLE_GROUP[role], "pooled")]
            elif w in ("run1", "run2", "sim") and cell in cohort:
                dest += [(w, cohort[cell]), (w, "pooled")]
            if not dest:
                skipped += 1
                continue
            gs, go = res["guard"].get("sol"), res["guard"].get("oars")
            sol_vs_oars_differ += int(gs != go)
            if gs:
                guard_runs.append((key, sorted({grp for grp, _ in dest}), len(gs)))
            for grp, s in dest:
                a = agg.setdefault(f"{s}|{grp}", {"runs": 0, "sol_units": 0, "guard_calls": 0, "runs_with_guard_calls": 0,
                                                   "no_staged_runs": 0, "unmapped": {x: 0 for x in F}, "classes": {}})
                a["runs"] += 1
                a["sol_units"] += res["sol_units"]
                a["no_staged_runs"] += int(res["no_staged"])
                for x in F:
                    a["unmapped"][x] += res["unmapped"][x]
                a["runs_with_guard_calls"] += int(bool(gs))
                for d_ in gs or []:
                    a["guard_calls"] += 1
                    k_ = json.dumps(d_, sort_keys=True)
                    a["classes"][k_] = a["classes"].get(k_, 0) + 1
    out = {"n_groups": len(groups), "n_runs_seen": len(seen), "dup_keys_across_folders": dup_keys,
           "key_conflicts_across_readers": key_conflicts, "runs_not_in_lead_inputs": skipped,
           "runs_where_sol_and_oars_guard_calls_differ": sol_vs_oars_differ, "cells": {}}
    for k, a in sorted(agg.items()):
        out["cells"][k] = {"runs": a["runs"], "sol_units": a["sol_units"], "guard_calls": a["guard_calls"],
                           "guard_calls_per_1000_sol_units": 1000 * a["guard_calls"] / a["sol_units"] if a["sol_units"] else None,
                           "runs_with_guard_calls": a["runs_with_guard_calls"], "no_staged_runs": a["no_staged_runs"],
                           "unmapped_units": a["unmapped"],
                           "classes": [{"n": n, **json.loads(c)} for c, n in sorted(a["classes"].items(), key=lambda kv: -kv[1])]}
    rng = np.random.default_rng(20261017)
    guard_runs.sort()
    pick = sorted(rng.choice(len(guard_runs), size=min(10, len(guard_runs)), replace=False).tolist()) if guard_runs else []
    out["sample_seed"] = 20261017
    out["sample_runs_with_guard_calls"] = [{"key": guard_runs[i][0], "groups": guard_runs[i][1], "guard_calls": guard_runs[i][2]}
                                           for i in pick]
    out["n_runs_with_guard_calls"] = len(guard_runs)
    return out


@app.local_entrypoint()
def main():
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    res = run.remote()
    path = Path(__file__).parent / "results" / f"guard_calls_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print("wrote", path)
