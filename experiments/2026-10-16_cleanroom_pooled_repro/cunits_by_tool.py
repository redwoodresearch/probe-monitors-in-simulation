"""C units by tool, over ALL runs in the pooled lead inputs (0d / 09, 10-17; a1). Context for a draft, not a correction.
Revision 3 after review rounds 1-2 (review/cunits_rounds.md): all three readers map against the folder's ONE staged file, the staged sha
that all three readers' file-map entries list (path from p2's manifest by folder name, sha-checked); every other listed sha is reported
("listed, not used"; Sol / own_all_rs extras with per-folder unmapped and no-staged counts). A guard counts Sol / own_all_rs calls that
are not p2 calls of the same run (expected 0: Sol = p2 minus fixed-slot calls). Run identity is checked across readers; nothing is
dropped silently.

For every run of pooled A's lead inputs (catalogue roles B, A, S; attack run1 / run2 / sim on the 1,313 cohort cells) and for each C
reader (p2 command = the public probes' registered command-parameter end; Sol cmd_A and own_all_rs C = content-only call_end):
  - each unit's tool (bash / text_editor / other, with other's tool names), via the staged row of the reader's kind at the unit's token;
    unit level and run level (runs with >= 1 text_editor unit);
  - for p2 text_editor command units: the gap (call_end readout - command readout) of the same call.
Grouped by set (each set whose pool admitted the run, or the cohort set of an attack cell) and pooled (each run once), per arm group.

Keys, tool names, counts and offsets only. The job reads score files' key / run_key / token columns and the staged rows' name, kind,
readout, irid and call_index; no text and no score leaves it. One Modal CPU container, folders in parallel processes.

  CR_CFG=<pinned input folder> uv run --no-project --python 3.12 --with modal==1.6.1 modal run cunits_by_tool.py
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
    """One C folder: each reader's units mapped to its own staged rows; per run (canonical key) tools, gaps and counts."""
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
        """The folder's one staged file (all three readers list its sha): the calls rows of the wanted runs."""
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
                        got[k] = [{q: c.get(q) for q in ("name", "kind", "readout", "irid", "call_index")} for c in it["calls"]]
        return got

    r = {x: rows(g[x]) for x in F}
    st_all = staged(g["staged"], set().union(*[{base(rk) for rk in r[x]} for x in F]))
    st = {x: st_all for x in F}
    out = {"runs": {}, "key_conflicts": 0, "folder": g["folder"]}
    for rk in sorted(set().union(*[set(r[x]) for x in F])):
        infos = {x: r[x][rk] for x in F if rk in r[x]}
        cells = {i["cell"] for i in infos.values()}
        worlds = {g["relabel"] or i["world"] for i in infos.values()}
        if len(cells) != 1 or len(worlds) != 1:  # one physical run, one canonical key, in every reader
            out["key_conflicts"] += 1
            continue
        key = f"{worlds.pop()}|{cells.pop()}"
        res = {"key": key, "tools": {}, "gaps": [], "unmapped": {x: 0 for x in F}, "no_staged_units": {x: 0 for x in F}}
        callset = {x: set() for x in F}
        for x, info in infos.items():
            calls = st[x].get(base(rk))
            if calls is None:  # counted, never dropped silently
                res["no_staged_units"][x] += len(info["toks"])
                continue
            at, ce = {}, {}
            for c in calls:
                if c["readout"] is not None:
                    at.setdefault((int(c["readout"]), c["kind"]), []).append(c)
                    if c["kind"] == "call_end":
                        ce[(c["irid"], c["call_index"])] = int(c["readout"])
            t = {}
            for tok in info["toks"]:
                cs = at.get((tok, WANT[x]), [])
                if len(cs) != 1:
                    res["unmapped"][x] += 1
                    continue
                callset[x].add((cs[0]["irid"], cs[0]["call_index"]))
                name = str(cs[0]["name"]) if TOOL.match(str(cs[0]["name"])) else "unlisted"
                t[name] = t.get(name, 0) + 1
                if x == "p2" and name == "text_editor":
                    e = ce.get((cs[0]["irid"], cs[0]["call_index"]))
                    res["gaps"].append(None if e is None else e - tok)
            res["tools"][x] = t
        # guard (Sol / own_all_rs = p2 minus fixed-slot calls, p2_unit_check): their calls must be p2 calls of the same run
        res["calls_outside_p2"] = {x: len(callset[x] - callset["p2"]) if "p2" in infos else None for x in ("sol", "oars")}
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
    # ---- assign each run (canonical key, once) to arm groups and sets
    agg, seen, skipped, dup_keys = {}, {}, 0, 0
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
            both = key in cat and {"A", "B"} <= set(cat[key]["roles"])
            if key in cat:
                for role in cat[key]["roles"]:
                    dest += [(ROLE_GROUP[role], s) for s in cat[key]["sets"]] + [(ROLE_GROUP[role], "pooled")]
            elif w in ("run1", "run2", "sim") and cell in cohort:
                dest += [(w, cohort[cell]), (w, "pooled")]
            if not dest:
                skipped += 1
                continue
            for grp, s in dest:
                a = agg.setdefault(f"{s}|{grp}", {"runs": 0, "runs_in_both_A_and_B": 0, "keys": set(),
                                                   "unmapped": {x: 0 for x in F}, "no_staged_units": {x: 0 for x in F},
                                                   "tools": {x: {} for x in F}, "runs_with_te": {x: 0 for x in F},
                                                   "runs_with_units": {x: 0 for x in F}, "gaps": [],
                                                   "calls_outside_p2": {"sol": 0, "oars": 0}, "runs_without_p2": 0})
                a["runs"] += 1
                a["keys"].add(key)
                a["runs_in_both_A_and_B"] += int(both)
                for x in F:
                    a["unmapped"][x] += res["unmapped"][x]
                    a["no_staged_units"][x] += res["no_staged_units"][x]
                for x, t in res["tools"].items():
                    a["runs_with_units"][x] += int(sum(t.values()) > 0)
                    a["runs_with_te"][x] += int(t.get("text_editor", 0) > 0)
                    for name, n in t.items():
                        a["tools"][x][name] = a["tools"][x].get(name, 0) + n
                a["gaps"] += res["gaps"]
                for x, v in res["calls_outside_p2"].items():
                    if v is None:
                        a["runs_without_p2"] += int(x == "sol")
                    else:
                        a["calls_outside_p2"][x] += v
    so_extra = {f: {x: v for x, v in e.items() if x != "p2"} for f, e in listed_not_used.items() if set(e) - {"p2"}}
    per_folder = {pt["folder"]: {"unmapped": {x: sum(res["unmapped"][x] for res in pt["runs"].values()) for x in F},
                                 "no_staged_units": {x: sum(res["no_staged_units"][x] for res in pt["runs"].values()) for x in F}}
                  for pt in parts if pt["folder"] in so_extra}
    out = {"n_groups": len(groups), "skipped_folders": sorted(set(skipped_folders)),
           "sol_or_oars_listed_not_used_staged_shas": {f: {"extra": e, **per_folder.get(f, {})} for f, e in so_extra.items()},
           "p2_folders_listing_other_batch_staged_files": sum(1 for e in listed_not_used.values() if "p2" in e),
           "n_runs_seen": len(seen), "dup_keys_across_folders": dup_keys, "key_conflicts_across_readers": key_conflicts,
           "runs_not_in_lead_inputs": skipped, "cells": {}}
    for k, a in sorted(agg.items()):
        cell = {"runs": a["runs"], "runs_in_both_A_and_B": a["runs_in_both_A_and_B"], "readers": {},
                "guard_calls_outside_p2": a["calls_outside_p2"], "runs_without_p2_units": a["runs_without_p2"]}
        for x in F:
            t = a["tools"][x]
            tot = sum(t.values())
            cell["readers"][x] = {
                "units": tot, "bash": t.get("bash", 0), "text_editor": t.get("text_editor", 0),
                "other": {n: c for n, c in sorted(t.items()) if n not in ("bash", "text_editor")},
                "share_text_editor": t.get("text_editor", 0) / tot if tot else None, "unmapped_units": a["unmapped"][x],
                "units_of_runs_without_staged_record": a["no_staged_units"][x], "runs_with_units": a["runs_with_units"][x],
                "runs_with_text_editor": a["runs_with_te"][x],
                "share_runs_with_text_editor": a["runs_with_te"][x] / a["runs_with_units"][x] if a["runs_with_units"][x] else None}
        gp = [v for v in a["gaps"] if v is not None]
        arr = np.array(gp, dtype=np.int64) if gp else np.zeros(0, np.int64)
        cell["p2_text_editor_gaps"] = {
            "units": len(a["gaps"]), "no_call_end": sum(v is None for v in a["gaps"]), "precedes": int((arr > 0).sum()),
            "zero": int((arr == 0).sum()), "negative": int((arr < 0).sum()),
            "share_precedes": float((arr > 0).mean()) if len(arr) else None,
            "median": float(np.median(arr)) if len(arr) else None, "p10": float(np.percentile(arr, 10)) if len(arr) else None,
            "p90": float(np.percentile(arr, 90)) if len(arr) else None, "max": int(arr.max()) if len(arr) else None}
        out["cells"][k] = cell
    # ---- expected counts: pools vs the catalogue's roles; attack arms vs the cohort (1,313 cells)
    pk = {g_: agg.get(f"pooled|{g_}", {}).get("keys", set()) for g_ in list(ROLE_GROUP.values()) + ["run1", "run2", "sim"]}
    out["pooled_pool_runs_vs_catalogue"] = {
        g_: {"seen": len(pk[g_]), "catalogue": sum(r_ in v["roles"] for v in cat.values()),
             "catalogue_keys_not_seen": sorted({k for k, v in cat.items() if r_ in v["roles"]} - pk[g_])[:20]}
        for r_, g_ in ROLE_GROUP.items()}
    out["attack_runs_vs_cohort"] = {
        w_: {"seen": len(pk[w_]), "cohort": len(cohort), "cohort_keys_not_seen": sorted({f"{w_}|{c}" for c in cohort} - pk[w_])[:20]}
        for w_ in ("run1", "run2", "sim")}
    return out


@app.local_entrypoint()
def main():
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    res = run.remote()
    path = Path(__file__).parent / "results" / f"cunits_by_tool_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print("wrote", path)
