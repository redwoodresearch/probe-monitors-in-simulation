"""Tier 2 v2 for route C (a1, 2026-10-18; addendum 5): as tier2_membership.py (reviewed, run @ 5bd79d6fd), with the catalogue and INPUTS_FULL
from config/routec_pins.json and the A25 pilot taken from its pinned root (INPUTS_FULL s_old_simh_extra_roots) by its score files' keys,
relabelled only by the config's extra_root_worlds; the pilot's cells must equal the pilot manifest's 20 tasks. Rebuild the pooled H_B
and H_sim membership WITHOUT the catalogue, from the registered per-set pins, then compare with the catalogue by key, per set and
per membership pattern. Keys and counts only; score files are opened for their `key` column only (to name each sim-honest root's
runs). One Modal CPU container.

  CR_CFG=<pinned input folder> uv run --no-project --python 3.12 --with modal==1.6.1 modal run tier2_membership_v2.py
"""

import hashlib
import io
import json
import os
import time
from pathlib import Path

import modal

SCORES = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
POOLED = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"


FKEY = "p2/fibs__command.npz"  # any headline file names a root's runs; keys only
SETS = ["S_old", "S_new", "S_newmains", "S_paper493"]
IN = {"full": "INPUTS_FULL_2026-10-16_110447.json", "v2": "INPUTS_FINAL_V2_FROZEN_2026-10-16_044120.json",
      "e12": "E12_BLOCK_2026-10-16_092322.json", "xerr": "xerr_scaleup_final_2026-10-15_013255.json",
      "m710": "membership710.json", "a25": "simh_counted_keys_primary_v3.json", "pilot": "simh_counted_keys_pilot_v3.json",
      "S_old": "pilot_tasks_amendment9_all196.json", "S_new": "scaleup_tasks_plus311_2026-10-14_125135.json",
      "S_newmains": "new_mains_cells_v1.json", "sp493": "sp493_facts_2026-10-15_165730.json"}
XERR = ["result", "calibration_records", "primary", "sol_a|R|0.01|primary|BuC|B", "schemes", "stratum_specific"]

app = modal.App("probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
if modal.is_local():
    image = image.add_local_dir(os.environ["CR_CFG"], "/cfg").add_local_file(str(Path(__file__).parent / "config/routec_pins.json"),
                                                                              "/pins.json")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def jl(name):
    return json.loads(Path("/cfg", IN[name]).read_bytes())


def dig(x, path):
    for p in path:
        x = x[p]
    return x


def listsha(keys):
    """The E12 block's epoch1_members_sha256 convention: sha256 of json.dumps(sorted(keys))."""
    return sha(json.dumps(sorted(keys)).encode())


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0,
              volumes={"/s": modal.Volume.from_name(SCORES, version=2).read_only(),
                       "/p": modal.Volume.from_name(POOLED, version=2).read_only()})
def tier2() -> dict:
    import numpy as np

    run_cfg = json.loads(Path("/pins.json").read_bytes())  # config/routec_pins.json
    tbd = [k for k, v in run_cfg.items() if k != "what" and "TBD" in json.dumps(v)]
    assert not tbd, f"run config still has TBD pins: {tbd}"
    assert run_cfg["inputs_full"] in {Path(p).name for _h, _c, p in run_cfg["files"]}, "INPUTS_FULL must be a pinned file"
    for h, _c, path in run_cfg["files"] + run_cfg.get("tier2_files", []):
        assert sha(Path("/cfg", Path(path).name).read_bytes()) == h, f"pin {path}"
    IN["full"] = run_cfg["inputs_full"]
    full, v2, e12 = jl("full"), jl("v2"), jl("e12")
    assert sha(Path("/cfg", IN["e12"]).read_bytes()) == full["e12"]["sha256"]
    assert sha(Path("/cfg", IN["v2"]).read_bytes()) == full["base"]["sha256"]
    cat_b = Path("/p", run_cfg["catalogue"]["path"]).read_bytes()
    assert sha(cat_b) == run_cfg["catalogue"]["sha256"], "catalogue sha"
    cat = json.loads(cat_b)["runs"]
    out = {"checks": {}, "sim_roots": {}, "compare": {}}
    main = {}
    for s in ("S_old", "S_new", "S_newmains"):
        for r in jl(s)["cells"]:
            main[r["key"]] = r["main"]
    for r in jl("sp493")["rows"]:
        main[r["key"]] = r["main"]

    # ---- H_B: epoch 1 (stage_a pins; hashes must equal the E12 block's) + epoch 2 (E12 replicate-only lists)
    xs = dig(jl("xerr"), XERR)
    m710 = jl("m710")
    sa = v2["pools"]["stage_a"]
    e1 = {"S_old": list(xs["pool_keys_old"]), "S_new": list(xs["pool_keys_new"]),
          "S_newmains": sorted(set(m710["pool_honest_b"]) | set(m710["pool_honest_b_4r_sol_only"]))}
    b2 = [k for k in e1["S_old"] if k.startswith("honest_b|")]
    assert listsha(b2) == e12["pools"]["S_old"]["H_B2"]["epoch1_members_sha256"], "S_old B-only epoch-1 list vs E12"
    for s in e1:
        assert listsha(e1[s]) == e12["pools"][s]["H_B"]["epoch1_members_sha256"], f"{s} epoch-1 list vs E12"
        assert len(e1[s]) == sa[s]["H_B"]["n"] == e12["pools"][s]["H_B"]["epoch1_members"], f"{s} epoch-1 n"
    HB = {s: set(e1[s]) for s in e1}
    for s in e1:
        e2 = e12["pools"][s]["H_B"]["e2"]
        ref = set(b2) if s == "S_old" else HB[s]
        bad = [d["key_e2"] for d in e2 if d["replicates"] not in ref
               or d["key_e2"].split("|", 1)[1] != d["replicates"].split("|", 1)[1]]
        assert not bad, f"{s}: epoch-2 entries that don't replicate an epoch-1 honest_b member: {bad[:5]}"
        HB[s] |= {d["key_e2"] for d in e2}
        out["checks"][f"H_B_{s}"] = {"epoch1": len(e1[s]), "epoch2": len(e2)}
    sp = v2["extensions"]["S_paper493"]
    excl = {d["key"] for d in sp["pool_exclusions"]}
    HB["S_paper493"] = set(sp["pools"]["H_B"]) - excl
    out["checks"]["H_B_S_paper493"] = {"listed": len(sp["pools"]["H_B"]), "excluded_by_key": len(set(sp["pools"]["H_B"]) & excl)}

    # ---- sim-honest roots -> keys: the `key` column of each root's file plus its runs_without_rows (cell from the path); worlds
    #      from the registries where they name one (S_paper493 simh, E12 fill_in), else the file's own single world
    ns_root = {d["root"]: d["world"] for d in sp["simh"]}
    for name, d in e12["fill_in"].items():
        assert ns_root.setdefault(d["root"], d["world"]) == d["world"], name
    extra_from = {}
    for root, v in (run_cfg.get("extra_root_worlds") or {}).items():  # addendum 6: the files' world must be v["from"]
        assert root not in ns_root, f"extra root {root} is already in a registry"
        ns_root[root], extra_from[root] = v["to"], v["from"]
    extra_roots = set(full.get("s_old_simh_extra_roots") or {})
    root_keys, root_zero, root_batch, relabels = {}, {}, {}, []
    for k, o in full["overrides"].items():
        ob = Path("/cfg", Path(o["path"]).name).read_bytes()
        assert sha(ob) == o["sha256"], k
        for e in json.loads(ob)["simh"].get(FKEY, []):
            r = e["root"]
            if r in root_keys:
                assert root_batch[r] == e["batch"], r
                continue
            b = Path("/s", e["path"]).read_bytes()
            assert sha(b) == e["sha256"], e["path"]
            keys = [str(x) for x in np.load(io.BytesIO(b), allow_pickle=False)["key"].tolist()]
            ws = {x.split("|", 1)[0] for x in keys}
            assert len(ws) <= 1, (r, ws)
            assert r not in extra_from or ws <= {extra_from[r]}, f"root {r}: files' world {sorted(ws)} is not the registered from-world"
            w = ns_root.get(r) or next(iter(ws), None)
            assert w, f"root {r}: no world (empty file, no registry name)"
            if ws and (ws != {w} or r in extra_from):  # the extra root is always recorded, identity included (addendum 13)
                relabels.append([r, sorted(ws), w])
            root_keys[r] = {f"{w}|{x.split('|', 1)[1]}" for x in keys}
            zs = set()
            for pk in e.get("runs_without_rows") or []:
                cells = [seg for seg in pk.split("/") if seg in main]
                assert len(cells) == 1, f"root {r}: cannot place zero-row run {pk}"
                zs.add(f"{w}|{cells[0]}")
            root_zero[r] = zs
            root_batch[r] = e["batch"]
    out["sim_roots"]["relabels"] = relabels

    def roots_pool(s):
        """stage_a: the roots, minus excluded (batch|cell) in non-replacement roots; replacement roots are authoritative (l.3480)."""
        rs, rep = set(sa[s]["H_sim_roots"]), set(sa[s]["replacement_roots"])
        assert rep <= rs, s
        ex = [x.split("|") for x in sa[s]["excluded_cells"]]
        hits = {f"{b}|{c}": 0 for b, c in ex}
        keys, zero, seen = set(), set(), {}

        def excluded(r, cell):
            for b, c in ex:
                if (root_batch[r] == b or root_batch[r].startswith(b + "_")) and (cell == c or cell.startswith(c + "_")):
                    return f"{b}|{c}"
            return None

        for r in sorted(rs):
            for key in root_keys[r] | root_zero[r]:
                cell = key.split("|", 1)[1]
                hit = None if r in rep else excluded(r, cell)
                if hit:
                    hits[hit] += 1
                    continue
                seen.setdefault(key, []).append(r)
                (zero if key in root_zero[r] else keys).add(key)
        rep_cells = {k.split("|", 1)[1] for r in rep for k in root_keys[r] | root_zero[r]}
        rep_not_excluded = sorted(c for c in rep_cells if not any(c == cc or c.startswith(cc + "_") for _, cc in ex))
        no_replacement = sorted(x for x in hits if not any(c.startswith(x.split("|")[1] + "_") for c in rep_cells))
        info = {"roots": len(rs), "keys": len(keys), "zero_row_keys": sorted(zero), "exclusion_hits": hits,
                "excluded_without_replacement": no_replacement, "replacement_cells_not_excluded": rep_not_excluded,
                "keys_in_several_roots": {k: v for k, v in seen.items() if len(v) > 1}}
        return keys | zero, zero, info

    HS, ZERO = {}, set()
    for s in ("S_new", "S_newmains"):
        HS[s], z, out["sim_roots"][s] = roots_pool(s)
        ZERO |= z
    # S_old: the "A25 + fill-in" pool (l.3984, l.4013-4019, l.4453 §3) = primary_v3 ∪ admitted fill-in ∪ admitted g4 ∪ the A25 pilot 20.
    #        t115 stays (l.4786: the paired-pool exclusion applies to S_paper493 only).
    a25_cells = set(jl("a25")["counted_tasks"])
    sold_map = json.loads(Path("/cfg", Path(full["overrides"]["S_old|C"]["path"]).name).read_bytes())
    sold_roots = sorted({e["root"] for e in sold_map["simh"].get(FKEY, []) if e["root"] not in ns_root and e["root"] not in extra_roots})
    a25_all = [k for r in sold_roots for k in root_keys[r] | root_zero[r]]
    a25 = {k for k in a25_all if k.split("|", 1)[1] in a25_cells}
    assert {k.split("|", 1)[1] for k in a25} == a25_cells, "primary_v3 cells vs S_old roots"
    fill = set(e12["fill_in"]["A25"]["admitted"])
    g4 = set(e12["fill_in"]["g4"]["admitted"])
    assert fill <= root_keys[e12["fill_in"]["A25"]["root"]] and g4 <= root_keys[e12["fill_in"]["g4"]["root"]], "admitted ⊆ root"
    pilot_tasks = {k["task"] for k in jl("pilot")["keys"]}
    assert len(extra_roots) == 1, f"route C: expected one extra S_old sim-honest root, got {sorted(extra_roots)}"
    pr = next(iter(extra_roots))
    pilot = root_keys[pr] | root_zero[pr]  # the pilot root's own keys (relabelled only by the config)
    assert {k.split("|", 1)[1] for k in pilot} == pilot_tasks and len(pilot) == 20, "pilot root keys vs the pilot manifest"
    # 09 on addendum 6, each a STOP: (1) full keys equal the registry's; (2) exactly one extra-root relabel, this root's, from the
    # config's from-world; (3) no other root emits the to-world. The catalogue's (A's) pilot keys are compared by key below.
    assert set(extra_from) == extra_roots, "addendum 6: the config's extra root is INPUTS_FULL's"
    assert sha(Path("/cfg", IN["pilot"]).read_bytes()) == full["s_old_simh_extra_roots"][pr]["sha256"], "pilot registry vs INPUTS_FULL"
    pv3 = jl("pilot")["keys"]
    want = {f"{k['world']}|{k['task']}" for k in pv3}
    assert {k["world"] for k in pv3} == {ns_root[pr]} and len(pv3) == len(want) == 20, "pilot registry: 20 keys in the config's to-world"
    assert pilot == want, "09 (1): the relabelled pilot keys equal the registry's 20 full keys"
    assert [x for x in relabels if x[0] in extra_from] == [[pr, [extra_from[pr]], ns_root[pr]]], "09 (2): exactly one extra-root relabel"
    other = sorted(r for r in root_keys if r != pr and any(k.split("|", 1)[0] == ns_root[pr] for k in root_keys[r] | root_zero[r]))
    assert not other, f"09 (3): other roots emit {ns_root[pr]} keys: {other}"
    cat_pilot = {k for k, v in cat.items() if "S" in v["roles"] and k.split("|", 1)[0] == ns_root[pr]}
    out["checks"]["pilot_vs_catalogue"] = {"mine": len(pilot), "catalogue": len(cat_pilot), "equal": pilot == cat_pilot,
                                           "only_mine": sorted(pilot - cat_pilot), "only_catalogue": sorted(cat_pilot - pilot),
                                           "verdict": "PASS" if pilot == cat_pilot else "FAIL (09 STOP: no compute)"}
    assert not (pilot & (a25 | fill | g4)), "pilot keys overlap primary_v3 / fill-in / g4"
    HS["S_old"] = a25 | fill | g4 | pilot
    out["sim_roots"]["S_old"] = {
        "sold_roots": sold_roots, "a25": len(a25), "a25_keys_outside_primary_v3": len(a25_all) - len(a25),
        "a25_cells_in_two_roots": len(a25_all) - len(set(a25_all)), "fill_admitted": len(fill),
        "fill_root_extra": len(root_keys[e12["fill_in"]["A25"]["root"]] - fill), "g4_admitted": len(g4),
        "g4_root_extra": len(root_keys[e12["fill_in"]["g4"]["root"]] - g4), "pilot_registered": len(pilot)}
    # S_paper493 (Append 39 am.1 §4; l.4364-4366; item 26 at l.4786): existing admitted runs on P_rep mains (A25 for T196 mains,
    #        S_new's pool for +311 mains) ∪ H_sim_new (g5) ∪ g4, minus pool_exclusions and the paired-pool exclusion (t115)
    prep = {o: set(v) for o, v in sp["pools"]["P_rep_mains"].items()}
    no_main = sorted(k for k in a25 | HS["S_new"] if main.get(k.split("|", 1)[1]) is None)
    exist = {k for k in a25 if main.get(k.split("|", 1)[1]) in prep["t196"]} | \
            {k for k in HS["S_new"] if main.get(k.split("|", 1)[1]) in prep["p311"]}
    paired = {d["key"] for d in full["paired_pool_exclusions"]}
    HS["S_paper493"] = (exist | set(sp["pools"]["H_sim_new"]) | g4) - excl - paired
    out["sim_roots"]["S_paper493"] = {"existing_on_P_rep": len(exist), "H_sim_new": len(sp["pools"]["H_sim_new"]), "g4": len(g4),
                                      "removed": sorted((exist | set(sp["pools"]["H_sim_new"]) | g4) & (excl | paired)),
                                      "keys_without_main": no_main[:20]}

    # ---- compare with the catalogue: union, per set, per membership pattern
    for role, mine in (("B", HB), ("S", HS)):
        cat_role = {k: set(v["sets"]) for k, v in cat.items() if role in v["roles"]}
        u = set().union(*mine.values())
        pat = {k: {s for s in mine if k in mine[s]} for k in u}
        only_mine, only_cat = sorted(u - set(cat_role)), sorted(set(cat_role) - u)
        c = {"union": len(u), "catalogue": len(cat_role), "both": len(u & set(cat_role)),
             "per_set": {s: {"mine": len(mine[s]), "catalogue": sum(1 for v in cat_role.values() if s in v)} for s in SETS},
             "n_only_mine": len(only_mine), "only_mine": only_mine[:40],
             "only_mine_pilot": sum(k.startswith("simh_pilot|") for k in only_mine),
             "only_mine_zero_row": sorted(set(only_mine) & ZERO),
             "n_only_catalogue": len(only_cat), "only_catalogue": only_cat[:40],
             "n_pattern_differs": sum(1 for k in u & set(cat_role) if pat[k] != cat_role[k]),
             "pattern_differs": [[k, sorted(pat[k]), sorted(cat_role[k])] for k in sorted(u & set(cat_role)) if pat[k] != cat_role[k]][:30]}
        out["compare"][role] = c
    return out


@app.local_entrypoint()
def main():
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    res = tier2.remote()
    path = Path(__file__).parent / "results" / f"tier2_membership_{stamp}.json"
    path.open("x").write(json.dumps(res, indent=1, default=str))
    print("wrote", path)
    print(json.dumps({r: {k: v for k, v in d.items() if not isinstance(v, list)} for r, d in res["compare"].items()}, indent=1))
    print(json.dumps({"pilot_vs_catalogue": res["checks"]["pilot_vs_catalogue"]}))
