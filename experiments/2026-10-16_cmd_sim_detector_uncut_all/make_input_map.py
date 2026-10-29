"""All-sets uncut detector (detAll, 4f fork): per-set input map by key + gate (1) per extraction root. Zero cost.

Reads committed records with `git show <commit>:<path>` (keys, roots, shas, counts only; no score or result files) and lists the
activations Volume with `modal volume ls` (metadata only, nothing downloaded). Attack worlds only: run1, run2, sim.

Gate (1), per extraction root: by_layer/ holds L48 and L63, and each of them holds exactly one .npy per chunk the root's
extraction record(s) wrote (names equal, no extras, none empty).
"""

import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = "/Users/aswinsurya/probe-distribution-shift-from-sims"
MODAL = "/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal"
VOLUME = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"
WORLDS = ("run1", "run2", "sim")
SP_WORLDS = {"run1", "run2", "sp493_sim"}  # S_paper493 routing records name the sim world sp493_sim
LAYERS = (48, 63)
REFUSED_ROOTS = {"nm360_s2_run2_s1_cutfull_extract_v1": "writer-race root; routings r1-r3 refused by sha (NOTE @ 2a3067fc3dca); r4 only (4f, 10-16)"}
# (batch__target) -> (routing file, sha256): the selected highest revision must be exactly this file, else STOP
PINNED_ROUTING = {"nm360-s2_m1__nm360_s2_run2_s1_cutfull": (
    "nm360-s2_m1__nm360_s2_run2_s1_cutfull_r4.json", "fa864d2c2fc402b8f815892619d722c4447ed90a0b8fac908b8a94d59b351614")}

PINS = {
    "folds": ("ec56720adde66baabe642a7fc842172c83737679", "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json"),
    # inventory_b_fix_v1 = inventory_b_v2 with nm360_s2_run2_s1 re-routed to r4 (4f, 10-16 ~02:55); replaces v2
    "inv_b": ("09b070617", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v1_2026-10-16_025058.json",
              "4a5d0fb3cad32740f849df5def3c7a03a40a18aa4c42bc360709f0c740973d03"),
    "inv_v3": ("b8a39f329", "experiments/2026-10-15_cut_sensitivity/results/inventory_v3_2026-10-15.json"),
}
# S_old: d9's standard routing records (0d option (a); @ 703ccbe0f3fb), one per root, sha256 prefixes as relayed by 4f.
# Attack-world arms only: q4f_main attack (run1) + sim_attack (sim), q4f_run2 attack (run2).
SOLD = {"q4f_main_cutfull": ("sold__q4f_main_cutfull_r3.json", "764df967", {"attack": "run1", "sim_attack@september": "sim"}),
        "q4f_run2_cutfull": ("sold__q4f_run2_cutfull_r2.json", "0e718de3", {"attack": "run2"})}
ROUTING_REF = sys.argv[1] if len(sys.argv) > 1 else "origin/aswin/q3-cut-option"
ROUTING_DIR = "experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing"
SOLD_EXTRACT = ("4e69ae2d9", "experiments/2026-10-14_q3_overnight_chain/results")  # S_old's 6 extraction records


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, check=True).stdout


def show(commit, fp):
    return json.loads(git("show", f"{commit}:{fp}"))


def vls(prefix, tries=3):
    """Volume listing (metadata only). A failed listing is retried; if every try fails the root fails gate (1)."""
    for _ in range(tries):
        out = subprocess.run([MODAL, "volume", "ls", "--json", VOLUME, prefix], capture_output=True, text=True)
        if not out.returncode:
            return json.loads(out.stdout)
        LS_FAILURES.append(prefix)
    return None


LS_FAILURES = []


def main():
    ref = git("rev-parse", ROUTING_REF).strip()
    folds = show(*PINS["folds"][:2])
    for k, v in PINS.items():
        if len(v) == 3 and hashlib.sha256(git("show", f"{v[0]}:{v[1]}").encode()).hexdigest() != v[2]:
            raise ValueError(f"pin {k}: sha256 mismatch")
    cell_set = folds["cell_set"]

    # d9's cutfull routing records: the HIGHEST revision per (batch, target). It is used only if its own attest record exists
    # with failing == {}; otherwise the target is listed as failed (never a fallback to a lower revision).
    names = [n for n in git("ls-tree", "--name-only", f"{ref}:{ROUTING_DIR}").split() if n.endswith(".json")]
    top = {}
    for n in names:
        if n.endswith("_attest.json") or n.startswith("sold__"):
            continue
        r = show(ref, f"{ROUTING_DIR}/{n}")
        k = (r["batch"], r["target"])
        if k in top and top[k][1]["revision"] == r["revision"]:
            raise ValueError(f"two records at revision {r['revision']} for {k}")
        if k not in top or r["revision"] > top[k][1]["revision"]:
            top[k] = (n, r)
    recs, failed_targets = {}, {}
    for k, (n, r) in top.items():
        att = n[:-5] + "_attest.json"
        a = show(ref, f"{ROUTING_DIR}/{att}") if att in names else None
        if a is None or a.get("failing") != {} or a.get("revision") != r["revision"]:
            failed_targets[f"{k[0]}__{k[1]}"] = "no attest" if a is None else f"attest failing/revision mismatch ({a.get('revision')})"
            continue
        recs[k] = (n, r)
    for t, (fname, sha) in PINNED_ROUTING.items():
        k = tuple(t.split("__", 1))
        got = recs.get(k)
        raw = git("show", f"{ref}:{ROUTING_DIR}/{fname}").encode() if got else b""
        if not got or got[0] != fname or hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f"pinned routing {t}: selected {got and got[0]} is not {fname} @ sha256 {sha[:16]}")
    by_staged = {}
    for n, r in recs.values():
        if r["staged_set"] in by_staged:
            raise ValueError(f"staged set {r['staged_set']} routed twice")
        by_staged[r["staged_set"]] = (n, r)

    sold_keys = {}  # key -> (world, root, routing)
    for t, (fname, pre, arm_world) in SOLD.items():
        revs = [n for n in names if n.startswith(f"sold__{t}") and not n.endswith("_attest.json")]
        if max(revs, key=lambda n: show(ref, f"{ROUTING_DIR}/{n}")["revision"]) != fname:
            raise ValueError(f"S_old {t}: highest revision is not the pinned {fname}")
        raw = git("show", f"{ref}:{ROUTING_DIR}/{fname}")
        if not hashlib.sha256(raw.encode()).hexdigest().startswith(pre):
            raise ValueError(f"S_old {fname}: sha256 does not start with {pre}")
        r = json.loads(raw)
        ap, ac = r["attestation"]["path"].split(" @ ")
        att_raw = git("show", f"{ac}:{ap}")
        att = json.loads(att_raw)
        if hashlib.sha256(att_raw.encode()).hexdigest() != r["attestation"]["sha256"] or att.get("failing") != {} \
                or att["revision"] != r["revision"] or r["attestation"]["revision"] != r["revision"]:
            raise ValueError(f"S_old {fname}: attestation sha/revision/failing check failed")
        for a in r["arms"]:
            if a["meta_arm"] in arm_world:
                if len(a["keys"]) != a["epochs"]:
                    raise ValueError(f"S_old {fname} {a['meta_arm']}: keys != epochs")
                for key in a["keys"]:
                    if key in sold_keys:
                        raise ValueError(f"S_old key routed twice: {key}")
                    sold_keys[key] = (arm_world[a["meta_arm"]], r["extraction_root"], f"{ROUTING_DIR}/{fname} @ {ref[:12]}")

    rows = []  # one per attack-world run: key, cell, world, set, root, routing
    for inv, src_set in (("inv_b", None), ("inv_v3", "S_new")):
        for x in show(*PINS[inv][:2])["runs"]:
            if x["world"] not in WORLDS:
                continue
            u = x.get("uncut_planned") or {}
            s = src_set or ("S_old" if x["task_set"] == "T196" else "S_newmains")
            hit = by_staged.get(u.get("staged_set"))
            # S_old alone may route by attestation (until d9's _cutfull records land); the other sets need a routing record
            if s == "S_old":
                so = sold_keys.get(x["key"])
                if not so or so[0] != x["world"] or so[1] != u.get("extraction_root"):
                    raise ValueError(f"S_old {x['key']}: not in the pinned S_old routing records with the same world and root")
            root = hit[1]["extraction_root"] if hit else (sold_keys[x["key"]][1] if s == "S_old" else None)
            if hit and u.get("extraction_root") and u["extraction_root"] != root:
                raise ValueError(f"{x['key']}: inventory root {u['extraction_root']} != routing root {root}")
            rows.append({"key": x["key"], "cell": x["task"], "world": x["world"], "set": s, "fold_set": cell_set.get(x["task"]),
                         "root": root, "staged_set": u.get("staged_set"),
                         "routing": (f"{ROUTING_DIR}/{hit[0]} @ {ref[:12]}" if hit else (sold_keys[x["key"]][2] if s == "S_old" else None)),
                         "routing_kind": "d9 cutfull routing record" if hit else (
                             "d9 S_old routing record (sold__)" if s == "S_old" else "NO routing record")})

    # extraction records per root -> expected chunk names
    expected, shard_errors = defaultdict(set), Counter()
    for n, r in recs.values():
        if r["extraction_root"] in REFUSED_ROOTS:
            continue
        for e in r["extraction_record"]:
            for sh in show(e["commit"], e["path"])["shards"]:
                if "chunks" not in sh:
                    shard_errors[r["extraction_root"]] += 1
                    continue
                expected[r["extraction_root"]] |= {c["chunk"] for c in sh["chunks"]}
    for fp in git("show", "--name-only", "--format=", SOLD_EXTRACT[0]).split():
        if "/extract_chain_" in fp:
            d = show(SOLD_EXTRACT[0], fp)
            root = d["config"]["run"]
            if True:  # union every record of the root (a later attempt adds its chunks and errors)
                for sh in d["shards"]:
                    if "chunks" not in sh:
                        shard_errors[root] += 1
                        continue
                    expected[root] |= {c["chunk"] for c in sh["chunks"]}

    # S_paper493: keys come from 2e's inventory addenda (pending); its routed attack-world roots are gated now.
    sp = {f"{r['batch']}__{r['target']}": {"revision": r["revision"], "root": r["extraction_root"], "routing": f"{ROUTING_DIR}/{n} @ {ref[:12]}",
                                           "epochs_by_world": {"+".join(a["world"]): a["epochs"] for a in r["arms"]}}
          for n, r in recs.values() if r["stratum"].startswith("S_paper493") and any(set(a["world"]) & SP_WORLDS for a in r["arms"])}
    gate = {}
    for root in sorted({r["root"] for r in rows if r["root"]} | {v["root"] for v in sp.values()}):
        if root in REFUSED_ROOTS:
            gate[root] = {"pass": False, "refused": REFUSED_ROOTS[root]}
            continue
        g = {"expected_chunks": len(expected.get(root, ())), "shard_errors_in_record": shard_errors[root]}
        top = vls(f"{root}/by_layer")
        have = {Path(e["filename"]).name for e in (top or [])}
        g["n_layers_listed"] = len(have)
        ok = bool(expected.get(root)) and not shard_errors[root]
        for l in LAYERS:
            files = vls(f"{root}/by_layer/L{l}") if f"L{l}" in have else None
            got = {Path(e["filename"]).name[:-4]: e["size"] for e in (files or [])
                   if e["type"] == "file" and e["filename"].endswith(".npy")}
            other = sorted(Path(e["filename"]).name for e in (files or []) if not (e["type"] == "file" and e["filename"].endswith(".npy")))
            g[f"L{l}"] = {"listed": files is not None, "n_npy": len(got), "empty": sum(1 for v in got.values() if not v),
                          "missing": sorted(expected.get(root, set()) - set(got))[:5],
                          "extra": sorted(set(got) - expected.get(root, set()))[:5], "non_npy_entries": other[:5]}
            ok &= files is not None and set(got) == expected.get(root, set()) and all(got.values()) and not other
        g["pass"] = ok
        gate[root] = g

    per_set = {}
    for s in ("S_old", "S_new", "S_newmains", "S_paper493"):
        rs = [r for r in rows if r["set"] == s]
        cells = defaultdict(set)
        for r in rs:
            if r["root"] and gate.get(r["root"], {}).get("pass"):
                cells[r["cell"]].add(r["world"])
        fold_cells = {c for c, v in cell_set.items() if v == s}
        per_set[s] = {
            "fold_list_cells": len(fold_cells),
            "runs_by_world": dict(Counter(r["world"] for r in rs)),
            "runs_without_root": sum(1 for r in rs if not r["root"]),
            "runs_in_refused_or_failed_roots": sum(1 for r in rs if r["root"] and not gate.get(r["root"], {}).get("pass")),
            "cells_with_run1_run2_sim_ready": sum(1 for c, w in cells.items() if set(WORLDS) <= w),
            "cells_with_run1_sim_ready": sum(1 for c, w in cells.items() if {"run1", "sim"} <= w),
            "inventory_cells_not_in_fold_set": sorted({r["cell"] for r in rs} - fold_cells)[:10],
            "routing_kinds": dict(Counter(r["routing_kind"] for r in rs)),
            "roots": sorted({r["root"] for r in rs if r["root"]}),
        }
    per_set["S_paper493"]["status"] = "keys PENDING (2e's inventory_sp493_w1/_w2/_w3_v1); routed attack-world targets gated below"
    per_set["S_paper493"]["routed_targets"] = {k: v | {"gate1": gate[v["root"]]["pass"]} for k, v in sorted(sp.items())}

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = Path(__file__).parent / "results" / f"input_map_gate1_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    n_sold_inv = sum(1 for r in rows if r["set"] == "S_old")
    if n_sold_inv != len(sold_keys):
        raise ValueError(f"S_old: inventory attack-world rows {n_sold_inv} != routed keys {len(sold_keys)}")
    rec = {"label": "detAll input map + gate (1) per root (zero cost; keys, roots, counts only; POST HOC; PRELIMINARY)",
           "stamp (PT)": stamp, "pins": {k: list(v) for k, v in PINS.items()}, "routing_ref": [ROUTING_REF, ref], "sold_routing": {t: v[:2] for t, v in SOLD.items()},
           "volume": VOLUME, "layers": list(LAYERS), "refused_roots": REFUSED_ROOTS,
           "n_routing_records_used": len(recs), "volume_ls_failed_tries": LS_FAILURES, "failed_targets": failed_targets, "per_set": per_set, "gate1": gate, "rows": rows}
    with open(out, "x") as f:
        json.dump(rec, f, indent=1)
    print(out)
    for s, v in per_set.items():
        print(s, {k: v[k] for k in v if k != "roots"})
    print("failed routing targets:", failed_targets)
    print("gate1:", sum(g["pass"] for g in gate.values()), "pass /", len(gate), "roots;",
          "fail:", [r for r, g in gate.items() if not g["pass"]])


if __name__ == "__main__":
    main()
