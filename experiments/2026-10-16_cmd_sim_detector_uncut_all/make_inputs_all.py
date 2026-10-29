"""All-sets uncut detector (detAll): input configs in detU's loader format (../2026-10-16_cmd_sim_detector_uncut/uncut_inputs.py),
one per set, so the loader runs unchanged. Zero cost: committed records via `git show`, keys, roots and shas only.

  --phase inventory: S_new has no (b)-contract inventory (inventory_v3 is the (a'') contract: no task_set, full_len, extraction root or
    staged sha). Writes results/inventory_S_new_derived_v1_<stamp>.json: v3's attack-world rows by key, with full_len = tokens_before
    and each row's extraction root and staged sha256 from its d9 cutfull routing record (status DERIVED; fail-closed).
  --phase configs --snew-inventory <commit>:<path>: writes config/inputs_all_{S_new,S_newmains,S_paper493_w1,S_paper493_w23}_v1.json.
    S_newmains is read with uncut_inputs_all.py (detU's loader + the refused S4 orphans counted in the whole-arm check).
    S_old uses detU's config/inputs_uncut_v2.json unchanged (d9's sold__ routing records @ 703ccbe0f3fb).

Routing selection (as make_input_map.py): the HIGHEST revision per (batch, target), used only if its own _attest.json has failing {}
and the same revision; nm360_s2_run2_s1 must select r4 by sha256 (4f, 10-16); any other outcome stops.
"""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = "/Users/aswinsurya/probe-distribution-shift-from-sims"
HERE = Path(__file__).parent
VOL = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"
ROUTING_DIR = "experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing"
WORLDS = ("run1", "run2", "sim")
INV_V3 = {"commit": "b8a39f329", "path": "experiments/2026-10-15_cut_sensitivity/results/inventory_v3_2026-10-15.json"}
INV_B_FIX = {"commit": "09b070617981342cd31ce95475da369f9c445b9b",
             "path": "experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v1_2026-10-16_025058.json",
             "sha256": "4a5d0fb3cad32740f849df5def3c7a03a40a18aa4c42bc360709f0c740973d03"}
PINNED = {"nm360-s2_m1__nm360_s2_run2_s1_cutfull": ("nm360-s2_m1__nm360_s2_run2_s1_cutfull_r4.json",
                                                    "fa864d2c2fc402b8f815892619d722c4447ed90a0b8fac908b8a94d59b351614")}


def gitb(*a) -> bytes:
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, check=True).stdout


def spec_of(commit: str, path: str) -> dict:
    full = gitb("rev-parse", f"{commit}^{{commit}}").decode().strip()
    return {"commit": full, "path": path, "sha256": hashlib.sha256(gitb("show", f"{full}:{path}")).hexdigest()}


def load(spec: dict):
    b = gitb("show", f"{spec['commit']}:{spec['path']}")
    if "sha256" in spec and hashlib.sha256(b).hexdigest() != spec["sha256"]:
        raise SystemExit(f"{spec['path']}: sha256 mismatch")
    return json.loads(b)


def select_routing(ref: str) -> dict:
    """(batch, target) -> (spec, record) for every non-sold__ cutfull routing target whose highest revision is attested clean."""
    names = [n for n in gitb("ls-tree", "--name-only", f"{ref}:{ROUTING_DIR}").decode().split() if n.endswith(".json")]
    top = {}
    for n in names:
        if n.endswith("_attest.json") or n.startswith("sold__"):
            continue
        r = json.loads(gitb("show", f"{ref}:{ROUTING_DIR}/{n}"))
        k = (r["batch"], r["target"])
        if k in top and top[k][1]["revision"] == r["revision"]:
            raise SystemExit(f"two records at revision {r['revision']} for {k}")
        if k not in top or r["revision"] > top[k][1]["revision"]:
            top[k] = (n, r)
    out = {}
    for k, (n, r) in top.items():
        att = n[:-5] + "_attest.json"
        if att not in names:
            continue  # not attested (e.g. IL arms): unusable, and never a fallback to a lower revision
        a = json.loads(gitb("show", f"{ref}:{ROUTING_DIR}/{att}"))
        if a.get("failing") != {} or a.get("revision") != r["revision"]:
            continue
        out[k] = (spec_of(ref, f"{ROUTING_DIR}/{n}"), r)
    st = Counter(r["staged_set"] for _, r in out.values())
    if any(v > 1 for v in st.values()):  # (Codex r1) one routing record per staged set
        raise SystemExit(f"staged sets routed by two targets: {sorted(k for k, v in st.items() if v > 1)}")
    for t, (fname, sha) in PINNED.items():
        got = out.get(tuple(t.split("__", 1)))
        if not got or not got[0]["path"].endswith("/" + fname) or got[0]["sha256"] != sha:
            raise SystemExit(f"pinned routing {t}: not {fname} @ sha256 {sha[:16]}")
    return out


def arm_sha(rec: dict, staged_set: str, arm: str) -> tuple:
    a = [x for x in rec["arms"] if x["meta_arm"] == arm and x["staged_inputs_file"] == f"{staged_set}/inputs/{arm}.jsonl"]
    if len(a) != 1:
        raise SystemExit(f"{staged_set}: routing record has {len(a)} arms named {arm}")
    return a[0]["staged_sha"], a[0]["epochs"]


def phase_inventory(ref: str):
    sel = {r["staged_set"]: (s, r) for s, r in select_routing(ref).values() if r.get("stratum") == "S_new"}
    v3 = load(INV_V3)
    rows, per = [], Counter()
    for x in v3["runs"]:
        if x["world"] not in WORLDS:
            continue
        u = x["uncut_planned"]
        if u["staged_set"] not in sel or u["volume"] != VOL:
            raise SystemExit(f"{x['key']}: staged set {u['staged_set']} has no attested S_new routing record, or another volume")
        s, r = sel[u["staged_set"]]
        if r["volume"] != VOL or r["cut"] != "full" or u["staged_inputs_file"] != f"{x['arm']}.jsonl":
            raise SystemExit(f"{x['key']}: routing volume/cut or staged inputs file mismatch")
        sha, n = arm_sha(r, u["staged_set"], x["arm"])
        per[(u["staged_set"], x["arm"])] += 1
        rows.append({"key": x["key"], "task_set": "S_new", "world": x["world"], "arm": x["arm"], "task": x["task"],
                     "full_len": x["tokens_before"], "tokens_before": x["tokens_before"], "original_folder": x["original_folder"],
                     "original_folder_registered": x["original_folder_registered"], "registered_source": f"inventory_v3 @ {INV_V3['commit']}",
                     "uncut_planned": {"volume": VOL, "staged_set": u["staged_set"], "folder": u["folder"], "extraction_root": r["extraction_root"],
                                       "staged_sha256": sha, "staged_inputs_file": u["staged_inputs_file"], "routing": f"{s['path']} @ {s['commit'][:12]}",
                                       "revision": r["revision"]}})
    for (st, arm), n in per.items():  # the whole routed arm is these rows (the loader checks it again)
        if arm_sha(sel[st][1], st, arm)[1] != n:
            raise SystemExit(f"{st}/{arm}: {n} inventory rows != routed epochs")
    if len({r["key"] for r in rows}) != len(rows):
        raise SystemExit("a key twice")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"inventory_S_new_derived_v1_{stamp}.json"
    rec = {"label": "detAll: S_new attack-world inventory in the (b) contract, DERIVED from inventory_v3 by key + d9's cutfull routing records "
                    "(roots, staged shas); POST HOC; keys/shas only", "status": "DERIVED", "inputs": {"inventory_v3": INV_V3, "routing_ref": ref},
           "n_runs": len(rows), "by_world": dict(Counter(r["world"] for r in rows)), "runs": rows}
    with open(out, "x") as f:
        json.dump(rec, f, indent=1)
    print(out, rec["n_runs"], rec["by_world"])


def config_for(label: str, inv_spec: dict, task_set: str, status: str, routed: list) -> dict:
    ex = {}
    for s, r in routed:
        recs = r["extraction_record"]
        if len(recs) != 1:
            raise SystemExit(f"{r['extraction_root']}: {len(recs)} extraction records (the loader takes exactly one)")
        new = spec_of(recs[0]["commit"], recs[0]["path"])
        if ex.setdefault(r["extraction_root"], new) != new:  # (Codex r1) the full spec: commit, path and sha256
            raise SystemExit(f"{r['extraction_root']}: two extraction records")
    return {"label": label, "task_set": task_set, "inventory_status": status, "inventory": inv_spec, "extraction_records": ex,
            "routing": {"mode": "routing", "records": [s for s, _ in routed]}}


def phase_configs(ref: str, snew_inv: str):
    sel = select_routing(ref)
    c, p = snew_inv.split(":", 1)
    inv_new = spec_of(c, p)
    def routed(stratum, staged_sets):
        return [(s, r) for s, r in sel.values() if r.get("stratum") == stratum and r["staged_set"] in staged_sets]
    new_sets = {x["uncut_planned"]["staged_set"] for x in load(inv_new)["runs"]}
    nm_sets = {x["uncut_planned"]["staged_set"] for x in load(INV_B_FIX)["runs"] if x["task_set"] == "new mains" and x["world"] in WORLDS}
    sp = {"S_paper493_w1": {"commit": "85447113f", "path": "experiments/2026-10-15_cut_sensitivity/results/inventory_sp493_w1_v1_2026-10-16_025411.json",
                            "sha_prefix": "3dc8a91a"},
          "S_paper493_w23": {"commit": "102b3741e", "path": "experiments/2026-10-15_cut_sensitivity/results/inventory_sp493_w23_v1_2026-10-16_033210.json",
                             "sha_prefix": "b11a63f5914c1c5d199c3401104acd9697b6d3e7095802f24e67635895dc98f5"}}
    extra = []
    for name, v in sp.items():
        spec = spec_of(v["commit"], v["path"])
        if not spec["sha256"].startswith(v["sha_prefix"]):
            raise SystemExit(f"{name}: inventory sha256 is not the relayed {v['sha_prefix']}")
        inv = load(spec)
        st = {x["uncut_planned"]["staged_set"] for x in inv["runs"] if x["world"] in WORLDS}
        rt = [(s, r) for s, r in sel.values() if r.get("stratum", "").startswith("S_paper493") and r["staged_set"] in st]
        extra.append((name, config_for(f"detAll uncut inputs, {name} (attack worlds): 2e's inventory + d9 cutfull routing records", spec,
                                       "S_paper493", inv["status"], rt)))
    for name, cfg in [
        ("S_new", config_for("detAll uncut inputs, S_new (attack worlds): derived inventory + d9 cutfull routing records", inv_new, "S_new",
                             "DERIVED", routed("S_new", new_sets))),
        ("S_newmains", config_for("detAll uncut inputs, S_newmains (attack worlds): inventory_b_fix_v1 + d9 cutfull routing records (nm360_s2_run2_s1 = r4)",
                                  INV_B_FIX, "new mains", load(INV_B_FIX)["status"], routed("S_newmains", nm_sets)))] + extra:
        cfg["routing_ref"] = ref
        out = HERE / "config" / f"inputs_all_{name}_v1.json"
        out.parent.mkdir(exist_ok=True)
        with open(out, "x") as f:
            json.dump(cfg, f, indent=1)
        print(out, len(cfg["extraction_records"]), "roots,", len(cfg["routing"]["records"]), "routing records")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("inventory", "configs"), required=True)
    ap.add_argument("--routing-ref", required=True)
    ap.add_argument("--snew-inventory", default="")
    a = ap.parse_args()
    ref = gitb("rev-parse", f"{a.routing_ref}^{{commit}}").decode().strip()
    phase_inventory(ref) if a.phase == "inventory" else phase_configs(ref, a.snew_inventory)
