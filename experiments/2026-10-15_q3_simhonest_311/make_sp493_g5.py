"""make_sp493_g5.py (S_paper493 sim-honest g5: the 35 new mains + t98's main keyed t699; 09's rulings via d9 ~19:25-19:35):
config_sp493/tasks_simh_sp493_new.json and config_sp493/pilot_simh_sp493_new.yaml, plus results/sp493_g5_plan_2026-10-15.json.
Each cell is the B fork's AS-RUN (prepped) entry for its key (results/asrun_honest_entries_2026-10-15_192615.json @ 24e7dc722), with
09's gates, all asserted here:
  1. every field of the committed input entry (config/tasks_sp493_box{A,B,C}.json @ 27c327b03) is present and EQUAL;
  2. the ONLY added fields are the prep fields, enumerated (PREP);
  3. approach_sha256 == facts v2's registered approach sha for the key (results/sp493_facts_2026-10-15_165730.json @ d800edc6a);
  plus a by-key 98364da05 -> 27c327b03 equality check (every key in the 2-box files is in the 3-box files with an identical entry).
Keys: pool-key v4 @ d7bc3398d: each new main's shared_key, and t699_model_registry for model_registry|locked_stage_filter_inverted
(T196 origin key t98). Admission cap = expected + K x $106 with cap // 106 = 25 (census room at start); pause at 2 x expected.
  python3 make_sp493_g5.py <repo>
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

repo = sys.argv[1]
HERE = Path(__file__).parent
show = lambda ref, p: subprocess.run(["git", "-C", repo, "show", f"{ref}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
D = "experiments/2026-10-15_q3_sp493/"
POOL = ("d7bc3398d", D + "results/sp493_poolkeys_PROVISIONAL_2026-10-15_192431.json")
ASRUN = ("24e7dc722", D + "results/asrun_honest_entries_2026-10-15_192615.json")
FACTS = ("d800edc6a", D + "results/sp493_facts_2026-10-15_165730.json")
INPUT3, INPUT2 = "27c327b03", "98364da05"
PREP = {"approach_sha256", "approach_chars", "approach_s3", "template_sha256", "attack_pair_launch"}
T98 = ("model_registry", "locked_stage_filter_inverted", "t699_model_registry")
pool = json.loads(show(*POOL))
keys = sorted(m["shared_key"] for m in pool["mains"] if m["origin"] == "new")
assert len(keys) == 35, len(keys)
t98 = [m for m in pool["mains"] if (m["env"], m["main"]) == T98[:2]]
assert len(t98) == 1 and t98[0]["shared_key"] == T98[2] and t98[0]["origin_task_keys"] == ["t98_model_registry"], t98
keys = sorted(keys + [T98[2]])
# 09 via d9 ~21:5x: t987_model_registry has no usable reference (refindex_L3 @ 627400b22, "no_usable_reference"), so it gets no
# sim-honest cell; t699 joins g5 with refindex_L3's pin (its usable rerun2 reference). g5 = 35 cells
DROPPED = {"t987_model_registry": "no_usable_reference (refindex_L3 @ 627400b22; the B fork's launch A recorded no usable reference after 2 reruns)"}
assert set(DROPPED) <= set(keys)
keys = [k for k in keys if k not in DROPPED]
assert len(keys) == 35, len(keys)
asrun = json.loads(show(*ASRUN))
assert asrun["pass"] is True
# the cells come from the LAUNCHED v2 files themselves (@ 548c75d30; shas pinned at the B fork's start: A 2bad5e45, B d05c9b42,
# C 1a2d3807); the extract @ 24e7dc722 is a cross-check (it annotates each entry with "_box", dropped for the comparison)
LAUNCHED = {"A": "2bad5e45", "B": "d05c9b42", "C": "1a2d3807"}
entries, tsh = {}, set()
for b, pref in LAUNCHED.items():
    raw = show("548c75d30", D + f"config/runlist_box{b}_launch_v2.json"); assert hashlib.sha256(raw).hexdigest().startswith(pref), b
    v2 = json.loads(raw); tsh.add(json.dumps(v2.get("template_sha256"), sort_keys=True))  # a per-env map; equal across boxes
    for c in v2["cells"]:
        if c["key"].endswith("_r2"):
            continue  # second-attack copies (B fork)
        assert c["key"] not in entries, c["key"]; entries[c["key"]] = c
for k in keys:
    x = {f: v for f, v in asrun["entries"][k].items() if f != "_box"}
    assert x == entries[k], f"{k}: the extract differs from the launched v2 entry"
inp3 = {}
for b in "ABC":
    for c in json.loads(show(INPUT3, D + f"config/tasks_sp493_box{b}.json"))["cells"]:
        assert c["key"] not in inp3, c["key"]; inp3[c["key"]] = c
eq23 = 0
for b in "AB":
    for c in json.loads(show(INPUT2, D + f"config/tasks_sp493_box{b}.json"))["cells"]:
        assert c["key"] in inp3 and inp3[c["key"]] == c, f"98364da05 -> 27c327b03 differs: {c['key']}"; eq23 += 1
facts = {r["key"]: r for r in json.loads(show(*FACTS))["rows"]}
cells, added = [], {}
for k in keys:
    e, i = entries[k], inp3[k]
    assert all(f in e and e[f] == v for f, v in i.items()), f"{k}: an input field is missing or differs (gate 1)"
    extra = set(e) - set(i)
    assert extra <= PREP, f"{k}: non-prep fields added: {sorted(extra - PREP)} (gate 2)"
    assert "real_arms" not in e, f"{k}: not an honest-key entry"
    assert e["approach_sha256"] == facts[k]["approach_sha256"], f"{k}: approach_sha256 != facts v2 (gate 3)"
    added[k] = sorted(extra); cells.append(e)
assert len(tsh) == 1, tsh
# cell order: round-robin over environments (envs sorted; within an env by key), so the frozen launcher's strict file order and
# its 90 s per-environment start spacing never serialize a same-env block (design fork's L1/L2 finding, d9 ~20:30)
by_env = {}
for c in sorted(cells, key=lambda c: c["key"]):
    by_env.setdefault(c["env"], []).append(c)
cells = [c for row in __import__("itertools").zip_longest(*(by_env[e] for e in sorted(by_env))) for c in row if c is not None]
assert sorted(c["key"] for c in cells) == keys
name = "sp493_new"
top = {"rule": "S_paper493 g5 sim-honest cells: the B fork's as-run honest entries (keys and hashes only; no approach text)",
       "template_sha256": json.loads(tsh.pop()), "source": {"asrun": f"{ASRUN[0]}:{ASRUN[1]}", "input": f"{INPUT3}:{D}config/tasks_sp493_box{{A,B,C}}.json"}}
# the source file the box script and writer compare the cells against: every r1 entry of the three launched v2 files
srcp = HERE / "config_sp493" / "sp493_v2_r1_entries.json"
srcp.write_text(json.dumps({"rule": "every r1 entry of the B fork's launched v2 files (@ 548c75d30; _r2 copies dropped)", "launched_sha256_prefix": LAUNCHED, "cells": [entries[x] for x in sorted(entries)]}, indent=1) + "\n")
tp = HERE / "config_sp493" / f"tasks_simh_{name}.json"
tp.write_text(json.dumps(dict(top, cells=cells), indent=1) + "\n")
n = len(cells); exp = round(n * 19.6); K = 19; cap = exp + K * 106; stop = 2 * exp
cfg = (HERE / "config" / "pilot_simh_q1.yaml").read_text()
cfg, c1 = re.subn(r"^tasks: .*$", f"tasks: config/tasks_simh_{name}.json   # make_sp493_g5.py: S_paper493 g5, 36 mains (as-run honest entries)", cfg, flags=re.M)
cfg, c2 = re.subn(r"^  cap_usd: .*$", f"  cap_usd: {cap}   # {n} cells: expected ${exp} + K {K} x $106 (admission at start: cap // 106 = {cap // 106})", cfg, flags=re.M)
cfg, c3 = re.subn(r"^  worst_case_limit_usd: .*$", f"  worst_case_limit_usd: {cap}   # = cap: the box-side admission bound", cfg, flags=re.M)
assert c1 == c2 == c3 == 1
cfg = f"# S_PAPER493 SIM-HONEST g5 launch simh_{name} (09 Append 39; rulings via d9 ~19:25-19:35), from config/pilot_simh_q1.yaml:\n" + cfg
cp = HERE / "config_sp493" / f"pilot_simh_{name}.yaml"
cp.write_text(cfg)
plan = {"launch": name, "registration": "S_paper493 g5", "keys": keys, "dropped": DROPPED, "n": n, "expected_usd": exp, "K": K, "cap_usd": cap, "sim_cap_stop_pause_at": stop,
        "t98_main": {"key": T98[2], "provenance": "main = model_registry|locked_stage_filter_inverted (T196 origin key t98)"},
        "cell_order": "round-robin over envs (sorted), within env by key",
        "gates": {"input_fields_equal": True, "prep_fields_added": sorted(set().union(*added.values())), "approach_eq_facts_v2": True,
                  "98364da05_to_27c327b03_keys_equal": eq23},
        "sources": {"pool_keys": POOL, "asrun": ASRUN, "facts_v2": FACTS, "input_3box": INPUT3, "input_2box": INPUT2},
        "tasks_file": tp.name, "tasks_sha256": hashlib.sha256(tp.read_bytes()).hexdigest(),
        "source_file": srcp.name, "source_sha256": hashlib.sha256(srcp.read_bytes()).hexdigest(),
        "config": cp.name, "config_sha256": hashlib.sha256(cp.read_bytes()).hexdigest()}
(HERE / "results" / "sp493_g5_plan_2026-10-15.json").write_text(json.dumps(plan, indent=1) + "\n")
print(name, n, "cells; cap", cap, "(admission", cap // 106, ") pause", stop, "| 2->3 box equal", eq23, "| prep", plan["gates"]["prep_fields_added"], "| tasks", plan["tasks_sha256"][:12], "config", plan["config_sha256"][:12])
