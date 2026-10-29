"""S_paper493 sim TAIL + L3 (d9 ~22:10, plan approved; 09: split @ 59f333d03, L3 = 37): two sims-only launches on the frozen IL
+311 harness, X (xbox1) and Y (box3), max_runs_per_server 5 (K = 30 each). Cells: g6 (L1's 137 unstarted), g7 (L2's 121 unstarted)
and g3 (L3's 37 pinned cells of refindex_L3 @ 627400b22; t987 has no usable reference and leaves the cohort) = 295.
Rules (09/d9): each moved cell keeps its key, entry, pinned reference and approach sha; only task-file order, box, server set,
launch id (and the caps) differ from L1/L2. Keys only; no outcome or approach text read.
- Entries fed to the on-box prep (prep_sp493_sim.sh, unchanged rule): g6/g7 = their L1/L2 input entries (config/sp493sim/
  tasks_sp493_sim_L{1,2}_in.json, = the B fork's committed entries); g3 = the B fork's committed entries @ 27c327b03, after
  asserting by key: (1) the as-run entry (@ 24e7dc722, t887/t962 @ af0c27a06) holds every committed field, equal; (2) it adds only
  the prep fields + attack_pair_launch (+ real_arms [attack] on t887/t962 only, dropped: 09 ~21:50); (3) its approach_sha256 =
  facts v2's (@ d800edc6a) by key = the committed q2_approach_sha256. After the prep, check_sp493_tail_built.py asserts each built
  entry equals the L1/L2 built entry (g6/g7) or the as-run entry less those fields (g3).
- Split (d9): within each environment, cells in sha256(seed:key) order are dealt alternately to X and Y (the side that starts
  alternates by environment), so both launches mix all groups and environments.
- Order within a launch (the interleave fix): each environment's cells in sha256(seed:key) order sit at fractional positions
  (j + u) / n_env (u from sha256(seed:env:j)); cells sorted by that position. check: the longest same-environment run.
- refindex_X/Y: refindex_L1/L2/L3's entries for the launch's keys, verbatim (same copy paths and pins); relays put the copies at
  those paths on the launch box (allowlists written here).
  python make_sp493_tail.py --seed 20261015
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
C = HERE / "config/sp493sim"
B = "experiments/2026-10-15_q3_sp493"
MEAN, BUF, RESERVE, K = 13.48, 1.3, 75, 30
PREP = {"approach_chars", "approach_s3", "approach_sha256", "template_sha256"}
BOX = {"X": ("xbox1", "i-04331f272d6778df0"), "Y": ("box3", "i-0cc7555acb95c694b")}
HOLDERS = {"box3": "i-0cc7555acb95c694b", "box4": "i-07bce857a818b6f69", "xbox1": "i-04331f272d6778df0",
           "scaleup4": "i-0ad4140e97b11d30f", "realhonest": "i-0abea93b8ad99ee41"}
H = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731


def git_json(commit, path):
    return json.loads(subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=HERE).stdout)


def main():
    global REFOLD  # the refs_old copies earlier allowlist relays put on box3 / box4 (their records: path + sha256)
    REFOLD = {b: {(f["path"], f["sha256"]) for f in json.loads((HERE / f"results/il_relay_sp493_refold_{b}_2026-10-15.json").read_text())["files"]}
              for b in ("box3", "box4")}
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    seed = ap.parse_args().seed
    split = json.loads((HERE / "results/sp493_sim_split_2026-10-15.json").read_text())["rows"]
    grp = {r["key"]: r["group"] for r in split if r["group"] in ("g6", "g7")}
    idx = {L: json.loads((C / f"refindex_{L}.json").read_text())["references"] for L in ("L1", "L2", "L3")}
    for k, g in list(grp.items()):
        assert k in idx["L1" if g == "g6" else "L2"], k
    for k in idx["L3"]:
        assert k not in grp
        grp[k] = "g3"
    assert Counter(grp.values()) == {"g6": 137, "g7": 121, "g3": 37}, Counter(grp.values())
    # entries fed to the prep
    ins = {L: {c["key"]: c for c in json.loads((C / f"tasks_sp493_sim_{L}_in.json").read_text())["cells"]} for L in ("L1", "L2")}
    committed = {c["key"]: c for b in "ABC" for c in git_json("27c327b03", f"{B}/config/tasks_sp493_box{b}.json")["cells"]}
    asrun = git_json("24e7dc722", f"{B}/results/asrun_honest_entries_2026-10-15_192615.json")["entries"]
    asrun |= git_json("af0c27a06", f"{B}/results/asrun_newmain_attackonly_2026-10-15_194135.json")["entries"]
    facts = {r["key"]: r for r in git_json("d800edc6a", f"{B}/results/sp493_facts_2026-10-15_165730.json")["rows"]}
    entry, g3_checks = {}, {}
    for k, g in grp.items():
        if g != "g3":
            entry[k] = ins["L1" if g == "g6" else "L2"][k]
            assert entry[k] == committed[k], k  # the L1/L2 inputs are the committed entries
            continue
        a, c = asrun[k], committed[k]
        extra = set(a) - set(c)
        allowed = PREP | {"attack_pair_launch", "_box"} | ({"real_arms"} if k in ("t887_auto_workflow", "t962_tiktok") else set())
        chk = {"1_committed_fields_equal": all(f in a and a[f] == c[f] for f in c),
               "2_only_enumerated_added": extra <= allowed and PREP <= extra,
               "3_approach_sha_eq_facts_v2": a["approach_sha256"] == facts[k]["approach_sha256"] == c["q2_approach_sha256"],
               "real_arms_dropped": a.get("real_arms") if "real_arms" in a else None}
        assert chk["1_committed_fields_equal"] and chk["2_only_enumerated_added"] and chk["3_approach_sha_eq_facts_v2"], (k, chk)
        assert ("real_arms" in a) == (k in ("t887_auto_workflow", "t962_tiktok")) and a.get("real_arms", ["attack"]) == ["attack"], k
        g3_checks[k] = chk
        entry[k] = c
    # split X / Y: per environment, alternate
    by_env = defaultdict(list)
    for k in grp:
        by_env[entry[k]["env"]].append(k)
    side = {}
    for e_i, env in enumerate(sorted(by_env)):
        for j, k in enumerate(sorted(by_env[env], key=lambda k: H(f"{seed}:{k}"))):
            side[k] = "XY"[(j + e_i) % 2]
    out = {}
    for L in "XY":
        keys = [k for k in grp if side[k] == L]
        envs = defaultdict(list)
        for k in sorted(keys, key=lambda k: H(f"{seed}:{k}")):
            envs[entry[k]["env"]].append(k)
        pos = {k: (j + int(H(f"{seed}:{env}:{j}")[:8], 16) / 16 ** 8) / len(ks) for env, ks in envs.items() for j, k in enumerate(ks)}
        order = sorted(keys, key=lambda k: (pos[k], H(f"{seed}:{k}")))
        run, longest = 1, 1
        for a, b in zip(order, order[1:]):
            run = run + 1 if entry[a]["env"] == entry[b]["env"] else 1
            longest = max(longest, run)
        name = f"sp493_tail_{L}"
        tf, rf, cf = C / f"tasks_{name}_in.json", C / f"refindex_{L}.json", C / f"pilot_{name}.yaml"
        for p in (tf, rf, cf):
            assert not p.exists(), f"{p} exists"
        tf.write_text(json.dumps({"launch": f"2026-10-15_{name}", "status": "S_paper493 sim tail (g6/g7) + L3 (g3); sims only (09: split @ 59f333d03, L3 = 37)",
                                  "source": "g6/g7: config/sp493sim/tasks_sp493_sim_L{1,2}_in.json; g3: aswin/q3-sp493 @ 27c327b03 (as-run asserts in make_sp493_tail.py)",
                                  "box": BOX[L][0], "order": f"per-environment fractional interleave, seed {seed} (make_sp493_tail.py)",
                                  "groups": {k: grp[k] for k in order}, "cells": [entry[k] for k in order]}, indent=1) + "\n")
        src = {"g6": "L1", "g7": "L2", "g3": "L3"}
        refs = {k: idx[src[grp[k]]][k] for k in order}
        rf.write_text(json.dumps({"launch": f"sp493_sim_refindex_{L}", "finished": "2026-10-15T22:15:00-07:00",
                                  "what": "reference index for --references-from: refindex_L1/L2/L3's pins for this launch's keys, verbatim",
                                  "references": refs}, indent=1) + "\n")
        n = len(order)
        cap = math.ceil(n * MEAN * BUF + K * RESERVE)
        cfg = (C / "pilot_sp493_sim_L1.yaml").read_text()
        body = cfg[cfg.index("tasks: "):]
        subs = [(r"(?m)^tasks: config/tasks_sp493_sim_L1\.json.*$", f"tasks: config/tasks_{name}.json   # S_paper493 sim tail + L3 {L}: the on-box prep's built cells (<= {n})"),
                (r"(?m)^  cap_usd: \d+ .*$", f"  cap_usd: {cap}   # {n} cells x ${MEAN} x 1.3 + K {K} x ${RESERVE} (d9: concurrency reserve)"),
                (r"(?m)^  worst_case_limit_usd: \d+ .*$", f"  worst_case_limit_usd: {cap}   # = cap")]
        for pat, rep in subs:
            body, cnt = re.subn(pat, rep, body)
            assert cnt == 1, pat
        cf.write_text(f"# S_PAPER493 SIM TAIL + L3 {L} ({BOX[L][0]}; make_sp493_tail.py): sims only, frozen IL design. From pilot_sp493_sim_L1.yaml:\n"
                      "# only this header, tasks and the caps differ (max_runs_per_server 5, real_arms [] unchanged).\n" + body)
        # relay allowlists: the reference copies not already at their path on this box, by holder
        need = defaultdict(dict)
        for k in order:
            r, g = refs[k], grp[k]
            holder = {"g6": "box3", "g7": "box4"}.get(g) or r["source_box"].split()[0]
            where = {holder}
            if "/refs_old/" in r["copy"]:  # il_refindex_old's copies (T196): on xbox1 (the source), and on every box an earlier
                holder, where = "xbox1", {"xbox1"} | {b for b, fs in REFOLD.items() if (r["copy"], r["sha256"]) in fs}  # relay put them (Codex r54)
            if BOX[L][0] not in where:
                need[holder][r["copy"]] = (r["sha256"], k)
        for holder, files in sorted(need.items()):
            alf = C / f"allowlist_tail_{L}_from_{holder}.json"
            assert not alf.exists()
            alf.write_text(json.dumps({"what": f"EXPLICIT ALLOWLIST (committed before the relay): {name}'s reference copies held on {holder} {HOLDERS[holder]}; destination {BOX[L][0]} {BOX[L][1]}",
                                       "written_pt": "2026-10-15T22:15:00-07:00", "index": f"config/sp493sim/refindex_{L}.json",
                                       "assertions": ["no cells.json, nothing under approaches/, no 'approach' in a path", "sha256 = the index pin"],
                                       "files": [{"path": p, "sha256": s, "role": "copy", "key": k} for p, (s, k) in sorted(files.items())]}, indent=1) + "\n")
        out[L] = {"box": BOX[L], "cells": n, "groups": dict(Counter(grp[k] for k in order)), "envs": dict(Counter(entry[k]["env"] for k in order)),
                  "longest_same_env_run": longest, "stagger_floor_min": round(max(Counter(entry[k]["env"] for k in order).values()) * 90 / 60, 1),
                  "cap": cap, "relays": {h: len(f) for h, f in need.items()},
                  "tasks_in_sha256": hashlib.sha256(tf.read_bytes()).hexdigest(), "refindex_sha256": hashlib.sha256(rf.read_bytes()).hexdigest(),
                  "config_sha256": hashlib.sha256(cf.read_bytes()).hexdigest()}
    (HERE / "results/sp493_tail_build_2026-10-15.json").write_text(json.dumps({"seed": seed, "launches": out, "g3_checks": g3_checks}, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
