"""S_paper493 sim-attack approach-provenance map v2 for the tail + L3 launches X (xbox1) and Y (box3): 295 cells by key (d9 ~01:00;
09's rule: provenance where the run happened). Keys and shas only, all from committed records; no approach text is read.
Per key: (1) the registered q2 sha (facts v2 @ d800edc6a, rows[].approach_sha256); (2) the sim on-box sha on the box that ran it
(the built task file tasks_sp493_tail_{X,Y}.json @ a400f9af0: prep_sp493_tail.sh's record_fleet_approaches output on xbox1 / box3,
which the prep itself checked against q2's sha); (3) the real arm's on-box sha: g6/g7 from map v1's real column
(sp493_sim_approach_map_2026-10-15.json @ c3931a963, from the box A/B/C v2 run lists), g3 from the B fork's as-run entries
(approach_sha256 @ 24e7dc722; t887/t962 @ af0c27a06). ADMITTED iff (1) = (2) = (3), all valid 64-hex; otherwise listed by key.
Keyed like v1 by the chain's row key sp493_sim|<cell>. Real-arm roots are not carried (v1's planned _try2 names were superseded).
  python make_sp493_approach_map_v2.py      # writes results/sp493_sim_approach_map_v2_2026-10-16.json (never overwrites)
"""

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "results/sp493_sim_approach_map_v2_2026-10-16.json"
SRC = {
    "facts_v2": ("d800edc6a", "experiments/2026-10-15_q3_sp493/results/sp493_facts_2026-10-15_165730.json"),
    "tasks_X": ("a400f9af0", "experiments/2026-10-15_q3_interleaved_run3_sim2/config/sp493sim/tasks_sp493_tail_X.json"),
    "tasks_Y": ("a400f9af0", "experiments/2026-10-15_q3_interleaved_run3_sim2/config/sp493sim/tasks_sp493_tail_Y.json"),
    "map_v1": ("c3931a963", "experiments/2026-10-15_sp493_sim_approach_map/results/sp493_sim_approach_map_2026-10-15.json"),
    "asrun_honest": ("24e7dc722", "experiments/2026-10-15_q3_sp493/results/asrun_honest_entries_2026-10-15_192615.json"),
    "asrun_attackonly": ("af0c27a06", "experiments/2026-10-15_q3_sp493/results/asrun_newmain_attackonly_2026-10-15_194135.json"),
}
SHA = re.compile(r"[0-9a-f]{64}")
ok = lambda v: isinstance(v, str) and SHA.fullmatch(v) is not None  # noqa: E731
EMPTY = hashlib.sha256(b"").hexdigest()


def blob(name):
    c, p = SRC[name]
    return subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=HERE).stdout


def main():
    assert not OUT.exists(), f"{OUT} exists"
    raw = {n: blob(n) for n in SRC}
    J = {n: json.loads(b) for n, b in raw.items()}
    facts = {r["key"]: r for r in J["facts_v2"]["rows"]}
    assert len(facts) == len(J["facts_v2"]["rows"]) == 493
    v1 = J["map_v1"]["map"]
    asrun = {**J["asrun_honest"]["entries"], **J["asrun_attackonly"]["entries"]}
    BOX = {"X": "xbox1 i-04331f272d6778df0", "Y": "box3 i-0cc7555acb95c694b"}
    rows, seen = [], set()
    for L in "XY":
        t = J[f"tasks_{L}"]
        assert t["dropped_at_preflight"] == [] and set(t["groups"]) == {c["key"] for c in t["cells"]}
        for c in t["cells"]:
            k, g = c["key"], t["groups"][c["key"]]
            assert k not in seen
            seen.add(k)
            q2 = facts[k]["approach_sha256"] if k in facts else None
            sim = c.get("approach_sha256")
            if g == "g3":
                real, real_src = (asrun[k]["approach_sha256"] if k in asrun else None), ("as-run @ af0c27a06" if k in J["asrun_attackonly"]["entries"] else "as-run @ 24e7dc722")
            else:
                e = v1.get(f"sp493_sim|{k}")
                real, real_src = (e["real_onbox_approach_sha256"] if e else None), f"map v1 real column (box {e['real_box'] if e else '?'})"
            reasons = [f"{n}: missing or not a 64-hex sha256" for n, v in (("q2", q2), ("sim", sim), ("real", real)) if not ok(v)]
            if not reasons:
                if sim != q2:
                    reasons.append("mismatch: sim on-box sha != registered q2 sha")
                if real != q2:
                    reasons.append("mismatch: real on-box sha != registered q2 sha")
            rows.append({"chain_key": f"sp493_sim|{k}", "key": k, "group": g, "sim_launch": f"2026-10-15_sp493_sim_{L}", "sim_box": BOX[L],
                         "q2_approach_sha256": q2, "sim_onbox_approach_sha256": sim, "real_onbox_approach_sha256": real, "real_source": real_src,
                         "approach_is_empty_string_sha": q2 == EMPTY, "admitted": not reasons, "reasons": reasons})
    assert len(rows) == 295
    counts = {}
    for r in rows:
        x = counts.setdefault(f"{r['sim_launch'][-1]}/{r['group']}", Counter())
        x["admitted" if r["admitted"] else "not_admitted"] += 1
    out = {"what": "S_paper493 sim-attack approach provenance map v2 (X/Y, 295 cells): registered q2 sha = sim on-box sha (the box that ran it) = real on-box sha; keys and shas only, committed records only",
           "rule": "09 via d9 (~01:00 10-16): admitted iff all three equal; any mismatch listed by key, not admitted; v2 goes to 09 for ack before the chain pins it",
           "sources": {n: {"commit": c, "path": p, "sha256": hashlib.sha256(raw[n]).hexdigest()} for n, (c, p) in SRC.items()},
           "counts_by_launch_group": {k: dict(v) for k, v in sorted(counts.items())},
           "counts_total": dict(Counter("admitted" if r["admitted"] else "not_admitted" for r in rows)),
           "not_admitted": [{"key": r["key"], "group": r["group"], "launch": r["sim_launch"], "reasons": r["reasons"]} for r in rows if not r["admitted"]],
           "n_empty_string_approach": sum(r["approach_is_empty_string_sha"] for r in rows),
           "empty_note": "the empty-string sha256 (e3b0c442...) is an intended empty approach, as in v1 (prep keeps empty approaches by design)",
           "keying": "by the chain's row key sp493_sim|<cell>, as v1; one entry per cell",
           "map": {r["chain_key"]: {k2: v for k2, v in r.items() if k2 != "chain_key"} for r in rows}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"counts": out["counts_by_launch_group"], "total": out["counts_total"], "not_admitted": out["not_admitted"][:5],
                      "empty": out["n_empty_string_approach"], "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}, indent=1))


if __name__ == "__main__":
    main()
