"""DRAFT completion launches for the stalled IL launches (not for launch until 09 registers them and the gate change lands).

Selection by RUN STATE only, from each launch's final per-cell states (results/il_final_states/cells_<host>.json, read with
il_cells_read.py after the graceful stops), 09's il410 estimation keys and the pinned stage maps:
  paired  (real_arms [attack] + sim_attack): estimation cells with NEITHER arm started (real and sim "not run", 0 attempts);
  simonly (real_arms [], sim_attack only):   estimation cells whose sim2 never started (sim "not run", 0 attempts; real ran).
Excluded: the 70 paired RL reruns (their own launches, f368ebab), selection-half cells (held), and blockers (no pinned
reference: results/il_no_reference_cells_2026-10-15.json; il_old preflight failures t88, t169). Task files hold the cells'
ORIGINAL entries in seeded order (sha256('<seed>:<host>_c<kind>')); configs = the host's IL config with tasks, real_arms
(simonly only) and caps (cells x $20.5) changed. Writes config/draft_completion/ and its manifest.
  python3 make_completion_files.py
"""

import hashlib
import json
import math
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
C = HERE / "config"
OUT = C / "draft_completion"
SEED, RATE = "q3-interleaved-run3-sim2-2026-10-14", 20.5
HOSTS = {"q1": ("box1", "q1_all"), "q2": ("box2", "q2_all"), "q3": ("box3", "q3_all"), "q4": ("box4", "q4_all"), "old": ("xbox1", "old_all")}
BLOCKERS = {"t404_slack_clone", "t139_model_registry", "t151_model_registry", "t98_model_registry", "t88_model_registry", "t169_clinical_trial"}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def main() -> None:
    est = set(json.loads((C / "il_keys_estimation.json").read_text()))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"rule": __doc__.split("\n\n")[1].replace("\n", " "), "hosts": {}}
    for h, (box, il) in HOSTS.items():
        cells = json.loads((HERE / "results" / "il_final_states" / f"cells_{box}.json").read_text())[0]
        assert cells["finished"], f"{h}: launch not finished"
        rr = {b["cell"] for b in json.loads((C / f"ilrr_rerun_of_{h}.json").read_text())["rerun_of"]}
        doc = json.loads((C / f"tasks_il_{il}.json").read_text())
        pick = {"paired": set(), "simonly": set()}
        for k, a in cells["cells"].items():
            if k not in est or k in rr or k in BLOCKERS:
                continue
            r, s = a.get("real", {}), a.get("sim", {})
            if s.get("state") != "not run" or s.get("attempts"):
                continue
            if r.get("state") == "not run" and not r.get("attempts"):
                pick["paired"].add(k)
            elif r.get("state") in ("done", "no policy call"):
                pick["simonly"].add(k)
        manifest["hosts"][h] = {}
        for kind, keys in pick.items():
            sel = [c for c in doc["cells"] if c["key"] in keys]
            assert {c["key"] for c in sel} == keys, (h, kind)
            random.Random(int(hashlib.sha256(f"{SEED}:{h}_c{kind}".encode()).hexdigest(), 16)).shuffle(sel)
            tp, cp = OUT / f"tasks_ilc_{h}_{kind}.json", OUT / f"pilot_ilc_{h}_{kind}.yaml"
            tp.write_text(json.dumps({k: v for k, v in doc.items() if k not in ("cells", "stages", "stage")}
                                     | {"completion_of": f"2026-10-14_il_{h}", "selection": kind, "cells": sel}, indent=1) + "\n")
            cap = math.floor(len(sel) * RATE)
            text, n = (C / f"pilot_il_{il}.yaml").read_text(), 0
            subs = [(r"(?m)^tasks: .*$", f"tasks: config/tasks_ilc_{h}_{kind}.json   # DRAFT completion ({kind}): {len(sel)} cells"),
                    (r"(?m)^  cap_usd: .*$", f"  cap_usd: {cap}   # {len(sel)} cells x ${RATE}"),
                    (r"(?m)^  worst_case_limit_usd: .*$", f"  worst_case_limit_usd: {cap}   # = cap")]
            if kind == "simonly":
                subs.append((r"(?m)^real_arms: .*$", "real_arms: []   # sim-only completion; references as the original"))
            for pat, rep in subs:
                text, k = re.subn(pat, rep, text)
                n += k
            assert n == len(subs), (h, kind, n)
            cp.write_text(f"# DRAFT COMPLETION {h} {kind} (not registered; do not launch)\n" + text)
            manifest["hosts"][h][kind] = {"cells": len(sel), "cap_usd": cap, "tasks_sha256": sha(tp), "config_sha256": sha(cp),
                                          "keys": sorted(keys)}
    (OUT / "draft_completion_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({h: {k: v["cells"] for k, v in m.items()} for h, m in manifest["hosts"].items()}))


if __name__ == "__main__":
    main()
