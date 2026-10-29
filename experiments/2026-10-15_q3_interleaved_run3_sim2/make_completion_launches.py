"""Completion launches for the five IL launches stopped at 02:43-02:45 PT 10-15 (0d's GO on priorities 1-3; d9 02:50:
09's option (a): ONE paired completion launch per stopped launch = tier 2 and tier 3, both arms, interleaved).

Selection rule (by RUN STATE only; no score, no outcome):
  source    each stopped launch's per-cell run states (real and simulated: state and attempt count), from its launch.json at
            the sha256 in --shas (read on the host with il_cells_read.py; identical to results/il_final_states/cells_<box>.json);
  universe  09's il410 estimation keys (config/il_keys_estimation.json) minus the blockers (no pinned reference,
            results/il_no_reference_cells_2026-10-15.json; il_old preflight failures t88, t169); selection-half cells are
            skipped (held);
  tier 1    the 70 paired rate-limit reruns (their own launches, config/ilrr_rerun_of_<h>.json): excluded here;
  tier 2    estimation cells with NEITHER arm started (real and simulated both "not run" with 0 attempts);
  tier 3    estimation cells whose sim2 never started (simulated "not run", 0 attempts) while run3 ran (done or no policy
            call): both arms run again in the completion launch; the original run3 stays in accounting and is superseded
            by the keys-only supersession map (config/completion/ilc_supersession_tier3_<h>.json) committed before scoring;
  il_order  a cell's 0-based position in its launch's IL task file (config/tasks_il_<h>_all.json, the pre-registered
            seeded order the launcher followed). Each completion task file holds tier 2 in il_order, then tier 3 in
            il_order (the launcher starts tasks in strict file order);
  budget    cells are priced at --price-usd each (the sim2 arm; run3 runs on our servers, priced per server-hour in the
            headroom line) and taken in the global order tier 2 then tier 3, within a tier by (il_order, host order
            q1 q2 q3 q4 old), while the running sum stays <= the usable expected headroom read from --headroom-json
            (il_headroom.py: headroom minus the 10% reserve of the ceiling). Cells past the cutoff are written as
            explicit "held for Aswin" rows, never dropped.
Placement (--place host=box, in launch order per box; 0d names the host per completion launch, d9 03:31): each completion
runs after the box's own launches tonight and after any completion placed on the same box earlier in --place. A host
placed off its own box needs its main (launch.json + honest_a reference copies) and its original's launch.json at the
same paths on the target: q4 on box 1 (relayed for q4's reruns, 03:23-03:25, results/il_relay_q4_box1_2026-10-15.json),
q1 on xbox2 and q3 on xbox3 (relay_box_to_box.sh: honest_a references only, hash manifest, no parsing, laptop cache
deleted after the destination verifies). ilc_box.sh pins the main's and the original's launch.json sha256s either way.
Task files hold the cells' ORIGINAL entries verbatim; each config is the launch's IL config with only tasks and the spend
lines changed. Cap (0d 03:52, K = 10): max(cells x $20.5, cells x --price-usd + K x $75). Reason: the frozen launcher reserves
~$75 per unfinished simulated run against the cap and starts tasks in strict file order, so it runs about
(cap - spend) / $75 cells at once; a cap below $75 admits nothing (il_s2_rlr, cap $41, ended at 03:21 with 0 runs) and
cells x $20.5 alone throttles a small launch to a few cells at a time. check_admission.py asserts cap >= K x $75 before launch. Writes config/completion/ (never overwriting).
  python3 make_completion_launches.py --shas results/il_stopped_launches_2026-10-15.json --headroom-json results/il_headroom_2026-10-15_0312.json --price-usd P \
      --place q4=box1,q2=box2,old=xbox1,q1=xbox2,q3=xbox3 --cap-k 10
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).parent
C = HERE / "config"
OUT = C / "completion"
RATE = 20.5
RESERVE = 75.0  # the launcher's admission reserve per unfinished simulated run (worst_per_sim_run_usd)
HOSTS = {"q1": ("box1", "q1_all"), "q2": ("box2", "q2_all"), "q3": ("box3", "q3_all"), "q4": ("box4", "q4_all"),
         "old": ("xbox1", "old_all")}
PREFLIGHT_FAILED = {"t88_model_registry", "t169_clinical_trial"}  # il_old preflight failures (RULE.md)
R311, RNM = "/home/aswin/phase2/runs", "/home/aswin/phase2/nm/runs"
MAINS = {  # references_from per host, and its launch.json sha256 read on the original box (each main finished long ago)
    "q1": (f"{R311}/2026-10-14_154401_scaleup311_q1", "7e8227cd859502e60ad718fc60215410ece9d4b86eea04040bd7bee0a82c53ac"),
    "q2": (f"{R311}/2026-10-14_154401_scaleup311_q2", "fa489bf06d2be9c4a2b5c432ded7fb6065df49763618fb19e96a5f5d01cf03fb"),
    "q3": (f"{R311}/2026-10-14_155838_scaleup311_q3", "289ffb1030625f24666b1c9fb86a2ad577cf66a55407b9f18db125f0ed121701"),
    "q4": (f"{R311}/2026-10-14_160246_scaleup311_q4", "f12ed4cd86ed516036bcc47857c7a4d90ba95e21595a8c8f484653b9055a9ad6"),
    "old": ("/home/aswin/phase2/il_refindex_old", "79acbfc1ef4523f9a57cc19dc876252159c6c1d62521001f1ac71e2644036940"),
}
BOX_PREVS = {  # each target box's own launches that must have finished before its completions start (its order tonight)
    "box1": [f"{R311}/2026-10-14_il_q1", f"{R311}/2026-10-15_il_q1_rlr", f"{R311}/2026-10-15_il_q4_rlr2"],  # q4_rlr never ran (addendum)
    "box2": [f"{R311}/2026-10-14_il_q2", f"{R311}/2026-10-15_il_q2_rlr"],
    "box3": [f"{R311}/2026-10-14_il_q3", f"{R311}/2026-10-15_il_q3_rlr", f"{RNM}/2026-10-14_il_s3"],
    "xbox1": [f"{R311}/2026-10-14_il_old", f"{R311}/2026-10-15_il_old_rlr"],
    "xbox2": [f"{RNM}/2026-10-14_il_s1", f"{RNM}/2026-10-14_il_s4d", f"{RNM}/2026-10-15_il_s1_rlr2"],  # il_s1_rlr never started
    "xbox3": [f"{RNM}/2026-10-14_il_s2", f"{RNM}/2026-10-15_il_s2_rlr", f"{RNM}/2026-10-15_il_s2_rlr2"],
    "astra": [f"{RNM}/2026-10-14_il_s5"],  # placement addendum (d9/0d 04:53): oldc on sims-astra after s5
}
RELAYED = {("q4", "box1"), ("q1", "xbox2"), ("q3", "xbox3"), ("old", "astra")}  # (host, box) pairs whose main and original are already on the box by a verified relay
MANIFEST = {"box2": "harness_manifest_fu_q2r.tsv"}  # else harness_manifest_fu.tsv (the freeze check passes on each, 03:04)
CAPSTOP_SHA = "6689a761a6a5d879310dcb6c7e0e10eea8840dd4eb88fd44b9b3403222ba1ccb"  # sim_cap_stop.py, aswin/q3-scaleup-311
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shas", required=True)
    ap.add_argument("--headroom-json", required=True)
    ap.add_argument("--price-usd", type=float, required=True)
    ap.add_argument("--place", required=True)
    ap.add_argument("--cap-k", type=int, required=True)
    a = ap.parse_args()
    porder = [tuple(x.split("=")) for x in a.place.split(",")]
    place = dict(porder)
    assert len(porder) == len(place) and set(place) == set(HOSTS) and set(place.values()) <= set(BOX_PREVS), place
    assert all(place[h] == box or (h, place[h]) in RELAYED for h, (box, _) in HOSTS.items()), "a placement needs a new relay"
    shas = json.loads(Path(a.shas).read_text())["launches"]
    est = set(json.loads((C / "il_keys_estimation.json").read_text()))
    sel = set(json.loads((C / "il_keys_selection.json").read_text()))
    noref = json.loads((HERE / "results" / "il_no_reference_cells_2026-10-15.json").read_text())
    blockers = {r["key"] for lj in noref["launches"] for r in lj["rows"]} | PREFLIGHT_FAILED
    assert noref["all_expected"] is True and len(blockers) >= 6, "no-reference file"
    tiers, recon, src = {}, {}, {}
    for h, (box, il) in HOSTS.items():
        st = json.loads((HERE / "results" / "il_final_states" / f"cells_{box}.json").read_text())
        assert len(st) == 1 and st[0]["finished"] and st[0]["run"].endswith(f"/2026-10-14_il_{h}"), h
        assert shas[h]["cells_identical_to_committed"] is True and shas[h]["run"] == st[0]["run"], h
        cf = f"results/il_final_states/cells_{box}.json"
        src[h] = {"run": st[0]["run"], "launch_json_sha256": shas[h]["launch_json_sha256"], "finished": st[0]["finished"],
                  "cells_file": cf, "cells_file_sha256": sha(HERE / cf)}
        cells = st[0]["cells"]
        order = [c["key"] for c in json.loads((C / f"tasks_il_{il}.json").read_text())["cells"]]
        assert sorted(order) == sorted(cells) and len(set(order)) == len(order), h
        rr = {b["cell"] for b in json.loads((C / f"ilrr_rerun_of_{h}.json").read_text())["rerun_of"]}
        cls, n_real_nr, n_sim_nr = {}, 0, 0
        for k in order:
            r, s = cells[k]["real"], cells[k]["sim"]
            rn = r["state"] == "not run" and r["attempts"] == 0
            sn = s["state"] == "not run" and s["attempts"] == 0
            n_real_nr += rn
            n_sim_nr += sn
            if k in rr:
                c = "tier 1 (paired rate-limit rerun)"
            elif k in blockers:
                c = "blocker"
            elif k in sel:
                c = "selection half (held)"
            elif k not in est:
                c = "outside il410"
            elif rn and sn:
                c = "tier 2"
            elif sn and r["state"] in ("done", "no policy call"):
                c = "tier 3"
            elif not rn and not sn:
                c = "ran (both arms started)"
            else:
                c = "other"
            cls[k] = (c, rn, sn)
        assert not [k for k, v in cls.items() if v[0] in ("other", "outside il410")], (h, "unclassified cells")
        recon[h] = {"real_not_run": n_real_nr, "sim_not_run": n_sim_nr, "by_class": {}}
        for c, rn, sn in cls.values():
            key = f"{c} | real {'not run' if rn else 'started'} | sim {'not run' if sn else 'started'}"
            recon[h]["by_class"][key] = recon[h]["by_class"].get(key, 0) + 1
        tiers[h] = {t: [(order.index(k), k) for k in order if cls[k][0] == t] for t in ("tier 2", "tier 3")}
    # the global budget order and cutoff
    glob = [(t, i, list(HOSTS).index(h), h, k) for t in ("tier 2", "tier 3") for h in HOSTS for i, k in tiers[h][t]]
    glob.sort(key=lambda x: (x[0], x[1], x[2]))
    hj = json.loads(Path(a.headroom_json).read_text())
    budget, run, take, held = hj["usable_expected_usd"], 0.0, set(), []  # read, never retyped (Codex r19)
    for t, i, _, h, k in glob:
        if run + a.price_usd <= budget:
            run += a.price_usd
            take.add((h, k))
        else:
            held.append({"host": h, "cell": k, "tier": t, "il_order": i, "status": "held for Aswin (past the budget cutoff)"})
    OUT.mkdir(parents=True, exist_ok=True)
    ins = [C / "il_keys_estimation.json", C / "il_keys_selection.json", HERE / "results" / "il_no_reference_cells_2026-10-15.json", Path(a.shas)]
    man = {"rule": __doc__.split("\n\n")[1], "inputs": [{"path": str(f.relative_to(HERE)) if f.is_absolute() else str(f), "sha256": sha(f)} for f in ins],
        "cap_rule": {"k": a.cap_k, "reserve_usd": RESERVE, "formula": "max(cells x 20.5, cells x price + k x reserve)"},
        "budget": {"headroom_json": {"path": a.headroom_json, "sha256": sha(a.headroom_json)}, "usable_headroom_usd": budget, "cutoff_usd": budget, "price_per_cell_usd": a.price_usd,
                   "taken_cells": len(take), "taken_usd": round(run, 2), "held": held},
        "sources": src, "reconciliation": recon, "hosts": {}}
    for h, (box, il) in HOSTS.items():
        doc = json.loads((C / f"tasks_il_{il}.json").read_text())
        by_key = {c["key"]: c for c in doc["cells"]}
        keys = [k for t in ("tier 2", "tier 3") for _, k in tiers[h][t] if (h, k) in take]
        tp, cp, sp = OUT / f"tasks_ilc_{h}.json", OUT / f"pilot_ilc_{h}.yaml", OUT / f"ilc_supersession_tier3_{h}.json"
        for p in (tp, cp, sp):
            assert not p.exists(), f"{p} exists"
        tp.write_text(json.dumps({k: v for k, v in doc.items() if k not in ("cells", "stages", "stage")}
                                 | {"completion_of": f"2026-10-14_il_{h}", "completion_rule": "tier 2 then tier 3, each in il_order",
                                    "cells": [by_key[k] for k in keys]}, indent=1) + "\n")
        cap = max(math.floor(len(keys) * RATE), round(len(keys) * a.price_usd + a.cap_k * RESERVE))
        text, n = (C / f"pilot_il_{il}.yaml").read_text(), 0
        for pat, rep in ((r"(?m)^tasks: .*$", f"tasks: config/tasks_ilc_{h}.json   # completion: {len(keys)} cells (tier 2 then tier 3)"),
                         (r"(?m)^  cap_usd: .*$", f"  cap_usd: {cap}   # max({len(keys)} cells x ${RATE}, {len(keys)} x ${a.price_usd} + {a.cap_k} x ${RESERVE:.0f})"),
                         (r"(?m)^  worst_case_limit_usd: .*$", f"  worst_case_limit_usd: {cap}   # = cap")):
            text, k = re.subn(pat, rep, text)
            n += k
        assert n == 3, (h, n)
        cp.write_text(f"# IL COMPLETION {h} (tier 2 + tier 3, both arms): from pilot_il_{il}.yaml with only tasks and caps changed.\n" + text)
        t3 = [k for _, k in tiers[h]["tier 3"] if (h, k) in take]
        sp.write_text(json.dumps({"host": h, "original_launch": src[h]["run"], "original_launch_json_sha256": src[h]["launch_json_sha256"],
                                  "replacement_launch": f"2026-10-15_il_{h}c", "replacement_cell_list_sha256": sha(tp),
                                  "rows": [{"cell": k, "arm": "run3", "key": f"run3|{k}", "authoritative": "replacement",
                                            "original": {"launch": f"2026-10-14_il_{h}", "run_key": f"attack/{k}"},
                                            "replacement": {"launch": f"2026-10-15_il_{h}c", "run_key": f"attack/{k}"},
                                            "replacement_state_done": None,
                                            "rerun_reason": "tier 3: sim2 never started (stall); run3 rerun with sim2 in one launch (09 option a)"}
                                           for k in t3]}, indent=1) + "\n")
        man["hosts"][h] = {"cells": len(keys), "cap_usd": cap,
                           "tier 2": [{"cell": k, "il_order": i} for i, k in tiers[h]["tier 2"] if (h, k) in take],
                           "tier 3": [{"cell": k, "il_order": i} for i, k in tiers[h]["tier 3"] if (h, k) in take],
                           "tasks_sha256": sha(tp), "config_sha256": sha(cp), "supersession_sha256": sha(sp)}
    gate = HERE / "gate_pause_311_rl.py"
    cases = []
    for i, (h, tb) in enumerate(porder):
        box, il = HOSTS[h]
        m = man["hosts"][h]
        orig = src[h]["run"]
        earlier = [f"{R311}/2026-10-15_il_{h2}c" for h2, tb2 in porder[:i] if tb2 == tb]
        prevs = [orig] + [x for x in BOX_PREVS[tb] if x != orig] + earlier
        m |= {"placed_on": tb, "relay": tb != box, "references_from": MAINS[h][0], "main_launch_json_sha256": MAINS[h][1]}
        cases.append(f'  {h}) KIND=311; MAIN={MAINS[h][0]}; ORIG={orig}; PREVS="{" ".join(prevs)}"; '
                     f'MANIFEST=/home/aswin/phase2/attack_check_311/config/{MANIFEST.get(tb, "harness_manifest_fu.tsv")}; '
                     f'CAP={m["cap_usd"]}; TSHA={m["tasks_sha256"]}; CSHA={m["config_sha256"]}; SSHA={m["supersession_sha256"]}; '
                     f'ILTASKS=config/tasks_il_{il}.json; ILTSHA={sha(C / f"tasks_il_{il}.json")}; RSHA={MAINS[h][1]}; '
                     f'OSHA={src[h]["launch_json_sha256"]}; CAPSTOP_SHA={CAPSTOP_SHA}; GATE_SHA={sha(gate)}; '
                     f'ILTASKS_STAGE={"staged" if tb != box else "-"} ;;  # on {tb}{" (placed: main and original relayed)" if tb != box else ""}')
    man["gate_pause_311_rl_sha256"] = sha(gate)
    (HERE / "ilc_box.sh").write_text((HERE / "ilc_box.sh.tmpl").read_text().replace("__CASES__", "\n".join(cases)))
    mp = OUT / "completion_manifest.json"
    assert not mp.exists()
    mp.write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({h: (v["cells"], len(v["tier 2"]), len(v["tier 3"]), v["cap_usd"]) for h, v in man["hosts"].items()}),
          "held", len(held), "taken_usd", round(run, 2), "cutoff", round(budget, 2))


if __name__ == "__main__":
    main()
