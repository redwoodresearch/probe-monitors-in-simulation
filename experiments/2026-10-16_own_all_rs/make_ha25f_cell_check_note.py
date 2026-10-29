"""25f H_A cell check (4f / 2e, 10-16), an APPEND-ONLY NOTE by key. The first 25f inventory (v1) names the task "references" for its 3 rows; v2 (@ b880c3c3a)
takes it from the manifests. BY KEY: (a) the own_all_rs_uncut fit's training epochs for the 3 keys (rebuilt with run_own_all's own _uncut_training from
the fit record's frozen routing commit / inventory / chain commit, identity sha checked) carry cell = v2's task, with that cell's main and re-split
fold; (b) the scored 25f messages of the given score records (rebuilt with run_own_all's own code; message sha256 = the scored one) carry the same
cells, and, where a manifest copy is given, its scored units equal them. Git only (+ manifest copies sha-checked against their score records)."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V2 = ("b880c3c3a", "experiments/2026-10-15_cut_sensitivity/results/inventory_ha25f_v2_2026-10-16_044849.json", "c01a6ef18869cd09f1c2d3e441f0deb05143cb2c2f29db84f0d2049a80f8af7f")
V1 = ("85447113f", "experiments/2026-10-15_cut_sensitivity/results/inventory_ha25f_v1_2026-10-16_025414.json", "a626647f118a84ca181e54dc69872a31b9ebc3c90793c49192e56ef9d3ce8c08")
FIT = "experiments/2026-10-14_p2_family_expansion/results/own_all_rs_uncut_fit_2026-10-16_030614.json"
INPUTS = "experiments/2026-10-14_p2_family_expansion/results/own_all_inputs_2026-10-15.json"
RS_FOLDS = "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json"


def main(records, man_dirs, out_rel):
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    show = lambda c, p: subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
    spec = importlib.util.spec_from_file_location("oa", REPO / "experiments/2026-10-14_p2_family_expansion/run_own_all.py")
    oa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oa)
    v2b = show(V2[0], V2[1])
    if hashlib.sha256(v2b).hexdigest() != V2[2]:
        raise SystemExit("inventory_ha25f_v2 differs from its pin")
    task = {r["key"]: r["task"] for r in json.loads(v2b)["runs"]}
    ib = show("HEAD", INPUTS)
    inp = json.loads(ib)
    main_of = {}
    for c_, p_, s_ in (inp["pins"]["t196"], inp["pins"]["plus311"], inp["pins"]["nm360"]):
        main_of |= {c["key"]: c["main"] for c in json.loads(show(c_, p_))["cells"]}
    flb = show("HEAD", RS_FOLDS)
    fold = {m: v[0] for m, v in json.loads(flb)["by_main"].items()}
    fr = json.loads(show("HEAD", FIT))
    if hashlib.sha256(flb).hexdigest() != fr["rs"]["fold_list"]["sha256"] or hashlib.sha256(ib).hexdigest() != fr["inputs"]["sha256"]:
        raise SystemExit("the fold list or the inputs record differs from the fit's (Codex r26)")
    ru = fr["rs_uncut"]
    by_root, urep = oa._uncut_training(git, ru["routing_commit"], (ru["inventory_b"]["commit"], ru["inventory_b"]["path"], ru["inventory_b"]["sha256"]), inp,
                                       ru["training_report"]["chain_commit"])
    if json.dumps(urep, sort_keys=True) != json.dumps(ru["training_report"], sort_keys=True):
        raise SystemExit("the rebuilt training report differs from the fit record's (Codex r26)")
    train = {e["key"]: (e["cell"], e["run"]) for v in by_root.values() for e in v if e["key"] in task}
    a = {k: {"v2_task": task[k], "training_cell": train.get(k, (None,))[0], "root": train.get(k, (None, None))[1], "main": main_of.get(task[k]),
             "fold": fold.get(main_of.get(task[k])), "equal": train.get(k, (None,))[0] == task[k]} for k in sorted(task)}
    a_ok = len(train) == 3 and all(v["equal"] and v["fold"] is not None for v in a.values())  # bound: same inputs, fold list and training report as the fit
    full_main_of = dict(main_of)
    fb = show(oa.SP493_FACTS[0], oa.SP493_FACTS[1])
    full_main_of |= {r["key"]: r["main"] for r in json.loads(fb)["rows"] if r["key"] not in oa.SP493_OUT}
    b, b_ok = {}, True
    for rel, md in zip(records.split(","), man_dirs.split(","), strict=True):
        rb = (REPO / rel).read_bytes()
        if subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"], capture_output=True).stdout != rb:
            raise SystemExit(f"{rel}: not committed as read")
        rec = json.loads(rb)
        jx, _ = oa._cut_jobs_b(git, rec["routing_commit"], V1, (), sold=False, main_of_=full_main_of)
        hits = [(m, h, k) for m, h, k, _ in jx if f"{k}|{h}" in rec["results"]]
        cells = {e["key"]: e["cell"] for m, _, _ in hits for e in m["epochs"]}
        units = {}
        for m, h, k in hits:
            for fo, x in rec["results"][f"{k}|{h}"]["folders"].items():
                mp = Path(md) / f"{fo}.json"
                if md and mp.exists():
                    mb = mp.read_bytes()
                    if hashlib.sha256(mb).hexdigest() != x["manifest_sha256"]:
                        raise SystemExit(f"{fo}: manifest copy differs from {rel}")
                    mn = json.loads(mb)
                    units[fo] = sorted(set(mn["counts"]["R"]["per_run"]) | set(mn["counts"]["R"]["runs_without_rows"]))
        ok = len(hits) == 3 and set(cells) == set(task) and all(cells[k] == task[k] for k in task) and \
            all(u == [f"honest_a|{cells[e['key']]}" for e in m["epochs"]] for m, h, k in hits for fo, u in units.items() if fo in {e["folder"] for e in m["epochs"]})
        b_ok &= ok
        b[rel] = {"messages_matched_by_sha": len(hits), "cells_by_key": cells, "manifest_units": units, "pass": ok}
    note = {"label": "25f H_A cell check by key (4f / 2e, 10-16): own_all_rs_uncut training + scored 25f rows carry the manifests' tasks; keys only; APPEND-ONLY",
            "inventory_v2": {"commit": V2[0], "path": V2[1], "sha256": V2[2]}, "inventory_v1_scored_with": {"commit": V1[0], "path": V1[1], "sha256": V1[2], "task_field": "references (all 3)"},
            "cell_rule_used": "run_own_all._cut_jobs_b: a 'references' task takes the reference .eval's stem as its cell, checked against the routing record's members; "
                              "_uncut_training: the training epoch's own cell (fit inputs), with the routed member's cell required equal",
            "a_fit_training": a, "a_pass": a_ok, "b_scored": b, "b_pass": b_ok, "pass": a_ok and b_ok}
    (REPO / out_rel).open("x").write(json.dumps(note, indent=1))
    print(json.dumps({"a": {k.split("/")[-1]: (v["training_cell"], v["main"], v["fold"], v["equal"]) for k, v in a.items()}, "a_pass": a_ok,
                      "b": {r.split("/")[-1]: (x["messages_matched_by_sha"], x["pass"], {fo: len(u) for fo, u in x["manifest_units"].items()}) for r, x in b.items()}, "pass": note["pass"]}))
    if not note["pass"]:
        raise SystemExit("STOP")


if __name__ == "__main__":
    main(*sys.argv[1:4])
