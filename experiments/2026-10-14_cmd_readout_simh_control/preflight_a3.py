"""RULE.md Amendment 3's local preflight, standard library only (no Modal import, no app, no spend). Run it before `modal run`;
run_controls_a3.main_a3 calls the same gate() again before any spawn.

The coverage gate: every run of run1, run2 and simulated attack in the corrected-B manifest (read with git show at its pinned commit)
must have exactly one entry in the attempt-label file (`labels` mapping, key "<world>|<task>", joined by run name, never by path);
duplicate keys anywhere in the file, a missing entry, or unexpected run counts stop it. Then the restriction to runs labelled exactly
"attempted", run-name and key uniqueness, and the #36 fold of each run (runs36.json only). Prints per-world counts; label values only.

  python3 experiments/2026-10-14_cmd_readout_simh_control/preflight_a3.py --r36 <runs36.json> --labels <claimc_labels_reduced.json>
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = "config/inputs_v1.json"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
R36_SHA = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
LABELS_SHA = "d00e88773fc2a71bef51e1ff0ccc2559932f5c3a306a8f5604c5248e797dc995"
WORLDS = {"run1": "run1", "run2": "run2", "sim": "sim_attack"}  # manifest world -> batch label (Amendment 2)
EXPECT_ALL = {"run1": 193, "run2": 192, "sim_attack": 191}  # the manifest's runs before the restriction
FIELDS = ("key", "world", "task", "arm", "extraction_run", "staged_inputs_run", "meta_file")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_bytes(spec: dict) -> bytes:  # run_simh_control._git_bytes
    b = subprocess.run(["git", "show", f"{spec['commit']}:{spec['path']}"], capture_output=True, check=True, cwd=REPO).stdout
    if hashlib.sha256(b).hexdigest() != spec["sha256"]:
        raise SystemExit(f"{spec['path']} @ {spec['commit'][:9]}: sha256 differs from the config's")
    return b


def gate(r36: str, labels: str, config: str = CONFIG) -> dict:
    """The coverage gate and the restricted epoch list (raises SystemExit on any failure)."""
    if sha(r36) != R36_SHA:
        raise SystemExit("runs36.json sha256 mismatch")
    if sha(labels) != LABELS_SHA:
        raise SystemExit("labels file sha256 mismatch")
    cfg = json.loads((HERE / config).read_text())
    mf = json.loads(git_bytes(cfg["real"]["manifest"]))
    epochs = [{k: e[k] for k in FIELDS} | {"batch": WORLDS[e["world"]], "cls": "attack", "vol": "b"} for e in mf["epochs"] if e["world"] in WORLDS]
    dups = []

    def no_dups(pairs):
        seen = {}
        for k, v in pairs:
            if k in seen:
                dups.append(k)
            seen[k] = v
        return seen

    lab_all = json.loads(Path(labels).read_text(), object_pairs_hook=no_dups)
    lab = lab_all["labels"]
    names = {}
    for e in epochs:
        names.setdefault(e["world"], []).append(f"{e['world']}|{e['task']}")
    cov = {w: {"runs": len(ns), "labelled_attempted": sum(lab.get(n) == "attempted" for n in ns),
               "labelled_not": sum(n in lab and lab[n] != "attempted" for n in ns), "missing": sum(n not in lab for n in ns),
               "label_values": {v: sum(lab.get(n) == v for n in ns) for v in sorted({str(lab[n]) for n in ns if n in lab})}} for w, ns in sorted(names.items())}
    run_counts = {WORLDS[w]: v["runs"] for w, v in cov.items()}
    out = {"coverage": cov, "duplicate_keys": len(dups), "lab_ref": lab_all.get("ref")}
    if dups or any(v["missing"] for v in cov.values()) or run_counts != EXPECT_ALL:
        raise SystemExit(f"coverage gate FAILED (missing or duplicate labels, or unexpected run counts): STOP, nothing launched: {json.dumps(out)}")
    fold_of_main = {}
    for r in json.loads(Path(r36).read_text()):  # folds: r36 ONLY
        if fold_of_main.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("a main task in two folds")
    main_of = {c["key"]: c["main"] for c in json.loads((REPO / T196).read_text())["cells"]}
    kept = [e for e in epochs if lab[f"{e['world']}|{e['task']}"] == "attempted"]
    for e in kept:
        e["name"] = f"{e['world']}|{e['task']}"
        e["fold"] = fold_of_main[main_of[e["task"]]]  # KeyError = a task outside T196 or without a #36 fold: stop
    if len({e["key"] for e in kept}) != len(kept) or len({e["name"] for e in kept}) != len(kept):
        raise SystemExit("duplicate epoch key or run name")
    kept.sort(key=lambda e: e["name"])
    keys = {e["key"] for e in kept}
    shas = {e["staged_inputs_file"]: e["staged_inputs_sha256"] for e in mf["epochs"] if e["key"] in keys}
    want = {}
    for e in kept:
        want.setdefault(e["staged_inputs_run"], {}).setdefault(e["arm"], []).append(e["key"])
    expect = {WORLDS[w]: v["labelled_attempted"] for w, v in cov.items()}
    if {b: sum(e["batch"] == b for e in kept) for b in expect} != expect:
        raise SystemExit("the restricted epoch list differs from the gate's counts")
    return out | {"epochs": kept, "expect": expect, "shas": shas, "want": want, "config": config}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r36", required=True)
    ap.add_argument("--labels", required=True)
    a = ap.parse_args()
    g = gate(a.r36, a.labels)
    print(json.dumps({"coverage_gate": g["coverage"], "duplicate_keys": g["duplicate_keys"], "expect": g["expect"], "kept_epochs": len(g["epochs"]),
                      "tasks": len({e["task"] for e in g["epochs"]}), "label_ref": g["lab_ref"], "passes": True}))
    sys.exit(0)
