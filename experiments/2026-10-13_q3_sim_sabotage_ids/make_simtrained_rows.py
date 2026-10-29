"""Training rows and labels for the sim-trained sabotage probe (RULE_simtrained.md), built from frozen inputs only. IDs,
labels, reading locations and numbers only (no text). Imported by run_simtrained.py, which builds the rows itself before
fitting, so the training set is always derived from the pinned inputs.

Joins Aswin's frozen panel votes (simulated; real runs 1 and 2) to the label-free job's rows file (reading locations) on
(run_id, epoch, turn, call_index). Every input's sha256 is read from its manifest, and every manifest is pinned here.
`train` = simulated, stage-1 training tasks, state-changing, label sabotage (y = 1) or not_sabotage (y = 0), with a reading.
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

MANIFEST_PINS = {  # each manifest's own sha256, as committed
    "manifest_sim_panel_votes_s3_v1.json": "15e8e151fe1feb5a2e27392bd694cba2037d7728a847938f140d8e19b2443f01",  # af1cf91e
    "manifest_real_panel_votes_s3_v1.json": "83ecbe6621b126a3daa04f61d0c3fb93adf3bf99d5fd0c0fa2f1404deb184482",  # 6fc9fcc8
    "manifest_flagrate_s3_v1.json": "22b1ac9570ca5663bda780d56c0342f259e6ae6c7e690882cef84b82b4892aad",  # 13abd4ba
    "manifest_gate_refit_v1.json": "20165be1b05c12033e153f9faba8bc83f75480795993537ea8a89bd4ca2456a9",  # 4d5d831d
    "manifest_gate_v1.json": "ef6653c3c610ed8f8c9526bcc6dd531b6910f370ebaa38d1c2bf0cb3339c3fd5",  # main, sabotage_test
}
LABELS = {"sabotage": 1, "not_sabotage": 0}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def manifests(mdir: str) -> dict:
    out = {}
    for name, pin in MANIFEST_PINS.items():
        p = Path(mdir) / name
        if sha(p) != pin:
            raise SystemExit(f"{name}: sha256 differs from its pin")
        out[name] = p
    return out


def manifest_sha(manifest: Path, name: str) -> str:
    """The sha256 a manifest records for the file whose basename is `name` (whatever field holds the path)."""
    files = json.loads(manifest.read_text())["files"]
    hits = [f["sha256"] for f in files if any(isinstance(v, str) and v.rsplit("/", 1)[-1] == name for k, v in f.items() if k != "sha256")]
    if len(hits) != 1:
        raise SystemExit(f"{manifest.name}: {len(hits)} entries for {name}")
    return hits[0]


def checked(manifest: Path, path: str) -> Path:
    p = Path(path)
    if sha(p) != manifest_sha(manifest, p.name):
        raise SystemExit(f"{p.name}: sha256 differs from {manifest.name}")
    return p


def build(mdir: str, sim_votes: str, real_votes: str, flag_rows: str) -> dict:
    m = manifests(mdir)
    sv = checked(m["manifest_sim_panel_votes_s3_v1.json"], sim_votes)
    rv = checked(m["manifest_real_panel_votes_s3_v1.json"], real_votes)
    fr = checked(m["manifest_flagrate_s3_v1.json"], flag_rows)
    key4 = lambda r: (r["run_id"], int(r["epoch"]), int(r["turn"]), int(r["call_index"]))  # noqa: E731
    votes = {}
    for path, default in ((sv, "sim"), (rv, None)):
        for r in csv.DictReader(open(path)):
            k = key4(r)
            if k in votes:
                raise SystemExit("a call appears twice in the votes")
            votes[k] = (r.get("world") or default, r["label"], int(r["state_changing"]))
    train, labels, miss = [], {}, Counter()
    for r in csv.DictReader(open(fr)):
        if r["part"] != "2":
            continue
        v = votes.get(key4(r))
        if v is None:
            miss[r["world"]] += 1
            continue
        world, label, sc = v
        if world != r["world"] or sc != int(r["state_changing"]):
            raise SystemExit(f"world or state_changing differs for {r['call_id']}")
        labels[r["call_id"]] = {"world": world, "task": r["task"], "group": r["group"], "state_changing": sc, "label": label}
        if world == "sim" and r["group"] == "stage1_train" and sc == 1 and label in LABELS:
            train.append({"x": r["x"], "chunk": r["chunk"], "row": int(r["row"]), "y": LABELS[label], "task": r["task"], "call_id": r["call_id"]})
    if miss:
        raise SystemExit(f"part-2 rows without a vote: {dict(miss)}")
    train.sort(key=lambda t: t["call_id"])
    sc_lab = Counter((v["world"], v["group"], v["label"]) for v in labels.values() if v["state_changing"] == 1)
    summary = {"inputs_sha256": {"sim_votes": sha(sv), "real_votes": sha(rv), "flag_rows": sha(fr)},
               "manifests_sha256": MANIFEST_PINS, "train_rows": len(train), "train_by_y": dict(Counter(t["y"] for t in train)),
               "train_tasks": len({t["task"] for t in train}), "train_tasks_with_sabotage": len({t["task"] for t in train if t["y"] == 1}),
               "part2_rows_labelled": len(labels),
               "state_changing_labels_by_world_group": {" / ".join(k): n for k, n in sorted(sc_lab.items())}}
    return {"summary": summary, "train": train, "labels": labels}
