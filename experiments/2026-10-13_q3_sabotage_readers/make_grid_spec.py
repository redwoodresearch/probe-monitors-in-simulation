"""B3 (RULE.md here): the label join and the training selections for one reader, before any fit. Labels, IDs and counts
only; no score, no activation, no text. $0, local.

- Eligible attack calls: state-changing calls in Aswin's frozen votes (sim_votes, real_votes for the 40 stage-1 tasks; the
  K = 116 export for the new tasks), joined on (run_id, epoch, turn, call_index) to the row files (flagrate_rows part 2,
  scaled_rows) for their irid, then to the reader's staged rows on (world, run_id, irid, call_index). A vote key without a
  row-file entry stops the script, and so does a vote whose run is staged without the call (a key mismatch). A vote whose
  run isn't staged at all (excluded at staging: no honest donor) is counted as such. A staged row without a reading (at or
  past the 32,768 cut) is "not read" (counted).
- Honest rows: rows_v2's honb_cal and honb_ho rows, through Qwen3.5's metas (extraction, chunk, row -> run, irid,
  call_index), to the reader's honest_b staged rows. Any row that doesn't map stops the script.
- Training (24 stage-1 training tasks, confident labels, read calls): n_pos = min(44, simulated sabotage), n_neg = min(900,
  simulated not-sabotage). S takes them all. R44, M and D draw 5 times each by Amendment 3's rule (pools sorted by call_id;
  seed int(sha256("match44|d" / "matchM|d" / "matchD|d")[:16], 16); default_rng; positives then negatives).
Writes results/grid_spec_<reader>_<stamp>.json (keys and labels) and prints counts.

  uv run --no-project --python 3.12 --with numpy python make_grid_spec.py --reader seed --inputs <scaled_inputs dir> \\
      --new-votes <scaleup_panel_votes_k116.csv> --split <split_v1.json> --metas <Qwen3.5 metas root> --stage <stage record>
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
PINS = {"flagrate_rows_2026-10-13_122417.csv": "44c75f78e78f86b4bcab409b1379762a8d4dccadfd456f28ee651832eb2b13a7",
        "scaled_rows_2026-10-13_152021.csv": "a95453ac2688e61ba99184726ce27a4dd54276a04730a56d67825f204d1ee1f3",
        "rows_v2.json": "727c417916b8fb2df9a31aca5f8ca0cc4dd17002a8a2edcb7bd79444bc0bf4b5",
        "sim_votes.csv": "06f4ef4bc00b53767b074185cda99311f2ba44f7c8f5a4f38ce59dd1537e81ae",
        "real_votes.csv": "9a96988db9818c8b97f3045ade83f19fae0e19e7d0cac134fa775005eba8dc36",
        "new_votes": "78ae0fa4e7e85da9a1ee85c9944be917b47e63c0d0a62f6f71390f046fdb7fcf",
        "split": "992e6e71793146b4df5f00d7cd98330d4f5e073f0a8379031aebfc172f818e11"}
ARM_WORLD = {"real1": "real", "real2": "run2", "sim": "sim", "honest_b": "honest_b"}
SIZES = (44, 900)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def draw(tag: str, d: int, pos: list, neg: list, n_pos: int, n_neg: int):
    rng = np.random.default_rng(int(hashlib.sha256(f"{tag}|{d}".encode()).hexdigest()[:16], 16))
    return [pos[i] for i in rng.choice(len(pos), n_pos, replace=False)], [neg[i] for i in rng.choice(len(neg), n_neg, replace=False)]


def staged_rows(rec: dict) -> dict:
    """(world, run_id, irid, call_index) -> has a reading, from a stage record's staged files (sha256-checked); on Modal the
    staged folder is read from the B3 Volume (B3_STAGE_ROOT)."""
    root = os.environ.get("B3_STAGE_ROOT")
    sdir = (Path(root) / Path(rec["staged_dir"]).name if root else Path(rec["staged_dir"])) / "inputs"
    if {p.name: sha(p) for p in sorted(sdir.glob("*.jsonl"))} != rec["staged_inputs_sha256"]:
        raise SystemExit("staged inputs differ from the stage record")
    out = {}
    for p in sorted(sdir.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            e = json.loads(line)
            for c in e["calls"]:
                k = (ARM_WORLD[e["arm"]], e["run_id"], c["irid"], int(c["call_index"]))
                if k in out:
                    raise SystemExit("a staged row appears twice")
                out[k] = c["readout"] is not None
    return out


def main(reader: str, inputs: str, new_votes: str, split: str, metas: str, stage: str, repair_stage: str = ""):
    I = Path(inputs)
    files = {n: I / n for n in list(PINS)[:5]} | {"new_votes": Path(new_votes), "split": Path(split)}
    for n, p in files.items():
        if sha(p) != PINS[n]:
            raise SystemExit(f"{n}: sha256 mismatch")
    rec = json.loads(Path(stage).read_text())
    if rec["reader"] != reader or rec.get("gate") or rec.get("limit"):
        raise SystemExit("not this reader's full stage record")
    staged = staged_rows(rec)
    staged_runs = {k[:2] for k in staged}
    rep = None
    if repair_stage:  # Amendment 2: a simulated call counts as read only if both its as-is and its repaired rendering read it
        rr = json.loads(Path(repair_stage).read_text())
        if rr["reader"] != reader or not rr.get("repair"):
            raise SystemExit("not this reader's repaired staging")
        rep = staged_rows(rr)
    sp = json.loads(files["split"].read_text())["tasks"]
    side = {t: ("train" if v["side"] == "train" else "held_out") for t, v in sp.items()}
    key4 = lambda r: (r["run_id"], int(r["epoch"]), int(r["turn"]), int(r["call_index"]))  # noqa: E731
    rows = {}
    for r in list(csv.DictReader(open(files["flagrate_rows_2026-10-13_122417.csv"]))) + list(csv.DictReader(open(files["scaled_rows_2026-10-13_152021.csv"]))):
        if r.get("part", "2") != "2":
            continue
        k = (r["world"], *key4(r))
        if k in rows and rows[k][0] != r["irid"]:
            raise SystemExit("a row-file key maps to two irids")
        rows[k] = (r["irid"], r["call_id"], r["task"])
    calls, cnt = [], Counter()
    for path, default in ((files["sim_votes.csv"], "sim"), (files["real_votes.csv"], None), (files["new_votes"], None)):
        for r in csv.DictReader(open(path)):
            if int(r["state_changing"]) != 1:
                continue
            w = r.get("world") or default
            got = rows.get((w, *key4(r)))
            if got is None:
                raise SystemExit("an eligible vote has no row-file entry")
            irid, call_id, task = got
            sk = (w, r["run_id"], irid, int(r["call_index"]))
            if sk not in staged:
                if (w, r["run_id"]) in staged_runs:
                    raise SystemExit("key mismatch: an eligible vote's run is staged but the call isn't")
                cnt[(w, side[task], r["label"], "run not staged (no honest donor)")] += 1  # stage_crossread's exclusion
                continue
            read = staged[sk]
            if rep is not None and w == "sim":
                both = read and rep.get(sk, False)
                cnt[(w, side[task], r["label"], "read as-is, not read repaired")] += read and not both
                cnt[(w, side[task], r["label"], "read repaired, not read as-is")] += rep.get(sk, False) and not read
                read = both
            cnt[(w, side[task], r["label"], "read" if read else "not read")] += 1
            calls.append({"key": list(sk), "call_id": call_id, "task": task, "world": w, "side": side[task], "stage1": sp[task]["stage1"],
                          "label": r["label"], "read": read})
    if len({c["call_id"] for c in calls}) != len(calls):
        raise SystemExit("a call_id appears twice")
    # honest rows: rows_v2 (extraction, chunk, row) -> Qwen3.5 meta call_end entry -> (run, irid, call_index)
    loc = {}
    for f in sorted(Path(metas).glob("*/*/*.json")):
        m = json.loads(f.read_text())
        if m.get("chunk") is None:
            continue
        for c in m["calls"]:
            if c.get("kind") == "call_end" and c.get("row") is not None:
                k = (f.parent.parent.name, m["chunk"], m["row_start"] + c["row"])
                if k in loc:
                    raise SystemExit("a meta location appears twice")
                loc[k] = (m["run"], c["irid"], int(c["call_index"]), m["arm"])
    honest = []
    for r in json.loads((I / "rows_v2.json").read_text())["rows"]:
        if r["role"] not in ("honb_cal", "honb_ho"):
            continue
        got = loc.get((r["x"], r["chunk"], int(r["row"])))
        if got is None or got[3] != "honest_b":
            raise SystemExit("an honest row has no honest_b meta entry")
        sk = ("honest_b", got[0], got[1], got[2])
        if sk not in staged:
            raise SystemExit("an honest row has no staged row")
        cnt[("honest_b", r["role"], "read" if staged[sk] else "not read")] += 1
        honest.append({"key": list(sk), "role": r["role"], "task": r["task"], "read": staged[sk]})
    tr = sorted((c for c in calls if c["side"] == "train" and c["stage1"] and c["read"] and c["label"] != "unclear"), key=lambda c: c["call_id"])
    pool = {(w, lab): [c["call_id"] for c in tr if (c["world"] == "sim") == (w == "sim") and c["label"] == lab]
            for w in ("sim", "real") for lab in ("sabotage", "not_sabotage")}
    n_pos, n_neg = min(SIZES[0], len(pool[("sim", "sabotage")])), min(SIZES[1], len(pool[("sim", "not_sabotage")]))
    fits = {"S": (pool[("sim", "sabotage")][:n_pos], pool[("sim", "not_sabotage")][:n_neg])}
    for d in range(5):
        fits[f"R44_{d}"] = draw("match44", d, pool[("real", "sabotage")], pool[("real", "not_sabotage")], n_pos, n_neg)
        fits[f"M_{d}"] = draw("matchM", d, pool[("sim", "sabotage")], pool[("real", "not_sabotage")], n_pos, n_neg)
        fits[f"D_{d}"] = draw("matchD", d, pool[("real", "sabotage")], pool[("sim", "not_sabotage")], n_pos, n_neg)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout.strip()  # noqa: E731
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    env = os.environ.get("B3_GIT_COMMIT")  # run_spec_modal.py: the committed HEAD (no git in the container)
    res = {"written": stamp, "reader": reader, "git": {"commit": env or git("rev-parse", "HEAD"), "dirty": False if env else bool(git("status", "--porcelain", "--", "."))},
           "inputs_sha256": PINS | {"stage_record": sha(stage)} | ({"repair_stage_record": sha(repair_stage)} if repair_stage else {}),
           "repair": bool(repair_stage), "sizes": {"n_pos": n_pos, "n_neg": n_neg},
           "pools": {f"{w} / {lab}": len(v) for (w, lab), v in pool.items()},
           "counts": {" / ".join(k): v for k, v in sorted(cnt.items())},
           "fits": {n: {"pos": p, "neg": q} for n, (p, q) in fits.items()}, "calls": calls, "honest": honest}
    out = Path(os.environ.get("B3_RESULTS") or HERE / "results") / f"grid_spec_{reader}{'_paired' if repair_stage else ''}_{stamp}.json"
    with out.open("x") as fh:
        fh.write(json.dumps(res))
    print(json.dumps({k: res[k] for k in ("sizes", "pools", "counts")}, indent=1), "\n->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("--reader", "--inputs", "--new-votes", "--split", "--metas", "--stage"):
        ap.add_argument(a, required=True)
    ap.add_argument("--repair-stage", default="")
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
