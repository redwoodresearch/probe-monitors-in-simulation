"""The follow-ups' redesign (scoring plan amendment of 2026-10-10 23:30 PT): per-call files in the format of a hon or
simhon observed draw, built from the main run's per-call observed draw and honest_scores.json. $0, local.

honest_scores.json holds the honest A and B read-only calls and every simulated-honest call (so its 20-task roster
stays whole), scored by draw 0's five saved real-trained models after gate b passed on Modal. This appends those calls
to the main file's meta and adds the cells analyze_b_honest.py and analyze_simhon_calibration.py read: real->honA and real->honB (each call scored by its own test
fold's model) and, for simhon, each fold model k's scores on the simulated-honest calls of calibration fold k + 1. Every
other number is the main file's, so both analyses run unchanged on the output.

  python3 make_honest_percall.py <main per-call file> <honest_scores.json> hon|simhon
"""

import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent


def main(main_path: str, honest_path: str, kind: str):
    raw_m, raw_h = Path(main_path).read_bytes(), Path(honest_path).read_bytes()
    r, h = json.loads(raw_m), json.loads(raw_h)
    if h["choices"] != r["draws"][0]["choices"]:
        raise ValueError("the saved models' draw is not the main run's draw 0")
    worlds = ("honA", "honB") + (("simH",) if kind == "simhon" else ())
    meta, draw = list(r["meta"]), copy.deepcopy(r["draws"][0])
    cells = draw["cells"]
    for w in worlds:
        if f"real->{w}" in cells:
            raise ValueError(f"the main file already has real->{w}")
    cal = cells["real->real"]["per_call"]["calibration"]
    added = {w: {"row": [], "score": []} for w in worlds}
    if kind == "simhon" and h["simH_tasks"] != 20:
        raise ValueError(f"{h['simH_tasks']} simulated-honest tasks, not 20")
    for x in h["rows"]:
        if x["world"] not in worlds:
            continue
        if x["world"] != "simH" and x["y"] != 0:
            raise ValueError("honest A and B rows must be read-only")
        i = len(meta)
        meta.append({"world": x["world"], "task": x["task"], "irid": None, "call": None, "y": x["y"]})
        added[x["world"]]["row"].append(i)
        added[x["world"]]["score"].append(x["s"][x["fold"]])  # its test fold's model
        if x["world"] == "simH":  # the analysis keeps its read-only calls (y == 0) for the threshold
            k = (x["fold"] - 1) % 5  # the model whose calibration fold this call is in
            cal["row"].append(i), cal["score"].append(x["s"][k]), cal["model_fold"].append(k)
    for w in worlds:
        cells[f"real->{w}"] = {"per_call": added[w]}
    out = r | {"label": f"scaleup196_{kind}_from_saved_probes", "meta": meta, "draws": [draw],
               "built_from": {"main_per_call": {"file": Path(main_path).name, "sha256": hashlib.sha256(raw_m).hexdigest()},
                              "honest_scores": {"file": Path(honest_path).name, "sha256": hashlib.sha256(raw_h).hexdigest(), "key": h["key"],
                                                "gate": h["gate"]},
                              "calls_added": {w: len(added[w]["row"]) for w in worlds}}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"onpolicy_2x2_scaleup196_{kind}_saved_probes_draws0-0_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    print(json.dumps(out["built_from"]["calls_added"]), "->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("main_path")
    ap.add_argument("honest_path")
    ap.add_argument("kind", choices=["hon", "simhon"])
    a = ap.parse_args()
    main(a.main_path, a.honest_path, a.kind)
