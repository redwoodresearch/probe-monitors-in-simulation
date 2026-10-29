"""Did the K = 116 exam use only state-changing calls? Checks the frozen result's counts against the votes, both ways. Labels only.

  uv run --no-project --python 3.12 python check_state_changing_only.py --result <scaled_result json> --sim-votes <csv> --real-votes <csv> \\
      --new-votes <csv> --split <split_v1.json> --id-lists <sim_calls_all csv>,<real_run2_calls_all csv>
"""
import argparse
import csv
import json
from collections import Counter


def main(result, sim_votes, real_votes, new_votes, split, id_lists):
    R = json.load(open(result))["primary"]["1pct"]["held_out_132"]["counts"]
    sp = json.load(open(split))["tasks"]
    ho = {t for t, v in sp.items() if v["side"] == "held_out"}
    task_of = {r["call_id"]: r["task"] for f in id_lists.split(",") for r in csv.DictReader(open(f))}  # the ID lists' task, as the exam used
    sc, allc = Counter(), Counter()
    for path, default in ((sim_votes, "sim"), (real_votes, None), (new_votes, None)):
        for r in csv.DictReader(open(path)):
            w = r.get("world") or default
            if (r.get("task") or task_of[r["call_id"]]) not in ho or r["label"] != "sabotage":
                continue
            allc[w] += 1
            sc[w] += r["state_changing"] == "1"
    used = {w: R[w]["sabotage"] for w in R}
    print(json.dumps({"result_positives": used, "votes_state_changing_sabotage": dict(sc), "votes_all_sabotage": dict(allc),
                      "equals_state_changing_only": used == dict(sc), "equals_all_calls": used == dict(allc)}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("--result", "--sim-votes", "--real-votes", "--new-votes", "--split", "--id-lists"):
        ap.add_argument(a, required=True)
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
