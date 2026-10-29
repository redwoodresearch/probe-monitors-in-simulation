"""Zoo C gate under amendment 1 (AMENDMENT_1.md): all disjoint same-batch overlap800 real pairs per task, with the
committed DE = max(1.25, 1.2 x Kish) recomputed on that pairing. Per-run positive counts from Aswin's numeric labels
(screening_labels_v1.csv's `credential` column, via #693's manifest) and the run list; it first reproduces gate_v3's four
counts. IDs and numbers only; $0.

  uv run --no-project python experiments/2026-10-11_q2_zoo_c_gate/analyze_zoo_c_gate_pairs.py data/zoo_c/screening_labels_v1.csv data/zoo_c/runs_phase1_honest.csv
"""
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
FROZEN = {"screening_labels_v1.csv": "edfeb7d43a104113453d5a768964b87b0ee8fc1bd9a6e286276e9b8fd58ae224",  # gate_v3's recorded inputs
          "runs_phase1_honest.csv": "75381ffed70ddc96ddf59b55eff3c7cf9df77db82063cea009a49bd5d37a83ce"}
GATE_V3 = {"simulated": 1385, "matched real (overlap800)": 2859, "same-batch real run 1": 546, "same-batch real run 2": 578}


def pass_prob(n_a: int, n_b: int, z: float, de: float) -> float:
    se = math.sqrt(0.25 * de * (1 / n_a + 1 / n_b))
    return max(0.0, math.erf((0.10 - z * se) / se / math.sqrt(2)))


def kish(per_task: dict) -> float:
    m = [v for v in per_task.values() if v > 0]
    return len(m) * sum(x * x for x in m) / sum(m) ** 2


def main(labels_p: str, runs_p: str):
    sha = {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (labels_p, runs_p)}
    if sha != FROZEN:
        raise SystemExit(f"inputs differ from gate_v3's recorded files: {sha}")
    pos = defaultdict(int)
    for r in csv.DictReader(open(labels_p)):
        pos[r["run_id"]] += int(r["credential"])
    runs = list(csv.DictReader(open(runs_p)))
    order_col = next(c for c in runs[0] if c.startswith("real_run_order"))
    batch = lambda rid: "overlap800" if "|overlap800_" in rid else "base200" if "|base200_" in rid else "other"
    sim_tasks = {r["main_task"] for r in runs if r["arm"] != "real-honest"}
    real = [r for r in runs if r["arm"] == "real-honest"]
    groups = {"simulated": [r for r in runs if r["arm"] != "real-honest"],
              "matched real (overlap800)": [r for r in real if batch(r["run_id"]) == "overlap800" and r["main_task"] in sim_tasks]}
    first2 = defaultdict(dict)
    for r in real:
        if r[order_col] in ("1", "2"):
            first2[r["main_task"]][r[order_col]] = r
    same = {t: d for t, d in first2.items() if len(d) == 2 and all(batch(x["run_id"]) == "overlap800" for x in d.values())}
    groups["same-batch real run 1"] = [d["1"] for d in same.values()]; groups["same-batch real run 2"] = [d["2"] for d in same.values()]
    got = {g: sum(pos[r["run_id"]] for r in rs) for g, rs in groups.items()}
    if got != GATE_V3:
        raise SystemExit(f"does not reproduce gate_v3's counts: {got}")
    by_task = defaultdict(list)
    for r in groups["matched real (overlap800)"]:
        by_task[r["main_task"]].append(r)
    A, B = [], []
    for t, rs in by_task.items():
        rs = sorted(rs, key=lambda r: r["run_id"])
        for i in range(0, len(rs) - 1, 2):
            A.append(rs[i]); B.append(rs[i + 1])
    groups["pairs, member A"], groups["pairs, member B"] = A, B
    n = {g: sum(pos[r["run_id"]] for r in rs) for g, rs in groups.items()}
    per_task = {g: defaultdict(int) for g in groups}
    for g, rs in groups.items():
        for r in rs:
            per_task[g][r["main_task"]] += pos[r["run_id"]]
    k = {g: kish(per_task[g]) for g in ("simulated", "matched real (overlap800)", "pairs, member A", "pairs, member B")}
    de = max(1.25, 1.2 * max(k.values()))
    p_rs = pass_prob(n["matched real (overlap800)"], n["simulated"], 2.24, de)
    p_rr = pass_prob(n["pairs, member A"], n["pairs, member B"], 1.96, de)
    at125 = (pass_prob(n["matched real (overlap800)"], n["simulated"], 2.24, 1.25), pass_prob(n["pairs, member A"], n["pairs, member B"], 1.96, 1.25))
    out = {"amendment": "AMENDMENT_1.md (a change for sample size, made before seeing results)", "inputs sha256": sha,
           "reproduces gate_v3": got, "pairs": len(A), "tasks with a pair": len({r["main_task"] for r in A}), "positives": n,
           "Kish design effect (positives within tasks)": k, "DE = max(1.25, 1.2 x largest Kish)": de,
           "real vs simulated, P(pass)": p_rs, "A vs B, P(pass)": p_rr, "predicted P(agree) (product)": p_rs * p_rr,
           "upper bound (smaller marginal)": min(p_rs, p_rr), "at DE 1.25 (sensitivity)": {"product": at125[0] * at125[1], "marginals": at125},
           "verdict": "gate passed under amendment 1" if p_rs * p_rr >= 0.8 else "too few to tell (amendment 1)",
           "original verdict on record": "too few to tell, 0.6996 (zoo_c_gate_2026-10-11_121752.json)"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"zoo_c_gate_pairs_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
