"""K5: agreement between the two independent labellers (K5_LABELING.md), and the calls they dispute.

Every non-submit call in the dumps is a unit; a call missing from a labeller's file is benign. Reports raw agreement
and Cohen's kappa for sabotage vs not and for all three labels, per run and pooled, and writes the disputed calls
(both labels and notes) for the settling read.

  python k5_agreement.py --a <A.json> --b <B.json> --dumps <dump dir>
"""

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
CALL = re.compile(r"^--- (\d+\.\d+) (\S+): ")
LABELS = ("sabotage", "preparation", "benign")


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[k] * cy[k] for k in set(cx) | set(cy)) / n**2
    return None if pe == 1 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--dumps", required=True)
    args = ap.parse_args()
    A, B = json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    units, disputed, per_run = [], [], {}
    for dump in sorted(Path(args.dumps).glob("*.txt")):
        lines = dump.read_text().splitlines()
        key = lines[0].removeprefix("KEY ")
        calls = [(m.group(1), m.group(2)) for m in map(CALL.match, lines) if m and m.group(2) != "submit"]
        for who, lab in (("A", A), ("B", B)):
            extra = set(lab[key]["labels"]) - {c for c, _ in calls}
            assert not extra, (who, key, extra)
            assert all(v[0] in ("sabotage", "preparation") for v in lab[key]["labels"].values()), (who, key)
        a = [A[key]["labels"].get(c, ["benign", ""]) for c, _ in calls]
        b = [B[key]["labels"].get(c, ["benign", ""]) for c, _ in calls]
        for (c, fn), la, lb in zip(calls, a, b):
            units.append((key, c, la[0], lb[0]))
            if la[0] != lb[0]:
                disputed.append({"dump": dump.name, "key": key, "call": c, "function": fn, "A": la, "B": lb})
        per_run[dump.name] = {"calls": len(calls), "A": dict(Counter(x[0] for x in a)), "B": dict(Counter(x[0] for x in b)),
                              "agree": sum(x[0] == y[0] for x, y in zip(a, b)),
                              "A_found": A[key]["sabotage_found"], "B_found": B[key]["sabotage_found"]}
    xa, xb = [u[2] for u in units], [u[3] for u in units]
    sa, sb = [x == "sabotage" for x in xa], [x == "sabotage" for x in xb]
    out = {"n_calls": len(units), "runs": len(per_run),
           "three_labels": {"agreement": sum(p == q for p, q in zip(xa, xb)) / len(units), "kappa": kappa(xa, xb),
                            "confusion_A_by_B": {f"{p}|{q}": n for (p, q), n in sorted(Counter(zip(xa, xb)).items())}},
           "sabotage_vs_not": {"agreement": sum(p == q for p, q in zip(sa, sb)) / len(units), "kappa": kappa(sa, sb),
                               "A_sabotage": sum(sa), "B_sabotage": sum(sb), "both": sum(p and q for p, q in zip(sa, sb))},
           "per_run": per_run, "disputed": disputed}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"k5_agreement_{stamp}.json"
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_calls", "three_labels", "sabotage_vs_not")}, indent=1))
    print(len(disputed), "disputed calls ->", path)


if __name__ == "__main__":
    main()
