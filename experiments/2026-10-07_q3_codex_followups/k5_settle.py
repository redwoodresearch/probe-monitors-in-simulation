"""K5: merge the two labellers' labels into the settled set label_sabotage_dump.py builds from.

Agreed calls keep their label (labeller A's note). Disputed calls take the settlement file's label, note and rule. Every
disputed call must be settled, and nothing else may be. Writes results/sabotage_step_labels_manual_2026-10-07.json (the
name label_sabotage_dump.py reads).

  python k5_settle.py --a results/k5_labels_A_2026-10-08.json --b results/k5_labels_B_2026-10-08.json \
      --settlements results/k5_settlements_2026-10-08.json --agreement results/k5_agreement_<stamp>.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    for a in ("a", "b", "settlements", "agreement"):
        ap.add_argument(f"--{a}", required=True)
    args = ap.parse_args()
    A, B = json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    settled = json.loads(Path(args.settlements).read_text())["settled"]
    disputed = {f"{d['dump']} {d['call']}": d["key"] for d in json.loads(Path(args.agreement).read_text())["disputed"]}
    if set(disputed) != set(settled):
        raise ValueError(f"unsettled {set(disputed) - set(settled)}, not disputed {set(settled) - set(disputed)}")
    out = {}
    for key in A:
        labels = {}
        for call in set(A[key]["labels"]) | set(B[key]["labels"]):
            a = A[key]["labels"].get(call, ["benign", ""])
            b = B[key]["labels"].get(call, ["benign", ""])
            if a[0] == b[0]:
                labels[call] = a
        out[key] = {"summary_A": A[key]["summary"], "summary_B": B[key]["summary"], "labels": labels}
    for dc, key in disputed.items():
        lab, note, _rule = settled[dc]
        if lab != "benign":
            out[key]["labels"][dc.split(" ")[1]] = [lab, note]
    for key in out:
        out[key]["labels"] = dict(sorted(out[key]["labels"].items(), key=lambda kv: tuple(map(int, kv[0].split(".")))))
    path = HERE / "results" / "sabotage_step_labels_manual_2026-10-07.json"
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1))
    print(Counter(v[0] for r in out.values() for v in r["labels"].values()), "->", path)


if __name__ == "__main__":
    main()
