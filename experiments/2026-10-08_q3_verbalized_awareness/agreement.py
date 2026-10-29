"""Agreement between the two coders (RUBRIC.md), before settling and before unmasking. Reads the codes only.

  python agreement.py --a data/codes_A.json --b data/codes_B.json
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
CATS = ("simulation", "evaluation", "monitoring", "stakes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    A, B = json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    assert set(A) == set(B) and len(A) == 14, "both coders must code the same 14 ids"
    out = {"per_category": {}, "disputes": [], "policy_seen_agree": sum(A[i]["policy_seen"] == B[i]["policy_seen"] for i in A)}
    for c in CATS:
        pairs = [(A[i][c]["summary"], B[i][c]["summary"]) for i in sorted(A)]
        out["per_category"][c] = {"agree": sum(a == b for a, b in pairs), "of": len(pairs), "pairs": dict(Counter(f"{a}|{b}" for a, b in pairs))}
        for i in sorted(A):
            if A[i][c]["summary"] != B[i][c]["summary"]:
                out["disputes"].append({"id": i, "category": c, "A": A[i][c], "B": B[i][c]})
    ic = [(A[i]["monitoring"].get("instructed_concealment"), B[i]["monitoring"].get("instructed_concealment")) for i in sorted(A)]
    out["instructed_concealment_agree"] = sum(a == b for a, b in ic)
    out["arm_guess_agree"] = sum(A[i]["arm_guess"] == B[i]["arm_guess"] for i in A)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path, text = HERE / "results" / f"agreement_{stamp}.json", json.dumps(out, indent=1)
    with path.open("x") as f:
        f.write(text)
    print(json.dumps({k: v for k, v in out.items() if k != "disputes"}, indent=1))
    for d in out["disputes"]:
        print("DISPUTE", d["id"], d["category"], "| A:", d["A"]["summary"], (d["A"].get("positive") or {}).get("paraphrase"),
              "| B:", d["B"]["summary"], (d["B"].get("positive") or {}).get("paraphrase"))
    print("->", path)


if __name__ == "__main__":
    main()
