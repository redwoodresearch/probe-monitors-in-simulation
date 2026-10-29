"""09's FINAL tail assert (01:2x PT 10-16), run before ANY S_paper493 g3/g6/g7 tail row is read: the admitted tail cells (from the
committed sp493 run records) together with the cells listed by key in an exclusion-class file must equal the 295 pinned tail cells
(258 moved + L3's 37); every gap classed. Writes results/sp493_tail_final_assert_<stamp>.json (exclusive create) for the read's record;
exits nonzero on STOP. Git and JSON only.

  python3 -I final_tail_assert.py --exclusions <json {"<cell key>": "<exclusion class>", ...}> --exclusions-source <who / path@sha>
"""

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_s = importlib.util.spec_from_file_location("sp493_rules", HERE / "sp493_rules.py")
R = importlib.util.module_from_spec(_s)
_s.loader.exec_module(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclusions", required=True)
    ap.add_argument("--exclusions-source", required=True)
    a = ap.parse_args()
    P = R.load()
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, check=True).stdout  # noqa: E731
    if git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/step3-retrain"):
        raise SystemExit("HEAD is not pushed: the assert reads committed records only")
    res = str((HERE / "results").relative_to(REPO))
    committed = [x for x in git("ls-tree", "--name-only", f"HEAD:{res}").decode().split() if re.fullmatch(r"sp493_.*\.json", x) and "_FAILED" not in x]
    admitted, recs = {}, []
    for name in sorted(committed):  # records read from HEAD, never the working tree (Codex r1)
        p = REPO / res / name
        r = json.loads(git("show", f"HEAD:{res}/{name}"))
        for fo, x in (r.get("result") or {}).items():
            if re.match(r"^sp493_g(?:3|6|7)_sim_[xy]", fo):
                recs.append(str(p.relative_to(REPO)))
                for wc in x["keys"]:
                    admitted.setdefault(wc.split("|", 1)[1], []).append(fo)
    eb = Path(a.exclusions).read_bytes()
    classed = json.loads(eb)
    bad = R.tail_union_assert(admitted, classed, P) + [f"{c}: in two admitted tail folders {fos}" for c, fos in sorted(admitted.items()) if len(fos) > 1]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"what": "09's FINAL tail assert (union half), before any S_paper493 tail row is read", "stamp (PT)": stamp, "git": git("rev-parse", "HEAD").decode().strip(),
           "pins": {"split": R.TAIL_SPLIT_PIN, "L3": R.L3_PIN, "tail_tasks": R.TAIL_TASK_PINS}, "records": sorted(set(recs)),
           "exclusions": {"source": a.exclusions_source, "sha256": hashlib.sha256(eb).hexdigest(), "n": len(classed), "by_class": {}},
           "admitted": len(admitted), "pinned": len(P["tail_group"]), "problems": bad, "pass": not bad}
    for v in classed.values():
        out["exclusions"]["by_class"][v] = out["exclusions"]["by_class"].get(v, 0) + 1
    q = HERE / "results" / f"sp493_tail_final_assert_{stamp}.json"
    q.open("x").write(json.dumps(out, indent=1))
    print(q.name, {k: out[k] for k in ("admitted", "pinned", "pass")}, out["exclusions"]["by_class"], bad[:5])
    if bad:
        raise SystemExit(f"STOP: {len(bad)} problems; no tail row may be read")


if __name__ == "__main__":
    main()
