"""S_paper493 X/Y sim-tail UNION assert for own_all_rs_L48 (32k, registered weights; 4f / d9 formal @ 195435019, 10-16): the three tail messages (s1, s2,
final; famexp's, pinned by commit + sha256) are disjoint by key and together hold 293 admitted runs (g3 36 / g6 137 / g7 120); the 2 classed exclusions
(t750 in g3 x s1, t899 in g7 y s2; no replacement) are in the messages' excluded_by_staging and in no epoch; 293 + 2 = 295; every message is scored in a
committed own_all_rs score record of the same fit, every folder with runs (R rows) + runs without rows = its epochs, 0 in-fold violations. Git only."""

import collections
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WANT = {"g3": 36, "g6": 137, "g7": 120}
EXCL = {"t750": "g3", "t899": "g7"}


def main(pins: str, records: str, out_rel: str):
    show = lambda c, p: subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
    recs, fit = {}, set()
    for r in records.split(","):
        b = (REPO / r).read_bytes()
        if subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{r}"], capture_output=True).stdout != b:
            raise SystemExit(f"{r}: not committed as read")
        d = json.loads(b)
        fit.add(d["fit_record_sha256"])
        for k, v in d["results"].items():
            recs[k.split("|", 1)[1]] = (r, hashlib.sha256(b).hexdigest(), v)
    if len(fit) != 1:
        raise SystemExit("the records are of more than one fit")
    keys, by_grp, excl, msgs = {}, collections.Counter(), {}, []
    for pin in pins.split(","):
        c, p, h = pin.split(":", 2)
        mb = show(c, p)
        if hashlib.sha256(mb).hexdigest() != h:
            raise SystemExit(f"{p} @ {c}: sha256 differs from the pin")
        m = json.loads(mb)
        if h not in recs:
            raise SystemExit(f"{p}: no given score record scored this message")
        rpath, rsha, res = recs[h]
        per = collections.Counter(e["folder"] for e in m["epochs"])
        for fo, n in per.items():
            x = res["folders"][fo]
            if x["counts"]["R"]["runs"] + x["counts"]["R"]["runs_without_rows"] != n or x["in_fold_violations"]:
                raise SystemExit(f"{fo}: scored runs differ from the message's epochs, or an in-fold violation")
        for e in m["epochs"]:
            if e["key"] in keys:
                raise SystemExit(f"{e['key']}: in two tail messages")
            keys[e["key"]] = p
            by_grp[re.match(r"^sp493_(g[0-9]+)_", e["folder"])[1]] += 1
        for pv in m.get("provenance", []):
            for x in pv.get("excluded_by_staging", []):
                excl[x["key"]] = {"cell": x.get("cell"), "folder_group": re.match(r"^sp493_(g[0-9]+)_", pv["staged_set"])[1] if re.match(r"^sp493_(g[0-9]+)_", str(pv.get("staged_set"))) else None,
                                  "message": p}
        msgs.append({"message": f"{p} @ {c}", "sha256": h, "epochs": len(m["epochs"]), "folders": dict(per), "score_record": {"path": rpath, "sha256": rsha}})
    ex_t = {k: v for k, v in excl.items() if str(v["cell"]).split("_", 1)[0] in EXCL}
    ok = (dict(by_grp) == WANT and len(keys) == 293 and sorted(str(v["cell"]).split("_", 1)[0] for v in ex_t.values()) == sorted(EXCL)
          and not set(ex_t) & set(keys) and all(v["folder_group"] == EXCL[str(v["cell"]).split("_", 1)[0]] for v in ex_t.values()))
    rec = {"label": "own_all_rs_L48 (32k) S_paper493 X/Y sim-tail union assert, by key; keys and counts only; PRELIMINARY", "fit_record_sha256": fit.pop(),
           "messages": msgs, "admitted_runs": len(keys), "by_group": dict(sorted(by_grp.items())), "want_by_group": WANT,
           "classed_exclusions": ex_t, "other_staging_exclusions_in_tail_messages": sorted(set(excl) - set(ex_t)), "union": len(keys) + len(ex_t), "pass": ok}
    (REPO / out_rel).open("x").write(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("admitted_runs", "by_group", "union", "pass")} | {"classed_exclusions": len(ex_t), "other_exclusions": len(rec["other_staging_exclusions_in_tail_messages"])}))
    if not ok:
        raise SystemExit("STOP: the tail union differs from 293 admitted + 2 classed = 295")


if __name__ == "__main__":
    main(*sys.argv[1:4])
