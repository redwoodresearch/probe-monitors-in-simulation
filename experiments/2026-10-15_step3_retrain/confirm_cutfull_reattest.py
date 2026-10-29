"""Append-only NOTE for the 45 cutfull run records (RULE F.9): confirm by sha that d9's re-attested routing revision changes nothing these
packs were scored from. Git and JSON only (no Volume read, no score). For each record: the highest revision of its routing record at
the pinned routing commit is newer; its unchanged_from_revision names exactly the revision and record file these packs used (bytes equal
by sha256) with all three flags true; we re-check the flags ourselves (staged set, extraction root, per-arm staged sha vs our committed
message); its attestation pointer and payload have failing {}; every arm's format_check is a record at a commit on origin with passes
true. Writes results/cutfull_reattest_note_<stamp>.json (exclusive create); exits nonzero on any failure.

  python3 -I confirm_cutfull_reattest.py --routing-commit <sha>
"""

import argparse
import glob
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LABEL_FORMAT = "format_check passed (d9 @ 5fe015495)"
git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731
on_origin = lambda c: any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", c).decode().splitlines())  # noqa: E731
truthy = lambda v: v is True or v == "true"  # noqa: E731


def at(ref):
    p, c = (x.strip() for x in ref.split(" @ "))
    return c, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing-commit", required=True)
    a = ap.parse_args()
    if not on_origin(a.routing_commit):
        raise SystemExit("the routing commit is not on origin")
    tracked = set(git("ls-files", "--", str((HERE / "results").relative_to(REPO))).decode().split())
    recs = [p for p in sorted(glob.glob(str(HERE / "results" / "cutfull_s*_r[0-9]_2026-10-15_23*.json"))) if str(Path(p).relative_to(REPO)) in tracked]
    out, fails = [], []
    for p in recs:
        r = json.loads(Path(p).read_text())
        rp, rc, rsha, k = r["routing_record"]["path"], r["routing_record"]["commit"], r["routing_record"]["sha256"], r["revision"]
        src = json.loads(git("show", f"{r['source_record']['commit']}:{r['source_record']['path']}"))
        msg = json.loads((REPO / src["message_path"]).read_bytes())
        if hashlib.sha256((REPO / src["message_path"]).read_bytes()).hexdigest() != r["message_sha256"] or str(Path(src["message_path"])) not in tracked:
            raise SystemExit(f"{p}: the committed message differs from the record's")
        ours = {"staged_set": msg["provenance"][0]["staged_set"], "extraction_root": msg["provenance"][0]["run"],
                "staged_sha": {e["arm"]: e["staged_sha256"] for e in msg["epochs"]}}
        stem = re.sub(r"(_r[0-9]+)?\.json$", "", Path(rp).name)
        ks = sorted(int(m.group(1) or 1) for x in git("ls-tree", "--name-only", f"{a.routing_commit}:{Path(rp).parent}").decode().split()
                    if (m := re.fullmatch(re.escape(stem) + r"(?:_r([0-9]+))?\.json", x)))
        kn = ks[-1]
        np_ = f"{Path(rp).parent}/{stem}_r{kn}.json"
        new = json.loads(git("show", f"{a.routing_commit}:{np_}"))
        uf = new.get("unchanged_from_revision") or {}
        uc, upath = at(uf.get("file", " @ "))
        same_file = upath == rp and hashlib.sha256(git("show", f"{uc}:{upath}")).hexdigest() == rsha if upath else False
        arms = {x["meta_arm"]: x for x in new["arms"]}
        ptr = json.loads(git("show", f"{a.routing_commit}:{Path(np_).parent}/{Path(np_).stem}_attest.json"))
        att = json.loads(git("show", f"{ptr['commit']}:{ptr['path']}"))
        fmt = {}
        for m_, x in arms.items():
            if not x.get("format_check"):
                fmt[m_] = False
                continue
            c_, p_ = at(x["format_check"])
            fmt[m_] = on_origin(c_) and truthy(json.loads(git("show", f"{c_}:{p_}")).get("passes"))
        checks = {"newer_revision": kn > k and new["revision"] == kn, "unchanged_from_this_revision": uf.get("revision") == k and same_file,
                  "flags_true": all(uf.get(f) is True for f in ("staged_set", "extraction_root", "staged_sha_per_arm")),
                  "recheck_staged_set_root_shas": new["staged_set"] == ours["staged_set"] and new["extraction_root"] == ours["extraction_root"]
                  and {m_: x["staged_sha"] for m_, x in arms.items()} == ours["staged_sha"],
                  "attestation_failing_empty": ptr.get("failing") == {} and att.get("failing") == {} and ptr.get("revision") == kn,
                  "format_check_every_arm_passes": bool(fmt) and all(fmt.values())}
        row = {"record": str(Path(p).relative_to(REPO)), "revision_scored": k, "routing_record_scored": f"{rp} @ {rc}",
               "new_revision": f"{np_} @ {git('rev-parse', a.routing_commit).decode().strip()[:12]}", "checks": checks}
        out.append(row)
        if not all(checks.values()):
            fails.append(row)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    note = {"what": "append-only NOTE on the 45 cutfull step-3 run records (RULE F.9); the records themselves are unchanged",
            "routing_commit": git("rev-parse", a.routing_commit).decode().strip(), "git": git("rev-parse", "HEAD").decode().strip(), "stamp (PT)": stamp,
            "records": len(out), "confirmed": len(out) - len(fails),
            "supersedes_labels": {"format_check": {"from": "format_check not run (stage format_gate only)", "to": LABEL_FORMAT},
                                  "read_status": {"from": "UNREAD until d9 re-attests", "to": f"confirmed by sha @ {a.routing_commit[:9]}"}} if not fails else None,
            "rows": out}
    q = HERE / "results" / f"cutfull_reattest_note_{stamp}.json"
    q.open("x").write(json.dumps(note, indent=1))
    print(q.name, {"records": len(out), "confirmed": note["confirmed"]})
    if fails or len(out) != 45:
        raise SystemExit(f"NOT confirmed: {len(fails)} records fail, {len(out)} records found")


if __name__ == "__main__":
    main()
