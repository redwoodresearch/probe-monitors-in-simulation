"""Route C pooled A's own selection record (09 RULE item 40, via 2e 10-17): it reads (A) and cites route C A's committed record by
path@commit and both sha256s (the committed .json.gz blob, and the decompressed JSON's, which is hashed, never parsed). When FROZEN pins no
IL extra-arm overrides it carries item 40's NOTE verbatim. Keys only: no record is parsed and no width or value is computed (unlike
select_estimator.py --b-not-evaluable, which parses the record on the laptop and computes HW(A)). Commit it BEFORE any route C value is read or
printed. Laptop, stdlib only, $0. A FROZEN whose overrides differ from 110447's twelve STOPs: 09 then words (B)'s status (if the IL overrides
are pinned, the lead is still (A) and (B) is the labelled update).

  uv run --no-project --python 3.12 python -I make_routec_selection.py --record <path>@<commit> --blob-sha <sha256> --json-sha <sha256> \\
      --inputs-full <path>@<commit> --inputs-full-sha <sha256>
  # writes results/estimator_selection_routec_<stamp>.json (exclusive create); prints {"file", "sha256", "lead"}
"""

import argparse
import gzip
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
RULE40 = {"commit": "bb0f24d801d7e6e75b44c68738cd93c195ca8588", "path": "experiments/2026-10-14_priority1_spec_09/RULE.md",
          "sha256": "223618171f028180c26846c3efd5512813fba4fa68eebe881c5866c79999bbd3", "line": 4873}
RULE40_HEAD = "40. **The (A)/(B) lead after the IL build omission"  # the pinned line starts so
B_NOT_EVALUABLE = "(B) not evaluable at the freeze: IL extra-arm overrides not pinned (build omission)"  # 09 RULE item 40, verbatim
RULE_B = ("09 RULE item 40: a build failure (not item 20's missing-arm fallback); the (B) + RMS addendum is a labelled update on the same "
          "frozen draws and union stream; no headline switch after (A)'s points are read")  # as select_estimator.py words it
OVERRIDES_110447 = sorted(f"{s}|{r}" for s in ("S_new", "S_new+25f", "S_newmains", "S_old", "S_paper493", "pools_ep2") for r in "CR")


def git(*a) -> bytes:
    return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, check=True).stdout


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pinned(ref: str, want: str, name: str) -> tuple:
    """<path>@<commit> -> (bytes, path, full commit); the bytes' sha256 must be want (checked before anything parses them)."""
    p, c = ref.rsplit("@", 1)
    c = git("rev-parse", "--verify", f"{c}^{{commit}}").decode().strip()
    b = git("show", f"{c}:{p}")
    if sha(b) != want:
        raise SystemExit(f"STOP: {name}: sha256 differs from the pin; nothing written")
    return b, p, c


def build(record: str, blob_sha: str, json_sha: str, inputs_full: str, inputs_full_sha: str, stamp: str) -> dict:
    rb, rp, rc = pinned(record, blob_sha, "the A record (committed blob)")
    if sha(gzip.decompress(rb) if rp.endswith(".gz") else rb) != json_sha:  # hashed only
        raise SystemExit("STOP: the A record's decompressed JSON sha256 differs from --json-sha; nothing written")
    del rb
    fb, fp, fc = pinned(inputs_full, inputs_full_sha, "INPUTS_FULL")
    fin = json.loads(fb)  # K: a config written by inputs_full.py build (blob 093bc2bb, L173-215)
    if not str(fin.get("status", "")).startswith("FROZEN") or fin.get("test") is not False:
        raise SystemExit("STOP: INPUTS_FULL is not a FROZEN, non-test file; nothing written")
    ov = sorted(fin.get("overrides") or {})
    if ov != OVERRIDES_110447:
        raise SystemExit(f"STOP: FROZEN pins overrides other than 110447's twelve ({sorted(set(ov) ^ set(OVERRIDES_110447))}): 09 words (B)'s "
                         "status (IL extra-arm overrides pinned: lead still (A), (B) the labelled update); nothing written")
    rule = git("show", f"{RULE40['commit']}:{RULE40['path']}")
    if sha(rule) != RULE40["sha256"] or not rule.decode().splitlines()[RULE40["line"] - 1].startswith(RULE40_HEAD):
        raise SystemExit("STOP: RULE item 40 is not at its pinned commit / line; nothing written")
    return {"label": "all-runs selection record, route C pooled A (09 RULE item 40: lead (A); keys only, no width computed)",
            "rule": RULE40, "lead": "A", "note": B_NOT_EVALUABLE, "rule_b_not_evaluable": RULE_B,
            "record": {"path": rp, "commit": rc, "blob_sha256": blob_sha, "json_sha256": json_sha},
            "inputs_full": {"path": fp, "commit": fc, "sha256": inputs_full_sha, "status": fin["status"]},
            "freeze_evidence": {"override_keys": ov, "il_extra_arm_overrides_pinned": False, "ilarms_block": fin.get("ilarms")},
            "stamp_pt": stamp, "git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": sha(Path(__file__).read_bytes())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for k in ("--record", "--blob-sha", "--json-sha", "--inputs-full", "--inputs-full-sha"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("STOP: uncommitted tracked changes")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = build(a.record, a.blob_sha, a.json_sha, a.inputs_full, a.inputs_full_sha, stamp)
    b = json.dumps(out, indent=1).encode()
    dest = HERE / "results" / f"estimator_selection_routec_{stamp}.json"
    with dest.open("xb") as fh:
        fh.write(b)
    print(json.dumps({"file": str(dest.relative_to(HERE.parents[1])), "sha256": sha(b), "lead": out["lead"]}), flush=True)


if __name__ == "__main__":
    main()
