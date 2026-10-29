"""Gate for the matched-FA drop-top redo (RULE.md): the copy run with --drop none must reproduce 2e's stored result
exactly: `failed`, the input sha256, and every `targets` value (points and both interval ends), compared with ==.
The stored result is the one-line git copy experiments/2026-10-08_q3_onpolicy_figures/results/
q22_matched_fa_POSTHOC_2026-10-11_033516.json (sha256 bd688327…, json.loads-equal to the S3 original f9a32211…).

  python3 check_gate.py <stored json> <copy's --drop none output>
"""
import hashlib
import json
import sys
from pathlib import Path

STORED_SHA = "bd688327176c96868397e8a2f31f6073ab00dc0eeceea907039fa60f437e53d7"
stored_raw = Path(sys.argv[1]).read_bytes()
if hashlib.sha256(stored_raw).hexdigest() != STORED_SHA:
    raise SystemExit("stored result sha256 differs")
a, b = json.loads(stored_raw), json.loads(Path(sys.argv[2]).read_text())
ok = {"failed": a["failed"] == b["failed"], "input_sha256": a["input"]["sha256"] == b["input"]["sha256"],
      "targets": a["targets"] == b["targets"], "drop_none": b["drop"]["key"] == "none" and b["drop"]["tasks"] == []}
print(json.dumps(ok), "PASS" if all(ok.values()) else "FAIL")
sys.exit(0 if all(ok.values()) else 1)
