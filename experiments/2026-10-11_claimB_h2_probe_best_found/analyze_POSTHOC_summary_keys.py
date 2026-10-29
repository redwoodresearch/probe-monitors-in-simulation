"""POST HOC, $0 (PR #774 review thread ...rbcEs): do the H2 summary's planted input hashes, recorded under plantN/<cfg>.npz,
equal the archived plantN_k1/<cfg>.npz arrays (results_manifest_v3.json)? Numbers and hashes only.
  uv run --no-project --python 3.12 python experiments/2026-10-11_claimB_h2_probe_best_found/analyze_POSTHOC_summary_keys.py \\
      <h2_summary_2026-10-11_132926.json, downloaded from S3 (manifest v2) and sha256-checked>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
raw = Path(sys.argv[1]).read_bytes()
want = json.loads((HERE / "results_manifest_v2.json").read_text())["files"]
assert hashlib.sha256(raw).hexdigest() == next(v["sha256"] for k, v in want.items() if k.endswith("h2_summary_2026-10-11_132926.json"))
inputs = json.loads(raw)["inputs"]
v3 = {k.split("h2_search_v1/")[1]: v["sha256"] for k, v in json.loads((HERE / "results_manifest_v3.json").read_text())["files"].items() if "h2_search_v1/" in k}
planted = {k: h for k, h in inputs.items() if k.startswith("plant")}
others = {k: h for k, h in inputs.items() if not k.startswith("plant") and k != "runs_json"}
out = {"planted_keys": len(planted),
       "planted_equal_k1_folder": sum(v3.get(k.replace("/", "_k1/", 1)) == h for k, h in planted.items()),
       "unsuffixed_planted_entries_in_v3": sum(k.startswith(("plant1/", "plant2/")) for k in v3),  # v3 archives only the _k1 reruns
       "planted_equal_unsuffixed_folder": sum(v3.get(k) == h for k, h in planted.items()),
       "other_keys": len(others), "other_equal": sum(v3.get(k) == h for k, h in others.items()),
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
with open(HERE / "results" / f"summary_keys_POSTHOC_{stamp}.json", "x") as fh:
    fh.write(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
