"""Laptop-only: resolve the run's pins to git bytes exactly as main does (run_numbers_verify.build_arts), and print counts and bytes
only. No Modal call is made (the module's Modal objects are lazy references). Writing's files are not parsed here.

  uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy==2.3.3 python -I check_payload.py tables34,rc1a,rc2,rc1b,post
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_numbers_verify as RN  # noqa: E402

cfg = json.loads(RN.CONFIG.read_text())
names = sys.argv[1].split(",")
arts = RN.build_arts(cfg, names)
post = arts.get("post") or {}
print(json.dumps({"artifacts": names, "bytes_sent": RN.bytes_sent(arts), "post_inputs": len(post.get("inputs", {})),
                  "post_records": len(post.get("records", [])), "post_code_blobs": sorted(post.get("code", {})),
                  "pure_files": list(RN.PURE)}))
