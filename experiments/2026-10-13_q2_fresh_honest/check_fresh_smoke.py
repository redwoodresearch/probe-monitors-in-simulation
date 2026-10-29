"""Fresh-task honest smoke check: a copy of ../2026-10-07_phase2_attack_check/check_zoo_c_smoke.py run on the devbox (no git checkout there): the git sha is passed in, and the result goes to --out-dir. Its docstring follows.

Zoo C smoke check (README "Zoo C's new real Sol honest runs", procedure step 1): does the smoke run match overlap800's
settings, and what did it cost? Reads the smoke run's .eval (an honest run) and one overlap800 log's header; keys,
settings and counts only, no message text. $0, local.

Checks: the plan's solvers and arena_policy parameters, the eval config (time limit, max_samples, epochs;
retry_on_error is reported, a known difference), reasoning effort on every policy call, the sample's status, error,
limit and steps, and the watcher's final priced spend.

Prediction (written before running): every check passes except retry_on_error (none against 1); cost $0.3-1.8.

  uv run --no-project --python 3.14 python check_zoo_c_smoke.py <smoke .eval> <overlap800 .eval> <policy_spend.jsonl>
"""

import argparse
import hashlib
import json
import subprocess
import zipfile

import evalzip  # noqa: F401  (registers zstd for Python 3.13 on the devbox)
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
POLICY = "openai/gpt-5.6-sol"


def header(p: Path) -> dict:
    with zipfile.ZipFile(p) as z:
        return json.loads(z.read("header.json"))


def main(smoke: Path, ref: Path, spend: Path, git: str, out_dir: Path):
    hs, hr = header(smoke), header(ref)
    plan = lambda h: [(s["solver"], s.get("params", {})) for s in h["plan"]["steps"]]  # noqa: E731
    cfg = lambda h: {k: h["eval"]["config"].get(k) for k in ("time_limit", "max_samples", "epochs", "retry_on_error")}  # noqa: E731
    with zipfile.ZipFile(smoke) as z:
        (name,) = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
        s = json.loads(z.read(name))
    calls = [e for e in s["events"] if e.get("event") == "model" and e.get("model") == POLICY]
    efforts = sorted({(e.get("config") or {}).get("reasoning_effort") for e in calls}, key=str)
    readings = [json.loads(x) for x in spend.read_text().splitlines()]
    out = {"label": "fresh-task honest smoke check", "git": git,
           "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (smoke, ref, spend)},
           "task": {"smoke": hs["eval"]["task"], "overlap800": hr["eval"]["task"]},
           "plan_matches": plan(hs) == plan(hr), "plan": {"smoke": plan(hs), "overlap800": plan(hr)},
           "config": {"smoke": cfg(hs), "overlap800": cfg(hr)},
           "config_matches_except_retry": {k: v for k, v in cfg(hs).items() if k != "retry_on_error"}
           == {k: v for k, v in cfg(hr).items() if k != "retry_on_error"},
           "policy_calls": len(calls), "reasoning_efforts": efforts, "effort_medium_on_every_call": efforts == ["medium"],
           "sample": {"id": s.get("id"), "error": bool(s.get("error")), "limit": (s.get("limit") or {}).get("type"),
                      "status": hs.get("status")},
           "watcher_final": readings[-1] if readings else None}
    fin = out["watcher_final"] or {}
    out["priced_complete"] = fin.get("running") == 0 and fin.get("finished") == 1  # the smoke stage's one attempt
    out["checks_passed"] = bool(out["plan_matches"] and out["config_matches_except_retry"] and calls
                                and out["effort_medium_on_every_call"] and hs.get("status") == "success"
                                and not out["sample"]["error"] and out["sample"]["limit"] is None and out["priced_complete"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = out_dir / f"fresh_smoke_check_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("plan", "inputs")}, indent=1))
    print("->", dst)
    if not out["checks_passed"]:
        raise SystemExit("smoke check FAILED: see the result file")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("smoke", "ref", "spend"):
        ap.add_argument(a, type=Path)
    ap.add_argument("--git", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    main(a.smoke, a.ref, a.spend, a.git, a.out_dir)
