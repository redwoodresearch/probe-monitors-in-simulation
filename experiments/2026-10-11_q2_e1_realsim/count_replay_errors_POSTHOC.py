"""POST HOC (Bugbot on #903, 10-12): how many replayed tool turns in the pilots' histories carried an Inspect `error`,
which build_realsim_index.py dropped (it kept each tool turn's content only). Counts only; no text is printed or saved.
Local, $0, from the cached .eval files.

  uv run --no-project --python 3.14 python experiments/2026-10-11_q2_e1_realsim/count_replay_errors_POSTHOC.py <index jsonl>
"""
import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
CACHE = Path.home() / ".cache/probe-shift/s3/rr-sce-sim-transcripts/transcripts"
COHORT = {"real": "2026-09-18_la-public10_real-honest-solmedium-step150-overlapaware-n1000",
          "sim": "2026-10-06_la-public10_sim-astra-medium-honest-origtraj-read-nocritic-fork0"}


def text_of(c):
    return c if isinstance(c, str) else "".join(x.get("text", "") for x in (c or []) if isinstance(x, dict))


def main(index: Path):
    me = Path(__file__).resolve()
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    rows = [json.loads(l) for l in index.open()]
    c, runs_with = Counter(), Counter()
    for r in rows:
        w0, file, sid, ep = r["run_id"].split("|"); world = "real" if w0.startswith("real") else "sim"
        with zipfile.ZipFile(CACHE / COHORT[world] / "logs" / file) as z:  # a missing file raises
            s = json.loads(z.read([n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")][0]))
        if str(s["id"]) != sid or str(s["epoch"]) != ep:
            raise SystemExit(f"{r['key']}: the eval file holds another sample")
        msgs = s["messages"]; la = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        tools = [m for m in msgs[:la] if m["role"] == "tool"]; errs = [m for m in tools if m.get("error")]
        a = r["arm"]; c[(a, "runs")] += 1; c[(a, "tool turns")] += len(tools); c[(a, "tool turns with an error")] += len(errs)
        c[(a, "errored tool turns replayed as empty")] += sum(not text_of(m.get("content")).strip() for m in errs); runs_with[a] += bool(errs)
    out = {"label": "POST HOC (Bugbot on #903); counts only", "git": head, "index": {index.name: hashlib.sha256(index.read_bytes()).hexdigest()},
           "by arm": {a: {k: v for (a2, k), v in sorted(c.items()) if a2 == a} | {"runs with an errored tool turn": runs_with[a]} for a in sorted({k[0] for k in c})}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"replay_errors_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
