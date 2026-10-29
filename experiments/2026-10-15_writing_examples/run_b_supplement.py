"""Index B supplement for the writing session: per row, the side task's id and name (metadata.side_task.id / .name) and side_task_success, read from the row's .eval
sample inside Modal. INDEX ONLY: the short id must be a plain identifier (<= 80 chars of [A-Za-z0-9_:.-]) and side_task_success a
short score code; anything else is refused, never returned. No side-task text, no command text, no message content leaves the job.

  --mode inspect  key names and types under metadata.side_task, metadata.real_side_task_id and scores.linux_scorer (no values)
  --mode run      the supplement record (exclusive create)
  uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-15_writing_examples/run_b_supplement.py --mode inspect|run
"""

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-p2probes_examples_kill-2026-11-13"
INDEX_B = (
    "256682752",
    "experiments/2026-10-15_writing_examples/results/examples_index_B_2026-10-15_221525.json",
    "a0f9fa14bd47672f861e2c829553aae8d6cb3328805468562aae3983f4d3f539",
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("zstandard==0.25.0")
    .add_local_file(HERE / "run_examples_index.py", "/root/run_examples_index.py")
)
app = modal.App(APP, image=image)
vol = {"/acts": modal.Volume.from_name("probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14").read_only()}
SHORT_ID = re.compile(r"[A-Za-z0-9_:.\-]{1,80}")
NAME = re.compile(r"[A-Za-z0-9 _:.()\-]{1,80}")


def _shape(x, depth=0):
    if depth > 2:
        return type(x).__name__
    if isinstance(x, dict):
        return {k: _shape(v, depth + 1) for k, v in list(x.items())[:30]}
    return type(x).__name__ if not isinstance(x, list) else [f"list len {len(x)}"]


@app.function(cpu=1, memory=4096, timeout=900, retries=0, volumes=vol)
def supp(rows: list, mode: str) -> list:
    import run_examples_index as rx

    out = []
    for e in rows:
        err = None
        try:
            s = rx.read_member(Path("/acts") / e["run_eval"]["eval"], e["run_eval"]["sample_member"])
            md, sc = s.get("metadata") or {}, s.get("scores") or {}
            if mode == "inspect":
                out.append(
                    {
                        "run": e["run"],
                        "metadata.side_task": _shape(md.get("side_task")),
                        "metadata.real_side_task_id": type(md.get("real_side_task_id")).__name__,
                        "scores": _shape(sc),
                    }
                )
                continue
            st = md.get("side_task") if isinstance(md.get("side_task"), dict) else {}
            sid, name = st.get("id"), st.get("name")  # metadata.side_task: id and name only, never its description fields (4f)
            sts = ((sc.get("linux_scorer") or {}).get("value") or {}).get("side_task_success")
            out.append(
                {
                    "run": e["run"],
                    "side_task_id": sid if isinstance(sid, str) and SHORT_ID.fullmatch(sid) else "refused: not a plain short identifier",
                    "side_task_name": name if isinstance(name, str) and NAME.fullmatch(name) else "refused: not a plain short name",
                    "side_task_success": sts
                    if (isinstance(sts, str) and len(sts) <= 3) or isinstance(sts, bool) or sts is None
                    else "refused: not a short score code",
                    "source": {"id": "metadata.side_task.id", "name": "metadata.side_task.name", "success": "scores.linux_scorer.value.side_task_success"},
                }
            )
        except Exception as x:  # noqa: BLE001 -- parse errors can hold transcript text: only the type crosses back
            err = f"stopped: {type(x).__name__} (message withheld)"
        if err:
            out.append({"run": e["run"], "status": err})
    return out


@app.local_entrypoint()
def main(mode: str = "inspect"):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    b = git("show", f"{INDEX_B[0]}:{INDEX_B[1]}")
    if hashlib.sha256(b).hexdigest() != INDEX_B[2]:
        raise SystemExit("index B sha256 differs from its pin")
    rows = [e for e in json.loads(b)["entries"] if e["status"] == "ok"]
    res = supp.remote(rows, mode)
    if mode == "inspect":
        print(json.dumps(res[0], indent=1)[:2500])
        return
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {
        "what": "Index B supplement: side task short id and side_task_success per row (index only; no text)",
        "index_b": {"commit": INDEX_B[0], "path": INDEX_B[1], "sha256": INDEX_B[2]},
        "git": git("rev-parse", "HEAD").decode().strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stamp (PT)": stamp,
        "rows": res,
    }
    dest = HERE / "results" / f"examples_index_B_supplement_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    print(dest.name, hashlib.sha256(dest.read_bytes()).hexdigest())
