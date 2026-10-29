"""Which of figure 1's honest runs were graded by a placeholder scorer? Read-only, $0, local.

Some LinuxArena tasks ship a placeholder main-task scorer that marks every run I with the explanation "Dummy scorer".
Such an I is not a failure. A run counts as placeholder-graded only when its own .eval's linux_scorer explanation says
so: read from the raw log, never inferred from the task name. plot_figure1_old_new.py then treats these runs as
unscored (N), so a task with no other scored run drops out of the matched set for every bar.

  uv run --no-project --python 3.14 python experiments/2026-10-08_q2_figure_pack/find_placeholder_runs.py \
      <Sol json> <Qwen3.5 json> <report_relaunch.json> <report_first.json> \
      <Sol transcripts cache> <strict rerun dir> <Qwen3-32B first run dir> <Qwen3-32B relaunch dir>

Writes results/placeholder_runs_<stamp>.json: per model, the run keys plot_figure1_old_new.py uses and their tasks.
"""
import glob
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
PILOT = HERE.parents[0] / "2026-10-07_phase2_attack_check" / "config" / "pilot_tasks_amendment9.json"


READ: dict[str, str] = {}  # every .eval read: S3 URI -> sha256, recorded in the output
ARCHIVE = "s3://individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/runs/"  # each run's manifest_v1 lists sha256s


def s3_uri(ev: Path) -> str:
    """Where the read file lives on S3: the Sol cache mirrors its bucket; a Phase 2 run is archived under ARCHIVE."""
    p = str(ev.resolve())
    if "/.cache/probe-shift/s3/" in p:
        return "s3://" + p.split("/.cache/probe-shift/s3/", 1)[1]
    if "/phase2_attack_check/runs/" in p:
        return ARCHIVE + p.split("/phase2_attack_check/runs/", 1)[1]
    raise ValueError(f"{ev}: no S3 location known for this file")


def explanation(ev: Path, sample_id: str | None = None, epoch: int | None = None) -> str:
    """The linux_scorer explanation of one sample: the given one, or the file's only sample (asserted)."""
    READ.setdefault(s3_uri(ev), hashlib.sha256(ev.read_bytes()).hexdigest())
    with zipfile.ZipFile(ev) as z:
        names = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
        assert sample_id is not None or len(names) == 1, f"{ev}: {len(names)} samples, none named"
        for n in names:
            s = json.loads(z.read(n))
            if sample_id is None or (s["id"] == sample_id and s.get("epoch", 1) == epoch):
                return ((s.get("scores") or {}).get("linux_scorer") or {}).get("explanation") or ""
    raise KeyError(f"{ev}: no sample {sample_id} epoch {epoch}")


def one(pattern: str) -> Path:
    (p,) = glob.glob(pattern)  # exactly one match, or this fails loudly
    return Path(p)


def main(sol_p, q35_p, rel_p, first_p, sol_cache, rerun, q32_first, q32_rel):
    main_of = {c["key"]: c["main"] for c in json.loads(PILOT.read_text())["cells"]}
    runs = {"Sol": [], "Qwen3.5-27B": [], "Qwen3-32B": []}  # (key, main task, placeholder?)
    for r in json.loads(sol_p.read_text())["rows"]:
        ev = one(str(sol_cache / "*" / "logs" / r["file"]))
        runs["Sol"].append((r["key"], r["main_task"], "Dummy scorer" in explanation(ev, r["sample_id"], r["epoch"])))
    for r in json.loads(q35_p.read_text())["Qwen3.5-27B"]["rows"]:
        ev = one(str(rerun / r["arm"] / r["task_dir"] / r["file"]))
        runs["Qwen3.5-27B"].append((f"{r['arm']}/{r['task_dir']}", r["sample_id"].split(":")[0],
                                    "Dummy scorer" in explanation(ev)))
    for path, run_dir in ((rel_p, q32_rel), (first_p, q32_first)):
        for r in json.loads(path.read_text())["rows"]:
            if r.get("main_task_raw") is None:  # never scored; plot_figure1_old_new.py drops these too
                continue
            ev = one(str(run_dir / r["arm"] / r["task"] / "*.eval"))
            runs["Qwen3-32B"].append((f"{r['arm']}/{r['task']}", main_of[r["task"]], "Dummy scorer" in explanation(ev)))
    out = {"rule": "a run is placeholder-graded when its own log's linux_scorer explanation contains 'Dummy scorer'",
           "inputs sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (sol_p, q35_p, rel_p, first_p)},
           "runs read": {m: len(rs) for m, rs in runs.items()},
           "placeholder runs": {m: sorted(k for k, _, d in rs if d) for m, rs in runs.items()},
           "placeholder tasks": {m: sorted({t for _, t, d in rs if d}) for m, rs in runs.items()},
           "eval files read (S3 URI: sha256)": dict(sorted(READ.items()))}
    print(json.dumps(out, indent=1))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = HERE / "results" / f"placeholder_runs_{stamp}.json"
    res.write_text(json.dumps(out, indent=1) + "\n")
    print(res)


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:9]))
