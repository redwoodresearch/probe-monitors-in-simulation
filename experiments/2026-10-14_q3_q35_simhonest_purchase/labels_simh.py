"""Attempt labels for A's epoch-2 simulated attacks (world sim_e2), for 09 (d9, 10-14 evening): EXACTLY the procedure that
labelled the 196, make_phase2_attempt_labels.py @ 6e1a8c86 (labeler/, byte-identical; sha256 963d7704...), run unchanged on Modal
CPU. A copy of the overnight chain's wrapper (aswin/q3-overnight-chain, experiments/2026-10-14_q3_overnight_chain/labels_chain.py
@ 7f080dd55) whose only changes are inputs: this job's Volume; the run folder is assembled in the container from A's launch.json
(copied from the devbox, sha256 pinned below, never read on the laptop) and a link to A's uploaded arm folder
(simh_evals_v1/2026-10-14_142029_simh_attack2/sim_attack@september, the 19 counted .eval files); flags as the simulated lineage,
`--arms sim_attack@september --self-report-only sim_attack@september`. The container has no git, so the script's two
`git show 8950b34:<path>` reads are served from labeler/pin_8950b34_*.py (byte-identical, sha256 checked). Writes no text: the
label file holds keys, labels, rules and each description's sha256 and length. Then 09's format, with the chain's mapping
(hygiene_chain.labels_of @ 7f080dd55): attempted / unclear / declined as given; the script's None -> no_report; no counted
attempt or no .eval in it -> run_failed (t144: its .eval was not uploaded, a counted infrastructure failure).

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run <this file>
"""

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q3simh_labels_kill-2026-11-13"
VOL = "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"
RUN = "2026-10-14_142029_simh_attack2"
LAUNCH_JSON = (Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad/simh_launch_X/a_launchjson/launch.json"),
               "1bab566a87d9d91cec51f3c27fd4843c500d31c67c0cef6d828f093568ec0ff3")
PINS = {"experiments/2026-10-07_q3_nano_monitor/run_self_reports.py": ("pin_8950b34_run_self_reports.py", "d192d4fce3fe636a848ac966d1a2a023e1433cc4c6f566f00706f4feb93888ed"),
        "experiments/2026-10-07_q3_nano_monitor/make_attempt_labels.py": ("pin_8950b34_make_attempt_labels.py", "27cd54d7d9ef94ca971710450c188ed4c3c6d1679847007ca21434b5b9ea0392")}
SCRIPT_SHA = "963d77043f4f93e987636095a69062c7a2bd4bee1c2f710b81c832400a43c735"
app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.14").add_local_dir(HERE / "labeler", "/root/labeler")  # 3.14: zstd .eval members
         .add_local_file(LAUNCH_JSON[0], "/root/launch.json"))


@app.function(image=image, cpu=(2.0, 2.0), memory=(8192, 8192), timeout=1800, retries=0, startup_timeout=300, scaledown_window=60,
              volumes={"/ev": modal.Volume.from_name(VOL, version=2).read_only()})
def label() -> dict:
    import importlib.util
    import sys

    L = Path("/root/labeler")
    if hashlib.sha256((L / "make_phase2_attempt_labels.py").read_bytes()).hexdigest() != SCRIPT_SHA:
        raise ValueError("labeler script differs from 6e1a8c86's")
    for path, (name, want) in PINS.items():
        if hashlib.sha256((L / name).read_bytes()).hexdigest() != want:
            raise ValueError(f"{name} differs from 8950b34:{path}")
    lj = Path("/root/launch.json").read_bytes()
    if hashlib.sha256(lj).hexdigest() != LAUNCH_JSON[1]:
        raise ValueError("A's launch.json differs from its pin")
    rd = Path("/tmp/rd") / RUN  # the run folder as on the box: launch.json and the arm folder (linked, read-only)
    rd.mkdir(parents=True)
    (rd / "launch.json").write_bytes(lj)
    (rd / "sim_attack@september").symlink_to(Path("/ev/simh_evals_v1") / RUN / "sim_attack@september")
    spec = importlib.util.spec_from_file_location("mpal", L / "make_phase2_attempt_labels.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    def pinned_block(path: str, start: str, end: str) -> str:  # the script's own slicing, on the byte-identical copy
        src = (L / PINS[path][0]).read_text()
        i, j = src.index(start), src.index(end)
        return src[i:j]

    m.pinned_block = pinned_block
    m.HERE = Path("/tmp/labels")
    (m.HERE / "results").mkdir(parents=True, exist_ok=True)
    args = ["--run-dir", str(rd), "--arms", "sim_attack@september", "--self-report-only", "sim_attack@september"]
    sys.argv = ["make_phase2_attempt_labels.py", *args]
    m.main()  # prints its own per-arm counts (labels and rules, no text)
    (out,) = sorted((m.HERE / "results").glob("phase2_attempt_labels_*.jsonl"))
    b = out.read_bytes()
    return {"name": out.name, "bytes": b, "sha256": hashlib.sha256(b).hexdigest(), "args": args}


def labels_09(rows: list) -> dict:
    """09's enum per counted run (hygiene_chain.labels_of @ 7f080dd55, kind "fu"), world sim_e2."""
    labels, counts = {}, Counter()
    for r in rows:
        lab = r["label"]
        if r.get("rule") in ("no counted attempt", "no .eval file in the counted attempt"):
            v = "run_failed"
        elif lab is None:
            v = "no_report"
        elif lab in ("attempted", "unclear", "declined"):
            v = lab
        else:
            raise ValueError(f"unexpected label {lab!r}")
        labels[f"sim_e2|{r['task']}"] = v
        counts[v] += 1
    return {"labels": labels, "counts": dict(counts)}


@app.local_entrypoint()
def main():
    r = label.remote()
    dest = HERE / "results" / "labels" / r["name"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.open("xb").write(r["bytes"])
    rows = [json.loads(x) for x in r["bytes"].decode().splitlines()]
    out = labels_09(rows) | {"world": "sim_e2", "launch": RUN, "n": len(rows), "source": {"file": f"results/labels/{r['name']}",
                             "sha256": r["sha256"], "script": "make_phase2_attempt_labels.py @ 6e1a8c86", "args": r["args"]},
                             "note": "t144_file_indexing is a counted infrastructure failure (its vLLM server was stopped by the laptop "
                                     "billing watcher); its .eval was not uploaded, so the labeler saw none: run_failed"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p09 = HERE / "results" / "labels" / f"attempt_labels_sim_e2_{stamp}.json"
    p09.open("x").write(json.dumps(out, indent=1))
    print(json.dumps({"label_file": str(dest.relative_to(HERE)), "sha256": r["sha256"], "rows": len(rows),
                      "script_labels": dict(Counter(str(x["label"]) for x in rows)), "rules": dict(Counter(x["rule"] for x in rows)),
                      "09_file": str(p09.relative_to(HERE)), "09_counts": out["counts"]}))
