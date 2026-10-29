"""Overnight chain (RULE.md here): attempt labels for a scale-up launch's attack arm, with EXACTLY the procedure that labelled the 196:
make_phase2_attempt_labels.py @ 6e1a8c86 (labeler/, byte-identical; sha256 963d7704...), run unchanged on Modal CPU against the
launch's relayed run folder on the chain's Volume (evals_v1/<run>: launch.json and the arm folders, as on the box). The one
wrapper change: the container has no git, so its two `git show 8950b34:<path>` reads are served from labeler/pin_8950b34_*.py,
byte-identical copies of exactly those files (sha256 checked here before use); the blocks it slices and executes are unchanged.
Flags as the 196's lineage: a main launch `--arms attack`; a simulated follow-up `--arms sim_attack@september
--self-report-only sim_attack@september` (`--arm` names another simulated attack arm, e.g. the ungrounded arm's
sim_attack@september_ungrounded, with the same flags). Only the relayed (counted) attempts' .eval files are on the Volume, which is what the
script reads (its counted-attempt rule picks the same attempt). Writes no text: descriptions as sha256 and length; the label file
comes back to results/labels/ (keys, labels, hashes).

  modal run experiments/2026-10-14_q3_overnight_chain/labels_chain.py --run <run> --kind main|fu [--arm <simulated attack arm>]
"""

import hashlib
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q3chain_labels_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
PINS = {"experiments/2026-10-07_q3_nano_monitor/run_self_reports.py": ("pin_8950b34_run_self_reports.py", "d192d4fce3fe636a848ac966d1a2a023e1433cc4c6f566f00706f4feb93888ed"),
        "experiments/2026-10-07_q3_nano_monitor/make_attempt_labels.py": ("pin_8950b34_make_attempt_labels.py", "27cd54d7d9ef94ca971710450c188ed4c3c6d1679847007ca21434b5b9ea0392")}
SCRIPT_SHA = "963d77043f4f93e987636095a69062c7a2bd4bee1c2f710b81c832400a43c735"
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.14").add_local_dir(HERE / "labeler", "/root/labeler")  # 3.14: zstd .eval members, as the 196 were labelled


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes={"/ev": modal.Volume.from_name(VOL, version=2).read_only()})
def label(run: str, kind: str, root: str = "evals_v1", arm: str = "sim_attack@september") -> dict:
    import importlib.util
    import sys

    L = Path("/root/labeler")
    if hashlib.sha256((L / "make_phase2_attempt_labels.py").read_bytes()).hexdigest() != SCRIPT_SHA:
        raise ValueError("labeler script differs from 6e1a8c86's")
    for path, (name, want) in PINS.items():
        if hashlib.sha256((L / name).read_bytes()).hexdigest() != want:
            raise ValueError(f"{name} differs from 8950b34:{path}")
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
    args = ["--run-dir", f"/ev/{root}/{run}"]  # root: evals_v1, or a snapshot's own folder (its members' launch records and .evals)
    args += ["--arms", "attack"] if kind == "main" else ["--arms", arm, "--self-report-only", arm]
    sys.argv = ["make_phase2_attempt_labels.py", *args]
    m.main()  # prints its own per-arm counts (labels and rules, no text)
    (out,) = sorted((m.HERE / "results").glob("phase2_attempt_labels_*.jsonl"))
    b = out.read_bytes()
    return {"name": out.name, "bytes": b, "sha256": hashlib.sha256(b).hexdigest(), "args": args}


@app.local_entrypoint()
def main(run: str, kind: str, root: str = "evals_v1", arm: str = "sim_attack@september"):
    if arm.startswith("sim_honest") or (kind == "main" and arm != "sim_attack@september"):
        raise SystemExit(f"--arm {arm}: attempt labels are for attack arms (a main launch labels its attack arm)")
    r = label.remote(run, kind, root, arm)
    dest = HERE / "results" / "labels" / r["name"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.open("xb").write(r["bytes"])
    rows = [json.loads(x) for x in r["bytes"].decode().splitlines()]
    from collections import Counter

    print(json.dumps({"file": str(dest.relative_to(HERE)), "sha256": r["sha256"], "args": r["args"], "rows": len(rows),
                      "labels": dict(Counter(str(x["label"]) for x in rows))}))
