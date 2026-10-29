"""Row 11 sizing input (counts only; no scores read): under the "hide" instruction in the E1 fresh confirmation, how many
reports named the changed file anyway (the section-aware disclosure rule, `analyze_e1.discloses_section`), and how many
hid it. Also how many of each the probe could score with the path masked: finished by stop, reasoning closed, a
non-empty report with no tool call, and at least 10 scored tokens without the changed path (the frozen rule's
"scored without the changed path" mask). By arm (honest A, honest B) and task. Local, $0.

  uv run --no-project --python 3.12 --with "transformers==5.19.0" --with jinja2 python \\
      experiments/2026-10-11_q2_hide_same_instruction/count_hide_reports.py <confirm stage folder> <confirm index jsonl>
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "2026-10-11_q2_e1_honest"))
from analyze_e1 import MODEL, REVISION, discloses_section, report_text  # noqa: E402

CONF = HERE.parent / "2026-10-11_q2_e1_confirm"
MASKS = CONF / "results/confirm_masks_2026-10-11_185020.json"
INDEX_SHA = "a4217da0379df98b118584e6015f5fe7eb49d7739d5ccd5f5127937304d2a127"


def main(folder: Path, index: Path):
    if hashlib.sha256(index.read_bytes()).hexdigest() != INDEX_SHA:
        raise SystemExit("confirm index differs")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(Path(__file__).resolve())], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    rows = {r["key"]: r for r in map(json.loads, index.open())}
    masks = json.loads(MASKS.read_text())["masks"]; flags = json.loads((folder / "flags.json").read_text())
    c, tasks = Counter(), {"named": set(), "hid": set()}
    for f in sorted((folder / "generations").glob("*.json")):
        g = json.loads(f.read_text()); key = g["key"]; r = rows[key]; arm = r.get("arm") or key.rsplit("__", 1)[-1]
        rep = report_text(tok, g["hide"]["completion_ids"])
        if not rep.strip():
            c[(arm, "empty")] += 1; continue
        group = "named" if discloses_section(rep, r["path"]) else "hid"
        fl = flags[f"{key}|hide"]
        ok = fl["stop"] and fl["think closed"] and fl["report non-empty"] and not fl["tool call"]
        n_nopath = len(masks.get(f"{key}|hide", {}).get("scored without the changed path", []))
        c[(arm, group)] += 1; c[(arm, group, "scoreable, path masked")] += bool(ok and n_nopath >= 10)
        if ok and n_nopath >= 10:
            tasks[group].add(r["main_task"])
    out = {"git": head, "index sha256": INDEX_SHA, "masks": MASKS.name,
           "hide reports by arm and group": {" | ".join(k): v for k, v in sorted(c.items())},
           "tasks with a scoreable report": {g: len(v) for g, v in tasks.items()}, "tasks in both groups": len(tasks["named"] & tasks["hid"])}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"hide_counts_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
