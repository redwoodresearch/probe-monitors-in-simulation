"""Census of unresolved Inspect attachment placeholders ('attachment://') in the run histories df's pipelines read
(Aswin's request, relayed 2026-10-11 ~16:55 PT). Modal CPU where the data lives; counts only, no text.

Per input group (its files listed): rows, rows with any placeholder, placeholder count, by message role and by arm, and rows carrying an
'attachments' key. Files: the honest-run inputs and claim B's level-1 packs (la-activations), and the built
dialogue files of the tool-output cue probe and N1 (evalaware acts).

  PYTHONPATH=src uv run --no-project --with modal modal run experiments/2026-10-11_tool_output_cue_probe/run_attachment_census.py
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-toc_census_kill-2026-11-10"
la = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only()
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09", version=2).read_only()
app = modal.App(APP)
FILES = {"honest-run inputs (toc, N1)": ("/in", "2026-10-06_la_segment-end-scrub1_qwen3-32b_v1/inputs/shard_*.jsonl"),
         "claim-B level-1 packs (toc planted/stage 2, N1 panel, Q3.3, concept probe)": ("/in", "2026-10-10_la_fixed-sim-v2_probe-inputs-b-l1_v1/inputs/shard_*.jsonl"),
         "toc built dialogues": ("/acts", "toc/dialogues_2026-10-11_154759.jsonl"), "toc built planted runs": ("/acts", "toc/planted_2026-10-11_154759.jsonl"),
         "N1 built pairs": ("/acts", "n1/pairs_*.jsonl")}


@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes={"/in": la, "/acts": acts}, cpu=4, memory=16384, timeout=1800, retries=0)
def census() -> dict:
    out = {}
    for name, (root, pattern) in FILES.items():
        files = sorted(Path(root).glob(pattern))
        if not files:
            raise FileNotFoundError(f"{name}: no file matches {pattern}")
        r = {"files": [str(f.relative_to(root)) for f in files], "rows": 0, "rows with a placeholder": 0, "placeholders": 0,
             "by role": {}, "rows with a placeholder by arm": {}, "rows with an 'attachments' key": 0}
        for f in files:
            for line in f.open():
                row = json.loads(line)
                r["rows"] += 1
                r["rows with an 'attachments' key"] += "attachments" in row
                n_row = 0
                for m in row.get("messages") or []:
                    k = json.dumps(m).count("attachment://")
                    if k:
                        r["by role"][m.get("role")] = r["by role"].get(m.get("role"), 0) + k
                        n_row += k
                if n_row:
                    r["rows with a placeholder"] += 1
                    r["placeholders"] += n_row
                    arm = row.get("arm") or row.get("version") or f"label {row.get('label')}"
                    r["rows with a placeholder by arm"][arm] = r["rows with a placeholder by arm"].get(arm, 0) + 1
        out[name] = r
    return out


@app.local_entrypoint()
def main():
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    out = {"git": git, "counted": census.remote()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"attachment_census_{stamp}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), f"\n-> {path}")
