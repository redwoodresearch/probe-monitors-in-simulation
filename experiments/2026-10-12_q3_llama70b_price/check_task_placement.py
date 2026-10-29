"""Where does the main-task description sit in the policy's prompt? Read-only, $0 (Aswin, 2026-10-12 about 15:16 PT).

  uv run --no-project --python 3.14 --with tokenizers python check_task_placement.py --llama-run <smoke run dir copy> \\
      --tokenizer-dir <Llama tokenizer dir> --compare <.eval of another model's run of the same task>

For each .eval (the Llama smoke's runs, and --compare): the sample's metadata main_task.description, the system and
first user messages' lengths, and whether and where the description appears in each in full. For the Llama runs also
the first policy request's decoded prompt (proxy 200 record joined by x-irid): which turn holds the description, at
what offset, the turn's length, and whether it is there in full. Writes results/task_placement_<stamp>.json (counts and
offsets only); the description's first line prints to the console.
"""

import argparse
import gzip
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from check_tool_outputs_llama import POLICY, decoder, load_sample, text_of
from tokenizers import Tokenizer

HERE = Path(__file__).parent
HEADER = re.compile(rb"<\|start_header_id\|>(\w+)<\|end_header_id\|>\n\n")


def placement(path: Path) -> tuple[dict, dict]:
    s = load_sample(path)
    desc = (s.get("metadata") or {}).get("main_task", {}).get("description") or ""
    msgs = s["messages"]
    out = {"eval": path.name, "eval_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "model": next((e.get("model") for e in s["events"] if e.get("event") == "model"), None),
           "description_chars": len(desc), "description_sha256": hashlib.sha256(desc.encode()).hexdigest()}
    for role in ("system", "user"):
        m = next((m for m in msgs if m["role"] == role), None)
        t = text_of(m["content"]) if m else ""
        out[f"first_{role}_message_chars"] = len(t)
        out[f"description_in_first_{role}_message_in_full"] = bool(desc) and desc in t
        out[f"description_offset_in_first_{role}_message"] = t.find(desc) if desc else None
    return out, s


def main(llama_run: str, tokenizer_dir: str, compare: str):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    run = Path(llama_run)
    to_bytes = decoder(Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json")))
    recs = {}
    for f in sorted((run.parent / f"{run.name}.request_logs").glob("requests_*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("kind") != "header" and r.get("status") == 200 and r.get("prompt_token_ids"):
                    recs[r["request_id"]] = r
    out = {"git": git, "llama_run": run.name, "runs": {}}
    first_line = None
    for ev in sorted(p for p in run.rglob("*.eval") if "references" not in p.parts):
        row, s = placement(ev)
        desc = (s.get("metadata") or {}).get("main_task", {}).get("description") or ""
        first_line = first_line or desc.splitlines()[0] if desc else first_line
        e = next(e for e in s["events"] if e.get("event") == "model" and e.get("model") == POLICY)
        r = recs.get(e["call"]["request"]["extra_headers"]["x-irid"])
        if r is not None:
            prompt = b"".join(to_bytes(i) for i in r["prompt_token_ids"])
            heads = list(HEADER.finditer(prompt))
            d = desc.encode()
            for k, h in enumerate(heads[:3]):
                turn = prompt[h.end():heads[k + 1].start() if k + 1 < len(heads) else len(prompt)]
                row[f"prompt_turn{k}_{h.group(1).decode()}_bytes"] = len(turn)
                row[f"prompt_turn{k}_description_offset"] = turn.find(d)
            row["prompt_description_occurrences"] = prompt.count(d)
            row["prompt_tokens"] = len(r["prompt_token_ids"])
        out["runs"]["/".join(ev.relative_to(run).parts[:2])] = row
    out["compare"], _ = placement(Path(compare))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"task_placement_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("task description, first line:", repr(first_line))
    print(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-run", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--compare", required=True)
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
