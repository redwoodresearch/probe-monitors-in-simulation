"""Gate (b): 5 same-cell pairs (real epoch 1, fixed sim) as one self-contained HTML page, plus a separate answer key.

Cells are drawn from all 939 paired cells with seed 20261009 (PLAN.md "Gate (b)"); the same seed puts the real run left
or right. Each transcript is shown as the reader sees it: the system prompt, the agent's text and tool calls, and the
tool outputs (the fixed run's reader inputs). Columns are labelled only "Transcript 1" and "Transcript 2"; nothing on
the page names an arm, run, environment or task beyond what the transcript itself says.

  uv run python experiments/2026-10-09_fixed_sim/plot_pair_page.py <local copy of the run, e.g. .../2026-10-09_la_fixed-sim_v2>
"""

import html
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REPO = HERE.parents[1]
SEED, N = 20261009, 5

CSS = """
:root { --bg: #fbfbf9; --fg: #1d1d1b; --muted: #6b6b66; --card: #ffffff; --line: #deded8; --call: #f2f5fb;
        --out: #f6f6f2; --sys: #fbf6ec; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #161615; --fg: #e8e8e3; --muted: #9a9a93;
        --card: #1e1e1c; --line: #34342f; --call: #1c2230; --out: #22221f; --sys: #2a2519; } }
:root[data-theme="dark"] { --bg: #161615; --fg: #e8e8e3; --muted: #9a9a93; --card: #1e1e1c; --line: #34342f;
        --call: #1c2230; --out: #22221f; --sys: #2a2519; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.45 system-ui, sans-serif; }
header { padding: 16px; max-width: 1400px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 6px; } h2 { font-size: 17px; margin: 28px 0 8px; }
p { margin: 4px 0; color: var(--muted); }
.pair { max-width: 1400px; margin: 0 auto; padding: 0 16px 24px; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 800px) { .cols { grid-template-columns: 1fr; } }
.col { background: var(--card); border: 1px solid var(--line); border-radius: 8px; min-width: 0; }
.col h3 { margin: 0; padding: 8px 12px; border-bottom: 1px solid var(--line); font-size: 15px; position: sticky; top: 0;
          background: var(--card); }
.scroll { max-height: 80vh; overflow: auto; padding: 8px 12px; }
.msg { margin: 8px 0; } .role { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
pre { white-space: pre-wrap; word-break: break-word; margin: 2px 0; padding: 6px 8px; border-radius: 6px;
      font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }
.system pre { background: var(--sys); } .call pre { background: var(--call); } .tool pre { background: var(--out); }
"""


def render(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m["role"]
        if role == "assistant":
            if m.get("content"):
                parts.append(f'<div class="msg"><div class="role">agent</div><pre>{html.escape(str(m["content"]))}</pre></div>')
            for c in m.get("tool_calls") or []:
                args = c["function"]["arguments"]
                args = json.loads(args) if isinstance(args, str) else args
                parts.append(f'<div class="msg call"><div class="role">tool call: {html.escape(c["function"]["name"])}</div>'
                             f'<pre>{html.escape(json.dumps(args, indent=1, ensure_ascii=False))}</pre></div>')
        else:
            label = {"system": "system prompt", "user": "user", "tool": "tool output"}.get(role, role)
            parts.append(f'<div class="msg {role}"><div class="role">{label}</div><pre>{html.escape(str(m.get("content") or ""))}</pre></div>')
    return "\n".join(parts)


def main(run_dir: str):
    run = Path(run_dir)
    cells = json.loads((REPO / "splits" / "exclusions_v4.json").read_text())["paired_attack_cells"]["cells"]
    rng = random.Random(SEED)
    picks = rng.sample(sorted(cells), N)
    real_left = [rng.random() < 0.5 for _ in picks]
    rows = {}
    for f in sorted((run / "inputs").glob("shard_*.jsonl")):
        for r in map(json.loads, f.read_text().splitlines()):
            if r["cell"] in picks and r["arm"] in ("real-attack-e1", "sim-attack"):
                rows[(r["cell"], r["arm"])] = r
    assert len(rows) == 2 * N, "a picked cell is missing an arm"
    body, key = [], []
    for i, (cell, left) in enumerate(zip(picks, real_left), 1):
        real, sim = rows[(cell, "real-attack-e1")], rows[(cell, "sim-attack")]
        one, two = (real, sim) if left else (sim, real)
        key.append({"pair": i, "real_is": "Transcript 1" if left else "Transcript 2", "cell": cell,
                    "environment": real["environment"], "real_key": real["key"], "sim_key": sim["key"]})
        body.append(f'<section class="pair"><h2>Pair {i}</h2><div class="cols">'
                    f'<div class="col"><h3>Transcript 1</h3><div class="scroll">{render(one["messages"])}</div></div>'
                    f'<div class="col"><h3>Transcript 2</h3><div class="scroll">{render(two["messages"])}</div></div></div></section>')
    page = ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Real or Simulated</title><style>{CSS}</style></head><body><header><h1>Real or simulated?</h1>"
            "<p>Each pair is two runs of the same task. In each pair, one transcript is a real run and the other is simulated. "
            "Which is real?</p></header>" + "\n".join(body) + "</body></html>")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results"
    (HERE / "data").mkdir(exist_ok=True)
    with open(HERE / "data" / f"pair_page_{stamp}.html", "x") as f:  # raw transcript text: data/ (git-ignored) and S3
        f.write(page)
    with open(out / f"pair_page_answer_key_{stamp}.json", "x") as f:
        f.write(json.dumps({"seed": SEED, "run": run.name, "pairs": key}, indent=1))
    print(HERE / "data" / f"pair_page_{stamp}.html")
    print(out / f"pair_page_answer_key_{stamp}.json")


if __name__ == "__main__":
    main(sys.argv[1])
