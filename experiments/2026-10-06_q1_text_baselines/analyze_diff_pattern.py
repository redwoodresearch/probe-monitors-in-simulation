"""Is level 5's top tell ("\\n+\\n ", an empty added line followed by a context line) diff structure or content?

Counts the pattern in level-5 text per arm, splits it by where in the hunk the empty added line sits, and writes
5 real and 5 simulated editor diffs that contain it, as the original text around the match and after the scrub, to
results/diff_pattern_<stamp>.md.

  uv run python experiments/2026-10-06_q1_text_baselines/analyze_diff_pattern.py
"""

import gzip
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from analyze_tfidf import EDITOR_DIFF, scrub

HERE = Path(__file__).parent
EXAMPLES = "examples_2026-10-06_142532"
PATTERN = "\n+\n "
ARMS = ("real-attack-e1", "sim-attack")


def blank_added_positions(text: str) -> Counter:
    """For each empty added line in a level-5 diff: what comes next (context, added, removed, hunk header, end)."""
    lines, out = text.split("\n"), Counter()
    for i, line in enumerate(lines):
        if line == "+":
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            out["next: " + ("end of output" if nxt is None else "hunk header" if nxt.startswith("@@") else
                            "added" if nxt.startswith("+") else "removed" if nxt.startswith("-") else "context")] += 1
    return out


def main():
    rows = [json.loads(l) for l in gzip.open(HERE / ".run_cache" / EXAMPLES / "outputs.jsonl.gz", "rt")]
    diffs = [r for r in rows if r["kept"] and r["tool"] == "text_editor" and EDITOR_DIFF in r["readout_text"] and r["arm"] in ARMS]
    stats, picks = {}, {}
    random.seed(0)
    for arm in ARMS:
        sub = [r for r in diffs if r["arm"] == arm]
        level5 = [scrub(r["readout_text"], 5) for r in sub]
        hits = [(r, t) for r, t in zip(sub, level5) if PATTERN in t]
        pos = sum((blank_added_positions(t) for t in level5), Counter())
        stats[arm] = {"diffs": len(sub), "with_pattern": len(hits), "share": round(len(hits) / len(sub), 3),
                      "empty_added_lines": sum(pos.values()), **dict(pos)}
        picks[arm] = random.sample(hits, min(5, len(hits)))
    lines = [f"# Level-5 tell `\\n+\\n ` (empty added line, then a context line): structure or content? ({EXAMPLES})", "",
             "Editor diffs (str_replace and similar), kept outputs, real epoch 1 vs simulated.", "",
             "| | diffs | containing the pattern | empty added lines | followed by context | by added | by removed | by hunk header | at end |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm, s in stats.items():
        lines.append(f"| {arm} | {s['diffs']} | {s['with_pattern']} ({s['share']:.1%}) | {s['empty_added_lines']} | "
                     f"{s.get('next: context', 0)} | {s.get('next: added', 0)} | {s.get('next: removed', 0)} | "
                     f"{s.get('next: hunk header', 0)} | {s.get('next: end of output', 0)} |")
    lines.append("")
    for arm in ARMS:
        lines += [f"## {arm}: 5 random diffs containing the pattern", ""]
        for k, (r, t) in enumerate(picks[arm], 1):
            i = t.find(PATTERN)
            lines += [f"### {k}. `{r['key'][:90]}` step {r['step']}", "", "After the level-5 scrub, around the match:", "",
                      "```", t[max(0, i - 300):i + 300], "```", "", "Original (cleaned text), first 900 characters of the diff:", "",
                      "```", r["readout_text"][r["readout_text"].find("@@"):][:900], "```", ""]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"diff_pattern_{stamp}.md"
    out.write_text("\n".join(lines) + "\n")
    print(json.dumps(stats, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
