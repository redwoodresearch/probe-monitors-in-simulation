"""Do text_editor edit results carry the real tool's numbered column (`  12|- old`), or plain unified-diff lines?

The September simulator was told to write plain unified diffs; sims commit 4b31bd30 (in today's code) requires the
real tool's numbered column. For each run folder given as `label=dir` (folders of `runs/<main>/*.eval`, or a flat
folder of .eval files), counts the text_editor results that are edit diffs and classifies each diff's body lines.

  uv run python experiments/2026-10-06_astra_sim_honest/analyze_editor_format.py today_honest=<dir> sept_honest=<dir> ...
"""

import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

from probe_shift.transcripts import tool_result_text

NUMBERED = re.compile(r"^\s*\d+\|[ +\-] ")
PLAIN = re.compile(r"^[ +\-](?![\-+]{2} )")  # unified-diff body line, not the ---/+++ headers


def edit_results(path: Path):
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                for m in json.loads(z.read(name))["messages"]:
                    text = tool_result_text(m) if m["role"] == "tool" and m.get("function") == "text_editor" else ""
                    if "has been edited. Here's a diff" in text:
                        yield text


for arg in sys.argv[1:]:
    label, folder = arg.split("=", 1)
    logs = sorted(Path(folder).glob("runs/*/*.eval")) or sorted(Path(folder).glob("*.eval"))
    c = Counter()
    for log in logs:
        for text in edit_results(log):
            body = text.split("@@", 1)[1].splitlines()[1:] if "@@" in text else []
            numbered = sum(bool(NUMBERED.match(l)) for l in body)
            plain = sum(bool(PLAIN.match(l)) and not NUMBERED.match(l) for l in body)
            c["edit results"] += 1
            c["all numbered" if numbered and not plain else "all plain" if plain and not numbered else
              "mixed" if numbered and plain else "no hunk lines"] += 1
    print(f"{label:16s} {len(logs):3d} logs | " + ", ".join(f"{k}: {v}" for k, v in c.items()))
