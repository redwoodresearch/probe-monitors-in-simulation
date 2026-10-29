"""09's header / label class (10-20): every number token in a markdown table's HEADER row of the frozen post, listed literally with the
table it belongs to and its position, for 09's approval by {sha256, count}. A header row is the line just above a table's separator
line (|---|...). Tokens are verify_post.tokens (hidden comments masked). The file holds header cell text (labels); nothing is printed
but counts.

  uv run --no-project --python 3.12 --with numpy==2.3.3 python -I tools/build_header_labels.py <post.md> <out.json>
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_post as VP  # noqa: E402

SEP = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")


def main(post, out):
    b = open(post, "rb").read()
    text = b.decode("utf-8")
    toks, kinds = VP.tokens(text)
    lines, starts = text.split("\n"), [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln) + 1)
    ents, n_tables = [], 0
    for i in range(1, len(lines)):
        if not SEP.match(lines[i]) or not lines[i - 1].lstrip().startswith("|"):
            continue
        n_tables += 1
        h0, h1 = starts[i - 1], starts[i - 1] + len(lines[i - 1])
        bars = [h0 + k for k, ch in enumerate(lines[i - 1]) if ch == "|"]
        for s, e, t in toks:
            if not h0 <= s < h1:
                continue
            c = next((k for k in range(len(bars) - 1) if bars[k] < s < bars[k + 1]), None)
            cell = text[bars[c] + 1:bars[c + 1]].strip() if c is not None else None
            ents.append({"table": n_tables, "header_line": lines[i - 1], "cell_index": None if c is None else c + 1, "cell": cell,
                         "token": t, "token_kind": kinds.get(s, "number"), "position": s})
    rec = {"what": "09's header / label class: number tokens in table header rows of the frozen post, literally, with table and position; "
                   "the class matches only these positions (header rows), never body text or cells",
           "post": {"path": post, "sha256": hashlib.sha256(b).hexdigest()}, "tables": n_tables, "entries": ents}
    json.dump(rec, open(out, "x", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({"tables": n_tables, "header_tokens": len(ents), "tables_with_header_tokens": len({x["table"] for x in ents}),
                      "entries_without_cell": sum(x["cell"] is None for x in ents)}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
