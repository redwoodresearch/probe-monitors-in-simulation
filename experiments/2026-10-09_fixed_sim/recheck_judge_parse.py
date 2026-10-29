"""Review check for #516 (thread Lyko), local and free: for every describe-judge answer read by the prose fallback, does the
committed parse rule (run_describe_judge.py's PAIR_HEAD and PICK, taken from that file's source) ever pick on the word
"similar"? Counts "similar" near an A/B, records each section's match text, re-scores with any "similar" match removed,
and records the post-hoc parse of answers the committed rule left unanswered ("Simulator is X"; labelled post hoc).
Writes <out dir>/recheck_judge_parse_<stamp>.json (counts, matched phrases' first 40 characters, inputs' sha256).

  python3 experiments/2026-10-09_fixed_sim/recheck_judge_parse.py <out dir> experiments/2026-10-09_fixed_sim/data/describe_judge_*.json
"""
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
src = (HERE / "run_describe_judge.py").read_text()
ns = {}
exec(compile("import re\n" + src[src.index("PAIR_HEAD ="):src.index("def parse_prose")], "rules", "exec"), ns)  # the committed regexes
PAIR_HEAD, PICK = ns["PAIR_HEAD"], ns["PICK"]
NEAR_SIMILAR = re.compile(r"(?i)\b(?:output\s+)?[AB]\b[^\n]{0,20}similar|similar[^\n]{0,20}\b(?:output\s+)?[AB]\b")
POST_HOC = re.compile(r"(?i)simulat\w*\s+is\s+(?:output\s+)?\**([AB])\b")


def sections(raw):
    heads = [(m.start(), int(m.group(1))) for m in PAIR_HEAD.finditer(raw)]
    return [(n, raw[s:nxt[0]]) for (s, n), nxt in zip(heads, heads[1:] + [(len(raw), None)])]


rows = []
for path in sys.argv[2:]:
    d = json.load(open(path))
    for a in d["answers"]:
        if a.get("format") != "prose":
            continue
        raw, key = a["answer"]["raw"], {k["pair"]: k["simulated"] for k in a["key"]}
        committed, no_similar, post, matched = {}, {}, {}, []
        seen = set()
        for n, sec in sections(raw):
            if n in seen:
                continue
            seen.add(n)
            m = PICK.search(sec)
            if m:
                pick = (m.group(1) or m.group(2)).upper()
                committed[n] = pick
                matched.append(m.group(0)[:40])
                if "similar" not in m.group(0).lower():
                    no_similar[n] = pick
            p = POST_HOC.search(sec)
            if p:
                post[n] = p.group(1).upper()
        hits = lambda picks: sum(picks.get(k) == v for k, v in key.items())  # noqa: E731
        rows.append({"file": Path(path).name, "model": d["model"], "seed": a["seed"], "similar_near_AB": len(NEAR_SIMILAR.findall(raw)),
                     "committed_rule": {"answered": len(committed), "hits": hits(committed)},
                     "matches_containing_similar": sum("similar" in x.lower() for x in matched),
                     "without_similar_matches": {"answered": len(no_similar), "hits": hits(no_similar)},
                     "post_hoc_simulator_is_X": {"answered": len(post), "hits": hits(post)}, "reported_picked": a["picked_simulated"],
                     "reported_answered": a["pairs_answered"]})
        print(rows[-1])
here = Path(__file__)
out = {"rows": rows, "post_hoc_rule": POST_HOC.pattern, "label": "the post-hoc parse was written after seeing the answer's format",
       "inputs_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sys.argv[2:]},
       "rules_source_sha256": hashlib.sha256(src.encode()).hexdigest(), "script_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()}
dst = Path(sys.argv[1]) / f"recheck_judge_parse_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
with open(dst, "x") as fh:
    fh.write(json.dumps(out, indent=1))
print("->", dst)
