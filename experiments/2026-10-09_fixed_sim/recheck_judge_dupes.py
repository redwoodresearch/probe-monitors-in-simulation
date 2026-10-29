"""Review check for #549 (thread MBiK), local and free: recheck_judge_parse.py keeps the first section when a pair number
repeats, while the committed rule (run_describe_judge.parse_prose) drops that pair. Counts repeated pair numbers in every
prose answer and re-scores both the committed rule and the post-hoc parse with repeats dropped, against the reported
numbers. Counts only. Writes <out dir>/recheck_judge_dupes_<stamp>.json.

  python3 experiments/2026-10-09_fixed_sim/recheck_judge_dupes.py <out dir> \
      experiments/2026-10-09_fixed_sim/data/describe_judge_*.json
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
src, chk = (HERE / "run_describe_judge.py").read_text(), (HERE / "recheck_judge_parse.py").read_text()
ns = {}
# the committed rule (PAIR_HEAD, PICK, parse_prose), then recheck_judge_parse.py's post-hoc parse
exec(compile("import re\n" + src[src.index("PAIR_HEAD ="):src.index("def prompt_for")], "rules", "exec"), ns)
exec(compile("import re\n" + chk[chk.index("NEAR_SIMILAR ="):chk.index("def sections")], "post_hoc", "exec"), ns)
PAIR_HEAD, parse_prose, POST_HOC = ns["PAIR_HEAD"], ns["parse_prose"], ns["POST_HOC"]


def post_hoc(raw, drop_repeats):
    heads = [(m.start(), int(m.group(1))) for m in PAIR_HEAD.finditer(raw)]
    picks, seen = {}, set()
    for (s, n), nxt in zip(heads, heads[1:] + [(len(raw), None)]):
        if n in seen:
            if drop_repeats:
                picks.pop(n, None)
            continue
        seen.add(n)
        p = POST_HOC.search(raw[s:nxt[0]])
        if p:
            picks[n] = p.group(1).upper()
    return picks


rows = []
for path in sys.argv[2:]:
    d = json.loads(Path(path).read_text())
    for a in d["answers"]:
        if a.get("format") != "prose":
            continue
        raw, key = a["answer"]["raw"], {k["pair"]: k["simulated"] for k in a["key"]}
        score = lambda p: {"answered": len(p), "hits": sum(p.get(k) == v for k, v in key.items())}
        reps = {n: c for n, c in Counter(int(m.group(1)) for m in PAIR_HEAD.finditer(raw)).items() if c > 1}
        rows.append({"file": Path(path).name, "model": d["model"], "seed": a["seed"], "repeated_pair_numbers": len(reps),
                     "extra_sections": sum(c - 1 for c in reps.values()), "committed_rule": score(parse_prose(raw)),
                     "reported": {"answered": a["pairs_answered"], "hits": a["picked_simulated"]},
                     "post_hoc_keep_first": score(post_hoc(raw, False)),
                     "post_hoc_drop_repeats": score(post_hoc(raw, True))})
        print(rows[-1])
out = {"rows": rows, "prose_answers": len(rows), "answers_with_repeats": sum(r["repeated_pair_numbers"] > 0 for r in rows),
       "committed_matches_reported": all(r["committed_rule"] == r["reported"] for r in rows),
       "post_hoc_unchanged_by_dropping_repeats": all(r["post_hoc_keep_first"] == r["post_hoc_drop_repeats"] for r in rows),
       "inputs_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sys.argv[2:]},
       "rules_source_sha256": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()
                                for f in ("run_describe_judge.py", "recheck_judge_parse.py")},
       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=HERE).stdout.strip()}
print({k: v for k, v in out.items() if k != "rows"})
stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')
dst = Path(sys.argv[1]) / f"recheck_judge_dupes_{stamp}.json"
with open(dst, "x") as fh:
    fh.write(json.dumps(out, indent=1))
print("->", dst)
