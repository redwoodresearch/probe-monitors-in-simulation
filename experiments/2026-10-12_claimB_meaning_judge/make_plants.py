"""Meaning judge: the planted check's edited copies (PLAN.md), by script from positions, seeded. $0, local.

From the e1 replies of data/sites.jsonl (honest tool outputs), random.Random("meaning-plants") draws:
20 line deletions (a reply with 3+ non-empty lines loses one line of 3+ visible characters that occurs once), 20 number
changes (a number that occurs once and is not inside a date or time gets +1, or 1 if it ends in 9: same length unless
it rolls over), 10 date/time controls (v2: one seeded ISO date or HH:MM[:SS] time in a reply gets the day's last digit - 1 or the
second's + 1), 10 self controls (unedited). Each copy is then labelled by Codex (`check_plants`): only edits it confirms
count. Text goes to data/plants.jsonl only; results/plants_*.json holds sites, kinds, offsets and hashes.

  python3 experiments/2026-10-12_claimB_meaning_judge/make_plants.py
"""

import hashlib
import json
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
DT = re.compile(r"\d{4}-\d{2}-\d{2}|\b\d{2}:\d{2}(?::\d{2})?\b")  # v2: HH:MM too
H = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731


def line_delete(t, rng):
    lines = t.split("\n")
    ok = [i for i, ln in enumerate(lines) if len(ln.strip()) >= 3 and lines.count(ln) == 1]
    if sum(bool(x.strip()) for x in lines) < 3 or not ok:
        return None
    i = rng.choice(ok)
    return "\n".join(lines[:i] + lines[i + 1:]), {"line": i}


def number_change(t, rng):
    spans = [m.span() for m in DT.finditer(t)]
    nums = [m for m in re.finditer(r"(?<![\w.])\d+(?![\w.])", t) if len(re.findall(rf"(?<![\w.]){m.group()}(?![\w.])", t)) == 1
            and not any(a <= m.start() < b for a, b in spans)]
    if not nums:
        return None
    m = rng.choice(nums)
    new = str(int(m.group()) + 1) if not m.group().endswith("9") else m.group()[:-1] + "1"
    return t[:m.start()] + new + t[m.end():], {"offset": m.start()}


def date_control(t, rng):
    ms = list(DT.finditer(t))
    if not ms:
        return None
    m = rng.choice(ms)  # v2: one seeded occurrence among any number
    g = m.group()
    d = int(g[-1])  # times: the second's last digit + 1 (mod 10); dates: the day's last digit - 1 (01 -> 02, x0 -> x1)
    new = g[:-1] + str((d + 1) % 10 if ":" in g else (d - 1 if d >= 1 and g[-2:] != "01" else d + 1))
    return t[:m.start()] + new + t[m.end():], {"offset": m.start()}


def main():
    sites = [json.loads(line) for line in (HERE / "data/sites.jsonl").read_text().splitlines()]
    rng = random.Random("meaning-plants")
    order = sites[:]
    rng.shuffle(order)
    plants, used = [], set()
    for kind, fn, n in (("delete_line", line_delete, 20), ("change_number", number_change, 20), ("date_control", date_control, 10)):
        got = 0
        for s in order:
            if got == n:
                break
            if s["site"] in used or not s["e1"]:
                continue
            r = fn(s["e1"], rng)
            if r and r[0] != s["e1"]:
                plants.append({"plant": len(plants), "site": s["site"], "kind": kind, "edited": r[0], "where": r[1]})
                used.add(s["site"])
                got += 1
        assert got == n, f"only {got} {kind}"
    for s in [s for s in order if s["site"] not in used][:10]:
        plants.append({"plant": len(plants), "site": s["site"], "kind": "self_control", "edited": s["e1"], "where": {}})
    with open(HERE / "data/plants.jsonl", "x") as fh:
        fh.writelines(json.dumps(p) + "\n" for p in plants)
    out = {"label": "Meaning judge plants (no text)", "seed": "meaning-plants",
           "plants": [{k: p[k] for k in ("plant", "site", "kind", "where")} | {"edited_sha256": H(p["edited"])} for p in plants],
           "data_sha256": hashlib.sha256((HERE / "data/plants.jsonl").read_bytes()).hexdigest(),
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"plants_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out) + "\n")
    print({k: sum(p["kind"] == k for p in plants) for k in ("delete_line", "change_number", "date_control", "self_control")})


if __name__ == "__main__":
    main()
