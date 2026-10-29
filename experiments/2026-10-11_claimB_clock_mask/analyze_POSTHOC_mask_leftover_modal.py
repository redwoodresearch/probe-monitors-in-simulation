"""POST HOC, EXPLORATORY (0d's go, before 23:14 PT 10-11, inside the clock mask's <= $50 pre-approval): which of R6's
timestamp families leaves the one R6 change the clock mask didn't hide (cells: 175 of 176 hidden)? Counts and class
names only; no text leaves the container.

One Modal CPU container reads the same stored rows as the runner's mask check (fixed-v2 rows, their R6 records, the
original simulated rows). For each R6-only output whose change survives level 1 + clock mask, it pairs the stamps R6's
own `clock.stamps` finds before and after (in order; R6 rewrites each stamp in place) and, for every stamp pair whose
level 1 + clock mask text differs, records the stamp's family (DATE_CMD, ISO, HTTP, LS, FAILED) and the classes of the
differing tokens (weekday, month, year, day number, other number, zone-like capitals, other letters, other). It also
counts differences outside any stamp.

  cd <worktree root> && PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run \\
      experiments/2026-10-11_claimB_clock_mask/analyze_POSTHOC_mask_leftover_modal.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import modal

HERE = Path(__file__).parent  # container-safe: only HERE and HERE.parent are used at module level
FIXED, ORIG = "2026-10-09_la_fixed-sim_v2", "2026-10-06_la_segment-end_qwen3-32b_v1"
app = modal.App("probe-shift_aswin-surya_sub-claimb-clockmask-diag_kill-2026-11-11")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations")
image = (modal.Image.debian_slim(python_version="3.12").add_local_python_source("probe_shift")
         .add_local_file(HERE / "clock_mask.py", "/root/clockmask/clock_mask.py")
         .add_local_file(HERE.parent / "2026-10-09_fixed_sim" / "clock.py", "/root/fixed_sim/clock.py"))


@app.function(image=image, cpu=4, memory=32768, volumes={"/acts": vol}, timeout=20 * 60, retries=0)
def diagnose() -> dict:
    import gzip
    import re
    import sys
    from collections import Counter

    sys.path[:0] = ["/root/clockmask", "/root/fixed_sim"]
    import clock
    import clock_mask as CM

    from probe_shift.scrub import scrub

    def rows(run):
        out = {}
        for f in sorted(Path(f"/acts/{run}/inputs").glob("shard_*.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["arm"] == "sim-attack":
                    out[r["key"]] = r
        return out

    def klass(tok):
        if re.fullmatch(CM.WEEKDAY_ALT, tok):
            return "weekday"
        if re.fullmatch(CM.MONTH_ALT, tok):
            return "month"
        if re.fullmatch(r"\d{4}", tok):
            return "year_like_4_digits"
        if re.fullmatch(r"\d{1,2}", tok):
            return "day_or_short_number"
        if tok.isdigit():
            return "other_number"
        if re.fullmatch(r"[A-Z]{2,6}", tok):
            return "zone_like_capitals"
        if tok.isalpha():
            return "other_letters"
        return "other"

    fixed, orig = rows(FIXED), rows(ORIG)
    recs = {}
    for f in sorted(Path(f"/acts/{FIXED}/records").glob("records_*.jsonl.gz")):
        for line in gzip.decompress(f.read_bytes()).decode().splitlines():
            x = json.loads(line)
            if x["changed"]:
                recs.setdefault((x["key"], x["msg"]), set()).add(x["rule"])
    c = Counter()
    for (key, msg), rules in recs.items():
        if rules != {"R6"}:
            continue
        pre, post = orig[key]["messages"][msg]["content"], fixed[key]["messages"][msg]["content"]
        if scrub(pre, 1) == scrub(post, 1) or CM.level1_clock(pre) == CM.level1_clock(post):
            continue
        c["leftover_outputs"] += 1
        sp, sq = clock.stamps(pre, 2026), clock.stamps(post, 2026)
        c[f"stamps_before={len(sp)}|after={len(sq)}"] += 1
        for a, b in zip(sp, sq):
            ta = re.findall(r"\w+", CM.level1_clock(pre[a["start"]:a["end"]]))
            tb = re.findall(r"\w+", CM.level1_clock(post[b["start"]:b["end"]]))
            if ta == tb:
                continue
            c[f"differing_stamp_family={a['kind']}->{b['kind']}"] += 1
            c[f"differing_stamp_token_counts={len(ta)}->{len(tb)}"] += 1
            for x, y in zip(ta, tb):
                if x != y:
                    c[f"differing_token_class={klass(x)}->{klass(y)}"] += 1
        # in context: where the whole-text masked tokens differ, relative to the stamps (pass 2, after pass 1 found no
        # difference within the stamps masked alone)
        def cc(ch):
            return ("none" if ch == "" else "digit" if ch.isdigit() else "letter" if ch.isalpha() else "space" if ch.isspace()
                    else {".": "dot", ":": "colon", "-": "hyphen", "/": "slash", ",": "comma", "[": "bracket", "(": "paren",
                          "'": "quote", '"': "quote", "_": "underscore", "+": "plus"}.get(ch, "other_punct"))
        for s in sp:
            c[f"stamp_family={s['kind']}|char_before={cc(pre[s['start'] - 1:s['start']])}|char_after={cc(pre[s['end']:s['end'] + 1])}"] += 1
            alone, ctx = CM.level1_clock(pre[s["start"]:s["end"]]), CM.level1_clock(pre)
            c[f"stamp_masked_alone_appears_in_context={alone in ctx}"] += 1
        fa, fb = re.findall(r"\w+", CM.level1_clock(pre)), re.findall(r"\w+", CM.level1_clock(post))
        c[f"whole_text_token_counts_equal={len(fa) == len(fb)}"] += 1
        for x, y in zip(fa, fb):
            if x != y:
                c[f"whole_text_differing_token_class={klass(x)}->{klass(y)}"] += 1
        # differences outside every stamp: blank the stamps (same length) and compare the rest
        def blank(text, st):
            chars = list(text)
            for s in st:
                chars[s["start"]:s["end"]] = " " * (s["end"] - s["start"])
            return "".join(chars)
        if CM.level1_clock(blank(pre, sp)) != CM.level1_clock(blank(post, sq)):
            c["differs_outside_stamps"] += 1
    return dict(sorted(c.items()))


@app.local_entrypoint()
def main():
    res = diagnose.remote()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out = {"label": "POST HOC, EXPLORATORY: R6 family of the clock mask's leftover, pass 2 with context classes (counts and class names only)",
           "counts": res, "git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"mask_leftover_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("->", dst)
