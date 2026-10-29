"""POST HOC (source audit follow-up 1; 0d's go after the audit, under Aswin's under-$5 rule): does rule R6 (the clock)
leave anything the level-1 mask doesn't hide, and are date-like strings balanced across fixed sim, e1 and e2 at level 1?

Level 1 (`probe_shift.scrub.scrub(text, 1)`) masks ISO dates, clock times, month-day dates and epoch seconds. It leaves
weekday names, day-month-year dates (the HTTP form `18 Sep 2026`) and bare years. R6 moves whole timestamps, and with
them weekday names and, when the simulator's clock was a year off, the year.

One Modal CPU container reads stored rows only (no model, no text leaves the container; numbers only):
- A. R6's own edits. For every simulated output R6 changed (`records/` of `2026-10-09_la_fixed-sim_v2`, rule R6), the
  text before R6 (the original simulated output from `2026-10-06_la_segment-end_qwen3-32b_v1`) and after (the fixed
  row) are masked at level 1 and compared token by token (`\\w+` tokens): position by position when the two token
  lists have the same length (R6 keeps each stamp's format), else by difflib, counted separately. Outputs another rule
  also changed are counted and skipped, since their text before R6 is not stored. The R6 output total must equal the
  build's 2,167 (fixed_sim/README.md:96), and every input shard must have its records shard. Reported: outputs and runs
  whose change survives level 1, and the surviving changed tokens by class (weekday, month, year, other number, other).
- B. Date-like strings at level 1 in every arm's tool outputs (fixed sim, e1, e2 from the same fixed-v2 rows; the real
  rows there are byte copies of the reader inputs): per run, the counts of weekday names and month names (the level-4
  masks' own patterns, `scrub.WEEKDAYS` and `scrub.MONTH_NAMES`, full names included), day-month-year dates and
  4-digit years 2000-2099 left after masking. Reported per arm: share of runs with any, and the mean and
  median count per run. All environments and the 5 test environments.
Counts cover each whole transcript, not only its first 32,768 tokens (the probe's cut), so they are upper bounds on what
the probe saw.

  cd <worktree root> && PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run \\
      experiments/2026-10-11_claimB_source_audit/analyze_POSTHOC_r6_level1_modal.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import modal

HERE = Path(__file__).parent
FIXED, ORIG = "2026-10-09_la_fixed-sim_v2", "2026-10-06_la_segment-end_qwen3-32b_v1"
ARMS = ("sim-attack", "real-attack-e1", "real-attack-e2")
TEST = {"clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"}
app = modal.App("probe-shift_aswin-surya_sub-claimb-audit_r6l1_kill-2026-11-11")
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations")
image = modal.Image.debian_slim(python_version="3.13").add_local_python_source("probe_shift")


@app.function(image=image, volumes={"/acts": acts}, cpu=8, memory=32768, timeout=40 * 60, retries=0)
def count() -> dict:
    import difflib
    import gzip
    import re
    from collections import Counter, defaultdict
    from statistics import median

    from probe_shift.scrub import MONTH_NAMES, WEEKDAYS, scrub

    month_alt = MONTH_NAMES.pattern.removeprefix(r"\b(?:").removesuffix(r")\b")
    assert "September" in month_alt and "Sept" in month_alt, month_alt
    rx = {"weekday": WEEKDAYS, "month": MONTH_NAMES,
          "day_month_year": re.compile(rf"\b\d{{1,2}}[ \t]+(?:{month_alt})\.?,?[ \t]+\d{{4}}\b"),
          "year": re.compile(r"\b20\d{2}\b")}

    def kind(tok: str) -> str:
        if WEEKDAYS.fullmatch(tok):
            return "weekday"
        if MONTH_NAMES.fullmatch(tok):
            return "month"
        if re.fullmatch(r"(19|20)\d{2}", tok):
            return "year"
        return "number" if tok.isdigit() else "other"

    def rows(run: str) -> dict:
        out = {}
        for f in sorted(Path(f"/acts/{run}/inputs").glob("shard_*.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["arm"] in ARMS:
                    out[(r["arm"], r["key"])] = r
        return out

    fixed = rows(FIXED)
    cells = Counter(r["arm"] for r in fixed.values())
    assert cells["sim-attack"] == cells["real-attack-e1"] == cells["real-attack-e2"] == 939, dict(cells)
    orig = rows(ORIG)
    recs = defaultdict(lambda: defaultdict(set))  # key -> msg -> rules that changed it
    base = Path(f"/acts/{FIXED}")
    manifest = json.loads((base / "manifest.json").read_text())["files"]
    in_ids = {p.name.removeprefix("shard_").removesuffix(".jsonl") for p in (base / "inputs").glob("shard_*.jsonl")}
    rec_files = sorted((base / "records").glob("records_*.jsonl.gz"))
    rec_ids = {p.name.removeprefix("records_").removesuffix(".jsonl.gz") for p in rec_files}
    assert in_ids and in_ids == rec_ids, (sorted(in_ids), sorted(rec_ids))
    for rel in [f"inputs/shard_{i}.jsonl" for i in in_ids] + [f"records/records_{i}.jsonl.gz" for i in rec_ids]:
        assert hashlib.sha256((base / rel).read_bytes()).hexdigest() == manifest[rel], f"{rel} differs from the build manifest"
    for f in rec_files:
        for line in gzip.decompress(f.read_bytes()).decode().splitlines():
            x = json.loads(line)
            if x["changed"]:
                recs[x["key"]][x["msg"]].add(x["rule"])

    # A. R6's own edits after level-1 masking
    a = Counter()
    tok_class = {"all": Counter(), "test": Counter()}
    tok_class_fallback = {"all": Counter(), "test": Counter()}
    runs_surviving = {"all": set(), "test": set()}
    r6_runs = {"all": set(), "test": set()}
    for (arm, key), r in fixed.items():
        if arm != "sim-attack":
            continue
        scope = ("all", "test") if r["environment"] in TEST else ("all",)
        before = orig[("sim-attack", key)]["messages"]
        assert len(before) == len(r["messages"]), key
        for msg, rules in recs.get(key, {}).items():
            if "R6" not in rules:
                continue
            for s in scope:
                r6_runs[s].add(key)
                a[(s, "r6_outputs")] += 1
            if rules != {"R6"}:
                for s in scope:
                    a[(s, "r6_outputs_also_changed_by_another_rule_skipped")] += 1
                continue
            pre, post = before[msg]["content"], r["messages"][msg]["content"]
            assert before[msg]["role"] == "tool" and pre != post, key
            l1_pre, l1_post = scrub(pre, 1), scrub(post, 1)
            if l1_pre == l1_post:
                for s in scope:
                    a[(s, "r6_outputs_fully_masked_at_level1")] += 1
                continue
            t_pre, t_post = re.findall(r"\w+", l1_pre), re.findall(r"\w+", l1_post)
            for s in scope:
                a[(s, "r6_outputs_surviving_level1")] += 1
                runs_surviving[s].add(key)
            if len(t_pre) == len(t_post):  # R6 keeps each stamp's format: positions line up
                changed = [b for a_, b in zip(t_pre, t_post) if a_ != b]
                for s in scope:
                    tok_class[s].update(kind(t) for t in changed)
            else:  # reported apart: difflib can misalign repeated stamps
                ops = difflib.SequenceMatcher(None, t_pre, t_post, autojunk=False).get_opcodes()
                for s in scope:
                    a[(s, "r6_outputs_surviving_with_unequal_token_counts")] += 1
                    tok_class_fallback[s].update(kind(t) for op, _, _, j1, j2 in ops if op in ("replace", "insert") for t in t_post[j1:j2])
                    a[(s, "fallback_tokens_deleted")] += sum(i2 - i1 for op, i1, i2, _, _ in ops if op in ("replace", "delete"))
    part_a = {s: {"sim_runs": sum(1 for (arm, _), r in fixed.items() if arm == "sim-attack" and (s == "all" or r["environment"] in TEST)),
                  "runs_with_r6_change": len(r6_runs[s]), "runs_with_r6_change_surviving_level1": len(runs_surviving[s]),
                  **{k: a[(s, k)] for k in ("r6_outputs", "r6_outputs_fully_masked_at_level1", "r6_outputs_surviving_level1",
                                            "r6_outputs_surviving_with_unequal_token_counts",
                                            "r6_outputs_also_changed_by_another_rule_skipped")},
                  "surviving_changed_tokens_by_class": dict(tok_class[s]),
                  "fallback_inserted_tokens_by_class": dict(tok_class_fallback[s]),
                  "fallback_tokens_deleted": a[(s, "fallback_tokens_deleted")]} for s in ("all", "test")}

    assert a[("all", "r6_outputs")] == 2167, f"R6 outputs {a[('all', 'r6_outputs')]} != the build's 2,167"

    # B. date-like strings left at level 1, every arm, tool outputs only
    per = {s: {arm: defaultdict(list) for arm in ARMS} for s in ("all", "test")}
    for (arm, key), r in fixed.items():
        text = "\n".join(scrub(m["content"], 1) for m in r["messages"] if m["role"] == "tool" and isinstance(m["content"], str))
        scope = ("all", "test") if r["environment"] in TEST else ("all",)
        for name, pat in rx.items():
            n = len(pat.findall(text))
            for s in scope:
                per[s][arm][name].append(n)
    part_b = {s: {arm: {name: {"runs": len(v), "share_with_any": sum(x > 0 for x in v) / len(v), "mean": sum(v) / len(v),
                               "median": median(v)} for name, v in d.items()} for arm, d in per[s].items()} for s in ("all", "test")}
    return {"rows": dict(cells), "A_r6_edits_after_level1": part_a, "B_date_like_strings_at_level1": part_b}


@app.local_entrypoint()
def main():
    res = count.remote()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    res = {"label": "POST HOC (source audit follow-up 1): R6's edits and date-like strings after level-1 masking (numbers only)",
           **res, "git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "note": "whole transcripts, not only the first 32,768 tokens the probe read: upper bounds"}
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")  # laptop local time (PT)
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"r6_level1_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("->", dst)
