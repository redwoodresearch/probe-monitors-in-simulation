"""Documentation of every fix (free; local CPU): what each applied rule does, the simulated outputs it changed per
environment, seeded before/after examples beside a real output of the same kind, and the rules derived but dropped.

  counts:   from the v2 run's records/ (one row per output a rule looked at; R5 and R6 rows only for outputs they
            changed), checked against a byte comparison of every simulated output, original vs v2. Real side: every
            real e1 and e2 row of v2 compared with the original reader inputs (byte-identical means no rule changed it).
  examples: per rule, its changed outputs sorted by id and shuffled with random.Random(SEED + rule); the first 2 not in
            --skip (the output, before or after, or its real neighbour is about the attack itself; decided by reading).
            Beside each, a real e1 output of the same kind (R1 insert, R2 str_replace, R5 dpkg marker, R6 a parseable
            clock stamp), seeded per example, from the same environment and type label if one exists.
  shown:    raw text (level 0), credential-like strings masked, 20 lines from just above the first changed line (R5, R6)
            or from the top (R1, R2); never the agent's calls. Writes data/fix_docs_<stamp>.md (transcript text; S3 only),
            results/fix_docs_counts_<stamp>.json (no text) and results/fix_counts_<stamp>.png.

  PYTHONPATH=src uv run --no-project --with numpy --with scipy --with scikit-learn --with matplotlib python experiments/2026-10-09_fixed_sim/make_fix_docs.py \
      <local 2026-10-09_la_fixed-sim_v2> <local 2026-10-06_la_segment-end_qwen3-32b_v1> [--skip id,id] [--draft out.md]
"""

import argparse
import gzip
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import clock  # noqa: E402
import fixes  # noqa: E402
from make_probe_examples import excerpt, label, mask, oid  # noqa: E402

SEED, N_EX = "fix-docs-2026-10-10:", 2
SPLIT = json.loads((HERE / "split.json").read_text())
ENVS = sorted(SPLIT["dev"]) + sorted(SPLIT["test"])
RULES = {
    "R1": "File-editor inserts: the simulator's diff becomes the real tool's numbered snippet of the file plus its "
          "\"Review the changes…\" note, with no trailing newline.",
    "R2": "File-editor replacements (str_replace): the plain diff gets the real tool's line-number gutter "
          "(`   N|+ code`) and loses its trailing newline.",
    "R5": "Package installs (dpkg/apt): dpkg's own lines end in a carriage return, and each run of \"Reading database\" "
          "lines becomes one real 20-step progress line.",
    "R6": "Clock (version 2): in each simulated run, the timestamps within 48 hours of the run's \"now\", in four formats "
          "(`date` output, ISO, HTTP dates, `ls -l` times), move by one whole-second offset so its \"now\" lands on a real "
          "run's \"now\" (donors: real runs of the dev environments; \"now\" read only from stamps dated September 14–21). "
          "Timestamps outside that window or in other formats stay as they are.",
}

DROPPED = """
## Rules derived but dropped

Numbers in this section are quoted from RULES.md (10e856e7), PLAN.md (f5c17889) and README.md (28275300), not
recomputed here.

- **R3, short diff headers** (`@@ -N +N @@` to the full `@@ -a,b +c,d @@`). Dropped because the dev count found none in
  either arm, and Python's difflib writes the short form for one-line ranges too, so it is not a known simulator
  mistake. **Caveat: this drop was never truly measured.** The pattern that counted full headers also accepted short
  ones, and it ran first, so the short-header count could not be anything but zero. Whether simulated runs print short
  headers is unknown.
- **R4, re-diffing empty added or removed lines** so they come out as the real tool would show them. Dropped because
  neither candidate copied the real tool well enough: re-diffing with Python's difflib reproduced the real line signs in
  80% of real dev hunks (3,005 of 3,750), and a block rendering in 74%, both under the 95% bar set in advance. Caveat: the re-diff check left out empty context lines, so it compared incomplete hunks; the rate on
  whole hunks is unmeasured.
- **R7, file-view formatting** (the `cat -n` number width and tab). Dropped because every dev file view in both arms
  matched the `cat -n` pattern (553 and 512 views), so no systematic difference was found. Caveat: the pattern accepted
  0 to 5 spaces before the number instead of the fixed six-character field, so unpadded simulated views would have
  matched too; whether the arms differ in padding is unmeasured.
- **The year swap** (2025 to 2026). Dropped because real outputs print 2025 dates too, as dates inside the data (90 dev
  outputs).

Two more counting caveats on rules that were kept, also unmeasured: R2's check of the real gutter accepted any padding
and no space after the sign, so a padding difference between the arms would have looked like the real form; R5's check
counted an output as using the real carriage-return form if any one progress line had one, so outputs mixing both
forms counted as real. Rechecking any of these needs new real-vs-simulated rendering counts, which wait for Aswin's call.

## Left unfixed by decision

- **"Invalid view_range" errors.** The simulator invents this error (150 simulated outputs against 20 real) because its
  idea of a file's length is wrong: a mistake about the world's state, not about rendering. Aswin's decision
  (2026-10-09): the main data keeps these outputs as they are; a secondary analysis drops them from both sides. An
  earlier drop test suggested they alone move the held-out text classifier by about 0.03.

## Not fixable by a general rule

- `undo_edit` outputs (they need the file's state; 61 outputs), the order of error and normal output, numbers and IDs,
  and everything the agent wrote (its messages and its calls).
"""


def kind(rule, tool, args, text):
    """Whether an output is of the kind the rule acts on (used to pick the real output shown beside an example)."""
    if rule in ("R1", "R2"):
        return tool == "text_editor" and args.get("command") == {"R1": "insert", "R2": "str_replace"}[rule]
    if rule == "R5":
        return tool != "text_editor" and fixes.DPKG_MARKER in text
    return any(s["kind"] != "FAILED" for s in clock.stamps(text, clock.LS_YEAR))


def load(run_dir, arms):
    rows = ((line, json.loads(line)) for f in sorted((Path(run_dir) / "inputs").glob("shard_*.jsonl")) for line in f.read_text().splitlines())
    return {(r["arm"], r["key"]): (line, r) for line, r in rows if r["arm"] in arms}


def first_diff_line(a, b):
    la, lb = a.split("\n"), b.split("\n")
    return next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))


def first_kind_line(rule, text):
    pos = (text.find(fixes.DPKG_MARKER) if rule == "R5" else
           min((s["start"] for s in clock.stamps(text, clock.LS_YEAR) if s["kind"] != "FAILED"), default=0))
    return text[:max(pos, 0)].count("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("v2")
    ap.add_argument("original")
    ap.add_argument("--skip", default="")
    ap.add_argument("--draft", default="", help="write the .md here only (no results files), for reading the draws")
    a = ap.parse_args()
    skip = {s for s in a.skip.split(",") if s}
    arms = ("real-attack-e1", "real-attack-e2", "sim-attack")
    v2, og = load(a.v2, arms), load(a.original, arms)
    real_identical = {arm: sum(v2[k][0] == og[k][0] for k in v2 if k[0] == arm) for arm in arms[:2]}
    real_runs = {arm: sum(k[0] == arm for k in v2) for arm in arms[:2]}
    print("real rows byte-identical to the original inputs:", real_identical, "of", real_runs)
    if real_identical != real_runs:
        raise ValueError("a real row differs from the original inputs")

    # every simulated output: (key, msg) -> (environment, step, action, tool, args, before, after)
    sim, n_out = {}, Counter()
    for (arm, key), (_, r) in v2.items():
        if arm != "sim-attack":
            continue
        before = og[(arm, key)][1]["messages"]
        for c in fixes.calls_of(r["messages"]):
            t0, t1 = before[c["msg"]]["content"], r["messages"][c["msg"]]["content"]
            sim[(key, c["msg"])] = (r["environment"], c["step"], c["action"], c["tool"], c["args"], t0 if isinstance(t0, str) else "",
                                    t1 if isinstance(t1, str) else "")
            n_out[r["environment"]] += 1
    recs = [json.loads(l) for f in sorted((Path(a.v2) / "records").glob("records_*.jsonl.gz")) for l in gzip.open(f, "rt")]
    changed_by_rule = defaultdict(set)
    considered, changed = Counter(), Counter()
    for r in recs:
        considered[(r["rule"], r["environment"])] += 1
        if r["changed"]:
            changed[(r["rule"], r["environment"])] += 1
            changed_by_rule[r["rule"]].add((r["key"], int(r["msg"])))
    for (key, msg), (env, *_, t0, t1) in sim.items():  # R5 and R6 records list changed outputs only: count what they consider
        if fixes.DPKG_MARKER in t0 and "\r" not in t0 and sim[(key, msg)][3] != "text_editor":
            considered[("R5x", env)] += 1
        if any(s["kind"] != "FAILED" for s in clock.stamps(t0, clock.LS_YEAR)):
            considered[("R6x", env)] += 1
    for env in ENVS:
        considered[("R5", env)], considered[("R6", env)] = considered.pop(("R5x", env), 0), considered.pop(("R6x", env), 0)
    differ = {k for k, v in sim.items() if v[5] != v[6]}
    union = set().union(*changed_by_rule.values())
    print("simulated outputs that differ, original vs v2:", len(differ), "| changed in the records (any rule):", len(union),
          "| in one and not the other:", len(differ ^ union))
    if differ != union:
        raise ValueError("the records do not match the byte comparison")
    multi = sum(sum(k in v for v in changed_by_rule.values()) > 1 for k in union)
    print("outputs changed by more than one rule:", multi)
    counts = {rule: {env: {"changed": changed[(rule, env)], "considered": considered[(rule, env)], "sim_outputs": n_out[env]} for env in ENVS}
              for rule in RULES}
    for rule in RULES:
        print(rule, "changed", sum(v["changed"] for v in counts[rule].values()), "| per environment:",
              {e: v["changed"] for e, v in counts[rule].items()})

    # real e1 outputs by kind, for the neighbour shown beside each example
    real = []
    for (arm, key), (_, r) in v2.items():
        if arm == "real-attack-e1":
            for c in fixes.calls_of(r["messages"]):
                t = r["messages"][c["msg"]]["content"]
                real.append((key, r["environment"], c["step"], c["action"], c["tool"], c["args"], t if isinstance(t, str) else ""))

    def block(title, key, env, step, action, tool, args, text, start):
        return (f"**{title}**: run `{key}` ({env}), step {step}.{action}; type: {label(tool, args)}\n\n"
                f"```text\n{excerpt(mask(text), start)}\n```\n")

    md = ["# How the simulated runs were fixed\n\n",
          "Four general rules, derived on the 5 dev environments and frozen before anything was measured, were applied once "
          "to every simulated run's tool outputs (RULES.md). The test environments are not a clean holdout: checks run "
          "before the freeze printed counts over them (README, \"Review caveats\"). Real runs were never passed to a rule: by script, all "
          f"{real_runs['real-attack-e1']} real epoch-1 and {real_runs['real-attack-e2']} real epoch-2 runs are byte-identical "
          "to the original reader inputs, so the rules changed 0 real outputs. Examples are drawn with a stated seed; text "
          "is the raw output (the rules act on it), credential-like strings masked, cut to 20 lines; the agent's commands "
          "are never shown; `\\r` marks a carriage return. In all, the rules changed "
          f"{len(differ):,} of {sum(n_out.values()):,} simulated tool outputs ({multi} of them by two rules).\n"]
    meta = {"seed": SEED, "examples": {}, "skipped": {}}
    for rule, desc in RULES.items():
        tot = {k: sum(v[k] for v in counts[rule].values()) for k in ("changed", "considered", "sim_outputs")}
        md.append(f"\n## {rule}\n\n{desc}\n\n| environment | split | outputs changed | outputs the rule considered | all simulated outputs |\n|---|---|---|---|---|\n")
        for env in ENVS:
            v = counts[rule][env]
            md.append(f"| {env} | {'dev' if env in SPLIT['dev'] else 'test'} | {v['changed']:,} | {v['considered']:,} | {v['sim_outputs']:,} |\n")
        md.append(f"| **all 10** | | **{tot['changed']:,}** | {tot['considered']:,} | {tot['sim_outputs']:,} |\n\n"
                  + {"R1": "Considered: every insert output (unchanged ones: the diff could not be rebuilt, or was not a diff).",
                     "R2": "Considered: every str_replace output (unchanged ones: not a diff, e.g. an error message).",
                     "R5": "Considered: non-editor outputs holding `Preparing to unpack` and no carriage return (original text).",
                     "R6": "Considered: outputs holding a timestamp the clock code can parse (original text); runs whose "
                           "\"now\" is not in the run week are not shifted."}[rule] + "\n")
        pool = sorted(changed_by_rule[rule], key=lambda k: oid("sim", *k, rule))
        random.Random(SEED + rule).shuffle(pool)
        picked, skipped = [], []
        for k in pool:
            if len(picked) == N_EX:
                break
            (skipped if oid("sim", *k, rule) in skip else picked).append(k)
        meta["skipped"][rule] = [oid("sim", *k, rule) for k in skipped]
        meta["examples"][rule] = []
        for i, (key, msg) in enumerate(picked, 1):
            env, step, action, tool, args, t0, t1 = sim[(key, msg)]
            start = 0 if rule in ("R1", "R2") else max(first_diff_line(t0, t1) - 2, 0)
            lab = label(tool, args)
            cands = [x for x in real if kind(rule, x[4], x[5], x[6])]
            near = ([x for x in cands if x[1] == env and label(x[4], x[5]) == lab] or [x for x in cands if x[1] == env] or cands)
            near = sorted(near, key=lambda x: oid("real", x[0], x[2], x[3]))
            ex_id = oid("sim", key, msg, rule)
            random.Random(f"{SEED}{rule}:real:{ex_id}").shuffle(near)  # seeded by the example, so a redraw elsewhere keeps it
            ids = [oid("real", x[0], x[2], x[3]) for x in near]
            j = next((j for j, r_id in enumerate(ids) if r_id not in skip), len(ids))
            meta["skipped"][rule] += ids[:j]
            rr = near[j] if j < len(near) else None
            md.append(f"\n### {rule} example {i} (id {ex_id})\n\n"
                      + block("simulated, before", key, env, step, action, tool, args, t0, start) + "\n"
                      + block("simulated, after", key, env, step, action, tool, args, t1, start) + "\n"
                      + (block(f"real, same kind (id {oid('real', rr[0], rr[2], rr[3])})", rr[0], rr[1], rr[2], rr[3], rr[4], rr[5], rr[6],
                               0 if rule in ("R1", "R2") else max(first_kind_line(rule, rr[6]) - 2, 0)) if rr else "_no real output of this kind_\n"))
            meta["examples"][rule].append({"id": ex_id, "key": key, "msg": msg, "step": step, "action": action, "environment": env, "label": lab,
                                           "real": {"id": oid("real", rr[0], rr[2], rr[3]), "key": rr[0], "step": rr[2], "action": rr[3],
                                                    "environment": rr[1], "label": label(rr[4], rr[5])} if rr else None})
        md.append(f"\n{len(pool)} changed outputs in the pool; {len(picked)} drawn; {len(meta['skipped'][rule])} drawn outputs "
                  "(simulated or real) skipped because their content was about the attack itself.\n")
        print(rule, "examples", [e["id"] for e in meta["examples"][rule]], "| skipped", meta["skipped"][rule], "| next in order:",
              [oid("sim", *k, rule) for k in pool[: N_EX + len(skipped) + 4]])
    md.append(DROPPED)
    text = "".join(md)
    if a.draft:
        Path(a.draft).write_text(text)
        return
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    fig = plot(counts, len(differ), HERE / "results" / f"fix_counts_{stamp}.png")
    (HERE / "data").mkdir(exist_ok=True)
    md_path = HERE / "data" / f"fix_docs_{stamp}.md"
    with open(md_path, "x") as f:
        f.write(text)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    meta |= {"counts": counts, "real_rows_identical": real_identical, "sim_outputs_differ": len(differ), "changed_by_two_rules": multi, "run": Path(a.v2).name,
             "original": Path(a.original).name, "git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "md": f"data/{md_path.name}", "md_sha256": hashlib.sha256(text.encode()).hexdigest(),
             "figure": f"results/{fig.name}", "figure_sha256": hashlib.sha256(fig.read_bytes()).hexdigest()}
    with open(HERE / "results" / f"fix_docs_counts_{stamp}.json", "x") as f:
        json.dump(meta, f, indent=1)
    print(md_path, meta["md_sha256"], fig)


def plot(counts, n_changed, path):
    """One stacked bar per rule: simulated outputs changed, dev and test environments."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    names = {"R1": "Editor insert (R1)", "R2": "Editor replace (R2)", "R5": "Package install (R5)", "R6": "Clock (R6)"}
    rules = sorted(counts, key=lambda r: sum(v["changed"] for v in counts[r].values()))
    dev = [sum(counts[r][e]["changed"] for e in SPLIT["dev"]) for r in rules]
    test = [sum(counts[r][e]["changed"] for e in SPLIT["test"]) for r in rules]
    top = max(d + t for d, t in zip(dev, test))
    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    ink, muted = "#0b0b0b", "#52514e"
    ax.barh(range(len(rules)), dev, height=0.5, color="#2a78d6", label="dev environments", edgecolor="white", linewidth=2)
    ax.barh(range(len(rules)), test, left=dev, height=0.5, color="#eb6834", label="test environments", edgecolor="white", linewidth=2)
    for i, (d, t) in enumerate(zip(dev, test)):
        ax.text(d + t + top * 0.01, i, f"{d + t:,}", va="center", ha="left", fontsize=13, fontweight="bold", color=ink)
    ax.set_yticks(range(len(rules)), [names[r] for r in rules], fontsize=11, color=ink)
    ax.set_xlim(0, top * 1.15)
    ax.set_xlabel("simulated tool outputs changed (count, all 10 environments)", fontsize=10, color=muted)
    fig.suptitle(f"{names[rules[-1]].split(' (')[0]} outputs got most of the fixes:\n{dev[-1] + test[-1]:,} of the {n_changed:,} "
                 "simulated outputs the rules changed", fontsize=12, color=ink, x=0.02, ha="left")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(muted)
    ax.tick_params(colors=muted)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    with open(path, "xb") as f:
        fig.savefig(f, format="png")
    return path


if __name__ == "__main__":
    main()
