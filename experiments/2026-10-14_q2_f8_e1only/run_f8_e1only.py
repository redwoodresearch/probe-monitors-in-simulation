"""POST HOC (RULE.md here): re-derive F8's error templates from real e1 errored outputs only and count what changes against
the as-run e1 + e2 derivation (Codex on #1654). Lines 49-228 of experiments/2026-10-13_q2_fixfull/stage_fixfull.py (as
packaged; last changed at 43ce3ec7) are copied verbatim below; only the app name differs above them. Gates first: with the
as-run templates, every staged arm is rebuilt and must equal the as-run stage records by messages_sha256. Modal CPU,
read-only on the inputs; the job writes its result (gates included) to q2_f8_e1only_results/ on the Volume; counts and\ntemplate hashes only, no output text leaves the container.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_q2_f8_e1only/run_f8_e1only.py
"""
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
SWAP = REPO / "experiments" / "2026-10-07_q2_rendering_swap"
FSIM = REPO / "experiments" / "2026-10-09_fixed_sim"
APP = "probe-shift_aswin-surya_sub-q2-f8e1only_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
BASE = "2026-10-06_la_segment-end_qwen3-32b_v1"  # #36's staged inputs: real-attack-e1/e2, sim-attack
HONEST = ("2026-10-06_la_segment-end-v4_qwen3-32b_v1", "sim-honest")  # q3's edited-honest source
FIX = ("2026-10-11_q3_fix939_inputs_v1", "57cfbfbf1e4571690769605d2093d73edc93b958e06fed5e8efe8fa88587530d")
DST = "2026-10-13_q2_fixfull_inputs_v2"  # v1 (14:16) lacked N complete, honest N and C; never read (Amendment 2)
VARIANTS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "NL", "C", "full")  # NL (= 0d's "N"): F2+F4+F5+F8; C: F1+F3 (the content fixes)
COMPLETE = ("NL", "full")  # staged for every run (unchanged included), attack and honest, for d9's training cells
SETS = {"fx": (VARIANTS, COMPLETE, DST), "f9": (("F9", "fullF9"), ("fullF9",), "2026-10-13_q2_fixfull_inputs_f9_v1")}  # f9: Amendment 3
RUNS36 = REPO / "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
HONEST_RUNS = REPO / "experiments/2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json"
ERRORS = REPO / "experiments/2026-10-06_q1_segment_end/results/tool_errors_2026-10-06_134908.jsonl"

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = (modal.Image.debian_slim(python_version="3.12").add_local_python_source("probe_shift")
         .add_local_file(SWAP / "rerender.py", "/root/swap/rerender.py")
         .add_local_file(SWAP / "str_replace_tells.py", "/root/swap/str_replace_tells.py")
         .add_local_file(FSIM / "fixes.py", "/root/fsim/fixes.py")
         .add_local_file(FSIM / "clock.py", "/root/fsim/clock.py"))

BT = re.compile(r"`[^`]*`")


def norm_first(text: str) -> str:
    """F8's template key: the first line, backtick spans and every token holding a digit or '/' replaced by <v>."""
    first = BT.sub("<v>", text.split("\n", 1)[0])
    return " ".join("<v>" if any(ch.isdigit() or ch == "/" for ch in t) else t for t in first.split(" "))


def trail(text: str) -> str:
    n = len(text) - len(text.rstrip("\n"))
    return "none" if n == 0 else "one" if n == 1 else "more"


def family(c: dict) -> str:
    return f"text_editor:{c['args'].get('command')}" if c["tool"] == "text_editor" else c["tool"]


def _init():
    sys.path[:0] = ["/root/swap", "/root/fsim"]


def run_job(mode: str, cohort: dict, honest: list, errored: list, dev: list, set_: str = "fx") -> dict:
    _init()
    import clock
    import fixes
    import rerender as rr
    import str_replace_tells as st

    def load(run_dir, arms, keys=None):
        out = {}
        for p in sorted((Path("/acts") / run_dir / "inputs").glob("shard_*.jsonl")):
            for line in p.read_text().splitlines():
                it = json.loads(line)
                if it["arm"] in arms and (keys is None or it["key"] in keys):
                    if (it["arm"], it["key"]) in out:
                        raise ValueError("a run appears twice in the source inputs")
                    out[(it["arm"], it["key"])] = it
        return out

    sim_keys = set(cohort["sim"])
    base = load(BASE, ("real-attack-e1", "real-attack-e2", "sim-attack"))
    sims = {k: v for (a, k), v in base.items() if a == "sim-attack" and k in sim_keys}
    if len(sims) != len(sim_keys):
        raise ValueError(f"{len(sims)} of {len(sim_keys)} simulated attack runs found")
    hon = {k: v for (a, k), v in load(HONEST[0], (HONEST[1],), set(honest)).items()}
    if len(hon) != len(honest):
        raise ValueError(f"{len(hon)} of {len(honest)} simulated honest runs found")
    c = Counter()

    # --- F8's templates, from REAL errored outputs only (attack cohorts' Inspect error flags)
    VARIANTS_, COMPLETE_, DST_ = SETS[set_]
    err = {(a, k, s, j): t for a, k, s, j, t in errored}
    real_err, tool_mismatch = defaultdict(Counter), 0
    for (a, k), it in base.items():
        if a == "sim-attack":
            continue
        for x in fixes.calls_of(it["messages"]):
            t = err.get((a, k, x["step"], x["action"]))
            if t is None:
                continue
            tool_mismatch += x["tool"] != t
            text = it["messages"][x["msg"]]["content"]
            if isinstance(text, str):
                real_err[norm_first(text)][trail(text)] += 1
    n_err = sum(sum(v.values()) for v in real_err.values())
    templates = {}
    for key, v in real_err.items():
        n = sum(v.values())
        if n < 5:
            continue
        templates[key] = "strip" if v["none"] >= 0.9 * n else "one" if v["one"] >= 0.9 * n else "keep"
    c["F8|real errored outputs"] = n_err
    c["F8|real errored outputs whose call tool differs from the error file's"] = tool_mismatch
    c["F8|templates"] = len(templates)
    c["F8|real errored outputs covered by templates"] = sum(sum(real_err[k].values()) for k in templates)
    for t in ("strip", "one", "keep"):
        c[f"F8|templates with rule {t}"] = sum(v == t for v in templates.values())
    tid = {k: hashlib.sha256(k.encode()).hexdigest()[:12] for k in templates}  # templates by hash only (no text out)
    f8_report = {tid[k]: {"real errored outputs": sum(real_err[k].values()), "real trailing newlines (none, one, more)":
                          [real_err[k]["none"], real_err[k]["one"], real_err[k]["more"]], "rule": templates[k],
                          "simulated matches": Counter(), "simulated changes": Counter()} for k in templates}

    def is_err(text):
        return isinstance(text, str) and norm_first(text) in templates

    # --- donors for R6 (dev real epoch-1 runs), as run_fixed_sim_modal.donors_job
    def outputs_of(msgs):
        return [m["content"] for m in msgs if m["role"] == "tool" and isinstance(m["content"], str)]
    cohort_cells = {it["cell"] for it in sims.values()}  # donors_job's pool: dev real e1 runs of the cohort's cells
    donors = {it["cell"]: n for (a, k), it in base.items() if a == "real-attack-e1" and it["environment"] in set(dev) and it["cell"] in cohort_cells
              if (n := clock.now_of(outputs_of(it["messages"]))) is not None}
    c["F7|donors"] = len(donors)

    # --- the fixes on one transcript
    def f1(msgs):
        new, recs = rr.rerender_messages(msgs, "diff_no_newline")  # only to find the inserts fix939 renders
        out = [dict(m) for m in msgs]
        for r in recs:
            if r["category"] in rr.RENDERED:
                o = msgs[r["msg"]]["content"]
                out[r["msg"]]["content"] = o.rstrip("\n") + "\n" + rr.REVIEW + o[len(o.rstrip("\n")):]
        return out

    def f2(msgs):
        return rr.rerender_messages(msgs, "diff_no_newline")[0]

    def f3(msgs):
        return st.swap_messages(msgs, "sr_gutter")[0]

    def f4(msgs):
        return st.swap_messages(msgs, "sr_no_newline")[0]

    def f5(msgs):
        out = [dict(m) for m in msgs]
        for x in fixes.calls_of(msgs):
            o = msgs[x["msg"]]["content"]
            if x["tool"] == "text_editor" and x["args"].get("command") == "create" and isinstance(o, str) and not is_err(o):
                out[x["msg"]]["content"] = o.rstrip("\n")
        return out

    def f6(msgs):
        return fixes.r5_dpkg(msgs)[0]

    def f7(msgs, cell, orig):
        now = clock.now_of(outputs_of(orig))  # from the uncorrected outputs, as the apply step
        out = [dict(m) for m in msgs]
        if now is None:
            return out, None
        _, donor_now = clock.donor_for(cell, donors)
        off = donor_now - now
        for i, m in enumerate(out):
            if m["role"] == "tool" and isinstance(m["content"], str):
                out[i] = m | {"content": clock.shift_text(m["content"], off, now)[0]}
        return out, round(off.total_seconds())

    def f8(msgs):
        out = [dict(m) for m in msgs]
        for x in fixes.calls_of(msgs):
            o = msgs[x["msg"]]["content"]
            if is_err(o):
                rule = templates[norm_first(o)]
                if rule == "strip":
                    out[x["msg"]]["content"] = o.rstrip("\n")
                elif rule == "one":
                    out[x["msg"]]["content"] = o.rstrip("\n") + "\n"
        return out

    def f9(msgs, key, group):
        """F9 (Amendment 3): every view output without Inspect's error field (attack: the tool-error file's flags; honest:
        none set, G3 fork's census) and not error-like (F8's templates) ends in exactly one trailing newline."""
        out = [dict(m) for m in msgs]
        for x in fixes.calls_of(msgs):
            o = msgs[x["msg"]]["content"]
            if x["tool"] == "text_editor" and x["args"].get("command") == "view" and isinstance(o, str) and not is_err(o) \
                    and (group != "attack" or ("sim-attack", key, x["step"], x["action"]) not in err):
                out[x["msg"]]["content"] = o.rstrip("\n") + "\n"
        return out

    def variant(v, msgs, cell, key=None, group=None):
        if v == "F9":
            return f9(msgs, key, group)
        if v == "fullF9":
            return f9(variant("full", msgs, cell), key, group)
        if v == "F7":
            return f7(msgs, cell, msgs)[0]
        if v == "NL":
            return f8(f5(f4(f2(msgs))))
        if v == "C":
            return f3(f1(msgs))
        if v != "full":
            return {"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "F6": f6, "F8": f8}[v](msgs)
        m = rr.rerender_messages(msgs, "diff_with_note")[0]
        m = st.swap_messages(m, "sr_gutter_no_newline")[0]
        m = f6(f5(m))
        m = f7(m, cell, msgs)[0]
        return f8(m)

    def msha(msgs):
        return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()
    # --- RULE.md (1): rebuild every staged arm with the as-run templates; each run must equal its as-run stage record
    recs = {}
    for d in (DST, SETS["f9"][2]):
        for r in json.loads((Path("/acts") / d / "stage_record.json").read_text())["runs"]:
            recs[(r["arm"], r["key"])] = r["messages_sha256"]
    plan = (("attack", sims, VARIANTS + ("F9", "fullF9")), ("honest", hon, COMPLETE + ("fullF9",)))
    complete = set(COMPLETE) | {"fullF9"}

    def hashes(msgs):
        return [hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest() for x in msgs]

    def build():
        out = {}
        for group, runs_, vs in plan:
            for k, it in sorted(runs_.items()):
                m = it["messages"]
                cell = it["cell"] if group == "attack" else f"{it['environment']}:{it['main_task']}"  # as the as-run loop
                for v in vs:
                    new = variant(v, m, cell, k, group)
                    out[(group, v, k)] = (msha(new), hashes(new), msha(new) != msha(m))
        return out

    T_asrun = dict(templates)
    A = build()
    for (group, v, k), (h, _, changed) in A.items():
        arm = f"{'sim-attack' if group == 'attack' else 'sim-honest'}-fx-{v}"
        c[f"gate|{group}|{v}|runs"] += 1
        if (arm, k) in recs:
            c[f"gate|{group}|{v}|staged runs reproduced"] += recs[(arm, k)] == h
        else:
            c[f"gate|{group}|{v}|unstaged runs"] += 1
            c[f"gate|{group}|{v}|unstaged runs unchanged (changed-only arm)"] += (not changed) and v not in complete
    staged = Counter(a for a, _ in recs)
    gates = {"F8 templates == 14": len(T_asrun) == 14, "F7 donors == 376": c["F7|donors"] == 376}
    for group, runs_, vs in plan:
        src = "sim-attack" if group == "attack" else "sim-honest"
        for v in vs:
            n_st = staged[f"{src}-fx-{v}"]
            gates[f"{group} {v}: {n_st} staged runs reproduced, the rest unchanged"] = (
                c[f"gate|{group}|{v}|staged runs reproduced"] == n_st
                and c[f"gate|{group}|{v}|unstaged runs unchanged (changed-only arm)"] == c[f"gate|{group}|{v}|unstaged runs"]
                and c[f"gate|{group}|{v}|runs"] == len(runs_))
    if not all(gates.values()):
        return {"gates": gates, "gates pass": False, "counts": {k: v for k, v in c.items() if k.startswith("gate|")}}

    # --- RULE.md (2): the e1-only templates, the as-run rule exactly, from real-attack-e1 errored outputs only
    real_err_e1 = defaultdict(Counter)
    for (a, k), it in base.items():
        if a != "real-attack-e1":
            continue
        for x in fixes.calls_of(it["messages"]):
            if err.get((a, k, x["step"], x["action"])) is None:
                continue
            text = it["messages"][x["msg"]]["content"]
            if isinstance(text, str):
                real_err_e1[norm_first(text)][trail(text)] += 1
    T_e1 = {}
    for key, v in real_err_e1.items():
        n = sum(v.values())
        if n < 5:
            continue
        T_e1[key] = "strip" if v["none"] >= 0.9 * n else "one" if v["one"] >= 0.9 * n else "keep"

    # --- RULE.md (3): compare templates, then rebuild every arm with the e1-only templates
    h12 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]  # noqa: E731
    tcmp = {"as run": len(T_asrun), "e1 only": len(T_e1),
            "only in the as-run set": sorted(h12(x) for x in set(T_asrun) - set(T_e1)),
            "only in the e1-only set": sorted(h12(x) for x in set(T_e1) - set(T_asrun)),
            "in both, rule differs": sorted(h12(x) for x in set(T_asrun) & set(T_e1) if T_asrun[x] != T_e1[x]),
            "by template (sha256[:12])": {h12(x): {"rule as run": T_asrun.get(x), "rule e1 only": T_e1.get(x),
                                                   "real e1 errored outputs": sum(real_err_e1[x].values()),
                                                   "real e1 + e2 errored outputs": sum(real_err[x].values())}
                                          for x in sorted(set(T_asrun) | set(T_e1))}}
    templates = T_e1  # the fixes' closures read this name
    B = build()
    templates = T_asrun
    arms = {}
    for (group, v, k), (h, hs, changed) in A.items():
        h2, hs2, changed2 = B[(group, v, k)]
        r = arms.setdefault(f"{group}|{v}", Counter())
        r["runs"] += 1
        r["runs whose messages differ"] += h != h2
        r["tool outputs whose content differs"] += sum(x != y for x, y in zip(hs, hs2)) + abs(len(hs) - len(hs2))
        r["runs whose changed status flips"] += changed != changed2
    total = {"templates differ": len(tcmp["only in the as-run set"]) + len(tcmp["only in the e1-only set"]),
             "rules differ": len(tcmp["in both, rule differs"]),
             "runs whose messages differ, all arms": sum(r["runs whose messages differ"] for r in arms.values()),
             "tool outputs whose content differs, all arms": sum(r["tool outputs whose content differs"] for r in arms.values())}
    return {"gates": gates, "gates pass": True, "templates": tcmp, "per arm": {a: dict(r) for a, r in sorted(arms.items())},
            "total changes": total, "any change": any(total.values()),
            "counts": dict(sorted(c.items()))}


OUT_DIR = "q2_f8_e1only_results"  # on the Volume: the job writes its own result, so nothing depends on the laptop


@app.function(image=image, volumes={"/acts": acts}, cpu=8, memory=65536, timeout=750, retries=0)  # ~10 min expected + 25%
def job(cohort: dict, honest: list, errored: list, dev: list, head: str) -> dict:
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    try:
        out = run_job("e1only", cohort, honest, errored, dev, "fx")
    except Exception as e:  # noqa: BLE001 — sanitized (no staged text crosses the boundary) and re-raised
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        out = {"gates pass": False, "error": f"{type(e).__name__}: {str(e)[:200]} at {Path(tb.filename).name}:{tb.lineno}"}
    rec = {"POST HOC": True, "rule": "RULE.md", "git": head, "stamp": stamp} | out
    d = Path("/acts") / OUT_DIR
    d.mkdir(exist_ok=True)
    with open(d / f"f8_e1only_POSTHOC_{stamp}.json", "x") as fh:  # never overwritten
        fh.write(json.dumps(rec, indent=1) + "\n")
    acts.commit()
    if "error" in out:
        raise RuntimeError(f"e1only failed: {out['error']}")
    return rec


@app.local_entrypoint()
def main():
    for f in (Path(__file__), HERE / "RULE.md"):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
                or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    runs = json.loads(RUNS36.read_text())
    cohort = {"sim": sorted(r["key"] for r in runs if r["arm"] == "sim-attack")}
    if len(cohort["sim"]) != 939:
        raise SystemExit(f"{len(cohort['sim'])} simulated attack runs in #36's cohort, not 939")
    honest = sorted(r["key"] for r in json.loads(HONEST_RUNS.read_text()) if r["arm"] == "sim-honest")
    if len(honest) != 437:
        raise SystemExit(f"{len(honest)} simulated honest runs, not 437")
    errored = [[r["arm"], r["key"], r["step"], r["action"], r["tool"]] for r in map(json.loads, ERRORS.read_text().splitlines())
               if r["error"] is not None and r["arm"] in ("real-attack-e1", "real-attack-e2", "sim-attack")]  # as stage_fixfull.py
    dev = json.loads((FSIM / "split.json").read_text())["dev"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out = job.remote(cohort, honest, errored, dev, head)  # also written to the Volume by the job (run with --detach)
    p = HERE / "results" / f"f8_e1only_POSTHOC_{out['stamp']}.json"
    p.write_text(json.dumps(out | {"inputs": {"#36 runs.json": hashlib.sha256(RUNS36.read_bytes()).hexdigest(),
                                              "honest runs.json": hashlib.sha256(HONEST_RUNS.read_bytes()).hexdigest(),
                                              "tool errors": hashlib.sha256(ERRORS.read_bytes()).hexdigest()}}, indent=1) + "\n")
    print(p)
    print("gates pass:", out["gates pass"])
    if out["gates pass"]:
        print("any change:", out["any change"], out["total changes"])
