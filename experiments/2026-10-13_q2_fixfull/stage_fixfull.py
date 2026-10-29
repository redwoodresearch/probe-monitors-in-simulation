"""POST HOC (PLAN.md): stage the per-fix variants F1-F8 and fix-full of the simulated runs, with the gates, change counts,
F8's error templates (from REAL errored outputs only) and the clock survey. Modal CPU. Writes no message text locally:
counts, hashes and the staged inputs on the Volume only (never overwritten).

  --mode survey: every count, gate and template, nothing written to the Volume.
  --mode stage:  the same, then the staged inputs `2026-10-13_q2_fixfull_inputs_v1/inputs/shard_<arm>.jsonl` (only runs a
                 variant changes; every other field of the staged row as in the source), and a per-run record.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run \\
      experiments/2026-10-13_q2_fixfull/stage_fixfull.py --mode survey
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
APP = "probe-shift_aswin-surya_sub-q2-fixfull_stage_kill-2026-11-13"
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

    # --- gates: F1 then F2 and F3 then F4 compose to fix939's edits; fix939's order reproduces its staged runs
    fx = load(FIX[0], ("sim-attack-note-sr-both",))
    fpath = Path("/acts") / FIX[0] / "inputs" / "shard_0.jsonl"
    if hashlib.sha256(fpath.read_bytes()).hexdigest() != FIX[1]:
        raise ValueError("fix939's staged inputs are not the pinned ones")
    for k, it in sims.items():
        m = it["messages"]
        a = rr.rerender_messages(m, "diff_with_note")[0]
        rendered = {r["msg"] for r in rr.rerender_messages(m, "diff_no_newline")[1] if r["category"] in rr.RENDERED}
        b = [dict(x) | ({"content": x["content"].rstrip("\n")} if i in rendered else {}) for i, x in enumerate(f1(m))]
        c["gate|F1 then F2's strip (eligibility from the original run) equals diff_with_note (runs)"] += msha(a) == msha(b)
        g = st.swap_messages(m, "sr_gutter_no_newline")[0]
        h = st.swap_messages(f3(m), "sr_no_newline")[0]
        c["gate|F3 then F4 equals sr_gutter_no_newline (runs)"] += msha(g) == msha(h)
        both = st.swap_messages(a, "sr_gutter_no_newline")[0]
        if ("sim-attack-note-sr-both", k) in fx:
            c["gate|fix939 runs"] += 1
            c["gate|fix939 runs reproduced"] += msha(both) == msha(fx[("sim-attack-note-sr-both", k)]["messages"])
        else:
            c["gate|fix939-unchanged runs"] += 1
            c["gate|fix939-unchanged runs left unchanged"] += msha(both) == msha(m)

    if not (c["gate|F1 then F2's strip (eligibility from the original run) equals diff_with_note (runs)"] == len(sims)
            and c["gate|F3 then F4 equals sr_gutter_no_newline (runs)"] == len(sims) and c["gate|fix939 runs"] == 823
            and c["gate|fix939 runs reproduced"] == 823 and c["gate|fix939-unchanged runs"] == 116 and c["gate|fix939-unchanged runs left unchanged"] == 116):
        raise ValueError("a composition or fix939 gate failed: nothing is staged")

    # --- variants: change counts, the untargeted-unchanged check, staged rows
    rows, record = defaultdict(list), []
    nl_outputs = {}  # per NL run, its tool outputs (step, action, family, changed): for the mechanism split (no text)
    for group, runs_, vs in (("attack", sims, VARIANTS_), ("honest", hon, COMPLETE_)):
        for k, it in sorted(runs_.items()):
            m = it["messages"]
            cell = it["cell"] if group == "attack" else f"{it['environment']}:{it['main_task']}"  # honest: PLAN's seed id (their cell is the main task)
            calls_ = fixes.calls_of(m)
            fam = {x["msg"]: family(x) for x in calls_}
            ren = {r["msg"] for r in rr.rerender_messages(m, "diff_no_newline")[1] if r["category"] in rr.RENDERED}
            tools = {x["msg"] for x in calls_}
            elig = {"F1": ren, "F2": ren, "F3": {i for i in tools if fam[i] == "text_editor:str_replace"}}
            elig["F4"] = elig["F3"]
            elig["F5"] = {i for i in tools if fam[i] == "text_editor:create" and not is_err(m[i]["content"])}
            elig["F6"] = {i for i in tools if not fam[i].startswith("text_editor")}
            elig["F7"] = tools
            elig["F8"] = {i for i in tools if is_err(m[i]["content"])}
            elig["NL"] = elig["F2"] | elig["F4"] | elig["F5"] | elig["F8"]
            elig["C"] = elig["F1"] | elig["F3"]
            elig["full"] = tools
            elig["F9"] = {i for i in tools if fam[i] == "text_editor:view" and isinstance(m[i]["content"], str) and not is_err(m[i]["content"])
                          and (group != "attack" or ("sim-attack", k, next(x for x in calls_ if x["msg"] == i)["step"], next(x for x in calls_ if x["msg"] == i)["action"]) not in err)}
            elig["fullF9"] = tools
            for i in elig["F8"]:
                f8_report[tid[norm_first(m[i]["content"])]]["simulated matches"][group] += 1
            for v in vs:
                new = variant(v, m, cell, k, group)
                if len(new) != len(m):
                    raise ValueError("a variant changed the number of messages")
                changed = [i for i, (x, y) in enumerate(zip(m, new)) if x != y]
                if any(m[i]["role"] != "tool" or i not in elig[v] or set(m[i]) != set(new[i])
                       or any(m[i][f_] != new[i][f_] for f_ in m[i] if f_ != "content") for i in changed):
                    raise ValueError(f"{v}: a non-tool or ineligible output changed, or a field other than content")
                if v == "F8":
                    for i in changed:
                        f8_report[tid[norm_first(m[i]["content"])]]["simulated changes"][group] += 1
                arm = f"{'sim-attack' if group == 'attack' else 'sim-honest'}-fx-{v}"
                c[f"{group}|{v}|runs changed"] += bool(changed)
                c[f"{group}|{v}|outputs changed"] += len(changed)
                for i in changed:
                    c[f"{group}|{v}|outputs changed|{fam.get(i, 'unpaired')}"] += 1
                if v == "NL":
                    nl_outputs[f"{group}|{k}"] = [[x["step"], x["action"], fam[x["msg"]], x["msg"] in changed] for x in calls_]
                if changed or v in COMPLETE_:  # complete arms (all 939 attack, all 437 honest) for d9's readers; per-fix arms changed-only
                    rows[arm].append(it | {"arm": arm, "messages": new})
                    record.append({"arm": arm, "key": k, "changed_messages": changed, "messages_sha256": msha(new),
                                   "first_changed_message": changed[0] if changed else None})

    # --- the clock survey: LS stamps' offset from the run's own "now" (b7's internal_ls: "now" from the run's OTHER
    # outputs). clock.now_of's logic over each output's stamps, parsed once (now_from reproduces now_of exactly).
    def now_from(stamp_lists):
        found = [s for sl in stamp_lists for s in sl if s["kind"] != "FAILED" and clock.in_run_week(s["utc"])]
        dts = [s["utc"] for s in found if s["kind"] in ("DATE_CMD", "ISO", "HTTP")] or [s["utc"] for s in found if s["kind"] == "LS"]
        if not dts:
            return None
        hour = Counter(d.replace(minute=0, second=0, microsecond=0) for d in dts).most_common(1)[0][0]
        inside = sorted(d for d in dts if d.replace(minute=0, second=0, microsecond=0) == hour)
        return inside[len(inside) // 2]

    def ls_offsets(msgs, arm):
        sts = [clock.stamps(o, clock.LS_YEAR) for o in outputs_of(msgs)]
        if now_from(sts) != clock.now_of(outputs_of(msgs)):
            raise ValueError("now_from differs from clock.now_of")
        for j, sl in enumerate(sts):
            now = now_from(sts[:j] + sts[j + 1:])
            if now is None:
                continue
            for s in sl:
                if s["kind"] == "LS":
                    mins = abs((s["utc"] - now).total_seconds()) / 60
                    c[f"clock|{arm}|LS stamps"] += 1
                    c[f"clock|{arm}|LS within 10 min of now"] += mins <= 10
                    c[f"clock|{arm}|LS within 48 h of now"] += mins <= 2880
    cells = {it["cell"] for it in sims.values()}
    for (a, k), it in base.items():
        if a == "real-attack-e2" and it["cell"] in cells:
            ls_offsets(it["messages"], "real-attack-e2")
    for k, it in sims.items():
        ls_offsets(it["messages"], "sim-attack as-is")
        ls_offsets(f7(it["messages"], it["cell"], it["messages"])[0], "sim-attack after R6")

    out = {"counts": dict(sorted(c.items())), "templates": {"n": len(templates), "rules": dict(Counter(templates.values())),
                                                         "by template (sha256[:12] of the normalized first line)": {k: v | {"simulated matches": dict(v["simulated matches"]), "simulated changes": dict(v["simulated changes"])} for k, v in f8_report.items()}},
           "runs by arm": {arm: sorted(x["key"] for x in rs) for arm, rs in sorted(rows.items())}, "NL outputs (step, action, family, changed)": nl_outputs}
    if mode == "stage":
        dst = Path("/acts") / DST_
        if dst.exists():
            raise FileExistsError(f"{DST_} exists: a staged run is never overwritten")
        (dst / "inputs").mkdir(parents=True)
        shas = {}
        for n, (arm, rs) in enumerate(sorted(rows.items())):
            body = "".join(json.dumps(x) + "\n" for x in sorted(rs, key=lambda x: x["key"])).encode()
            (dst / "inputs" / f"shard_{n}.jsonl").write_bytes(body)
            shas[f"inputs/shard_{n}.jsonl"] = {"arm": arm, "runs": len(rs), "sha256": hashlib.sha256(body).hexdigest()}
        rec = json.dumps({"runs": record, "shards": shas, "f8_templates_sha256": hashlib.sha256(json.dumps(sorted(templates.items())).encode()).hexdigest()}).encode()
        (dst / "stage_record.json").write_bytes(rec)
        acts.commit()
        out |= {"staged": DST_, "shards": shas, "record_sha256": hashlib.sha256(rec).hexdigest()}
    return out


@app.function(image=image, volumes={"/acts": acts}, cpu=8, memory=65536, timeout=3600, retries=0)
def job(mode: str, cohort: dict, honest: list, errored: list, dev: list, set_: str = "fx") -> dict:
    try:
        return run_job(mode, cohort, honest, errored, dev, set_)
    except Exception as e:  # noqa: BLE001 — sanitized (no staged text crosses the boundary) and re-raised
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"stage failed: {type(e).__name__}: {str(e)[:200]} at {Path(tb.filename).name}:{tb.lineno}") from None


@app.local_entrypoint()
def main(mode: str = "survey", set: str = "fx"):
    if mode not in ("survey", "stage") or set not in SETS:
        raise SystemExit("--mode survey|stage --set fx|f9")
    for f in (Path(__file__), HERE / "PLAN.md"):
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
               if r["error"] is not None and r["arm"] in ("real-attack-e1", "real-attack-e2", "sim-attack")]  # sim-attack: F9's flags only
    dev = json.loads((FSIM / "split.json").read_text())["dev"]
    out = job.remote(mode, cohort, honest, errored, dev, set)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    p = HERE / "results" / f"{mode}{'' if set == 'fx' else '_' + set}_{stamp}.json"
    p.write_text(json.dumps({"POST HOC": True, "git": head, "mode": mode, "inputs": {"#36 runs.json": hashlib.sha256(RUNS36.read_bytes()).hexdigest(),
                             "honest runs.json": hashlib.sha256(HONEST_RUNS.read_bytes()).hexdigest(),
                             "tool errors": hashlib.sha256(ERRORS.read_bytes()).hexdigest()}} | out, indent=1) + "\n")
    print(p)
    for k, v in out["counts"].items():
        print(f"  {k}: {v}")
