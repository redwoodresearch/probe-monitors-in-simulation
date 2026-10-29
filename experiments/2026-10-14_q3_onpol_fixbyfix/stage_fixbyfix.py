"""Fix-by-fix on Qwen3.5-27B's own runs (RULE.md here), POST HOC: each of q2's fixes F1-F9 applied ALONE to Qwen3.5's own
simulated runs (191 attack, 19 honest), mirroring q2's per-fix ablation (experiments/2026-10-13_q2_fixfull/PLAN.md, aswin/q2-f10-tabs
@ 43ce3ec7). A copy of ../2026-10-14_q3_onpolicy_titlecells/stage_repair_onpol.py (be3539ed): the fixers are q2's
stage_fixfull.py closures, verbatim (F1-F9), with q2's single-fix dispatch (stage_fixfull.variant: F7 = f7(msgs, cell, msgs)[0],
F9 = f9(msgs, key, group), the rest one closure each). Changed from the copy: the app name, the output (one staged run holding
every single-fix arm, CHANGED RUNS ONLY, as q2's per-fix arms), the gate (q2's composition gates as before, plus a per-fix
byte-exact gate against q2's own single-fix Sol arms), and the loop over F1-F9. "All together" is the on-policy job's existing
full / full + F9 copies (not restaged here).
Gate (before anything is written):
  - q2's composition gates on #36's 939 simulated attack runs (as the copy);
  - per fix F1-F9: on seeded samples of #36's simulated attack runs that q2 staged in its single-fix arm (sim-attack-fx-<F>,
    2026-10-13_q2_fixfull_inputs_v2; F9 in 2026-10-13_q2_fixfull_inputs_f9_v1), this copy's fix reproduces q2's staged
    messages BYTE-EXACT; and on seeded samples q2 did NOT stage for that fix, it leaves the run unchanged.
Qwen3.5 (F9's error flags from the .eval samples, F7's donors and F8's templates as in the copy): per fix and run, the fixed
messages; only tool-output contents may change (else stop). Changed runs only are staged, arms q35-sim-attack-fx-<F> and
q35-sim-honest-fx-<F>, in one staged run (2026-10-14_q3onpol_q35own_fx_inputs_v1) with the Qwen3-32B position metas Seed's
keep flags need (2026-10-14_q3onpol_q35own_fx_qmeta_v1). A fix that changes no run gets no arm (counted). Counts and hashes
only leave the container; no text.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-14_q3_onpol_fixbyfix/stage_fixbyfix.py
"""
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")  # in the container only the mounted files are read
APP = "probe-shift_aswin-surya_sub-q3onpol_fxstage_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
EVALS = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
HF = "probe-shift_aswin-surya_hf-cache"
STAGED = "2026-10-14_q3onpol_q35own_inputs_v1"  # run_onpol_stage.py's staged run (scrub level 0)
OUT = ("2026-10-14_q3onpol_q35own_fx_inputs_v1", "2026-10-14_q3onpol_q35own_fx_qmeta_v1")
FIXES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9")  # q2's single fixes, its order
BASE = "2026-10-06_la_segment-end_qwen3-32b_v1"  # #36's staged inputs: real-attack-e1/e2, sim-attack
HONEST = ("2026-10-06_la_segment-end-v4_qwen3-32b_v1", "sim-honest")
FIX = ("2026-10-11_q3_fix939_inputs_v1", "57cfbfbf1e4571690769605d2093d73edc93b958e06fed5e8efe8fa88587530d")
Q2_FX, Q2_F9 = "2026-10-13_q2_fixfull_inputs_v2", "2026-10-13_q2_fixfull_inputs_f9_v1"
RUNS36 = "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
HONEST_RUNS = "experiments/2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json"
ERRORS = "experiments/2026-10-06_q1_segment_end/results/tool_errors_2026-10-06_134908.jsonl"
SPLIT = "experiments/2026-10-09_fixed_sim/split.json"
SWAP, FSIM = "experiments/2026-10-07_q2_rendering_swap", "experiments/2026-10-09_fixed_sim"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"  # Qwen3.5 tasks' environments
N_GATE, N_NEG, SEED, CUT = 6, 3, 20261014, 32768

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
evals = modal.Volume.from_name(EVALS, version=2).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==4.56.2", "jinja2", "zstandard==0.25.0", "numpy", "boto3", "pyyaml")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift")
         .add_local_file(REPO / SWAP / "rerender.py", "/root/swap/rerender.py")
         .add_local_file(REPO / SWAP / "str_replace_tells.py", "/root/swap/str_replace_tells.py")
         .add_local_file(REPO / FSIM / "fixes.py", "/root/fsim/fixes.py")
         .add_local_file(REPO / FSIM / "clock.py", "/root/fsim/clock.py")
         .add_local_file(HERE / "evalzip.py", "/root/q4/evalzip.py")
         .add_local_file(REPO / T196, "/root/t196.json"))

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


def msha(msgs):
    return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()


def filename(key: str) -> str:
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def fixers(templates, donors, err, rr, st, fixes, clock):
    """q2's stage_fixfull.run_job closures, verbatim apart from their inputs (templates, donors, err) being arguments."""
    def is_err(text):
        return isinstance(text, str) and norm_first(text) in templates

    def outputs_of(msgs):
        return [m["content"] for m in msgs if m["role"] == "tool" and isinstance(m["content"], str)]

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
        out = [dict(m) for m in msgs]
        for x in fixes.calls_of(msgs):
            o = msgs[x["msg"]]["content"]
            if x["tool"] == "text_editor" and x["args"].get("command") == "view" and isinstance(o, str) and not is_err(o) \
                    and (group != "attack" or ("sim-attack", key, x["step"], x["action"]) not in err):
                out[x["msg"]]["content"] = o.rstrip("\n") + "\n"
        return out

    def variant(v, msgs, cell, key=None, group=None):  # q2's stage_fixfull.variant (43ce3ec7), its single fixes and full / fullF9
        if v == "F9":
            return f9(msgs, key, group)
        if v == "fullF9":
            return f9(variant("full", msgs, cell), key, group)
        if v == "F7":
            return f7(msgs, cell, msgs)[0]
        if v != "full":
            return {"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "F6": f6, "F8": f8}[v](msgs)
        m = rr.rerender_messages(msgs, "diff_with_note")[0]
        m = st.swap_messages(m, "sr_gutter_no_newline")[0]
        m = f6(f5(m))
        m = f7(m, cell, msgs)[0]
        return f8(m)
    return variant, outputs_of, f1, f3


def load_inputs(run_dir, arms, keys=None):
    out = {}
    for p in sorted((Path("/acts") / run_dir / "inputs").glob("shard_*.jsonl")):
        for line in p.read_text().splitlines():
            it = json.loads(line)
            if it["arm"] in arms and (keys is None or it["key"] in keys):
                if (it["arm"], it["key"]) in out:
                    raise ValueError("a run appears twice in the source inputs")
                out[(it["arm"], it["key"])] = it
    return out


def run_job(cohort: list, honest: list, errored: list, dev: list, stamp: str) -> dict:
    sys.path[:0] = ["/root/swap", "/root/fsim", "/root/q4"]
    import clock
    import evalzip  # noqa: F401
    import fixes
    import rerender as rr
    import str_replace_tells as st
    from transformers import AutoTokenizer

    from probe_shift import reader
    from probe_shift.models import QWEN3_32B
    from probe_shift.transcripts import resolve_attachments
    c = Counter()

    # --- #36's inputs, F8's templates and q2's donors, exactly as stage_fixfull.run_job builds them
    base = load_inputs(BASE, ("real-attack-e1", "real-attack-e2", "sim-attack"))
    sims = {k: v for (a, k), v in base.items() if a == "sim-attack" and k in set(cohort)}
    hon = {k: v for (a, k), v in load_inputs(HONEST[0], (HONEST[1],), set(honest)).items()}
    if len(sims) != 939 or len(hon) != 437:
        raise ValueError(f"{len(sims)} of 939 simulated attack runs, {len(hon)} of 437 honest runs found")
    err36 = {(a, k, s, j): t for a, k, s, j, t in errored}
    real_err = defaultdict(Counter)
    for (a, k), it in base.items():
        if a == "sim-attack":
            continue
        for x in fixes.calls_of(it["messages"]):
            if err36.get((a, k, x["step"], x["action"])) is None:
                continue
            text = it["messages"][x["msg"]]["content"]
            if isinstance(text, str):
                real_err[norm_first(text)][trail(text)] += 1
    templates = {}
    for key, v in real_err.items():
        n = sum(v.values())
        if n < 5:
            continue
        templates[key] = "strip" if v["none"] >= 0.9 * n else "one" if v["one"] >= 0.9 * n else "keep"
    c["F8 templates"] = len(templates)
    tsha = hashlib.sha256(json.dumps(sorted(templates.items())).encode()).hexdigest()

    def outputs_of(msgs):
        return [m["content"] for m in msgs if m["role"] == "tool" and isinstance(m["content"], str)]
    cohort_cells = {it["cell"] for it in sims.values()}
    donors36 = {it["cell"]: n for (a, k), it in base.items() if a == "real-attack-e1" and it["environment"] in set(dev) and it["cell"] in cohort_cells
                if (n := clock.now_of(outputs_of(it["messages"]))) is not None}
    variant36, _, f1, f3 = fixers(templates, donors36, err36, rr, st, fixes, clock)

    # --- q2's composition gates (all 939), then the byte-exact gate on seeded samples against q2's staged rows
    fx = load_inputs(FIX[0], ("sim-attack-note-sr-both",))
    if hashlib.sha256((Path("/acts") / FIX[0] / "inputs" / "shard_0.jsonl").read_bytes()).hexdigest() != FIX[1]:
        raise ValueError("fix939's staged inputs are not the pinned ones")
    for k, it in sims.items():
        m = it["messages"]
        a = rr.rerender_messages(m, "diff_with_note")[0]
        rendered = {r["msg"] for r in rr.rerender_messages(m, "diff_no_newline")[1] if r["category"] in rr.RENDERED}
        b = [dict(x) | ({"content": x["content"].rstrip("\n")} if i in rendered else {}) for i, x in enumerate(f1(m))]
        c["gate|F1 then F2 equals diff_with_note"] += msha(a) == msha(b)
        c["gate|F3 then F4 equals sr_gutter_no_newline"] += msha(st.swap_messages(m, "sr_gutter_no_newline")[0]) == msha(st.swap_messages(f3(m), "sr_no_newline")[0])
        both = st.swap_messages(a, "sr_gutter_no_newline")[0]
        if ("sim-attack-note-sr-both", k) in fx:
            c["gate|fix939 runs reproduced"] += msha(both) == msha(fx[("sim-attack-note-sr-both", k)]["messages"])
        else:
            c["gate|fix939-unchanged runs left unchanged"] += msha(both) == msha(m)
    # --- per-fix byte-exact gate: q2's single-fix arms (changed runs only) on #36's simulated attack runs
    q2 = load_inputs(Q2_FX, tuple(f"sim-attack-fx-{f}" for f in FIXES if f != "F9")) | load_inputs(Q2_F9, ("sim-attack-fx-F9",))
    rng = random.Random(SEED)
    gate_rows = []
    for v in FIXES:
        arm = f"sim-attack-fx-{v}"
        staged_keys = sorted(k for k in sims if (arm, k) in q2)
        others = sorted(k for k in sims if (arm, k) not in q2)
        for kind, pool, n in (("q2 changed it", staged_keys, N_GATE), ("q2 left it unchanged", others, N_NEG)):
            for k in rng.sample(pool, min(n, len(pool))):
                it = sims[k]
                ours = variant36(v, it["messages"], it["cell"], k, "attack")
                ref = q2[(arm, k)]["messages"] if kind == "q2 changed it" else it["messages"]
                same = json.dumps(ours, sort_keys=True).encode() == json.dumps(ref, sort_keys=True).encode()
                gate_rows.append({"fix": v, "kind": kind, "same": same, "changed outputs": sum(x != y for x, y in zip(it["messages"], ours))})
        c[f"gate|q2 runs staged|{v}"] = len(staged_keys)
    gates = {"composition": {k: v for k, v in c.items() if k.startswith("gate|") and "q2 runs staged" not in k},
             "byte-exact vs q2's single-fix rows": gate_rows, "q2 single-fix runs staged": {f: c[f"gate|q2 runs staged|{f}"] for f in FIXES}}
    pos_ok = all(sum(1 for r in gate_rows if r["fix"] == f and r["kind"] == "q2 changed it") == min(N_GATE, c[f"gate|q2 runs staged|{f}"]) for f in FIXES)
    ok = (c["gate|F1 then F2 equals diff_with_note"] == 939 and c["gate|F3 then F4 equals sr_gutter_no_newline"] == 939
          and c["gate|fix939 runs reproduced"] == 823 and c["gate|fix939-unchanged runs left unchanged"] == 116
          and pos_ok and all(r["same"] for r in gate_rows) and all(c[f"gate|q2 runs staged|{f}"] > 0 for f in FIXES))
    if not ok:
        return {"STOP": "a repair gate failed: nothing is staged", "gates": gates, "counts": dict(c)}

    # --- Qwen3.5's runs: donors, F9's error flags from the .eval samples, the repairs
    staged = load_inputs(STAGED, ("q35-real-attack-1", "q35-sim-attack", "q35-sim-honest"))
    q_sims = {k: it for (a, k), it in staged.items() if a == "q35-sim-attack"}
    q_hon = {k: it for (a, k), it in staged.items() if a == "q35-sim-honest"}
    if (len(q_sims), len(q_hon)) != (191, 19):
        raise ValueError(f"{len(q_sims)} simulated attack and {len(q_hon)} simulated honest runs staged, not 191 and 19")
    env_of = {x["key"]: x["env"] for x in json.loads(Path("/root/t196.json").read_text())["cells"]}  # the 196-task table's environments
    # F7 (Amendment 1b): clock.now_of only counts stamps in the fix's run week (September 14-21, Sol's cohort). Qwen3.5's real
    # runs ran in October, so they cannot be donors; F7 runs exactly as on the Sol side, with q2's own donor pool (Sol's
    # dev-environment real e1 runs), shifting only Qwen3.5 simulated runs whose own clock falls in the run week. Counted:
    c["Qwen3.5 dev-environment real run-1 attack runs with a run-week clock (could be donors)"] = sum(
        1 for (a, k), it in staged.items() if a == "q35-real-attack-1" and env_of[it["cell"]] in set(dev) and clock.now_of(outputs_of(it["messages"])) is not None)
    c["Qwen3.5 simulated attack runs with a run-week clock (F7 shifts them)"] = sum(clock.now_of(outputs_of(it["messages"])) is not None for it in q_sims.values())
    c["Qwen3.5 simulated honest runs with a run-week clock (F7 shifts them)"] = sum(clock.now_of(outputs_of(it["messages"])) is not None for it in q_hon.values())
    donors = donors36
    if not donors:
        raise ValueError("q2's donor pool is empty")
    err = set()
    import zipfile
    for k, it in q_sims.items():
        launch, rest = it["staged_epoch_key"].split("/", 1)
        ev = Path("/ev/evals_v1") / launch / rest.rsplit("#", 1)[0]
        with zipfile.ZipFile(ev) as z:
            names = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
            if len(names) != 1:
                raise ValueError("an .eval without exactly one sample")
            s = resolve_attachments(json.loads(z.read(names[0])))
        submit_ids = {tc.get("id") for m in s["messages"] if m["role"] == "assistant" for tc in (m.get("tool_calls") or []) if tc["function"] == "submit"}
        flags = [m.get("error") is not None for m in s["messages"] if m["role"] == "tool" and not (m.get("tool_call_id") in submit_ids or m.get("function") == "submit")]
        tool_ix = [i for i, m in enumerate(it["messages"]) if m["role"] == "tool"]
        if len(flags) != len(tool_ix):
            raise ValueError("the sample's tool messages and the staged reader messages differ in number")
        by_msg = dict(zip(tool_ix, flags))
        for x in fixes.calls_of(it["messages"]):
            if by_msg.get(x["msg"]):
                err.add(("sim-attack", k, x["step"], x["action"]))
        c["Qwen3.5 simulated attack tool outputs with an Inspect error"] += sum(flags)
    variant, _, _, _ = fixers(templates, donors, err, rr, st, fixes, clock)
    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"], local_files_only=True)
    run_dir, qdir = OUT
    rows, record, qmeta = defaultdict(list), [], []
    for v in FIXES:
        for group, runs_ in (("attack", q_sims), ("honest", q_hon)):
            for k, it in sorted(runs_.items()):
                m = it["messages"]
                cell = it["cell"] if group == "attack" else f"{it['environment']}:{it['main_task']}"
                new = variant(v, m, cell, k, group)
                if len(new) != len(m):
                    raise ValueError("a variant changed the number of messages")
                changed = [i for i, (x, y) in enumerate(zip(m, new)) if x != y]
                if any(m[i]["role"] != "tool" or set(m[i]) != set(new[i]) or any(m[i][f_] != new[i][f_] for f_ in m[i] if f_ != "content") for i in changed):
                    raise ValueError(f"{v}: a non-tool message or a field other than content changed")
                arm = f"q35-sim-{'attack' if group == 'attack' else 'honest'}-fx-{v}"
                fam = {x["msg"]: family(x) for x in fixes.calls_of(m)}
                c[f"{v}|{group}|runs changed"] += bool(changed)
                c[f"{v}|{group}|outputs changed"] += len(changed)
                for i in changed:
                    c[f"{v}|{group}|outputs changed|{fam.get(i, 'unpaired')}"] += 1
                # unchanged runs are not staged: the scoring reuses their as-is rows (their fixed messages hash equal to the as-is ones)
                record.append({"fix": v, "arm": arm, "key": k, "changed_messages": changed, "messages_sha256": msha(new),
                               "as_is_messages_sha256": msha(m), "staged": bool(changed)})
                if not changed:
                    continue
                seg = reader.segment_end_positions(tok, new, max_tokens=CUT)
                rows[arm].append(it | {"arm": arm, "messages": new, "tokens": seg["n_tokens"], "truncated": seg["truncated"], "repair": v})
                qmeta.append({"key": k, "arm": arm, "positions": seg["positions"]})
    if any(r["messages_sha256"] != r["as_is_messages_sha256"] for r in record if not r["staged"]) or any(r["messages_sha256"] == r["as_is_messages_sha256"] for r in record if r["staged"]):
        raise ValueError("the changed/unchanged split disagrees with the message hashes")
    dst, qd = Path("/acts") / run_dir, Path("/acts") / qdir
    if dst.exists() or qd.exists():
        raise FileExistsError(f"{run_dir} or {qdir} exists: never overwritten")
    (dst / "inputs").mkdir(parents=True)
    shas = {}
    for n, (arm, rs) in enumerate(sorted(rows.items())):
        body = "".join(json.dumps(x) + "\n" for x in sorted(rs, key=lambda x: x["key"])).encode()
        (dst / "inputs" / f"shard_{n}.jsonl").write_bytes(body)
        shas[f"inputs/shard_{n}.jsonl"] = {"arm": arm, "runs": len(rs), "sha256": hashlib.sha256(body).hexdigest()}
    for q in qmeta:
        p = qd / "meta" / q["arm"] / f"{filename(q['key'])}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(q))
    rec = json.dumps({"runs": record, "shards": shas, "f8_templates_sha256": tsha, "fixes": list(FIXES), "source": STAGED}).encode()
    (dst / "stage_record.json").write_bytes(rec)
    written = {"mode": "stage", "staged": run_dir, "qmeta": qdir, "gates pass": True, "shards_files": shas,
               "runs by arm": {a: sorted(x["key"] for x in rs) for a, rs in sorted(rows.items())},
               "record_sha256": hashlib.sha256(rec).hexdigest(),
               "forward tokens by arm": {a: sum(x["tokens"] for x in rs) for a, rs in sorted(rows.items())},
               "fixes with no changed run": [v for v in FIXES if not any(r["fix"] == v and r["staged"] for r in record)]}
    acts.commit()
    return {"gates": gates, "gates pass": True, "f8 templates sha256": tsha, "counts": dict(sorted(c.items())), "staged": written}


@app.function(image=image, volumes={"/acts": acts, "/ev": evals, "/hf": modal.Volume.from_name(HF).read_only()}, cpu=8, memory=65536, timeout=3600, retries=0)
def job(cohort: list, honest: list, errored: list, dev: list, stamp: str) -> dict:
    try:
        out = run_job(cohort, honest, errored, dev, stamp)
    except Exception as e:  # noqa: BLE001 — type and location only (no staged text crosses the boundary)
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"repair failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}") from None
    d = Path("/acts") / OUT[0]
    d.mkdir(parents=True, exist_ok=True)
    (d.parent / f"{OUT[0]}_result_{stamp}.json").write_text(json.dumps(out, indent=1))
    acts.commit()
    return out


@app.local_entrypoint()
def main():
    for f in (Path(__file__), HERE / "RULE.md"):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"commit {f.name} before running")
    cohort = sorted(r["key"] for r in json.loads((REPO / RUNS36).read_text()) if r["arm"] == "sim-attack")
    honest = sorted(r["key"] for r in json.loads((REPO / HONEST_RUNS).read_text()) if r["arm"] == "sim-honest")
    errored = [[r["arm"], r["key"], r["step"], r["action"], r["tool"]] for r in map(json.loads, (REPO / ERRORS).read_text().splitlines())
               if r["error"] is not None and r["arm"] in ("real-attack-e1", "real-attack-e2", "sim-attack")]
    dev = json.loads((REPO / SPLIT).read_text())["dev"]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    call = job.spawn(cohort, honest, errored, dev, stamp)
    print(f"fix-by-fix staging spawned (function call {call.object_id}); result at {OUT[0]}_result_{stamp}.json on {ACTS}", flush=True)
