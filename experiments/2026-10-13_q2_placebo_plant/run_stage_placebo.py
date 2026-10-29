"""G1 placebo (PLAN.md): staging. One Modal CPU container; a copy of `../2026-10-12_plantslip_transfer/q2_reply_stage.py`
(P4's staging, on aswin/plantslip-q2-reply @ 27dd1702) with `plant_placebo()` in place of `plant_both()`, real e2 only, and no
new controls (the gate re-extracts P4's staged controls: PLAN.md). From the staged messages #36's dump read, it:
- picks the placebo's end character by PLAN.md Addendum 2's rule: the first of `placebo.END_CANDIDATES` that ends 0
  str_replace outputs in each of #36's four arms (real e1, real e2, real honest, simulated); none: stages nothing;
- applies `plant_placebo()` to the 939 real e2 runs and stages the runs it changed as run
  `2026-10-13_q2_placebo_reply_inputs_v2` (arm `real-attack-e2-placebo`) on the shared activations Volume (v1, the
  first staging with a trailing space, is superseded by Addendum 2 and kept);
- checks the matching against P4 on every run (`plant_both()` applied to the same messages): the same outputs, the
  same lines, the same place and length (PLAN.md, "Matching"), and fails on the first mismatch;
- counts the placebo's three forms in every arm, and stages nothing if the simulated arm writes any of them (PLAN.md,
  "Gate before launch");
- counts, per changed run unplanted, with P4's plant and with the placebo: Qwen3-32B tokens, and the output_end rows
  kept under the 32k cut (which outputs enter the run's mean).
Counts and keys only leave the container.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-13_q2_placebo_plant/run_stage_placebo.py --runs <#36 runs.json> --p4-stage <P4's staging record>
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
EXP = HERE.parent
FILES = {"rerender.py": EXP / "2026-10-07_q2_rendering_swap", "str_replace_tells.py": EXP / "2026-10-07_q2_rendering_swap",
         "fixes.py": EXP / "2026-10-09_fixed_sim", "plant_both.py": EXP / "2026-10-12_plantslip_transfer",
         "placebo.py": EXP / "2026-10-13_q2_placebo_plant"}  # named, not HERE: in the container HERE is /root
APP = "probe-shift_aswin-surya_sub-q2-placebo_stage_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"  # the staged messages the readers read; the new staged run goes here too
SOURCE = "2026-10-06_la_segment-end_qwen3-32b_v1"
PLACEBO_RUN, ARM, SRC_ARM = "2026-10-13_q2_placebo_reply_inputs_v2", "real-attack-e2-placebo", "real-attack-e2"
TOKENIZER = ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137")
N_SHARD_FILES = 8
PIN = {"runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",  # #36's run list
       "P4 stage": "f1a68c90a5d26799925bda1c0d574728a547cde6c2ec3b76bfa18c3691c75157"}  # q2_reply_stage_2026-10-12_135318.json
SLIPS = {"gutter+newline": "gutter-restyle+end", "note": "note-text"}
GATE_FORMS = ("text_editor outputs with a placebo gutter line in the edited section", "tool outputs containing the placebo note")
ARMS4 = ("real-attack-e1", "real-attack-e2", "real-honest", "sim-attack")
SUPERSEDED = "stage_placebo_2026-10-13_014508.json @ f2f86a25 (staged run 2026-10-13_q2_placebo_reply_inputs_v1): its trailing space is a real-side form (PLAN.md Addendum 2)"

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2==3.1.6", "huggingface_hub").add_local_python_source("probe_shift")
for _n, _f in FILES.items():
    image = image.add_local_file(_f / _n, f"/root/{_f.name}/{_n}")


def _mods():
    import sys
    for folder in {f for f in FILES.values()}:
        for p in (f"/root/{folder.name}", str(folder)):
            if p not in sys.path:
                sys.path.insert(0, p)
    import fixes  # noqa: E402  (it puts the swap folder on the path)
    import placebo
    import rerender
    from plant_both import plant_both
    return fixes, rerender, plant_both, placebo


@app.function(image=image, volumes={"/acts": acts}, cpu=16, memory=32768, timeout=3600, retries=0)
def stage(e2_keys: list[str], cohort: list[list[str]]) -> dict:
    try:
        return _stage(e2_keys, cohort)
    except Exception as e:  # nothing from the exception (which may quote staged text) crosses the boundary
        import traceback  # the failing frame's file and line, and a missing module's name: never staged text
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"staging failed: {type(e).__name__} {getattr(e, 'name', '') or ''} at {Path(tb.filename).name}:{tb.lineno}") from None


def _check(o: str, p4: str, pl: str, slip: str, rerender, placebo, end: str, n) -> None:
    """PLAN.md's matching, for one output: raises on any mismatch; adds the size counts to n."""
    if slip == "gutter+newline":
        cut = "+++ edited_version\n"
        h0, h4, hp = (x.partition(cut)[0] for x in (o, p4, pl))
        assert h0 == h4 == hp and cut in o, "header"
        assert p4.endswith("\n") and not p4[:-1].endswith("\n") and pl.endswith(end) and not pl[:-1].endswith("\n"), "end"
        b0, b4, bp = o.rstrip("\n").partition(cut)[2].split("\n"), p4[:-1].partition(cut)[2].split("\n"), pl[:-1].partition(cut)[2].split("\n")
        assert len(b0) == len(b4) == len(bp), "line count"
        ch4 = {i for i, (a, b) in enumerate(zip(b0, b4)) if a != b}
        chp = {i for i, (a, b) in enumerate(zip(b0, bp)) if a != b}
        assert ch4 == chp and ch4, "changed lines"
        for i in chp:  # each changed line: the prefix restyled, every other character the original's
            p = placebo.PREFIX.match(b0[i])
            assert p and bp[i] == f"{p.group(2)}:".ljust(p.end()) + b0[i][p.end():] and len(bp[i]) == len(b0[i]), "placebo line"
        n["str_replace: outputs whose original ended in newlines (both plants normalize them away first)"] += o.endswith("\n")
        n["str_replace: lines changed (P4 = placebo)"] += len(chp)
        n["str_replace: characters P4 removes from those lines"] += sum(len(b0[i]) - len(b4[i]) for i in ch4)
        n["str_replace: characters the placebo rewrites on those lines"] += sum(sum(x != y for x, y in zip(b0[i], bp[i])) for i in chp)
        n["str_replace: characters appended at the end (P4 newline = the placebo's end character)"] += 1
    else:
        R = rerender.REVIEW
        i = len(o.rstrip()) - len(R)
        assert o[i:i + len(R)] == R and len(pl) == len(o), "note place"
        assert pl[:i] == o[:i] and pl[i + len(R):] == o[i + len(R):] and pl[i:i + len(R)] != R, "note span"
        assert p4 == o.rstrip()[:i].rstrip("\n"), "P4 removes that note"
        n["insert: characters P4 removes"] += len(o) - len(p4)
        n["insert: notes followed by whitespace (P4 removes it, the placebo keeps it)"] += o[i + len(R):] != ""
        n["insert: characters the placebo rewrites (the note, in place)"] += sum(x != y for x, y in zip(o[i:i + len(R)], pl[i:i + len(R)]))


def _stage(e2_keys, cohort):
    import re
    from collections import Counter

    from transformers import AutoTokenizer

    from probe_shift.reader import segment_end_positions
    fixes, rerender, plant_both, placebo = _mods()
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    items = {}
    for shard in sorted((Path("/acts") / SOURCE / "inputs").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            items[(it["arm"], it["key"])] = it
    cohort = {(a, k) for a, k in cohort}  # #36's 3,807 runs (runs.json): the source dump holds more simulated runs
    if not {(SRC_ARM, k) for k in e2_keys} <= cohort <= set(items):
        raise ValueError("a cohort run is missing from the staged inputs")
    # the gate's counts: the placebo's three forms, in every arm of #36's cohort (PLAN.md)
    line_re = re.compile(r"^\d+: *(?:[ +\-](?: .*)?)?$")
    forms = Counter()
    for (arm, k) in sorted(cohort):
        it = items[(arm, k)]
        forms[(arm, "runs")] += 1
        for c in fixes.calls_of(it["messages"]):
            o = it["messages"][c["msg"]]["content"]
            if not isinstance(o, str):
                continue
            body = o.partition("+++ edited_version\n")[2] if c["tool"] == "text_editor" else ""
            forms[(arm, "text_editor outputs with a placebo gutter line in the edited section")] += any(line_re.match(l) for l in body.split("\n"))
            for e in placebo.END_CANDIDATES:  # Addendum 2's count
                forms[(arm, f"str_replace outputs ending in {e!r}")] += c["tool"] == "text_editor" and c["args"].get("command") == "str_replace" and o.endswith(e)
                forms[(arm, f"reference: text_editor outputs ending in {e!r}")] += c["tool"] == "text_editor" and o.endswith(e)
            forms[(arm, "tool outputs containing the placebo note")] += placebo.NOTE in o
            forms[(arm, "reference: tool outputs of any tool with a line of the placebo gutter form")] += any(line_re.match(l) for l in o.split("\n"))
    gate = {c: int(forms[("sim-attack", c)]) for c in GATE_FORMS} | {"runs": int(forms[("sim-attack", "runs")])}
    if gate["runs"] != 939:
        raise ValueError("not 939 simulated runs in the cohort")
    end_counts = {repr(e): {a: int(forms[(a, f"str_replace outputs ending in {e!r}")]) for a in ARMS4} for e in placebo.END_CANDIDATES}
    if any(forms[(a, "runs")] == 0 for a in ARMS4):
        raise ValueError("an arm of the cohort is empty")
    end = next((e for e in placebo.END_CANDIDATES if not any(end_counts[repr(e)].values())), None)  # Addendum 2's rule
    gate |= {"end character counts (str_replace outputs ending in it, by arm)": end_counts, "end character": None if end is None else repr(end)}
    if end is None or any(gate[c] for c in GATE_FORMS):  # PLAN.md: the simulator must never write the placebo; Addendum 2: no arm writes its end
        return {"gate": "fail", "simulated arm counts": gate, "placebo forms by arm": [{"arm": a, "count": c, "n": int(v)} for (a, c), v in sorted(forms.items())]}
    def survives(content):  # as q2_reply_stage.py's survives(): does the final character change the rendered prompt?
        def render(c):
            msgs = [{"role": "user", "content": "x"},
                    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "text_editor", "arguments": {"command": "str_replace"}}}]},
                    {"role": "tool", "content": c}]
            return tok.apply_chat_template(msgs, tokenize=False)
        return render(content) != render(content[:-1])
    n, staged, per_run = Counter(), [], []
    for k in sorted(e2_keys):
        it = items[(SRC_ARM, k)]
        p4, c4 = plant_both(it["messages"], fixes, rerender)
        new, changed = placebo.plant_placebo(it["messages"], fixes, rerender, end)
        assert [c[:3] for c in changed] == [c[:3] for c in c4] and [SLIPS[c[3]] for c in c4] == [c[3] for c in changed], "the same outputs"
        if not changed:
            n["runs unchanged (P4 = placebo)"] += 1
            continue
        diff = [i for i, (a, b) in enumerate(zip(it["messages"], new)) if a != b]
        assert len(new) == len(it["messages"]) and diff == [c[0] for c in changed]
        assert all({x for x in new[i] if new[i][x] != it["messages"][i][x]} == {"content"} for i in diff)
        for (i, _, _, slip) in c4:
            _check(it["messages"][i]["content"], p4[i]["content"], new[i]["content"], slip, rerender, placebo, end, n)
            if slip == "gutter+newline":  # does the end character reach the Qwen3-32B prompt? (P4's survives() check)
                n["str_replace outputs whose end character reaches the Qwen3-32B prompt"] += survives(new[i]["content"])
            n[f"outputs changed: {slip} (P4) = {SLIPS[slip]} (placebo)"] += 1
        n["runs changed (P4 = placebo)"] += 1; n["outputs changed (P4 = placebo)"] += len(changed)
        se = [segment_end_positions(tok, m, 32768) for m in (it["messages"], p4, new)]  # the extractor's positions and cut
        t0, t4, tp = (x["n_tokens"] for x in se)
        k0, k4, kp = ({(q["step"], q["action"]) for q in x["positions"] if q["kind"] == "output_end" and q["kept"]} for x in se)
        n["kept output_end rows, changed runs: unplanted"] += len(k0); n["kept output_end rows, changed runs: P4"] += len(k4); n["kept output_end rows, changed runs: placebo"] += len(kp)
        n["changed runs whose kept output_end set differs: P4 vs unplanted"] += k4 != k0
        n["changed runs whose kept output_end set differs: placebo vs unplanted"] += kp != k0
        n["changed runs whose kept output_end set differs: placebo vs P4"] += kp != k4
        n["Qwen3-32B tokens, changed runs: unplanted"] += t0; n["Qwen3-32B tokens, changed runs: P4"] += t4; n["Qwen3-32B tokens, changed runs: placebo"] += tp
        n["Qwen3-32B tokens within the 32k cut, changed runs: placebo"] += min(tp, 32768)
        n["changed runs over 32,768 tokens: unplanted"] += t0 > 32768; n["changed runs over 32,768 tokens: P4"] += t4 > 32768; n["changed runs over 32,768 tokens: placebo"] += tp > 32768
        staged.append(it | {"arm": ARM, "messages": new, "tokens": tp, "truncated": tp > 32768, "plant": "placebo", "orig_arm": SRC_ARM,
                            "changed": [list(c) for c in changed]})
        per_run.append({"arm": ARM, "orig_arm": SRC_ARM, "key": k, "outputs changed": len(changed), "tokens: unplanted, P4, placebo": [t0, t4, tp],
                        "kept output_end rows: unplanted, P4, placebo": [len(k0), len(k4), len(kp)],
                        "kept output_end (step, action) differences": {"only unplanted vs P4": sorted(k0 - k4), "only P4 vs unplanted": sorted(k4 - k0),
                                                                       "only unplanted vs placebo": sorted(k0 - kp), "only placebo vs unplanted": sorted(kp - k0)}})
    shas = {}
    d = Path("/acts") / PLACEBO_RUN / "inputs"
    if d.exists() and any(d.iterdir()):
        raise FileExistsError(f"{PLACEBO_RUN} is already staged")
    d.mkdir(parents=True)
    for s in range(N_SHARD_FILES):
        body = "".join(json.dumps(x) + "\n" for x in staged[s::N_SHARD_FILES]).encode()
        (d / f"shard_{s}.jsonl").write_bytes(body)
        shas[f"{PLACEBO_RUN}/shard_{s}.jsonl"] = hashlib.sha256(body).hexdigest()
    acts.commit()
    return {"gate": "pass", "simulated arm counts": gate, "end character": repr(end), "matching (every changed output passed PLAN.md's checks)": {k: int(v) for k, v in sorted(n.items())},
            "placebo forms by arm": [{"arm": a, "count": c, "n": int(v)} for (a, c), v in sorted(forms.items())],
            "per run": per_run, "inputs_sha256": shas, "placebo": [{"key": x["key"], "orig_arm": x["orig_arm"], "arm": x["arm"]} for x in staged]}


@app.local_entrypoint()
def main(runs: str, p4_stage: str):
    import subprocess
    for f in [Path(__file__).resolve(), HERE / "PLAN.md"] + [folder / name for name, folder in FILES.items()]:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    if sha(runs) != PIN["runs.json"] or sha(p4_stage) != PIN["P4 stage"]:
        raise SystemExit("not #36's runs.json, or not P4's staging record")
    e2 = sorted(r["key"] for r in json.loads(Path(runs).read_text()) if r["arm"] == SRC_ARM)
    if len(e2) != 939:
        raise SystemExit("not 939 e2 runs")
    cohort = [[r["arm"], r["key"]] for r in json.loads(Path(runs).read_text())]
    res = stage.remote(e2, cohort)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    if res["gate"] != "pass":  # nothing was staged: the record says why, and nothing launches
        with open(dest := HERE / "results" / f"stage_placebo_gate_fail_{stamp}.json", "x") as fh:
            fh.write(json.dumps({"plan": "PLAN.md (this folder)", "git": head} | res, indent=1) + "\n")
        raise SystemExit(f"gate failed: the simulated arm writes a placebo form {res['simulated arm counts']}; {dest}")
    p4_keys = sorted(x["key"] for x in json.loads(Path(p4_stage).read_text())["planted"] if x["orig_arm"] == SRC_ARM)
    if sorted(x["key"] for x in res["placebo"]) != p4_keys:
        raise SystemExit("the placebo changed a different set of e2 runs than P4")
    out = {"plan": "PLAN.md (this folder)", "git": head, "runs.json sha256": PIN["runs.json"], "P4 stage sha256": PIN["P4 stage"], "source": SOURCE,
           "staged run (Volume probe-shift_aswin-surya_sub-la-activations)": PLACEBO_RUN, "supersedes": SUPERSEDED, "e2 runs changed = P4's 845": True} | res
    with open(dest := HERE / "results" / f"stage_placebo_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("per run", "placebo")}, indent=1)); print(dest)
