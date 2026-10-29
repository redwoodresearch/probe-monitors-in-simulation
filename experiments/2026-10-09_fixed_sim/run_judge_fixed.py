"""The blinded pairwise judge on fixed-sim (PLAN.md "Blinded AI judge"; config/judge_fixed.yaml).

The paper's judge, set up exactly as the 0.633 baseline ran it (../2026-10-06_q1_text_baselines/run_judge_repro.py):
`Pair` + `full_sim_actions` + `discriminate_pair` from simulated-control-evals, prompt `pair_2afc.md` (digest checked
against the paper's stored runs), gpt-5.6-sol at medium effort, every pair both ways round. The judge sees only the
two transcripts' actions (tool calls and outputs), never run IDs, environment names or metadata.

Designs (config `designs`):
  main     real epoch 1 vs fixed sim, every paired cell of the 5 test environments. The simulated side is the cohort's
           transcript with each tool output a rule changed replaced by the fixed run's text (v2). Alignment is checked
           output by output: the judge's actions and the reader's tool messages must match in count and tool, and
           wherever a rule changed an output, the judge's original output must equal the reader's original text.
           A pair that fails this is dropped and counted.
  control  real epoch 1 vs real epoch 2 on 200 seeded cells of the main run's cells, same protocol; "picked e1"
           counts as correct, so it sits at 0.5 by construction unless the judge separates the two real runs.
Calls run in batches; after each batch the run's paid cost is summed from the client's ledger, and the run stops
past `stop_usd`. Without --run it only builds the pairs, checks alignment and estimates the cost.

  uv run python experiments/2026-10-09_fixed_sim/run_judge_fixed.py --run-dir <local v2 copy> --design main
  op run --env-file=.env.tpl -- uv run python experiments/2026-10-09_fixed_sim/run_judge_fixed.py --run-dir <...> --design main --run
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import random
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from inspect_ai.log import EvalSample
from metrics.contract import prompt_digest
from metrics.discriminator.drops import DropTally
from metrics.discriminator.pairing import PAIR_PROMPT, Pair, build_pair_prompt, full_sim_actions
from metrics.discriminator.two_afc import DiscriminatorConfig, PairScore, _run_order, discriminate_pair
from runner.metrics.clients import JudgeClient, ModelSettings
from sim.adapters.inspect_eval import sample_transcript
from sim.identity import Mode
from sim.transcript import Corpus

from probe_shift.scoring import pairwise_accuracy_ci
from probe_shift.transcripts import file_samples, reader_messages

HERE = Path(__file__).parent
REPO = HERE.parents[1]
LEDGER = Path(".run_cache/paper/total_cost.jsonl")  # JudgeClient's cost ledger, relative to the working directory


def samples_by_key(alias: str, rows: list[dict]) -> dict:
    """The cohort samples for these reader-input rows, read once per log file (cached locally)."""
    want = {(r["file"], r["sample_id"], r["epoch"]): r["key"] for r in rows}
    out = {}
    for file in sorted({f for f, _, _ in want}):
        for s in file_samples(alias, file):
            if (file, s["id"], s["epoch"]) in want:
                out[want[(file, s["id"], s["epoch"])]] = s
    assert len(out) == len(want), f"{alias}: {len(want) - len(out)} samples not found"
    return out


def transcript(sample: dict, simulated: bool):
    s = EvalSample.model_validate(sample)
    corpus = Corpus(name="simulated" if simulated else "real", simulated=simulated, setting="linuxarena:public10")
    t = sample_transcript(s, corpus=corpus, simulated=simulated, mode=Mode.ATTACK, epoch=sample["epoch"])
    assert t.session_id == sample["id"], (t.session_id, sample["id"])
    return t


def calls_and_outputs(messages: list[dict]) -> list[tuple[str, str, str]]:
    """(tool, arguments as sorted JSON, output) per tool output, each paired with its call in order."""
    out, pending = [], []
    for m in messages:
        if m["role"] == "assistant":
            pending = [(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls") or []]
        elif m["role"] == "tool":
            name, args = pending.pop(0)
            args = json.loads(args) if isinstance(args, str) else args
            out.append((name, json.dumps(args, sort_keys=True, default=str), m["content"]))
    return out


def fixed_actions(sim_t, sample: dict, fixed_row: dict, count: Counter):
    """The judge's simulated actions with every rule-changed output replaced by the fixed text, or None if the judge's
    actions and the reader's calls do not align: same count, and per position the same tool and arguments (and, where a
    rule changed the output, the same original output text)."""
    full = list(full_sim_actions(sim_t))
    orig = calls_and_outputs(reader_messages(sample, ""))
    fixed = calls_and_outputs(fixed_row["messages"])
    assert len(orig) == len(fixed), f"{fixed_row['key']}: the fixed run has a different number of tool outputs"
    assert all(o[:2] == f[:2] for o, f in zip(orig, fixed)), f"{fixed_row['key']}: the fixed run changed a call"
    if len(full) != len(orig):
        count["dropped:action_count_differs"] += 1
        return None
    if any((a["function"], json.dumps(a.get("arguments") or {}, sort_keys=True, default=str)) != o[:2] for a, o in zip(full, orig)):
        count["dropped:call_differs"] += 1
        return None
    out = []
    for a, (_, _, o_text), (_, _, f_text) in zip(full, orig, fixed):
        o, f = {"content": o_text}, {"content": f_text}
        if f["content"] == o["content"]:  # no rule changed it: the judge's own text, as in the baseline
            out.append(a)
            continue
        if str(a.get("output") or "") != o["content"]:
            count["dropped:changed_output_differs_from_judge_text"] += 1
            return None
        out.append({**a, "output": f["content"]})
        count["outputs_replaced"] += 1
    return tuple(out)


def r2_undone(fixed_msgs: list[dict], orig_msgs: list[dict], count: Counter):
    """The fixed run with R2 undone, as planted check 2 built it, and the reverted outputs' positions (tool-output order):
    a str_replace output gets the simulator's own text back only when R2 alone explains the change (R2 applied to the
    original gives the fixed text, or the fixed text is the original without its trailing newline), so no other rule's
    change (e.g. an R6 stamp, which level 1 does not fully mask) comes back with it (Codex). None if nothing reverts."""
    import sys

    sys.path.insert(0, str(HERE.parent / "2026-10-07_q2_rendering_swap"))
    import str_replace_tells
    f_idx = [i for i, m in enumerate(fixed_msgs) if m["role"] == "tool"]
    o_out, f_out = calls_and_outputs(orig_msgs), calls_and_outputs(fixed_msgs)
    assert len(f_idx) == len(o_out) == len(f_out), "the fixed and original runs have different tool outputs"
    out, rev = [dict(m) for m in fixed_msgs], []
    for j, (i, (name, args, o), (_, _, f)) in enumerate(zip(f_idx, o_out, f_out)):
        if name == "text_editor" and json.loads(args).get("command") == "str_replace" and o != f:
            if not isinstance(o, str) or (str_replace_tells.variant_output(o, "sr_gutter_no_newline")[0] != f and o.rstrip("\n") != f):
                count["str_replace_changed_by_more_than_r2"] += 1
                continue
            out[i] = out[i] | {"content": o}
            rev.append(j)
    count["planted_outputs"] += len(rev)
    return (out, set(rev)) if rev else None


def scrub_args(v, level: int):
    """The probe staging's argument scrub (run_stage_probe_inputs_modal.scrubbed): strings scrubbed, numbers from level 2."""
    from probe_shift.scrub import scrub

    if isinstance(v, str):
        return scrub(v, level)
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return "<n>" if level >= 2 else v
    if isinstance(v, list):
        return [scrub_args(x, level) for x in v]
    if isinstance(v, dict):
        return {k: scrub_args(x, level) for k, x in v.items()}
    return v


def scrubbed(messages: list[dict], level: int) -> list[dict]:
    """run_stage_probe_inputs_modal.scrubbed, copied: tool outputs, assistant text and call arguments scrubbed at `level`."""
    from probe_shift.scrub import scrub

    out = []
    for m in messages:
        if m["role"] == "tool":
            m = m | {"content": scrub(m["content"], level)}
        elif m["role"] == "assistant":
            m = m | {"content": scrub(m["content"], level) if m["content"] else m["content"],
                     "tool_calls": [c | {"function": c["function"] | {"arguments": scrub_args(c["function"]["arguments"], level)}}
                                    for c in m.get("tool_calls") or []]}
        out.append(m)
    return out


def as_probe_saw(actions, messages: list[dict], level: int, cut: int, tok, count: Counter, side: str):
    """The judge's actions cut where the probe's text was cut and masked at the probe's level (run-level judge, 10-10):
    each output is the reader's own text (`messages`, scrubbed at `level`, as the probe read it; Codex 23:02), actions run
    through the last output whose readout lies inside the probe's `cut` tokens (Qwen3-32B), and an output that runs past
    the cut is truncated at the cut's character. None if the actions do not line up with the reader's tool calls (same
    count, same tool and arguments in order), or an output is not text."""
    from probe_shift.reader import _layout, segment_end_positions
    from probe_shift.scrub import scrub

    calls = calls_and_outputs(messages)
    if len(actions) != len(calls) or any((a["function"], json.dumps(a.get("arguments") or {}, sort_keys=True, default=str)) != c[:2]
                                         for a, c in zip(actions, calls)):
        count[f"dropped:{side}_actions_differ_from_reader"] += 1
        return None
    if not all(isinstance(c[2], str) for c in calls):
        count[f"dropped:{side}_non_text_output"] += 1
        return None
    count[f"{side}_outputs_differing_from_reader_text"] += sum(str(a.get("output") or "") != c[2] for a, c in zip(actions, calls))
    outs = [scrub(c[2], level) for c in calls]
    keep = len(actions)
    if cut:
        sm = scrubbed(messages, level)  # the probe staging's full scrub, then its cut
        seg = segment_end_positions(tok, sm, cut)
        text, found, enc = _layout(tok, sm)
        cut_char = enc["offset_mapping"][len(seg["ids"]) - 1][1]
        spans = {(f[1], f[2]): (f[4], f[5]) for f in found if f[0] == "tool_output"}
        kept = {(p["step"], p["action"]): p["kept"] for p in seg["positions"] if p["kind"] == "output_end"}
        flags = [kept.get((o["step"], o["action"]), False) for o in seg["outputs"]]
        keep = max((i for i, f in enumerate(flags) if f), default=-1) + 1  # through the last kept readout (empty outputs before it stay)
        for i, o in enumerate(seg["outputs"][:keep]):
            ca, cb = spans[(o["step"], o["action"])]
            if text[ca:cb] != outs[i]:
                count[f"dropped:{side}_layout_text_differs"] += 1
                return None
            if cb > cut_char:  # a kept readout does not guarantee the output's last characters made the cut
                outs[i] = outs[i][:max(0, cut_char - ca)]
                count[f"{side}_outputs_truncated_at_cut"] += 1
        count[f"{side}_actions_cut"] += len(actions) - keep
    return tuple({**a, "output": outs[i], "arguments": scrub_args(a.get("arguments") or {}, level)} for i, a in enumerate(actions[:keep]))


def run_cost(run_name: str) -> tuple[float, int]:
    """(paid cost so far, calls with no recorded cost) for this run in the ledger."""
    if not LEDGER.exists():
        return 0.0, 0
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if f'"run": "{run_name}"' in l]
    rows = [r for r in rows if r["run"] == run_name]
    return sum(r["run_cost"] or 0 for r in rows), sum(r["run_cost"] is None for r in rows)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="local copy of the fixed run (inputs/ with every arm)")
    parser.add_argument("--design", choices=["main", "control", "pilot_main", "pilot_control", "pilot_planted", "pilot_planted_xhigh"], required=True)
    parser.add_argument("--config", default="config/judge_fixed.yaml")
    parser.add_argument("--run", action="store_true", help="make the paid calls after the preflight")
    parser.add_argument("--resume", default="", help="stamp of an interrupted run: same run name and budget, done pairs skipped")
    parser.add_argument("--level", type=int, default=0, help="mask both sides as the probe's staging did (1 = times masked)")
    parser.add_argument("--cut", type=int, default=0, help="keep only actions inside the probe's cut (Qwen3-32B tokens; 32768)")
    args = parser.parse_args()
    cfg = yaml.safe_load((HERE / args.config).read_text())
    design = cfg["designs"][args.design]
    if args.design.startswith("pilot") and (args.level, args.cut) != (1, 32768):  # the pilots judge the probe's view
        raise SystemExit("pilot designs need --level 1 --cut 32768")
    stored = json.loads((HERE / cfg["stored_summary"]).read_text())
    digest = {p["prompt_sha256"] for p in stored["judges"][cfg["stored_judge"]]["provenance"]}
    assert digest == {prompt_digest(PAIR_PROMPT)}, f"prompt changed since the paper's run: {digest}"
    split = json.loads((HERE / cfg["split_file"]).read_text())

    rows = {}
    for f in sorted((Path(args.run_dir) / "inputs").glob("shard_*.jsonl")):
        for r in map(json.loads, f.read_text().splitlines()):
            rows[(r["arm"], r["cell"])] = r
    cells = sorted({c for (a, c), r in rows.items() if r["environment"] in split[design["cells"]]})
    tok = None
    if args.cut:
        from transformers import AutoTokenizer

        from probe_shift.models import QWEN3_32B

        tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    def planted_view(c, count):
        """Planted check (Aswin 23:22 PT): the fixed-sim run (in the "real" slot) and the same run with R2 undone, both in
        the probe's view; None unless a planted output survives the cut."""
        r = rows[("sim-attack", c)]
        smp = samples_by_key(cfg["cohorts"]["sim-attack"], [r])[r["key"]]
        sim_t = transcript(smp, simulated=True)
        got = r2_undone(r["messages"], reader_messages(smp, ""), count)
        if got is None:
            count["dropped:no_str_replace_change"] += 1
            return None
        pm, rev = got
        fx, pl = fixed_actions(sim_t, smp, r, count), fixed_actions(sim_t, smp, r | {"messages": pm}, count)
        if fx is None or pl is None:
            return None
        fx, pl = as_probe_saw(fx, r["messages"], args.level, args.cut, tok, count, "fixed"), as_probe_saw(pl, pm, args.level, args.cut, tok, count, "planted")
        diffs = {j for j, (a, b) in enumerate(zip(fx or (), pl or ())) if a != b}
        if not fx or not pl or len(fx) != len(pl) or not diffs <= rev:  # same actions kept; differences only where R2 was undone
            count["dropped:copies_differ_beyond_the_plant"] += 1
            return None
        if not diffs:  # every planted output fell past the cut
            count["dropped:no_planted_output_inside_cut"] += 1
            return None
        count["planted_outputs_inside_cut"] += len(diffs)
        return sim_t, fx, pl

    if design.get("sample"):  # drawn from the cells the main design accepts (aligned pairs), so the control matches it
        sim_rows = [rows[("sim-attack", c)] for c in cells]
        sim_s = samples_by_key(cfg["cohorts"]["sim-attack"], sim_rows)
        accepted = [c for c, r in zip(cells, sim_rows) if fixed_actions(transcript(sim_s[r["key"]], True), sim_s[r["key"]], r, Counter()) is not None]
        if args.level or args.cut:  # the probe's view: the first seeded cells whose e1, fixed-sim and e2 views all pass, shared by both designs
            order = sorted(accepted)
            random.Random(design["sample_seed"]).shuffle(order)
            chosen = []
            for c in order:
                if len(chosen) == design["sample"]:
                    break
                ok = True
                if design.get("planted"):
                    ok = planted_view(c, Counter()) is not None
                    if ok:
                        chosen.append(c)
                    continue
                for arm, sim in (("real-attack-e1", False), ("real-attack-e2", False), ("sim-attack", True)):
                    r = rows[(arm, c)]
                    smp = samples_by_key(cfg["cohorts"][arm], [r])[r["key"]]
                    acts = fixed_actions(transcript(smp, True), smp, r, Counter()) if sim else tuple(transcript(smp, False).actions)
                    msgs = r["messages"]
                    v = None if acts is None else as_probe_saw(acts, msgs, args.level, args.cut, tok, Counter(), arm)
                    ok = ok and bool(v)
                if ok:
                    chosen.append(c)
            cells = sorted(chosen)
        else:
            cells = sorted(random.Random(design["sample_seed"]).sample(accepted, design["sample"]))
    real_rows = [rows[(design["real"], c)] for c in cells]
    other_rows = [rows[(design["other"], c)] for c in cells]
    real_s = samples_by_key(cfg["cohorts"][design["real"]], real_rows)
    other_s = samples_by_key(cfg["cohorts"][design["other"]], other_rows)

    count, pairs, kept = Counter(), [], []
    for c, rr, orow in zip(cells, real_rows, other_rows):
        if design.get("planted"):  # the judge should call the R2-undone copy simulated (correct = picks the fixed copy as real)
            v = planted_view(c, count)
            if v is None:
                continue
            sim_t, real_a, other = v
            pairs.append(Pair(id=sim_t.session_id, sim_id=sim_t.session_id, real_actions=real_a, sim_actions=other, source="attack"))
            kept.append((c, orow["key"], orow["key"] + "__r2_undone"))
            continue
        real_t = transcript(real_s[rr["key"]], simulated=False)
        if design["other"] == "sim-attack":
            sim_t = transcript(other_s[orow["key"]], simulated=True)
            other = fixed_actions(sim_t, other_s[orow["key"]], orow, count)
            if other is None:
                continue
        else:
            sim_t = transcript(other_s[orow["key"]], simulated=False)
            other = tuple(sim_t.actions)
        real_a = tuple(real_t.actions)
        if args.level or args.cut:  # the probe's view: the same cut and masking on both sides
            real_a = as_probe_saw(real_a, rr["messages"], args.level, args.cut, tok, count, "real")
            other = None if real_a is None else as_probe_saw(other, orow["messages"], args.level, args.cut, tok, count, "other")
            if real_a is None or other is None or not real_a or not other:
                count["dropped:probe_view"] += 1
                continue
        pairs.append(Pair(id=real_t.session_id, sim_id=sim_t.session_id, real_actions=real_a,
                          sim_actions=other, source="attack"))
        kept.append((c, rr["key"], orow["key"]))
    assert all(p.real_actions and p.sim_actions for p in pairs)
    chars = sum(len(build_pair_prompt(p, slot)) for p in pairs for slot in ("A", "B"))
    est = chars / cfg["chars_per_token"] * cfg["price_per_mtok_input"] / 1e6 \
        + 2 * len(pairs) * cfg["output_tokens_per_call"] * cfg["price_per_mtok_output"] / 1e6
    print(f"{args.design}: {len(cells)} cells, {len(pairs)} pairs kept, {dict(count)}; {2 * len(pairs)} calls, "
          f"{chars:,} prompt characters; estimated ${est:.0f} (stop at ${design['stop_usd']})", flush=True)
    assert est <= design["stop_usd"], "the estimate is over the stop; ask before running"
    if not args.run:
        return

    stamp = args.resume or datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    run_name = f"{cfg['run_name']}_{args.design}{f'_l{args.level}c{args.cut}' if args.level or args.cut else ''}_{stamp}"
    cache = HERE / ".run_cache" / f"judge_{args.design}{f'_l{args.level}c{args.cut}' if args.level or args.cut else ''}_{stamp}"
    cache.mkdir(parents=True, exist_ok=bool(args.resume))
    if not args.resume:
        with gzip.open(cache / "prompts.jsonl.gz", "wt") as fh:  # every rendered prompt, both orders, before any call
            for (c, rk, ok), p in zip(kept, pairs):
                for slot in ("A", "B"):
                    fh.write(json.dumps({"cell": c, "real_key": rk, "other_key": ok, "real_slot": slot,
                                         "prompt": build_pair_prompt(p, slot)}) + "\n")
    revisions = {name: subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=path).stdout.strip()
                 + ("-dirty" if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                               text=True, cwd=path).stdout.strip() else "")
                 for name, path in (("probe-shift", REPO), ("simulated-control-evals", REPO.parent / "simulated-control-evals"))}
    effort = design.get("reasoning_effort", cfg["reasoning_effort"])  # a design may override the effort (pilot_planted_xhigh)
    judge = JudgeClient(model=cfg["model"], settings=ModelSettings(reasoning_effort=effort, run_name=run_name))
    config, tally = DiscriminatorConfig(reasoning_effort=effort, max_concurrent=cfg["max_concurrent"]), DropTally()
    semaphore = asyncio.Semaphore(config.max_concurrent)
    done_path = cache / "done.jsonl"  # one line per judged pair, written after every batch (checkpoint)
    done = {json.loads(l)["cell"]: json.loads(l)["outcome"] for l in done_path.read_text().splitlines()} if done_path.exists() else {}
    todo = [(k, p) for k, p in zip(kept, pairs) if k[0] not in done]
    worst = {k[0]: sum(len(build_pair_prompt(p, s)) for s in ("A", "B")) / cfg["chars_per_token"] * cfg["price_per_mtok_input"] / 1e6
             + 2 * design.get("worst_output_tokens_per_call", cfg["worst_output_tokens_per_call"]) * cfg["price_per_mtok_output"] / 1e6
             for k, p in zip(kept, pairs)}

    async def seeded_pair(cell, p):
        """Both orders, sent one after the other in a seeded order per pair (Aswin 23:4x PT: order randomized per pair)."""
        first = random.Random(f"{design['order_seed']}:{cell}").choice("AB")
        got = []
        for slot in (first, "B" if first == "A" else "A"):
            async with semaphore:
                got.append(await _run_order(p, slot, judge, config, tally))
        orders = [o for o in got if o is not None]
        return PairScore(id=p.id, sim_id=p.sim_id, source=p.source, orders=orders) if orders else None
    stopped = None
    step = design.get("batch_pairs", cfg["batch_pairs"])  # the pilots check the budget between pairs (batch_pairs 1)
    for i in range(0, len(todo), step):
        batch = todo[i:i + step]
        spent, unpriced = run_cost(run_name)
        judged = len(done)
        # reserve each pair's worst case (its input plus worst_output_tokens_per_call per call) before sending it; no output
        # cap is settable, so this is a stopping threshold, not a hard cap (Codex 23:02)
        if unpriced or spent + sum(worst[k[0]] for k, _ in batch) > design["stop_usd"]:
            stopped = {"before_pairs": judged, "spent_usd": spent, "calls_without_recorded_cost": unpriced}
            print("stopping:", stopped, flush=True)
            break
        if design.get("order_seed"):
            outs = [await seeded_pair(k[0], p) for k, p in batch]
        else:
            outs = await asyncio.gather(*(discriminate_pair(p, judge, config, semaphore, tally) for _, p in batch))
        with open(done_path, "a") as fh:
            for (k, _), o in zip(batch, outs):
                done[k[0]] = o.as_dict() if o is not None else None
                fh.write(json.dumps({"cell": k[0], "outcome": done[k[0]]}) + "\n")
        spent, unpriced = run_cost(run_name)
        print(f"{len(done)}/{len(pairs)} pairs, ${spent:.2f} so far ({unpriced} calls without a recorded cost)", flush=True)

    result_rows, full = [], []
    for c, rk, ok in kept:
        if c not in done:
            continue
        d = done[c]
        full.append({"cell": c, "outcome": d})
        by_slot = {o["real_slot"]: o["correct"] for o in d["orders"]} if d else {}
        result_rows.append({"cell": c, "main_task": c.split(":")[0], "real_key": rk, "other_key": ok,
                            "accuracy": None if d is None else d["accuracy"], "real_in_a_correct": by_slot.get("A"),
                            "real_in_b_correct": by_slot.get("B")})
    with gzip.open(cache / "outcomes.jsonl.gz", "wt") as fh:  # the judge's full answers (text), kept out of results/
        fh.writelines(json.dumps(x) + "\n" for x in full)
    scored = [r for r in result_rows if r["accuracy"] is not None]
    groups = [r["main_task"] for r in scored]
    per_order = {slot: [float(r[f"real_in_{slot}_correct"]) for r in result_rows if r[f"real_in_{slot}_correct"] is not None]
                 for slot in ("a", "b")}
    spent, unpriced = run_cost(run_name)
    summary = {"config": cfg, "design": args.design, "run_name": run_name, "revisions": revisions, "stopped": stopped,
               "prompts": {"path": str((cache / "prompts.jsonl.gz").relative_to(HERE)),
                           "sha256": hashlib.sha256((cache / "prompts.jsonl.gz").read_bytes()).hexdigest()},
               "outcomes": {"path": str((cache / "outcomes.jsonl.gz").relative_to(HERE)),
                            "sha256": hashlib.sha256((cache / "outcomes.jsonl.gz").read_bytes()).hexdigest()},
               "cells": len(cells), "pairs_built": len(pairs), "alignment": dict(count), "n_scored": len(scored),
               "drops": tally.as_dict(),
               "accuracy": pairwise_accuracy_ci([r["accuracy"] for r in scored], groups, cfg["bootstrap"], cfg["seed"]),
               "per_order_accuracy": {f"real_in_{s}": {"n": len(v), "accuracy": sum(v) / len(v) if v else None} for s, v in per_order.items()},
               "spent_usd": spent, "calls_without_recorded_cost": unpriced,
               "spent_is_lower_bound": unpriced > 0}
    out = HERE / "results" / f"judge_{args.design}_{stamp}.json"
    with open(out, "x") as f:
        f.write(json.dumps(summary | {"pairs": result_rows}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    asyncio.run(main())
