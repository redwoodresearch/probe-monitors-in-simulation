"""On-policy title cells (RULE.md here), POST HOC, Seed-OSS-36B: q2's run_extract_seedcall_fresh.py (aswin/q2-fresh-honest @
f55ccfea) for one batch, "onpol": run_onpol_stage.py's staged Qwen3.5-27B runs (scrub level 0, #36's reader format) and its
8 repro-real-honest runs, keep flags from the staged Qwen3-32B position metas (QMETA, reader.segment_end_positions on the same
messages: what the Qwen3-32B extraction's metas hold, checked by check_onpol_qwen.py G3), so it need not wait for that
extraction. Changed: the app, the batch table, all 64 layers (the standing all-layers rule; the readouts and their rules are
unchanged), the shard count, and RULE.md as the plan file. Its docstring follows.

Fresh-task honest runs (PLAN.md here), Seed-OSS-36B: run_extract_seedcall_f11.py (aswin/q2-f11-insert @ bfc0bddd) for the
fresh batches (--batch smoke|parity|s1..s7: the staged run from stage_fresh_modal.py, its arms from the stage record; keep flags
from this batch's own Qwen3-32B dump, run_extract_qwen3_fresh.py, which must finish first). Changed: the app, the batch table,
work_items' "f10" branch (staged run and Qwen3-32B dump from the batch), the render gate's plan arms (the stage record's),
main's stage check and record names. No output_end reproduction here (new runs); the batch's reproduction runs are not read
by this copy. Its docstring follows.

F11 Seed-OSS-36B (PLAN_F11.md, POST HOC; Aswin via 0d: "Price and LAUNCH F11 as its OWN arm"): run_extract_seedcall_f10.py with
F10's names replaced by F11's (app, run name, input set, arms and counts, shard count, PLAN file, record prefixes), plus a remote
orchestrator (`orchestrate`, `--remote` with `modal run --detach`; 0d's 10-13 launch rule); it was:
F10 Seed-OSS-36B (../2026-10-13_q2_fixfull/PLAN_F10.md Amendment 8b, POST HOC; Aswin "only as its own arm" and "run all
this stuff on modal asap", via 0d): a copy of ../2026-10-13_q2_seed_callend/run_extract_seedcall.py (aswin/q2-seed-callend @
51be31a0; its readouts: output_end, output_end_mean, call_end raw / primary / Seed content-only, call_end_mean) for one input
set "f10": the 110 runs of 2026-10-13_q2_f10_inputs_v1 (stage_f10.py; arms sim-attack-fx-fullF9F10 74, sim-honest-fx-fullF9F10
36), each shard pinned by the stage record, keep flags from the original runs' Qwen3-32B metas as the "all" set. Changed: the
app, run name, arms and shard count, work_items' "f10" branch, and main's "f10" path (no output_end reference: F10 moves token
positions, so the reproduction is PLAN_F10.md's prefix gate, a separate script). The seedcall docstring follows.

Seed call_end (PLAN.md here), POST HOC: a copy of ../2026-10-13_q2_fixfull/run_extract_g3_fx.py (f9f2ea34) that also
locates and keeps call_end, the command readout, in Seed-OSS-36B's template, for one input set "all" (5,701 runs): #36's
real arms and as-simulated attack runs, the n437 as-simulated honest runs, fix-full's complete full arms and the F9 re-read
subset. Changed: the app, run name and constants; positions() locates every call's parameter spans in the rendering and
reads call_end (raw: the last token wholly inside the last parameter value that has one, else the function name; content-
only: not a non-string value, not a whitespace-only token, not the name), kept where Qwen3-32B kept the call's call_end;
work_items' "all" branch; the render job reads G3's stored metas and computes the render gate in the job (PLAN.md gate 1);
check counts every kept row and runs the output_end reproduction gate against G3's stored rows (PLAN.md gate 2); shard
stats record load and forward time; chunks of 10. Output_end is G3's, unchanged. The fx copy's docstring follows.

Fix-full (PLAN.md here), Seed-OSS-36B: a copy of ../2026-10-13_g3_followups/run_extract_g3_rep.py (at 1c277bfc, sha256
e793a0c3...) with one more input set, "fx": every arm staged by stage_fixfull.py (the per-fix variants F1-F8 and fix-full
of the 939 simulated attack runs, and fix-full of the 437 simulated honest runs; only runs a variant changes), each shard
checked against the staging record's sha256. Keep flags: the ORIGINAL run's Qwen3-32B meta (#36's dump for attack runs,
the v4 all-layers dump for honest runs), as G3's planted arm read its originals'. Its own app and run name; GPU fallback
["H200", "B200"] (Aswin 14:00 via 0d), each shard's GPU name in its stats; 48 shards. No forecast barrier (post hoc, after
G3's verdict). Everything else is G3's; the rep copy's docstring follows.

G3 follow-up (2), POST HOC (PLAN.md here; Aswin's "do both", 12:53 PT 10-13, via 0d): Seed-OSS-36B on fix939's repaired
runs. A copy of ../2026-10-13_g3_new_reader/run_extract_g3.py (sha256 e48933c2..., frozen in G3's forecast.json) with one
new arm, "rep": the 823 FIX runs of fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl, each read from fix939's re-rendered
inputs (2026-10-11_q3_fix939_inputs_v1, its single shard pinned by sha256), with its Qwen3-32B keep flags from fix939's own
Qwen3-32B dump (2026-10-11_q3_fix939_qwen3-32b_l48-63_v1, meta/sim-attack-note-sr-both), as the Olmo and Llama readers did.
Everything else is G3's: model, revision, rendering, readout (Amendment 1), the 7 layers, chunk layout, resume, local
orchestration, check. For d9's reader of this dump: run 2026-10-13_g3_rep_seed-oss-36b_v1 on G3's activation Volume, meta
arm "sim-attack-fix939"; a repaired run's key and file stem are its #36 sim-attack run's key and file stem, unchanged (so
meta/sim-attack-fix939/<stem>.json here pairs with meta/sim-attack/<stem>.json in 2026-10-13_g3_sim_seed-oss-36b_v1). The
other 116 cells are not re-extracted (their as-simulated runs are what fix939 left unchanged; PLAN.md (2)); the manifest
lists their keys. Changed lines: the app, the constants and arm "rep" below, work_items' "rep" branch, the render gate's
scope "rep", and main's --fix input; G3's docstring follows.

G3 steps 2 and 7 (RULE.md here): ByteDance-Seed/Seed-OSS-36B-Instruct, the screen's pick
(results/screen_2026-10-13_012449.json @ c2785c4d), reads #36's real runs, the planted e2 runs and P4-P6's 20 controls
(step 2); step 7's as-simulated runs only past the barrier. A copy of ../2026-10-12_q2_olmo_reader/run_extract_olmo.py (at
726c953e) with the staged-run inputs of ../2026-10-12_plantslip_transfer/q2_reply_extract_olmo.py (at a4fdf613). Changed:
  - the reader: the screened revision, on one H200 (72.3 GB of bf16 weights leave an H100 too little for 32k-token passes);
  - positions(): template-agnostic spans (each tool output replaced by a sentinel, located, checked verbatim in the real
    rendering, which is the screen's: prep() unchanged + apply_chat_template, and the whole rendering rebuilt). The
    readout is RULE.md Amendment 1's (dbb1cd03): the last non-whitespace content token, i.e. the last non-special token
    whose offsets have positive width, lie wholly inside the tool output's span, and pass Olmo's pinned bad() test
    (copied below). Beside it, the render counts R0 (the last token wholly inside the span, the RULE's first
    output_end), tokens crossing a span's end, and tool outputs quoting a special token's text. Rows are #36's: the
    outputs Qwen3-32B kept (C2), from its meta. Amendment 2 (0ce416b2): a required row with no eligible token is left
    out (the render records each run's left-out rows, by step and action, for the gate and the scoring), and the
    coverage gate (>= 99% per arm) is analyze_render_g3.py's;
  - layers: RULE.md's 7 (round_half_up(n k / 8), k = 2..7, plus n - 1), not all 64;
  - inputs (--which): real (#36's real-honest, real-attack-e1, real-attack-e2), planted (the 845 staged
    real-attack-e2-planted runs), controls (the 20 staged real-attack-e2-control runs), sim (#36's sim-attack: step 7).
    Before anything is spawned, an input manifest of every run id is written, and for real, planted and controls it
    asserts that no as-simulated run is present; every shard re-checks its items' arms;
  - orchestration is local: the entrypoint spawns each shard once, saves its call id at once, and waits (10-12: a
    restarted remote orchestrator re-spawned running shards);
  - sim: the barrier (--forecast-commit, --forecast-sha256) runs before any mode, pilots included: the step-6 commit must
    be an ancestor of HEAD, and forecast.json there and here must match the recorded sha256. The workers refuse
    simulated work unless their config carries that barrier record and forecast.json's bytes hashing to it;
  - missing rows are counted, never dropped silently: every output Qwen3-32B kept (#36's rows) must get a kept readout
    here; render, the shard metas and check count the ones that don't, and render and check exit 1 on any (RULE.md
    step 3: a missing row makes G3 unevaluable). Extraction exits 1 on a crashed shard or a non-finite value;
  - a launch refuses while another launch of this app is live (no second writer), and every shard checks that its
    items are exactly the manifest's run ids;
  - every mode refuses unless the committed final screen (Amendment 1) picked this model at this revision; extract
    (smoke test included) and check refuse without a passing render gate (analyze_render_g3.py) whose render record for
    this arm was made by this exact script on these exact run ids; check requires each meta's left-out rows to equal
    that render's, so only gated omissions pass.

  --mode render    (CPU, counts only): spans, readouts, kept rows and forward tokens per arm. Any render failure exits 1.
  --mode download  (CPU): the pinned weights into the shared HF cache Volume, once.
  --mode extract   (one H200 per shard): the residual stream after the 7 blocks at the kept readouts; #36's layout.
                   --test: the 4 longest transcripts on one shard under a separate run name (the smoke test).
  --mode check     (CPU, counts only): an extraction's files against its index and meta.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-13_g3_new_reader/run_extract_g3.py --mode <render|download|extract|check> --which <real|planted|controls|sim> \\
      --runs experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json \\
      --stage <q2_reply_stage_2026-10-12_135318.json> [--test] [--forecast-commit SHA --forecast-sha256 HEX]
"""
import bisect
import hashlib
import json
import os
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
# One app name per batch (this copy): the no-second-writer check is by app name, and the two batches write different runs.
APP = os.environ.get("ONPOL_SEED_APP") or "probe-shift_aswin-surya_sub-q3onpol_seed_kill-2026-11-13"
if APP not in ("probe-shift_aswin-surya_sub-q3onpol_seed_kill-2026-11-13", "probe-shift_aswin-surya_sub-q3onpol_seedrep_kill-2026-11-13"):
    raise ValueError(f"app name {APP!r} is not this job's")
ACTS = "probe-shift_aswin-surya_sub-la-activations"  # read-only: #36's dump (staged inputs, Qwen3-32B meta) and the staged plants
OUT = "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
DUMP = "2026-10-06_la_segment-end_qwen3-32b_v1"
STAGED = "2026-10-12_q2_plantslip_reply_{which}_inputs_v1"  # on ACTS, staged by q2_reply_stage.py
RUN = "2026-10-13_g3_{which}_seed-oss-36b_v1"
BATCHES = {"onpol": {"staged": "2026-10-14_q3onpol_q35own_inputs_v1", "qwen_run": "2026-10-14_q3onpol_q35own_qmeta_v1",
                     "run": "2026-10-14_q3onpol_q35own_seed-oss-36b_all_v1",
                     "arms": ("q35-real-attack-1", "q35-real-attack-2", "q35-sim-attack", "q35-sim-honest", "q35-real-honest-a",
                              "q35-real-honest-b", "repro-real-honest")},  # run_onpol_stage.py's arms (repro-xread-s1 is Qwen3-32B's gate only)
           "onpol_rep": {"staged": "2026-10-14_q3onpol_q35own_repair_full_inputs_v1", "qwen_run": "2026-10-14_q3onpol_q35own_repair_full_qmeta_v1",
                         "run": "2026-10-14_q3onpol_q35own_repair_full_seed-oss-36b_all_v1",
                         "arms": ("q35-sim-attack-fx-full", "q35-sim-honest-fx-full")}}  # RULE.md Amendment 1: Seed's cell's (seed_ff) fix-full repair
F10_INPUTS = "2026-10-13_q2_f11_inputs_v1"  # F11's set, on ACTS, staged by stage_f11.py (name kept)
FX_INPUTS = "2026-10-13_q2_fixfull_inputs_v2"  # on ACTS, staged by stage_fixfull.py
F9R_INPUTS = "2026-10-13_q2_fixfull_inputs_f9r_v1"  # on ACTS: F9's re-read subset (subset_f9.py)
HONEST_INPUTS = "2026-10-06_la_segment-end-v4_qwen3-32b_v1"  # on ACTS: the n437 as-simulated honest runs (arm sim-honest)
V4AL = "2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1"  # the honest runs' Qwen3-32B meta (meta/sim-honest)
ALL_ARMS = {a: None for b in BATCHES.values() for a in b["arms"]}  # the fresh batches' arms (counts come from each stage record)
G3_META = {"real-honest": "2026-10-13_g3_real_seed-oss-36b_v1", "real-attack-e1": "2026-10-13_g3_real_seed-oss-36b_v1",
           "real-attack-e2": "2026-10-13_g3_real_seed-oss-36b_v1", "sim-attack": "2026-10-13_g3_sim_seed-oss-36b_v1",
           "sim-attack-fx-full": "2026-10-13_fixfull_seed-oss-36b_v1", "sim-honest-fx-full": "2026-10-13_fixfull_seed-oss-36b_v1",
           "sim-attack-fx-fullF9": "2026-10-13_fixfull_f9_seed-oss-36b_v1", "sim-honest-fx-fullF9": "2026-10-13_fixfull_f9_seed-oss-36b_v1"}  # G3's runs on OUT
CALL_OPEN, CALL_CLOSE = "\n<seed:tool_call>\n<function=", "</function>\n</seed:tool_call>"
DROP = ("timeout number", "whitespace-only token", "function name")  # Seed content-only drops (PLAN.md Amendment 1.2)
FIXED_QWEN = ("0", "n")  # q1_eval.fixed_call_tokens for unscrubbed text: #36's content-only (the primary call set)
MODEL, REVISION, CTX, H, N_LAYERS = "ByteDance-Seed/Seed-OSS-36B-Instruct", "497f1dca95ebdec98e41d517b9f060ee753c902f", 65536, 5120, 64
LAYERS = list(range(N_LAYERS))  # all 64 (this copy; G3's 7 are 16, 24, 32, 40, 48, 56, 63)
assert len(LAYERS) == 64
REAL_ARMS = ("real-honest", "real-attack-e1", "real-attack-e2")
REP_LABEL = "sim-attack-fix939"
FIX_DUMP, FIX_ARM = "2026-10-11_q3_fix939_qwen3-32b_l48-63_v1", "sim-attack-note-sr-both"
FIX_INPUTS, FIX_INPUTS_SHA = "2026-10-11_q3_fix939_inputs_v1", "57cfbfbf1e4571690769605d2093d73edc93b958e06fed5e8efe8fa88587530d"
FIX_PIN = "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd"  # fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl
ARMS = {"real": REAL_ARMS, "planted": ("real-attack-e2-planted",), "controls": ("real-attack-e2-control",), "sim": ("sim-attack",), "rep": (REP_LABEL,)}
HONEST_RUNS_PIN = "f420ed2401dcc776cbbcfd414bbe9d6d1132a3bbd012b529a33aba7fa5ecd068"  # sim_honest_scores_2026-10-07_124717/runs.json
EXPECT = {"real-honest": 990, "real-attack-e1": 939, "real-attack-e2": 939, "real-attack-e2-planted": 845, "real-attack-e2-control": 20, "sim-attack": 939,
          REP_LABEL: 823}
GPU, CHUNK, TIMEOUT_S = ["H200", "B200"], 10, 3 * 3600
SHARDS = {"real": 24, "planted": 8, "controls": 1, "sim": 8, "rep": 8, "fx": 48, "all": 96, "f10": 24}  # f10 (= onpol here): 24
SENT = "G3SPAN{:06d}Q"
PIP_CPU = ["transformers==4.57.1", "jinja2==3.1.6", "huggingface_hub[hf_transfer]==0.35.3", "numpy"]
PIP_GPU = ["torch==2.8.0", "accelerate==1.10.1", *PIP_CPU]
FINAL_SCREEN = ("fe755dc9", "experiments/2026-10-13_g3_new_reader/results/screen_final_2026-10-13_020610.json",
                "f8192bb5065d138a7e4de037f424af8dc872643f1040e3648f9ecac76e04472d")  # Amendment 1's final pick
PIN = {"runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",
       "stage": "f1a68c90a5d26799925bda1c0d574728a547cde6c2ec3b76bfa18c3691c75157"}  # q2_reply_stage_2026-10-12_135318.json (b0a3b620)

ARMS["all"] = tuple(ALL_ARMS)
ARMS["f10"] = tuple(ALL_ARMS)
app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
hf_cache = modal.Volume.from_name(HF_CACHE)
out_vol = modal.Volume.from_name(OUT, version=2, create_if_missing=True)
out_ro = modal.Volume.from_name(OUT, version=2).read_only()  # the render reads G3's stored metas
cpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP_CPU).env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "ONPOL_SEED_APP": APP})
gpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP_GPU).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "ONPOL_SEED_APP": APP})


class Fail(Exception):
    def __init__(self, stage):
        super().__init__(stage); self.stage = stage


def prep(msgs):
    """Olmo's template takes string content and tool-call arguments as a dict; the staged messages may hold either form."""
    out = []
    for m in msgs:
        m = dict(m); c = m.get("content")
        m["content"] = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
        if m.get("tool_calls"):
            m["tool_calls"] = [{**tc, "function": {**(tc.get("function") or {}), "arguments": (lambda a: json.loads(a) if isinstance(a, str) else (a or {}))(
                (tc.get("function") or {}).get("arguments"))}} for tc in m["tool_calls"]]
        out.append(m)
    return out


def bad(t: str) -> bool:
    """R2 skips a token that is whitespace-only or has a newline anywhere in it."""
    return t.strip() == "" or "\n" in t


def call_blocks(tok, m):
    """The text Seed's own template writes for an assistant message's calls: the message rendered alone with its content
    and reasoning removed (`<seed:bos>assistant` + the calls + `<seed:eos>`), the two ends cut off."""
    s = tok.apply_chat_template([{"role": "assistant", "content": "", "tool_calls": m["tool_calls"]}], tokenize=False)
    head, tail = "<seed:bos>assistant", "<seed:eos>"
    if not (s.startswith(head) and s.endswith(tail)):
        raise Fail("a call block renders unexpectedly")
    return s[len(head):-len(tail)]


def call_spans(blocks, m):
    """Per call of message m, in order: (name, the name's [start, end) in blocks, [(parameter, is a string, value start,
    value end)] in written order). Walks the template's fixed structure with the message's own names and values, so a
    value quoting the template's tags cannot shift it; a non-string value (written `| string`) ends at the next
    `</parameter>\\n`."""
    out, p = [], 0
    for tc in m["tool_calls"]:
        fn = tc.get("function") or tc
        name, args = fn.get("name") or "", fn.get("arguments") or {}
        if not blocks.startswith(CALL_OPEN + name + ">\n", p):
            raise Fail("a call's header not where the template writes it")
        n0 = p + len(CALL_OPEN); n1 = n0 + len(name); p = n1 + 2
        spans = []
        for k, v in args.items():
            ph = f"<parameter={k}>"
            if not blocks.startswith(ph, p):
                raise Fail("a call's parameter not where the template writes it")
            v0 = p + len(ph)
            if isinstance(v, str):
                v1 = v0 + len(v)
                if blocks[v0:v1] != v:
                    raise Fail("a string value not verbatim")
            else:
                if not (v is None or isinstance(v, (bool, int, float))) and "</parameter>" in json.dumps(v):
                    raise Fail("a non-scalar value quoting the closing tag")
                v1 = blocks.find("</parameter>\n", v0)
                if v1 < 0:
                    raise Fail("a non-string value without its closing tag")
            if not blocks.startswith("</parameter>\n", v1):
                raise Fail("a value without its closing tag")
            spans.append((k, isinstance(v, str), v0, v1)); p = v1 + len("</parameter>\n")
        if not blocks.startswith(CALL_CLOSE, p):
            raise Fail("a call's close not where the template writes it")
        p += len(CALL_CLOSE)
        out.append((name, (n0, n1), spans))
    if p != len(blocks):
        raise Fail("text after the last call")
    return out


def positions(tok, msgs, q, special, special_text=(), qc=None):
    """One staged transcript under the reader's template. q: Qwen3-32B's output_end positions, {"step,action": [kept, tool]};
    qc: its call_end positions, {"step,action": [kept, #36 content-only]}. Returns ids, the meta positions (every call's call_end, kept where
    Qwen3-32B kept it and the readout is below CTX; every output's output_end, kept under C2 and below CTX) and per output
    (qwen kept, readout index, readout is a newline token, a token crosses the span's end, the output ends in a newline),
    plus per call the content-only drop reason and the dropped readouts' safe token strings. Raises Fail(stage) on a span
    that can't be located verbatim. Required rows (Qwen3-32B-kept outputs) without a kept readout here are listed by
    "step,action" in two kinds: no eligible token in the span, and any other reason (a readout at or past the context
    length, or a Qwen3-32B position with no span here). Output_end is G3's (the fx copy's), unchanged."""
    import re
    msgs = prep(msgs)
    qc = qc or {}
    text = tok.apply_chat_template(msgs, tokenize=False)
    if "G3SPAN" in text:
        raise Fail("sentinel text in the transcript")
    tool_ix = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
    quoting = sum(any(t in msgs[i]["content"] for t in special_text) for i in tool_ix)  # counted, not refused (Amendment 1 skips specials)
    sent = [dict(m) for m in msgs]
    for k, i in enumerate(tool_ix):
        sent[i]["content"] = SENT.format(k)
    stext = tok.apply_chat_template(sent, tokenize=False)
    spans, calls, step, action, shift, pos, k = [], [], -1, 0, 0, 0, 0
    span_of = {}  # message index -> its tool output's [a, b)
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            step, action = step + 1, 0
            calls += [(step, j, (tc.get("function") or {}).get("name"), i) for j, tc in enumerate(m.get("tool_calls") or [])]
        elif m["role"] == "tool":
            s_ = SENT.format(k); j = stext.find(s_, pos)
            if j < 0 or stext.count(s_) != 1:
                raise Fail("a tool output not located")
            a = j + shift; b = a + len(m["content"])
            if text[a:b] != m["content"]:
                raise Fail("a tool output not verbatim")
            spans.append((step, action, a, b)); span_of[i] = (a, b); action += 1; k += 1
            shift += len(m["content"]) - len(s_); pos = j + len(s_)
    if len(text) != len(stext) + shift:
        raise Fail("the renderings differ outside the tool outputs")
    rebuilt, prev = [], 0
    for k, i in enumerate(tool_ix):  # the sentinel rendering with the real contents put back must be the real rendering
        j = stext.index(SENT.format(k), prev); rebuilt += [stext[prev:j], msgs[i]["content"]]; prev = j + len(SENT.format(k))
    if "".join(rebuilt) + stext[prev:] != text:
        raise Fail("the renderings differ outside the tool outputs")
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]; starts, ends = [s for s, _ in enc["offset_mapping"]], [e for _, e in enc["offset_mapping"]]
    if any(starts[i] < starts[i - 1] or ends[i] < ends[i - 1] for i in range(1, len(ids))):
        raise Fail("offsets not monotonic")
    tt = lambda i: text[starts[i]:ends[i]]  # noqa: E731
    # call_end: each assistant message's calls located in its own region (between the neighbouring tool outputs), then the
    # last token wholly inside the last parameter value that has one (else the name), non-special, positive width
    cpos, dropped, cmean = [], [], []
    by_msg = {}
    for s, j, name, i in calls:
        by_msg.setdefault(i, []).append((s, j, name))
    for i, cl in by_msg.items():
        lo = max((b for mi, (_, b) in span_of.items() if mi < i), default=0)
        hi = min((a for mi, (a, _) in span_of.items() if mi > i), default=len(text))
        blocks = call_blocks(tok, msgs[i])
        needle = blocks + "<seed:eos>"
        at = text.find(needle, lo, hi)
        if at < 0 or text.find(needle, at + 1, hi) >= 0:
            raise Fail("an assistant message's calls not located once in its region")
        parsed = call_spans(blocks, msgs[i])
        if len(parsed) != len(cl) or any(p_[0] != (nm or "") for p_, (_, _, nm) in zip(parsed, cl)):
            raise Fail("located calls differ from the message's")
        for (s, j, name), (nm, (n0, n1), pspans) in zip(cl, parsed):
            cands = [(pk, isstr, at + v0, at + v1) for pk, isstr, v0, v1 in reversed(pspans)] + [(None, None, at + n0, at + n1)]
            r, pk_, isstr_ = None, None, None
            sp = None
            for pk, isstr, c0, c1 in cands:
                i0, i1 = bisect.bisect_left(starts, c0), bisect.bisect_right(ends, c1)
                r = next((t for t in range(i1 - 1, i0 - 1, -1) if ends[t] > starts[t] and ids[t] not in special), None)
                if r is not None:
                    pk_, isstr_, sp = pk, isstr, (i0, i1)
                    break
            drop = None if r is None else (DROP[2] if pk_ is None else DROP[0] if (pk_ == "timeout" and not isstr_) else DROP[1] if tt(r).strip() == "" else None)
            qv = qc.get(f"{s},{j}") or [False, False]
            kept = bool(qv[0]) and r is not None and r < CTX
            cpos.append({"kind": "call_end", "step": s, "action": j, "tool": name, "token": r, "kept": kept, "token_text": None,
                         "param": pk_, "content_only": kept and drop is None, "qwen_content_only": bool(qv[0] and qv[1]),
                         "primary": kept and bool(qv[1]), "drop": drop})
            # call_end_mean, d9's (mean_final over call_end's span): the read value's tokens lying entirely inside it
            kcm = kept and sp is not None and sp[1] <= CTX
            cmean.append({"kind": "call_end_mean", "step": s, "action": j, "tool": name, "token": sp[1] - 1 if kcm else None, "kept": kcm,
                          "token_text": None, "span": list(sp) if kcm else None, "param": pk_, "content_only": kcm and drop is None,
                          "primary": kcm and bool(qv[1])})
            if kept and drop is not None:  # safe strings only: whitespace, a tool name, a value with no letters (else a placeholder)
                t_ = tt(r)
                safe = (repr(t_) if drop == DROP[1] else name if drop == DROP[2]
                        else t_ if (re.fullmatch(r"[\W\d_]*", t_) or t_ in ("True", "False", "None")) else "<has letters>")
                dropped.append((drop, pk_, safe, ids[r]))
    if len(cpos) != len(calls):
        raise Fail("a call not located")
    out = sorted(cpos + cmean, key=lambda p: (p["step"], p["action"], p["kind"]))
    per, no_content = [], set()
    for s, a_, a, b in spans:
        i0, i1 = bisect.bisect_left(starts, a), bisect.bisect_right(ends, b)  # the tokens lying entirely inside [a, b) (and zero-width ones)
        r0 = next((i for i in range(i1 - 1, i0 - 1, -1) if ends[i] > starts[i]), None)  # the last token wholly inside
        r2 = next((i for i in range(i1 - 1, i0 - 1, -1) if ends[i] > starts[i] and ids[i] not in special and not bad(tt(i))), None)  # the readout
        crosses = i1 < len(ids) and starts[i1] < b < ends[i1]
        qk = q.get(f"{s},{a_}")
        kept = bool(qk and qk[0]) and r2 is not None and r2 < CTX
        if r2 is None:
            no_content.add(f"{s},{a_}")
        out.append({"kind": "output_end", "step": s, "action": a_, "tool": qk[1] if qk else None, "token": r2, "kept": kept, "token_text": None})
        # output_end_mean, d9's averaged readout (run_extract_all_layers.py @ 82c87660, mean_final; reader.py's span): the plain
        # mean over the tokens lying entirely inside the output's text, [i0, i1); kept where output_end is kept and the whole
        # span lies below the context (an output reaching past it is dropped, not clipped, as d9's extractor does)
        km = kept and i1 <= CTX
        out.append({"kind": "output_end_mean", "step": s, "action": a_, "tool": qk[1] if qk else None, "token": i1 - 1 if km else None, "kept": km,
                    "token_text": None, "span": [i0, i1] if km else None})
        if km and not (i0 < i1 and starts[i0] >= a and ends[i1 - 1] <= b and i0 <= r2 < i1):
            raise Fail("an empty mean span, or one outside the output's characters")
        per.append((qk, [(r, None if r is None else tt(r)) for r in (r0, r2)], crosses, text[a:b].endswith("\n")))
    got = {f"{p['step']},{p['action']}" for p in out if p["kind"] == "output_end" and p["kept"]}
    lost = [k_ for k_, v in q.items() if v[0] and k_ not in got]  # Qwen-kept outputs with no kept readout here
    cgot = {f"{p['step']},{p['action']}" for p in cpos if p["kept"]}
    return {"ids": ids, "n_tokens": len(ids), "positions": out, "outputs": per,
            "qwen_kept_no_content": sum(k_ in no_content for k_ in lost), "qwen_kept_missing_other": sum(k_ not in no_content for k_ in lost),
            "left_out": sorted(lost, key=lambda k_: tuple(map(int, k_.split(",")))), "left_out_other": sorted(k_ for k_ in lost if k_ not in no_content),
            "required": sum(1 for v in q.values() if v[0]), "outputs_quoting_special": quoting,
            "calls": len(cpos), "calls_qwen_kept": sum(1 for v in qc.values() if v[0]), "calls_qwen_kept_lost": sorted(k_ for k_, v in qc.items() if v[0] and k_ not in cgot),
            "calls_qwen_content_only": sum(1 for v in qc.values() if v[0] and v[1]),
            "calls_without_qwen_position": sum(f"{p['step']},{p['action']}" not in qc for p in cpos), "dropped": dropped}


def special_ids(tok):
    return set(tok.all_special_ids) | {i for i, t in tok.added_tokens_decoder.items() if t.special}


def special_texts(tok):
    return tuple(sorted({t.content for t in tok.added_tokens_decoder.values() if t.special and t.content}))


def staged(run_dir, by_key):
    out = {}
    for shard in sorted((Path("/acts") / run_dir / "inputs").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line); out[it["key"] if by_key else (it["arm"], it["key"])] = it
    return out


def qsum(meta):  # Qwen3-32B's output_end positions: [kept, tool] by "step,action"
    return {f"{p['step']},{p['action']}": [bool(p["kept"]), p.get("tool")] for p in meta["positions"] if p["kind"] == "output_end"}


def qcalls(meta):  # Qwen3-32B's call_end positions: [kept, #36 content-only (token not in fixed_call_tokens)] by "step,action"
    return {f"{p['step']},{p['action']}": [bool(p["kept"]), bool(p["kept"]) and p.get("token_text") is not None and p.get("token_text") not in FIXED_QWEN]
            for p in meta["positions"] if p["kind"] == "call_end"}


def qcalls_no_text(meta):  # kept call_end positions without a token_text (the primary set needs it): must be 0
    return sum(1 for p in meta["positions"] if p["kind"] == "call_end" and p["kept"] and p.get("token_text") is None)


def msg_sha(it) -> str:
    return hashlib.sha256(json.dumps(it["messages"], sort_keys=True).encode()).hexdigest()


def filename(key: str) -> str:  # the extractors' file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def work_items(which, cohort, shas):
    """(arm, filename, staged item, the (original) run's Qwen3-32B meta path). real and sim: #36's staged runs of the
    cohort; planted and controls: q2_reply_stage.py's staged runs, each shard file checked against the staging record.
    Every item's arm must be one of `which`'s arms (an as-simulated run outside sim fails the job)."""
    if which in ("real", "sim"):
        base = staged(DUMP, False)
        items = [(r["arm"], r["filename"], base[(r["arm"], r["key"])], Path("/acts") / DUMP / "acts" / r["arm"] / f"{r['filename']}.json") for r in cohort]
    elif which == "fx":  # stage_fixfull.py's staged arms, each shard pinned by the staging record; the originals' keep flags
        d = Path("/acts") / FX_INPUTS / "inputs"
        files = sorted(d.glob("shard_*.jsonl"))
        if {f.name for f in files} != set(shas):
            raise Fail("the staged shard files differ from the staging record's list")
        items = []
        for f in files:
            if hashlib.sha256(f.read_bytes()).hexdigest() != shas[f.name]:
                raise Fail("a staged shard differs from the staging record")
            for line in f.read_text().splitlines():
                x = json.loads(line)
                q = (Path("/acts") / DUMP / "acts" / "sim-attack" / f"{filename(x['key'])}.json" if x["arm"].startswith("sim-attack-fx-")
                     else Path("/acts") / V4AL / "meta" / "sim-honest" / f"{filename(x['key'])}.json")
                items.append((x["arm"], filename(x["key"]), x, q))
        want = {(r["arm"], r["key"]) for r in cohort}
        if {(a, x["key"]) for a, _, x, _ in items} != want:
            raise Fail("the staged runs differ from the manifest's")
    elif which == "f10":  # F10's staged arms, each shard pinned by stage_f10.py's record; the originals' keep flags
        items = []
        for name, rec in sorted(shas["f10"].items()):
            b = (Path("/acts") / shas["staged"] / "inputs" / name).read_bytes()
            if hashlib.sha256(b).hexdigest() != rec["sha256"]:
                raise Fail("a staged shard differs from its staging record")
            for line in b.decode().splitlines():
                x = json.loads(line)
                if x["arm"] != rec["arm"] or x["arm"] not in shas["arms"]:
                    raise Fail("a staged shard holds another arm")
                q = Path("/acts") / shas["qwen_run"] / "meta" / x["arm"] / f"{filename(x['key'])}.json"  # this batch's own Qwen3-32B dump
                items.append((x["arm"], filename(x["key"]), x, q))
        if {(a, x["key"]) for a, _, x, _ in items} != {(r["arm"], r["key"]) for r in cohort}:
            raise Fail("the staged runs differ from the manifest's")
    elif which == "all":  # PLAN.md: #36's real and as-simulated attack runs, the n437 honest runs, fix-full's full arms and F9's subset
        base = staged(DUMP, False)
        items = [(r["arm"], r["filename"], base[(r["arm"], r["key"])], Path("/acts") / DUMP / "acts" / r["arm"] / f"{r['filename']}.json")
                 for r in cohort if r["arm"] in REAL_ARMS + ("sim-attack",)]
        hon = staged(HONEST_INPUTS, False)
        items += [("sim-honest", r["filename"], hon[("sim-honest", r["key"])], Path("/acts") / V4AL / "meta" / "sim-honest" / f"{r['filename']}.json")
                  for r in cohort if r["arm"] == "sim-honest"]
        for src, run_dir in (("fx", FX_INPUTS), ("f9r", F9R_INPUTS)):
            for name, rec in sorted(shas[src].items()):
                b = (Path("/acts") / run_dir / "inputs" / name).read_bytes()
                if hashlib.sha256(b).hexdigest() != rec["sha256"]:
                    raise Fail("a staged shard differs from its staging record")
                for line in b.decode().splitlines():
                    x = json.loads(line)
                    if x["arm"] != rec["arm"]:
                        raise Fail("a staged shard holds another arm")
                    q = (Path("/acts") / DUMP / "acts" / "sim-attack" / f"{filename(x['key'])}.json" if x["arm"].startswith("sim-attack-fx-")
                         else Path("/acts") / V4AL / "meta" / "sim-honest" / f"{filename(x['key'])}.json")
                    items.append((x["arm"], filename(x["key"]), x, q))
        if {(a, x["key"]) for a, _, x, _ in items} != {(r["arm"], r["key"]) for r in cohort}:
            raise Fail("the staged runs differ from the manifest's")
    elif which == "rep":  # fix939's re-rendered inputs, its pinned single shard; Qwen3-32B keep flags from fix939's own dump
        f = Path("/acts") / FIX_INPUTS / "inputs" / "shard_0.jsonl"
        if sorted(p.name for p in f.parent.glob("shard_*.jsonl")) != ["shard_0.jsonl"] or hashlib.sha256(f.read_bytes()).hexdigest() != FIX_INPUTS_SHA:
            raise Fail("fix939's inputs are not the pinned ones")
        fixed = {}
        for line in f.read_text().splitlines():
            x = json.loads(line)
            if x["key"] in fixed:
                raise Fail("a fix939 input appears twice")
            fixed[x["key"]] = x
        items = [(REP_LABEL, r["filename"], fixed[r["key"]], Path("/acts") / FIX_DUMP / "meta" / FIX_ARM / f"{r['filename']}.json") for r in cohort]
    else:
        d = Path("/acts") / STAGED.format(which=which) / "inputs"
        files = sorted(d.glob("shard_*.jsonl"))
        if {f"{STAGED.format(which=which)}/{f.name}" for f in files} != set(shas):
            raise Fail("the staged shard files differ from the staging record's list")
        for f in files:
            if hashlib.sha256(f.read_bytes()).hexdigest() != shas[f"{STAGED.format(which=which)}/{f.name}"]:
                raise Fail("a staged shard differs from the staging record")
        it = staged(STAGED.format(which=which), False)
        items = [(arm, filename(k), x, Path("/acts") / DUMP / "acts" / x["orig_arm"] / f"{filename(k)}.json")
                 for (arm, k), x in sorted(it.items()) if arm in ARMS[which]]
    if (which != "fx" and any(arm not in ARMS[which] for arm, *_ in items)) or (which not in ("sim", "rep", "fx", "all", "f10") and any("sim" in arm for arm, *_ in items)):
        raise Fail("an item outside this job's arms")
    if len({(arm, x["key"]) for arm, _, x, _ in items}) != len(items):
        raise Fail("duplicate run ids")
    return items


def ids_sha(pairs) -> str:
    return hashlib.sha256(json.dumps(sorted([list(p) for p in pairs])).encode()).hexdigest()


def sim_gate(cfg):
    """Workers' half of the barrier: simulated work needs the barrier record and forecast.json's bytes that hash to it."""
    if cfg["which"] == "sim":
        b = cfg.get("barrier") or {}
        if not (b.get("step-6 commit") and b.get("forecast.json sha256") and cfg.get("forecast.json")
                and hashlib.sha256(cfg["forecast.json"].encode()).hexdigest() == b["forecast.json sha256"]):
            raise Fail("simulated work without the step-7 barrier")


# ---------- render (CPU, counts only) ----------
def _init():
    from transformers import AutoTokenizer
    global TOK, SPECIAL, SPECIAL_TEXT
    TOK = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    SPECIAL, SPECIAL_TEXT = special_ids(TOK), special_texts(TOK)


def count_one(item):
    """Counts for one run, and its per-run eligibility (Amendment 2: its required rows and the left-out ones by step and
    action; positions only, no text). A failure is counted by its stage and type only (an exception may quote content),
    never raised. Seed call_end: call counts by set (raw, primary = #36's content-only calls, Seed content-only), their
    disagreements, the dropped readouts' safe strings, and output_end indices against the reference meta (G3's run, or for
    as-simulated honest runs fix-full's run when the messages are byte-identical)."""
    from collections import Counter
    label, _, it, qpath, gpath = item
    c = Counter(); c[(label, "runs")] += 1
    c[(label, "runs with attachment:// in the staged messages")] += "attachment://" in json.dumps(it["messages"])
    try:
        meta = json.loads(qpath.read_text())
        r = positions(TOK, it["messages"], qsum(meta), SPECIAL, SPECIAL_TEXT, qcalls(meta))
    except Exception as e:
        c[(label, f"render failure: {getattr(e, 'stage', 'render')} ({type(e).__name__})")] += 1
        return c, {"arm": label, "key": it["key"], "render failure": True}, []
    c[(label, "Qwen3-32B-kept call_end without token_text")] += qcalls_no_text(meta)
    kept_idx = [p["token"] for p in r["positions"] if p["kept"]]  # a mean row's token is its span's last token
    fwd = max(kept_idx) + 1 if kept_idx else 0
    for kind in ("output_end_mean", "call_end_mean"):
        c[(label, f"{kind} kept")] += sum(p["kept"] for p in r["positions"] if p["kind"] == kind)
        c[(label, f"{kind} tokens averaged")] += sum(p["span"][1] - p["span"][0] for p in r["positions"] if p["kept"] and p["kind"] == kind)
    oe_k = {(p["step"], p["action"]) for p in r["positions"] if p["kind"] == "output_end" and p["kept"]}
    om_k = {(p["step"], p["action"]) for p in r["positions"] if p["kind"] == "output_end_mean" and p["kept"]}
    c[(label, "output_end kept but output_end_mean dropped (span past the context)")] += len(oe_k - om_k)
    c[(label, "output_end_mean kept without output_end")] += len(om_k - oe_k)
    ck = [p["token"] for p in r["positions"] if p["kept"] and p["kind"] == "call_end"]
    fwd_call = max(ck) + 1 if ck else 0
    c[(label, "Qwen3-32B-kept outputs with no readout: no non-whitespace content token")] += r["qwen_kept_no_content"]
    c[(label, "tool outputs quoting a special token's text")] += r["outputs_quoting_special"]
    c[(label, "Qwen3-32B-kept outputs with no readout: any other reason (missing rows)")] += r["qwen_kept_missing_other"]
    c[(label, "reader tokens")] += r["n_tokens"]; c[(label, "forward tokens (to the last kept readout)")] += fwd
    c[(label, "forward tokens to the last kept call_end (a command-rows-only cut)")] += fwd_call
    c[(label, "forward tokens squared, summed over runs")] += fwd * fwd  # for the attention term of the re-price
    c[(label, "runs with no kept readout")] += not kept_idx
    c[(label, "outputs Qwen3-32B kept")] += r["required"]  # every required row, with or without a span here
    calls = [p for p in r["positions"] if p["kind"] == "call_end"]
    c[(label, "calls")] += len(calls)
    c[(label, "calls Qwen3-32B kept")] += r["calls_qwen_kept"]
    c[(label, "calls Qwen3-32B kept with no kept Seed readout")] += len(r["calls_qwen_kept_lost"])
    c[(label, "calls with no Qwen3-32B position")] += r["calls_without_qwen_position"]
    c[(label, "call_end kept (raw)")] += sum(p["kept"] for p in calls)
    c[(label, "#36 content-only calls (Qwen3-32B)")] += r["calls_qwen_content_only"]
    c[(label, "call_end kept, primary set (#36's content-only calls)")] += sum(p["primary"] for p in calls)
    c[(label, "call_end kept, Seed content-only")] += sum(p["content_only"] for p in calls)
    c[(label, "primary but not Seed content-only")] += sum(p["primary"] and not p["content_only"] for p in calls)
    c[(label, "Seed content-only but not primary")] += sum(p["content_only"] and not p["primary"] for p in calls)
    for p in calls:
        if p["kept"] and p["drop"]:
            c[(label, f"Seed drop: {p['drop']}")] += 1
        if p["kept"] and p["param"] is None:
            c[(label, "call_end read at the function name")] += 1
    c[(label, "runs with 0 primary-set call_end rows")] += not any(p["primary"] for p in calls)
    c[(label, "runs with 0 Seed content-only call_end rows")] += not any(p["content_only"] for p in calls)
    oe = [(p["step"], p["action"], p["token"], p["kept"]) for p in r["positions"] if p["kind"] == "output_end"]
    if gpath is None:
        c[(label, "output_end reference: none (changed by the fixes, or no G3 run)")] += 1
    elif not gpath.exists():
        c[(label, "output_end reference: G3 meta missing")] += 1
    else:
        g = json.loads(gpath.read_text())
        go = [(p["step"], p["action"], p["token"], p["kept"]) for p in g["positions"] if p["kind"] == "output_end"]
        c[(label, "output_end reference: compared")] += 1
        c[(label, "output_end reference: index mismatch (runs)")] += go != oe
    for qk, rules, crosses, ends_nl in r["outputs"]:
        c[(label, "tool outputs")] += 1
        c[(label, "tool outputs with no Qwen3-32B position")] += qk is None
        if not (qk and qk[0]):
            continue
        r2 = rules[1][0]
        c[(label, "of those, kept (R2 readout below the context length)")] += r2 is not None and r2 < CTX
        c[(label, "of those, R2 at index 32,768 or beyond (kept if below 65,536)")] += r2 is not None and r2 >= 32768
    return c, {"arm": label, "key": it["key"], "required": r["required"], "left out": r["left_out"], "left out, other reason": r["left_out_other"],
               "calls Qwen3-32B kept, lost": r["calls_qwen_kept_lost"], "forward tokens": fwd, "reference": None if gpath is None else str(gpath)}, \
        [(label,) + d for d in r["dropped"]]


@app.function(image=cpu_image, volumes={"/acts": acts, "/out": out_ro}, cpu=32, memory=65536, timeout=3600, retries=0)
def render(cfg: dict, cohort: list[dict], shas: dict) -> dict:
    try:
        sim_gate(cfg)
        return _render(cfg["which"], cohort, shas)
    except Exception as e:  # nothing from the exception (which may quote staged text) crosses the boundary
        raise RuntimeError(f"render job failed: {type(e).__name__}") from None


def refs(items):
    """Per item, its output_end reference meta on OUT: G3's run for the arm; for an as-simulated honest run, fix-full's honest
    run when the staged messages are byte-identical (PLAN.md Amendment 1.4); else None."""
    fx_hon = {x["key"]: msg_sha(x) for a, _, x, _ in items if a == "sim-honest-fx-full"}
    out = []
    for a, stem, x, q in items:
        if a in G3_META:
            g = Path("/out") / G3_META[a] / "meta" / a / f"{stem}.json"
        elif a == "sim-honest" and fx_hon.get(x["key"]) == msg_sha(x):
            g = Path("/out") / G3_META["sim-honest-fx-full"] / "meta" / "sim-honest-fx-full" / f"{stem}.json"
        else:
            g = None
        out.append((a, stem, x, q, g))
    return out


def _render(which, cohort, shas):
    from collections import Counter
    from multiprocessing import Pool

    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(MODEL, revision=REVISION)  # download once; the pool's processes load from the cache
    items = refs(work_items(which, cohort, shas))
    total, runs, dropped = Counter(), [], Counter()
    with Pool(32, initializer=_init) as pool:
        for cnt, run, drops in pool.imap(count_one, items, chunksize=4):
            total.update(cnt); runs.append(run); dropped.update(drops)
    n = lambda a, k: total.get((a, k), 0)  # noqa: E731
    arms = sorted({a for a, _ in total})
    checks = {
        "render failures": sum(v for (a, k), v in total.items() if k.startswith("render failure")),
        "runs with attachment://": sum(n(a, "runs with attachment:// in the staged messages") for a in arms),
        "Qwen3-32B-kept call_end without token_text": sum(n(a, "Qwen3-32B-kept call_end without token_text") for a in arms),
        "output_end index mismatches vs reference (runs)": sum(n(a, "output_end reference: index mismatch (runs)") for a in arms),
        "G3 reference metas missing (runs)": sum(n(a, "output_end reference: G3 meta missing") for a in arms),
        "runs by arm vs plan": {a: [n(a, "runs"), shas["arms"].get(a)] for a in arms},
        "call_end coverage (kept / Qwen3-32B-kept calls)": {a: round(n(a, "call_end kept (raw)") / max(n(a, "calls Qwen3-32B kept"), 1), 5) for a in arms},
        "primary coverage (kept / #36 content-only calls)": {a: round(n(a, "call_end kept, primary set (#36's content-only calls)") / max(n(a, "#36 content-only calls (Qwen3-32B)"), 1), 5) for a in arms},
        "output_end_mean kept without output_end": sum(n(a, "output_end_mean kept without output_end") for a in arms),
        "output_end_mean dropped where output_end kept (span past the context)": {a: n(a, "output_end kept but output_end_mean dropped (span past the context)") for a in arms},
        "output_end coverage ((required - left out) / required)": {a: round(1 - (n(a, "Qwen3-32B-kept outputs with no readout: no non-whitespace content token")
                                                                                 + n(a, "Qwen3-32B-kept outputs with no readout: any other reason (missing rows)")) / max(n(a, "outputs Qwen3-32B kept"), 1), 5) for a in arms},
    }
    ok = (checks["render failures"] == 0 and checks["runs with attachment://"] == 0 and checks["Qwen3-32B-kept call_end without token_text"] == 0
          and checks["output_end index mismatches vs reference (runs)"] == 0 and checks["G3 reference metas missing (runs)"] == 0
          and all(v[0] == v[1] for v in checks["runs by arm vs plan"].values()) and set(arms) == set(shas["arms"])
          and all(v >= 0.99 for v in checks["call_end coverage (kept / Qwen3-32B-kept calls)"].values())
          and all(v >= 0.99 for v in checks["primary coverage (kept / #36 content-only calls)"].values())
          and all(v >= 0.99 for v in checks["output_end coverage ((required - left out) / required)"].values())
          and checks["output_end_mean kept without output_end"] == 0)
    return {"rows": [{"arm": a, "count": k, "n": int(v)} for (a, k), v in sorted(total.items())],
            "dropped readouts (arm, reason, parameter, safe token string, token id)": [list(k) + [v] for k, v in sorted(dropped.items(), key=lambda kv: -kv[1])][:400],
            "gate": {"pass": bool(ok), "checks": checks},
            "runs": sorted(runs, key=lambda x: (x["arm"], x["key"]))}


# ---------- download (CPU) ----------
@app.function(image=cpu_image, volumes={"/hf": hf_cache}, cpu=8, memory=16384, timeout=3600, retries=0)
def download() -> dict:
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL, revision=REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    hf_cache.commit()
    return {"path": path, "bytes": sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())}


# ---------- extract (GPU) ----------
# No automatic retry (a retry once started a second container on a running shard: two writers, double cost). After a
# crash, relaunch: the resume skips every transcript already done.
@app.function(image=gpu_image, gpu=GPU, memory=65536, volumes={"/hf": hf_cache, "/out": out_vol, "/acts": acts}, timeout=TIMEOUT_S, retries=0)
def extract_shard(cfg: dict, shard: int, cohort: list[dict], shas: dict) -> list[dict]:
    t_fn0 = time.time()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sim_gate(cfg)
    items = work_items(cfg["which"], cohort, shas)
    if ids_sha((arm, x["key"]) for arm, _, x, _ in items) != cfg["ids sha256"]:
        raise Fail("the items differ from the manifest's run ids")
    if cfg.get("assign"):  # all: the entrypoint's longest-first balanced assignment from the render's forward tokens
        mine = [it for it in items if cfg["assign"][f"{it[0]}|{it[2]['key']}"] == shard][: cfg.get("limit_items") or None]
    else:
        items.sort(key=lambda it: (-(it[2].get("tokens") or 0), it[0], it[2]["key"]))  # longest first (Qwen's count), dealt round-robin
        mine = items[shard::cfg["shards"]][: cfg.get("limit_items") or None]
    if cfg["which"] in ("fx", "all", "f10"):  # one arm per chunk (G3's frozen _score refuses a chunk holding another arm): arm by arm, longest first
        mine.sort(key=lambda it: (it[0], -(it[2].get("tokens") or 0), it[2]["key"]))
    run_dir = Path("/out") / cfg["run"]
    if all((run_dir / "meta" / label / f"{stem}.json").exists() for label, stem, _, _ in mine) and (run_dir / "stats" / f"shard_{shard}.json").exists():
        return [json.loads((run_dir / "stats" / f"shard_{shard}.json").read_text())]  # done: no weights loaded
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    special, special_text = special_ids(tok), special_texts(tok)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    t_loaded = time.time()
    base = lm.model
    if lm.config.num_hidden_layers != N_LAYERS or lm.config.hidden_size != H:
        raise ValueError(f"unexpected shape: {lm.config.num_hidden_layers} layers x {lm.config.hidden_size}")
    layers = LAYERS
    for l in layers:
        (run_dir / "by_layer" / f"L{l}").mkdir(parents=True, exist_ok=True)
    (run_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (run_dir / "chunk_markers").touch()  # always a fresh, marker-mode run (no legacy chunks)
    names = {p.stem for l in layers for p in (run_dir / "by_layer" / f"L{l}").glob(f"shard{shard:03d}_chunk*.npy")}
    names |= {p.stem for p in (run_dir / "chunks").glob(f"shard{shard:03d}_chunk*.json")}
    for name in sorted(n for n in names if not (run_dir / "chunks" / f"{n}.done").exists()):  # drop incomplete chunks
        idx = run_dir / "chunks" / f"{name}.json"
        for t in (json.loads(idx.read_text())["transcripts"] if idx.exists() else []):
            meta = run_dir / "meta" / t["arm"] / f"{t['filename']}.json"
            if meta.exists() and json.loads(meta.read_text()).get("chunk") == name:
                meta.unlink()
        for l in layers:
            (run_dir / "by_layer" / f"L{l}" / f"{name}.npy").unlink(missing_ok=True)
        idx.unlink(missing_ok=True)
        print(f"shard {shard}: dropped incomplete chunk {name} before resuming", flush=True)
    out_vol.commit()
    chunk = len(list((run_dir / "by_layer" / f"L{layers[0]}").glob(f"shard{shard:03d}_chunk*.npy")))
    buf, buf_meta, timing = {l: [] for l in layers}, [], []

    def flush():
        nonlocal chunk
        if not buf_meta:
            return
        name = f"shard{shard:03d}_chunk{chunk:03d}"
        nonfinite = 0
        for l in layers:
            x = np.concatenate(buf[l]) if buf[l] else np.zeros((0, H), np.float16)
            nonfinite += int((~np.isfinite(x)).sum())
            tmp = run_dir / "by_layer" / f"L{l}" / f".{name}.npy.tmp"
            with open(tmp, "wb") as fh:
                np.save(fh, x)
            os.replace(tmp, run_dir / "by_layer" / f"L{l}" / f"{name}.npy")
            buf[l].clear()
        index, start = [], 0
        for m in buf_meta:
            index.append({"arm": m["arm"], "key": m["key"], "filename": m["filename"], "row_start": start, "n_rows": m["n_rows"]})
            m |= {"chunk": name, "row_start": start}
            start += m["n_rows"]
        tmp = run_dir / "chunks" / f".{name}.json.tmp"
        tmp.write_text(json.dumps({"layers": layers, "rows": start, "nonfinite": nonfinite, "transcripts": index}))
        os.replace(tmp, run_dir / "chunks" / f"{name}.json")
        for m in buf_meta:
            d = run_dir / "meta" / m["arm"]
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f".{m['filename']}.json.tmp"
            tmp.write_text(json.dumps(m))
            os.replace(tmp, d / f"{m['filename']}.json")
        (run_dir / "chunks" / f"{name}.done").touch()
        buf_meta.clear()
        chunk += 1
        out_vol.commit()

    t_all = time.time()
    for n, (label, stem, it, qpath) in enumerate(mine, 1):
        if (run_dir / "meta" / label / f"{stem}.json").exists():
            continue
        if buf_meta and buf_meta[-1]["arm"] != label:  # fx: close the chunk at an arm boundary
            flush()
        qm = json.loads(qpath.read_text())
        r = positions(tok, it["messages"], qsum(qm), special, special_text, qcalls(qm))
        rows = [p for p in r["positions"] if p["kept"]]
        t0 = time.time()
        if rows:
            pts = [j for j, p in enumerate(rows) if not p["kind"].endswith("_mean")]
            mns = [(j, p["span"]) for j, p in enumerate(rows) if p["kind"].endswith("_mean")]
            if any(not (0 <= a < b <= CTX) for _, (a, b) in mns):
                raise ValueError("an empty or out-of-range content span")
            last = max([rows[j]["token"] for j in pts] + [b - 1 for _, (a, b) in mns])
            idx = torch.tensor([rows[j]["token"] for j in pts], device="cuda", dtype=torch.long)
            pts_t = torch.tensor(pts, device="cuda", dtype=torch.long)
            got = {}

            def hook_for(layer):
                def hook(module, args, output):
                    h = (output[0] if isinstance(output, tuple) else output)[0]
                    arr = torch.empty((len(rows), H), device="cuda", dtype=torch.float32)
                    if pts:
                        arr[pts_t] = h.index_select(0, idx).float()
                    for j, (a, b) in mns:
                        arr[j] = h[a:b].float().mean(0)
                    got[layer] = arr.to(torch.float16).cpu().numpy()
                return hook
            handles = [base.layers[l].register_forward_hook(hook_for(l)) for l in layers]
            try:
                with torch.inference_mode():
                    base(input_ids=torch.tensor([r["ids"][: last + 1]], device="cuda"), use_cache=False)
            finally:
                for h_ in handles:
                    h_.remove()
            for l in layers:
                buf[l].append(got[l])
            timing.append([last + 1, round(time.time() - t0, 3)])
        row = iter(range(len(rows)))
        positions_ = [p | {"row": next(row) if p["kept"] else None} for p in r["positions"]]
        buf_meta.append({"key": it["key"], "arm": label, "filename": stem,
                         **{k: it.get(k) for k in ("cohort", "file", "sample_id", "epoch", "environment", "main_task", "side_task", "cell", "orig_arm")},
                         "n_tokens": r["n_tokens"], "forward_tokens": (last + 1) if rows else 0,
                         "qwen_kept_no_content": r["qwen_kept_no_content"], "qwen_kept_missing_other": r["qwen_kept_missing_other"],
                         "left_out": r["left_out"],
                         "n_rows": len(rows), "positions": positions_})
        if len(buf_meta) == cfg["chunk_size"]:
            flush()
            print(f"shard {shard}: {n}/{len(mine)}", flush=True)
    flush()
    stats = []
    for p in sorted((run_dir / "chunks").glob(f"shard{shard:03d}_chunk*.json")):
        c = json.loads(p.read_text())
        live = [t for t in c["transcripts"] if (f := run_dir / "meta" / t["arm"] / f"{t['filename']}.json").exists()
                and json.loads(f.read_text())["chunk"] == p.stem]
        stats.append({"chunk": p.stem, "transcripts": len(live), "rows": sum(t["n_rows"] for t in live), "nonfinite": c["nonfinite"]})
    summary = {"shard": shard, "transcripts": len(mine), "chunks": stats, "seconds": round(time.time() - t_all), "forward [tokens, seconds]": timing,
               "gpu": torch.cuda.get_device_name(0), "load seconds (function start to weights on the GPU)": round(t_loaded - t_fn0, 1),
               "function seconds": round(time.time() - t_fn0, 1)}
    (run_dir / "stats").mkdir(exist_ok=True)
    (run_dir / "stats" / f"shard_{shard}.json").write_text(json.dumps(summary))
    out_vol.commit()
    return [summary]


@app.function(image=cpu_image, cpu=2, memory=8192, volumes={"/out": out_vol}, timeout=TIMEOUT_S + 1800, retries=0)
def orchestrate(cfg: dict, cohort: list[dict], shas: dict) -> dict:
    """F11 addition (0d's 10-13 rule: every launch `modal run --detach` with a remote orchestrator): spawns the shards from
    inside Modal, so they survive the laptop client, waits for each, and writes the summary to the run's folder."""
    calls = [extract_shard.spawn(cfg, i, cohort, shas) for i in range(cfg["shards"])]
    stats = []
    for shard, call in enumerate(calls):
        try:
            stats.extend(call.get())
        except Exception as e:  # noqa: BLE001 — recorded by type only (a message may quote staged text)
            stats.append({"shard": shard, "crashed": type(e).__name__})
    ok = [x for x in stats if "chunks" in x]
    summary = {"shards": stats, "n_ok": len(ok), "n_crashed": len(stats) - len(ok), "transcripts": sum(c["transcripts"] for x in ok for c in x["chunks"]),
               "rows": sum(c["rows"] for x in ok for c in x["chunks"]), "nonfinite": sum(c["nonfinite"] for x in ok for c in x["chunks"]),
               "function calls": [c.object_id for c in calls]}
    d = Path("/out") / cfg["run"] / "stats"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    out_vol.commit()
    return summary


@app.function(image=cpu_image, cpu=8, memory=32768, volumes={"/out": out_vol}, timeout=3600, retries=0)
def check(run: str, expected: dict, left_out: dict | None = None, repro: dict | None = None) -> dict:
    """Counts only. Meta files by arm against `expected` (arm -> transcripts); each meta's kept readouts (call_end and
    output_end), rows 0..n-1 in order, and its chunk and row_start against that chunk's index; per chunk, its done marker,
    the 7 layer files of shape (rows, 5120), and non-finite values (every layer, from the index; the first and last layer
    recounted here); call_end kept per arm (raw, primary, Seed content-only); forward timing and load time from the shard
    stats; and the reproduction gate (PLAN.md gate 2): output_end rows of the sampled runs against the reference run's rows
    at the same (step, action), every layer, by this run's GPU type."""
    import numpy as np
    from collections import Counter
    out_vol.reload()
    d = Path("/out") / run
    metas = Counter(); kept = 0; problems = Counter(); rows_total = 0; nonfinite_index = 0; nonfinite_recount = 0; no_content = Counter()
    calls = Counter()
    index = {}
    for idx in sorted((d / "chunks").glob("shard*_chunk*.json")):
        c = json.loads(idx.read_text())
        problems["chunk without its done marker"] += not (d / "chunks" / f"{idx.stem}.done").exists()
        problems["chunk layers != the plan's"] += c["layers"] != LAYERS
        problems["chunk index rows != its transcripts' rows"] += c["rows"] != sum(t["n_rows"] for t in c["transcripts"])
        problems["chunk index offsets not cumulative from 0"] += [t["row_start"] for t in c["transcripts"]] != [sum(u["n_rows"] for u in c["transcripts"][:j]) for j in range(len(c["transcripts"]))]
        for t in c["transcripts"]:
            problems["duplicate index entry"] += (t["arm"], t["filename"], idx.stem) in index
            index[(t["arm"], t["filename"], idx.stem)] = (t["row_start"], t["n_rows"])
        nonfinite_index += c["nonfinite"]
        for l in LAYERS:
            x = np.load(d / "by_layer" / f"L{l}" / f"{idx.stem}.npy", mmap_mode="r")
            problems["layer file with the wrong shape"] += x.shape != (c["rows"], H)
            if l in (LAYERS[0], LAYERS[-1]):
                nonfinite_recount += int((~np.isfinite(np.asarray(x))).sum())
        rows_total += c["rows"]
    for arm_dir in sorted((d / "meta").iterdir()):
        for f in arm_dir.glob("*.json"):
            m = json.loads(f.read_text()); metas[arm_dir.name] += 1
            if left_out is not None:  # only the render's (gated) left-out rows may be missing
                problems["meta left_out differs from the gated render's"] += m.get("left_out") != left_out.get(arm_dir.name, {}).get(m["key"])
            kr = [p["row"] for p in m["positions"] if p["kept"]]; kept += len(kr)
            ce = [p for p in m["positions"] if p["kind"] == "call_end" and p["kept"]]
            calls[(arm_dir.name, "call_end kept (raw)")] += len(ce)
            calls[(arm_dir.name, "call_end kept, primary set")] += sum(bool(p.get("primary")) for p in ce)
            calls[(arm_dir.name, "call_end kept, Seed content-only")] += sum(bool(p.get("content_only")) for p in ce)
            calls[(arm_dir.name, "output_end kept")] += sum(1 for p in m["positions"] if p["kind"] == "output_end" and p["kept"])
            calls[(arm_dir.name, "output_end_mean kept")] += sum(1 for p in m["positions"] if p["kind"] == "output_end_mean" and p["kept"])
            calls[(arm_dir.name, "call_end_mean kept")] += sum(1 for p in m["positions"] if p["kind"] == "call_end_mean" and p["kept"])
            problems["meta rows != kept readouts"] += m["n_rows"] != len(kr)
            problems["kept rows not 0..n-1 in order"] += kr != list(range(len(kr)))
            problems["unkept position with a row"] += any(p.get("row") is not None for p in m["positions"] if not p["kept"])
            problems["meta without its left-out counts"] += "qwen_kept_no_content" not in m or "qwen_kept_missing_other" not in m
            no_content[arm_dir.name] += m.get("qwen_kept_no_content", 0) + m.get("qwen_kept_missing_other", 0)
            problems["meta not in its chunk's index, or row_start/n_rows differ"] += index.pop((arm_dir.name, m["filename"], m["chunk"]), None) != (m["row_start"], m["n_rows"])
    problems["index entries with no live meta"] += len(index)
    problems["kept readouts (meta) != rows (chunks)"] += kept != rows_total
    problems["transcripts by arm != expected"] += (sum(metas.values()) != expected["total"]) if "total" in expected else (dict(metas) != expected)
    shard_stats = {int(f.stem.split("_")[1]): json.loads(f.read_text()) for f in sorted((d / "stats").glob("shard_*.json"))}
    timing = [t for s in shard_stats.values() for t in s.get("forward [tokens, seconds]", [])]
    toks, secs = sum(t[0] for t in timing), sum(t[1] for t in timing)
    util = [{"shard": i, "gpu": s.get("gpu"), "load s": s.get("load seconds (function start to weights on the GPU)"), "function s": s.get("function seconds"),
             "forward s": round(sum(t[1] for t in s.get("forward [tokens, seconds]", [])), 1),
             "tokens per forward second": round(sum(t[0] for t in s.get("forward [tokens, seconds]", [])) / max(sum(t[1] for t in s.get("forward [tokens, seconds]", [])), 1e-9))}
            for i, s in sorted(shard_stats.items())]

    def rows_of(rd, m):  # {(step, action): {layer: vector}} for kept output_end
        want = {(p["step"], p["action"]): m["row_start"] + p["row"] for p in m["positions"] if p["kind"] == "output_end" and p["kept"]}
        got = {}
        for l in LAYERS:
            x = np.load(Path("/out") / rd / "by_layer" / f"L{l}" / f"{m['chunk']}.npy", mmap_mode="r")
            for k_, row in want.items():
                got.setdefault(k_, {})[l] = np.asarray(x[row], dtype=np.float32)
        return got
    rep = {}
    for arm, lst in (repro or {}).items():
        for stem, ref_run, ref_arm in lst:
            m = json.loads((d / "meta" / arm / f"{stem}.json").read_text())
            g = json.loads((Path("/out") / ref_run / "meta" / ref_arm / f"{stem}.json").read_text())
            a_, b_ = rows_of(run, m), rows_of(ref_run, g)
            gpu = shard_stats.get(int(m["chunk"][5:8]), {}).get("gpu")
            r_ = rep.setdefault(gpu, {"runs": 0, "rows": 0, "rows missing in the reference": 0, "bit-exact rows": 0, "min cosine": 1.0, "max abs diff": 0.0})
            r_["runs"] += 1
            for k_, by_l in a_.items():
                if k_ not in b_:
                    r_["rows missing in the reference"] += 1; continue
                exact = True
                for l in LAYERS:
                    u, v = by_l[l], b_[k_][l]
                    cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
                    r_["min cosine"] = min(r_["min cosine"], cos); r_["max abs diff"] = max(r_["max abs diff"], float(np.abs(u - v).max()))
                    exact &= bool(np.array_equal(u, v))
                r_["rows"] += 1; r_["bit-exact rows"] += exact
    repro_pass = bool(rep) and all(v["min cosine"] >= 0.999 and v["rows missing in the reference"] == 0 and v["rows"] > 0 for v in rep.values())
    return {"run": run, "meta files by arm": dict(metas), "kept readouts (meta)": kept, "rows (chunks)": rows_total,
            "non-finite values: every layer (chunk indexes)": nonfinite_index, "non-finite values: first and last layer recounted": nonfinite_recount,
            "expected transcripts by arm": expected, "problems": {k: v for k, v in problems.items()},
            "required rows left out (Amendment 2), by arm (compare with the render gate's)": dict(no_content),
            "kept rows by arm and kind": {f"{a} | {k}": v for (a, k), v in sorted(calls.items())},
            "forward passes": len(timing), "forward tokens": toks, "forward seconds": round(secs, 1),
            "seconds per 1,000 forward tokens": round(1000 * secs / toks, 4) if toks else None,
            "longest forward pass [tokens, seconds]": max(timing) if timing else None,
            "per shard (gpu, load, function and forward seconds, tokens per forward second)": util,
            "reproduction (output_end vs reference rows, by this run's GPU type)": rep, "reproduction gate": "pass" if repro_pass else "FAIL"}


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, check=check)


def barrier(commit: str, sha: str) -> dict:
    """RULE.md step 7: before anything simulated, the step-6 commit is an ancestor of HEAD and forecast.json matches the
    recorded sha256, both at that commit and here."""
    fc = HERE / "forecast.json"
    rel = str(fc.resolve().relative_to(Path(git("rev-parse", "--show-toplevel").stdout.strip())))
    if not commit or not sha or git("merge-base", "--is-ancestor", commit, "HEAD", check=False).returncode:
        raise SystemExit("barrier: the step-6 commit is not an ancestor of HEAD")
    at = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=HERE, capture_output=True, check=True).stdout
    if hashlib.sha256(at).hexdigest() != sha or hashlib.sha256(fc.read_bytes()).hexdigest() != sha:
        raise SystemExit("barrier: forecast.json does not match the recorded sha256")
    return {"step-6 commit": git("rev-parse", commit).stdout.strip(), "forecast.json sha256": sha}


def final_pick():
    blob = subprocess.run(["git", "show", f"{FINAL_SCREEN[0]}:{FINAL_SCREEN[1]}"], cwd=HERE, capture_output=True, check=True).stdout
    if hashlib.sha256(blob).hexdigest() != FINAL_SCREEN[2]:
        raise SystemExit("the final screen record differs from its pin")
    rec = json.loads(blob)
    rev = next((r["revision"] for r in rec["candidates"] if r["repo"] == rec["pick"]), None)
    if rec["pick"] != MODEL or rev != REVISION:
        raise SystemExit(f"the final screen picked {rec['pick']} @ {rev}, not this script's {MODEL} @ {REVISION}")


def render_gate(path: str, ids, src: str):
    """extract and check need a render record of this arm set whose in-job gate passed, made by this script's bytes (at the
    record's commit) on these run ids with this reader."""
    if not path:
        raise SystemExit("--gate <seedcall_render record>: extraction needs a passing render gate")
    rp = Path(path)
    r = json.loads(rp.read_text())
    if not (r.get("gate") or {}).get("pass"):
        raise SystemExit("the render gate did not pass")
    rel = str(Path(__file__).resolve().relative_to(Path(git("rev-parse", "--show-toplevel").stdout.strip())))
    then = subprocess.run(["git", "show", f"{r['git']}:{rel}"], cwd=HERE, capture_output=True, check=True).stdout
    if hashlib.sha256(then).hexdigest() != src or r["ids sha256"] != ids_sha(ids) or r["reader"]["revision"] != REVISION:
        raise SystemExit("the render record was made by other code, on other runs, or with another reader")
    if git("ls-files", "--error-unmatch", str(rp.resolve()), check=False).returncode:
        raise SystemExit("commit the render record before using it as the gate")
    return {"render record": rp.name, "render record sha256": hashlib.sha256(rp.read_bytes()).hexdigest()}, r


def balance(r, n):
    """Longest-first greedy assignment of runs to n shards by the render's forward tokens (with the attention term)."""
    import heapq
    w = sorted(((x.get("forward tokens") or 0, x["arm"], x["key"]) for x in r["runs"]), key=lambda z: (-z[0], z[1], z[2]))
    heap = [(0.0, i) for i in range(n)]
    out = {}
    for f, a, k in w:
        load, i = heapq.heappop(heap)
        out[f"{a}|{k}"] = i
        heapq.heappush(heap, (load + f + f * f / 2e5, i))  # attention grows with length; 2e5 tokens per unit is a rough weight
    return out, max(l for l, _ in heap)


@app.local_entrypoint()
def main(mode: str, runs: str, stage: str, batch: str = "", honest_runs: str = "", stage_f9: str = "", which: str = "f10", test: bool = False, gate: str = "", shards: int = 0, remote: bool = False):
    """--which all --runs <#36 runs.json> --honest-runs <sim_honest_scores runs.json> --stage <fix-full stage record>
    --stage-f9 <stage_f9r record> --mode render|extract|check [--gate <render record>]."""
    if mode not in ("render", "extract", "check") or which != "f10" or batch not in BATCHES:
        raise SystemExit("this copy runs only --which f10 (--mode render|extract|check)")
    for f in (Path(__file__).resolve(), HERE / "RULE.md"):
        if git("ls-files", "--error-unmatch", str(f), check=False).returncode or git("diff", "--quiet", "HEAD", "--", str(f), check=False).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    final_pick()
    head = git("rev-parse", "HEAD").stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    if hashlib.sha256(Path(runs).read_bytes()).hexdigest() != PIN["runs.json"]:
        raise SystemExit(f"{Path(runs).name} is not the pinned runs.json")
    st = json.loads(Path(stage).read_text())
    BT = BATCHES[batch]
    if st.get("mode") != "stage" or st.get("staged") != BT["staged"] or not st.get("gates pass"):
        raise SystemExit("not stage_fresh_modal.py's passing stage record of this batch")
    plan_arms = {a: len(st["runs by arm"][a]) for a in BT["arms"]}
    shas = {"f10": {Path(k).name: {"arm": v["arm"], "sha256": v["sha256"]} for k, v in st["shards_files"].items() if v["arm"] in plan_arms},
            "staged": BT["staged"], "qwen_run": BT["qwen_run"], "arms": plan_arms}  # the reproduction arm's shard is not read
    r36 = json.loads(Path(runs).read_text())
    ids = sorted({(a, k) for a in plan_arms for k in st["runs by arm"][a]})
    by_arm = {a: sum(1 for b, _ in ids if b == a) for a in plan_arms}
    if by_arm != plan_arms:
        raise SystemExit(f"runs by arm {by_arm} differ from the stage record's {plan_arms}")
    cohort = [{"arm": a, "key": k, "filename": filename(k)} for a, k in ids]
    stems = {x["key"]: x["filename"] for x in r36}
    if any(filename(k) != stems[k] for a, k in ids if a.startswith("sim-attack")):
        raise SystemExit("a #36 run's file stem differs from runs.json's")
    manifest = {"made": stamp, "git": head, "which": which, "runs": [{"arm": a, "key": k} for a, k in ids], "by arm": by_arm,
                "POST HOC": f"fresh-task honest runs, batch {batch} (PLAN.md), Seed-OSS-36B with the seedcall readouts",
                "inputs sha256": {"#36 runs.json": PIN["runs.json"], Path(stage).name: hashlib.sha256(Path(stage).read_bytes()).hexdigest(),
                                  **{f"{BT['staged']}/inputs/{k}": v["sha256"] for k, v in sorted(shas["f10"].items())}}}
    with open(HERE / "results" / f"freshseed_{batch}_manifest_{stamp}.json", "x") as fh:
        fh.write(json.dumps(manifest, indent=1) + "\n")
    src = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run = BT["run"] + (f"_test_{src[:8]}" if test else "")
    common = {"plan": "PLAN.md (this folder), POST HOC", "git": head, "which": which, "reader": {"model": MODEL, "revision": REVISION, "context": CTX, "layers": LAYERS},
              "inputs sha256": manifest["inputs sha256"], "barrier": None, "ids sha256": ids_sha(ids), "forecast.json": None}

    def write(name, out):
        with open(dest := HERE / "results" / f"{name}_{stamp}.json", "x") as fh:
            fh.write(json.dumps(out, indent=1) + "\n")
        print(dest)
    if mode == "render":
        out = {k: v for k, v in common.items() if k != "forecast.json"} | render.remote(common, cohort, shas)
        write(f"freshseed_{batch}_render", out)
        print(json.dumps(out["gate"], indent=1))
        if not out["gate"]["pass"]:
            raise SystemExit("the render gate FAILED: see the record")
        return
    gated, rr = render_gate(gate, ids, src)
    if mode == "check":
        lo = {}
        for x in rr["runs"]:
            lo.setdefault(x["arm"], {})[x["key"]] = x["left out"]
        import random
        rng = random.Random(20261013)
        del rng  # F10 runs have no position-identical reference: the reproduction is PLAN_F10.md's prefix gate (separate)
        out = {"git": head, "render gate": gated} | check.remote(run, {"total": 4} if test else by_arm, lo, {})
        out["reproduction gate"] = "not applicable here (PLAN_F10.md prefix gate, a separate script)"
        write(f"extract_check_{run}", out)
        if (any(out["problems"].values()) or out["non-finite values: every layer (chunk indexes)"] or out["non-finite values: first and last layer recounted"]):
            raise SystemExit("check failed: see the record")
        return
    rows_ = json.loads(subprocess.run(["uv", "run", "--no-project", "--with", "modal==1.6.1", "modal", "app", "list", "--json"],
                                      capture_output=True, text=True, check=True, timeout=180).stdout)
    if live := [r["app_id"] for r in rows_ if r["description"] == APP and r["app_id"] != app.app_id and r["state"] != "stopped"]:
        raise SystemExit(f"another launch of {APP} is live ({live}): stop it first (no second writer)")
    n_sh = 1 if test else (shards or SHARDS[which])
    assign, heaviest = balance(rr, n_sh)
    cfg = {"run": run, "shards": n_sh, "chunk_size": 2 if test else CHUNK, "limit_items": 4 if test else None, "source sha256": src,
           "modal": {"app": APP, "gpu": GPU, "pip": PIP_GPU}, "render gate": gated, "assign": assign,
           "heaviest shard (forward tokens + attention term)": round(heaviest)} | common
    try:
        stored = json.loads(b"".join(out_vol.read_file(f"{cfg['run']}/config.json")))
    except (FileNotFoundError, modal.exception.NotFoundError):
        stored = None
    if stored is not None and {k: v for k, v in stored.items() if k not in ("git", "plan")} != {k: v for k, v in cfg.items() if k not in ("git", "plan")}:
        raise SystemExit(f"{cfg['run']} exists with different settings or code (source sha256); use a new run name")
    import io
    with out_vol.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(json.dumps(cfg, indent=1).encode()), f"{cfg['run']}/config.json")
    rec_p = HERE / "results" / f"extract_launch_{cfg['run']}_{stamp}.json"
    rec = {"launched": stamp, "app": app.app_id, "function calls": []} | {k: v for k, v in cfg.items() if k != "assign"}
    if remote:  # F11: the remote orchestrator (with `modal run --detach`): the shards are spawned from inside Modal
        call = orchestrate.spawn(cfg, cohort, shas)
        rec["orchestrator function call"] = call.object_id
        tmp = rec_p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(rec, indent=1) + "\n"); tmp.replace(rec_p)
        print(f"{cfg['run']}: orchestrating {cfg['shards']} shards on Modal (function call {call.object_id}); summary at {cfg['run']}/stats/summary.json", flush=True)
        return
    calls = []
    for i in range(cfg["shards"]):  # spawned once, from here only; each call id is saved as soon as it exists
        calls.append(extract_shard.spawn(cfg, i, cohort, shas))
        rec["function calls"].append(calls[-1].object_id)
        tmp = rec_p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(rec, indent=1) + "\n"); tmp.replace(rec_p)
    print(f"{cfg['run']}: {len(calls)} shards spawned from the local entrypoint (app {app.app_id})", flush=True)
    stats = []
    for shard, call in enumerate(calls):  # the wait loop, run locally
        for _ in range(240):
            try:
                stats.extend(call.get())
                break
            except (socket.gaierror, ConnectionError, TimeoutError, modal.exception.ConnectionError, modal.exception.ServiceError):
                time.sleep(30)
            except Exception as e:  # the shard itself failed: recorded by type only (a message may quote staged text)
                stats.append({"shard": shard, "crashed": type(e).__name__})
                break
        else:
            stats.append({"shard": shard, "unreachable": True})
    ok = [x for x in stats if "chunks" in x]
    summary = {"shards": stats, "n_ok": len(ok), "n_crashed": len(stats) - len(ok), "transcripts": sum(c["transcripts"] for x in ok for c in x["chunks"]),
               "rows": sum(c["rows"] for x in ok for c in x["chunks"]), "nonfinite": sum(c["nonfinite"] for x in ok for c in x["chunks"])}
    write(f"extract_summary_{cfg['run']}", summary)
    if summary["n_crashed"] or summary["nonfinite"]:
        raise SystemExit("extraction failed: a crashed or unreachable shard, or a non-finite value (relaunch resumes)")
