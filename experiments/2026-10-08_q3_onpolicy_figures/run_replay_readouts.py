"""Figure 3, step 1: the rule-1 readout on the replay tokens, with its checks. Tokenizer only (no model); Modal CPU.

Local part: from a Phase 2 run folder's launch.json and .eval logs, an index of every policy call in the chosen arms:
the acting model's events only (the simulator's own model events carry request ids too), each event's request id
(`x-irid`), arm, task, turn, and per emitted tool call its name, argument count, rule 1's
command parameter (`reader.command_param`), the parameter value's sha256 and the Q2 label (`calls.label_call`,
fingerprint eb014307fd1aab31; the terminal submit is excluded, amendment 7, and any other tool without a rule raises).
No text leaves the laptop beyond these hashes and labels. `reader.command_param` is #163's (a9f49c2): run with that
branch's src on PYTHONPATH until it merges.
Remote part (CPU, the request-log Volume and the HF cache): for each indexed request, the logged prompt and completion
token ids, and:
- the adapter: rebuild the completion from each token's exact bytes (checked against the tokenizer's decode; never
  re-encode), parse its XML tool calls, require the same call names and parameter names as the eval (no repeats), find the command value verbatim (raw, else stripped, up to
  `\n</parameter>`) with the eval's value hash, and take the last token whose whole span lies inside the value;
  readout position = len(prompt ids) + its index;
- context epochs: within a run, whether each request's prompt starts with the previous request's prompt plus
  completion, and the token volume of one forward pass per call vs one per epoch.
Every mismatch is counted by kind; nothing is scored. Output: counts and per-call positions (ids and flags only).
`--segment-ends`: also the two per-segment readouts, each its own row with `kind` (the rule-1 rows are kind "command"):
- call_end: per tool call, the last token wholly inside its last parameter value, in the completion;
- output_end: per tool output, the last token wholly inside the `<tool_response>` content, read once, in the prompt of
  the first request that shows it (the next request of the run; the run's final outputs are never shown, so never read).
  Its row's `irid`, `n` and `sha256` are the request that shows it (the extractor's gate), `src_irid` the request whose
  call produced it. The outputs are found in the prompt's last user turn before the final assistant turn, must match
  the eval's tool messages of the previous turn in number, and each must equal its eval text (raw or stripped) by sha256;
  a failed tool call's text is "Error: <its error message>", which is what the policy was shown.
A request with no ordinary completion in the logs is listed and counted, and its calls get no readout.
`--unoffered a,b`: calls to these tool names (invented by the agent; the environment offers no such tool, so they ran
nothing) get no readout and no label, and are counted. The labeler (eb014307) is unchanged.
`--stage <name>` also writes each context epoch for run_extract_replay.py to `<name>/inputs/<arm folder>.jsonl` on the
activations Volume: its ids (the last request's prompt + completion) and per call its readout, its request's length n
and the sha256 of its logged prompt + completion (int32 bytes); every attempt is staged with its kind and index, and the
analysis chooses which count. It refuses if any check above mismatched, if an acting-model event has no request id, if
nothing was indexed, if a request is not a prefix of its epoch, or if `<name>` exists. Coverage per arm folder (attempts,
attempts without an .eval, samples, errored samples, samples without an acting-model call) is reported.

  PYTHONPATH=src uv run --no-project --python 3.14 --with 'modal>=1.2' modal run \
      experiments/2026-10-08_q3_onpolicy_figures/run_replay_readouts.py --run-dir <run folder> \
      --arms attack,sim_attack --acting-model vllm/Qwen/Qwen3.5-27B [--keys attack/t17_slack_clone,...] \
      --logs <request-log file>,... --label <name>
"""

import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3onpol_readouts_kill-2026-11-08"
REQUEST_LOG = "probe-shift_aswin-surya_sub-phase2_request-log_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
DIFFER, DIFFER_MAX = "parameter names differ from the eval's, or repeat (calls kept, no readout)", 5
OUT_DIFFER = "output_end: tool output differs from the eval (raw and stripped; no readout)"  # also stops above DIFFER_MAX

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))


def bytes_to_unicode() -> dict:
    """GPT-2's byte-level BPE alphabet (Qwen's tokenizer uses it): byte -> the character a token string writes it as."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


REF = re.compile(r"attachment://[0-9a-f]{32}")  # Inspect's attachment reference: the whole value


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@app.function(image=image, cpu=4, memory=32768, timeout=3600, retries=0,
              volumes={"/rl": modal.Volume.from_name(REQUEST_LOG).read_only(), "/hf": modal.Volume.from_name(HF_CACHE).read_only(),
                       "/acts": modal.Volume.from_name(ACTS, create_if_missing=True, version=2)})
def remote(index: list, logs: list, stage: str = "", segment_ends: bool = False) -> dict:
    import gzip
    import itertools
    import re

    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    byte_of = {u: b for b, u in bytes_to_unicode().items()}
    added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    asst_ids = tok.encode("assistant", add_special_tokens=False)

    def token_bytes(seq):
        return [added[t].encode() if t in added else bytes(byte_of[ch] for ch in tok.convert_ids_to_tokens(t)) for t in seq]

    def last_inside(tb, lo, hi):
        """index of the last token whose whole byte span lies in [lo, hi), or None"""
        ends = list(itertools.accumulate(len(x) for x in tb))
        inside = [i for i, (s, e) in enumerate(zip([0] + ends[:-1], ends)) if s >= lo and e <= hi and e > s]
        return inside[-1] if inside else None

    def outputs_seen(prompt):
        """the tool outputs in the prompt's last user turn before the final assistant turn: (start, token bytes, text)"""
        p = prompt.tolist()
        starts = [i for i, x in enumerate(p) if x == im_start]
        asst = [i for i in starts if p[i + 1: i + 1 + len(asst_ids)] == asst_ids]
        if len(asst) < 2 or asst[-1] != starts[-1]:
            return None
        lo = next(i for i in range(asst[-2], asst[-1]) if p[i] == im_end) + 1  # after the previous assistant turn
        tb = token_bytes(p[lo: asst[-1]])
        return lo, tb, b"".join(tb).decode("utf-8", errors="strict")
    ids = [r["irid"] for r in index]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{len(ids) - len(set(ids))} request ids appear twice in the index")
    want = set(ids)
    rec, seen = {}, {}  # rec: the ordinary completions per id; seen: every record's status per id
    truncated = {}
    for n in logs:
        with gzip.open(Path("/rl") / n, "rt") as fh:
            lines = iter(fh)
            while True:
                try:
                    line = next(lines)
                except StopIteration:
                    break
                except EOFError:  # a log still being written: read what is complete, and say so
                    truncated[n] = "compressed stream ends early"
                    break
                if not line.endswith("\n"):
                    truncated[n] = "last line incomplete"
                    break
                r = json.loads(line)
                if r.get("request_id") not in want:
                    continue
                ok = r.get("status") == 200 and r.get("complete") and len(r.get("choices") or []) == 1 and r.get("prompt_token_ids") is not None
                seen.setdefault(r["request_id"], []).append("ordinary" if ok else str(r.get("status")))
                if ok:
                    rec.setdefault(r["request_id"], []).append((np.asarray(r["prompt_token_ids"], np.int32), r["choices"][0]["token_ids"]))
    statuses = Counter(" + ".join(sorted(v)) for v in seen.values())
    bad, info, calls_out, epochs, where = Counter(), Counter(), [], {}, []
    by_run = {}
    for ix in index:
        rs = rec.get(ix["irid"])
        if rs is None:  # no replayable record (e.g. an incomplete token record): its calls get no readout; listed, counted
            info["request without an ordinary completion in the logs (excluded)"] += 1
            where.append({"kind": "no ordinary completion", "run": ix["run"], "turn": ix["turn"], "irid": ix["irid"],
                          "statuses": seen.get(ix["irid"])})
            continue
        if len(rs) != 1:
            bad["several ordinary completions for one request id"] += 1
            continue
        prompt, comp = rs[0]
        by_run.setdefault(ix["run"], []).append((ix["turn"], prompt, comp, ix["irid"]))
        # each token's exact bytes (byte-level BPE, or an added token's own text), so spans are byte-exact
        tb = token_bytes(comp)
        raw = b"".join(tb)
        text = raw.decode("utf-8", errors="strict") if comp else ""
        if not comp or text != tok.decode(comp, skip_special_tokens=False):
            bad["token bytes do not rebuild the decoded completion"] += 1
            continue
        b_end = list(itertools.accumulate(len(x) for x in tb))
        b_start = [0] + b_end[:-1]
        to_byte = lambda i: len(text[:i].encode())
        found = list(re.finditer(r"<tool_call>\n<function=([^>\n]+)>\n(.*?)</function>\n</tool_call>", text, re.S))
        if [m.group(1) for m in found] != [c["name"] for c in ix["calls"]]:
            no_canonical = not found and ix["calls"] and not all(c["name"] == "submit" for c in ix["calls"])
            if no_canonical:  # vLLM's parser read calls this regex does not (scale-up launch 2: 2 requests); listed, no readout
                info["completion without a canonical tool call (calls kept, no command or call_end readout)"] += 1
                where.append({"kind": "no canonical tool call", "run": ix["run"], "turn": ix["turn"], "irid": ix["irid"],
                              "eval": [c["name"] for c in ix["calls"]]})
            if ix["calls"] and (no_canonical or all(c["name"] == "submit" for c in ix["calls"])):  # submit: excluded (amendment 7)
                if not no_canonical:
                    info["submit-only completion not parsed"] += 1
                calls_out.extend({"run": ix["run"], "arm": ix["arm"], "arm_dir": ix["arm_dir"], "task": ix["task"], "turn": ix["turn"],
                                  "irid": ix["irid"], "name": c["name"], "param": None, "label": None, "matched_subset": False,
                                  "prompt_len": len(prompt), "readout": None} for c in ix["calls"])
                continue
            bad["call names differ from the eval"] += 1
            where.append({"kind": "call names differ", "run": ix["run"], "turn": ix["turn"], "irid": ix["irid"],
                          "completion": [m.group(1) for m in found], "eval": [c["name"] for c in ix["calls"]]})
            continue
        for ci, (m, c) in enumerate(zip(found, ix["calls"])):
            params = [(p.group(1), p.start(2) + m.start(2), p.group(2)) for p in
                      re.finditer(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>", m.group(2), re.S)]
            names = [k for k, _, _ in params]
            row = {"run": ix["run"], "arm": ix["arm"], "arm_dir": ix["arm_dir"], "task": ix["task"], "turn": ix["turn"], "irid": ix["irid"], "name": c["name"],
                   "param": c["param"], "label": c["label"], "matched_subset": c["matched_subset"], "prompt_len": len(prompt),
                   "kind": "command", "src_irid": ix["irid"], "call_index": ci}
            differ = len(set(names)) != len(names) or sorted(names) != sorted(c["arg_names"])
            if segment_ends and not c["excluded"] and params and not differ:
                k_last, s_last, v_last = max(params, key=lambda q: q[1])  # the last parameter in the text
                i_last = last_inside(tb, to_byte(s_last), to_byte(s_last + len(v_last)))
                if i_last is None:
                    info["call_end: no token inside the last value"] += 1
                calls_out.append(row | {"kind": "call_end", "param": k_last, "readout": None if i_last is None else len(prompt) + i_last})
            if differ and c["name"] == "submit":
                info["submit parameters repeat or differ (excluded anyway)"] += 1
                row["readout"] = None
                calls_out.append(row)
                continue
            if differ:  # this regex and vLLM's parser read the call differently, so no readout is placed in it; listed, and
                # staging stops above DIFFER_MAX per batch (second attack runs: 2 of about 12,700 calls, a str_replace whose
                # old_str only vLLM's parser read, and a bash call with editor parameters that the tool rejected)
                info[DIFFER] += 1
                where.append({"kind": "parameter names differ", "run": ix["run"], "turn": ix["turn"], "irid": ix["irid"], "tool": c["name"],
                              "completion": names, "eval": c["arg_names"]})
                row["readout"] = None
                calls_out.append(row)
                continue
            if c["param"] is None:
                row["readout"] = None  # rule 1: no command value, or the terminal submit; dropped and counted
                calls_out.append(row)
                continue
            hit = [(s0, v) for k, s0, v in params if k == c["param"]]
            if len(hit) != 1:
                bad["command parameter missing from the completion"] += 1
                row["readout"] = None
                calls_out.append(row)
                continue
            s0, v = hit[0]
            if sha(v) != c["value_sha256"] and sha(v.strip()) != c["value_sha256"]:
                bad["command value differs from the eval (raw and stripped)"] += 1
                row["readout"] = None
                calls_out.append(row)
                continue
            lo, hi = to_byte(s0), to_byte(s0 + len(v))
            inside = [i for i in range(len(comp)) if b_start[i] >= lo and b_end[i] <= hi and b_end[i] > b_start[i]]
            if not inside:
                bad["no token lies wholly inside the value"] += 1
                row["readout"] = None
            else:
                row["readout"] = len(prompt) + inside[-1]
                row["readout_token_id"] = comp[inside[-1]]
            calls_out.append(row)
    if segment_ends:  # output_end: the previous turn's tool outputs, in this request's prompt
        for ix in index:
            rs = rec.get(ix["irid"])
            if rs is None or len(rs) != 1 or ix["turn"] == 0 or not ix["prev_outputs"]:
                continue
            prompt = rs[0][0]
            seen_out = outputs_seen(prompt)
            if seen_out is None:
                bad["output_end: no previous assistant turn found in the prompt"] += 1
                continue
            lo, tb, text = seen_out
            blocks = list(re.finditer(r"<tool_response>\n(.*?)\n</tool_response>", text, re.S))
            if len(blocks) != len(ix["prev_outputs"]):
                bad["output_end: tool outputs in the prompt differ in number from the eval"] += 1
                where.append({"kind": "output count", "run": ix["run"], "turn": ix["turn"], "irid": ix["irid"],
                              "prompt": len(blocks), "eval": len(ix["prev_outputs"])})
                continue
            to_b = lambda i: len(text[:i].encode())
            for j, (b, o) in enumerate(zip(blocks, ix["prev_outputs"])):
                v = b.group(1)
                row = {"run": ix["run"], "arm": ix["arm"], "arm_dir": ix["arm_dir"], "task": ix["task"], "turn": ix["turn"], "irid": ix["irid"],
                       "name": o["name"], "param": None, "label": o["label"], "matched_subset": False, "prompt_len": len(prompt),
                       "kind": "output_end", "src_irid": o["src_irid"], "call_index": j}
                if o["excluded"]:  # the output of a rule-E call or of the submit: no readout, counted
                    info["output_end: output of an excluded call (rule E or submit)"] += 1
                    row["readout"] = None
                elif sha(v) != o["sha256"] and sha(v.strip()) != o["sha256_stripped"]:
                    # listed, no readout (honest batch C: 1 output, t26 turn 88, 2 characters shorter in the prompt than in the eval,
                    # whose eval text has two ESC characters; `matches_without_controls` was false: not that cause alone)
                    info[OUT_DIFFER] += 1
                    where.append({"kind": "output differs", "run": ix["run"], "turn": ix["turn"], "irid": ix["irid"], "call": j,
                                  "prompt_chars": len(v), "eval_chars": o["chars"],
                                  "matches_without_controls": sha(v.strip()) == o["sha256_stripped_no_controls"]})
                    row["readout"] = None
                else:
                    i_in = last_inside(tb, to_b(b.start(1)), to_b(b.end(1)))
                    if i_in is None:
                        info["output_end: empty tool output (no token inside)"] += 1
                    row["readout"] = None if i_in is None else lo + i_in
                calls_out.append(row)
    # context epochs: a request starts a new epoch unless its prompt begins with the previous request's prompt + completion
    rows_of = {}
    for row in calls_out:
        rows_of.setdefault(row["irid"], []).append(row)
    meta = {ix["irid"]: ix for ix in index}
    vol = {"tokens, one pass per request": 0, "tokens, one pass per epoch": 0, "requests": 0, "epochs": 0}
    staged = {}
    for run, reqs in by_run.items():
        reqs.sort(key=lambda x: x[0])
        eps = []  # per epoch: [(irid, full ids)]
        for turn, p, c, irid in reqs:
            full = np.concatenate([p, np.asarray(c, np.int32)])
            vol["tokens, one pass per request"] += len(full)
            vol["requests"] += 1
            last = eps[-1][-1][1] if eps else None
            if last is None or len(p) < len(last) or not np.array_equal(p[: len(last)], last):
                eps.append([])
            eps[-1].append((irid, full))
        for k, ep in enumerate(eps):
            ids = ep[-1][1]
            vol["tokens, one pass per epoch"] += len(ids)
            calls = []
            for irid, full in ep:
                if not np.array_equal(ids[: len(full)], full):  # the gate, at staging (the extractor repeats it)
                    raise ValueError(f"fidelity gate: request {irid} is not a prefix of its epoch's ids")
                for j, row in enumerate(rows_of.get(irid, [])):
                    row["epoch"] = k
                    calls.append({k2: row[k2] for k2 in ("irid", "turn", "name", "param", "label", "matched_subset", "readout")}
                                 | {k2: row[k2] for k2 in ("kind", "src_irid", "call_index") if k2 in row}
                                 | {"call": j, "n": len(full), "sha256": hashlib.sha256(full.tobytes()).hexdigest()})
            ix = meta[ep[0][0]]
            staged.setdefault(ix["arm_dir"], []).append({"key": f"{run}#e{k}", "arm": ix["arm_dir"], "run": run, "task": ix["task"],
                                                          **{k2: ix[k2] for k2 in ("attempt_index", "attempt_kind", "redrawn_because", "policy_called")},
                                                          "epoch": k, "ids": ids.tolist(), "calls": calls})
        vol["epochs"] += len(eps)
        epochs[run] = len(eps)
    written = {}
    for kind in (DIFFER, OUT_DIFFER):
        if info[kind] > DIFFER_MAX:
            bad[kind] = info[kind]
    if stage:
        if bad:
            raise ValueError(f"not staging: {dict(bad)}; details: {json.dumps(where)[:3000]}")
        d = Path("/acts") / stage / "inputs"
        if d.exists():
            raise FileExistsError(f"{d} exists; stage under a new name")
        d.mkdir(parents=True)
        for arm, eps in staged.items():
            b = "".join(json.dumps(e) + "\n" for e in eps).encode()
            (d / f"{arm}.jsonl").write_bytes(b)
            written[f"{arm}.jsonl"] = hashlib.sha256(b).hexdigest()
        modal.Volume.from_name(ACTS, version=2).commit()
    return {"mismatches": dict(bad), "calls": calls_out, "volume": vol, "epochs_per_run": epochs, "requests_found": len(seen),
            "record_statuses_per_request_id": dict(statuses), "staged_inputs_sha256": written, "logs_truncated": truncated,
            "mismatch_details": where, "notes": dict(info)}


@app.local_entrypoint()
def main(run_dir: str, arms: str, acting_model: str, logs: str, label: str, stage: str = "", keys: str = "", unoffered: str = "",
         segment_ends: bool = False):
    from probe_shift.calls import fingerprint, label_call
    from probe_shift.reader import command_param

    rd = Path(run_dir).expanduser()
    launch = json.loads((rd / "launch.json").read_text())
    index, skipped, coverage = [], Counter(), Counter()  # coverage: per arm folder, attempts and samples by outcome
    for r in launch["runs"]:
        if r["arm"] not in arms.split(",") or (keys and r["key"] not in keys.split(",")):  # keys: a batch of runs
            continue
        for ai, a in enumerate(r.get("attempts") or []):
            ld = a.get("log_dir") or ""
            local = rd / ld.split(f"/{rd.name}/", 1)[1] if f"/{rd.name}/" in ld else Path(ld)
            evs = sorted(local.glob("*.eval"))
            folder = local.relative_to(rd).parts[0] if local.is_relative_to(rd) else "?"
            coverage[f"{folder}: attempts"] += 1
            if not evs:
                coverage[f"{folder}: attempts without an .eval"] += 1
            for ev in evs:
                with zipfile.ZipFile(ev) as z:
                    sample_files = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
                    if len(sample_files) != 1:
                        raise ValueError(f"{ev}: {len(sample_files)} samples; one per .eval is expected")
                    for n in sample_files:
                        s = json.loads(z.read(n))
                        att = s.get("attachments") or {}
                        coverage[f"{folder}: samples"] += 1
                        coverage[f"{folder}: samples errored"] += bool(s.get("error"))
                        n_before = len(index)
                        turn = 0

                        def resolved(x):  # Inspect's whole-value `attachment://<32 hex>` references, at any depth; a missing one raises
                            if isinstance(x, str) and REF.fullmatch(x):
                                return att[x.removeprefix("attachment://")]
                            if isinstance(x, list):
                                return [resolved(v) for v in x]
                            if isinstance(x, dict):
                                return {k: resolved(v) for k, v in x.items()}
                            return x

                        def refs_left(x):  # a reference-shaped value after resolution (a chained one); text that only mentions the scheme is fine
                            if isinstance(x, str):
                                return bool(REF.fullmatch(x))
                            return any(map(refs_left, x.values() if isinstance(x, dict) else x)) if isinstance(x, (list, dict)) else False

                        def text_of(c):  # a tool message's content as the template sees it: attachments resolved, text parts joined
                            c = resolved(c)
                            if refs_left(c):
                                raise ValueError(f"{ev}: attachment reference left in a tool output")
                            if isinstance(c, list):
                                c = "".join(p.get("text", "") for p in c if p.get("type") == "text")
                            return c or ""
                        msgs = s.get("messages") or []
                        asst_msgs = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
                        outputs_after = []  # per assistant message: its tool messages (text, tool_call_id, function)
                        for k, i in enumerate(asst_msgs):
                            nxt = asst_msgs[k + 1] if k + 1 < len(asst_msgs) else len(msgs)
                            outputs_after.append([m for m in msgs[i + 1: nxt] if m["role"] == "tool"])
                        n_acting = sum(e.get("event") == "model" and e.get("model") == acting_model for e in s.get("events") or [])
                        # an errored sample keeps only its first messages: no tool outputs to check, so no output_end (counted)
                        outputs_known = len(asst_msgs) == n_acting
                        if segment_ends and not outputs_known:
                            if not s.get("error"):
                                raise ValueError(f"{ev}: {n_acting} acting-model calls but {len(asst_msgs)} assistant messages")
                            coverage[f"{folder}: errored samples without messages (no output_end)"] += 1
                        for e in s.get("events") or []:
                            if e.get("event") != "model":
                                continue
                            if e.get("model") != acting_model:
                                skipped[f"model event of {e.get('model')}"] += 1
                                continue
                            irid = (((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid")
                            msg = (((e.get("output") or {}).get("choices") or [{}])[0].get("message") or {})
                            tcs = msg.get("tool_calls") or []
                            if irid is None:
                                skipped["model event without x-irid"] += 1
                                continue
                            calls = []
                            for tc in tcs:
                                args = resolved(tc.get("arguments") or {})  # nested values too (2e's census, 10-11: one in a submit call)
                                if refs_left(args):
                                    raise ValueError(f"{ev}: attachment reference left in a call's arguments")
                                name = tc.get("function")
                                submit = name == "submit"  # the terminal submit: excluded (amendment 7)
                                offered = {t.get("name") for t in (e.get("tools") or [])}
                                fake = name in unoffered.split(",")  # a tool the environment does not offer: ran nothing; excluded, counted
                                if offered and name not in offered and not fake:  # Aswin's rule E: every such call is listed and excluded
                                    raise ValueError(f"{ev}: call to {name!r}, not offered in that model call; add it to --unoffered")
                                if fake:
                                    skipped[f"call to a tool not offered: {name}"] += 1
                                # A text_editor call without `command` fails the tool's input parsing, so it ran nothing (scale-up
                                # launch 2: one call, honest_b/t158, `cmd` for `command`); excluded and counted like rule E.
                                malformed = name == "text_editor" and args.get("command") is None
                                if malformed:
                                    skipped["text_editor call without command (rejected by the tool's parser)"] += 1
                                fake = fake or malformed
                                param = None if submit or fake else command_param(name, args)
                                lab = None if submit or fake else label_call(name, args)["label"]
                                calls.append({"name": name, "n_args": len(args), "arg_names": sorted(args), "param": param, "label": lab,
                                              "excluded": submit or fake,  # rule E and the terminal submit: no readout of any kind
                                              "value_sha256": sha(args[param]) if param else None,
                                              "matched_subset": bool(param) and list(args)[-1] == param})
                            prev = index[-1] if len(index) > n_before else None
                            prev_outputs = None
                            if segment_ends and prev is not None and outputs_known:
                                ids_prev = [tc.get("id") for tc in (msgs[asst_msgs[turn - 1]].get("tool_calls") or [])]
                                prev_outputs = []
                                for tm in outputs_after[turn - 1]:
                                    err = tm.get("error")  # a failed tool call: the policy saw "Error: <message>", not the content
                                    txt = f"Error: {err.get('message', '')}" if isinstance(err, dict) else text_of(tm.get("content"))
                                    ci = ids_prev.index(tm.get("tool_call_id")) if tm.get("tool_call_id") in ids_prev else None
                                    no_ctl = "".join(ch for ch in txt if ch in "\n\t" or unicodedata.category(ch) != "Cc")
                                    prev_outputs.append({"sha256": sha(txt), "sha256_stripped": sha(txt.strip()), "chars": len(txt),
                                                         "sha256_stripped_no_controls": sha(no_ctl.strip()),
                                                         "name": tm.get("function"), "src_irid": prev["irid"],
                                                         "excluded": ci is None or ci >= len(prev["calls"]) or prev["calls"][ci]["excluded"],
                                                         "label": prev["calls"][ci]["label"] if ci is not None and ci < len(prev["calls"]) else None})
                            index.append({"irid": irid, "run": str(ev.relative_to(rd)), "arm": r["arm"], "task": r["task"], "prev_outputs": prev_outputs,
                                          "arm_dir": ev.relative_to(rd).parts[0],  # e.g. sim_attack@september: the simulator
                                          "attempt_index": ai, "attempt_kind": a.get("kind"), "redrawn_because": a.get("redrawn_because"),
                                          "policy_called": a.get("policy_called"), "turn": turn, "calls": calls})
                            turn += 1
                        coverage[f"{folder}: samples without an acting-model call"] += len(index) == n_before
                        if segment_ends and outputs_known and len(index) - n_before != len(asst_msgs):
                            raise ValueError(f"{ev}: {len(index) - n_before} indexed calls but {len(asst_msgs)} assistant messages")
    unmatched = set(arms.split(",")) - {r["arm"] for r in launch["runs"]}
    if unmatched:
        raise ValueError(f"no run of arm {sorted(unmatched)} in launch.json (arms are e.g. attack, sim_attack; the simulator is a run field)")
    if stage and (skipped["model event without x-irid"] or not index):
        raise ValueError(f"not staging: {skipped['model event without x-irid']} acting-model events without a request id, "
                         f"{len(index)} indexed requests")
    res = remote.remote(index, logs.split(","), stage, segment_ends)
    rows = res["calls"]
    cmd = [r for r in rows if r.get("kind", "command") == "command"]  # the rule-1 rows (the per-arm counts below)
    summary = {"label": label, "acting_model": acting_model, "arms": arms, "keys": keys, "logs": logs, "logs_truncated": res["logs_truncated"], "labeler": fingerprint(), "indexed_requests": len(index), "requests_found": res["requests_found"],
               "coverage": dict(sorted(coverage.items())), "skipped_locally": dict(skipped),
               "record_statuses_per_request_id": res["record_statuses_per_request_id"],
               "mismatches": res["mismatches"], "mismatch_details": res["mismatch_details"], "notes": res["notes"], "volume": res["volume"],
               "rows_by_kind": {k: {"rows": sum(r.get("kind", "command") == k for r in rows), "with_readout": sum(r.get("kind", "command") == k and r["readout"] is not None for r in rows)}
                                for k in sorted({r.get("kind", "command") for r in rows})},
               "calls": sum(r.get("kind", "command") == "command" for r in rows),
               "calls_with_readout": sum(r.get("kind", "command") == "command" and r["readout"] is not None for r in rows),
               "by_arm": {a: {"calls": sum(r["arm_dir"] == a for r in cmd), "with_readout": sum(r["arm_dir"] == a and r["readout"] is not None for r in cmd),
                              "impactful": sum(r["arm_dir"] == a and r["readout"] is not None and r["label"] == "impactful" for r in cmd),
                              "read_only": sum(r["arm_dir"] == a and r["readout"] is not None and r["label"] == "read_only" for r in cmd),
                              "matched_subset": sum(r["arm_dir"] == a and r["readout"] is not None and r["matched_subset"] for r in cmd),
                              "no_readout_by_tool": dict(Counter(f"{r['name']}/{r['param']}" for r in cmd if r["arm_dir"] == a and r["readout"] is None))}
                          for a in sorted({r["arm_dir"] for r in cmd})},
               "segment_ends_by_arm": {a: {k: sum(r["arm_dir"] == a and r.get("kind") == k and r["readout"] is not None for r in rows) for k in ("call_end", "output_end")}
                                       for a in sorted({r["arm_dir"] for r in rows})},
               "epochs_per_run": res["epochs_per_run"], "stage": stage, "staged_inputs_sha256": res["staged_inputs_sha256"]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"replay_readouts_{label}_{stamp}.json"
    with path.open("x") as f:
        f.write(json.dumps({"summary": summary, "calls": rows}))
    print(json.dumps({k: v for k, v in summary.items() if k != "epochs_per_run"}, indent=1))
    print("->", path)
