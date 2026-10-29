"""B2 (RULE.md): stage Qwen3.5-27B's September simulated runs with "fix-full minus the clock (F7)" applied to their tool outputs,
as replay inputs for run_extract_b2.py. Modal CPU; writes no message text locally (counts and hashes only).

The repair is q2's fix-full (aswin/q2-fixfull @ 43ce3ec7, stage_fixfull.py, variant "full") without F7, in its order:
fix939's rerender (diff_with_note), the str_replace gutter (sr_gutter_no_newline), F5 (create outputs lose trailing newlines),
F6 (fixes.r5_dpkg), F8 (error-like outputs, by the 14 templates learned from REAL errored outputs exactly as q2 learns them:
#36's real-attack-e1/e2 staged inputs and the tool-error file). Modules on main, byte-identical to 43ce3ec7's.
Messages are each run's .eval transcript (attachments resolved), with each tool message's text as the policy saw it
(run_replay_readouts.py: "Error: <message>" for a failed call). Only tool outputs may change (else it raises).

Tokens: each staged epoch's logged ids (the original stagings, pinned by sha256). Every `<tool_response>` block's tokens are
the encoding of "\\n" + output + "\\n" on their own (the special tokens split the text). Per block:
- align: its text equals the run's next tool message text, raw or stripped (consecutive after the first match);
- render gate: re-encoding its decoded text gives exactly its logged tokens;
- if the repair changed that output: the block becomes encode("\\n" + repaired + "\\n"), stripped like the original was.
Positions after a changed block shift; every command and call_end readout lands on the same token id (gate), and each call's n
and sha256 are recomputed on the new ids. output_end readouts in a changed block are dropped. An epoch whose changed block fails
alignment or the render gate is excluded and counted; over 2% of epochs excluded stops staging. Unchanged epochs are written
byte-identical (checked).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run \\
      experiments/2026-10-13_q3_sabotage_avg_repair/stage_b2.py --bundle <run folders root> [--mode survey|stage]
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
SWAP, FSIM = REPO / "experiments" / "2026-10-07_q2_rendering_swap", REPO / "experiments" / "2026-10-09_fixed_sim"
ERRORS = REPO / "experiments/2026-10-06_q1_segment_end/results/tool_errors_2026-10-06_134908.jsonl"
APP = "probe-shift_aswin-surya_sub-q3sabavg_stage_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
LA = "probe-shift_aswin-surya_sub-la-activations"
BASE = "2026-10-06_la_segment-end_qwen3-32b_v1"  # q2's F8 source: #36's real-attack-e1/e2
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
EVALS, DST = "2026-10-13_b2fx_evals_v1", "2026-10-13_b2fx_{}"
STAGINGS = {  # original staging -> (run folder, {arm: inputs sha256 from its extraction config on main})
    "rerun_strict_segends_v2": ("2026-10-09_000523_rerun_strict", {
        "sim_attack@september": "6111e4e8c4469876f937524536b7c319dade99ea1860853a85ee841c00b044b1",
        "sim_honest@september": "2fd004c38289837d7916fa9676941e166ba8c8a74b997c1d50a49db692c33654"}),
    "scaleup_canary_segends_v1": ("2026-10-09_173815_scaleup_canary", {"sim_attack@september": "831d79c8474a52ec99da5468cfdf84ffc990a9d5258238491e6b0f9e33da274a"}),
    "scaleup_l2_segends_v1": ("2026-10-09_182505_scaleup", {"sim_attack@september": "95cc51603c80d7f1d9ae8020b27f093b30c4800fee0ae3006690b90c2388538c"}),
    "scaleup_topup_segends_v1": ("2026-10-10_001314_scaleup_topup", {"sim_attack@september": "0e15a2fed0ea2225f70c2f6ad0f6026fd7593f4126d8448befff198b51cdc114"})}
TR, TRC = 248066, 248067  # <tool_response>, </tool_response>
REF = re.compile(r"attachment://[0-9a-f]{32}")
BT = re.compile(r"`[^`]*`")

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "numpy", "zstandard==0.23.0")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift")
         .add_local_file(SWAP / "rerender.py", "/root/swap/rerender.py").add_local_file(SWAP / "str_replace_tells.py", "/root/swap/str_replace_tells.py")
         .add_local_file(FSIM / "fixes.py", "/root/fsim/fixes.py").add_local_file(FSIM / "clock.py", "/root/fsim/clock.py"))


def norm_first(text: str) -> str:  # stage_fixfull.norm_first, unchanged
    first = BT.sub("<v>", text.split("\n", 1)[0])
    return " ".join("<v>" if any(ch.isdigit() or ch == "/" for ch in t) else t for t in first.split(" "))


def trail(text: str) -> str:  # stage_fixfull.trail, unchanged
    n = len(text) - len(text.rstrip("\n"))
    return "none" if n == 0 else "one" if n == 1 else "more"


def bytes_to_unicode() -> dict:  # run_replay_readouts.bytes_to_unicode, unchanged
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def run_job(mode: str, eval_sha: dict, errored: list) -> dict:
    import zipfile

    import numpy as np
    from transformers import AutoTokenizer

    sys.path[:0] = ["/root/swap", "/root/fsim"]
    import fixes
    import rerender as rr
    import str_replace_tells as st

    c = Counter()
    # --- F8's templates, exactly as stage_fixfull.run_job learns them (REAL errored outputs of #36's real attack runs)
    base = {}
    for p in sorted((Path("/la") / BASE / "inputs").glob("shard_*.jsonl")):
        for line in p.read_text().splitlines():
            it = json.loads(line)
            if it["arm"] in ("real-attack-e1", "real-attack-e2"):
                base[(it["arm"], it["key"])] = it
    err = {(a, k, s, j): t for a, k, s, j, t in errored}
    real_err = defaultdict(Counter)
    for (a, k), it in base.items():
        for x in fixes.calls_of(it["messages"]):
            if err.get((a, k, x["step"], x["action"])) is None:
                continue
            text = it["messages"][x["msg"]]["content"]
            if isinstance(text, str):
                real_err[norm_first(text)][trail(text)] += 1
    templates = {}
    for key, v in real_err.items():
        n = sum(v.values())
        if n >= 5:
            templates[key] = "strip" if v["none"] >= 0.9 * n else "one" if v["one"] >= 0.9 * n else "keep"
    if len(templates) != 14 or set(templates.values()) != {"strip"}:
        raise ValueError(f"F8: {len(templates)} templates {Counter(templates.values())}, not q2's 14 all-strip table")
    c["F8|templates (14, all strip)"] = len(templates)

    def is_err(text):
        return isinstance(text, str) and norm_first(text) in templates

    def f5(msgs):  # stage_fixfull.f5, unchanged
        out = [dict(m) for m in msgs]
        for x in fixes.calls_of(msgs):
            o = msgs[x["msg"]]["content"]
            if x["tool"] == "text_editor" and x["args"].get("command") == "create" and isinstance(o, str) and not is_err(o):
                out[x["msg"]]["content"] = o.rstrip("\n")
        return out

    def f8(msgs):  # stage_fixfull.f8, unchanged
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

    def repair(msgs):  # stage_fixfull's variant "full" with F7 removed
        m = rr.rerender_messages(msgs, "diff_with_note")[0]
        m = st.swap_messages(m, "sr_gutter_no_newline")[0]
        m = fixes.r5_dpkg(f5(m))[0]  # f6(f5(m)), as in variant("full")
        return f8(m)

    # --- each run's messages from its .eval (pinned by sha256), tool text as the policy saw it
    def member(z, info):
        """a zip member's bytes; zstd members (method 93, Inspect's .eval) decompressed by hand on Python 3.12, CRC-checked"""
        if info.compress_type != 93:
            return z.read(info)
        import struct
        import zlib

        import zstandard

        z.fp.seek(info.header_offset)
        h = z.fp.read(30)
        if h[:4] != b"PK\x03\x04":
            raise ValueError("bad local file header")
        n, m = struct.unpack("<HH", h[26:30])
        z.fp.seek(info.header_offset + 30 + n + m)
        out = zstandard.ZstdDecompressor().decompress(z.fp.read(info.compress_size), max_output_size=info.file_size)
        if len(out) != info.file_size or zlib.crc32(out) != info.CRC:
            raise ValueError("zstd member fails its size or CRC check")
        return out

    def messages_of(path):
        with zipfile.ZipFile(path) as z:
            infos = [i for i in z.infolist() if i.filename.startswith("samples/") and i.filename.endswith(".json")]
            if len(infos) != 1:
                raise ValueError("one sample per .eval is expected")
            s = json.loads(member(z, infos[0]))
        att = s.get("attachments") or {}

        def resolved(x):
            if isinstance(x, str) and REF.fullmatch(x):
                return att[x.removeprefix("attachment://")]
            if isinstance(x, list):
                return [resolved(v) for v in x]
            if isinstance(x, dict):
                return {k: resolved(v) for k, v in x.items()}
            return x

        def text_of(x):
            x = resolved(x)
            if isinstance(x, list):
                x = "".join(p.get("text", "") for p in x if p.get("type") == "text")
            return x or ""
        out = []
        for m in s.get("messages") or []:
            if m["role"] == "tool":
                e = m.get("error")
                out.append({"role": "tool", "content": f"Error: {e.get('message', '')}" if isinstance(e, dict) else text_of(m.get("content"))})
            elif m["role"] == "assistant":
                out.append({"role": "assistant", "content": text_of(m.get("content")), "tool_calls": [
                    {"function": {"name": tc.get("function"), "arguments": resolved(tc.get("arguments") or {})}} for tc in m.get("tool_calls") or []]})
            else:
                out.append({"role": m["role"], "content": text_of(m.get("content"))})
        return out

    fam = lambda x: f"text_editor:{x['args'].get('command')}" if x["tool"] == "text_editor" else x["tool"]  # noqa: E731
    repaired = {}  # run -> (tool texts as-is, repaired tool texts, family per tool message)
    for rel, want in sorted(eval_sha.items()):
        p = Path("/acts") / EVALS / rel
        if hashlib.sha256(p.read_bytes()).hexdigest() != want:
            raise ValueError("an uploaded .eval differs from its local sha256")
        m = messages_of(p)
        new = repair(m)
        if len(new) != len(m):
            raise ValueError("the repair changed the number of messages")
        changed = [i for i, (x, y) in enumerate(zip(m, new)) if x != y]
        if any(m[i]["role"] != "tool" or set(m[i]) != set(new[i]) or any(m[i][f] != new[i][f] for f in m[i] if f != "content") for i in changed):
            raise ValueError("the repair changed a non-tool message or a field other than content")
        tools = [i for i, x in enumerate(m) if x["role"] == "tool"]
        fm = {x["msg"]: fam(x) for x in fixes.calls_of(m)}
        repaired[rel] = ([m[i]["content"] for i in tools], [new[i]["content"] for i in tools], [fm.get(i, "unpaired") for i in tools])
        c["runs read"] += 1
        c["runs changed"] += bool(changed)
        c["outputs changed"] += len(changed)
        for i in changed:
            c[f"outputs changed|{fm.get(i, 'unpaired')}"] += 1

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    byte_of = {u: b for b, u in bytes_to_unicode().items()}
    added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    if tok.convert_tokens_to_ids("<tool_response>") != TR or tok.convert_tokens_to_ids("</tool_response>") != TRC:
        raise ValueError("unexpected tool_response ids")

    def text(seq):
        return b"".join(added[t].encode() if t in added else bytes(byte_of[ch] for ch in tok.convert_ids_to_tokens(t)) for t in seq).decode("utf-8")

    sha32 = lambda ids: hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()  # noqa: E731
    out_files, record = {}, []
    for stg, (folder, arms) in STAGINGS.items():
        for arm, want in arms.items():
            raw = (Path("/acts") / stg / "inputs" / f"{arm}.jsonl").read_bytes()
            if hashlib.sha256(raw).hexdigest() != want:
                raise ValueError(f"{stg}/{arm}: staged inputs differ from the extraction config's sha256")
            new_eps = []
            for line in raw.decode().splitlines():
                ep = json.loads(line)
                c["epochs"] += 1
                run = f"{folder}/{ep['run']}"
                if run not in repaired:
                    raise ValueError("a staged run has no uploaded .eval")
                orig, fixed, fams = repaired[run]
                ids = ep["ids"]
                opens = [i for i, x in enumerate(ids) if x == TR]
                closes = [i for i, x in enumerate(ids) if x == TRC]
                if len(opens) != len(closes) or any(not o < e for o, e in zip(opens, closes)) or any(e > o2 for e, o2 in zip(closes, opens[1:])):
                    c["epochs kept unrepaired|tool_response markers unpaired"] += 1
                    new_eps.append(ep | {"b2_blocks_changed": 0, "b2_unrepaired": "tool_response markers unpaired", "orig_ids_sha256": sha32(ids)})
                    continue
                blocks, bad = [], None
                for o, e in zip(opens, closes):
                    seg = ids[o + 1: e]
                    try:
                        t = text(seg)
                    except UnicodeDecodeError:
                        t = None
                    if t is None or not (t.startswith("\n") and t.endswith("\n") and len(t) >= 2):
                        bad = "block is not newline-wrapped text"
                        break
                    blocks.append((o + 1, e, t[1:-1], tok.encode(t, add_special_tokens=False) == seg))
                match = lambda v, q: q < len(orig) and v in (orig[q], orig[q].strip())  # noqa: E731
                p0 = None if bad or not blocks else next((q for q in range(len(orig)) if all(match(b[2], q + j) for j, b in enumerate(blocks))), None)
                if not bad and blocks and p0 is None:
                    bad = "blocks do not align with consecutive tool messages"
                edits = []
                if not bad and blocks:
                    c[f"first block at tool message {'0' if p0 == 0 else '>0'}"] += 1
                    for j, (s, e, v, rendered) in enumerate(blocks):
                        q = p0 + j
                        c["blocks"] += 1
                        c[f"render gate|{'pass' if rendered else 'fail'}"] += 1
                        if fixed[q] == orig[q]:
                            continue
                        if not rendered:
                            bad = "changed block fails the render gate"
                            break
                        nv = fixed[q] if v == orig[q] else fixed[q].strip()
                        c[f"changed outputs|{'raw' if v == orig[q] else 'stripped'}"] += 1
                        if nv != v:
                            edits.append((s, e, tok.encode("\n" + nv + "\n", add_special_tokens=False), fams[q]))
                if bad:
                    c[f"epochs kept unrepaired|{bad}"] += 1  # Amendment 3: staged as-is (flagged), so every call keeps a reading
                    new_eps.append(ep | {"b2_blocks_changed": 0, "b2_unrepaired": bad, "orig_ids_sha256": sha32(ids)})
                    record.append({"staging": stg, "arm": arm, "key": ep["key"], "blocks_changed": 0, "unrepaired": bad, "orig_ids_sha256": sha32(ids),
                                   "ids_sha256": sha32(ids)})
                    continue
                new_ids, shifts, prev, d = [], [], 0, 0
                for s0, e0, toks, f in edits:
                    new_ids += ids[prev:s0] + toks
                    d += len(toks) - (e0 - s0)
                    shifts.append((s0, e0, d))
                    prev = e0
                    c[f"blocks re-rendered|{f}"] += 1
                new_ids += ids[prev:]

                def shift(pos):  # shifts: (start, end, cumulative delta after the block), in order
                    d = 0
                    for s0, e0, dd in shifts:
                        if s0 <= pos < e0:
                            return None
                        if pos >= e0:
                            d = dd
                    return pos + d
                calls = []
                for cl in ep["calls"]:
                    nn, nr = shift(cl["n"]), (shift(cl["readout"]) if cl["readout"] is not None else None)
                    if nn is None:
                        raise ValueError("a request ends inside a tool output")
                    if cl["readout"] is not None and nr is None:
                        if cl.get("kind") != "output_end":
                            raise ValueError("a command or call_end readout lies inside a tool output")
                        c["output_end readouts dropped (inside a changed block)"] += 1
                    if nr is not None and new_ids[nr] != ids[cl["readout"]]:
                        raise ValueError("a shifted readout lands on another token")
                    calls.append(cl | {"n": nn, "readout": nr, "sha256": sha32(new_ids[:nn])})
                if not edits and (new_ids != ids or any(a["sha256"] != b["sha256"] for a, b in zip(calls, ep["calls"]))):
                    raise ValueError("an unchanged epoch is not byte-identical")
                c["epochs changed"] += bool(edits)
                c["tokens before"] += len(ids)
                c["tokens after"] += len(new_ids)
                new_eps.append(ep | {"ids": new_ids, "calls": calls, "b2_blocks_changed": len(edits), "orig_ids_sha256": sha32(ids)})
                record.append({"staging": stg, "arm": arm, "key": ep["key"], "blocks_changed": len(edits), "orig_ids_sha256": sha32(ids),
                               "ids_sha256": sha32(new_ids)})
            out_files[(stg, arm)] = new_eps
    excluded = sum(v for k, v in c.items() if k.startswith("epochs kept unrepaired|"))
    if excluded > 0.02 * c["epochs"]:
        raise ValueError(f"{excluded} of {c['epochs']} epochs left unrepaired (> 2%): nothing is staged")
    res = {"counts": dict(sorted(c.items()))}
    if mode == "stage":
        shas = {}
        for (stg, arm), eps in out_files.items():
            d = Path("/acts") / DST.format(stg) / "inputs"
            if (d / f"{arm}.jsonl").exists():
                raise FileExistsError(f"{d}/{arm}.jsonl exists: a staged file is never overwritten")
            d.mkdir(parents=True, exist_ok=True)
            body = "".join(json.dumps(e) + "\n" for e in eps).encode()
            (d / f"{arm}.jsonl").write_bytes(body)
            shas[f"{DST.format(stg)}/inputs/{arm}.jsonl"] = {"epochs": len(eps), "sha256": hashlib.sha256(body).hexdigest()}
        rec = json.dumps(record).encode()
        (Path("/acts") / DST.format("record")).mkdir(parents=True, exist_ok=True)
        (Path("/acts") / DST.format("record") / "stage_record.json").write_bytes(rec)
        acts.commit()
        res |= {"staged": shas, "record_sha256": hashlib.sha256(rec).hexdigest()}
    return res


@app.function(image=image, cpu=8, memory=65536, timeout=3600, retries=0,
              volumes={"/acts": acts, "/la": modal.Volume.from_name(LA, version=2).read_only(),
                       "/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only()})
def job(mode: str, eval_sha: dict, errored: list) -> dict:
    try:
        return run_job(mode, eval_sha, errored)
    except Exception as e:  # noqa: BLE001 - sanitized (no transcript text crosses the boundary) and re-raised
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"stage failed: {type(e).__name__}: {str(e)[:200]} at {Path(tb.filename).name}:{tb.lineno}") from None


@app.local_entrypoint()
def main(bundle: str, mode: str = "survey"):
    if mode not in ("survey", "stage"):
        raise SystemExit("--mode survey|stage")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(Path(__file__))], cwd=HERE).returncode:
        raise SystemExit("commit this file before running")
    root = Path(bundle)
    eval_sha = {}
    for folder, arms in STAGINGS.values():
        for arm in arms:
            for p in sorted((root / folder / arm).rglob("*.eval")):
                eval_sha[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    errored = [[r["arm"], r["key"], r["step"], r["action"], r["tool"]] for r in map(json.loads, ERRORS.read_text().splitlines())
               if r["error"] is not None and r["arm"] in ("real-attack-e1", "real-attack-e2", "sim-attack")]
    out = job.remote(mode, eval_sha, errored)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"stage_b2_{mode}_{stamp}.json"
    with p.open("x") as fh:
        fh.write(json.dumps({"git": head, "mode": mode, "evals": len(eval_sha), "evals_sha256_digest": hashlib.sha256(json.dumps(eval_sha, sort_keys=True).encode()).hexdigest(),
                             "tool_errors_sha256": hashlib.sha256(ERRORS.read_bytes()).hexdigest()} | out, indent=1) + "\n")
    print(json.dumps(out, indent=1)[:5000])
    print("->", p)
