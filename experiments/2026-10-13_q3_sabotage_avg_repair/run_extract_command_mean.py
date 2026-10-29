"""B1 (RULE.md): the averaged command readout. Per policy call, the MEAN of Qwen3.5-27B's layer-30 residual stream over the
tokens of the rule-1 command value, beside a re-read of the last-token row at the stored command readout.

A copy of experiments/2026-10-13_onpolicy_reply/run_extract_pooled.py (main @ 95f029dc; itself run_extract_replay.py with the
same fidelity gate, leases, chunking, resume and padding). What differs:
- Rows: the staged kind == "command" calls (rule 1's command readout r), not output_end. Layer 30 only; the forward pass
  stops after block 30 (later blocks can't change it).
- Span: tokens wholly inside the command parameter's value, found as the staging did (run_replay_readouts.py: each token's
  exact bytes, the same tool-call and parameter regexes), in the call's own assistant turn: from the last
  `<|im_start|>assistant` before r up to the request's length n. Never the call_end readout: a trailing fixed-slot
  parameter (timeout) is outside the span.
- Mean: fp32 sum over the span's rows (index_add over gathered positions, no prefix sums), stored fp16.
Gates:
1. Span (CPU, every epoch of the shard, before the model loads): the region decodes as UTF-8; the call_index-th tool call
   exists with the staged tool name; exactly one parameter has the staged name; tokens lie wholly inside its value and are
   contiguous; the last of them is exactly the staged readout r. A failing call is excluded and counted; more than 1%
   failing in a shard stops it before any forward pass.
2. Reproduction (every epoch): the re-read last-token row at r equals the original extraction's stored layer-30 row for
   the same call: max |diff| <= 1e-2 on an H200 (2e's smoke read 0.0); on a B200, cosine >= 0.999 per row. Else the
   shard stops before writing that epoch.

  PYTHONPATH=src uv run --no-project --python 3.14 --with 'modal==1.6.1' --with pyyaml modal run \\
      experiments/2026-10-13_q3_sabotage_avg_repair/run_extract_command_mean.py [--test --only <config>] [--attempt 1]
"""

import bisect
import hashlib
import itertools
import json
import os
import re
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3sabavg_extract_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
LEASES = "probe-shift_aswin-surya_sub-q3onpol_leases_kill-2026-11-08"
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]
TIMEOUT_S, LAYER, SPAN_FAIL_MAX, CHUNK_POS = 4 * 3600, 30, 0.01, 32768
BASE4 = ["attack", "honest_a", "honest_b", "sim_attack@september"]
CONFIGS = {  # original config on main -> (arms read, shards)
    "extract_rerun_segends.yaml": (BASE4 + ["sim_honest@september"], 3), "extract_scaleup_canary.yaml": (BASE4, 1),
    "extract_scaleup_l2.yaml": (BASE4, 18), "extract_scaleup_topup.yaml": (BASE4, 2),
    "extract_second_attack.yaml": (["attack"], 4), "extract_honest_c.yaml": (["honest_b"], 6)}
CALL = re.compile(r"<tool_call>\n<function=([^>\n]+)>\n(.*?)</function>\n</tool_call>", re.S)  # run_replay_readouts.py's
PARAM = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>", re.S)

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, create_if_missing=False, version=2)
leases = modal.Dict.from_name(LEASES, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})


def bytes_to_unicode() -> dict:  # run_replay_readouts.bytes_to_unicode, unchanged
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def ids_sha(ids) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()


def gate(ep: dict) -> None:  # run_extract_replay.gate, unchanged
    bad = [c["irid"] for c in ep["calls"] if c["n"] > len(ep["ids"]) or ids_sha(ep["ids"][: c["n"]]) != c["sha256"]
           or (c["readout"] is not None and not 0 <= c["readout"] < c["n"])]
    if bad:
        raise ValueError(f"fidelity gate: {len(bad)} calls of {ep['key']} differ from the logged tokens, e.g. {bad[:3]}")


def save_npy(path: Path, x) -> None:
    import numpy as np

    with open(path, "wb") as fh:
        np.save(fh, x)


def filename(key: str) -> str:  # run_extract_replay.filename, unchanged (the original meta's file name)
    return hashlib.sha1(key.encode()).hexdigest()[:20]


class Stop(Exception):
    pass


@app.function(image=image, gpu=["H200", "B200"], memory=65536, timeout=TIMEOUT_S, retries=0, max_containers=40,
              volumes={"/hf": modal.Volume.from_name(HF_CACHE).read_only(), "/acts": acts_vol})
def extract_shard(cfg: dict, shard: int) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    me = f"{os.environ.get('MODAL_TASK_ID', '?')}@{time.time():.0f}"
    key = f"{cfg['run']}/shard{shard:03d}"
    if not leases.put(f"{key}/a{cfg['attempt']}", me, skip_if_exists=True):
        raise RuntimeError(f"shard {shard} attempt {cfg['attempt']} already has a writer; rerun with a new --attempt")
    held = leases.get(key)
    if held is not None and held["attempt"] >= cfg["attempt"]:
        raise RuntimeError(f"shard {shard} is held by attempt {held['attempt']}; rerun with a later --attempt")
    leases[key] = {"writer": me, "attempt": cfg["attempt"]}

    def check():
        if (leases.get(key) or {}).get("writer") != me:
            raise RuntimeError(f"shard {shard}: a later attempt took over; stopping before writing")

    def put(path: Path, write):
        tmp = path.with_name(f".{path.name}.{cfg['attempt']}.tmp")
        write(tmp)
        check()
        os.replace(tmp, path)

    acts_vol.reload()
    run_dir, src, orig = Path("/acts") / cfg["run"], Path("/acts") / cfg["inputs"] / "inputs", Path("/acts") / cfg["orig_run"]
    raw = {n: (src / n).read_bytes() for n in sorted(cfg["inputs_sha256"])}
    if {n: hashlib.sha256(b).hexdigest() for n, b in raw.items()} != cfg["inputs_sha256"]:
        raise ValueError(f"the staged inputs in {src} differ from the config's sha256s")
    if missing := set(cfg["arms"]) - {n[:-6] for n in raw}:
        raise ValueError(f"arms {sorted(missing)} are not among the config's pinned files")
    eps = sorted((json.loads(line) for n, b in raw.items() if n[:-6] in cfg["arms"] for line in b.decode().splitlines()),
                 key=lambda e: (-len(e["ids"]), e["key"]))
    mine = eps[:1] if cfg.get("test") else eps[shard:: cfg["shards"]]
    for ep in mine:
        gate(ep)

    def done(ep) -> bool:
        f = run_dir / "meta" / ep["arm"] / f"{filename(ep['key'])}.json"
        if not f.exists():
            return False
        m = json.loads(f.read_text())
        idx = json.loads((run_dir / "chunks" / f"{m['chunk']}.json").read_text())
        if m["inputs_sha256"] != cfg["inputs_sha256"][f"{ep['arm']}.jsonl"] or m["key"] not in {e["key"] for e in idx["epochs"]}:
            raise ValueError(f"{f} was written from other inputs or its chunk does not list it")
        return True

    todo = [ep for ep in mine if not done(ep)]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    byte_of = {u: b for b, u in bytes_to_unicode().items()}
    added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    IMS, ASST = tok.convert_tokens_to_ids("<|im_start|>"), tok.encode("assistant", add_special_tokens=False)

    def token_bytes(seq):  # run_replay_readouts.py's: each token's exact bytes
        return [added[t].encode() if t in added else bytes(byte_of[ch] for ch in tok.convert_ids_to_tokens(t)) for t in seq]

    def spans_of(ep):
        """(call, first, r) per staged command call with a readout; failures counted by kind (gate 1)"""
        ids, ok, fail, cache = ep["ids"], [], Counter(), {}
        heads = [i + 1 + len(ASST) for i, x in enumerate(ids) if x == IMS and ids[i + 1: i + 1 + len(ASST)] == ASST]
        for c in ep["calls"]:
            if c.get("kind", "command") != "command" or c["readout"] is None:
                continue
            r, n = c["readout"], c["n"]
            h = bisect.bisect_right(heads, r) - 1
            if h < 0:
                fail["no assistant turn before the readout"] += 1
                continue
            a = heads[h]
            if (a, n) not in cache:
                tb = token_bytes(ids[a:n])
                try:
                    text = b"".join(tb).decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    text = None
                ends = list(itertools.accumulate(len(x) for x in tb))
                cache[a, n] = (text, [0] + ends[:-1], ends, list(CALL.finditer(text)) if text is not None else [])
            text, b_start, b_end, found = cache[a, n]
            if text is None:
                fail["region is not UTF-8"] += 1
                continue
            ci = c.get("call_index", 0)
            if ci >= len(found) or found[ci].group(1) != c["name"]:
                fail["tool call missing or another name"] += 1
                continue
            m = found[ci]
            hit = [(m.start(2) + p.start(2), p.group(2)) for p in PARAM.finditer(m.group(2)) if p.group(1) == c["param"]]
            if len(hit) != 1:
                fail["command parameter missing or repeated"] += 1
                continue
            s0, v = hit[0]
            lo, hi = len(text[:s0].encode()), len(text[: s0 + len(v)].encode())
            inside = [i for i in range(len(b_end)) if b_start[i] >= lo and b_end[i] <= hi and b_end[i] > b_start[i]]
            if not inside or inside != list(range(inside[0], inside[-1] + 1)):
                fail["no contiguous token span inside the value"] += 1
                continue
            if a + inside[-1] != r:
                fail["span end differs from the staged readout"] += 1
                continue
            ok.append((c, a + inside[0], r))
        return ok, fail

    spans, n_rows, fails = {}, 0, Counter()  # gate 1 on the whole shard before the model loads
    for ep in todo:
        spans[ep["key"]], f = spans_of(ep)
        n_rows, fails = n_rows + len(spans[ep["key"]]) + sum(f.values()), fails + f
    if n_rows and sum(fails.values()) / n_rows > SPAN_FAIL_MAX:
        raise ValueError(f"shard {shard}: {dict(fails)} of {n_rows} command calls failed the span gate (> 1%); nothing is read")
    span_s = round(time.time() - t0)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lm = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=torch.bfloat16, device_map="cuda",
                                              attn_implementation="sdpa").eval()
    base, gpu = lm.model, torch.cuda.get_device_name(0)
    load_s = round(time.time() - t0) - span_s

    def read(ids, F, R):
        """layer-30 means over [F_j, R_j] (fp32 sums) and rows at R_j; the forward pass stops after block 30"""
        got = {}

        def hook(m, a, out):
            h = (out[0] if isinstance(out, tuple) else out)[0]
            pos = torch.cat([torch.arange(f, r + 1, device="cuda") for f, r in zip(F, R)])
            seg = torch.cat([torch.full((r - f + 1,), j, device="cuda") for j, (f, r) in enumerate(zip(F, R))])
            sums = torch.zeros(len(F), h.shape[1], device="cuda", dtype=torch.float32)
            for s in range(0, len(pos), CHUNK_POS):
                sums.index_add_(0, seg[s: s + CHUNK_POS], h.index_select(0, pos[s: s + CHUNK_POS]).float())
            cnt = torch.tensor([r - f + 1 for f, r in zip(F, R)], device="cuda", dtype=torch.float32).unsqueeze(1)
            got["mean"] = (sums / cnt).to(torch.float16).cpu().numpy()
            got["last"] = h.index_select(0, torch.tensor(R, device="cuda")).to(torch.float16).cpu().numpy()
            raise Stop

        hk = base.layers[LAYER].register_forward_hook(hook)
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            with torch.inference_mode():
                base(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
            raise RuntimeError("the forward pass did not reach layer 30's hook")
        except Stop:
            pass
        finally:
            hk.remove()
        torch.cuda.synchronize()
        return got, round(time.time() - t, 2), round(torch.cuda.max_memory_allocated() / 2**30, 1)

    def reproduce(ep, kept, last):
        """gate 2: re-read last-token rows vs the original extraction's stored layer-30 rows for the same calls"""
        om = json.loads((orig / "meta" / ep["arm"] / f"{filename(ep['key'])}.json").read_text())
        orow = {(c["irid"], c.get("call_index", 0), c["readout"]): c["row"] for c in om["calls"]
                if c.get("kind", "command") == "command" and c.get("row") is not None}
        stored = np.load(orig / "by_layer" / f"L{LAYER}" / f"{om['chunk']}.npy", mmap_mode="r")
        if any((c["irid"], c.get("call_index", 0), r) not in orow for c, _, r in kept):
            raise ValueError(f"{ep['key']}: a command row is missing from the original extraction")
        o = np.stack([np.asarray(stored[om["row_start"] + orow[(c["irid"], c.get("call_index", 0), r)]], np.float32) for c, _, r in kept])
        x = last.astype(np.float32)
        diff = float(np.abs(x - o).max(initial=0))
        cos = float(((x * o).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(o, axis=1) + 1e-12)).min(initial=1))
        if ("H200" in gpu and diff > 1e-2) or ("H200" not in gpu and cos < 0.999):
            raise ValueError(f"{ep['key']}: last-token rows differ from the original on {gpu} (max |diff| {diff}, min cos {cos})")
        return {"max_abs_diff_vs_original": diff, "min_cos_vs_original": cos, "rows_compared": len(kept)}

    for d in ("by_layer", "last"):
        (run_dir / d / f"L{LAYER}").mkdir(parents=True, exist_ok=True)
    (run_dir / "chunks").mkdir(parents=True, exist_ok=True)
    buf, metas, stats, peaks, chunk, worst = {"by_layer": [], "last": []}, [], [], [], 0, {"diff": 0.0, "cos": 1.0}

    def flush():
        nonlocal chunk
        if not metas:
            return
        name = f"shard{shard:03d}_a{cfg['attempt']}_chunk{chunk:03d}"
        xs = {d: np.concatenate(v) for d, v in buf.items()}
        if nonfinite := sum(int((~np.isfinite(x)).sum()) for x in xs.values()):
            raise ValueError(f"{name}: {nonfinite} non-finite activations; nothing of it is published")
        for d, x in xs.items():
            put(run_dir / d / f"L{LAYER}" / f"{name}.npy", lambda t, x=x: save_npy(t, x))
            buf[d].clear()
        start = 0
        for m in metas:
            m |= {"chunk": name, "row_start": start, "inputs_sha256": cfg["inputs_sha256"][f"{m['arm']}.jsonl"]}
            start += m["n_rows"]
        index = {"layers": [LAYER], "rows": start, "epochs": [{k: m[k] for k in ("arm", "key", "row_start", "n_rows")} for m in metas]}
        put(run_dir / "chunks" / f"{name}.json", lambda t: t.write_text(json.dumps(index)))
        for m in metas:
            dd = run_dir / "meta" / m["arm"]
            dd.mkdir(parents=True, exist_ok=True)
            put(dd / f"{filename(m['key'])}.json", lambda t, m=m: t.write_text(json.dumps(m)))
        stats.append({"chunk": name, "epochs": len(metas), "rows": start})
        metas.clear()
        chunk += 1
        acts_vol.commit()

    for ep in todo:
        check()
        gate(ep)
        kept = spans[ep["key"]]
        ids = ep["ids"] + [pad_id] * (-len(ep["ids"]) % cfg["pad_to"])
        F, R = [f for _, f, _ in kept], [r for _, _, r in kept]
        if kept:
            got, seconds, peak = read(ids, F, R)
            rep = reproduce(ep, kept, got["last"])
            worst = {"diff": max(worst["diff"], rep["max_abs_diff_vs_original"]), "cos": min(worst["cos"], rep["min_cos_vs_original"])}
        else:
            got, seconds, peak, rep = {d: np.zeros((0, 5120), np.float16) for d in ("mean", "last")}, 0.0, 0.0, None
        peaks.append(peak)
        timing = {"forward_s": seconds, "peak_gib": peak, "padded_tokens": len(ids), "gpu": gpu, "reproduction": rep}
        if cfg.get("test") and kept:
            again, s2, _ = read(ids, F, R)
            timing |= {"forward_s_repeat": s2, "repeat_max_abs_diff": float(np.abs(got["mean"].astype(np.float32) - again["mean"].astype(np.float32)).max(initial=0))}
        buf["by_layer"].append(got["mean"])
        buf["last"].append(got["last"])
        metas.append({k: ep[k] for k in ("key", "arm", "run", "task", "epoch", "attempt_index", "attempt_kind", "redrawn_because", "policy_called")}
                     | {"n_tokens": len(ep["ids"]), "n_rows": len(kept), **timing,
                        "calls": [{k: v for k, v in c.items() if k != "sha256"} | {"span": [f, r], "n_span_tokens": r - f + 1, "row": j}
                                  for j, (c, f, r) in enumerate(kept)]})
        if len(metas) == cfg["chunk_size"]:
            flush()
    flush()
    return {"run": cfg["run"], "shard": shard, "epochs": len(mine), "done_before": len(mine) - len(todo), "chunks": stats, "gpu": gpu,
            "span_s": span_s, "load_s": load_s, "seconds": round(time.time() - t0), "max_peak_gib": max(peaks, default=None),
            "command_rows": n_rows, "span_fail": dict(fails), "reproduction_worst": worst}


@app.local_entrypoint()
def main(test: bool = False, only: str = "", attempt: int = 1):
    import io

    import yaml

    git = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
           "dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=HERE).stdout.strip())}
    if git["dirty"]:
        raise SystemExit("uncommitted tracked changes")
    from probe_shift.models import QWEN35_27B

    jobs, cfgs = [], []
    for name, (arms, shards) in CONFIGS.items():
        if only and name != only:
            continue
        orig = yaml.safe_load(subprocess.run(["git", "show", f"95f029dc:experiments/2026-10-08_q3_onpolicy_figures/config/{name}"],
                                             capture_output=True, text=True, check=True, cwd=HERE).stdout)
        cfg = {k: orig[k] for k in ("inputs", "inputs_sha256", "model", "chunk_size", "pad_to")} | {
            "orig_run": orig["run"], "run": f"2026-10-13_sabavg_{orig['run']}" + ("_test" if test else ""), "arms": arms, "shards": shards,
            "layer": LAYER, "revision": QWEN35_27B["revision"], "attempt": attempt, "test": test, "source_config": name}
        if cfg["model"] != QWEN35_27B["name"]:
            raise ValueError(f"{name}: {cfg['model']} is not the pinned Qwen3.5-27B")
        settings = {k: v for k, v in cfg.items() if k != "attempt"}
        try:
            stored = json.loads(b"".join(acts_vol.read_file(f"{cfg['run']}/config.json")))
        except (FileNotFoundError, modal.exception.NotFoundError):
            stored = None
        if stored is not None and stored != settings:
            raise ValueError(f"{cfg['run']} exists with other settings; use a new run name")
        with acts_vol.batch_upload(force=True) as b:
            if stored is None:
                b.put_file(io.BytesIO(json.dumps(settings, indent=1).encode()), f"{cfg['run']}/config.json")
            b.put_file(io.BytesIO(json.dumps({"attempt": attempt, "git": git}, indent=1).encode()), f"{cfg['run']}/attempt_{attempt}.json")
        cfgs.append(cfg)
        jobs += [(cfg, s) for s in range(1 if test else shards)]
    res = list(extract_shard.starmap(jobs, return_exceptions=True))
    out = {"git": git, "app": APP, "configs": cfgs, "shards": [r if isinstance(r, dict) else {"error": repr(r)[:500]} for r in res]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"extract_command_mean_{'test_' if test else ''}a{attempt}_{stamp}.json"
    path.open("x").write(json.dumps(out, indent=1))
    print(json.dumps(out["shards"], indent=1)[:4000])
    print("->", path)
    if not test:
        for cfg in cfgs:
            mine = [s for s in out["shards"] if s.get("run") == cfg["run"]]
            if len(mine) == cfg["shards"] and not any("error" in s for s in mine):
                with acts_vol.batch_upload(force=True) as b:
                    b.put_file(io.BytesIO(json.dumps({"epochs": sum(s["epochs"] for s in mine), "attempt": attempt}).encode()), f"{cfg['run']}/complete.json")
        failed = [s for s in out["shards"] if "error" in s]
        if failed:
            raise SystemExit(f"{len(failed)} shards failed; rerun with --attempt {attempt + 1} (done epochs are kept)")
