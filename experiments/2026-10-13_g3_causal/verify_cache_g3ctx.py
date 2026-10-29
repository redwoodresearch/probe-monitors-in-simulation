"""G3 causal test, the cache check at the logit level (PLAN.md here; POST HOC). The extraction's own check compared
prefix-cached and full-forward activations as max |Δ| / max |full| over layers, and failed its 2e-2 tolerance: median 0.0206,
max 0.209 over 64 rows. This job measures what matters for the analysis, the probe logits. It runs on one H200.

Rows, in this order:
- the run behind the 0.209 reading: shard 27's first run in the extractor's order (its row 1 gave 0.209);
- per arm, the run with the most kept rows, so later rows sit after many cache crops;
- a seeded sample (seed 20261013) of 3 runs per arm.
All of each run's kept rows are read in the extractor's order, so a row's index is the number of crops before it. Each row is read three ways: the prefix-cached forward (as extracted), a full forward without the
cache, and the stored dump row. Its logit is the mean over v != f of G3's frozen model (f, v), at layer CHOSEN[f].

Rule, written before the run: the cache path is accepted if all three hold:
- the mean Δlogit (cached − full) over all sampled rows is within ±0.01;
- the max |Δlogit| is ≤ 0.1;
- the mean Δlogit over rows after 20 or more crops is within ±0.01 (a stale cache would bias every later row). Otherwise the analysis does not use the cached extraction, and it is re-extracted
without the cache.

The helpers (prep through ids_sha) are copied verbatim from run_extract_g3ctx.py, along with its constants.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run experiments/2026-10-13_g3_causal/verify_cache_g3ctx.py \
      --runs experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json --fit-runs <G3 fit runs.json>
"""
import bisect
import hashlib
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-g3ctx_extract_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"  # read-only: #36's staged inputs
G3OUT = "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"  # read-only: G3's dumps (the kept rows)
G3WORK = "probe-shift_aswin-surya_sub-q2-g3_work_kill-2026-11-12"  # read-only: G3's fit (the frozen probes)
OUT = "probe-shift_aswin-surya_sub-q2-g3ctx_acts_kill-2026-11-12"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
DUMP = "2026-10-06_la_segment-end_qwen3-32b_v1"
G3RUN = {"real-honest": "2026-10-13_g3_real_seed-oss-36b_v1", "real-attack-e1": "2026-10-13_g3_real_seed-oss-36b_v1",
         "real-attack-e2": "2026-10-13_g3_real_seed-oss-36b_v1", "sim-attack": "2026-10-13_g3_sim_seed-oss-36b_v1"}
RUN = "2026-10-13_g3_ctx_seed-oss-36b_v1"
FIT_RUN = "2026-10-13_111525"
MODEL, REVISION, CTX, H, N_LAYERS = "ByteDance-Seed/Seed-OSS-36B-Instruct", "497f1dca95ebdec98e41d517b9f060ee753c902f", 65536, 5120, 64
LAYERS = [16, 24, 32, 40, 48, 56, 63]
CHOSEN = {0: 40, 1: 63, 2: 63, 3: 63, 4: 63}  # forecast.json's chosen layer by test fold
ARMS = ("real-honest", "real-attack-e1", "real-attack-e2", "sim-attack")
EXPECT = {"real-honest": 990, "real-attack-e1": 939, "real-attack-e2": 939, "sim-attack": 939}
GPU, CHUNK, TIMEOUT_S, SHARDS, VERIFY_ROWS, VERIFY_TOL = "H200", 25, 3 * 3600, 32, 2, 2e-2
SENT = "G3SPAN{:06d}Q"
PIP_CPU = ["transformers==4.57.1", "jinja2==3.1.6", "huggingface_hub[hf_transfer]==0.35.3", "numpy"]
PIP_GPU = ["torch==2.8.0", "accelerate==1.10.1", *PIP_CPU]
PIN = {"runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",  # #36's runs.json
       "fit runs.json": "5c987d6a79315344c76ccad0f71c91a4150e762952109838fdbed43819872a00",  # forecast.json's "inputs of x"
       "forecast.json": "4243f2cd38d93d33e5b63ef08bc7e3fb3b387198af43e667c8821966504fff77"}
PROBES = {16: "70aeb87dbc2c372a7ed9103d35be21fea6cd3b52383d287e30fd37954b706343", 24: "f4d24e77014914f7746912e709c0dbc14a4559d12b57d2756a42996758a1b790",
          32: "7f18518ad0565de58551009d795279875aa100aaa7a9037554165c0d12f8f980", 40: "5d41c69b9cb118e783be004760618cbbbc0b3069c7043671853c819ad2fabc00",
          48: "41dbd919a0110a6d892daacea5a9529314161919a8eec4d6046770a9f136a608", 56: "1f66a1813761183ab4b1995159b5753c7e5027273070459c87fab9dbc34042b3",
          63: "3b3646b0edb49b2ae738c359c7965bc846ea756024f6f3d62ba26058b18ebf9e"}  # forecast.json's "probes sha256 by layer"

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


def positions(tok, msgs, q, special, special_text=()):
    """One staged transcript under the reader's template. q: Qwen3-32B's output_end positions, {"step,action": [kept, tool]}.
    Returns ids, the meta positions (every call's call_end, not kept; every output's output_end, kept under C2 and below
    CTX) and per output (qwen kept, readout index, readout is a newline token, a token crosses the span's end, the output
    ends in a newline). Raises Fail(stage) on a span that can't be located verbatim. Required rows (Qwen3-32B-kept
    outputs) without a kept readout here are listed by "step,action" in two kinds: no eligible token in the span, and any
    other reason (a readout at or past the context length, or a Qwen3-32B position with no span here)."""
    msgs = prep(msgs)
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
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            step, action = step + 1, 0
            calls += [(step, j, (tc.get("function") or {}).get("name")) for j, tc in enumerate(m.get("tool_calls") or [])]
        elif m["role"] == "tool":
            s_ = SENT.format(k); j = stext.find(s_, pos)
            if j < 0 or stext.count(s_) != 1:
                raise Fail("a tool output not located")
            a = j + shift; b = a + len(m["content"])
            if text[a:b] != m["content"]:
                raise Fail("a tool output not verbatim")
            spans.append((step, action, a, b)); action += 1; k += 1
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
    out = [{"kind": "call_end", "step": s, "action": i, "tool": name, "token": None, "kept": False, "token_text": None} for s, i, name in calls]
    tt = lambda i: text[starts[i]:ends[i]]  # noqa: E731
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
        per.append((qk, [(r, None if r is None else tt(r)) for r in (r0, r2)], crosses, text[a:b].endswith("\n")))
    got = {f"{p['step']},{p['action']}" for p in out if p["kind"] == "output_end" and p["kept"]}
    lost = [k_ for k_, v in q.items() if v[0] and k_ not in got]  # Qwen-kept outputs with no kept readout here
    return {"ids": ids, "n_tokens": len(ids), "positions": out, "outputs": per,
            "qwen_kept_no_content": sum(k_ in no_content for k_ in lost), "qwen_kept_missing_other": sum(k_ not in no_content for k_ in lost),
            "left_out": sorted(lost, key=lambda k_: tuple(map(int, k_.split(",")))), "left_out_other": sorted(k_ for k_ in lost if k_ not in no_content),
            "required": sum(1 for v in q.values() if v[0]), "outputs_quoting_special": quoting}


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


def filename(key: str) -> str:  # the extractors' file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


# ---------- new in this copy ----------
def g3_kept(arm, stem):
    """G3's kept output_end rows for this run, in its meta's order: [(step, action, tool)]."""
    m = json.loads((Path("/g3") / G3RUN[arm] / "meta" / arm / f"{stem}.json").read_text())
    return [(p["step"], p["action"], p.get("tool")) for p in m["positions"] if p["kind"] == "output_end" and p["kept"]]


def minis(msgs, kept):
    """The prefix (messages before the first assistant turn) and, per kept (step, action), the minimal transcript: the
    prefix, an assistant turn holding only that output's tool call, then the output. step and action are positions()'s:
    step counts assistant turns, action counts tool outputs after the latest one."""
    first = next((i for i, m in enumerate(msgs) if m["role"] == "assistant"), None)
    if first is None:
        raise Fail("no assistant turn")
    prefix, by, step, action, cur = msgs[:first], {}, -1, 0, None
    for m in msgs:
        if m["role"] == "assistant":
            step, action, cur = step + 1, 0, m
        elif m["role"] == "tool":
            by[(step, action)] = (cur, m); action += 1
    rows = []
    for s, a, tool in kept:
        if (s, a) not in by:
            raise Fail("a kept row with no output here")
        am, tm = by[(s, a)]
        tcs, tid = am.get("tool_calls") or [], tm.get("tool_call_id")
        tc = next((t for t in tcs if tid and t.get("id") == tid), None) or (tcs[a] if a < len(tcs) else None)
        if tc is None:
            raise Fail("no tool call for a kept output")
        rows.append((s, a, tool, prefix + [{"role": "assistant", "content": "", "tool_calls": [tc]}, tm]))
    return prefix, rows


def render_run(tok, special, special_text, it, kept):
    """Every kept row's minimal rendering: ids, readout index, sha256 of the rendered text; plus the run's shared prefix
    length P (the longest common token prefix of the prefix rendering and every row's rendering)."""
    prefix, rows = minis(it["messages"], kept)
    pre = tok(tok.apply_chat_template(prep(prefix), tokenize=False), add_special_tokens=False)["input_ids"]
    out, lost = [], []
    for s, a, tool, mini in rows:
        try:
            r = positions(tok, mini, {"0,0": [True, tool]}, special, special_text)
        except Fail as e:
            lost.append({"step": s, "action": a, "reason": f"render failure: {e.stage}", "render failure": True}); continue
        oe = [p for p in r["positions"] if p["kind"] == "output_end"]
        if len(oe) != 1 or not oe[0]["kept"]:
            lost.append({"step": s, "action": a, "reason": "no eligible readout"}); continue
        text = tok.apply_chat_template(prep(mini), tokenize=False)
        out.append({"step": s, "action": a, "tool": tool, "ids": r["ids"], "readout": oe[0]["token"],
                    "sha256": hashlib.sha256(text.encode()).hexdigest()})
    P = len(pre)
    for x in out:
        n = 0
        while n < min(P, len(x["ids"])) and x["ids"][n] == pre[n]:
            n += 1
        P = min(P, n)
    P = min([P] + [x["readout"] for x in out])  # the readout always lies in the suffix
    return pre[:P], out, lost


def work_items(cohort):
    base = staged(DUMP, False)
    return [(r["arm"], r["filename"], base[(r["arm"], r["key"])]) for r in cohort]


def ids_sha(pairs) -> str:
    return hashlib.sha256(json.dumps(sorted([list(p) for p in pairs])).encode()).hexdigest()



VAPP = "probe-shift_aswin-surya_sub-q2-g3ctx_verify_kill-2026-11-12"
app = modal.App(VAPP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
g3out = modal.Volume.from_name(G3OUT, version=2).read_only()
g3work = modal.Volume.from_name(G3WORK, version=2).read_only()
hf_cache = modal.Volume.from_name(HF_CACHE)
out_vol = modal.Volume.from_name(OUT, version=2).read_only()
gpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP_GPU).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})


@app.function(image=gpu_image, gpu=GPU, memory=65536, volumes={"/hf": hf_cache, "/out": out_vol, "/acts": acts, "/g3": g3out, "/work": g3work}, timeout=3600, retries=0)
def verify(items: list[dict], cohort: list[dict], fold_of: dict) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    try:
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        special, special_text = special_ids(tok), special_texts(tok)
        lm = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
        base = lm.model
        Z, shas = {}, {}
        for l in sorted(set(CHOSEN.values())):
            f_ = Path("/work") / "runs" / FIT_RUN / f"probes_L{l}.npz"
            shas[str(l)] = hashlib.sha256(f_.read_bytes()).hexdigest()
            with np.load(f_) as zf:
                Z[l] = {k: zf[k] for k in zf.files}
        want, got = [None], {}

        def hook_for(layer):
            def hook(module, args, output):
                if want[0] is not None:
                    h = output[0] if isinstance(output, tuple) else output
                    got[layer] = h[0, want[0]].to(torch.float16).cpu().numpy()
            return hook
        handles = [base.layers[l].register_forward_hook(hook_for(l)) for l in LAYERS]

        def logit(vec, f):
            x = vec[CHOSEN[f]].astype(np.float32)
            return float(np.mean([((x - Z[CHOSEN[f]][f"output_end__task_nested_{f}_{v}__mean"]) / Z[CHOSEN[f]][f"output_end__task_nested_{f}_{v}__std"])
                                  @ Z[CHOSEN[f]][f"output_end__task_nested_{f}_{v}__w"] + np.float32(Z[CHOSEN[f]][f"output_end__task_nested_{f}_{v}__b"]) for v in range(5) if v != f]))
        base_items = staged(DUMP, False)
        order = work_items(cohort)  # the extractor's order: longest first, dealt round-robin over 32 shards
        order.sort(key=lambda it: (-(it[2].get("tokens") or 0), it[0], it[2]["key"]))
        a27, s27, x27 = order[27::32][0]
        items = [{"arm": a27, "key": x27["key"], "filename": s27, "tag": "shard 27's first run (the 0.209 reading)", "fold": fold_of[f"{a27}|{x27['key']}"]}] + items
        out = []
        for it in items:
            st = base_items[(it["arm"], it["key"])]
            pre, rows, lost = render_run(tok, special, special_text, st, g3_kept(it["arm"], it["filename"]))
            P = len(pre)
            m = json.loads((Path("/out") / RUN / "meta" / it["arm"] / f"{it['filename']}.json").read_text())
            X = {l: np.load(Path("/out") / RUN / "by_layer" / f"L{l}" / f"{m['chunk']}.npy", mmap_mode="r") for l in sorted(set(CHOSEN.values()))}
            with torch.inference_mode():
                cache = DynamicCache() if P else None
                if P:
                    want[0] = None
                    base(input_ids=torch.tensor([pre], device="cuda"), past_key_values=cache, use_cache=True)
                for j, x in enumerate(rows):
                    want[0] = x["readout"] - P; got.clear()
                    if P:
                        base(input_ids=torch.tensor([x["ids"][P: x["readout"] + 1]], device="cuda"), past_key_values=cache, use_cache=True)
                        cache.crop(P)
                    else:
                        base(input_ids=torch.tensor([x["ids"][: x["readout"] + 1]], device="cuda"), use_cache=False)
                    cv = dict(got)
                    want[0] = x["readout"]; got.clear()
                    base(input_ids=torch.tensor([x["ids"][: x["readout"] + 1]], device="cuda"), use_cache=False)
                    fv = dict(got)
                    r = m["rows"][j]
                    if (r["step"], r["action"], r["sha256"]) != (x["step"], x["action"], x["sha256"]):
                        raise Fail("the stored row differs from the re-rendered one")
                    sv = {l: np.asarray(X[l][m["row_start"] + j]) for l in X}
                    f = it["fold"]
                    rel = max(float(np.abs(cv[l].astype(np.float32) - fv[l].astype(np.float32)).max() / (np.abs(fv[l].astype(np.float32)).max() + 1e-6)) for l in LAYERS)
                    cos = {str(l): float(np.dot(cv[l].astype(np.float64), fv[l].astype(np.float64)) / (np.linalg.norm(cv[l].astype(np.float64)) * np.linalg.norm(fv[l].astype(np.float64)))) for l in (40, 63)}
                    out.append({"arm": it["arm"], "run": it["tag"], "row": j, "tokens": x["readout"] + 1, "prefix": P,
                                "logit cached": logit(cv, f), "logit full": logit(fv, f), "logit stored": logit(sv, f),
                                "max rel diff (the extraction's metric)": rel, "cosine cached vs full": cos})
                want[0] = None
        for h_ in handles:
            h_.remove()
        return {"probes sha256": shas, "rows": out}
    except Exception as e:  # nothing from the exception (which may quote staged text) crosses the boundary
        raise RuntimeError(f"verify failed: {type(e).__name__}: {getattr(e, 'stage', '')}") from None


@app.local_entrypoint()
def main(runs: str, fit_runs: str):
    import statistics
    here = Path(__file__).resolve().parent
    for f in (Path(__file__).resolve(), here / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=here, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=here).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    if hashlib.sha256(Path(runs).read_bytes()).hexdigest() != PIN["runs.json"] or hashlib.sha256(Path(fit_runs).read_bytes()).hexdigest() != PIN["fit runs.json"]:
        raise SystemExit("not the pinned runs.json files")
    rs = [r for r in json.loads(Path(runs).read_text()) if r["arm"] in ARMS]
    fr = json.loads(Path(fit_runs).read_text())
    fold = {(r["arm"], r["key"]): int(r["fold"]) for r in fr}; e2cell = {r["cell"]: int(r["fold"]) for r in fr if r["arm"] == "real-attack-e2"}
    rr = json.loads((here / "results" / "ctx_render_2026-10-13_134134.json").read_text())
    nrows = {(x["arm"], x["key"]): x.get("rendered", 0) for x in rr["runs"]}
    launch = json.loads((here / "results" / "ctx_launch_2026-10-13_g3_ctx_seed-oss-36b_v1_2026-10-13_134423.json").read_text())
    if launch["shards"] != 32:
        raise SystemExit("the launch record's shard count is not the 32 this job assumes")
    rng = random.Random(20261013)
    pick = []
    for a in ARMS:
        arm_runs = sorted((x for x in rs if x["arm"] == a), key=lambda x: x["key"])
        pick.append((max(arm_runs, key=lambda x: (nrows[(a, x["key"])], x["key"])), f"most rows {a}"))
        for r in rng.sample(arm_runs, 3):
            pick.append((r, f"seeded {a}"))
    fold_of = {f"{r['arm']}|{r['key']}": (e2cell[r["cell"]] if r["arm"] == "sim-attack" else fold[(r["arm"], r["key"])]) for r in rs}
    items = [{"arm": r["arm"], "key": r["key"], "filename": r["filename"], "tag": tag, "fold": fold_of[f"{r['arm']}|{r['key']}"]} for r, tag in pick]
    cohort = [{"arm": r["arm"], "key": r["key"], "filename": r["filename"]} for r in rs]
    res = verify.remote(items, cohort, fold_of)
    if any(res["probes sha256"][str(l)] != PROBES[l] for l in set(CHOSEN.values())):
        raise SystemExit("the probes read differ from forecast.json's")
    rows = res["rows"]
    d = [r["logit cached"] - r["logit full"] for r in rows]; ds = [r["logit stored"] - r["logit cached"] for r in rows]
    summary = {"rows": len(rows), "runs": len(items), "mean Δlogit (cached - full)": statistics.mean(d), "max |Δlogit| (cached - full)": max(map(abs, d)),
               "mean |Δlogit| (cached - full)": statistics.mean(map(abs, d)), "max |Δlogit| (stored - cached recompute)": max(map(abs, ds)),
               "max rel diff (the extraction's metric): median, max": [statistics.median(r["max rel diff (the extraction's metric)"] for r in rows), max(r["max rel diff (the extraction's metric)"] for r in rows)],
               "min cosine cached vs full (L40, L63)": [min(r["cosine cached vs full"]["40"] for r in rows), min(r["cosine cached vs full"]["63"] for r in rows)]}
    late = [r["logit cached"] - r["logit full"] for r in rows if r["row"] >= 20]
    summary["rows after 20 or more crops"] = len(late)
    summary["mean Δlogit (cached - full), rows after 20 or more crops"] = statistics.mean(late) if late else None
    by_bin = {}
    for r in rows:
        b = "0" if r["row"] == 0 else "1-4" if r["row"] < 5 else "5-19" if r["row"] < 20 else "20-49" if r["row"] < 50 else "50+"
        by_bin.setdefault(b, []).append(r["logit cached"] - r["logit full"])
    summary["Δlogit by crops before the row: [rows, mean, max |Δ|]"] = {b: [len(v), statistics.mean(v), max(map(abs, v))] for b, v in by_bin.items()}
    first = [r for r in rows if r["run"].startswith("shard 27")]
    summary["shard 27's first run: rows 0-1 [Δlogit, max rel diff]"] = [[r["logit cached"] - r["logit full"], r["max rel diff (the extraction's metric)"]] for r in first[:2]]
    summary["rule"] = "accept if |mean Δlogit| <= 0.01, max |Δlogit| <= 0.1, and |mean Δlogit| over rows after >= 20 crops <= 0.01"
    summary["verdict"] = "accept" if (abs(summary["mean Δlogit (cached - full)"]) <= 0.01 and summary["max |Δlogit| (cached - full)"] <= 0.1
                                      and (not late or abs(summary["mean Δlogit (cached - full), rows after 20 or more crops"]) <= 0.01)) else "reject"
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"plan": "PLAN.md (this folder), POST HOC", "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip(),
           "app": VAPP, "summary": summary, "items": [{k: v for k, v in it.items() if k != "key"} | {"key sha256": hashlib.sha256(it["key"].encode()).hexdigest()[:16]} for it in items],
           "rows": rows, "probes sha256": res["probes sha256"]}
    with open(dest := here / "results" / f"ctx_verify_cache_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(summary, indent=1)); print(dest)
