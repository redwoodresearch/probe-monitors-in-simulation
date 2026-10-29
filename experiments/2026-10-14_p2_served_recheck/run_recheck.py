"""Independent recheck of the served-context p2 scoring (claims 1-7 of 4f's brief; RULE.md here). Modal CPU only.

Written from the brief, RULE.md / ALIGNMENT.md of experiments/2026-10-14_p2_replay_check, 09's Appends 28-28e, the probes'
own RULEs and weight files, and the record schemas. It does not import or copy the served scorer or the stager.

It never decodes token ids to text and prints no score values: only counts, hashes and max |diff|. The one vocabulary
property it uses is structural (whether a token's bytes contain / start / end with a newline), to split the system turn's
tools block into lines; no transcript token is turned into text. No thresholds, catches, AUROCs or sim-vs-real numbers
are computed. The record is written inside the job to
  probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13:/2026-10-14_p2_served_recheck/recheck_<stamp>.json

  cd <repo worktree>
  uv run --no-project --with 'modal>=1.2' --with 'numpy>=2' modal run --detach \
      experiments/2026-10-14_p2_served_recheck/run_recheck.py [--smoke]
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-p2probes_servedrecheck_kill-2026-11-13"
V_ACTS = "probe-shift_aswin-surya_sub-p2served_acts_kill-2026-11-14"
V_CW = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
V_Q3 = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
V_Q4 = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
V_RL = "probe-shift_aswin-surya_sub-phase2_request-log_kill-2026-11-06"
V_HF = "probe-shift_aswin-surya_hf-cache"
V_OUT = "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13"
OUT_DIR = "2026-10-14_p2_served_recheck"
HF_SNAP = "hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"

STAGE = "served_2026-10-14_200651"
EXTRACT = {L: f"{STAGE}_extract_{L}_v1" for L in ("A0", "A", "B")}
LAYOUTS = ("A0", "A", "B")
WORLDS = ("run1", "run2", "sim", "honest_a", "honest_b", "honest_c")
ATTACK = ("run1", "run2", "sim")
READOUTS = ("reply_mean", "reply_last", "command", "call_end_content")
KIND_OF = {"reply_mean": "output_end", "reply_last": "output_end", "command": "command", "call_end_content": "call_end"}
PROBES = ("fibs", "refusal", "harmfulness", "highstakes", "apollo_d2_tight")
FIXED = (15, 77)  # "0" and "n": content-only call_end drops these readout tokens (ALIGNMENT.md)
SEED = 20261014
N_SAMPLE_PER_WORLD = 10
TOL = 1e-6
EPS32 = 2.0 ** -24  # float32 unit roundoff

# Claimed values under check (4f's brief; manifest slim copy @ d024bf7a0).
CLAIM = {
    "manifest_sha256": {"A0": "c7a0dddafd77baf0f3d13d4951062b92b84703d1ba583ec618b6f578351c70ee",
                        "A": "56b9459b91f2770b89075b97aec6dd738c9be11d502a895755fae88bbda31741",
                        "B": "7755bd63b9567130da24ba2280a4a7f504b591414a90c582dd3a78394d2dd75e"},
    "unit_assignment": ("served/unit_assignment_2026-10-14_204921.json",
                        "4dc6a4039f11c4e901e900892aca1579e6e2280104ac2b8070b902e6d10338ae", 93139),
    "units": {"reply_mean": 28588, "reply_last": 28588, "command": 32291, "call_end_content": 28569},
    "runs": 846,
    "28d_reply": {"run1": (47, 3292, 21), "run2": (35, 3047, 18), "sim": (44, 3384, 18), "honest_a": (18, 6295, 14),
                  "honest_b": (34, 6179, 19), "honest_c": (39, 6391, 24)},
    "28d_command_mismatched": {"run1": 16, "run2": 19, "sim": 19, "honest_a": 16, "honest_b": 16, "honest_c": 16},
    "28d_command_units": {"run1": 3744, "run2": 3475, "sim": 3827, "honest_a": 7040, "honest_b": 7124, "honest_c": 7081},
    "28d_call_end_units": {"run1": 3324, "run2": 3080, "sim": 3349, "honest_a": 6280, "honest_b": 6204, "honest_c": 6332},
}
# Git pins (read on the laptop; the job gets bytes and checks sha256s).
PIN_SIMH = "f14b1253"  # run_simh_scores.py: LINEAR, CONV, RECORDS
SIMH = "experiments/2026-10-14_p2_simh_scores/run_simh_scores.py"
RECORDS = {"fibs": ("d74ccaba0", "experiments/2026-10-14_p2_fibs_pair/results/fibs_scores_2026-10-14_124948.json", None),
           "refusal": ("a8d786425", "experiments/2026-10-14_p2_refusal_harm/results/scores_2026-10-14_145748.json", "refusal"),
           "harmfulness": ("a8d786425", "experiments/2026-10-14_p2_refusal_harm/results/scores_2026-10-14_145748.json", "harmfulness"),
           "highstakes": ("3d04b7936", "experiments/2026-10-14_p2_highstakes/results/scores_2026-10-14_133519.json", "highstakes"),
           "apollo_d2_tight": ("54ea9724", "experiments/2026-10-14_p2_apollo/results/scores_2026-10-14_145858.json", "apollo_d2_tight")}
LINEAR = {"refusal": "experiments/2026-10-14_p2_refusal_harm/results/h200_check_2026-10-14_145255.json",
          "harmfulness": "experiments/2026-10-14_p2_refusal_harm/results/h200_check_2026-10-14_145255.json",
          "highstakes": "experiments/2026-10-14_p2_highstakes/results/read_fit_2026-10-14_131233.json",
          "apollo_d2_tight": "experiments/2026-10-14_p2_apollo/results/fit_tight_2026-10-14_145508.json"}
CONV = "experiments/2026-10-14_q2_fibs_probe/results/convert_fibs_linear_2026-10-14_115759.json"
PIN_CHECKED = "d024bf7a0da5a9df7d5c8fe47291bda6e778190c"  # origin/aswin/p2-replay-check head (records under check)
STAGE_REC = ("experiments/2026-10-14_p2_replay_check/results/stage_served_2026-10-14_200651.json",
             "3f097843eec6bfd9f286a5a7e6e12d1d1c695c038eeba178b3ca2b85f313213f")
MANIFEST_B = ("46ca25fc", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_manifest_v1.json",
              "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc")

# Token ids (Qwen3.5-27B @ fc05daec; checked against the tokenizer inside the job).
IM_START, IM_END, THINK_O, THINK_C = 248045, 248046, 248068, 248069
TOOL_CALL_O, TR_O, TR_C = 248058, 248066, 248067
ASSISTANT, NL, NLNL = 74455, 198, 271
TOOLS_O, TOOLS_C = (27, 15449, 29), (510, 15449, 29)

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.3", "tokenizers==0.22.1")
ro = lambda n: modal.Volume.from_name(n).read_only()  # noqa: E731
out_vol = modal.Volume.from_name(V_OUT)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------------------------------- shared helpers (job)
def _np():
    import numpy as np

    return np


def i32(x):
    return _np().asarray(x, dtype="<i4")


def read_jsonl(path, keep=None, keyfield="key"):
    """Parse a jsonl file; with `keep` (a set), parse only lines whose key (found in the first 2 kB) is in it."""
    import re

    rx = re.compile(r'"' + keyfield + r'":\s*"([^"]+)"')
    out = []
    with open(path, "rb") as f:
        for line in f:
            if keep is not None:
                m = rx.search(line[:2048].decode("utf-8", "replace"))
                if not m or m.group(1) not in keep:
                    continue
            out.append(json.loads(line))
    return out


def ukey(c):
    """A unit's key (irid, kind, src_irid, call_index); None for the few legacy rule-1 rows without kind/src_irid/call_index."""
    if any(k not in c for k in ("kind", "src_irid", "call_index")):
        return None
    return (c["irid"], c["kind"], c["src_irid"], c["call_index"])


def sysend(ids) -> int:
    np = _np()
    w = np.flatnonzero(ids == IM_END)
    if not len(w):
        raise ValueError("no <|im_end|> in ids")
    return int(w[0]) + 1


def cprefix(a, b) -> int:
    np = _np()
    n = min(len(a), len(b))
    d = np.flatnonzero(a[:n] != b[:n])
    return int(d[0]) if len(d) else n


def find_sub(hay, needle, start: int = 0) -> int:
    hb, nb = i32(hay).tobytes(), i32(needle).tobytes()
    if not nb:
        return start
    s = max(start, 0) * 4
    while True:
        k = hb.find(nb, s)
        if k < 0:
            return -1
        if k % 4 == 0:
            return k // 4
        s = k + 1


def empty_think(a):
    """B from A: each assistant turn's OPENING think block (<|im_start|>assistant\\n<think>) gets interior [271] if it is
    closed by </think> before the next <|im_start|>/<|im_end|>; unclosed blocks stay. Returns (b, blocks, stats)."""
    np = _np()
    hdr = np.flatnonzero((a[:-3] == IM_START) & (a[1:-2] == ASSISTANT) & (a[2:-1] == NL) & (a[3:] == THINK_O))
    stops = np.flatnonzero((a == IM_START) | (a == IM_END))
    closes = np.flatnonzero(a == THINK_C)
    blocks, unclosed, already = [], 0, 0
    for h in hdr:
        j = int(h) + 3  # the <think> position
        ic, im = np.searchsorted(closes, j + 1), np.searchsorted(stops, j + 1)
        k = int(closes[ic]) if ic < len(closes) else None
        m = int(stops[im]) if im < len(stops) else None
        if k is None or (m is not None and m < k):
            unclosed += 1
            continue
        if k - (j + 1) == 1 and a[j + 1] == NLNL:
            already += 1
        blocks.append((j + 1, k))  # interior [j+1, k)
    parts, last = [], 0
    for s, e in blocks:
        parts += [a[last:s], np.array([NLNL], dtype=a.dtype)]
        last = e
    parts.append(a[last:])
    b = np.concatenate(parts)
    stats = {"blocks emptied": len(blocks), "already empty": already, "unclosed kept": unclosed,
             "interior tokens": int(sum(e - s for s, e in blocks)),
             "think markers outside opening blocks": int(((a == THINK_O) | (a == THINK_C)).sum()) - 2 * len(blocks) - unclosed,
             "assistant think headers": len(hdr)}
    return b, blocks, stats


def posmap_b(p: int, blocks):
    """A position -> B position; None inside an emptied interior."""
    shift = 0
    for s, e in blocks:
        if p >= e:
            shift += (e - s) - 1
        elif p >= s:
            return None
    return p - shift


# ----------------------------------------------------------------------------------------------------- projection
@app.function(image=image, volumes={"/acts": ro(V_ACTS), "/q4": ro(V_Q4)}, cpu=4, memory=16384, timeout=600, retries=0)
def project(src: str, layer: int, vecs: dict, chunks: list) -> dict:
    """Project every row of the listed chunk files at one layer. vecs: name -> {kind, tensors (lists)}.
    Returns {chunk_id: {name: {"f32", "f64", "absdot"}}}, chunk_id = folder/chunk. absdot = sum_i |x_i w_i| (float64), the
    scale of float32 rounding in the dot product."""
    import numpy as np

    V = {}
    for name, v in vecs.items():
        V[name] = (v["kind"], {k: np.asarray(x) for k, x in v["tensors"].items()})
    out = {"_rows": 0, "_nonfinite": 0}
    for folder, chunk in chunks:
        base = f"/acts/{folder}" if src in LAYOUTS else f"/q4/{folder}"
        X = np.load(f"{base}/by_layer/L{layer}/{chunk}.npy")
        if X.dtype != np.float16 or X.ndim != 2 or X.shape[1] != 5120:
            raise ValueError(f"{folder}/{chunk} L{layer}: unexpected {X.dtype} {X.shape}")
        out["_nonfinite"] += int((~np.isfinite(X)).any(axis=1).sum())
        Xf, Xd = X.astype(np.float32), X.astype(np.float64)
        Xa = np.abs(Xd)
        res = {}
        for name, (kind, t) in V.items():
            if kind == "fibs":
                w, s, b = t["linear.weight"].reshape(-1), t["input_scale"].reshape(()), t["linear.bias"].reshape(())
                f32 = (Xf / s.astype(np.float32)) @ w.astype(np.float32) + b.astype(np.float32)
                f64 = (Xd / float(s)) @ w.astype(np.float64) + float(b)
                ad = (Xa / abs(float(s))) @ np.abs(w.astype(np.float64)) + abs(float(b))
            elif kind == "linear":
                d = t["direction"].reshape(-1)
                b = t["bias"].reshape(()) if "bias" in t else np.zeros((), np.float32)
                f32 = Xf @ d.astype(np.float32) + b.astype(np.float32)
                f64 = Xd @ d.astype(np.float64) + float(b)
                ad = Xa @ np.abs(d.astype(np.float64)) + abs(float(b))
            elif kind == "unfolded":  # high-stakes: W . ((x - mean) / std) + intercept
                W, mu, sd, c = t["W"].reshape(-1), t["mean"].reshape(-1), t["std"].reshape(-1), t["intercept"].reshape(())
                f32 = ((Xf - mu.astype(np.float32)) / sd.astype(np.float32)) @ W.astype(np.float32) + c.astype(np.float32)
                f64 = ((Xd - mu.astype(np.float64)) / sd.astype(np.float64)) @ W.astype(np.float64) + float(c)
                ad = np.abs((Xd - mu.astype(np.float64)) / sd.astype(np.float64)) @ np.abs(W.astype(np.float64)) + abs(float(c))
            else:
                raise ValueError(kind)
            res[name] = {"f32": f32.astype(np.float32), "f64": f64, "absdot": ad}
        out[f"{folder}/{chunk}"] = res
        out["_rows"] += X.shape[0]
    return out


# ------------------------------------------------------------------------------------- claims 2, 4, 5, 6 per world
@app.function(image=image, volumes={"/acts": ro(V_ACTS), "/q3": ro(V_Q3), "/q4": ro(V_Q4), "/hf": ro(V_HF)},
              cpu=4, memory=32768, timeout=1200, retries=0)
def ids_world(world: str, ctx: dict) -> dict:
    """Layout construction (claim 6) on every staged epoch of one world; per-unit token, position, n and span checks in
    served and layout coordinates; each layout's extraction meta joined to the reader's units by the full unit key; the
    assignment's candidate multiplicity (claim 4); the reader's own unit set for the scored runs (claim 2)."""
    import os
    from collections import Counter, defaultdict

    import numpy as np
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(f"/hf/{HF_SNAP}/tokenizer.json")
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
    marks = {"<|im_start|>": [IM_START], "<|im_end|>": [IM_END], "<think>": [THINK_O], "</think>": [THINK_C],
             "<tool_call>": [TOOL_CALL_O], "<tool_response>": [TR_O], "</tool_response>": [TR_C], "\n\n": [NLNL],
             "\n": [NL], "<tools>": list(TOOLS_O), "</tools>": list(TOOLS_C),
             "<|im_start|>assistant\n": [IM_START, ASSISTANT, NL]}
    bad_marks = [k for k, v in marks.items() if enc(k) != v]
    if bad_marks:
        raise ValueError(f"marker ids differ from the tokenizer: {bad_marks}")
    PARAM_C = enc("</parameter")
    IMPORTANT_C = enc("</IMPORTANT>")
    SUBMIT = [enc('name": "submit')]  # the tool's name field, '{"name": "submit", ...' (pretokens: name | ": | ' "' | submit)
    # structural vocabulary property only: does a token's byte string contain / start / end with "\n"
    bu = {}
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n_ = bs[:], 0
    for b_ in range(256):
        if b_ not in bs:
            bs.append(b_)
            cs.append(256 + n_)
            n_ += 1
    bu = {chr(c_): b_ for b_, c_ in zip(bs, cs, strict=True)}
    V = tok.get_vocab_size(with_added_tokens=True)
    nl_any, nl_start, nl_end = np.zeros(V + 1, bool), np.zeros(V + 1, bool), np.zeros(V + 1, bool)
    special = {t.content for t in tok.get_added_tokens_decoder().values()}
    for i in range(V):
        t_ = tok.id_to_token(i)
        if t_ is None:
            continue
        bb = t_.encode() if t_ in special else bytes(bu.get(ch, 0) for ch in t_)
        nl_any[i], nl_start[i], nl_end[i] = b"\n" in bb, bb.startswith(b"\n"), bb.endswith(b"\n")
    C = Counter()
    fails = defaultdict(list)

    def fail(kind, key):
        C["FAIL " + kind] += 1
        if len(fails[kind]) < 15:
            fails[kind].append(key)

    def check(name, ok, key):
        ok = bool(ok)
        C[name + (": pass" if ok else ": FAIL")] += 1
        if not ok:
            fail(name, key)
        return ok

    scored = set(ctx["scored_keys"])
    # ---- layout inputs
    Ls = ("A0", "A", "B") if world in ATTACK else ("A", "B")
    lay = {}
    for L in Ls:
        recs = read_jsonl(f"/acts/{STAGE}/{L}/inputs/{world}.jsonl")
        lay[L] = {(r["key"], r["epoch"]): r for r in recs}
        for r in recs:
            r["ids"] = i32(r["ids"])
    keysA = set(lay["A"])
    for L in lay:
        check(f"{L}: epoch set == A's", set(lay[L]) == keysA, L)
    C["epochs"] = len(keysA)
    # ---- each layout's extraction meta and chunk index
    xmeta = {}
    for L in Ls:
        cfg = json.load(open(f"/acts/{EXTRACT[L]}/config.json"))
        chunk_idx = {}
        for f in sorted(os.listdir(f"/acts/{EXTRACT[L]}/chunks")):
            cj = json.load(open(f"/acts/{EXTRACT[L]}/chunks/{f}"))
            for x in cj["epochs"]:
                chunk_idx[(f[:-5], x["key"])] = (x["row_start"], x["n_rows"])
        for f in sorted(os.listdir(f"/acts/{EXTRACT[L]}/meta/{world}")):
            m = json.load(open(f"/acts/{EXTRACT[L]}/meta/{world}/{f}"))
            check(f"{L} meta: file name == sha1(key)[:20]", hashlib.sha1(m["key"].encode()).hexdigest()[:20] == f[:-5], f)
            check(f"{L} meta: inputs_sha256 == config's", m.get("inputs_sha256") == cfg["inputs_sha256"].get(f"{world}.jsonl"), f)
            check(f"{L} meta: chunk index has (row_start, n_rows)", chunk_idx.get((m["chunk"], m["key"])) == (m["row_start"], m["n_rows"]), f)
            rows = [c["row"] for c in m["calls"]] + [c["row_mean"] for c in m["calls"] if c.get("row_mean") is not None]
            check(f"{L} meta: rows partition 0..n_rows-1", sorted(rows) == list(range(m["n_rows"])), f)
            xmeta[(L, m["key"], m["epoch"])] = m
    # ---- served stagings (vLLM ids): every epoch of each needed served key
    need = defaultdict(set)
    for (k, e), r in lay["A"].items():
        need[r["served_staging"]].add(k)
    served = {}
    for st, ks in need.items():
        for f in sorted(os.listdir(f"/q3/{st}/inputs")):
            for r in read_jsonl(f"/q3/{st}/inputs/{f}", keep=ks):
                kk = (r["key"], r["epoch"])
                if kk in served:
                    fail("served staging: duplicate (key, epoch)", f"{st}/{f}")
                r["ids"] = i32(r["ids"])
                r["_file"] = f"{st}/{f}"
                served[kk] = r
    served_epochs_of = defaultdict(list)
    served_ukeys = {}
    for (k, e), r in served.items():
        served_epochs_of[k].append(e)
        served_ukeys[(k, e)] = {ukey(c) for c in r["calls"] if ukey(c) is not None and c["readout"] is not None}
    # ---- donors (attack): first epoch of the donor's served key
    donors = {}
    if world in ATTACK:
        dneed = defaultdict(set)
        for d in ctx["donor_of"].values():
            dneed[(d["staging"], d["file"])].add(d["served_key"])
        for (st, f), ks in dneed.items():
            for r in read_jsonl(f"/q3/{st}/inputs/{f}", keep=ks):
                k = r["key"]
                if k not in donors or r["epoch"] < donors[k]["epoch"]:
                    donors[k] = {"epoch": r["epoch"], "ids": i32(r["ids"])}
    # ---- reader: staged ids and extraction meta
    reader_ids, reader_meta = {}, {}
    for src in ctx["reader_sources"]:
        reader_meta[src["run_key"]] = json.load(open(f"/q4/{src['meta']}"))
    by_file = defaultdict(set)
    for src in ctx["reader_sources"]:
        by_file[src["inputs"]].add(src["key"])
    rk_of = {src["key"]: src["run_key"] for src in ctx["reader_sources"]}
    for f, ks in by_file.items():
        for r in read_jsonl(f"/q4/{f}", keep=ks):
            reader_ids[rk_of[r["key"]]] = (i32(r["ids"]), {ukey(c): c for c in r["calls"] if ukey(c) is not None})
    C["reader runs (meta)"] = len(reader_meta)
    C["reader runs (staged ids)"] = len(reader_ids)
    # ---- claim 2: the reader's own units (calls with a row) for the scored runs
    reader_units = {}  # (run_key, uk) -> dict
    for rk, m in reader_meta.items():
        if rk not in scored:
            continue
        rid = reader_ids.get(rk)
        if rid is None:
            fail("reader staged ids missing", rk)
            continue
        ids_r, calls_r = rid
        for c in m["calls"]:
            if c.get("row") is None:
                C["reader calls without a row (beyond the reader's cut)"] += 1
                continue
            uk = ukey(c)
            sc = calls_r.get(uk)
            if not check("reader meta call == its staged call (readout)", sc is not None and sc["readout"] == c["readout"], rk):
                continue
            reader_units[(rk, uk)] = {"kind": c["kind"], "row": c["row"], "row_mean": c.get("row_mean"),
                                      "t_reader": int(ids_r[c["readout"]])}
    # ---- system-turn parts (token ids; lines split at newline-carrying tokens)
    def parts(sysids):
        ts, te = find_sub(sysids, TOOLS_O), find_sub(sysids, TOOLS_C)
        im = find_sub(sysids, IMPORTANT_C, max(te, 0))
        if ts < 0 or te < 0 or im < 0 or sysids[-1] != IM_END:
            return None
        blk = sysids[ts + 3:te]
        cut = np.flatnonzero(nl_any[blk]) + 1
        lines = [tuple(x) for x in np.split(blk, cut) if len(x)]
        sub = [i for i, ln in enumerate(lines) if any(find_sub(np.array(ln), p) >= 0 for p in SUBMIT)]
        return {"head": sysids[:ts + 3], "lines": lines, "mid": sysids[te:im + len(IMPORTANT_C)],
                "content": sysids[im + len(IMPORTANT_C):-1], "submit": sub}

    # ---- per epoch: claim 6 and per-unit checks
    units = {}  # (run_key, uk) -> facts
    think_tot = Counter()
    sys_hash = Counter()
    span_conv = Counter()
    for (k, e) in sorted(keysA):
        A = lay["A"][(k, e)]
        rk = f"{world}|{A['task']}"
        S = served.get((k, e))
        if not check("served staging epoch found", S is not None, k):
            continue
        check("served staging folder == layout's served_staging", S["_file"].split("/")[0] == A["served_staging"], k)
        a, s = A["ids"], S["ids"]
        sa, ss = sysend(a), sysend(s)
        nA = len(a) - sa
        check("A: after the system turn == served ids", nA <= len(s) - ss and np.array_equal(a[sa:], s[ss:ss + nA]), k)
        check("A: cut at a served request boundary", (ss + nA) in {c["n"] for c in S["calls"]}, k)
        if world not in ATTACK:
            check("honest A == served ids (whole epoch, incl. its own system turn)", sa == ss and np.array_equal(a, s[:len(a)]), k)
        else:
            a0 = lay["A0"][(k, e)]["ids"]
            sa0 = sysend(a0)
            check("A0: after the system turn == A's", np.array_equal(a0[sa0:], a[sa:]), k)
            d = ctx["donor_of"].get(f"{k}|{e}")
            if check("donor found", d is not None and d["served_key"] in donors, k):
                D = donors[d["served_key"]]["ids"]
                Dsys, Ssys, Asys, A0sys = D[:sysend(D)], s[:ss], a[:sa], a0[:sa0]
                for fmt, bb in (("int32 LE", i32(Dsys).tobytes()), ("int64 LE", np.asarray(Dsys, "<i8").tobytes()),
                                ("json list", json.dumps([int(x) for x in Dsys]).encode()),
                                ("json list, no spaces", json.dumps([int(x) for x in Dsys], separators=(",", ":")).encode())):
                    sys_hash[f"donor system ids sha256 == stage record ({fmt})"] += sha(bb) == d.get("system_ids_sha256")
                P = {n_: parts(x) for n_, x in (("S", Ssys), ("D", Dsys), ("A", Asys), ("A0", A0sys))}
                if check("system turns parse (head, tools lines, instructions, content)", all(P.values()), k):
                    check("head (through <tools>) equal in served, donor, A, A0",
                          all(np.array_equal(P[x]["head"], P["S"]["head"]) for x in ("D", "A", "A0")), k)
                    check("instructions (</tools> .. </IMPORTANT>) equal in served, donor, A, A0",
                          all(np.array_equal(P[x]["mid"], P["S"]["mid"]) for x in ("D", "A", "A0")), k)
                    check("A0 content == donor content", np.array_equal(P["A0"]["content"], P["D"]["content"]), k)
                    check("A content == donor content", np.array_equal(P["A"]["content"], P["D"]["content"]), k)
                    C["served content != donor content (prompt actually replaced)"] += not np.array_equal(P["S"]["content"], P["D"]["content"])
                    check("A0 tools lines == served tools lines (all, incl. submit)", P["A0"]["lines"] == P["S"]["lines"], k)
                    oks = len(P["S"]["submit"]) == 1 and len(P["D"]["submit"]) == 1
                    C[f"submit-line matches (served, donor) = ({len(P['S']['submit'])}, {len(P['D']['submit'])})"] += 1
                    if check("exactly one submit line in served and in donor tools blocks", oks, k):
                        js, jd = P["S"]["submit"][0], P["D"]["submit"][0]
                        la, ls_ = P["A"]["lines"], P["S"]["lines"]
                        ok_rep = (len(la) == len(ls_) and la[js] == P["D"]["lines"][jd]
                                  and all(la[i] == ls_[i] for i in range(len(ls_)) if i != js))
                        check("A tools lines == served's with ONLY the submit line replaced by the donor's", ok_rep, k)
                        C["served submit line != donor submit line"] += ls_[js] != P["D"]["lines"][jd]
                    C["A system turn == donor's (expected for run1/run2: same non-submit tools)"] += np.array_equal(Asys, Dsys)
                    C["A0 system turn == donor's"] += np.array_equal(A0sys, Dsys)
        # B = empty_think(A)
        b = lay["B"][(k, e)]["ids"]
        eb, blocks, st = empty_think(a)
        for kk2, v in st.items():
            think_tot[kk2] += v
        check("B == A with each opening think block emptied (interior -> [271])", np.array_equal(b, eb), k)
        check("B system turn == A's", np.array_equal(b[:sysend(b)], a[:sa]), k)
        # per-unit checks
        scalls = {ukey(c): c for c in S["calls"] if ukey(c) is not None}
        C["served staging calls without a unit key (legacy rule-1 rows)"] += sum(1 for c in S["calls"] if ukey(c) is None)
        lcalls = {L: {ukey(c): c for c in lay[L][(k, e)]["calls"]} for L in lay}
        check("unit set equal across layouts (per epoch)", all(set(lcalls[L]) == set(lcalls["A"]) for L in lcalls), k)
        mcalls = {L: {ukey(c): c for c in xmeta[(L, k, e)]["calls"]} if (L, k, e) in xmeta else None for L in Ls}
        for L in Ls:
            if check(f"{L}: extraction meta exists for the epoch", mcalls[L] is not None, k):
                check(f"{L}: meta unit set == staged unit set", set(mcalls[L]) == set(lcalls[L]), k)
        rid = reader_ids.get(rk)
        rmeta_calls = {ukey(c): c for c in reader_meta.get(rk, {}).get("calls", []) if c.get("row") is not None}
        for uk, ca in lcalls["A"].items():
            if not check("unit staged once per run", (rk, uk) not in units, rk):
                continue
            sc = scalls.get(uk)
            if not check("unit in its served staging epoch with a served readout", sc is not None and sc["readout"] is not None, rk):
                continue
            r_s, n_s = sc["readout"], sc["n"]
            t_s = int(s[r_s])
            u = {"kind": uk[1], "t_served": t_s, "tool": ca.get("tool"), "epoch": e, "key": k}
            # served coordinates: inside its request; command/call_end in its completion turn, output_end in the final
            # user turn of the request that shows it
            check("served readout < request length n <= epoch length", r_s < n_s <= len(s), rk)
            starts = np.flatnonzero(s[:n_s] == IM_START)
            h = int(starts[-1]) if len(starts) else -1
            if uk[1] in ("command", "call_end"):
                check(f"served {uk[1]}: after the request's final '<|im_start|>assistant' (its completion)",
                      h >= 0 and s[h + 1] == ASSISTANT and h < r_s, rk)
                tc = np.flatnonzero(s[:r_s] == TOOL_CALL_O)
                check(f"served {uk[1]}: inside a <tool_call> of that completion", len(tc) > 0 and tc[-1] > h, rk)
                check(f"served {uk[1]}: next token carries the newline and '</parameter' follows at +2",
                      r_s + 2 + len(PARAM_C) <= n_s and nl_end[s[r_s + 1]] and np.array_equal(s[r_s + 2:r_s + 2 + len(PARAM_C)], PARAM_C), rk)
            else:
                check("served output_end: in the final user turn of its request",
                      len(starts) >= 2 and h >= 0 and s[h + 1] == ASSISTANT and int(starts[-2]) < r_s < h, rk)
                tro = np.flatnonzero(s[:r_s] == TR_O)
                trc = np.flatnonzero(s[r_s:] == TR_C)
                check("served output_end: readout == '</tool_response>' - 2, newline-ended token between",
                      len(tro) > 0 and len(trc) > 0 and trc[0] == 2 and nl_end[s[r_s + 1]] and int(tro[-1]) + 2 <= r_s, rk)
            # each layout: readout, n, span from served coordinates, token at the readout
            for L in lcalls:
                cl = lcalls[L][uk]
                ids_l = lay[L][(k, e)]["ids"]
                if L == "B":
                    er, en = posmap_b(ca["readout"], blocks), (posmap_b(ca["n"] - 1, blocks) + 1 if posmap_b(ca["n"] - 1, blocks) is not None else None)
                else:
                    sl = sysend(ids_l)
                    er, en = r_s - ss + sl, n_s - ss + sl
                check(f"{L}: readout == served readout mapped", er is not None and cl["readout"] == er, rk)
                check(f"{L}: n == served n mapped", en is not None and cl["n"] == en, rk)
                tl = int(ids_l[cl["readout"]]) if 0 <= cl["readout"] < len(ids_l) else -1
                check(f"{L}: token at readout == served token", tl == t_s, rk)
                sp = cl.get("span")
                if uk[1] == "output_end":
                    tro = np.flatnonzero(ids_l[:cl["readout"]] == TR_O)
                    trc = np.flatnonzero(ids_l[cl["readout"]:] == TR_C)
                    exp = [int(tro[-1]) + 2, cl["readout"] + 1] if len(tro) and len(trc) and trc[0] == 2 else None
                    span_conv["exclusive end (readout + 1)" if isinstance(sp, list) and sp[1] == cl["readout"] + 1 else "other"] += 1
                    check(f"{L}: span == [<tool_response> + 2, readout + 1) with newline boundary tokens",
                          exp is not None and sp == exp and nl_start[ids_l[exp[0] - 1]] and nl_end[ids_l[cl["readout"] + 1]]
                          and exp[0] <= cl["readout"], rk)
                else:
                    check(f"{L}: no span on {uk[1]}", sp is None, rk)
                # extraction meta for this layout: same readout, same reader row labels as the reader's own unit
                mc = mcalls[L].get(uk) if mcalls[L] is not None else None
                rmc = rmeta_calls.get(uk)
                ok_m = (mc is not None and mc["readout"] == cl["readout"] and rmc is not None and mc.get("reader_row") == rmc["row"]
                        and (uk[1] != "output_end" or mc.get("reader_row_mean") == rmc.get("row_mean"))
                        and cl.get("reader_row") == rmc["row"])
                check(f"{L}: meta readout == staged; meta and staged reader_row(_mean) == the reader's own row for the unit key", ok_m, rk)
                u[L] = [cl["readout"], cl["n"], sp]
            # reader token (claim 5) and the staged flags
            rc = rid[1].get(uk) if rid is not None else None
            if check("unit in the reader's staged calls", rc is not None and rc["readout"] is not None, rk):
                t_r = int(rid[0][rc["readout"]])
                u["t_reader"] = t_r
                check("staged reader_readout_token == reader ids at reader readout", ca.get("reader_readout_token") == t_r, rk)
                check("staged served_readout_token == served ids at served readout", ca.get("served_readout_token") == t_s, rk)
                check("staged flag == (reader token == served token)", ca.get("readout_token_matches_reader") == (t_r == t_s), rk)
            rmc = rmeta_calls.get(uk)
            u["reader_row"], u["reader_row_mean"] = (rmc["row"], rmc.get("row_mean")) if rmc else (None, None)
            # candidate multiplicity: served epochs of this run whose calls list the same unit key
            cands = [ee for ee in served_epochs_of[k] if uk in served_ukeys[(k, ee)]]
            u["candidates"] = cands
            units[(rk, uk)] = u
    # ---- unit assignment rows for this world vs the staged units
    ci = {c: i for i, c in enumerate(ctx["ua_columns"])}
    ua_seen = Counter()
    mult = Counter()
    for row in ctx["ua_rows"]:
        rk = ctx["ua_runkey"][row[ci["run key"]]]
        uk = (row[ci["irid"]], row[ci["kind"]], row[ci["src_irid"]], row[ci["call_index"]])
        ua_seen[(rk, uk)] += 1
        u = units.get((rk, uk))
        if not check("UA row has a staged unit with the same key", u is not None, rk):
            continue
        ok = (row[ci["served epoch"]] == u["epoch"] and row[ci["served epoch key"]] == u["key"] and row[ci["reader_row"]] == u["reader_row"]
              and row[ci["reader_row_mean"]] == u["reader_row_mean"]
              and all(row[ci[f"{L} [readout, n, span]"]] == u[L if L in u else "A"] for L in ("A0", "A", "B")))
        check("UA row == staged unit (epoch, key, reader rows, per-layout readout/n/span)", ok, rk)
        mult[len(u["candidates"])] += 1
        check("UA epoch == the earliest served epoch listing the unit", u["candidates"] and u["epoch"] == min(u["candidates"]), rk)
    C["UA rows (this world)"] = len(ctx["ua_rows"])
    C["UA duplicate unit keys"] = sum(1 for v in ua_seen.values() if v > 1)
    C["staged units (this world)"] = len(units)
    C["staged units missing from UA"] = sum(1 for key in units if ua_seen[key] == 0)
    # ---- sampled runs: every served epoch's calls, for the log-based check
    sample = {}
    for rk in ctx["sample"]:
        task = rk.split("|", 1)[1]
        run_keys = {k for (k, e) in keysA if lay["A"][(k, e)]["task"] == task}
        recs = sorted(kk for kk in served if kk[0] in run_keys)
        irids = set()
        epochs = []
        for kk in recs:
            S = served[kk]
            for c in S["calls"]:
                irids.update(x for x in (c["irid"], c.get("src_irid")) if x)
            epochs.append({"key": kk[0], "epoch": kk[1], "file": S["_file"],
                           "calls": [{"irid": c["irid"], "src_irid": c.get("src_irid"), "kind": c.get("kind"), "call_index": c.get("call_index"),
                                      "readout": c["readout"], "n": c["n"], "sha256": c["sha256"], "turn": c.get("turn")} for c in S["calls"]]})
        sample[rk] = {"irids": sorted(irids), "epochs": epochs}
    utab = [[rk, uk[1], u.get("reader_row"), u.get("reader_row_mean"), u.get("t_reader"), u["t_served"], u.get("tool")]
            for (rk, uk), u in units.items()]
    rtab = [[rk, v["kind"], v["row"], v["row_mean"], v["t_reader"]] for (rk, uk), v in reader_units.items()]
    return {"world": world, "counts": dict(C), "fails": dict(fails), "think": dict(think_tot), "sys_hash": dict(sys_hash),
            "span_convention": dict(span_conv), "multiplicity": {str(k): v for k, v in mult.items()},
            "units": utab, "reader_units": rtab, "sample": sample}


# -------------------------------------------------------------------------------------------------- claim 4: logs
@app.function(image=image, volumes={"/rl": ro(V_RL)}, cpu=2, memory=16384, timeout=600, retries=0)
def scan_log(fname: str, wanted: list) -> dict:
    """Every record in one request log whose request_id is wanted: logged prompt and completion ids (int32)."""
    import gzip
    import re

    import numpy as np

    W = set(wanted)
    rx = re.compile(rb'"request_id":\s*"([A-Za-z0-9]+)"')
    out, C = {}, {"lines": 0, "matched": 0, "truncated_tail": 0, "bad_json": 0}
    with gzip.open(f"/rl/{fname}", "rb") as f:
        while True:
            try:
                line = f.readline()
            except (EOFError, OSError):
                C["truncated_tail"] += 1
                break
            if not line:
                break
            C["lines"] += 1
            m = rx.search(line[:200])
            if not m or m.group(1).decode() not in W:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                C["bad_json"] += 1
                continue
            C["matched"] += 1
            ch = (r.get("choices") or [{}])[0]
            out.setdefault(r["request_id"], []).append({
                "file": fname, "received": r.get("received"), "status": r.get("status"), "complete": r.get("complete"),
                "prompt": np.asarray(r["prompt_token_ids"], dtype="<i4") if r.get("prompt_token_ids") is not None else None,
                "completion": np.asarray(ch["token_ids"], dtype="<i4") if ch.get("token_ids") is not None else None})
    return {"recs": out, "counts": C}


@app.function(image=image, volumes={"/q3": ro(V_Q3)}, cpu=2, memory=32768, timeout=600, retries=0)
def check_sample(world: str, sample: dict, logs: dict, ua: list, ua_columns: list) -> dict:
    """Claim 4: rebuild each sampled run's epochs from the logged ids with my own prefix test; check every unit's
    request, epoch and position against the frozen assignment and the served staging; check the request timeline's
    completeness by the staged turn indices."""
    from collections import Counter, defaultdict
    from datetime import datetime

    import numpy as np

    C = Counter()
    fails = defaultdict(list)

    def check(name, ok, key):
        ok = bool(ok)
        C[name + (": pass" if ok else ": FAIL")] += 1
        if not ok:
            C["FAIL " + name] += 1
            if len(fails[name]) < 15:
                fails[name].append(key)
        return ok

    ci = {c: i for i, c in enumerate(ua_columns)}
    ua_by_run = defaultdict(list)
    for r in ua:
        ua_by_run[r[0]].append(r)
    staged_ids = {}
    need = defaultdict(set)
    for rk, sm in sample.items():
        for ep in sm["epochs"]:
            need[ep["file"]].add(ep["key"])
    for f, ks in need.items():
        for r in read_jsonl(f"/q3/{f.split('/')[0]}/inputs/{f.split('/', 1)[1]}", keep=ks):
            staged_ids[(r["key"], r["epoch"])] = i32(r["ids"])
    for rk, sm in sample.items():
        C["runs"] += 1
        staged_sha = defaultdict(set)
        for ep in sm["epochs"]:
            for c in ep["calls"]:
                staged_sha[c["irid"]].add(c["sha256"])
        req = {}
        for irid in sm["irids"]:
            recs = [r for r in logs.get(irid, []) if r["status"] == 200 and r["complete"] and r["prompt"] is not None
                    and r["completion"] is not None]
            uniq = {}
            for r in recs:
                full = np.concatenate([r["prompt"], r["completion"]])
                uniq.setdefault(sha(full.tobytes()), (r, full))
            if not check("request has a usable logged record", len(uniq) > 0, f"{rk}:{irid}"):
                continue
            if len(uniq) > 1:
                C["requests with >1 distinct usable logged records"] += 1
                pick = [v for h_, v in uniq.items() if h_ in staged_sha.get(irid, set())]
                if not check("several logged records resolved by the staged sha256", len(pick) == 1, f"{rk}:{irid}"):
                    continue
                r, full = pick[0]
            else:
                r, full = next(iter(uniq.values()))
            req[irid] = {"t": datetime.fromisoformat(r["received"]), "lp": len(r["prompt"]), "full": full}
        order = sorted(req, key=lambda i: req[i]["t"])
        # my epochs: a new epoch whenever the next prompt does not begin with the previous prompt + completion
        my_epoch, e, prev = {}, -1, None
        for i in order:
            p = req[i]["full"][:req[i]["lp"]]
            if prev is None or len(p) < len(prev) or not np.array_equal(p[:len(prev)], prev):
                e += 1
            my_epoch[i] = e
            prev = req[i]["full"]
        C["my epochs"] += e + 1
        nxt = {order[j]: order[j + 1] for j in range(len(order) - 1)}
        # completeness of the request timeline by the staged turn indices (each request's own calls carry its turn)
        turn_of = defaultdict(set)
        for ep in sm["epochs"]:
            for c in ep["calls"]:
                if c["kind"] in ("command", "call_end", None) and c.get("turn") is not None and c["src_irid"] in (None, c["irid"]):
                    turn_of[c["irid"]].add(c["turn"])
        check("each request has one turn index", all(len(v) == 1 for v in turn_of.values()), rk)
        tl = sorted((min(v), i) for i, v in turn_of.items())
        turns = [t for t, _ in tl]
        consumer_only = [i for i in req if i not in turn_of]
        C["requests without own calls (consumer-only, summed)"] += len(consumer_only)
        check("time order == turn order (requests with own calls)", [i for _, i in tl if i in req] == [i for i in order if i in turn_of], rk)
        # completeness against the eval's turn indices: each consumer-only request takes turn = (turn of the previous
        # request in time) + 1; together the requests must cover every turn from the first to the last exactly once
        inferred, last_t = {}, None
        for i in order:
            if i in turn_of:
                last_t = min(turn_of[i])
            elif last_t is not None:
                last_t += 1
                inferred[i] = last_t
        allt = sorted([min(v) for i, v in turn_of.items() if i in req] + list(inferred.values()))
        C["turn-index gaps among requests with their own calls (summed)"] += (turns[-1] - turns[0] + 1 - len(set(turns))) if turns else 0
        check("request timeline covers every turn index exactly once (gaps filled one-to-one by consumer-only requests in time order)",
              len(allt) > 0 and allt == list(range(allt[0], allt[0] + len(allt))) and allt[0] == min(turns, default=allt[0]), rk)
        # staged epochs vs mine; staged ids vs logged ids, request by request
        st_to_mine = defaultdict(set)
        for ep in sm["epochs"]:
            S = staged_ids.get((ep["key"], ep["epoch"]))
            if not check("staged epoch found", S is not None, rk):
                continue
            for c in ep["calls"]:
                q = req.get(c["irid"])
                if q is None:
                    continue
                st_to_mine[ep["epoch"]].add(my_epoch[c["irid"]])
                check("staged call: n == logged prompt + completion length", c["n"] == len(q["full"]), f"{rk}:{c['irid']}")
                check("staged call: sha256 == sha256 of logged int32 ids", c["sha256"] == sha(q["full"].tobytes()), f"{rk}:{c['irid']}")
                check("staged call: staged ids[:n] == logged prompt + completion",
                      c["n"] <= len(S) and np.array_equal(S[:c["n"]], q["full"]), f"{rk}:{c['irid']}")
        for st_e, mine in st_to_mine.items():
            check("staged epoch == exactly my epoch with the same index", mine == {st_e}, f"{rk}:{st_e}")
        # units of this run in the frozen assignment
        for row in ua_by_run.get(rk, []):
            kind, irid, src = row[ci["kind"]], row[ci["irid"]], row[ci["src_irid"]]
            C[f"UA units checked: {kind}"] += 1
            q = req.get(irid)
            if not check("UA unit's request found in the logs", q is not None, f"{rk}:{irid}"):
                continue
            ep_key, ep_no = row[ci["served epoch key"]], row[ci["served epoch"]]
            ep = next((x for x in sm["epochs"] if x["key"] == ep_key and x["epoch"] == ep_no), None)
            if not check("UA unit's epoch is staged", ep is not None, rk):
                continue
            sc = next((c for c in ep["calls"] if c["irid"] == irid and c["kind"] == kind and c["src_irid"] == src
                       and c["call_index"] == row[ci["call_index"]]), None)
            if not check("UA unit in its staged epoch with a readout", sc is not None and sc["readout"] is not None, rk):
                continue
            r_s = sc["readout"]
            if kind in ("command", "call_end"):
                check(f"{kind}: read in its originating request's completion (logged prompt length <= readout < n)",
                      irid == src and q["lp"] <= r_s < len(q["full"]), f"{rk}:{irid}")
            else:
                qs = req.get(src)
                check("output_end: its request is the FIRST logged request after the source (time order)", nxt.get(src) == irid, f"{rk}:{irid}")
                check("output_end: readout inside that request's prompt, after the source's prompt + completion",
                      qs is not None and r_s < q["lp"] and (my_epoch[src] != my_epoch[irid] or r_s >= len(qs["full"])), f"{rk}:{irid}")
            check("UA served epoch == my epoch index of its request", my_epoch[irid] == ep_no, f"{rk}:{irid}")
            check("UA unit's readout token: staged ids == logged ids", int(staged_ids[(ep_key, ep_no)][r_s]) == int(q["full"][r_s])
                  if r_s < len(q["full"]) else False, f"{rk}:{irid}")
    return {"world": world, "counts": dict(C), "fails": dict(fails)}


# --------------------------------------------------------------------------------------------- claim 7 per layout
@app.function(image=image, volumes={"/acts": ro(V_ACTS), "/cw": ro(V_CW)}, cpu=4, memory=32768, timeout=900, retries=0)
def compare_layout(layout: str, proj: dict, projA: dict, probe_layers: dict, probes: list) -> dict:
    """Every npz row of one layout recomputed from its stored activation row; run means; npz unit sets (claim 2);
    A0 vs A honest rows (claim 3, for A0). Missing or non-finite rows count as failures."""
    import os
    from collections import Counter, defaultdict

    import numpy as np

    C = Counter()
    fails = defaultdict(list)

    def fail(kind, key):
        C["FAIL " + kind] += 1
        if len(fails[kind]) < 15:
            fails[kind].append(key)

    def meta_index(L):
        last, mean = {}, {}
        base = f"/acts/{EXTRACT[L]}/meta"
        for w in sorted(os.listdir(base)):
            for f in sorted(os.listdir(f"{base}/{w}")):
                m = json.load(open(f"{base}/{w}/{f}"))
                rk = f"{m['arm']}|{m['task']}"
                ch = f"{EXTRACT[L]}/{m['chunk']}"
                for c in m["calls"]:
                    k = (rk, c["reader_row"])
                    if k in last:
                        fail(f"{L} meta: (run, reader_row) twice", rk)
                    last[k] = {"kind": c["kind"], "epoch": m["epoch"], "chunk": ch, "row": m["row_start"] + c["row"]}
                    if c["kind"] == "output_end":
                        km = (rk, c["reader_row_mean"])
                        if km in mean:
                            fail(f"{L} meta: (run, reader_row_mean) twice", rk)
                        mean[km] = {"kind": "output_end", "epoch": m["epoch"], "chunk": ch, "row": m["row_start"] + c["row_mean"]}
        return last, mean

    idx = {"self": meta_index(layout)}
    if layout == "A0":
        idx["A"] = meta_index("A")
    res = {}
    unitsets = {}
    for probe in probes:
        run_means = json.load(open(f"/cw/served/{layout}/p2/{probe}__run_means.json"))
        for ro_ in READOUTS:
            z = np.load(f"/cw/served/{layout}/p2/{probe}__{ro_}.npz", allow_pickle=False)
            keys, rr, ep, score = z["key"], z["reader_row"], z["served_epoch"], z["score"]
            n = len(keys)
            mine32, mine64, ratio = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
            lg32 = np.full((n, 6), np.nan) if probe == "fibs" else None
            lgratio = np.full((n, 6), np.nan) if probe == "fibs" else None
            unmatched = 0
            for i in range(n):
                k = str(keys[i])
                src = "A" if (layout == "A0" and not k.startswith(ATTACK)) else "self"
                P = projA if src == "A" else proj
                m = idx[src][1 if ro_ == "reply_mean" else 0].get((k, int(rr[i])))
                if m is None or m["kind"] != KIND_OF[ro_] or m["epoch"] != int(ep[i]):
                    unmatched += 1
                    fail(f"{ro_}: npz row has no meta unit with the same (run, reader row), kind and epoch", k)
                    continue
                row = m["row"]
                try:
                    if probe == "fibs":
                        zz = [P[L][m["chunk"]]["fibs"] for L in probe_layers["fibs"]]
                        z32 = np.array([x["f32"][row] for x in zz], dtype=np.float32)
                        z64 = np.array([x["f64"][row] for x in zz])
                        ad = np.array([x["absdot"][row] for x in zz])
                        lg32[i] = z32
                        lgratio[i] = np.abs(z["logits"][i] - z64) / (EPS32 * ad)
                        mine32[i] = np.mean(1.0 / (1.0 + np.exp(-z32.astype(np.float64))))
                        mine64[i] = np.mean(1.0 / (1.0 + np.exp(-z64)))
                        ratio[i] = abs(score[i] - mine64[i]) / (0.25 * EPS32 * float(np.mean(ad)) + 1e-300)
                    else:
                        L = probe_layers[probe][0]
                        x = P[L][m["chunk"]][probe]
                        mine32[i], mine64[i] = float(x["f32"][row]), float(x["f64"][row])
                        ratio[i] = abs(score[i] - mine64[i]) / (EPS32 * float(x["absdot"][row]) + 1e-300)
                        if probe == "highstakes":
                            u = P[L][m["chunk"]].get("highstakes_unfolded")
                            C["highstakes unfolded form: rows |diff| > 1e-6 vs stored"] += int(not abs(float(u["f32"][row]) - float(score[i])) <= TOL)
                            C["highstakes folded (direction+bias) form: rows |diff| > 1e-6 vs stored"] += int(not abs(mine32[i] - float(score[i])) <= TOL)
                except KeyError:
                    unmatched += 1
                    fail("row's activation chunk/layer not projected", k)
            d32, d64 = np.abs(mine32 - score), np.abs(mine64 - score)
            bad = ~np.isfinite(d32) | ~np.isfinite(d64)
            worlds = np.array([str(k).split("|")[0] for k in keys])

            def summ(sel):
                fin = sel & ~bad
                return {"rows": int(sel.sum()), "nonfinite_or_missing": int((sel & bad).sum()),
                        "max_abs_diff_f32": float(d32[fin].max()) if fin.any() else None,
                        "n_gt_1e-6_f32": int((sel & ~(d32 <= TOL)).sum()),
                        "max_abs_diff_f64": float(d64[fin].max()) if fin.any() else None,
                        "n_gt_1e-6_f64": int((sel & ~(d64 <= TOL)).sum()),
                        "max_rounding_ratio |stored - exact| / (eps32 * sum|x_i w_i|)": float(ratio[fin].max()) if fin.any() else None}

            cell = summ(np.ones(n, bool)) | {"unmatched_rows": unmatched, "per_world": {w: summ(worlds == w) for w in WORLDS}}
            if probe == "fibs":
                dl = np.abs(lg32 - z["logits"])
                cell["fibs_logits_max_abs_diff_f32"] = float(np.nanmax(dl)) if np.isfinite(dl).any() else None
                cell["fibs_logits_nonfinite"] = int((~np.isfinite(dl)).sum())
                cell["fibs_logits_max_rounding_ratio"] = float(np.nanmax(lgratio)) if np.isfinite(lgratio).any() else None
                cell["fibs_layers_stored == mine"] = [int(x) for x in z["layers"]] == list(probe_layers["fibs"])
            by_run = defaultdict(list)
            for i in range(n):
                by_run[str(keys[i])].append(mine32[i])
            stored = run_means[ro_]
            dm = [abs(float(np.mean(v)) - float(stored[k])) if k in stored else float("nan") for k, v in by_run.items()]
            cell["run_means"] = {"runs_mine": len(by_run), "runs_stored": len(stored), "same_keys": set(by_run) == set(stored),
                                 "max_abs_diff": float(np.nanmax(dm)) if np.isfinite(dm).any() else None,
                                 "n_gt_1e-6_or_nonfinite": int(sum(not x <= TOL for x in dm))}
            ok_fibs = probe != "fibs" or (cell["fibs_logits_nonfinite"] == 0 and cell["fibs_layers_stored == mine"])
            cell["verified (every row matched, finite, within 1e-6; FIBS: logits finite, layer order equal)"] = (
                unmatched == 0 and not bad.any() and cell["n_gt_1e-6_f32"] == 0 and ok_fibs)
            res[f"{probe}|{ro_}"] = cell
            us = sorted(zip(map(str, keys), map(int, rr), strict=True))
            sig = sha(json.dumps(us).encode())
            if ro_ not in unitsets:
                unitsets[ro_] = {"sha256": sig, "n": len(us), "runs": len({k for k, _ in us}),
                                 "rows": [[str(k), int(r), int(e_), bool(f)] for k, r, e_, f in
                                          zip(keys, rr, ep, z["readout_token_matches_reader"], strict=True)]}
            elif unitsets[ro_]["sha256"] != sig:
                fail(f"{ro_}: unit set differs between probes", probe)
    out = {"layout": layout, "cells": res, "counts": dict(C), "fails": dict(fails),
           "unitsets": {k: {"sha256": v["sha256"], "n": v["n"], "runs": v["runs"]} for k, v in unitsets.items()},
           "unit_rows": {k: v["rows"] for k, v in unitsets.items()}}
    if layout == "A0":  # claim 3: A0's honest rows vs A's, every field, every file
        ident = {}
        for probe in probes:
            for ro_ in READOUTS:
                z0 = np.load(f"/cw/served/A0/p2/{probe}__{ro_}.npz", allow_pickle=False)
                zA = np.load(f"/cw/served/A/p2/{probe}__{ro_}.npz", allow_pickle=False)
                h0 = np.array([not str(k).startswith(ATTACK) for k in z0["key"]])
                hA = np.array([not str(k).startswith(ATTACK) for k in zA["key"]])
                eq = set(z0.files) == set(zA.files)
                for f in z0.files if eq else []:
                    a0, aA = z0[f], zA[f]
                    if a0.ndim and len(a0) == len(h0):
                        eq &= a0[h0].shape == aA[hA].shape and a0[h0].tobytes() == aA[hA].tobytes()
                    else:
                        eq &= np.array_equal(a0, aA)
                ident[f"{probe}__{ro_}.npz"] = {"identical": bool(eq), "honest_rows": int(h0.sum())}
            rm0 = json.load(open(f"/cw/served/A0/p2/{probe}__run_means.json"))
            rmA = json.load(open(f"/cw/served/A/p2/{probe}__run_means.json"))
            ident[f"{probe}__run_means.json (honest keys)"] = {"identical": all(
                {k: v for k, v in rm0[ro_].items() if not k.startswith(ATTACK)} == {k: v for k, v in rmA[ro_].items() if not k.startswith(ATTACK)}
                for ro_ in READOUTS)}
        out["A0_vs_A_honest"] = ident
    return out


# ------------------------------------------------------------------------------------------------- orchestrator
@app.function(image=image, volumes={"/acts": ro(V_ACTS), "/cw": ro(V_CW), "/q4": ro(V_Q4), "/rl": ro(V_RL), "/out": out_vol},
              cpu=2, memory=32768, timeout=2700, retries=0)
def orchestrate(payload: dict) -> dict:
    import os
    import random
    from collections import Counter, defaultdict
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import numpy as np

    smoke = payload["smoke"]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "PRELIMINARY: independent recheck of the served-context p2 scoring (claims 1-7); counts, hashes and max |diff| only",
           "stamp (PT)": stamp, "git": payload["git"], "smoke": smoke, "inputs": payload["inputs"], "claims": {}, "hashes": {}}
    # ---- hashes of the scorer's outputs and the unit assignment
    man = {}
    for L in LAYOUTS:
        b = open(f"/cw/served/{L}/p2/manifest_served.json", "rb").read()
        man[L] = json.loads(b)
        rec["hashes"][f"manifest_served {L}"] = {"sha256": sha(b), "claimed": CLAIM["manifest_sha256"][L], "match": sha(b) == CLAIM["manifest_sha256"][L]}
    uab = open(f"/cw/{CLAIM['unit_assignment'][0]}", "rb").read()
    ua = json.loads(uab)
    rec["hashes"]["unit_assignment"] = {"sha256": sha(uab), "claimed": CLAIM["unit_assignment"][1],
                                        "match": sha(uab) == CLAIM["unit_assignment"][1], "rows": len(ua["rows"])}
    stage = payload["stage_record"]
    # ---- scored keys (from A's npz) and the sample
    z = np.load("/cw/served/A/p2/fibs__command.npz", allow_pickle=False)
    scored = sorted(set(map(str, z["key"])))
    by_w = {w: sorted(k for k in scored if k.split("|")[0] == w) for w in WORLDS}
    rng = random.Random(SEED)
    sample = {w: rng.sample(by_w[w], min(N_SAMPLE_PER_WORLD, len(by_w[w]))) for w in WORLDS}
    rec["sample"] = {"seed": SEED, "rule": "one random.Random(20261014); for w in run1, run2, sim, honest_a, honest_b, honest_c in this "
                     "order: .sample(sorted scored run keys of w, 10)", "runs": sample}
    # ---- UA run keys -> world|task
    ci = {c: i for i, c in enumerate(ua["columns"])}
    reader2run = {x["key"]: f"{w}|{x['task']}" for w, xs in stage["cohort"]["worlds"].items() for x in xs}
    ua_runkey = {}
    for row in ua["rows"]:
        k = row[ci["run key"]]
        if k not in ua_runkey:
            if k.split("|")[0] in WORLDS and len(k) < 60:
                ua_runkey[k] = k
            elif k in reader2run:
                ua_runkey[k] = reader2run[k]
            else:
                raise SystemExit("unit assignment run key not resolvable")
    ua_by_w = defaultdict(list)
    for row in ua["rows"]:
        ua_by_w[ua_runkey[row[ci["run key"]]].split("|")[0]].append(row)
    # ---- reader sources (corrected B manifest + honest C)
    mb = payload["manifest_b"]
    reader_sources = defaultdict(list)
    for e in mb["epochs"]:
        reader_sources[e["world"]].append({"run_key": f"{e['world']}|{e['task']}", "key": e["key"], "inputs": e["staged_inputs_file"], "meta": e["meta_file"]})
    for f in sorted(os.listdir("/q4/q4b_extract_c_v1/meta/honest_b")):
        m = json.load(open(f"/q4/q4b_extract_c_v1/meta/honest_b/{f}"))
        reader_sources["honest_c"].append({"run_key": f"honest_c|{m['task']}", "key": m["key"],
                                           "inputs": "q4f_c_v1/inputs/honest_b.jsonl", "meta": f"q4b_extract_c_v1/meta/honest_b/{f}"})
    donor_of = {f"{r['served_key']}|{r['served_epoch']}": r["donor"] for w in stage["worlds"] for r in w["recipients"] if "donor" in r}
    # ---- launch: projections (served layouts + the reader's honest_b) and the per-world id checks
    W = payload["weights"]
    probe_layers = {p: W[p]["layers"] for p in W}
    probes_run = list(PROBES) if not smoke else ["refusal", "harmfulness"]
    layers = sorted({L for p in probes_run for L in probe_layers[p]})
    vec_at = {L: {} for L in layers}
    for p in probes_run:
        for L in probe_layers[p]:
            vec_at[L][p] = {"kind": W[p]["kind"], "tensors": W[p]["tensors"][str(L)]}
            if p == "highstakes":
                vec_at[L]["highstakes_unfolded"] = {"kind": "unfolded", "tensors": W[p]["tensors"][str(L)]}
    chunks = {}
    for Lo in LAYOUTS:
        xs = sorted(f[:-5] for f in os.listdir(f"/acts/{EXTRACT[Lo]}/chunks") if f.endswith(".json"))
        chunks[Lo] = [(EXTRACT[Lo], c) for c in (xs[:2] if smoke else xs)]
    hb = [e for e in mb["epochs"] if e["world"] == "honest_b"]
    hb_meta = {e["key"]: json.load(open(f"/q4/{e['meta_file']}")) for e in (hb[:6] if smoke else hb)}
    rchunks = sorted({(e["extraction_run"], hb_meta[e["key"]]["chunk"]) for e in hb if e["key"] in hb_meta})
    calls = {}
    for Lo in LAYOUTS:
        for L in layers:
            calls[(Lo, L)] = project.spawn(Lo, L, vec_at[L], chunks[Lo])
    for L in layers:
        calls[("R", L)] = project.spawn("R", L, vec_at[L], rchunks)
    worlds = ("run1", "honest_c") if smoke else WORLDS
    idc = {}
    for w in worlds:
        ctx = {"scored_keys": by_w[w], "sample": sample[w][:2] if smoke else sample[w], "ua_rows": ua_by_w[w],
               "ua_columns": ua["columns"], "ua_runkey": {k: v for k, v in ua_runkey.items() if v.split("|")[0] == w},
               "reader_sources": reader_sources[w], "donor_of": donor_of if w in ATTACK else {}}
        idc[w] = ids_world.spawn(w, ctx)
    idr = {w: idc[w].get() for w in worlds}
    # ---- claim 4: the log scan for the sampled runs, then the per-world sample checks
    wanted = sorted({i for w in worlds for s in idr[w]["sample"].values() for i in s["irids"]})
    logs_files = sorted(f for f in os.listdir("/rl") if f.startswith("requests_") and "2026-10-08" <= f[9:19] <= "2026-10-11")
    if smoke:
        logs_files = [f for f in logs_files if f[9:19] == "2026-10-11"]
    scans = [scan_log.spawn(f, wanted) for f in logs_files]
    logs, scan_counts = defaultdict(list), Counter()
    for fc in scans:
        r = fc.get()
        for k, v in r["counts"].items():
            scan_counts[k] += v
        for irid, xs in r["recs"].items():
            logs[irid].extend(xs)
    sc_calls = {}
    for w in worlds:
        sub = {i: logs[i] for s in idr[w]["sample"].values() for i in s["irids"] if i in logs}
        rows = [[ua_runkey[r[ci["run key"]]]] + r[1:] for r in ua_by_w[w] if ua_runkey[r[ci["run key"]]] in idr[w]["sample"]]
        sc_calls[w] = check_sample.spawn(w, idr[w]["sample"], sub, rows, ua["columns"])
    # ---- projections -> claim 1 (gate) and claim 7 (per-row)
    proj = {Lo: {} for Lo in LAYOUTS + ("R",)}
    proj_stats = Counter()
    for (Lo, L), fc in calls.items():
        r = fc.get()
        proj_stats[f"{Lo} rows projected (summed over layers)"] += r.pop("_rows")
        proj_stats[f"{Lo} non-finite activation rows (summed over layers)"] += r.pop("_nonfinite")
        proj[Lo][L] = r
    cmp_calls = {Lo: compare_layout.spawn(Lo, proj[Lo], proj["A"] if Lo == "A0" else {}, probe_layers, probes_run) for Lo in LAYOUTS}
    # gate: honest_b run scores on the reader layout vs the committed records
    staged_hb = {}
    for f in sorted({e["staged_inputs_file"] for e in hb if e["key"] in hb_meta}):
        for r in read_jsonl(f"/q4/{f}", keep=set(hb_meta)):
            staged_hb[r["key"]] = (i32(r["ids"]), {ukey(c): c["readout"] for c in r["calls"] if ukey(c) is not None})
    gate, gate_rows = {}, Counter()
    for probe in probes_run:
        L0 = probe_layers[probe]
        per = {ro_: {} for ro_ in READOUTS}
        for e in hb:
            if e["key"] not in hb_meta:
                continue
            m = hb_meta[e["key"]]
            ch = f"{e['extraction_run']}/{m['chunk']}"
            ids_r, ro_of = staged_hb[e["key"]]
            vals = defaultdict(list)

            def sc(row, m=m, ch=ch, probe=probe, L0=L0):
                if probe == "fibs":
                    zz = np.array([proj["R"][L][ch]["fibs"]["f32"][m["row_start"] + row] for L in L0], dtype=np.float32)
                    return float(np.mean(1.0 / (1.0 + np.exp(-zz.astype(np.float64)))))
                return float(proj["R"][L0[0]][ch][probe]["f32"][m["row_start"] + row])

            for c in m["calls"]:
                if c.get("row") is None:
                    continue
                gate_rows["meta call readout == staged readout"] += ro_of.get(ukey(c)) == c["readout"]
                if c["kind"] == "output_end":
                    vals["reply_last"].append(sc(c["row"]))
                    vals["reply_mean"].append(sc(c["row_mean"]))
                elif c["kind"] == "command":
                    vals["command"].append(sc(c["row"]))
                elif c["kind"] == "call_end":
                    if int(ids_r[c["readout"]]) in FIXED:
                        gate_rows["call_end rows skipped (fixed token)"] += 1
                    else:
                        vals["call_end_content"].append(sc(c["row"]))
            for ro_ in READOUTS:
                if vals[ro_]:
                    per[ro_][f"honest_b|{e['task']}"] = float(np.mean(vals[ro_]))
        for ro_ in READOUTS:
            ref, mine = payload["records"][probe][ro_], per[ro_]
            d = [abs(mine[k] - ref[k]) if k in mine else float("nan") for k in ref]
            gate[f"{probe}|{ro_}"] = {"runs_record": len(ref), "runs_mine": len(mine), "same_keys": set(ref) == set(mine),
                                      "extra_mine": sorted(set(mine) - set(ref))[:5], "missing_mine": sorted(set(ref) - set(mine))[:5],
                                      "max_abs_diff": float(np.nanmax(d)) if np.isfinite(d).any() else None,
                                      "n_gt_1e-6_or_missing": int(sum(not x <= TOL for x in d)), "n_exactly_equal": int(sum(x == 0.0 for x in d))}
    gate_claim = man["A"]["gate_honest_b"]
    rec["claims"]["1_gate"] = {
        "mine": gate, "row_counts": dict(gate_rows),
        "verified_cells": sum(1 for v in gate.values() if v["same_keys"] and v["runs_record"] == 194 and v["n_gt_1e-6_or_missing"] == 0),
        "cells": len(gate),
        "claimed (manifest A)": {"pass": gate_claim["pass"], "n_cells": len(gate_claim["by_cell"]),
                                 "cells byte_identical": sum(1 for v in gate_claim["by_cell"].values() if v["byte_identical"]),
                                 "n_runs": sorted({v["n_runs"] for v in gate_claim["by_cell"].values()}),
                                 "call_end_rows_skipped": gate_claim.get("call_end_rows_skipped")},
        "claimed (manifests A0, B) pass": [man["A0"]["gate_honest_b"]["pass"], man["B"]["gate_honest_b"]["pass"]]}
    # ---- claim 7 + 2 + 3 from compare_layout
    cmp = {Lo: cmp_calls[Lo].get() for Lo in LAYOUTS}
    rec["claims"]["7_per_row"] = {Lo: {"cells": cmp[Lo]["cells"], "counts": cmp[Lo]["counts"], "fails": cmp[Lo]["fails"]} for Lo in LAYOUTS}
    rec["claims"]["7_per_row"]["projection_stats"] = dict(proj_stats)
    a0i = cmp["A0"].get("A0_vs_A_honest", {})
    rec["claims"]["3_A0_vs_A_honest"] = {"mine": a0i, "identical_npz_files": sum(1 for k, v in a0i.items() if k.endswith(".npz") and v["identical"]),
                                         "claimed (manifest A0)": man["A0"].get("A0_A_honest_identity")}
    # claim 2: npz unit sets vs the reader's own units (reader rows: last-token row; reply_mean: the span-mean row)
    R = [x for w in worlds for x in idr[w]["reader_units"]]  # [rk, kind, row, row_mean, t_reader]
    U = {(x[0], x[2]): x for w in worlds for x in idr[w]["units"]}  # (rk, reader_row) -> [rk, kind, rr, rrm, t_r, t_s, tool]
    Um = {(x[0], x[3]): x for x in U.values() if x[1] == "output_end"}
    exp = {"reply_last": {(x[0], x[2]) for x in R if x[1] == "output_end"},
           "reply_mean": {(x[0], x[3]) for x in R if x[1] == "output_end"},
           "command": {(x[0], x[2]) for x in R if x[1] == "command"},
           "call_end_content (reader token not 0/n)": {(x[0], x[2]) for x in R if x[1] == "call_end" and x[4] not in FIXED},
           "call_end_content (served token not 0/n)": {(x[0], x[2]) for x in R if x[1] == "call_end" and (x[0], x[2]) in U and U[(x[0], x[2])][5] not in FIXED},
           "call_end (all)": {(x[0], x[2]) for x in R if x[1] == "call_end"}}
    npz_sets = {Lo: {ro_: {(r[0], r[1]) for r in cmp[Lo]["unit_rows"][ro_]} for ro_ in READOUTS} for Lo in LAYOUTS}
    cmpset = {}
    for name, s in exp.items():
        ro_ = name.split(" ")[0] if not name.startswith("call_end") else "call_end_content"
        got = npz_sets["A"][ro_]
        cmpset[name] = {"reader_units": len(s), "npz_units": len(got), "npz_extra": len(got - s), "npz_unread": len(s - got),
                        "runs_reader": len({k for k, _ in s}), "runs_npz": len({k for k, _ in got}),
                        "extra_examples": sorted(got - s)[:5], "unread_examples": sorted(s - got)[:5]}
    rec["claims"]["2_units"] = {"per_layout": {Lo: cmp[Lo]["unitsets"] for Lo in LAYOUTS},
                                "identical_across_layouts": {ro_: len({cmp[Lo]["unitsets"][ro_]["sha256"] for Lo in LAYOUTS}) == 1 for ro_ in READOUTS},
                                "npz_vs_reader_units (layout A)": cmpset, "scored runs": len(scored),
                                "claimed": {"units": CLAIM["units"], "runs": CLAIM["runs"]}}
    # claim 5: 28d counts from my token comparison over each readout's npz units
    t5 = {}
    for ro_ in READOUTS:
        look = Um if ro_ == "reply_mean" else U
        for w in worlds:
            rows = [kr for kr in npz_sets["A"][ro_] if kr[0].split("|")[0] == w]
            pair = {kr: look.get(kr) for kr in rows}
            mism = [kr for kr, x in pair.items() if x is not None and x[4] is not None and x[4] != x[5]]
            flags = {(r[0], r[1]): r[3] for r in cmp["A"]["unit_rows"][ro_] if r[0].split("|")[0] == w}
            t5[f"{ro_}|{w}"] = {"mismatched": len(mism), "units": len(rows), "affected_runs": len({k for k, _ in mism}),
                                "units_without_token_pair": sum(1 for x in pair.values() if x is None or x[4] is None),
                                "npz_flag_agrees_with_mine": sum(1 for kr, x in pair.items() if x is not None and x[4] is not None and flags.get(kr) == (x[4] == x[5])),
                                "npz_flag_false_per_layout": {Lo: sum(1 for r in cmp[Lo]["unit_rows"][ro_] if r[0].split("|")[0] == w and not r[3]) for Lo in LAYOUTS},
                                "mismatched_by_tool": dict(Counter(pair[kr][6] for kr in mism))}
    man28 = {}
    for Lo in LAYOUTS:
        rtc = man[Lo].get("readout_token_counts (Append 28d)", {})
        for Lx, byro in rtc.items():
            for ro_, d in byro.items():
                for k, v in d.items():
                    if k.endswith("| ALL TOOLS"):
                        man28[f"manifest {Lo}: {Lx}|{ro_}|{k.split(' |')[0]}"] = [v.get("mismatched"), v.get("units"), v.get("affected_runs")]
    rec["claims"]["5_28d"] = {"mine": t5, "manifest (mismatched, units, affected runs)": man28,
                              "claimed": {"reply": CLAIM["28d_reply"], "command_mismatched": CLAIM["28d_command_mismatched"],
                                          "command_units": CLAIM["28d_command_units"], "call_end_units": CLAIM["28d_call_end_units"]}}
    # claim 6 + per-unit structure (from ids_world)
    rec["claims"]["6_layouts_and_positions"] = {w: {k: idr[w][k] for k in ("counts", "fails", "think", "sys_hash", "span_convention", "multiplicity")}
                                                for w in worlds}
    rec["claims"]["6_layouts_and_positions"]["stage_record_B_audit"] = {w["world"]: {k: v for k, v in w["audit"].items() if k.startswith("B")}
                                                                       for w in stage["worlds"]}
    # claim 4
    rec["claims"]["4_unit_assignment"] = {
        "ua": {"rows": len(ua["rows"]), "claimed_rows": CLAIM["unit_assignment"][2], "columns": ua["columns"],
               "duplicate unit keys": sum(idr[w]["counts"].get("UA duplicate unit keys", 0) for w in worlds),
               "rows per kind": dict(Counter(r[ci["kind"]] for r in ua["rows"])),
               "candidate multiplicity (served epochs listing the unit) per world": {w: idr[w]["multiplicity"] for w in worlds}},
        "manifest_provenance (claimed)": {k: man["A"]["provenance"].get(k) for k in ("unit_assignment_units", "epoch_prefix_checks_pass",
                                                                                      "ambiguous_units", "unit_assignment_sha256")},
        "log_scan": {"files": len(logs_files), "wanted_requests": len(wanted), "found_requests": sum(1 for i in wanted if i in logs),
                     "counts": dict(scan_counts)},
        "sample_checks": {w: sc_calls[w].get() for w in worlds}}
    rec["finished (PT)"] = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    os.makedirs(f"/out/{OUT_DIR}", exist_ok=True)
    path = f"/out/{OUT_DIR}/recheck_{stamp}{'_SMOKE' if smoke else ''}.json"
    blob = json.dumps(rec, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode()
    with open(path, "xb") as f:  # append-only: never overwrite
        f.write(blob)
    out_vol.commit()
    return {"path": path, "sha256": sha(blob), "bytes": len(blob)}


# ------------------------------------------------------------------------------------------------ local entrypoint
def _git(c: str, p: str) -> bytes:
    return subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=REPO).stdout


def _safetensors(b: bytes) -> dict:
    import struct

    import numpy as np

    n = struct.unpack("<Q", b[:8])[0]
    h = json.loads(b[8:8 + n])
    dt = {"F32": np.float32, "F64": np.float64}
    out = {}
    for k, v in h.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]
        out[k] = np.frombuffer(b[8 + n + s:8 + n + e], dtype=dt[v["dtype"]]).reshape(v["shape"]).tolist()
    return out


@app.local_entrypoint()
def main(smoke: bool = False):
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(HERE)], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    if dirty and not smoke:
        raise SystemExit("commit this folder before a full launch")
    inputs, weights = {}, {}
    for p, rp in LINEAR.items():
        d = json.loads(_git(PIN_SIMH, rp))["directions"][p]
        wpath = str(Path(rp).parent / d["file"]["path"])
        wb = _git(PIN_SIMH, wpath)
        if sha(wb) != d["file"]["sha256"]:
            raise SystemExit(f"{p}: weight sha256 differs from its record")
        t = _safetensors(wb)
        weights[p] = {"kind": "linear", "layers": [int(d["layer"])], "tensors": {str(d["layer"]): t}}
        inputs[f"weights {p}"] = {"commit": PIN_SIMH, "path": wpath, "sha256": sha(wb), "layer": int(d["layer"]), "tensors": sorted(t)}
    conv = json.loads(_git(PIN_SIMH, CONV))
    fl = sorted(int(L) for L in conv["layers"])
    weights["fibs"] = {"kind": "fibs", "layers": fl, "tensors": {}}
    for L in fl:
        v = conv["layers"][str(L)]
        wb = _git(PIN_SIMH, str(Path(CONV).parent / v["output"]))
        if sha(wb) != v["output_sha256"]:
            raise SystemExit(f"FIBS L{L}: sha256 differs from the conversion record")
        t = _safetensors(wb)
        if float(t["input_mean"] if not isinstance(t["input_mean"], list) else t["input_mean"][0]) != 0.0:
            raise SystemExit("FIBS input_mean is not 0")
        weights["fibs"]["tensors"][str(L)] = t
        inputs[f"weights fibs L{L}"] = {"commit": PIN_SIMH, "sha256": sha(wb), "tensors": sorted(t)}
    records = {}
    for p, (c, path, key) in RECORDS.items():
        rb = _git(c, path)
        r = json.loads(rb)
        rs = r["run_scores"] if key is None else r["probes"][key]["run_scores"]
        records[p] = {ro_: {k: v for k, v in rs[ro_].items() if k.startswith("honest_b|")} for ro_ in READOUTS}
        inputs[f"record {p}"] = {"commit": c, "path": path, "sha256": sha(rb)}
    sb = _git(PIN_CHECKED, STAGE_REC[0])
    if sha(sb) != STAGE_REC[1]:
        raise SystemExit("stage record sha256 differs")
    mbb = _git(MANIFEST_B[0], MANIFEST_B[1])
    if sha(mbb) != MANIFEST_B[2]:
        raise SystemExit("corrected B manifest sha256 differs")
    inputs["run_simh_scores.py (pins only)"] = {"commit": PIN_SIMH, "sha256": sha(_git(PIN_SIMH, SIMH))}
    inputs["stage_record"] = {"commit": PIN_CHECKED, "path": STAGE_REC[0], "sha256": STAGE_REC[1]}
    inputs["manifest_b"] = {"commit": MANIFEST_B[0], "path": MANIFEST_B[1], "sha256": MANIFEST_B[2]}
    payload = {"smoke": smoke, "git": {"commit": head, "dirty_folder": bool(dirty)}, "inputs": inputs, "weights": weights,
               "records": records, "stage_record": json.loads(sb), "manifest_b": json.loads(mbb)}
    print(json.dumps({"launch": "orchestrate", "smoke": smoke, "git": head}))
    print(json.dumps(orchestrate.remote(payload)))
