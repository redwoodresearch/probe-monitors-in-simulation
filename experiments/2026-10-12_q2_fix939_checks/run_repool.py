"""fix939 (G), the per-row part (PLAN.md addendum G, committed with this script before it runs): one Modal CPU container
scores every kept output_end row with #36's 20 frozen nested L63 models and returns only per-run pooled nested scores and
row counts, under three poolings (all; no insert outputs; content-only readout tokens). No text or activations leave it.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-12_q2_fix939_checks/run_repool.py --g <folder: probes_L63.npz, tables.npz, runs_data.json> \\
      --fix <fix939 per-run jsonl>
"""
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-fix939-repool_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
DUMP = "2026-10-06_la_segment-end_qwen3-32b_v1"
SWAP = HERE.parent / "2026-10-07_q2_rendering_swap"
INPUT_FILES = ("probes_L63.npz", "tables.npz", "runs_data.json")  # their sha256 is recorded in the output
POOLINGS = ("all", "no insert", "content only")
FIX_DUMP, FIX_INPUTS, FIX_ARM = "2026-10-11_q3_fix939_qwen3-32b_l48-63_v1", "2026-10-11_q3_fix939_inputs_v1", "sim-attack-note-sr-both"
app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
         .add_local_file(SWAP / "rerender.py", "/root/rerender.py"))
MODELS = [(f, v) for f in range(5) for v in range(5) if f != v]


# (D)'s readout-token rule, copied unchanged from run_readout_tokens.py
def bytes_to_unicode() -> dict:
    """GPT-2 byte-level BPE's byte -> printable-character map (Qwen's tokenizer uses it): vocabulary strings like 'Ċ' (newline)
    and 'Ġ' (space) are these characters."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_OF = {c: b for b, c in bytes_to_unicode().items()}


def decode(tok: str) -> str:
    """A vocabulary string to its text; an already-decoded or special token (a character outside the map) is kept as is."""
    return bytes(BYTE_OF[ch] for ch in tok).decode("utf-8", "replace") if tok and all(ch in BYTE_OF for ch in tok) else tok


def category(tok: str, ends_with_note: bool, review: str) -> str:
    t = decode(tok).strip()
    if t == "":
        return "newline or whitespace"
    if ends_with_note and t in review:
        return "review-note template"
    if set(t) <= set("0123456789|"):
        return "gutter or digit"
    return "other content"


@app.function(image=image, volumes={"/acts": acts}, cpu=8, memory=32768, timeout=3600, retries=0)
def repool(runs: list[dict], tables: bytes, probes: bytes, fix: list[dict]) -> dict:
    import sys
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    sys.path.insert(0, "/root")
    from rerender import REVIEW
    review = REVIEW.strip()
    P = np.load(io.BytesIO(probes)); T = np.load(io.BytesIO(tables))
    k = lambda f, v, s: P[f"output_end__task_nested_{f}_{v}__{s}"]
    W = np.stack([k(f, v, "w") / k(f, v, "std") for f, v in MODELS], 1).astype(np.float64)  # [5120, 20]
    B = np.array([float(k(f, v, "b")) - float((k(f, v, "mean") / k(f, v, "std")) @ k(f, v, "w")) for f, v in MODELS])
    run_r, row_r, step_r, act_r = T["output_end__run"], T["output_end__row"], T["output_end__step"], T["output_end__action"]
    if (np.diff(run_r) < 0).any():
        raise ValueError("table rows are not in run order")
    bounds = np.searchsorted(run_r, np.arange(len(runs) + 1))

    def staged(run_dir, by_key):
        out = {}
        for shard in sorted((Path("/acts") / run_dir / "inputs").glob("shard_*.jsonl")):
            for line in shard.read_text().splitlines():
                it = json.loads(line)
                out[it["key"] if by_key else (it["arm"], it["key"])] = it["messages"]
        return out

    def text_of(c):
        return "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")

    def outputs(msgs):
        res, asst = {}, [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        for s, i in enumerate(asst):
            nxt = asst[s + 1] if s + 1 < len(asst) else len(msgs)
            tools = [m for m in msgs[i + 1: nxt] if m["role"] == "tool"]
            for a, tc in enumerate(msgs[i].get("tool_calls") or []):
                fn = tc.get("function") or {}
                args = fn.get("arguments") or {}
                args = json.loads(args) if isinstance(args, str) else args
                cmd = args.get("command") if fn.get("name") == "text_editor" else None
                if a < len(tools):
                    res[(s, a)] = (cmd, text_of(tools[a].get("content")))
        return res
    base, fixed = staged(DUMP, False), staged(FIX_INPUTS, True)

    def pooled(S, cmds, cats):
        masks = {"all": np.ones(len(S), bool), "no insert": np.array([c != "insert" for c in cmds], bool),
                 "content only": np.array([c == "other content" for c in cats], bool)}
        return [[float(x) for x in S[m].mean(0)] if m.any() else None for m in (masks[p] for p in POOLINGS)], [int(masks[p].sum()) for p in POOLINGS]

    def score(x):  # #36's packed features are float16: round as it did, then score in float64
        return x.astype(np.float16).astype(np.float64) @ W + B

    def one36(i):
        r = runs[i]; a, b = bounds[i], bounds[i + 1]
        meta = json.loads((Path("/acts") / DUMP / "acts" / r["arm"] / f"{r['filename']}.json").read_text())
        tok = {(p["step"], p["action"]): p["token_text"] for p in meta["positions"] if p["kind"] == "output_end" and p["kept"]}
        outs = outputs(base[(r["arm"], r["key"])])
        x = np.load(Path("/acts") / DUMP / "acts" / r["arm"] / f"{r['filename']}.npz")["L63"][row_r[a:b]]
        S = score(x); cmds, cats = [], []
        for s_, a_ in zip(step_r[a:b], act_r[a:b]):
            cmd, text = outs[(int(s_), int(a_))]
            cmds.append(cmd); cats.append(category(tok[(int(s_), int(a_))] or "", text.rstrip().endswith(review), review))
        pv, n = pooled(S, cmds, cats)
        return {"i": i, "pooled": pv, "n": n}

    def onefix(fr):
        meta = json.loads((Path("/acts") / FIX_DUMP / "meta" / FIX_ARM / f"{fr['filename']}.json").read_text())
        pos = [p for p in meta["positions"] if p["kind"] == "output_end" and p["kept"]]
        blk = np.load(Path("/acts") / FIX_DUMP / "by_layer" / "L63" / f"{meta['chunk']}.npy", mmap_mode="r")
        x = np.asarray(blk[[meta["row_start"] + p["row"] for p in pos]])
        outs = outputs(fixed[fr["key"]])
        S = score(x); cmds, cats = [], []
        for p in pos:
            cmd, text = outs[(p["step"], p["action"])]
            cmds.append(cmd); cats.append(category(p["token_text"] or "", text.rstrip().endswith(review), review))
        pv, n = pooled(S, cmds, cats)
        return {"key": fr["key"], "pooled": pv, "n": n}
    with ThreadPoolExecutor(16) as pool:
        r36 = list(pool.map(one36, range(len(runs))))
        rfx = list(pool.map(onefix, fix))
    return {"models": MODELS, "poolings": POOLINGS, "runs": r36, "fix": rfx}


@app.local_entrypoint()
def main(g: str, fix: str):
    g = Path(g)
    runs = json.loads((g / "runs_data.json").read_text())
    fx = [json.loads(l) for l in open(fix)]
    fkeys = sorted({r["key"] for r in fx if r["label"] == "FIX"})
    fname = {r["key"]: r["filename"] for r in runs if r["arm"] == "sim-attack"}
    if len(fkeys) != 823 or not set(fkeys) <= set(fname):
        raise SystemExit("not fix939's 823 re-rendered simulated runs")
    res = repool.remote([{"arm": r["arm"], "key": r["key"], "filename": r["filename"]} for r in runs], (g / "tables.npz").read_bytes(),
                        (g / "probes_L63.npz").read_bytes(), [{"key": k, "filename": fname[k]} for k in fkeys])
    out = {"plan": "PLAN.md addendum G (this folder)", "inputs sha256": {n: hashlib.sha256((g / n).read_bytes()).hexdigest() for n in INPUT_FILES}
           | {Path(fix).name: hashlib.sha256(Path(fix).read_bytes()).hexdigest()}} | res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    d = HERE.parents[1] / "data" / "fix939_repool"; d.mkdir(parents=True, exist_ok=True)
    with open(dest := d / f"repool_{stamp}.json", "x") as fh:  # numbers only, but bulky: data/, then S3 with a manifest
        fh.write(json.dumps(out) + "\n")
    print(dest)
