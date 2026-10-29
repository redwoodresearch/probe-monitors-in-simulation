"""Layer-0 control (PLAN.md), part 1: the packs (Modal CPU). Two readings at each output_end row of step B's level-1
pack (`claimB/level1/packedB_2cfd1aec393c`, whose runs and row order the 0.822 used), written in that pack's layout so
step B's grid copy runs on them unchanged with `--layers 0 --readouts output_end`:
  E   the raw token embedding (Hugging Face hidden_states[0]): row `embed_tokens.weight[id]` of Qwen/Qwen3-32B @ 9216db57,
      id = tokenizer.convert_tokens_to_ids(token_text) of the readout token in step A's dump metadata; read from the
      safetensors shard directly (bf16 bits to float32), no model run;
  B0  block 0's stored output (`by_layer/L0`, hidden_states[1]) from step A's dumps, whole chunk files read once each.
Sources per arm: real e1/e2 from Q1's level-1 dump, fixed sim and plants from the fixed-sim-v2 level-1 dump.
Gates (all before any pack file is written; gates JSON written either way):
  1 every pack row's (arm, key, step, action, token position, tool) matches a kept output_end position in the source
    metadata, and the pack rows are exactly 0..N-1 (N = step B's L58 array length);
  2 (E) every token string maps to an id and back to the same string, ids inside the embedding matrix;
  3 (E) 20 seeded runs re-tokenized from their staged inputs give the same kept output_end positions and the same ids;
  4 the 20 repro runs: same readout tokens as Q1's e1 rows (E); L0 cosine >= 0.9999 against Q1's rows (B0);
  5 no non-finite or all-zero row. E is float32 (exact from bf16); E also saves the row token ids (int32,
    step-table order, no text) for the negative control. Any exception is recorded by type only, gates written either way.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal deploy experiments/2026-10-11_claimB_layer0_control/run_layer0_pack.py
  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' python experiments/2026-10-11_claimB_layer0_control/run_layer0_pack.py spawn
  ... run_layer0_pack.py collect <call id> [<call id>]
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-claimb-layer0_pack_kill-2026-11-11"
B_PACK, POS, SEED, N_CHECK, COS_MIN = "claimB/level1/packedB_2cfd1aec393c", "output_end", 20261011, 20, 0.9999
Q1 = "2026-10-06_la_segment-end-scrub1_qwen3-32b_alllayers_v1"
FS = "2026-10-10_la_fixed-sim-v2-scrub1_qwen3-32b_alllayers_v1"
SOURCE = {"real-attack-e1": Q1, "real-attack-e2": Q1, "sim-attack-fixed": FS, "sim-attack-fixed-plant1": FS, "sim-attack-fixed-plant2": FS}
SNAP = "/hf/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"

app = modal.App(APP)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
hf = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "transformers==4.56.2", "jinja2")  # jinja2: apply_chat_template (gate 3)
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


def stem(key: str) -> str:  # run_extract_all_layers.filename
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def embeddings(ids):
    """Rows of model.embed_tokens.weight (bf16) for ids, as float32, read from the safetensors shard that holds it."""
    import numpy as np

    shard = json.loads(Path(f"{SNAP}/model.safetensors.index.json").read_text())["weight_map"]["model.embed_tokens.weight"]
    with open(f"{SNAP}/{shard}", "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        h = json.loads(fh.read(n))["model.embed_tokens.weight"]
        assert h["dtype"] == "BF16" and len(h["shape"]) == 2, h
        fh.seek(8 + n + h["data_offsets"][0])
        W = np.frombuffer(fh.read(h["data_offsets"][1] - h["data_offsets"][0]), np.uint16).reshape(h["shape"])
    if max(ids) >= len(W) or min(ids) < 0:
        raise ValueError("token id outside the embedding matrix")
    return (W[np.asarray(ids)].astype(np.uint32) << 16).view(np.float32), h["shape"]


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": acts, "/hf": hf}, timeout=45 * 60, retries=0)
def pack(reading: str, code: str) -> dict:
    import numpy as np

    A = Path("/acts")
    g = {"reading": reading, "b_pack": B_PACK, "failures": {}, "stage": "start"}
    digest = hashlib.sha1(json.dumps({"reading": reading, "b_pack": B_PACK, "sources": SOURCE, "code": code}, sort_keys=True).encode()).hexdigest()[:12]
    out = A / "claimB" / "level1" / f"packedB_{digest}"
    g |= {"pack": digest, "code": code}
    try:  # any exception is recorded by type only (messages can quote token text) and the gates are written either way
        X, extra, runs = build(reading, g)
    except Exception as e:  # noqa: BLE001
        g["failures"].setdefault("exception", []).append(f"{g['stage']}:{type(e).__name__}")
        X = None
    g["pass"] = X is not None and not g["failures"]
    out.mkdir(parents=True, exist_ok=True)
    if g["pass"]:
        for name, arr in [(f"L0_{POS}.npy", X)] + [(f"{k}.npy", v) for k, v in extra.items()]:
            with open(out / f".{name}.tmp", "wb") as fh:
                np.save(fh, arr)
            (out / f".{name}.tmp").replace(out / name)
        (out / ".runs.L0.tmp").write_text(json.dumps(runs))
        (out / ".runs.L0.tmp").replace(out / "runs.L0.json")
    (out / "gates.json").write_text(json.dumps(g))
    acts.commit()
    return g


def build(reading: str, g: dict):
    """The gated pack: features in step-table order (step B's pack layout), extra arrays, and the run list (unchanged)."""
    import random

    import numpy as np
    from transformers import AutoTokenizer

    from probe_shift.q1_eval import step_table
    from probe_shift.reader import segment_end_positions

    A = Path("/acts")
    fail = lambda k, v: g["failures"].setdefault(k, []).append(v)
    g["stage"] = "gate1"
    runs = json.loads((A / B_PACK / "runs.L58.json").read_text())
    t = step_table(runs, POS)
    n = int(np.load(A / B_PACK / "L58_output_end.npy", mmap_mode="r").shape[0])
    g |= {"rows": n, "runs": len(runs)}
    if len(t["run"]) != n:
        fail("gate1_rows", "step-table rows differ from step B's array length")
    src, meta, glob_of = [], {}, {}  # src in step-table order (the grid reads X row j as step-table row j)
    for i, m in enumerate(runs):
        d = SOURCE[m["arm"]]
        a = json.loads((A / d / "meta" / m["arm"] / f"{stem(m['key'])}.json").read_text())
        meta[i] = a
        pos = {(p["step"], p["action"]): p for p in a["positions"] if p["kind"] == POS}
        mine = [p for p in m["positions"] if p["kind"] == POS and p["kept"]]
        if sum(p["kept"] for p in pos.values()) != len(mine):
            fail("gate1_kept_count", [m["arm"], i])
        for p in mine:
            q = pos.get((p["step"], p["action"]))
            glob_of[(i, p["step"], p["action"])] = len(src)
            if not q or not q["kept"] or q["token"] != p["token"] or (q["tool"] or "") != (p["tool"] or "") or q.get("token_text") is None:
                fail("gate1_position", [m["arm"], i, p["step"], p["action"]])
                src.append(None)
                continue
            src.append((d, a["chunk"], a["row_start"] + q["row"], q["token_text"]))
    if len(src) != n or any(s is None for s in src):
        fail("gate1_unmatched_rows", sum(s is None for s in src))
    if g["failures"]:
        return None, {}, runs
    g["gate1"], X, extra = "pass", None, {}
    if reading == "E":
        g["stage"] = "gate2"
        tok = AutoTokenizer.from_pretrained(SNAP)
        texts = [s[3] for s in src]
        ids = tok.convert_tokens_to_ids(texts)
        if any(not isinstance(x, int) for x in ids) or tok.convert_ids_to_tokens(ids) != texts:
            fail("gate2_roundtrip", "a token string does not map to one id and back")
            return None, {}, runs
        X, shape = embeddings(ids)  # float32, exact from bf16
        g |= {"embedding_shape": shape, "distinct_ids": len(set(ids))}
        extra["token_ids"] = np.asarray(ids, np.int32)  # step-table order
        g["stage"] = "gate3"
        picks = random.Random(SEED).sample(range(len(runs)), N_CHECK)  # re-tokenize from the staged inputs
        cfgs = {d: json.loads((A / d / "config.json").read_text()) for d in set(SOURCE.values())}
        items = {}
        for d in sorted({SOURCE[runs[i]["arm"]] for i in picks}):  # one pass over each dump's staged inputs
            want = {(runs[i]["arm"], runs[i]["key"]) for i in picks if SOURCE[runs[i]["arm"]] == d}
            needles = [json.dumps(k)[1:-1] for _, k in want]
            for f in sorted((A / cfgs[d]["inputs_from"] / "inputs").glob("shard_*.jsonl")):
                for line in f.open():
                    if any(s in line for s in needles):
                        it = json.loads(line)
                        if (it["arm"], it["key"]) in want:
                            items[(it["arm"], it["key"])] = it
        for i in picks:
            m, item = runs[i], items.get((runs[i]["arm"], runs[i]["key"]))
            if item is None:
                fail("gate3_missing_input", [m["arm"], i])
                continue
            r = segment_end_positions(tok, item["messages"], cfgs[SOURCE[m["arm"]]]["max_tokens"])
            got = {(p["step"], p["action"], p["token"]) for p in r["positions"] if p["kind"] == POS and p["kept"]}
            mine = [p for p in m["positions"] if p["kind"] == POS and p["kept"]]
            if got != {(p["step"], p["action"], p["token"]) for p in mine} or any(
                    r["ids"][p["token"]] != ids[glob_of[(i, p["step"], p["action"])]] for p in mine):
                fail("gate3_retokenize", [m["arm"], i])
        g["gate3_checked"] = N_CHECK
    else:
        g["stage"] = "b0_read"
        X = np.zeros((n, 5120), np.float16)
        by_chunk = {}
        for row, (d, c, cr, _) in enumerate(src):
            by_chunk.setdefault((d, c), []).append((row, cr))
        for (d, c), rows in by_chunk.items():  # whole chunk files, read once each
            C = np.load(A / d / "by_layer" / "L0" / f"{c}.npy")
            X[[r for r, _ in rows]] = C[[cr for _, cr in rows]]
    g["stage"] = "gate4"  # the 20 repro runs against Q1's e1 rows: positions, tokens, and (B0) cosine
    reps = sorted((A / FS / "meta" / "real-attack-e1-repro").glob("*.json"))
    g["gate4_runs"], cos_min = len(reps), None
    if len(reps) != N_CHECK:
        fail("gate4_count", len(reps))
    for f in reps:
        r1 = json.loads(f.read_text())
        q1 = json.loads((A / Q1 / "meta" / "real-attack-e1" / f.name).read_text())
        pa = {(p["step"], p["action"]): p for p in r1["positions"] if p["kind"] == POS and p["kept"]}
        pb = {(p["step"], p["action"]): p for p in q1["positions"] if p["kind"] == POS and p["kept"]}
        if pa.keys() != pb.keys() or any(pa[k]["token"] != pb[k]["token"] or pa[k]["token_text"] != pb[k]["token_text"] for k in pa):
            fail("gate4_tokens", f.name)
            continue
        if reading == "B0":
            ra = np.load(A / FS / "by_layer" / "L0" / f"{r1['chunk']}.npy")[[r1["row_start"] + pa[k]["row"] for k in pa]].astype(np.float32)
            rb = np.load(A / Q1 / "by_layer" / "L0" / f"{q1['chunk']}.npy")[[q1["row_start"] + pb[k]["row"] for k in pa]].astype(np.float32)
            cos = (ra * rb).sum(1) / np.linalg.norm(ra, axis=1) / np.linalg.norm(rb, axis=1)
            if not np.isfinite(cos).all():
                fail("gate4_nonfinite", f.name)
            else:
                cos_min = float(cos.min()) if cos_min is None else min(cos_min, float(cos.min()))
    if reading == "B0":
        g["gate4_cos_min"] = None if cos_min is None else round(cos_min, 6)
        if cos_min is None or cos_min < COS_MIN:
            fail("gate4_cosine", g["gate4_cos_min"])
    g["stage"] = "gate5"
    if not np.isfinite(X.astype(np.float32)).all() or not (np.abs(X).sum(1) > 0).all():
        fail("gate5_values", reading)
    g["stage"] = "done"
    return (None if g["failures"] else X), extra, runs


def main():
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "spawn":
        code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        f = modal.Function.from_name(APP, "pack")
        if not set(rest) <= {"E", "B0"}:
            raise SystemExit(f"unknown readings {rest}; use E and/or B0")
        calls = {r: f.spawn(r, code).object_id for r in (rest or ["E", "B0"])}
        print(json.dumps({"code": code, "calls": calls}))
        return
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    for cid in rest:
        g = modal.FunctionCall.from_id(cid).get()
        dst = HERE / "results" / f"layer0_pack_{g['reading']}_{stamp}.json"
        with open(dst, "x") as fh:
            fh.write(json.dumps(g | {"call": cid}) + "\n")
        print(json.dumps(g), "->", dst)


if __name__ == "__main__":
    main()
