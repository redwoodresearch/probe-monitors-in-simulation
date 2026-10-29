"""Read tool-output ends with Qwen3-32B (README sections 1, 3, 4), sharded over H200s, on the build's frozen files.

Per dialogue, one forward pass (Qwen3-32B @ 9216db57, bf16, sdpa, batch 1, end-padded to a multiple of 1,024 tokens),
stopped after the last layer asked for, storing fp16 rows at:
- C, the content readout: each read tool output's `output_end` (its last token wholly inside the output text);
- M, beside only: the `</tool_response>` token closing it (`output_marker`).
Two kinds of job:
- `pairs <probe>`: the build's dialogues for probe A (with the attack panel) or B, every layer 0-63, the final tool
  output only. Assembled as run_fit.py's runs `toc<probe>C_<stamp>` and `toc<probe>M_<stamp>` (one row per dialogue),
  plus `tocPanelC/M_<stamp>` for the panel.
- `runs <source> <layers>`: every tool output inside the 32,768-token cut, at the given layers, for the planted file
  (`planted`) or claim B's packs outside P (`packs`). Assembled as `toc_runs_<source>_<stamp>` (rows + meta).
Each shard checkpoints every 100 dialogues on the Volume, under a fingerprint of code, reader, model and inputs, so a
stopped job resumes with `--resume <stamp>`. Only counts and timings return; failures return type and line numbers.
Retries 0; an external billing watcher stops the app at its stop.

  PYTHONPATH=src uv run modal deploy experiments/2026-10-11_tool_output_cue_probe/run_extract_toc.py
  PYTHONPATH=src uv run python experiments/2026-10-11_tool_output_cue_probe/run_extract_toc.py launch results/build_<stamp>.json pairs A --shards 4
  PYTHONPATH=src uv run python experiments/2026-10-11_tool_output_cue_probe/run_extract_toc.py launch results/build_<stamp>.json runs planted 24 --shards 2
  PYTHONPATH=src uv run python experiments/2026-10-11_tool_output_cue_probe/run_extract_toc.py collect <stamp>
"""

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-toc_extract_kill-2026-11-10"
VOL, LA = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09", "probe-shift_aswin-surya_sub-la-activations"
PACKS, PACKS_SHA = "2026-10-10_la_fixed-sim-v2_probe-inputs-b-l1_v1", "e56a27e46c4f0093ab07c08fa5c994f2d76391bc6e36e12cc583fdaf6ec9bd1f"
BUCKET, MAX_TOKENS, CHUNK = 1024, 32768, 100
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub"]

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
la = modal.Volume.from_name(LA, version=2).read_only()
hf_cache = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift"))
LOG_FD = 1
# sha256 over the bundled probe_shift package's .py files (relative path and bytes), at launch and in the container
pkg_digest = lambda d: hashlib.sha256(b"".join(p.relative_to(d).as_posix().encode() + p.read_bytes() for p in sorted(Path(d).rglob("*.py")))).hexdigest()


def log(msg):  # counts only, to the real stdout (the Modal log)
    import os
    os.write(LOG_FD, (msg + "\n").encode())


@app.function(image=image, gpu="H200", volumes={"/hf": hf_cache, "/acts": vol, "/in": la}, memory=65536, timeout=86400, retries=0)
def extract(job: dict, shard: int, prov: dict) -> dict:
    global LOG_FD
    import os
    import traceback

    null = os.open(os.devnull, os.O_WRONLY)  # no LinuxArena text in logs or errors
    LOG_FD = os.dup(1)
    os.dup2(null, 1), os.dup2(null, 2)
    try:
        import probe_shift

        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != prov["extractor_sha256"]:
            raise ValueError("the deployed extractor is not the launching commit's: redeploy")
        if pkg_digest(Path(probe_shift.__file__).parent) != prov["probe_shift_sha256"]:
            raise ValueError("the deployed probe_shift package is not the launching checkout's: redeploy")
        return _extract(job, shard, prov)
    except BaseException as e:
        lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("run_extract_toc.py")]
        raise RuntimeError(f"extract failed: {type(e).__name__} at run_extract_toc.py lines {lines}") from None


def items_of(job):
    """The job's dialogues, in a fixed order, from files checked against their recorded sha256."""
    if job.get("source") == "packs":
        base = Path("/in") / PACKS
        if hashlib.sha256((base / "manifest.json").read_bytes()).hexdigest() != PACKS_SHA:
            raise ValueError("the packs' manifest does not match its sha256")
        p_cells, rows = set(job["p_cells"]), []
        if sorted(f"{PACKS}/{f.name}" for f in (base / "inputs").glob("shard_*.jsonl")) != sorted(k for k in job["shard_sha256"] if k.startswith(PACKS)):
            raise ValueError("the packs' shard files are not the build's")
        for f in sorted((base / "inputs").glob("shard_*.jsonl")):
            raw = f.read_bytes()
            if hashlib.sha256(raw).hexdigest() != job["shard_sha256"][f"{PACKS}/{f.name}"]:
                raise ValueError(f"{f.name} differs from the build's record")
            rows += [r | {"id": r["key"], "version": "original", "arm": r["arm"].removeprefix("real-attack-").removeprefix("sim-attack-")}
                     for r in map(json.loads, raw.decode().splitlines())
                     if r["arm"] in ("real-attack-e1", "real-attack-e2", "sim-attack-fixed") and r["cell"] not in p_cells]
        return sorted(rows, key=lambda r: r["id"])
    raw = (Path("/acts") / job["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != job["sha256"]:
        raise ValueError("the input file does not match the build's sha256")
    rows = [json.loads(x) for x in raw.decode().splitlines()]
    return [r for r in rows if job["kind"] == "runs" or r["probe"] in job["probes"]]


def _extract(job, shard, prov):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import probe_shift.reader as reader_module
    from probe_shift.models import QWEN3_32B
    from probe_shift.reader import segment_end_positions

    vol.reload()
    items = items_of(job)[shard::job["shards"]]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    lm = AutoModelForCausalLM.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"], dtype=torch.bfloat16,
                                              device_map="cuda", attn_implementation="sdpa").eval()
    load_s = time.time() - t0
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    layers = job["layers"]

    class Stop(Exception):
        pass

    ck = Path("/acts") / "toc" / f"chunks_{prov['stamp']}" / f"shard{shard}"
    ck.mkdir(parents=True, exist_ok=True)
    fp = json.dumps({"extractor": prov["extractor_sha256"], "reader": hashlib.sha256(Path(reader_module.__file__).read_bytes()).hexdigest(),
                     "model": QWEN3_32B, "job": job, "shard": shard, "chunk": CHUNK, "bucket": BUCKET})
    have = (ck / "fingerprint.json").read_text() if (ck / "fingerprint.json").exists() else None
    if (have is None and any(ck.iterdir())) or (have is not None and have != fp):
        raise ValueError("these checkpoints were made from other inputs or code: not resumed")
    (ck / "fingerprint.json").write_text(fp)
    fwd_s, n_tok, resumed, skipped = 0.0, 0, 0, 0
    for c in range(0, len(items), CHUNK):
        if (ck / f"c{c:05d}.done").exists():
            resumed += 1
            continue
        rows = {v: {l: [] for l in layers} for v in "CM"}
        meta = []
        for it in items[c:c + CHUNK]:
            r = segment_end_positions(tok, it["messages"], MAX_TOKENS)
            ends = {(p["step"], p["action"]): p for p in r["positions"] if p["kind"] == "output_end" and p["kept"] and p["token"] is not None}
            marks = {(p["step"], p["action"]): p["token"] for p in r["positions"] if p["kind"] == "output_marker" and p["kept"]}
            keys = sorted(ends)  # the content readout decides which outputs count; a marker past the cut gives a NaN M row
            if job["kind"] == "pairs":
                last = max((p["step"], p["action"]) for p in r["positions"] if p["kind"] == "output_end")
                if r["truncated"] or last not in keys:
                    raise ValueError("a pair dialogue's final output is not read inside the cut")
                keys = [last]
            if not keys:
                skipped += 1
                meta.append({x: it.get(x) for x in ("id", "cell", "arm", "version", "task", "main_task")} | {"n_rows": 0, "keys": []})
                continue
            tc = [ends[k]["token"] for k in keys]
            tm, has_m = [marks.get(k, tc[i]) for i, k in enumerate(keys)], np.array([k in marks for k in keys])
            acts = {}

            def hook_for(layer):
                def hook(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    acts[layer] = (h[0, tc].to(torch.float16).cpu().numpy(), h[0, tm].to(torch.float16).cpu().numpy())
                    if layer == layers[-1]:
                        raise Stop
                return hook

            handles = [lm.model.layers[l].register_forward_hook(hook_for(l)) for l in layers]
            t1 = time.time()
            try:
                with torch.inference_mode():
                    lm.model(input_ids=torch.tensor([r["ids"] + [pad] * (-len(r["ids"]) % BUCKET)], device="cuda"), use_cache=False)
            except Stop:
                pass
            finally:
                for h in handles:
                    h.remove()
            fwd_s += time.time() - t1
            n_tok += len(r["ids"])
            for l in layers:
                rows["C"][l].append(acts[l][0])
                rows["M"][l].append(np.where(has_m[:, None], acts[l][1], np.float16(np.nan)))
            keep = ("id", "probe", "split", "pair", "version", "label", "family", "mechanism", "task", "cell", "arm", "main_task",
                    "len_diff", "distance", "neutral_k")
            meta.append({x: it.get(x) for x in keep} | {"n_rows": len(keys), "keys": keys, "n_tokens": r["n_tokens"]})
        for v in "CM":
            for l in layers:
                np.save(ck / f"c{c:05d}_{v}_L{l}.npy", np.concatenate(rows[v][l]) if rows[v][l] else np.zeros((0, 5120), np.float16))
        (ck / f"c{c:05d}_meta.json").write_text(json.dumps(meta))
        (ck / f"c{c:05d}.done").write_text("")
        vol.commit()
        log(f"shard {shard}: {min(c + CHUNK, len(items))}/{len(items)} dialogues, {n_tok / max(fwd_s, 1e-9):.0f} tok/s, {time.time() - t0:.0f} s")
    skipped = sum(m["n_rows"] == 0 for f in ck.glob("c*_meta.json") for m in json.loads(f.read_text()))  # every chunk, resumed or not
    return {"shard": shard, "dialogues": len(items), "skipped (no readable output)": skipped, "chunks resumed": resumed,
            "tokens (this attempt)": n_tok, "forward_s": round(fwd_s, 1), "load_s": round(load_s, 1),
            "tokens_per_s": round(n_tok / max(fwd_s, 1e-9)), "gpu": torch.cuda.get_device_properties(0).name}


@app.function(image=image, volumes={"/acts": vol}, cpu=4, memory=65536, timeout=7200, retries=0)
def assemble(job: dict, prov: dict) -> dict:
    """Concatenate the shards' chunks, in shard and chunk order, into run_fit.py's layout (pairs) or one rows file per
    layer and readout (runs)."""
    import numpy as np

    vol.reload()
    base = Path("/acts") / "toc" / f"chunks_{prov['stamp']}"
    parts = [(s, c) for s in range(job["shards"]) for c in sorted(int(p.stem[1:6]) for p in (base / f"shard{s}").glob("c*.done"))]
    meta = [m for s, c in parts for m in json.loads((base / f"shard{s}" / f"c{c:05d}_meta.json").read_text())]
    groups = ({"probe": [i for i, m in enumerate(meta) if m["probe"] != "panel"], "panel": [i for i, m in enumerate(meta) if m["probe"] == "panel"]}
              if job["kind"] == "pairs" else {"runs": list(range(len(meta)))})
    starts = np.cumsum([0] + [m["n_rows"] for m in meta])
    out = {}
    for g, idx in groups.items():
        if not idx:
            continue
        rows_idx = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in idx])
        for v in "CM":
            name = (f"toc{job['probes'][0] if g == 'probe' else 'Panel'}{v}_{prov['stamp']}" if job["kind"] == "pairs"
                    else f"toc_runs_{job['source']}_{v}_{prov['stamp']}")
            d = Path("/acts") / name
            if (d / "manifest.json").exists():  # assembled already (the manifest is written last): resumable
                out[name] = {"dialogues": len(idx), "rows": int(sum(meta[i]["n_rows"] for i in idx)), "already assembled": True}
                continue
            d.mkdir(parents=True, exist_ok=True)
            for l in job["layers"]:
                X = np.concatenate([np.load(base / f"shard{s}" / f"c{c:05d}_{v}_L{l}.npy") for s, c in parts])
                np.save(d / f"L{l}.npy", X[rows_idx])
            row, sel = 0, []
            for i in idx:
                sel.append(meta[i] | {"row_start": row})
                row += meta[i]["n_rows"]
            (d / "meta.json").write_text(json.dumps(sel))
            (d / ".manifest.tmp").write_text(json.dumps(prov | {"layers": job["layers"], "sets_sha256": job.get("sha256"), "readout": v,
                                                                "model": "Qwen/Qwen3-32B @ 9216db57, bf16, sdpa, batch 1"}))
            (d / ".manifest.tmp").rename(d / "manifest.json")
            out[name] = {"dialogues": len(idx), "rows": int(row)}
    vol.commit()
    return out


if __name__ == "__main__":
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
    cache = ROOT / ".run_cache" / "tool_output_cue_probe"
    cache.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "launch":
        st = git("status", "--porcelain", "--untracked-files=no")
        if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
            raise SystemExit("commit first: the extraction must record the commit of the code that made it")
        build = json.loads((HERE / sys.argv[2]).read_text())
        kind, arg = sys.argv[3], sys.argv[4]
        shards = int(sys.argv[sys.argv.index("--shards") + 1])
        if kind == "pairs":
            job = {"kind": "pairs", "probes": [arg] + (["panel"] if arg == "A" else []), "file": build["dialogues"]["file"],
                   "sha256": build["dialogues"]["sha256"], "layers": list(range(64))}
        else:
            layers = sorted({int(x) for x in sys.argv[5].split(",")})
            job = {"kind": "runs", "source": arg, "layers": layers}
            if arg == "planted":
                job |= {"file": build["planted"]["file"], "sha256": build["planted"]["sha256"]}
            else:  # the build's P cells (its launch record) and its packs shard hashes
                pl = json.loads((cache / f"build_launch_{build['provenance']['stamp']}.json").read_text())["plan"]
                job |= {"p_cells": pl["p_cells"], "shard_sha256": build["shard sha256"]}
        job["shards"] = shards
        stamp = sys.argv[sys.argv.index("--resume") + 1] if "--resume" in sys.argv else datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        prov = {"git": git("rev-parse", "HEAD").stdout.strip(), "stamp": stamp, "build": sys.argv[2],
                "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "probe_shift_sha256": pkg_digest(ROOT / "src" / "probe_shift"),
                "selection_sha256": hashlib.sha256(json.dumps({k: job.get(k) for k in ("p_cells", "shard_sha256")}, sort_keys=True).encode()).hexdigest()}
        fn = modal.Function.from_name(APP, "extract")
        calls = [fn.spawn(job, s, prov).object_id for s in range(shards)]
        (cache / f"extract_launch_{stamp}.json").write_text(json.dumps({"calls": calls, "job": job, "prov": prov}))
        print(json.dumps({"calls": calls, "job": {k: v for k, v in job.items() if k not in ("p_cells", "shard_sha256")}, **prov}, indent=1))
    elif sys.argv[1] == "collect":
        rec = json.loads((cache / f"extract_launch_{sys.argv[2]}.json").read_text())
        shards = [modal.FunctionCall.from_id(c).get() for c in rec["calls"]]
        out = {"job": rec["job"], "provenance": rec["prov"], "shards": shards,  # the resolved selection (P cells, shard hashes) kept
               "assembled": modal.Function.from_name(APP, "assemble").remote(rec["job"], rec["prov"])}
        path = HERE / "results" / f"extract_{rec['prov']['stamp']}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1), f"\n-> {path}")
