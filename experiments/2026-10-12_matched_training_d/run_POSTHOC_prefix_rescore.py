"""Item D, POST HOC (POSTHOC_RULE_prefix.md, frozen at caf3dac7): rescore the stored probes with token positions 0-28
(the opening shared by every version) masked from token-max. Aswin's go, before 00:24 PT 2026-10-13: "(A) Rescore with
stored probes, $6.5 (Recommended)". Numbers, hashed run keys, task and environment names only leave.

- `check` (Modal CPU): the longest token prefix shared by all 1,204 item D versions; the same for item C's 120 untouched
  transcripts, and whether C's opening equals D's (a hash of the shared ids); a replay of D's seeded background draws
  (how many of the 10,000 kept background rows fall in positions 0-28).
- `rescore` (one H200): the 404 selection and test versions read to layer 63; for every stored probe column, token-max
  over all eligible positions (unmasked, for the reproduction gate) and over eligible positions from 29 on (masked), and
  the downstream score. Each run is saved to the Volume as it finishes, with a numbers-only progress.json.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal modal deploy experiments/2026-10-12_matched_training_d/run_POSTHOC_prefix_rescore.py
  ... python experiments/2026-10-12_matched_training_d/run_POSTHOC_prefix_rescore.py launch check | rescore ; ... collect <stamp>
"""

import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import d_lib as L  # noqa: E402

APP = "probe-shift_aswin-surya_sub-matched-d-prefix_kill-2026-11-13"
VOL, LA = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09", "probe-shift_aswin-surya_sub-la-activations"
STAGED = "/in/2026-10-06_la_segment-end-v4_qwen3-32b_v1/inputs"
FITS, FITS_SHA = "/acts/matched_d/2026-10-12_172812/fits.pt", "80156b182ccc"  # results/s3_manifest_v2.json
MASK, LAYERS = 29, (2, 4, 8, 16, 32, 48, 63)
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]
app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
la = modal.Volume.from_name(LA, version=2).read_only()
hf = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift", "d_lib"))
h16 = lambda s: hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()[:16]


def code_sha():
    import importlib.util
    pkg = Path(importlib.util.find_spec("probe_shift").origin).parent
    return hashlib.sha256(b"".join(p.read_bytes() for p in (Path(__file__), Path(L.__file__), pkg / "reader.py", pkg / "models.py"))).hexdigest()


def guarded(fn):
    """stdout and stderr discarded (flushed first); failures leave as type and line numbers only."""
    def run(*a):
        import os
        import traceback
        sys.stdout.flush(), sys.stderr.flush()
        saved, null = (os.dup(1), os.dup(2)), os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1), os.dup2(null, 2)
        try:
            if code_sha() != a[-1]["code_sha256"]:
                raise RuntimeError("the deployed code is not the launching commit's: redeploy")
            return fn(*a)
        except BaseException as e:
            lines = [(Path(f.filename).name, f.lineno) for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith(("prefix_rescore.py", "d_lib.py"))]
            raise RuntimeError(f"{fn.__name__} failed: {type(e).__name__} at {lines}") from None
        finally:
            sys.stdout.flush(), sys.stderr.flush()
            os.dup2(saved[0], 1), os.dup2(saved[1], 2)
            for fd in (*saved, null):
                os.close(fd)
    return run


def tokenizer():
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN35_27B
    return AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])


def by_hash():
    out = {}
    for p in sorted(Path(STAGED).glob("shard_*.jsonl")):
        for it in map(json.loads, p.read_text().splitlines()):
            if it["arm"] == "real-honest":
                out[h16(it["key"])] = it
    return out


def lcp(seqs):
    n = min(map(len, seqs))
    return next((i for i in range(n) if any(s[i] != seqs[0][i] for s in seqs[1:])), n)


@app.function(image=image, volumes={"/in": la, "/hf": hf}, cpu=8, memory=32768, timeout=3600, retries=0)
def check(prov: dict) -> dict:
    return guarded(_check)(prov)


def _check(prov):
    import numpy as np

    from probe_shift.reader import segment_end_positions
    tok, H = tokenizer(), by_hash()
    S = {k: [H[h] for h in v] for k, v in prov["roster"].items()}
    ids, rng, picks = [], np.random.default_rng(int(L.SEED)), []  # the full run's development loop, replayed for its draws
    for split in ("dev", "sel", "test"):
        for it in S[split]:
            vs, _ = L.versions(tok, it["messages"], split, it["main_task"])
            for v in vs:
                ids.append(v["ids"])
                if split == "dev":
                    in_slot = set(v["e"]) | set(v["s"])
                    others = [i for i in v["eligible"] if i not in in_slot]
                    picks += sorted(rng.choice(others, size=min(13, len(others)), replace=False).tolist())
    keep = np.sort(rng.choice(len(picks), size=10_000, replace=False))
    n_d = lcp(ids)
    c_ids = [segment_end_positions(tok, H[h]["messages"], None)["ids"] for v in prov["c_runs"].values() for h in v]
    n_c = lcp(c_ids)
    return {"D versions": len(ids), "D shared prefix": n_d, "C untouched transcripts": len(c_ids), "C shared prefix": n_c,
            "C opening equals D's (first 29 ids)": h16(json.dumps(c_ids[0][:MASK])) == h16(json.dumps(ids[0][:MASK])),
            "background pool rows": len(picks), "kept background rows in positions 0-28": int(sum(picks[i] < MASK for i in keep)),
            "pool background rows in positions 0-28": int(sum(p < MASK for p in picks)), "provenance": prov}


@app.function(image=image, gpu="H200", volumes={"/in": la, "/hf": hf, "/acts": vol}, cpu=4, memory=65536, timeout=7200,
              startup_timeout=600, retries=0)
def rescore(prov: dict) -> dict:
    return guarded(_rescore)(prov)


def _rescore(prov):
    import torch
    from transformers import AutoModelForCausalLM

    from probe_shift.models import QWEN35_27B
    t0, tok, H = time.time(), tokenizer(), by_hash()
    raw = Path(FITS).read_bytes()
    if hashlib.sha256(raw).hexdigest()[:12] != FITS_SHA:
        raise ValueError("fits.pt is not the archived one")
    P = {k: {"mu": p["mu"].cuda().double(), "sd": p["sd"].cuda().double(), "W": [w.cuda() for w in p["W"]]}
         for k, p in torch.load(FITS, weights_only=False).items()}  # our own file, sha256 checked above
    ck = Path("/acts") / "matched_d_prefix" / prov["stamp"]
    ck.mkdir(parents=True, exist_ok=True)
    fp = {k: prov[k] for k in ("code_sha256", "roster_sha256", "stamp")}
    if (ck / "fingerprint.json").exists() and json.loads((ck / "fingerprint.json").read_text()) != fp:
        raise ValueError("this checkpoint folder was made by another launch")
    (ck / "fingerprint.json").write_text(json.dumps(fp))
    lm = AutoModelForCausalLM.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"], dtype=torch.bfloat16,
                                              device_map="cuda", attn_implementation="sdpa").eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    class Stop(Exception):
        pass

    def forward(ids):
        acts = {}

        def hook_for(layer):
            def hook(module, args, output):
                acts[layer] = (output[0] if isinstance(output, tuple) else output)[0]
                if layer == LAYERS[-1]:
                    raise Stop
            return hook
        hs = [lm.model.layers[l].register_forward_hook(hook_for(l)) for l in LAYERS]
        try:
            with torch.inference_mode():
                lm.model(input_ids=torch.tensor([ids + [pad] * (-len(ids) % 4096)], device="cuda"), use_cache=False)
        except Stop:
            pass
        finally:
            for h in hs:
                h.remove()
        return acts

    runs = [(s, H[h]) for s in ("sel", "test") for h in prov["roster"][s]]
    done = 0
    for n_run, (split, it) in enumerate(runs):
        vs, place = L.versions(tok, it["messages"], split, it["main_task"])
        for (e, sm), v in zip(L.VERSIONS, vs):
            rid = h16(f"{it['key']}|{e}{sm}")
            if (ck / f"row_{rid}.json").exists():
                done += 1
                continue
            acts = forward(v["ids"])
            el = torch.tensor(v["eligible"], device="cuda")
            late = el >= MASK
            sc, masked = {}, {}
            for (t, l, kind), p in P.items():
                if kind == "tok":
                    Zt = (acts[l][el].double() - p["mu"]) / p["sd"]
                    z = [Zt @ W[:-1] + W[-1] for W in p["W"]]
                    sc[f"{t}|{l}|tok"] = [x.max(0).values.tolist() for x in z]
                    masked[f"{t}|{l}|tok"] = [x[late].max(0).values.tolist() for x in z]
                else:
                    Z = (torch.stack([acts[l][i0:i1].float().mean(0) for i0, i1 in v["calls"]]).double() - p["mu"]) / p["sd"]
                    sc[f"{t}|{l}|down"] = [(Z @ W[:-1] + W[-1]).mean(0).tolist() for W in p["W"]]
            if not all(math.isfinite(x) for d in (sc, masked) for vals in d.values() for col in vals for x in col):
                raise ValueError("a non-finite score")
            row = {"split": split, "run": h16(it["key"]), "task": it["main_task"], "environment": it["environment"], "placement": place,
                   "eval": e, "sim": sm, "masked eligible": int((~late).sum()), "scores": sc, "scores masked": masked}
            (ck / f".row_{rid}.tmp").write_text(json.dumps(row))
            (ck / f".row_{rid}.tmp").rename(ck / f"row_{rid}.json")
        (ck / ".progress.tmp").write_text(json.dumps({"runs done": n_run + 1, "of": len(runs), "elapsed_s": round(time.time() - t0)}))
        (ck / ".progress.tmp").rename(ck / "progress.json")
        vol.commit()
    return {"resumed rows": done, "seconds": round(time.time() - t0), "gpu": torch.cuda.get_device_properties(0).name}


@app.function(image=image, volumes={"/acts": vol}, cpu=2, memory=8192, timeout=900, retries=0)
def rows(stamp: str) -> list:
    vol.reload()
    return [json.loads(f.read_text()) for f in sorted((Path("/acts") / "matched_d_prefix" / stamp).glob("row_*.json"))]


if __name__ == "__main__":
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
    cache = ROOT / ".run_cache" / "matched_d_prefix"
    cache.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "launch":
        st = git("status", "--porcelain", "--untracked-files=no")
        if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
            raise SystemExit("commit first")
        dry = json.loads((HERE / "results" / "dry_2026-10-12_171629.json").read_text())["result"]
        c_dry = json.loads((ROOT / "experiments/2026-10-12_cue_pilot/results/dry_2026-10-12_150605.json").read_text())["result"]
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        if "--resume" in sys.argv:
            stamp = sys.argv[sys.argv.index("--resume") + 1]
        prov = {"git": git("rev-parse", "HEAD").stdout.strip(), "stamp": stamp, "phase": sys.argv[2], "code_sha256": code_sha(),
                "roster": dry["roster"], "roster_sha256": dry["roster_sha256"], "c_runs": c_dry["runs"]}
        if hashlib.sha256(json.dumps(dry["roster"]).encode()).hexdigest() != dry["roster_sha256"]:
            raise SystemExit("the dry run's roster does not match its hash")
        call = modal.Function.from_name(APP, sys.argv[2]).spawn(prov)
        (cache / f"launch_{stamp}_{sys.argv[2]}.json").write_text(json.dumps({"call": call.object_id, "prov": {k: v for k, v in prov.items() if k not in ("roster", "c_runs")}}))
        print(json.dumps({"call": call.object_id, "collect key": f"{stamp}_{sys.argv[2]}", "git": prov["git"]}, indent=1))
    elif sys.argv[1] == "collect":
        rec = json.loads((cache / f"launch_{sys.argv[2]}.json").read_text())
        out = {"provenance": rec["prov"], "result": modal.FunctionCall.from_id(rec["call"]).get()}
        if rec["prov"]["phase"] == "rescore":
            rs = modal.Function.from_name(APP, "rows").remote(rec["prov"]["stamp"])
            f = cache / f"rows_{rec['prov']['stamp']}.json"
            f.write_text(json.dumps(rs))
            out["rows"] = {"file": f.name, "n": len(rs), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        path = HERE / "results" / f"prefix_POSTHOC_{sys.argv[2]}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1)[:3000], f"\n-> {path}")
