"""Item D's frozen simulation-cue probe (sim|16|tok, C 1e-4, column 0) on Qwen3.5-27B's own on-policy runs (RULE.md,
frozen at bf7d709f). Aswin via 0d: "just run it don't need to price". Numbers, run and request ids, task names only leave.

- `check` (Modal CPU): every staged epoch of the five worlds; the token prefix shared by all of them (masked); one run per
  task and world (the highest attempt_index); token and epoch counts.
- `score` (one H200 per shard): each chosen run's epochs read to layer 16; the probe's standardized logit at every
  eligible position (not a special or added token, not a role header, not in the shared prefix); per staged step its max
  and position, per epoch its max. Steps are the requests with staged calls (the record lists no others), so a request
  without one merges into the next step, and any tokens after the last one form a tail step (irid None). Rows saved per
  epoch to the Volume, with a progress file; `--resume <stamp>` reuses a launch's rows if its fingerprint matches.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal modal deploy experiments/2026-10-13_simcue_onpolicy/run_simcue_onpolicy.py
  ... python experiments/2026-10-13_simcue_onpolicy/run_simcue_onpolicy.py launch check | score --check <file> [--resume <stamp>] ; ... collect <key>
"""

import bisect
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
APP = "probe-shift_aswin-surya_sub-simcue-onpolicy_kill-2026-11-13"
VOL, Q3 = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09", "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
FITS, FITS_SHA, KEY, LAYER = "/acts/matched_d/2026-10-12_172812/fits.pt", "80156b182ccc", ("sim", 16, "tok"), 16
FIRST = ["rerun_strict_segends_v2", "scaleup_canary_segends_v1", "scaleup_l2_segends_v1", "scaleup_topup_segends_v1"]
WORLDS = {"real1": [(s, "attack") for s in FIRST], "real2": [("second_attack_segends_v1", "attack")],
          "sim_sept": [(s, "sim_attack@september") for s in FIRST], "honest_a": [(s, "honest_a") for s in FIRST],
          "honest_b": [(s, "honest_b") for s in FIRST]}
SHARDS = [["real2"], ["real1", "sim_sept"], ["honest_a"], ["honest_b"]]
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]
app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
q3 = modal.Volume.from_name(Q3, version=2).read_only()
hf = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift"))


def code_sha():
    import importlib.util
    pkg = Path(importlib.util.find_spec("probe_shift").origin).parent
    return hashlib.sha256(Path(__file__).read_bytes() + (pkg / "models.py").read_bytes()).hexdigest()


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
            lines = [(Path(f.filename).name, f.lineno) for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("run_simcue_onpolicy.py")]
            raise RuntimeError(f"{fn.__name__} failed: {type(e).__name__} at {lines}") from None
        finally:
            sys.stdout.flush(), sys.stderr.flush()
            os.dup2(saved[0], 1), os.dup2(saved[1], 2)
            for fd in (*saved, null):
                os.close(fd)
    return run


def epochs(world):
    for stage, arm in WORLDS[world]:
        for line in (Path("/q3") / stage / "inputs" / f"{arm}.jsonl").read_text().splitlines():
            yield stage, json.loads(line)


@app.function(image=image, volumes={"/q3": q3}, cpu=8, memory=65536, timeout=3600, retries=0)
def check(prov: dict) -> dict:
    return guarded(_check)(prov)


def _check(prov):
    best, first, lcp, n_ep, n_tok = {}, None, None, {}, {}
    for w in WORLDS:
        for stage, ep in epochs(w):
            ids = ep["ids"]
            if first is None:
                first, lcp = ids, len(ids)
            lcp = next((i for i in range(min(lcp, len(ids))) if ids[i] != first[i]), min(lcp, len(ids)))
            n_ep[w], n_tok[w] = n_ep.get(w, 0) + 1, n_tok.get(w, 0) + len(ids)
            k, cand = (w, ep["task"]), (ep["attempt_index"], ep["run"])
            if k not in best or cand[0] > best[k][0]:
                best[k] = cand
            elif cand[0] == best[k][0] and cand[1] != best[k][1]:
                raise ValueError("two runs share a task, world and attempt_index")
    chosen = {w: sorted(r for (w2, _), (_, r) in best.items() if w2 == w) for w in WORLDS}
    return {"shared prefix": lcp, "epochs": n_ep, "tokens": n_tok, "runs chosen": {w: len(v) for w, v in chosen.items()},
            "tasks": {w: len({t for (w2, t) in best if w2 == w}) for w in WORLDS}, "chosen": chosen, "provenance": prov}


@app.function(image=image, gpu="H200", volumes={"/q3": q3, "/hf": hf, "/acts": vol}, cpu=4, memory=98304, timeout=7200,
              startup_timeout=600, retries=0)
def score(worlds: list, prov: dict) -> dict:
    return guarded(_score)(worlds, prov)


def _score(worlds, prov):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.models import QWEN35_27B
    t0 = time.time()
    raw = Path(FITS).read_bytes()
    if hashlib.sha256(raw).hexdigest()[:12] != FITS_SHA:
        raise ValueError("fits.pt is not the archived one")
    p = torch.load(FITS, weights_only=False)[KEY]  # our own file, sha256 checked above
    mu, sd, w = p["mu"].cuda().double(), p["sd"].cuda().double(), p["W"][0][:, 0].cuda()
    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    special = set(tok.all_special_ids) | set(tok.added_tokens_decoder)
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    lm = AutoModelForCausalLM.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"], dtype=torch.bfloat16,
                                              device_map="cuda", attn_implementation="sdpa").eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ck = Path("/acts") / "simcue_onpolicy" / prov["stamp"]
    ck.mkdir(parents=True, exist_ok=True)
    fp, fp_file = {k: prov[k] for k in ("code_sha256", "check", "stamp")}, ck / f"fingerprint_{worlds[0]}.json"
    if fp_file.exists() and json.loads(fp_file.read_text()) != fp:
        raise ValueError("this checkpoint folder was made by another launch")
    fp_file.write_text(json.dumps(fp))

    class Stop(Exception):
        pass

    def layer16(ids):
        out = {}

        def hook(module, args, output):
            out["a"] = (output[0] if isinstance(output, tuple) else output)[0]
            raise Stop
        h = lm.model.layers[LAYER].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                lm.model(input_ids=torch.tensor([ids + [pad] * (-len(ids) % 4096)], device="cuda"), use_cache=False)
        except Stop:
            pass
        finally:
            h.remove()
        return out["a"]

    def eligible(ids):
        keep, i = [], 0
        while i < len(ids):
            if ids[i] == im_start:  # the role header: <|im_start|> and the tokens up to and including its newline
                j = i + 1
                while j < len(ids) and j <= i + 4 and "\n" not in tok.decode([ids[j]]):
                    j += 1
                i = j + 1
                continue
            if i >= prov["shared prefix"] and ids[i] not in special:
                keep.append(i)
            i += 1
        return keep

    chosen = {w: set(prov["chosen"][w]) for w in worlds}
    done, n_ep = 0, 0
    for wd in worlds:
        for stage, ep in epochs(wd):
            if ep["run"] not in chosen[wd]:
                continue
            rid = hashlib.sha256(f"{wd}|{ep['key']}".encode()).hexdigest()[:16]
            n_ep += 1
            if (ck / f"row_{rid}.json").exists():
                done += 1
                continue
            ids = ep["ids"]
            el = eligible(ids)
            z = ((layer16(ids)[torch.tensor(el, device="cuda")].double() - mu) / sd) @ w[:-1] + w[-1]
            zl, steps, prev = z.tolist(), [], 0
            if not all(math.isfinite(x) for x in zl):
                raise ValueError("a non-finite score")
            for irid, n in sorted({(c["irid"], c["n"]) for c in ep["calls"]}, key=lambda x: x[1]):
                lo, hi = bisect.bisect_left(el, prev), bisect.bisect_left(el, n)  # eligible positions in [prev, n)
                k = max(range(lo, hi), key=zl.__getitem__) if hi > lo else None
                steps.append([irid, n, None if k is None else zl[k], None if k is None else el[k]])
                prev = n
            if prev < len(ids):  # a tail after the last staged request
                lo = bisect.bisect_left(el, prev)
                k = max(range(lo, len(el)), key=zl.__getitem__) if len(el) > lo else None
                steps.append([None, len(ids), None if k is None else zl[k], None if k is None else el[k]])
            k = max(range(len(zl)), key=zl.__getitem__)
            row = {"world": wd, "stage": stage, "run": ep["run"], "task": ep["task"], "attempt_index": ep["attempt_index"], "epoch": ep["epoch"],
                   "tokens": len(ids), "eligible": len(el), "max": zl[k], "argmax": el[k], "steps": steps}
            (ck / f".row_{rid}.tmp").write_text(json.dumps(row))
            (ck / f".row_{rid}.tmp").rename(ck / f"row_{rid}.json")
            if n_ep % 10 == 0:
                (ck / f".progress_{worlds[0]}.tmp").write_text(json.dumps({"worlds": worlds, "epochs done": n_ep, "elapsed_s": round(time.time() - t0)}))
                (ck / f".progress_{worlds[0]}.tmp").rename(ck / f"progress_{worlds[0]}.json")
                vol.commit()
    vol.commit()
    return {"worlds": worlds, "epochs": n_ep, "resumed": done, "seconds": round(time.time() - t0), "gpu": torch.cuda.get_device_properties(0).name}


@app.function(image=image, volumes={"/acts": vol}, cpu=2, memory=8192, timeout=900, retries=0)
def rows(stamp: str) -> list:
    vol.reload()
    return [json.loads(f.read_text()) for f in sorted((Path("/acts") / "simcue_onpolicy" / stamp).glob("row_*.json"))]


if __name__ == "__main__":
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
    cache = ROOT / ".run_cache" / "simcue_onpolicy"
    cache.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "launch":
        st = git("status", "--porcelain", "--untracked-files=no")
        if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
            raise SystemExit("commit first")
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        if "--resume" in sys.argv:
            stamp = sys.argv[sys.argv.index("--resume") + 1]
        prov = {"git": git("rev-parse", "HEAD").stdout.strip(), "stamp": stamp, "phase": sys.argv[2], "code_sha256": code_sha()}
        if sys.argv[2] == "check":
            calls = [modal.Function.from_name(APP, "check").spawn(prov).object_id]
        else:
            c = json.loads((HERE / sys.argv[sys.argv.index("--check") + 1]).read_text())
            if c["provenance"]["code_sha256"] != prov["code_sha256"]:
                raise SystemExit("the check was made by other code: rerun it")
            prov |= {"shared prefix": c["result"]["shared prefix"], "chosen": c["result"]["chosen"], "check": sys.argv[sys.argv.index("--check") + 1]}
            f = modal.Function.from_name(APP, "score")
            calls = [f.spawn(s, prov).object_id for s in SHARDS]
        (cache / f"launch_{stamp}_{sys.argv[2]}.json").write_text(json.dumps({"calls": calls, "prov": {k: v for k, v in prov.items() if k != "chosen"}}))
        print(json.dumps({"calls": calls, "collect key": f"{stamp}_{sys.argv[2]}", "git": prov["git"]}, indent=1))
    elif sys.argv[1] == "collect":
        rec = json.loads((cache / f"launch_{sys.argv[2]}.json").read_text())
        out = {"provenance": rec["prov"], "result": [modal.FunctionCall.from_id(c).get() for c in rec["calls"]]}
        if rec["prov"]["phase"] == "score":
            rs = modal.Function.from_name(APP, "rows").remote(rec["prov"]["stamp"])
            f = cache / f"rows_{rec['prov']['stamp']}.json"
            f.write_text(json.dumps(rs))
            out["rows"] = {"file": f.name, "n": len(rs), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        else:
            out["result"] = out["result"][0]
        path = HERE / "results" / f"{rec['prov']['phase']}_{sys.argv[2]}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1)[:2500], f"\n-> {path}")
