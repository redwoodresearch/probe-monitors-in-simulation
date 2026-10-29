"""Item D, POST HOC (after the screen): what sets the evaluation probe's tied maximum (228 of 404 versions score exactly
3.047247196586617 at layer 48, token-max)? 0d's go under Aswin's "<$5 just run", about 20:20 PT 2026-10-12. Numbers and
hashes only leave: no text, no token ids.

- `spans` (Modal CPU): for every selection and test version (404), every system and user message span found the way
  d_lib.layout finds it (text.find(content, cursor)) is checked to sit between its own role header and the next
  <|im_end|>; counts of failures by role. Also the longest token prefix shared by all 404 versions (and per environment),
  and how many eligible positions lie inside it.
- `tokens` (one H200): 24 versions picked from the rows file by a seeded rule (below). Read to layer 48; the stored
  probes (fits.pt, sha256 checked) at evaluation layers 2, 8 and 48 and simulation layer 16 (C 1e-4, column 0) scored at
  every eligible position. Per version and probe: the argmax position, its message index, role and offsets, its logit
  against the stored run score, how many positions tie with it, a hash of the token prefix up to it and a hash of its
  activation vector; the spans check for that version.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal modal deploy experiments/2026-10-12_matched_training_d/run_POSTHOC_floor_check.py
  ... python experiments/2026-10-12_matched_training_d/run_POSTHOC_floor_check.py launch spans | tokens ; ... collect <stamp>
"""

import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import d_lib as L  # noqa: E402

APP = "probe-shift_aswin-surya_sub-matched-d-floor_kill-2026-11-12"
VOL, LA = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09", "probe-shift_aswin-surya_sub-la-activations"
STAGED = "/in/2026-10-06_la_segment-end-v4_qwen3-32b_v1/inputs"
FITS, FITS_SHA = "/acts/matched_d/2026-10-12_172812/fits.pt", "80156b182ccc"  # results/s3_manifest_v2.json
PROBES = [("eval", 48, "tok"), ("eval", 8, "tok"), ("eval", 2, "tok"), ("sim", 16, "tok")]  # all at C 1e-4 (index 0), column 0
FLOOR = 3.047247196586617
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]
app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2).read_only()
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
    """stdout and stderr discarded; failures leave as type and line numbers only."""
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
            lines = [(Path(f.filename).name, f.lineno) for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith(("run_POSTHOC_floor_check.py", "d_lib.py"))]
            raise RuntimeError(f"{fn.__name__} failed: {type(e).__name__} at {lines}") from None
        finally:
            sys.stdout.flush(), sys.stderr.flush()
            os.dup2(saved[0], 1), os.dup2(saved[1], 2)
            for fd in (*saved, null):
                os.close(fd)
    return run


def message_spans(tok, msgs):
    """d_lib.layout's span walk (text.find(content, cursor) for system and user messages), returning every message's
    (index, role, a, b) and, for system and user ones, whether the span sits between its role header and <|im_end|>."""
    from probe_shift.reader import _layout, prepare
    text, found, enc = _layout(tok, msgs)
    agents, tools = iter([f for f in found if f[0] == "agent"]), iter([f for f in found if f[0] == "tool_output"])
    out, cursor = [], 0
    for j, m in enumerate(prepare(tok, msgs)):
        if m["role"] == "assistant":
            _, _, _, _, a, b = next(agents)
            out.append((j, "assistant", a, b, None, None))
            cursor = b
            continue
        if m["role"] == "tool":
            _, _, _, _, a, b = next(tools)
            out.append((j, "tool", a, b, None, None))
        else:
            c = m.get("content") or ""
            a = text.find(c, cursor)
            b = a + len(c)
            head = f"<|im_start|>{m['role']}\n"
            out.append((j, m["role"], a, b, a >= len(head) and text[a - len(head):a] == head and text[b:b + 10] == "<|im_end|>", len(c)))
        cursor = b
    return text, enc, out


def splits(prov):
    by_h = {}
    for p in sorted(Path(STAGED).glob("shard_*.jsonl")):
        for it in map(json.loads, p.read_text().splitlines()):
            if it["arm"] == "real-honest":
                by_h[h16(it["key"])] = it
    return {k: [by_h[h] for h in prov["roster"][k]] for k in ("sel", "test")}


@app.function(image=image, volumes={"/in": la, "/hf": hf}, cpu=8, memory=32768, timeout=3600, retries=0)
def spans(prov: dict) -> dict:
    return guarded(_spans)(prov)


def _spans(prov):
    from collections import Counter

    from transformers import AutoTokenizer

    from probe_shift.models import QWEN35_27B
    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    bad, total, ids_all, elig_all, env_of, detail = Counter(), Counter(), [], [], [], Counter()
    for split, runs in splits(prov).items():
        for it in runs:
            place = L.placement(it["main_task"])
            (ec, en), (sc, sn) = L.wording(split, it["main_task"])
            vs = L.versions(tok, it["messages"], split, it["main_task"])[0]
            for (e, s), v in zip(L.VERSIONS, vs):
                m, _ = L.edit(it["messages"], place, ec if e else en, sc if s else sn)
                text, enc, sp = message_spans(tok, m)
                starts = [x for x, _ in enc["offset_mapping"]]
                for _, role, a, b, ok, n_c in sp:
                    if ok is not None:
                        total[role] += 1
                        bad[role] += not ok
                        if not ok:  # (role, content length, span length, eligible positions inside the span)
                            detail[f"{role}|{n_c}|{b - a}|{sum(a <= starts[i] < b for i in v['eligible'])}"] += 1
                ids_all.append(v["ids"])
                elig_all.append(set(v["eligible"]))
                env_of.append(it["environment"])

    def lcp(seqs):
        n = min(map(len, seqs))
        for i in range(n):
            if any(s[i] != seqs[0][i] for s in seqs[1:]):
                return i
        return n
    n_all = lcp(ids_all)
    per_env = {env: lcp([x for x, e2 in zip(ids_all, env_of) if e2 == env]) for env in sorted(set(env_of))}
    return {"versions": len(ids_all), "system/user spans checked": dict(total), "spans not between their header and <|im_end|>": dict(bad),
            "failing spans by role|content length|span length|eligible positions inside": dict(detail),
            "shared token prefix, all versions": n_all, "eligible positions inside it (min, max over versions)": [min(sum(i < n_all for i in el) for el in elig_all), max(sum(i < n_all for i in el) for el in elig_all)],
            "shared token prefix per environment": per_env, "provenance": prov}


@app.function(image=image, gpu="H200", volumes={"/in": la, "/hf": hf, "/acts": vol}, cpu=4, memory=65536, timeout=3600,
              startup_timeout=600, retries=0)
def tokens(picks: list, prov: dict) -> dict:
    return guarded(_tokens)(picks, prov)


def _tokens(picks, prov):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.models import QWEN35_27B
    raw = Path(FITS).read_bytes()
    if hashlib.sha256(raw).hexdigest()[:12] != FITS_SHA:
        raise ValueError("fits.pt is not the archived one")
    P = torch.load(FITS, weights_only=False)  # our own file, sha256 checked above
    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    lm = AutoModelForCausalLM.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"], dtype=torch.bfloat16,
                                              device_map="cuda", attn_implementation="sdpa").eval()
    pad, layers = (tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id), sorted({l for _, l, _ in PROBES})

    class Stop(Exception):
        pass

    def forward(ids):
        acts = {}

        def hook_for(layer):
            def hook(module, args, output):
                acts[layer] = (output[0] if isinstance(output, tuple) else output)[0]
                if layer == layers[-1]:
                    raise Stop
            return hook
        hs = [lm.model.layers[l].register_forward_hook(hook_for(l)) for l in layers]
        try:
            with torch.inference_mode():
                lm.model(input_ids=torch.tensor([ids + [pad] * (-len(ids) % 4096)], device="cuda"), use_cache=False)
        except Stop:
            pass
        finally:
            for h in hs:
                h.remove()
        return acts

    S, out = splits(prov), []
    by = {(sp, h16(it["key"])): it for sp, runs in S.items() for it in runs}
    for pk in picks:
        it = by[(pk["split"], pk["run"])]
        v = L.versions(tok, it["messages"], pk["split"], it["main_task"])[0][L.VERSIONS.index((pk["eval"], pk["sim"]))]
        place = L.placement(it["main_task"])
        (ec, en), (sc, sn) = L.wording(pk["split"], it["main_task"])
        m, _ = L.edit(it["messages"], place, ec if pk["eval"] else en, sc if pk["sim"] else sn)
        text, enc, sp = message_spans(tok, m)
        starts = [s for s, _ in enc["offset_mapping"]]
        acts, el = forward(v["ids"]), torch.tensor(v["eligible"], device="cuda")
        rec = {k: pk[k] for k in ("split", "run", "eval", "sim", "why")} | {"placement": place, "tokens": len(v["ids"]),
               "spans ok": all(x[4] for x in sp if x[4] is not None), "probes": {}}
        for t, l, kind in PROBES:
            p = P[(t, l, kind)]
            W = p["W"][0].cuda()
            z = ((acts[l][el].double() - p["mu"].cuda()) / p["sd"].cuda()) @ W[:-1, 0] + W[-1, 0]
            k = int(z.argmax())
            pos = v["eligible"][k]
            j, role, a, b, _, _ = next(x for x in sp if x[2] <= starts[pos] < max(x[3], x[2] + 1))
            rec["probes"][f"{t}|{l}|{kind}"] = {
                "max logit": float(z[k]), "stored run score": pk["stored"].get(f"{t}|{l}|{kind}"), "positions tied with the max": int((z == z[k]).sum()),
                "argmax position": pos, "message index": j, "role": role, "chars into message": starts[pos] - a,
                "eligible tokens before it in its message": sum(i < pos and a <= starts[i] < b for i in v["eligible"]),
                "prefix hash": h16(json.dumps(v["ids"][:pos + 1])), "activation hash": h16(acts[l][pos].float().cpu().numpy().tobytes())}
        out.append(rec)
    return {"picks": out, "gpu": torch.cuda.get_device_properties(0).name, "provenance": prov}


def choose(rows, seed=20261020):
    """12 tied cue-absent versions and 4 tied cue-present versions (random, seeded) and every test false alarm."""
    key, rnd = "eval|48|tok", random.Random(seed)
    sc = lambda r: r["scores"][key][0][0]
    tied_abs = [r for r in rows if not r["eval"] and sc(r) == FLOOR]
    tied_pre = [r for r in rows if r["eval"] and sc(r) == FLOOR]
    alarms = [r for r in rows if r["split"] == "test" and not r["eval"] and sc(r) > FLOOR]
    pick = [(r, "tied, cue absent") for r in rnd.sample(tied_abs, 12)] + [(r, "tied, cue present") for r in rnd.sample(tied_pre, 4)] + [(r, "test false alarm") for r in alarms]
    return [{"split": r["split"], "run": r["run"], "eval": r["eval"], "sim": r["sim"], "why": why,
             "stored": {f"{t}|{l}|{k}": r["scores"][f"{t}|{l}|{k}"][0][0] for t, l, k in PROBES}} for r, why in pick]


if __name__ == "__main__":
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
    cache = ROOT / ".run_cache" / "matched_d_floor"
    cache.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "launch":
        st = git("status", "--porcelain", "--untracked-files=no")
        if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
            raise SystemExit("commit first")
        dry = json.loads((HERE / "results" / "dry_2026-10-12_171629.json").read_text())["result"]
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        prov = {"git": git("rev-parse", "HEAD").stdout.strip(), "stamp": stamp, "phase": sys.argv[2], "code_sha256": code_sha(),
                "roster": {k: dry["roster"][k] for k in ("sel", "test")}}
        if sys.argv[2] == "spans":
            call = modal.Function.from_name(APP, "spans").spawn(prov)
        else:
            rf = ROOT / ".run_cache" / "matched_d" / "rows_2026-10-12_172812.json"
            if hashlib.sha256(rf.read_bytes()).hexdigest()[:12] != "276a95dfa7d2":
                raise SystemExit("the rows file is not the collected one")
            picks = choose(json.loads(rf.read_text()))
            if len(picks) != 24:
                raise SystemExit(f"{len(picks)} picks, not 24")
            call = modal.Function.from_name(APP, "tokens").spawn(picks, prov)
        (cache / f"launch_{stamp}.json").write_text(json.dumps({"call": call.object_id, "prov": {k: v for k, v in prov.items() if k != "roster"}}))
        print(json.dumps({"call": call.object_id, "stamp": stamp, "phase": sys.argv[2], "git": prov["git"]}, indent=1))
    elif sys.argv[1] == "collect":
        rec = json.loads((cache / f"launch_{sys.argv[2]}.json").read_text())
        res = modal.FunctionCall.from_id(rec["call"]).get()
        res["provenance"] = rec["prov"]
        path = HERE / "results" / f"floor_check_POSTHOC_{rec['prov']['phase']}_{sys.argv[2]}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1)[:4000], f"\n-> {path}")
