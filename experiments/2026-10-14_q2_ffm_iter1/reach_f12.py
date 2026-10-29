"""RULE.md Addition 3, part B, POST HOC: does F12's inserted \\r reach each reader's input? For every simulated run F12
changes (full + F9 + F11, attack and honest; F12 derived exactly as stage_f12.py derives it, nothing staged), each reader
renders the run before and after F12 exactly as its extractor does, and the token ids are compared: whole run, inside the
reader's cut, and per changed output (the run with only that output changed). Modal CPU; counts only, no text.

Readers and their renderings (the extractors' own code; the prep functions below are verbatim copies):
  Qwen3-32B      probe_shift.reader.segment_end_positions (transformers 4.56.2; d9's all-layers extractor), cut 32,768
  Qwen3.5-27B    probe_shift.reader.segment_end_positions (transformers 5.19.0; q2_reply_extract_qwen35), cut 32,768
  Olmo-3.1-32B   prep + apply_chat_template + tokenize (4.57.1; experiments/2026-10-12_plantslip_transfer/
                 q2_reply_extract_olmo.py), cut 32,768
  Llama-3.3-70B  prep (one message per call) + apply_chat_template(date_string) + tokenize (4.57.1; experiments/
                 2026-10-12_q2_llama_reader/run_extract_llama.py), cut 32,768
  Seed-OSS-36B   prep + apply_chat_template + tokenize (4.57.1; run_extract_seedcall_f11.py on aswin/q2-f10-tabs), context
                 65,536

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_q2_ffm_iter1/reach_f12.py
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
FOLDER = "experiments/2026-10-14_q2_ffm_iter1"
if modal.is_local():
    sys.path.insert(0, str(REPO / "src"))
APP = "probe-shift_aswin-surya_sub-q2-ffm1_reach_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
V4 = "2026-10-06_la_segment-end-v4_qwen3-32b_v1"
OUT = "2026-10-14_q2_ffm_iter1/results"
B7_SHA = "8bdaa55d0a395e5a36415bb001f1eff13c9eaabb9ff9f5c71e56bdd26e9e5042"
SIM = {"attack": (("2026-10-13_q2_fxavg_inputs_v1/inputs/shard_2.jsonl", "009466e7b8eb0f3f265a840157c800826ff459cb387f9824f05fabdd6f09fe55", "sim-attack-fx-fullF9", 939),
                  ("2026-10-13_q2_f11q_inputs_v1/inputs/shard_2.jsonl", "7a97837283ab161820ff78fbb87d645fc8753ab19a13aa694f77b0a78a7bdd6b", "sim-attack-fx-fullF9F11", 427)),
       "honest": (("2026-10-13_q2_fxavg_inputs_v1/inputs/shard_4.jsonl", "eaa15a3c136de18916ae66b4bfd28a9a9779f7d14e94dc9e83cb82e20c7bdd0c", "sim-honest-fx-fullF9", 437),
                  ("2026-10-13_q2_f11q_inputs_v1/inputs/shard_3.jsonl", "78445677dc32700b49ccd9f9d30e73fe096c895584ce5bce109db0b663df3e91", "sim-honest-fx-fullF9F11", 226))}
READERS = {  # name -> (model, revision, rendering, cut, image key)
    "Qwen3-32B": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137", "segment_end", 32768, "t4562"),
    "Qwen3.5-27B": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", "segment_end", 32768, "t519"),
    "Olmo-3.1-32B": ("allenai/Olmo-3.1-32B-Instruct", "ac0587e4a7744a551c059d8cd17ba220bc940dae", "olmo", 32768, "t4571"),
    "Llama-3.3-70B": ("meta-llama/Llama-3.3-70B-Instruct", "6f6073b423013f6a7d4d9f39144961bfbfbc386b", "llama", 32768, "llama"),
    "Seed-OSS-36B": ("ByteDance-Seed/Seed-OSS-36B-Instruct", "497f1dca95ebdec98e41d517b9f060ee753c902f", "seed", 65536, "t4571"),
}
LLAMA_DATE = "26 Jul 2024"  # run_extract_llama.py's DATE
# Llama's snapshot is not on the HF cache Volume (the first run's OSError, 10:03 PT 10-14); its extractor reads it from this
# Volume, read-only, as HF_HUB_CACHE=/llama (run_extract_llama.py lines 36, 50). No token is needed offline.
LLAMA_WEIGHTS = "probe-shift_aswin-surya_sub-apollo_llama-cache"

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
hf_cache = modal.Volume.from_name(HF_CACHE).read_only()
llama_weights = modal.Volume.from_name(LLAMA_WEIGHTS).read_only()


def _img(pips, env=None):
    return (modal.Image.debian_slim(python_version="3.12").pip_install(*pips)
            .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "AUTHOR_SPLIT_STAGE": "/tmp/author_split_stage", **(env or {})})
            .add_local_python_source("probe_shift")
            .add_local_file(HERE / "f12_lib.py", f"/root/{FOLDER}/f12_lib.py")
            .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/rerender.py", "/root/swap/rerender.py")
            .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/str_replace_tells.py", "/root/swap/str_replace_tells.py")
            .add_local_file(REPO / "experiments/2026-10-09_fixed_sim/fixes.py", "/root/fsim/fixes.py"))


derive_image = _img(["numpy==2.5.4", "scipy==1.18.1", "scikit-learn==1.9.1", "joblib==1.6.0", "boto3==1.43.113", "pyyaml==6.0.3"]) \
    .add_local_file(REPO / "splits" / "exclusions_v4.json", "/root/splits/exclusions_v4.json") \
    .add_local_file(HERE / "analyze_POSTHOC_author_split.py", f"/root/{FOLDER}/analyze_POSTHOC_author_split.py")
IMAGES = {"t4562": _img(["transformers==4.56.2", "jinja2==3.1.6", "numpy"]),
          "t519": _img(["transformers==5.19.0", "jinja2==3.1.6", "numpy"]),
          "t4571": _img(["transformers==4.57.1", "jinja2==3.1.6", "huggingface_hub==0.35.3", "numpy"])}
IMAGES["llama"] = _img(["transformers==4.57.1", "jinja2==3.1.6", "huggingface_hub==0.35.3", "numpy"], {"HF_HUB_CACHE": "/llama"})


class Gate(Exception):
    """A failed check; its message holds counts only."""


def say(*a):
    print(datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S"), *a, flush=True)


def _paths():
    for p in ("/root/swap", "/root/fsim", f"/root/{FOLDER}"):
        if p not in sys.path:
            sys.path.insert(0, p)


def sim_rows() -> dict:
    """side -> key -> messages, full + F9 with the F11 rows replacing theirs (sha256-pinned shard files)."""
    out = {}
    for side, specs in SIM.items():
        parts = []
        for p, h, arm, n in specs:
            raw = (Path("/acts") / p).read_bytes()
            if hashlib.sha256(raw).hexdigest() != h:
                raise Gate(f"{p}: sha256 differs from its stage record")
            rows = {r["key"]: r for r in map(json.loads, raw.decode().splitlines())}
            if len(rows) != n or {r["arm"] for r in rows.values()} != {arm}:
                raise Gate(f"{p}: not {n} runs of {arm}")
            parts.append(rows)
        out[side] = {k: r["messages"] for k, r in (parts[0] | parts[1]).items()}
    return out


@app.function(image=derive_image, volumes={"/acts": acts}, cpu=8, memory=32768, timeout=1800, retries=0)
def derive() -> dict:
    """stage_f12.py's derivation, unchanged: the IN cells, and per simulated run F12 changes, its changed messages."""
    _paths()
    import f12_lib as L
    import fixes

    if hashlib.sha256(Path(f"/root/{FOLDER}/analyze_POSTHOC_author_split.py").read_bytes()).hexdigest() != B7_SHA:
        raise Gate("b7's analysis module differs from its pin")
    import analyze_POSTHOC_author_split as A

    _, first, second, _ = A.cohort_rows()
    want = {r["key"] for r in [*first.values(), *second.values()]}
    v4 = {}
    for p in sorted((Path("/acts") / V4 / "inputs").glob("shard_*.jsonl")):
        for line in open(p):
            it = json.loads(line)
            if it["arm"] == "real-honest" and it["key"] in want:
                v4[it["key"]] = it["messages"]
    if set(v4) != want:
        raise Gate("wanted real runs missing from the v4 inputs")
    c = Counter()
    cells, table = L.derive([v4[r["key"]] for r in second.values()], [v4[r["key"]] for r in first.values()], fixes, c)
    changed = {}
    for side, rows in sim_rows().items():
        for k, m in rows.items():
            _, ch = L.f12(m, cells, fixes)
            if ch:
                changed[f"{side}|{k}"] = ch
    say("derived", {f: sorted(t) for f, t in cells.items()}, len(changed))
    return {"cells": {f: sorted(t) for f, t in cells.items()}, "cell table": table, "changed": changed}


# --- the extractors' prep functions, verbatim ---
def prep_olmo(msgs):
    """Olmo's template takes string content and tool-call arguments as a dict; the staged messages may hold either form."""
    out = []
    for m in msgs:
        m = dict(m); c = m.get("content")  # noqa: E702
        m["content"] = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
        if m.get("tool_calls"):
            m["tool_calls"] = [{**tc, "function": {**(tc.get("function") or {}), "arguments": (lambda a: json.loads(a) if isinstance(a, str) else (a or {}))(
                (tc.get("function") or {}).get("arguments"))}} for tc in m["tool_calls"]]
        out.append(m)
    return out


prep_seed = prep_olmo  # run_extract_seedcall_f11.py's prep is the same text


def prep_llama(msgs):
    """Llama's template, per PLAN.md: string content; tool-call arguments as a dict; each assistant turn split into its
    text (if any, after trimming) and one message per tool call. Tool outputs and everything else in their order."""
    out = []
    for m in msgs:
        m = dict(m); c = m.get("content")  # noqa: E702
        m["content"] = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = [{**tc, "function": {**(tc.get("function") or {}), "arguments": (lambda a: json.loads(a) if isinstance(a, str) else (a or {}))(
                (tc.get("function") or {}).get("arguments"))}} for tc in m["tool_calls"]]
            if m["content"].strip():
                out.append({"role": "assistant", "content": m["content"]})
            out += [{"role": "assistant", "content": "", "tool_calls": [tc]} for tc in calls]
        else:
            m.pop("tool_calls", None)
            out.append(m)
    return out


_tok = None


def _load(name):
    global _tok
    from transformers import AutoTokenizer
    model, rev = READERS[name][:2]
    _tok = AutoTokenizer.from_pretrained(model, revision=rev)


def _render(name, msgs):
    """(rendered text, token ids, uncut) as the reader's extractor renders the run."""
    kind = READERS[name][2]
    if kind == "segment_end":
        from probe_shift.reader import _layout
        text, _, enc = _layout(_tok, msgs)
        return text, enc["input_ids"]
    if kind == "llama":
        text = _tok.apply_chat_template(prep_llama(msgs), tokenize=False, date_string=LLAMA_DATE)
    else:
        text = _tok.apply_chat_template(prep_olmo(msgs) if kind == "olmo" else prep_seed(msgs), tokenize=False)
    return text, _tok(text, add_special_tokens=False)["input_ids"]


def _one(job):
    name, side, key, old, ch = job
    _paths()
    import f12_lib as L
    import fixes
    cut = READERS[name][3]
    cells = job_cells
    new, ch2 = L.f12(old, cells, fixes)
    if ch2 != ch:
        raise Gate("F12 changed other messages than derive() recorded")
    t0, i0 = _render(name, old)
    t1, i1 = _render(name, new)
    r = Counter({"runs": 1, "outputs changed by F12": len(ch), "\\r inserted": sum(len(new[i]["content"]) - len(old[i]["content"]) for i in ch),
                 "runs whose rendered text differs": t0 != t1, "runs whose token ids differ": i0 != i1,
                 "runs whose token ids differ inside the cut": i0[:cut] != i1[:cut],
                 "\\r characters in the rendering (after - before)": t1.count("\r") - t0.count("\r"),
                 "escaped \\\\r in the rendering (after - before)": t1.count("\\r") - t0.count("\\r")})
    for i in ch:
        one = [dict(m) for m in old]
        one[i] = new[i]
        _, ij = _render(name, one)
        r["outputs whose change alters the token ids"] += ij != i0
        r["outputs whose change alters the token ids inside the cut"] += ij[:cut] != i0[:cut]
    return side, r


job_cells = {}


def _init(name, cells):
    global job_cells
    job_cells = {f: set(t) for f, t in cells.items()}
    _paths()
    _load(name)


def read_with(name: str, derived: dict) -> dict:
    from multiprocessing import get_context
    _paths()
    rows = sim_rows()
    jobs = [(name, s, k, rows[s][k], ch) for sk, ch in sorted(derived["changed"].items()) for s, k in [sk.split("|", 1)]]
    _load(name)  # preflight: a tokenizer that fails to load stops here
    out = {"attack": Counter(), "honest": Counter()}
    with get_context("fork").Pool(16, initializer=_init, initargs=(name, derived["cells"])) as pool:
        for side, r in pool.imap_unordered(_one, jobs, chunksize=2):
            out[side] += r
    tot = out["attack"] + out["honest"]
    share = tot["outputs whose change alters the token ids"] / tot["outputs changed by F12"] if tot["outputs changed by F12"] else None
    verdict = None if share is None else ("survives" if share >= 0.99 else "dropped (F12 is a no-op)" if share == 0 else "partly survives")
    return {"reader": name, "model": READERS[name][0], "revision": READERS[name][1], "rendering": READERS[name][2], "cut": READERS[name][3],
            "by side": {s: dict(sorted(c.items())) for s, c in out.items()},
            "share of changed outputs whose change alters the token ids": share, "verdict": verdict}


RKW = dict(volumes={"/acts": acts, "/hf": hf_cache}, cpu=16, memory=32768, timeout=3600, retries=0)


@app.function(image=IMAGES["t4562"], **RKW)
def read_t4562(name: str, derived: dict) -> dict:
    return read_with(name, derived)


@app.function(image=IMAGES["t519"], **RKW)
def read_t519(name: str, derived: dict) -> dict:
    return read_with(name, derived)


@app.function(image=IMAGES["t4571"], **RKW)
def read_t4571(name: str, derived: dict) -> dict:
    return read_with(name, derived)


@app.function(image=IMAGES["llama"], volumes={"/acts": acts, "/llama": llama_weights}, cpu=16, memory=32768, timeout=3600, retries=0)
def read_llama(name: str, derived: dict) -> dict:
    return read_with(name, derived)


READ = {"t4562": read_t4562, "t519": read_t519, "t4571": read_t4571, "llama": read_llama}


@app.function(image=derive_image, volumes={"/acts": acts}, cpu=1, memory=2048, timeout=3600, retries=0)
def orchestrate(git: str, stamp: str, only: list, label: str = "") -> dict:
    d = derive.remote()
    calls = {name: READ[spec[4]].spawn(name, d) for name, spec in READERS.items() if not only or name in only}
    res = {}
    for name, fc in calls.items():
        try:
            res[name] = fc.get()
        except Exception as e:  # noqa: BLE001 — the type only (a tokenizer not cached, a template error); never text
            res[name] = {"reader": name, "error": type(e).__name__}
    out = {"POST HOC": True, "status": "preliminary", "rule": "RULE.md Addition 3 (part B)", "git": git, "stamp": stamp,
           "IN F12": d["cells"], "cell table": d["cell table"],
           "runs changed by F12": {s: sum(1 for k in d["changed"] if k.startswith(s + "|")) for s in ("attack", "honest")},
           "readers": res, "only": only, "label": label}
    dst = Path("/acts") / OUT / f"reach_f12_{stamp}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    acts.commit()
    return out


@app.local_entrypoint()
def main(only: str = "", label: str = ""):
    """--only "Llama-3.3-70B" reruns one reader (comma-separated names); --label is written into the result."""
    for f in (Path(__file__), HERE / "RULE.md", HERE / "f12_lib.py", HERE / "analyze_POSTHOC_author_split.py"):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
                or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    names = [x for x in only.split(",") if x]
    if any(x not in READERS for x in names):
        raise SystemExit("--only takes reader names")
    out = orchestrate.remote(git, stamp, names, label)
    p = HERE / "results" / f"reach_f12_{stamp}.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", p, {n: r.get("verdict", r.get("error")) for n, r in out["readers"].items()})
