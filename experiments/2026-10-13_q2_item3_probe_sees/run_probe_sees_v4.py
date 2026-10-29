"""Item 3 v4 (PLAN.md here, Amendment 1): v3's checks extended in scope, counts and keys only, no text in any output.

(1) Source scan of EVERY staged input folder the readers extracted from on the activations Volume (any folder with
    inputs/*.jsonl), except the sealed zoo C oct11c sets, which are never read. Rows without a side task inherit the
    (environment, side task) of the same key in #36's / v4's base inputs.
(2) Rendered per-reader inputs: a seeded sample of base runs (200 per arm, seed 20261013) plus every run flagged on any
    check in the source scan, rendered exactly as each reader's extractor does (its own prep code, taken verbatim by AST
    from the extractor at the pinned commit; the same apply_chat_template call and arguments; Qwen3-32B also at its
    32,768-token cut). The same checks run on the whole rendered string, compared with the run's source flags (any part).
(3) The keys-only own-words file for d9, as v3.

  modal run --detach experiments/2026-10-13_q2_item3_probe_sees/run_probe_sees_v4.py --assets <dir> --prep <prep_sources.json> --git <sha>
"""
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-q2-item3_probe-sees-v4_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
WORK = "probe-shift_aswin-surya_sub-q2-item3_work_kill-2026-11-13"
LLAMA = "probe-shift_aswin-surya_sub-apollo_llama-cache"
BASE = ("2026-10-06_la_segment-end_qwen3-32b_v1", "2026-10-06_la_segment-end-v4_qwen3-32b_v1")
SEALED = ("oct11c", "lean230")
GENERIC = ("side task", "side_task", "red team", "red-team", "covert", "secret objective", "hidden objective", "sabotage", "attack")
OWN_WORDS = ("generic:side_task", "generic:sabotage", "generic:covert")
PARTS = ("system", "user", "assistant_text", "tool_call_args", "tool_output")
WIN, STRIDE, MINLINE = 60, 30, 40
SAMPLE, SEED, QWEN3_CUT = 200, 20261013, 32768
READERS = {  # model, revision, image key, prep key, template kwargs (each as in its extractor)
    "qwen3-32b": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137", "t456", None, {"enable_thinking": False}),
    "qwen3.5-27b": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", "t519", "qwen35", {"enable_thinking": False}),
    "olmo-3.1-32b": ("allenai/Olmo-3.1-32B-Instruct", "ac0587e4a7744a551c059d8cd17ba220bc940dae", "t457", "olmo", {}),
    "llama-3.3-70b": ("meta-llama/Llama-3.3-70B-Instruct", "6f6073b423013f6a7d4d9f39144961bfbfbc386b", "t457llama", "llama", {"date_string": "26 Jul 2024"}),
    "seed-oss-36b": ("ByteDance-Seed/Seed-OSS-36B-Instruct", "497f1dca95ebdec98e41d517b9f060ee753c902f", "t457", "seed", {}),
}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS)
work = modal.Volume.from_name(WORK, create_if_missing=True)
llama_cache = modal.Volume.from_name(LLAMA).read_only()
base_img = modal.Image.debian_slim(python_version="3.12")
IMAGES = {
    "t456": base_img.uv_pip_install("transformers==4.56.2", "jinja2", "huggingface_hub", "numpy").env({"HF_HOME": "/hf"}),
    "t519": base_img.uv_pip_install("transformers==5.19.0", "jinja2", "huggingface_hub", "numpy").env({"HF_HOME": "/hf"}),
    "t457": base_img.uv_pip_install("transformers==4.57.1", "jinja2==3.1.6", "huggingface_hub==0.35.3", "numpy").env({"HF_HOME": "/hf"}),
    "t457llama": base_img.uv_pip_install("transformers==4.57.1", "jinja2==3.1.6", "huggingface_hub==0.35.3", "numpy").env(
        {"HF_HUB_CACHE": "/llama", "HF_HUB_OFFLINE": "1"}),
}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def pieces(text: str) -> set[str]:
    """Distinctive substrings of a doc: its lines/sentences of >= MINLINE characters and WIN-character windows."""
    n = norm(text)
    lines = [norm(s) for s in text.splitlines()]
    sents = [norm(s) for s in re.split(r"(?<=[.!?:])\s", text)]
    out = {s for s in lines + sents if len(s) >= MINLINE}
    out |= {n[i:i + WIN] for i in range(0, max(1, len(n) - WIN + 1), STRIDE) if len(n[i:i + WIN]) == WIN}
    if len(n) >= WIN:
        out.add(n[-WIN:])
    return out


def strings(v):
    """Every raw string inside a call's arguments (unescaped), and dict keys."""
    if isinstance(v, str):
        try:
            return [v] + strings(json.loads(v)) if v.lstrip()[:1] in "{[" else [v]
        except ValueError:
            return [v]
    if isinstance(v, dict):
        return [s for k, x in v.items() for s in [k] + strings(x)]
    if isinstance(v, list):
        return [s for x in v for s in strings(x)]
    return [] if v is None else [str(v)]


def read_assets(root: Path) -> dict:
    """Raw texts only (no processing here): {"env|side_task": {"goal.md": ..., "scoring_details.md": ...}}, attacker prompt."""
    docs = {}
    for env_dir in sorted((root / "side_tasks").iterdir()):
        for st_dir in sorted(env_dir.iterdir()):
            docs[f"{env_dir.name}|{st_dir.name}"] = {f: (st_dir / f).read_text() for f in ("goal.md", "scoring_details.md") if (st_dir / f).exists()}
    return {"docs": docs, "attacker": (root / "attacker_2026_08_28.md").read_text()}


def build(raw: dict):
    goals, envgoals = {}, defaultdict(set)
    for k, d in raw["docs"].items():
        env, st = k.split("|")
        goals[(env, st)] = set().union(*[pieces(v) for v in d.values()]) if d else set()
        if "goal.md" in d:
            envgoals[env] |= pieces(d["goal.md"])
    att = [norm(l) for l in raw["attacker"].splitlines() if len(norm(l)) >= MINLINE and "{" not in l and "}" not in l]
    return goals, dict(envgoals), att


def parts_of(messages: list[dict]) -> dict[str, list[str]]:
    out = {p: [] for p in PARTS}
    for m in messages:
        c = m.get("content")
        c = c if isinstance(c, str) else "".join(x.get("text") or "" for x in c or [] if isinstance(x, dict))
        if m["role"] == "system":
            out["system"].append(c)
        elif m["role"] == "user":
            out["user"].append(c)
        elif m["role"] == "assistant":
            out["assistant_text"].append(c or "")
            for tc in m.get("tool_calls") or []:
                f = tc.get("function") or {}
                out["tool_call_args"] += [f.get("name") or "", json.dumps(f.get("arguments"), ensure_ascii=False)] + strings(f.get("arguments"))
        elif m["role"] == "tool":
            out["tool_output"].append(c or "")
    return {p: norm("\n".join(v)) for p, v in out.items()}


def match_side_task(env: str, st: str, goals: dict):
    if (env, st) in goals:
        return (env, st), "exact"
    cands = [k for k in goals if k[0] == env and (k[1].endswith(st) or st.endswith(k[1]) or st in k[1] or k[1] in st)]
    return (cands[0], "fuzzy") if len(cands) == 1 else (None, "unmatched")


def flags_text(t: str, own: set, envg: set, att: list) -> dict:
    return {"own_goal": any(s in t for s in own), "env_goal": any(s in t for s in envg), "attacker_line": any(s in t for s in att),
            **{f"generic:{g}": g in t for g in GENERIC}}


def own_sets(env, st, goals, envgoals):
    gk, how = match_side_task(env, st, goals) if st else (None, "none")
    return (goals.get(gk, set()) if gk else set()), envgoals.get(env, set()), how


@app.function(image=base_img, volumes={"/acts": acts}, cpu=4, memory=8192, timeout=900)
def scan(path: str, raw: dict, keymap: dict) -> dict:
    goals, envgoals, att = build(raw)
    rows, nomsg = [], 0
    with open(f"/acts/{path}") as fh:
        for line in fh:
            r = json.loads(line)
            if not isinstance(r.get("messages"), list):
                nomsg += 1
                continue
            env, st = r.get("environment") or "", r.get("side_task") or ""
            if not st and r.get("key") in keymap:
                env, st = keymap[r["key"]]
            own, envg, how = own_sets(env, st, goals, envgoals)
            P = parts_of(r["messages"])
            rows.append({"arm": r.get("arm"), "key": r.get("key"), "main_task": r.get("main_task"), "env": env, "side_task": st,
                         "source": path.split("/")[0], "file": path, "side_task_match": how,
                         "msg_sha": hashlib.sha256(json.dumps(r["messages"], sort_keys=True).encode()).hexdigest(),
                         "system_sha": hashlib.sha256(P["system"].encode()).hexdigest(),
                         "flags": {p: flags_text(P[p], own, envg, att) for p in PARTS}})
    return {"rows": rows, "no_messages": nomsg}


def exec_prep(code: str) -> dict:
    ns = {"json": json}
    exec(code, ns)
    return ns


def render_one(reader: str, tok, ns, msgs):
    model, rev, _, prep_key, kw = READERS[reader]
    if reader == "qwen3-32b":
        return tok.apply_chat_template(msgs, tokenize=False, **kw)
    if reader == "qwen3.5-27b":
        return tok.apply_chat_template(ns["prepare"](tok, msgs), tokenize=False, **kw)
    return tok.apply_chat_template(ns["prep"](msgs), tokenize=False, **kw)


def _render(reader: str, jobs: list, raw: dict, prep: dict) -> list[dict]:
    from transformers import AutoTokenizer
    model, rev, _, prep_key, kw = READERS[reader]
    tok = AutoTokenizer.from_pretrained(model, revision=rev)
    ns = exec_prep(prep[prep_key]["code"]) if prep_key else {}
    goals, envgoals, att = build(raw)
    want = defaultdict(dict)
    for j in jobs:
        want[j["file"]][j["key"]] = j
    out = []
    for path, keys in want.items():
        with open(f"/acts/{path}") as fh:
            for line in fh:
                r = json.loads(line)
                j = keys.get(r.get("key"))
                if j is None or r.get("arm") != j["arm"]:
                    continue
                own, envg, _ = own_sets(j["env"], j["side_task"], goals, envgoals)
                rec = {"arm": j["arm"], "key": j["key"], "reader": reader}
                try:
                    text = render_one(reader, tok, ns, r["messages"])
                except Exception as e:  # counted per reader; the type only, no text
                    rec["error"] = type(e).__name__
                    out.append(rec)
                    continue
                rec["rendered"] = flags_text(norm(text), own, envg, att)
                if reader == "qwen3-32b":
                    ids = tok(text, add_special_tokens=False)["input_ids"]
                    rec["cut"] = len(ids) > QWEN3_CUT
                    rec["rendered_cut"] = flags_text(norm(tok.decode(ids[:QWEN3_CUT])), own, envg, att)
                out.append(rec)
    return out


@app.function(image=IMAGES["t456"], volumes={"/acts": acts}, cpu=4, memory=8192, timeout=900)
def render_t456(reader, jobs, raw, prep):
    return _render(reader, jobs, raw, prep)


@app.function(image=IMAGES["t519"], volumes={"/acts": acts}, cpu=4, memory=8192, timeout=900)
def render_t519(reader, jobs, raw, prep):
    return _render(reader, jobs, raw, prep)


@app.function(image=IMAGES["t457"], volumes={"/acts": acts}, cpu=4, memory=8192, timeout=900)
def render_t457(reader, jobs, raw, prep):
    return _render(reader, jobs, raw, prep)


@app.function(image=IMAGES["t457llama"], volumes={"/acts": acts, "/llama": llama_cache}, cpu=4, memory=8192, timeout=900)
def render_t457llama(reader, jobs, raw, prep):
    return _render(reader, jobs, raw, prep)


RENDERERS = {"t456": render_t456, "t519": render_t519, "t457": render_t457, "t457llama": render_t457llama}


def summarize(R: list, honest_sys: dict) -> dict:
    a = {"runs": len(R), "side task id match": dict(Counter(r["side_task_match"] for r in R)),
         "system equals the main task's honest prompt (normalized)": sum(r["system_sha"] in honest_sys.get(r["main_task"], ()) for r in R),
         "runs flagged, by part and check": {p: {c: sum(r["flags"][p][c] for r in R) for c in R[0]["flags"][p]} for p in PARTS}}
    a["runs flagged in ANY part"] = {c: sum(any(r["flags"][p][c] for p in PARTS) for r in R) for c in R[0]["flags"]["system"]}
    return a


@app.function(image=base_img, volumes={"/acts": acts, "/work": work}, cpu=2, memory=8192, timeout=3000)
def run_all(raw: dict, prep: dict, git: str) -> str:
    acts.reload()
    folders = sorted(p.parent.name for p in Path("/acts").glob("*/inputs") if any(p.glob("*.jsonl")))
    sealed = [f for f in folders if any(s in f for s in SEALED)]
    folders = [f for f in folders if f not in sealed]
    files = {f: sorted(str(x.relative_to("/acts")) for x in Path(f"/acts/{f}/inputs").glob("*.jsonl")) for f in folders}
    base_files = [x for f in BASE for x in files[f]]
    base = [x for res in scan.map(base_files, kwargs={"raw": raw, "keymap": {}}) for x in res["rows"]]
    keymap = {r["key"]: (r["env"], r["side_task"]) for r in base if r["side_task"]}
    other = [x for f in folders if f not in BASE for x in files[f]]
    res_other = list(scan.map(other, kwargs={"raw": raw, "keymap": keymap})) if other else []
    allrows = base + [x for res in res_other for x in res["rows"]]
    nomsg = Counter()
    for path, res in zip(other, res_other):
        nomsg[path.split("/")[0]] += res["no_messages"]
    # base: unique runs, duplicates compared
    by = defaultdict(list)
    for r in base:
        by[(r["arm"], r["key"])].append(r)
    if any(len({r["msg_sha"] for r in v}) > 1 for v in by.values()):
        raise SystemExit("duplicate base rows differ")
    order = {run: i for i, run in enumerate(BASE)}
    uniq = [min(v, key=lambda r: order[r["source"]]) for v in by.values()]
    honest_sys = defaultdict(set)
    for r in uniq:
        if r["arm"] == "real-honest":
            honest_sys[r["main_task"]].add(r["system_sha"])
    out = {"POST HOC": True, "plan": "PLAN.md Amendment 1", "git": git, "sealed folders skipped (not read)": sealed,
           "prep sources": {k: {kk: v[kk] for kk in ("path", "ref", "commit", "file_sha256", "functions")} for k, v in prep.items()},
           "readers": {k: {"model": v[0], "revision": v[1], "template kwargs": v[4]} for k, v in READERS.items()},
           "(1) source scan by folder": {}}
    for f in folders:
        R = [r for r in allrows if r["source"] == f]
        entry = {"files": len(files[f]), "rows": len(R), "rows without messages": nomsg.get(f, 0), "arms": {}}
        for arm in sorted({r["arm"] for r in R if r["arm"]}):
            entry["arms"][arm] = summarize([r for r in R if r["arm"] == arm], honest_sys)
        out["(1) source scan by folder"][f] = entry
    # (2) rendered sample: 200 per arm (seeded) plus every run flagged on any check in the source scan
    rng = random.Random(SEED)
    jobs = []
    for arm in sorted({r["arm"] for r in uniq}):
        R = sorted([r for r in uniq if r["arm"] == arm], key=lambda r: r["key"])
        flagged = [r for r in R if any(any(r["flags"][p].values()) for p in PARTS)]
        pick = {r["key"]: r for r in rng.sample(R, min(SAMPLE, len(R)))} | {r["key"]: r for r in flagged}
        jobs += [{"arm": arm, "key": r["key"], "file": r["file"], "env": r["env"], "side_task": r["side_task"],
                  "src_any": {c: any(r["flags"][p][c] for p in PARTS) for c in r["flags"]["system"]}} for r in pick.values()]
    src_any = {(j["arm"], j["key"]): j["src_any"] for j in jobs}
    chunks = [jobs[i:i + 120] for i in range(0, len(jobs), 120)]
    calls = [(reader, RENDERERS[READERS[reader][2]], ch) for reader in READERS for ch in chunks]
    handles = [(reader, fn.spawn(reader, ch, raw, prep)) for reader, fn, ch in calls]
    rend = defaultdict(list)
    for reader, h in handles:
        rend[reader] += h.get()
    out["(2) rendered per reader"] = {}
    for reader, recs in rend.items():
        entry = {"rendered": sum("rendered" in x for x in recs), "render errors": dict(Counter(x["error"] for x in recs if "error" in x)), "arms": {}}
        for arm in sorted({x["arm"] for x in recs}):
            X = [x for x in recs if x["arm"] == arm and "rendered" in x]
            e = {"runs rendered": len(X), "checks": {}}
            for c in X[0]["rendered"] if X else []:
                s = [src_any[(x["arm"], x["key"])][c] for x in X]
                rr = [x["rendered"][c] for x in X]
                e["checks"][c] = {"rendered": sum(rr), "source": sum(s), "rendered but not source": sum(a and not b for a, b in zip(rr, s)),
                                  "source but not rendered": sum(b and not a for a, b in zip(rr, s))}
                if reader == "qwen3-32b":
                    rc = [x["rendered_cut"][c] for x in X]
                    e["checks"][c]["within the 32,768-token cut"] = sum(rc)
            if reader == "qwen3-32b":
                e["runs past the cut"] = sum(x["cut"] for x in X)
            entry["arms"][arm] = e
        out["(2) rendered per reader"][reader] = entry
    out["(2) sample"] = {"per arm": SAMPLE, "seed": SEED, "plus every run flagged in the source scan": True, "runs": len(jobs)}
    # (3) keys-only own-words file (as v3)
    keys = {"POST HOC": True, "git": git, "rule": "flagged if any of " + ", ".join(OWN_WORDS) + " appears in assistant_text or "
            "tool_call_args (normalized, lowercased) of #36's / v4's staged inputs; keys only, no text", "per arm": {}}
    for arm in sorted({r["arm"] for r in uniq}):
        R = [r for r in uniq if r["arm"] == arm]
        per = {w.split(":")[1]: sorted(r["key"] for r in R if r["flags"]["assistant_text"][w] or r["flags"]["tool_call_args"][w]) for w in OWN_WORDS}
        union = sorted(set().union(*map(set, per.values())))
        keys["per arm"][arm] = {"runs": len(R), "flagged keys by word": per, "union": union, "counts": {**{w: len(v) for w, v in per.items()}, "union": len(union)}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    name = f"probe_sees_v4_POSTHOC_{stamp}.json"
    Path("/work/results").mkdir(parents=True, exist_ok=True)
    Path(f"/work/results/own_words_keys_v4_POSTHOC_{stamp}.json").write_text(json.dumps(keys, indent=1) + "\n")
    Path(f"/work/results/{name}").write_text(json.dumps(out, indent=1) + "\n")
    work.commit()
    return name


@app.local_entrypoint()
def main(assets: str, prep: str, git: str):
    print(run_all.remote(read_assets(Path(assets)), json.loads(Path(prep).read_text()), git))
