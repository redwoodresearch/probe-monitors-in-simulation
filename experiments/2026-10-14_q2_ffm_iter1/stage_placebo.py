"""RULE.md Additions 8-9, Test P, POST HOC: the donor-swap placebo, simulated attack arm only. The insert replies F11 changed
(full + F9 text) are swapped one-to-one among themselves within each (environment, fold) group, minimizing the total
character-length difference, never from the same run (scipy linear_sum_assignment); infeasible groups stay unchanged.
Counts, hashes and token counts only; never message text (attack runs). Staged inputs are never overwritten; nothing is
staged unless every gate passes. Staged: changed runs (arm sim-attack-fx-fullF9P) plus one reproduction run per extractor
shard copied byte-identically from d9's mean-pool dumps' inputs (stage_f12.py's layout and dealing).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_q2_ffm_iter1/stage_placebo.py
"""
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
if modal.is_local():
    sys.path.insert(0, str(REPO / "src"))
APP = "probe-shift_aswin-surya_sub-q2-ffm1_stageP_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
DST, DST_ARM = "2026-10-14_q2_ffmP_inputs_v1", "sim-attack-fx-fullF9P"
RUNS36 = REPO / "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
BASE = ("2026-10-13_q2_fxavg_inputs_v1/inputs/shard_2.jsonl", "009466e7b8eb0f3f265a840157c800826ff459cb387f9824f05fabdd6f09fe55", "sim-attack-fx-fullF9", 939)
F11 = ("2026-10-13_q2_f11q_inputs_v1/inputs/shard_2.jsonl", "7a97837283ab161820ff78fbb87d645fc8753ab19a13aa694f77b0a78a7bdd6b", "sim-attack-fx-fullF9F11", 427)
F11_TARGETS, F11_TOKENS_DELTA = (1075, 427), 11341895 - 11276105  # stage_f11_stage_2026-10-13_183525.json (aswin/q2-f10-tabs)
D9 = {"repro-sim-attack": ("2026-10-12_la_mean-pool36_qwen3-32b_l7_v1", "sim-attack"),
      "repro-sim-honest": ("2026-10-13_q3_mean-pool36-honestsim_qwen3-32b_l7_v1", "sim-honest")}
PRICE = {"qwen3 $/1M forward tokens (F11 billed: $17.28 for 18.6M)": 17.28 / 18.6}
PER_SHARD, MAX_TOKENS, SEED = 1.0e6, 32768, 20261014
REVIEW = "Review the changes and make sure they are as expected"
DIFF = "Here's a diff showing the changes made:"
TELLS = {"tab": lambda t: "\t" in t, "CRLF": lambda t: "\r\n" in t, "cat -n gutter": lambda t: bool(re.search(r"(?m)^ *\d+\t", t)),
         "review note": lambda t: REVIEW in t, "diff marker": lambda t: DIFF in t[:300], "trailing newline": lambda t: t.endswith("\n")}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
hf_cache = modal.Volume.from_name(HF_CACHE).read_only()
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy==2.5.4", "scipy==1.18.1", "transformers==4.56.2", "jinja2==3.1.6")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift")
         .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/rerender.py", "/root/swap/rerender.py")
         .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/str_replace_tells.py", "/root/swap/str_replace_tells.py")
         .add_local_file(REPO / "experiments/2026-10-09_fixed_sim/fixes.py", "/root/fsim/fixes.py"))


class Gate(Exception):
    """A failed check; its message holds counts only."""


def msha(msgs):
    return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()


def tsha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def filename(key: str) -> str:
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def run_job(git: str, groups_of: dict, qwen_rev: str) -> dict:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from transformers import AutoTokenizer

    sys.path[:0] = ["/root/swap", "/root/fsim"]
    import fixes

    from probe_shift.reader import segment_end_positions

    root, c = Path("/acts"), Counter()

    def pinned(spec):
        p, h, arm, n = spec
        raw = (root / p).read_bytes()
        if hashlib.sha256(raw).hexdigest() != h:
            raise Gate(f"{p}: sha256 differs from its stage record")
        rows = {r["key"]: r for r in map(json.loads, raw.decode().splitlines())}
        if len(rows) != n or {r["arm"] for r in rows.values()} != {arm}:
            raise Gate(f"{p}: not {n} runs of {arm}")
        return rows

    base, f11 = pinned(BASE), pinned(F11)
    targets = []  # (key, message index, text)
    for k, r in sorted(f11.items()):
        old = base[k]["messages"]
        for i, (a, b) in enumerate(zip(old, r["messages"])):
            if a != b:
                if a["role"] != "tool" or not isinstance(a.get("content"), str):
                    raise Gate("an F11 change outside a text tool message")
                targets.append((k, i, a["content"]))
    if (len(targets), len({k for k, _, _ in targets})) != F11_TARGETS:
        raise Gate(f"targets {len(targets)} in {len({k for k, _, _ in targets})} runs, not F11's {F11_TARGETS}")
    if any(k not in groups_of for k, _, _ in targets):
        raise Gate("a target run has no environment or fold in #36's runs.json")

    # one-to-one assignment within (environment, fold), minimizing length difference, never from the same run
    by_group = defaultdict(list)
    for t in targets:
        by_group[groups_of[t[0]]].append(t)
    new_text, infeasible = {}, Counter()
    for g, ts in sorted(by_group.items()):
        runs = Counter(k for k, _, _ in ts)
        if len(runs) < 2 or max(runs.values()) > len(ts) / 2:
            infeasible["groups"] += 1
            infeasible["targets"] += len(ts)
            continue
        L = np.array([len(x) for _, _, x in ts], float)
        same = np.array([[a[0] == b[0] for b in ts] for a in ts])
        cost = np.abs(L[:, None] - L[None, :]) + 1e12 * same
        ri, ci = linear_sum_assignment(cost)
        if same[ri, ci].any():
            raise Gate("the assignment used a same-run donor in a feasible group")
        if sorted(ci.tolist()) != list(range(len(ts))):
            raise Gate("a donor used twice")
        for r_, d_ in zip(ri, ci):
            new_text[(ts[r_][0], ts[r_][1])] = ts[d_][2]
        c["swapped groups"] += 1
        c["length |difference| total (characters)"] += int(cost[ri, ci].sum())
    swapped = [(k, i, x) for k, i, x in targets if (k, i) in new_text]
    if Counter(tsha(x) for _, _, x in swapped) != Counter(tsha(new_text[(k, i)]) for k, i, _ in swapped):
        raise Gate("the content-hash multiset changed")
    c["targets"], c["targets swapped"] = len(targets), len(swapped)
    c["targets byte-identical to their donor"] = sum(new_text[(k, i)] == x for k, i, x in swapped)
    c["infeasible groups"], c["targets left unchanged (infeasible groups)"] = infeasible["groups"], infeasible["targets"]

    # new rows, gates, tell checks (counts only)
    staged, record, tok_in = [], [], []
    tell = Counter()
    per_run_before, per_run_after = defaultdict(Counter), defaultdict(Counter)
    for k in sorted({k for k, _, _ in swapped}):
        old = base[k]["messages"]
        new = [dict(m) for m in old]
        for (kk, i), txt in new_text.items():
            if kk == k:
                new[i] = old[i] | {"content": txt}
        changed = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
        if any((k, i) not in new_text or set(old[i]) != set(new[i]) for i in changed):
            raise Gate("a non-target message or a field other than content changed")
        for i in [i for kk, i in new_text if kk == k]:
            for name, f in TELLS.items():
                b_, a_ = f(old[i]["content"]), f(new[i]["content"])
                tell[f"replies with {name}|before"] += b_
                tell[f"replies with {name}|after"] += a_
                per_run_before[k][name] += b_
                per_run_after[k][name] += a_
        c["runs changed"] += bool(changed)
        c["messages changed"] += len(changed)
        staged.append(base[k] | {"arm": DST_ARM, "messages": new})
        record.append({"arm": DST_ARM, "key": k, "changed_messages": changed, "messages_sha256": msha(new), "base_messages_sha256": msha(old)})
        tok_in.append((k, old, new))
    for name in TELLS:
        tell[f"runs with {name} in a target reply|before"] = sum(per_run_before[k][name] > 0 for k in per_run_before)
        tell[f"runs with {name} in a target reply|after"] = sum(per_run_after[k][name] > 0 for k in per_run_after)
        tell[f"runs whose count of target replies with {name} changed"] = sum(per_run_before[k][name] != per_run_after[k][name] for k in per_run_before)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", revision=qwen_rev)
    deltas, kept = [], Counter()
    for k, old, new in tok_in:
        so, sn = segment_end_positions(tok, old, MAX_TOKENS), segment_end_positions(tok, new, MAX_TOKENS)
        deltas.append(len(sn["ids"]) - len(so["ids"]))
        msg_of = {(x["step"], x["action"]): x["msg"] for x in fixes.calls_of(old)}
        tgt = {i for kk, i in new_text if kk == k}
        for tag, s in (("before", so), ("after", sn)):
            kept[f"target outputs with a kept output_end|{tag}"] += sum(1 for p in s["positions"] if p["kind"] == "output_end" and p["kept"]
                                                                        and msg_of.get((p["step"], p["action"])) in tgt)
    q_tok = sum(len(segment_end_positions(tok, new, MAX_TOKENS)["ids"]) for _, _, new in tok_in)
    dose = {"tokens delta total (placebo)": int(sum(deltas)), "tokens delta total (F11, attack)": F11_TOKENS_DELTA,
            "per-run |token delta| mean": float(np.mean(np.abs(deltas))), "per-run |token delta| max": int(np.max(np.abs(deltas))),
            **kept}
    S = min(64, max(8, 2 * math.ceil(q_tok / PER_SHARD / 2)))
    price = {"basis": PRICE, "qwen3 forward tokens": q_tok, "shards": S,
             "$ raw": round(q_tok / 1e6 * PRICE["qwen3 $/1M forward tokens (F11 billed: $17.28 for 18.6M)"], 2)}
    price["$ with 30%"] = round(price["$ raw"] * 1.3, 2)
    out = {"git": git, "counts": dict(c), "tell checks (counts)": dict(tell), "dose": dose, "price": price, "runs staged": len(staged)}
    if (root / DST).exists():
        raise FileExistsError(f"{DST} exists: a staged run is never overwritten")

    def rows(run, arms):
        o = {}
        for p in sorted((root / run / "inputs").glob("shard_*.jsonl")):
            for line in open(p):
                it = json.loads(line)
                if it["arm"] in arms:
                    o[(it["arm"], it["key"])] = it
        return o

    n_rep = {"repro-sim-attack": (S + 1) // 2, "repro-sim-honest": S // 2}
    rng, repro, chosen, n = random.Random(SEED), {}, {}, 0
    for new_arm, (dump, arm) in D9.items():
        cfg = json.loads((root / dump / "config.json").read_text())
        have = {p.stem for p in (root / dump / "meta" / arm).glob("*.json")}
        srcr = {filename(k): it for (a, k), it in rows(cfg["inputs_from"], {arm}).items()}
        pool = sorted(s for s in have if s in srcr)
        pick = rng.sample(pool, n_rep[new_arm])
        chosen[new_arm] = {"dump": dump, "dump arm": arm, "inputs_from": cfg["inputs_from"], "pool": len(pool), "keys": sorted(srcr[s]["key"] for s in pick)}
        for s in pick:
            n += 1
            repro.setdefault(new_arm, []).append(srcr[s] | {"arm": new_arm, "tokens": -n, "source_tokens": srcr[s]["tokens"]})
    allrows = staged + [r for rs in repro.values() for r in rs]
    order = sorted(allrows, key=lambda it: (-it["tokens"], it["arm"], it["key"]))
    shard_of = {(it["arm"], it["key"]): i % S for i, it in enumerate(order)}
    if sorted(shard_of[(r["arm"], r["key"])] for rs in repro.values() for r in rs) != list(range(S)):
        raise Gate("the reproduction runs do not cover every shard exactly once")
    (root / DST / "inputs").mkdir(parents=True)
    shas = {}
    for i, (arm, rs) in enumerate(sorted(({DST_ARM: staged} | repro).items())):
        body = "".join(json.dumps(x) + "\n" for x in sorted(rs, key=lambda x: x["key"])).encode()
        (root / DST / "inputs" / f"shard_{i}.jsonl").write_bytes(body)
        shas[f"inputs/shard_{i}.jsonl"] = {"arm": arm, "runs": len(rs), "sha256": hashlib.sha256(body).hexdigest()}
    rec = json.dumps({"shards": shas, "runs": record, "reproduction draw": chosen, "seed": SEED, "git": git,
                      "planned shard": {f"{a}|{k}": s for (a, k), s in sorted(shard_of.items())}}).encode()
    (root / DST / "stage_record.json").write_bytes(rec)
    acts.commit()
    return out | {"staged": DST, "shard files": shas, "record_sha256": hashlib.sha256(rec).hexdigest(), "gates pass": True}


@app.function(image=image, volumes={"/acts": acts, "/hf": hf_cache}, cpu=16, memory=65536, timeout=3600, retries=0)
def job(git: str, groups_of: dict, qwen_rev: str) -> dict:
    try:
        return run_job(git, groups_of, qwen_rev)
    except Exception as e:  # noqa: BLE001 — type, location and our own count-only messages; never staged text
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        msg = str(e) if isinstance(e, (Gate, FileExistsError)) else ""
        raise RuntimeError(f"stage failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}: {msg[:200]}") from None


@app.local_entrypoint()
def main():
    for f in (Path(__file__), HERE / "RULE.md"):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
                or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    from probe_shift.models import QWEN3_32B
    runs = json.loads(RUNS36.read_text())
    groups_of = {r["key"]: f"{r['environment']}|{r['fold']}" for r in runs if r["arm"] == "sim-attack"}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out = job.remote(git, groups_of, QWEN3_32B["revision"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"stage_placebo_{stamp}.json"
    with open(p, "x") as fh:
        fh.write(json.dumps({"POST HOC": True, "status": "preliminary", "rule": "RULE.md Additions 8-9 (Test P)", "stamp": stamp,
                             "inputs sha256": {"#36 runs.json": hashlib.sha256(RUNS36.read_bytes()).hexdigest()}} | out, indent=1) + "\n")
    print("wrote", p, "price:", out.get("price"), "counts:", out.get("counts"))
