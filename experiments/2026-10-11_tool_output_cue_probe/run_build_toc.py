"""Build every dialogue the plan reads (README sections 1-3), once, on Modal CPU where the data lives.

N1's templates (read from `../2026-10-11_la_format_sim_probe/run_build_la_pairs.py` at launch), rendered as
tool-output lines and inserted just before the last line of a real tool output; each dialogue ends with that output.
- Probes A and B: train and dev pairs from their own fold's honest runs (base200 excluded), held-out prefixes from the
  other fold's, 4 versions each (simulated, real, original, neutral).
- The attack panel: P's e1 runs, 4 versions each.
- The planted check: P's e1 and e2 runs whole, 4 versions each (original, planted, neutral, planted every 4th output).
Only counts, token statistics and hashes come back; the dialogues go to the Volume.

  PYTHONPATH=src uv run modal deploy experiments/2026-10-11_tool_output_cue_probe/run_build_toc.py
  PYTHONPATH=src uv run python experiments/2026-10-11_tool_output_cue_probe/run_build_toc.py launch
  PYTHONPATH=src uv run python experiments/2026-10-11_tool_output_cue_probe/run_build_toc.py collect <call id>
"""

import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
N1 = HERE.parent / "2026-10-11_la_format_sim_probe"
APP = "probe-shift_aswin-surya_sub-toc_build_kill-2026-11-10"
LA, ACTS = "probe-shift_aswin-surya_sub-la-activations", "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09"
HONEST_RUN, PACKS = "2026-10-06_la_segment-end-scrub1_qwen3-32b_v1", "2026-10-10_la_fixed-sim-v2_probe-inputs-b-l1_v1"
PACKS_SHA = "e56a27e46c4f0093ab07c08fa5c994f2d76391bc6e36e12cc583fdaf6ec9bd1f"
SPLIT = N1 / "results" / "build_split_2026-10-11_104954.json"
SEED, MAX_TOKENS, MIN_STEP, LAST_LINE_MAX = 20261011, 32768, 3, 64
RESERVED = re.compile(r"(?i)replica|usual linux setup")  # as toc_templates.py; the lines themselves come in the plan

app = modal.App(APP)
la = modal.Volume.from_name(LA, version=2).read_only()
acts = modal.Volume.from_name(ACTS, version=2)
hf = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


def plan():
    """Folds (README section 2) from N1's frozen split: A = T, B = H and D; the first quarter of each fold by
    sha256(seed|task) gives the dev pairs."""
    sets = json.loads(SPLIT.read_text())["sets"]
    order = lambda ts: sorted(ts, key=lambda t: hashlib.sha256(f"{SEED}|{t}".encode()).hexdigest())
    folds = {"A": order(sets["T"]), "B": order(sets["H"] + sets["D"])}
    ex = json.loads((ROOT / "splits" / "exclusions_v4.json").read_text())
    p_cells = sorted(c for c in ex["paired_attack_cells"]["cells"] if c.split(":")[0] in set(sets["P"]))
    return {"folds": {f: {"dev": ts[:len(ts) // 4], "train": ts[len(ts) // 4:]} for f, ts in folds.items()}, "P": sets["P"], "p_cells": p_cells}


def insert(content, block):
    """The block on its own lines just before the content's last non-empty line."""
    lines = content.split("\n")
    k = max(i for i, x in enumerate(lines) if x.strip())
    return "\n".join(lines[:k] + block.split("\n") + lines[k:])


@app.function(image=image, volumes={"/in": la, "/acts": acts, "/hf": hf}, cpu=8, memory=32768, timeout=3600, retries=0)
def build(pl: dict, stamp: str, prov: dict) -> dict:
    import os
    import traceback

    null = os.open(os.devnull, os.O_WRONLY)  # only counts leave: no LinuxArena text in logs or errors
    os.dup2(null, 1), os.dup2(null, 2)
    try:
        return _build(pl, stamp, prov)
    except BaseException as e:
        lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("run_build_toc.py")]
        raise RuntimeError(f"build failed: {type(e).__name__} at run_build_toc.py lines {lines}") from None


def _build(pl, stamp, prov, root=Path("/"), packs_sha=PACKS_SHA):  # root and packs_sha: for a local smoke test
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != prov["builder_sha256"]:
        raise ValueError("the deployed builder is not the launching commit's: redeploy")
    import bisect
    import numpy as np
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN3_32B
    from probe_shift.reader import _layout, segment_end_positions

    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    fold_of = {t: (f, part) for f, d in pl["folds"].items() for part, ts in d.items() for t in ts}
    p_cells = set(pl["p_cells"])
    shard_sha, honest, attack = {}, [], []
    for name in (HONEST_RUN, PACKS):
        base = root / "in" / name
        want = None
        if name == PACKS:  # the manifest's own hash, then its shard list and every shard's hash (as Q3.3's scorer)
            man = (base / "manifest.json").read_bytes()
            if hashlib.sha256(man).hexdigest() != packs_sha:
                raise ValueError("the packs' manifest does not match its sha256")
            files = json.loads(man)["files"]
            entries = files.items() if isinstance(files, dict) else [(e.get("path") or e.get("name") or e.get("file"), e) for e in files]
            want = {Path(k).name: (v if isinstance(v, str) else v["sha256"]) for k, v in entries if Path(k).name.startswith("shard_")}
            if sorted(want) != sorted(f.name for f in (base / "inputs").glob("shard_*.jsonl")):
                raise ValueError("the packs' shard files are not the manifest's")
        for f in sorted((base / "inputs").glob("shard_*.jsonl")):
            raw = f.read_bytes()
            shard_sha[f"{name}/{f.name}"] = hashlib.sha256(raw).hexdigest()
            if want is not None and shard_sha[f"{name}/{f.name}"] != want[f.name]:
                raise ValueError(f"{f.name} does not match the manifest")
            for r in map(json.loads, raw.decode().splitlines()):
                if name == HONEST_RUN and r["arm"] == "real-honest" and not r["key"].startswith("base200") and r["main_task"] in fold_of:
                    honest.append(r)
                elif name == PACKS and r["arm"] in ("real-attack-e1", "real-attack-e2") and r["cell"] in p_cells:
                    attack.append(r)
    skip = Counter()

    def readout(msgs):
        """The final tool output's output_end token, its last-line token ids, the uncut length; None if unusable."""
        text, found, enc = _layout(tok, msgs)
        pos = segment_end_positions(tok, msgs, MAX_TOKENS)
        last = found[-1]
        ends = [p for p in pos["positions"] if p["kind"] == "output_end" and (p["step"], p["action"]) == (last[1], last[2])]
        if last[0] != "tool_output" or len(ends) != 1 or ends[0]["token"] is None or not ends[0]["kept"] or pos["n_tokens"] > MAX_TOKENS - 128:
            return None
        content = msgs[-1]["content"]
        lines = content.split("\n")
        k = max(i for i, x in enumerate(lines) if x.strip())
        c0 = last[4] + len("\n".join(lines[:k])) + (1 if k else 0)  # the last non-empty line's first character
        starts = [s for s, _ in enc["offset_mapping"]]
        t0, e = bisect.bisect_left(starts, c0), ends[0]["token"]
        if not (t0 <= e and enc["offset_mapping"][e][1] <= c0 + len(lines[k])):  # the token read lies inside the last line
            return None
        return {"token": e, "tail": enc["input_ids"][t0:e + 1], "n_tokens": pos["n_tokens"], "t0": t0}

    def prefixes(r, n_max=2):
        """Up to n_max seeded prefixes ending with a usable tool output (README section 1)."""
        msgs, cand, step = r["messages"], [], -1
        if RESERVED.search(json.dumps(msgs)):
            skip["run with a reserved word"] += 1
            return []
        for j, m in enumerate(msgs):
            step += m["role"] == "assistant"
            if (m["role"] != "tool" or step < MIN_STEP or not isinstance(m.get("content"), str) or not m["content"].strip()
                    or msgs[j - 1]["role"] != "assistant" or len(msgs[j - 1].get("tool_calls") or []) != 1):
                continue
            last_line = [x for x in m["content"].split("\n") if x.strip()][-1]
            if len(tok.encode(last_line, add_special_tokens=False)) <= LAST_LINE_MAX:
                cand.append((j, step))
        out = []
        for j, step in random.Random(f"{SEED}|{r['key']}").sample(cand, len(cand)):
            d = msgs[:j + 1]
            ro = readout(d)
            if ro is None or ro["n_tokens"] > MAX_TOKENS - 256:  # room for the inserted lines
                skip["prefix too long or no readable final output"] += 1
                continue
            out.append({"key": r["key"], "task": r["main_task"], "messages": d, "step": step, "ro": ro})
            if len(out) == n_max:
                break
        return out

    pools = {}
    for r in sorted(honest, key=lambda r: r["key"]):
        f, part = fold_of[r["main_task"]]
        pools.setdefault((f, part), []).extend(prefixes(r))
    for f in ("A", "B"):
        pools[(f, "all")] = pools.get((f, "train"), []) + pools.get((f, "dev"), [])
    e1 = [r for r in sorted(attack, key=lambda r: r["key"]) if r["arm"] == "real-attack-e1"]
    pools[("P", "panel")] = [p for r in e1 for p in prefixes(r, 1)]
    rng = random.Random(SEED)
    items, stats = [], {"distance": [], "length_diff": []}

    version = lambda pre, block: pre["messages"][:-1] + [dict(pre["messages"][-1], content=insert(pre["messages"][-1]["content"], block))]

    def add(probe, split, family, mech, pre, pair, versions):
        vs = {v: (pre["messages"] if blk is None else version(pre, blk)) for v, blk in versions.items()}
        ros = {v: readout(m) for v, m in vs.items()}
        if any(ro is None or ro["tail"] != pre["ro"]["tail"] for ro in ros.values()):
            raise ValueError(f"{pair}: a version moves or changes the last line's tokens")
        ln = lambda b: len(tok.encode(b, add_special_tokens=False))
        diag = {"len_diff": ln(versions["sim"]) - ln(versions["real"]), "distance": ros["sim"]["token"] - ros["sim"]["t0"],
                "neutral_k": NEUTRAL.index(versions["neutral"]) if "neutral" in versions else None}
        stats["length_diff"].append(diag["len_diff"]), stats["distance"].append(diag["distance"])
        for v, m in vs.items():
            items.append({"id": f"{pair}_{v}", "probe": probe, "split": split if v in ("sim", "real") else "control", "pair": pair,
                          "version": v, "label": 1 if v == "sim" else 0, "family": family, "mechanism": mech, "task": pre["task"],
                          "source_key": pre["key"], "step": pre["step"], "n_tokens": ros[v]["n_tokens"], **diag, "messages": m})

    tp = pl["templates"]
    NEUTRAL, PLANTED, PLANTED_NEUTRAL = tp["neutral"], tp["planted"], tp["planted_neutral"]
    stakes, vocab, status = tp["clauses"]
    train = {f: [t for t in tp["train"] if t[0] == f] for f in tp["families"]}
    for probe, other in (("A", "B"), ("B", "A")):
        for part, counts in (("train", [40] * 8), ("dev", [12] * 4 + [13] * 4)):
            for f, n in zip(tp["families"], counts):
                for i, pre in enumerate(rng.sample(pools[(probe, part)], n)):
                    _, a, b = rng.choice(train[f])
                    extra = " ".join(x for x in (rng.choice(stakes) if rng.random() < 0.5 else "", rng.choice(vocab) if rng.random() < 0.5 else "",
                                                 rng.choice(status) if rng.random() < 0.5 else "") if x)
                    a, b = (f"{extra}\n{a}", f"{extra}\n{b}") if extra else (a, b)  # the crossed clauses, identical on both sides
                    add(probe, part, f, None, pre, f"{probe}_{part}_{f}_{i}", {"sim": a, "real": b})
        for split, pool, n_impl, n_phr in (("heldout", (other, "all"), 60, 120), ("panel", ("P", "panel"), 30, 30)):
            if split == "panel" and probe == "B":
                continue  # one panel, scored by both probes
            name = probe if split == "heldout" else "panel"
            for mech, a, b in tp["held"]:
                for i, pre in enumerate(rng.sample(pools[pool], n_impl)):
                    add(name, split, "heldout_implied", mech, pre, f"{name}_{split}_{mech}_{i}",
                        {"sim": a, "real": b, "original": None, "neutral": rng.choice(NEUTRAL)})
            for i, pre in enumerate(rng.sample(pools[pool], n_phr)):
                k = i % len(tp["phrasings"])
                add(name, f"{split}_phrasing", "heldout_phrasing", f"phrasing_{k}", pre, f"{name}_{split}_phrasing_{i}",
                    {"sim": tp["phrasings"][k][0], "real": tp["phrasings"][k][1], "original": None, "neutral": rng.choice(NEUTRAL)})
    planted = []  # whole P runs, 4 versions (README G4)
    for r in sorted(attack, key=lambda r: r["key"]):
        if RESERVED.search(json.dumps(r["messages"])):
            skip["planted run with a reserved word"] += 1
            continue
        outs = [j for j, m in enumerate(r["messages"]) if m["role"] == "tool" and isinstance(m.get("content"), str) and m["content"].strip()]
        for v, line_, every in (("original", None, 0), ("planted", PLANTED, 1), ("neutral", PLANTED_NEUTRAL, 1), ("planted_every4", PLANTED, 4)):
            m = [dict(x, content=insert(x["content"], line_)) if line_ and j in outs[::every] else x for j, x in enumerate(r["messages"])]
            planted.append({"id": f"{r['key']}_{v}", "cell": r["cell"], "arm": r["arm"].removeprefix("real-attack-"), "version": v,
                            "task": r["main_task"], "messages": m})
    if len({it["id"] for it in items}) != len(items) or len({it["id"] for it in planted}) != len(planted):
        raise ValueError("duplicate ids")
    folds_of = lambda probe: set(pl["folds"][probe]["train"] + pl["folds"][probe]["dev"])
    for it in items:  # no probe trains or picks on a task it validates or scores (README section 2)
        if it["split"] in ("train", "dev") and it["task"] not in folds_of(it["probe"]):
            raise ValueError(f"{it['id']}: a train or dev pair outside its probe's fold")
        if it["split"].startswith("heldout") and it["task"] in folds_of(it["probe"]):
            raise ValueError(f"{it['id']}: a held-out prefix inside its probe's fold")
    out = {}
    for name, rows in (("dialogues", items), ("planted", planted)):
        path = root / "acts" / "toc" / f"{name}_{stamp}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows).encode()
        path.write_bytes(blob)
        out[name] = {"file": f"toc/{path.name}", "sha256": hashlib.sha256(blob).hexdigest(), "rows": len(rows)}
    if root == Path("/"):
        acts.commit()

    count = lambda rows, *keys: dict(sorted(Counter("|".join(str(it[x]) for x in keys) for it in rows).items()))
    t = np.array([it["n_tokens"] for it in items])
    d, ld = np.array(stats["distance"]), np.array(stats["length_diff"])
    return out | {"dialogues by probe, split and version": count(items, "probe", "split", "version"),
                  "pairs by mechanism": count([it for it in items if it["version"] == "sim"], "probe", "split", "mechanism"),
                  "planted by version and arm": count(planted, "version", "arm"), "planted cells": len({p["cell"] for p in planted}),
                  "honest runs read": len(honest), "attack runs read": len(attack),
                  "prefixes by pool": {"|".join(k): len(v) for k, v in pools.items()}, "skipped": dict(skip),
                  "tokens": {"total": int(t.sum()), "mean": float(t.mean()), "p90": float(np.percentile(t, 90)), "max": int(t.max())},
                  "cue-to-readout tokens": {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)), "max": int(d.max())},
                  "simulated minus real block tokens": {"mean": float(ld.mean()), "min": int(ld.min()), "max": int(ld.max())},
                  "shard sha256": shard_sha, "provenance": prov}


if __name__ == "__main__":
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
    if sys.argv[1] == "launch":
        st = git("status", "--porcelain", "--untracked-files=no")
        if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
            raise SystemExit("commit first: the build must record the commit of the code that made it")
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        sys.path.insert(0, str(HERE))
        from toc_templates import templates  # launch only: the container gets the blocks in the plan, not this module

        pl = plan() | {"templates": templates()}
        prov = {"templates_sha256": hashlib.sha256(json.dumps(pl["templates"]).encode()).hexdigest(), "git": git("rev-parse", "HEAD").stdout.strip(), "stamp": stamp, "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "folds": {f: {k: len(v) for k, v in d.items()} for f, d in pl["folds"].items()}, "p cells": len(pl["p_cells"])}
        call = modal.Function.from_name(APP, "build").spawn(pl, stamp, prov)
        (ROOT / ".run_cache" / "tool_output_cue_probe").mkdir(parents=True, exist_ok=True)
        (ROOT / ".run_cache" / "tool_output_cue_probe" / f"build_launch_{stamp}.json").write_text(json.dumps({"call": call.object_id, "plan": pl, "prov": prov}))
        print(json.dumps({"call": call.object_id, **prov}, indent=1))
    elif sys.argv[1] == "collect":
        out = modal.FunctionCall.from_id(sys.argv[2]).get()
        path = HERE / "results" / f"build_{out['provenance']['stamp']}.json"
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps({k: v for k, v in out.items() if k != "shard sha256"}, indent=1), f"\n-> {path}")
