"""Halves number-shuffle and whitespace arms: staging-only counts (RULE.md here @9709d984 + amendment 1; PRICE.md's
"cheapest consistent next step"; 0d's go under Aswin's under-$5 rule). One Modal CPU container; no reads, no text leaves
it, only counts. For the halves run set (staged fixed-sim, real e1, real e2 and plant-2 runs of all 10 environments, plus
B-dev's 100 dev planted copies), each evened as the mask split did, and each arm's text:
  V   value resample: each eligible digit run replaced by a same-length one from an external frozen pool (digit runs of
      the real-honest runs' tool outputs, v4 staged inputs), drawn by sha256(cell | message index | occurrence index);
  P   permutation: the eligible digit runs of each length permuted within the run (seed sha256(cell));
  W1  TRAILING_SPACE.sub("", t) alone; W2 BLANK_RUNS.sub("\\n\\n", t) alone; W3 t.rstrip() alone (scrub.py line 93's parts).
Eligible digit runs (V, P): `\\d+` outside level-2 UUID, IP and hex-ID matches and outside line-number gutters (`^ *\\d+\\|`,
`^ *\\d+(\\t|$)`); integer JSON arguments get the arm, floats are unchanged. Counted per arm and world: tool outputs changed,
content tokens changed (approx., difflib on Qwen3-32B tokens of each changed output), plant-2 pairs still differing
inside the 32,768-token cut per half (the 135 / 90 staging stop), runs equal to the evened text (reusable halves rows).

  PYTHONPATH=<asrun>/src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal deploy \\
      experiments/2026-10-12_claimB_halves_number_shuffle/count_staging_modal.py
  PYTHONPATH=<asrun>/src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy python \\
      experiments/2026-10-12_claimB_halves_number_shuffle/count_staging_modal.py
"""

import hashlib
import json
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import modal

HERE = Path(__file__).parent
H3_DIR = HERE.parent / "2026-10-11_claimB_h3_fix_to_plateau"
sys.path[:0] = [str(H3_DIR), str(HERE.parent / "2026-10-11_claimB_slips_evened_probe")] if modal.is_local() else ["/root"]
import run_h3_small_modal as R  # noqa: E402  (H3's runner: B-dev's code, staged inputs, the evening)

B, M = R.B, R.M
SIM, PL2 = M.SIM, "sim-attack-fixed-plant2"
ARMS = (M.SIM, M.REAL, "real-attack-e2", PL2)
DONOR = "2026-10-06_la_segment-end-v4_qwen3-32b_v1"  # v4 staged inputs; arm real-honest donates digit runs (arm V)
STOP = {"test": 135, "dev": 90}
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

app = modal.App("probe-shift_aswin-surya_sub-claimb-numshuf-count_kill-2026-11-12")
image = R.image.add_local_file(H3_DIR / "run_h3_small_modal.py", "/root/run_h3_small_modal.py")


def eligible(text):
    """Spans of digit runs outside UUID / IP / hex-ID matches and outside line-number gutters."""
    from probe_shift.scrub import ID_PATTERNS

    out = []
    for p, _ in ID_PATTERNS[:3]:
        out += [m.span() for m in p.finditer(text)]
    out += [m.span() for m in re.finditer(r"^ *\d+(?:\||\t|$)", text, re.M)]
    return [m.span() for m in re.finditer(r"\d+", text) if not any(a <= m.start() and m.end() <= b for a, b in out)]


def h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def apply_fields(msgs, text_fn, int_fn):
    """Apply text_fn(text, field_key) to tool outputs, agent text and string arguments, int_fn(int, field_key) to integer
    arguments; field_key identifies the position (message index, field)."""
    def args(v, key):
        if isinstance(v, str):
            return text_fn(v, key)
        if isinstance(v, bool) or v is None or isinstance(v, float):
            return v
        if isinstance(v, int):
            return int_fn(v, key)
        if isinstance(v, list):
            return [args(x, f"{key}.{i}") for i, x in enumerate(v)]
        if isinstance(v, dict):
            return {k: args(x, f"{key}.{k}") for k, x in v.items()}
        return v
    out = []
    for i, m in enumerate(msgs):
        if m["role"] == "tool":
            m = m | {"content": text_fn(m["content"], f"{i}")}
        elif m["role"] == "assistant":
            new = m | {"content": text_fn(m["content"], f"{i}.c") if m["content"] else m["content"]}
            if m.get("tool_calls"):  # absent or empty tool_calls stay exactly as they were
                new["tool_calls"] = [c | {"function": c["function"] | {"arguments": args(c["function"]["arguments"], f"{i}.a{j}")}}
                                     for j, c in enumerate(m["tool_calls"])]
            m = new
        out.append(m)
    return out


def arm_v(msgs, cell, pool):
    def text_fn(t, key):
        spans, parts, last = eligible(t), [], 0
        for k, (a, b) in enumerate(spans):
            cands = pool.get(b - a)
            parts += [t[last:a], cands[h(cell, key, k) % len(cands)] if cands else t[a:b]]
            last = b
        return "".join(parts) + t[last:]

    def int_fn(v, key):  # the integer's digits (sign kept), same length; a leading zero only where the length is 1
        s = str(abs(v))
        cands = [c for c in pool.get(len(s), []) if len(c) == 1 or not c.startswith("0")]
        return (-1 if v < 0 else 1) * int(cands[h(cell, key, 0) % len(cands)]) if cands else v
    return apply_fields(msgs, text_fn, int_fn)


def arm_p(msgs, cell):
    seen = []  # pass 1: every eligible value in run order, as (kind, length): text digit runs, and integer arguments' digits
    apply_fields(msgs, lambda t, k: (seen.extend(("text", t[a:b]) for a, b in eligible(t)), t)[1],
                 lambda v, k: (seen.append(("int", str(abs(v)))), v)[1])
    groups = {}  # integers permute only among integers of the same length (no leading zeros), text among text
    for kind, x in seen:
        groups.setdefault((kind, len(x)), []).append(x)
    rng = random.Random(h(cell))
    for g in sorted(groups):
        rng.shuffle(groups[g])
    it = {g: iter(v) for g, v in groups.items()}  # pass 2: reassign in run order

    def text_fn(t, key):
        spans, parts, last = eligible(t), [], 0
        for a, b in spans:
            parts += [t[last:a], next(it[("text", b - a)])]
            last = b
        return "".join(parts) + t[last:]
    return apply_fields(msgs, text_fn, lambda v, k: (-1 if v < 0 else 1) * int(next(it[("int", len(str(abs(v))))])))


def arm_w(op):
    from probe_shift.scrub import BLANK_RUNS, TRAILING_SPACE

    f = {"W1": lambda t: TRAILING_SPACE.sub("", t), "W2": lambda t: BLANK_RUNS.sub("\n\n", t), "W3": lambda t: t.rstrip()}[op]
    return lambda msgs: apply_fields(msgs, lambda t, k: f(t), lambda v, k: v)


def _work(args):
    """(evened, armed) -> tokens of outputs and approx. tokens changed; and the armed text's cut-ids sha256."""
    from difflib import SequenceMatcher

    from transformers import AutoTokenizer

    from probe_shift.models import QWEN3_32B
    from probe_shift.reader import segment_end_positions

    global _TOK
    if "_TOK" not in globals():
        _TOK = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    e, t, need_cut = args
    tot = chg = 0
    for x, y in zip(e, t):
        if x["role"] != "tool":
            continue
        a = _TOK(x["content"], add_special_tokens=False)["input_ids"]
        tot += len(a)
        if x["content"] != y["content"]:
            b = _TOK(y["content"], add_special_tokens=False)["input_ids"]
            chg += len(a) - sum(m.size for m in SequenceMatcher(None, a, b).get_matching_blocks())
    cut = hashlib.sha256(json.dumps(segment_end_positions(_TOK, t, M.MAX_TOKENS)["ids"]).encode()).hexdigest() if need_cut else None
    return tot, chg, cut


@app.function(image=image, cpu=16, memory=65536, volumes={"/acts": B.P.vol, "/hf": B.P.hf}, timeout=60 * 60, retries=0)
def count() -> dict:
    from multiprocessing import Pool

    R.check_b()
    runs = R.staged()
    bd = json.loads((Path("/acts") / R.DEV_READS / "cells.json").read_text())
    if (bd["script_sha256"], bd["deps_sha256"]) != (B.SCRIPT_SHA, B.DEPS_SHA):
        raise ValueError("B-dev's cells.json was made by other code")
    test = set(R.TEST)
    jobs = []
    for (c, a) in sorted(runs):
        if a in ARMS:
            e, _ = R.evened_version(runs, c, a)
            jobs.append((c, a, e, runs[(c, a)]["environment"], runs[(c, a)]["key"]))
    for c in bd["planted"]:
        e, _, _ = B.evened(runs[(c, SIM)]["messages"])
        p, n = B.planted(e)
        if not n:
            raise ValueError(f"{c}: no R2 undo")
        jobs.append((c, PL2, p, runs[(c, SIM)]["environment"], runs[(c, SIM)]["key"] + "#plant2-dev"))
    if len(jobs) != 3067:
        raise ValueError(f"{len(jobs)} jobs, expected 3,067")
    halves_keys = {k for *_, k in jobs}
    pool = {}  # arm V's donor pool: digit runs of the real-honest runs' tool outputs (none is a halves run)
    donors = 0
    for f in sorted((Path("/acts") / DONOR / "inputs").glob("shard_*.jsonl")):
        for line in f.open():
            if '"real-honest"' not in line:
                continue
            it = json.loads(line)
            if it.get("arm") != "real-honest":
                continue
            if it.get("key") in halves_keys:
                raise ValueError("a donor run is a halves run")
            donors += 1
            for m in it["messages"]:
                if m["role"] == "tool" and isinstance(m.get("content"), str):
                    for a, b in eligible(m["content"]):
                        pool.setdefault(b - a, []).append(m["content"][a:b])
    pool = {L: sorted(v) for L, v in pool.items()}
    out = {"jobs": len(jobs), "donor_runs": donors, "donor_pool_sha256": hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest(),
           "donor_pool_sizes": {str(L): len(v) for L, v in sorted(pool.items())}, "script_sha256": SCRIPT_SHA}
    fns = {"V": lambda e, c: arm_v(e, c, pool), "P": lambda e, c: arm_p(e, c)} | {w: (lambda op: lambda e, c: arm_w(op)(e))(w) for w in ("W1", "W2", "W3")}
    for arm, fn in fns.items():
        texts = [(c, a, e, fn(e, c), env) for c, a, e, env, _ in jobs]
        pair_cells = {c for c, a, *_ in texts if a == PL2}
        with Pool(16) as pool_:
            res = pool_.map(_work, [(e, t, (a == PL2 or (a == SIM and c in pair_cells))) for c, a, e, t, _ in texts], chunksize=8)
        by, cut = {}, {}
        for (c, a, e, t, env), (tot, chg, cs) in zip(texts, res):
            world = "real" if a in (M.REAL, "real-attack-e2") else "sim"
            g = by.setdefault(world, {"runs": 0, "runs_changed": 0, "runs_equal_to_evened": 0, "outputs": 0, "outputs_changed": 0,
                                      "content_tokens": 0, "content_tokens_changed_approx": 0})
            tools = [(x["content"], y["content"]) for x, y in zip(e, t) if x["role"] == "tool"]
            g["runs"] += 1
            g["runs_changed"] += t != e
            g["runs_equal_to_evened"] += t == e
            g["outputs"] += len(tools)
            g["outputs_changed"] += sum(x != y for x, y in tools)
            g["content_tokens"] += tot
            g["content_tokens_changed_approx"] += chg
            if cs:
                cut[(c, a)] = (cs, env)
        plants = {}
        for c in sorted(pair_cells):
            half = "test" if cut[(c, PL2)][1] in test else "dev"
            ph = plants.setdefault(half, {"pairs": 0, "differ_inside_cut": 0})
            ph["pairs"] += 1
            ph["differ_inside_cut"] += cut[(c, PL2)][0] != cut[(c, SIM)][0]
        out[arm] = {"by_world": by, "plant_pairs": plants,
                    "staging_stop_passes": all(plants[x]["differ_inside_cut"] >= STOP[x] for x in STOP)}
    return out


@app.function(image=image, cpu=1, memory=1024, timeout=120, retries=0)
def version() -> str:
    return SCRIPT_SHA


def main():
    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    R.check_b()
    if modal.Function.from_name(app.name, "version").remote() != SCRIPT_SHA:
        raise SystemExit("the deployed app runs other code: `modal deploy` this revision first")
    res = modal.Function.from_name(app.name, "count").remote()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / f"staging_counts_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
