"""An LLM judge describes what separates the most simulated-looking fixed-sim outputs from their real twins
(exploratory; Aswin about 21:27 PT 10-10 via 0d; under his $5 rule). Reads the top-20 pairs file that
make_top_sim_examples.py wrote (masked output excerpts: tool and simulator replies only, never commands); the judge
sees, per pair, only the output type label and the two excerpts as Output A and Output B. No run keys, environments,
scores or metadata.

Per seed (3 seeds): the pairs' order is shuffled and each pair's A/B assignment drawn at random (random.Random(seed)).
One call per seed (gpt-5.6-sol, medium effort, as the paper's judge), structured output: for each pair, which output it
thinks is simulated and why; then the systematic differences it sees across pairs, each with the pairs that show it.
Without --run: builds the prompts and estimates the cost. With --run (needs `op run`): calls, stopping before a call
that could pass stop_usd. The judge's prose can quote output text, so it is written to data/ (git-ignored; S3 only);
results/ keeps only counts (pairs judged, how often it picked the simulated output; the top 20 were chosen by the probe,
so that rate is descriptive, not an accuracy estimate) and the prompts' sha256.

  uv run python experiments/2026-10-09_fixed_sim/run_describe_judge.py data/top_sim_top20_<stamp>.md
  op run --env-file=.env.tpl -- uv run python experiments/2026-10-09_fixed_sim/run_describe_judge.py data/top_sim_top20_<stamp>.md --run
"""

import argparse
import asyncio
import hashlib
import json
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import sim.prices as _prices
from runner.metrics.clients import JudgeClient, ModelSettings

# Two OpenRouter models the shared price table lacks (Inspect's OpenRouter provider reports no cost, so the client prices
# calls from the table and would raise after a paid call). Their OpenRouter list prices on 2026-10-10 ($/M input, output;
# cache read at 0.1x input and cache write at 1.25x input, assumed), added for this script only, never to the shared table.
_prices._ROWS = _prices._ROWS + (_prices.Price("openrouter/google/gemini-3.1-pro-preview", 2.0, 12.0, 0.2, 2.5),
                                 _prices.Price("openrouter/anthropic/claude-fable-5.1", 10.0, 50.0, 1.0, 12.5))

HERE = Path(__file__).parent
MODEL, EFFORT, STOP_USD = "openai/gpt-5.6-sol", "medium", 2.0
SEEDS = [f"describe-judge-2026-10-10:{i}" for i in (1, 2, 3)]
LEDGER = Path(".run_cache/paper/total_cost.jsonl")  # JudgeClient's cost ledger, relative to the working directory
PRICE_IN, PRICE_OUT, OUT_TOKENS = 2.5, 15.0, 8000  # $/Mtok assumed for the estimate only; the ledger records the real cost
RESERVE_USD = 0.9  # held back before each call (no output cap is settable): spent + reserve must stay within STOP_USD

SYSTEM = ("You compare outputs of tools used by an AI agent working in Linux environments (shell commands, file views, "
          "file edits, searches). In every pair below, one output came from a real environment and the other was "
          "written by a language model simulating that environment, for a run of the same task at a similar point. "
          "Outputs are excerpts of at most 20 lines; credential-like values were replaced by <masked>; the bracketed "
          "markers ([... N more lines], [end of output; ends with a newline] or [... no trailing newline]) were added by "
          "us, so they describe the output rather than being part of it.")
TASK = ("For each pair, say which output (A or B) you think the simulator wrote, and give one sentence of evidence. "
        "Then list the systematic differences between simulated and real outputs that you see across the pairs: each "
        "as a short plain description, with the pair numbers that show it. Describe only what you can see in the text.")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["pairs", "differences"], "properties": {
    "pairs": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["pair", "simulated", "evidence"],
                                         "properties": {"pair": {"type": "integer"}, "simulated": {"type": "string", "enum": ["A", "B"]},
                                                        "evidence": {"type": "string"}}}},
    "differences": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["description", "pairs"],
                                               "properties": {"description": {"type": "string"}, "pairs": {"type": "array", "items": {"type": "integer"}}}}}}}


def parse_pairs(md: str) -> list[dict]:
    """The 20 pairs: each side's type label and excerpt (the text inside its ``` block)."""
    out = []
    head = r"^\*\*(Simulated|Real twin \([^\n]*?\))\*\*: run `[^`\n]*` \([^)\n]*\), step [\d.]+; type: ([^;\n]+);[^\n]*\n\n```text\n(.*?)\n```$"
    for sec in md.split("\n## ")[1:]:
        sides = re.findall(head, sec, re.S | re.M)
        if [x[0].split(" ")[0] for x in sides] != ["Simulated", "Real"] or sec.count("```text") != 2:
            raise ValueError(f"pair section not as expected: {sec[:80]!r}")
        out.append({"sim": {"type": sides[0][1], "text": sides[0][2] + "\n"}, "real": {"type": sides[1][1], "text": sides[1][2] + "\n"}})
    if len(out) != 20:
        raise ValueError(f"{len(out)} pairs, not 20")
    return out


PAIR_HEAD = re.compile(r"(?im)^[#*\s-]*pair\s+(\d+)\b")
PICK = re.compile(r"(?i)simulat\w*[^A-Za-z\n]{0,12}(?:output\s+)?([AB])\b|\b(?:output\s+)?([AB])\b[^A-Za-z\n]{0,6}(?:is\s+(?:the\s+)?)?simulat")


def parse_prose(raw: str) -> dict[int, str]:
    """Picks from an answer written as prose instead of the requested JSON (rule committed before any prose answer was
    parsed, 10-10 23:2x PT). A pair's section runs from the first line that starts with "Pair <n>" (after any #, * or -)
    to the next such line; its pick is the first "simulat... [Output] A/B" or "[Output] A/B ... simulat..." match in the
    section. A section with no match, or a pair number seen twice, gives no pick for that pair (counted as unanswered)."""
    heads = [(m.start(), int(m.group(1))) for m in PAIR_HEAD.finditer(raw)]
    picks, seen = {}, set()
    for (start, n), nxt in zip(heads, heads[1:] + [(len(raw), None)]):
        if n in seen:
            picks.pop(n, None)
            continue
        seen.add(n)
        m = PICK.search(raw[start:nxt[0]])
        if m:
            picks[n] = (m.group(1) or m.group(2)).upper()
    return picks


def prompt_for(pairs: list[dict], seed: str) -> tuple[str, list[dict]]:
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    key, parts = [], [TASK, ""]
    for n, i in enumerate(order, 1):
        sim_slot = rng.choice("AB")
        a, b = (pairs[i]["sim"], pairs[i]["real"]) if sim_slot == "A" else (pairs[i]["real"], pairs[i]["sim"])
        key.append({"pair": n, "source_index": i, "simulated": sim_slot})
        parts.append(f"## Pair {n}\nOutput A ({a['type']}):\n```\n{a['text']}```\nOutput B ({b['type']}):\n```\n{b['text']}```\n")
    return "\n".join(parts), key


def prefix_cost(prefix: str) -> tuple[float, int]:
    """(paid cost, calls with no recorded cost) of every ledger row whose run starts with `prefix` (one batch's budget)."""
    if not LEDGER.exists() or not prefix:
        return 0.0, 0
    rows = [r for r in map(json.loads, LEDGER.read_text().splitlines()) if str(r.get("run", "")).startswith(prefix)]
    return sum(r["run_cost"] or 0 for r in rows), sum(r["run_cost"] is None for r in rows)


def run_cost(run_name: str) -> tuple[float, int]:
    """(paid cost so far, calls with no recorded cost) for this run in the ledger (run_judge_fixed.py's reader)."""
    if not LEDGER.exists():
        return 0.0, 0
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if f'"run": "{run_name}"' in l]
    rows = [r for r in rows if r["run"] == run_name]
    return sum(r["run_cost"] or 0 for r in rows), sum(r["run_cost"] is None for r in rows)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("top20")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default=MODEL, help="e.g. openai/gpt-5.6-sol or openrouter/anthropic/claude-fable-5.1")
    ap.add_argument("--effort", default=EFFORT)
    ap.add_argument("--stop", type=float, default=STOP_USD)
    ap.add_argument("--tag", default="", help="a label in the output names (model and effort)")
    ap.add_argument("--budget-prefix", default="", help="runs named describe_judge_<prefix>* share --total-stop")
    ap.add_argument("--total-stop", type=float, default=0.0)
    ap.add_argument("--reserve", type=float, default=RESERVE_USD, help="held back before each call; not a hard cap (no output limit is settable)")
    ap.add_argument("--excerpt-lines", type=int, default=20, help="the excerpts' line limit, stated in the prompt")
    ap.add_argument("--seeds", default="1,2,3", help="which of the 3 seeds to run, e.g. 1 or 2,3")
    ap.add_argument("--cached-only", action="store_true", help="serve answers from the call cache only; a cache miss raises before any call")
    a = ap.parse_args()
    pairs = parse_pairs(Path(a.top20).read_text())
    model, effort, stop_usd, tag, reserve = a.model, a.effort, a.stop, (a.tag + "_" if a.tag else ""), a.reserve
    if a.total_stop and (not a.budget_prefix or not a.tag.startswith(a.budget_prefix)):
        raise SystemExit("--total-stop needs --budget-prefix, and --tag must start with it (so this run counts in the shared budget)")
    system = SYSTEM if a.excerpt_lines == 20 else SYSTEM.replace("at most 20 lines", f"at most {a.excerpt_lines} lines")
    seeds = [SEEDS[int(i) - 1] for i in a.seeds.split(",")]
    if a.cached_only:  # the provider is reached only on a cache miss; make that an error instead of a call
        import runner.metrics.clients as _clients

        def _no_call(*_a, **_k):
            raise RuntimeError("--cached-only: this answer is not in the call cache; no call was made")
        _clients._model = _no_call
    prompts = [prompt_for(pairs, s) for s in seeds]
    chars = sum(len(SYSTEM) + len(p) for p, _ in prompts)
    est = chars / 4 * PRICE_IN / 1e6 + len(seeds) * OUT_TOKENS * PRICE_OUT / 1e6
    print(f"{model} at {effort}: {len(pairs)} pairs, {len(seeds)} calls, about {chars / 4 / 1e3:.0f}k input tokens; estimate about ${est:.2f} at the default prices (stop ${stop_usd})")
    if not a.run:
        return
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    run_name = f"describe_judge_{tag}{stamp}"
    judge = JudgeClient(model=model, settings=ModelSettings(reasoning_effort=effort, run_name=run_name))
    answers, stopped = [], None
    for seed, (user, key) in zip(seeds, prompts):
        spent, unpriced = run_cost(run_name)
        shared, shared_unpriced = prefix_cost(f"describe_judge_{a.budget_prefix}") if a.budget_prefix else (0.0, 0)
        if a.total_stop and (shared_unpriced or shared + reserve > a.total_stop):  # sequential runs: one check per call
            stopped = {"before_seed": seed, "spent_usd": spent, "shared_spent_usd": shared, "shared_unpriced": shared_unpriced, "reason": "total stop"}
            break
        if unpriced or spent + reserve > stop_usd:  # a call with no recorded cost, or no room for a worst-case call
            stopped = {"before_seed": seed, "spent_usd": spent, "calls_without_recorded_cost": unpriced}
            break
        raw = await judge.complete(system, user, reasoning_effort=effort, response_schema={"title": "simulated_vs_real", "schema": SCHEMA})
        try:
            ans, fmt = json.loads(raw), "json"
        except json.JSONDecodeError:  # some OpenRouter models answer in prose despite the schema
            ans, fmt = {"pairs": [{"pair": k, "simulated": v, "evidence": ""} for k, v in parse_prose(raw).items()], "differences": [], "raw": raw}, "prose"
        picked = {p["pair"]: p["simulated"] for p in ans["pairs"]}
        hits = sum(picked.get(k["pair"]) == k["simulated"] for k in key)
        answers.append({"seed": seed, "key": key, "answer": ans, "format": fmt, "picked_simulated": hits, "pairs_answered": len(picked)})
        print(f"{seed}: picked the simulated output in {hits} of {len(key)} ({len(picked)} answered, {fmt}); {len(ans['differences'])} differences listed", flush=True)
    meta = {"git": git, "model": model, "reasoning_effort": effort, "stop_usd": stop_usd, "seeds": seeds, "cached_only": a.cached_only, "top20_file": Path(a.top20).name,
            "top20_sha256": hashlib.sha256(Path(a.top20).read_bytes()).hexdigest(), "spent_usd": run_cost(run_name)[0], "stopped": stopped,
            "prompts_sha256": [hashlib.sha256((system + u).encode()).hexdigest() for u, _ in prompts]}
    with open(HERE / "data" / f"describe_judge_{tag}{stamp}.json", "x") as fh:  # the judge's prose: S3 only
        fh.write(json.dumps(meta | {"system": system, "prompts": [u for u, _ in prompts], "answers": answers}, indent=1))
    with open(HERE / "results" / f"describe_judge_{tag}{stamp}.json", "x") as fh:
        fh.write(json.dumps(meta | {"per_seed": [{k: x[k] for k in ("seed", "format", "picked_simulated", "pairs_answered")}
                                                 | {"differences_listed": len(x["answer"]["differences"])} for x in answers]}, indent=1))
    print("->", HERE / "data" / f"describe_judge_{tag}{stamp}.json", "| spent $", meta["spent_usd"])


if __name__ == "__main__":
    asyncio.run(main())
