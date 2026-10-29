"""Planted-slip transfer, P4-P6, POST HOC check for review thread #1084 rw2pY: does each planted str_replace output's
appended trailing newline reach the token ids of the FULL transcript, as each reader's extraction rendered it? The
staging record's reach counts (q2_reply_stage.py `survives()`) rendered each output alone, in a minimal one-call
context. Local, $0; counts only leave this script (no transcript text).

  rebuild: rebuilds the staged planted runs (q2_reply_stage.py `_stage`, planted half, unchanged logic) from a local
           copy of #36's staged inputs (2026-10-06_la_segment-end_qwen3-32b_v1/inputs) and checks all 8 shards' sha256
           against the staging record. The rebuilt shards go to <out dir>, outside git.
  reach:   needs every rebuilt shard to match. For each gutter+newline output, renders the whole planted transcript
           through the reader's own extraction rendering, with and without that output's final newline, and compares
           the token ids. Qwen3-32B (P4) and Qwen3.5-27B (P6): probe_shift.reader._layout, as segment_end_positions
           uses it. Olmo-3.1-32B (P5): q2_reply_extract_olmo.prep + apply_chat_template, as positions() uses them.

  HF_HUB_OFFLINE=1 PYTHONPATH=src uv run --no-project --python 3.12 --with transformers==5.19.0 --with jinja2==3.1.6 \\
      python experiments/2026-10-12_plantslip_transfer/check_reach_full.py rebuild <source inputs dir> <out dir>
  ... --with transformers==<the reader's extractor pin> ... check_reach_full.py reach <reader> <out dir>
      (Qwen3-32B: 4.56.2, run_extract_all_layers.py; Olmo-3.1-32B: 4.57.1 + modal==1.6.1, q2_reply_extract_olmo.py;
       Qwen3.5-27B: 5.19.0, q2_reply_extract_qwen35.py)
"""
import bisect
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
EXP = HERE.parent
STAGE_REC = HERE / "results" / "q2_reply_stage_2026-10-12_135318.json"
STAGE_REC_SHA = "f1a68c90a5d26799925bda1c0d574728a547cde6c2ec3b76bfa18c3691c75157"  # as pinned by the P4-P6 scores records
PLANTED_RUN, N_SHARD_FILES = "2026-10-12_q2_plantslip_reply_planted_inputs_v1", 8
ARMS = {"real-attack-e2": "real-attack-e2-planted", "real-attack-e1": "real-attack-e1-planted"}
FOLDERS = [EXP / "2026-10-07_q2_rendering_swap", EXP / "2026-10-09_fixed_sim", HERE]
TOKENIZERS = {"Qwen3-32B": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137"),
              "Olmo-3.1-32B": ("allenai/Olmo-3.1-32B-Instruct", "ac0587e4a7744a551c059d8cd17ba220bc940dae"),
              "Qwen3.5-27B": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654")}
QWEN3_CUT = 32768  # #36's dump and P4's extraction: max_tokens (alllayers_plantslip_q2_reply_planted.yaml)
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
_tok = None


def _paths():
    for f in FOLDERS:
        if str(f) not in sys.path:
            sys.path.insert(0, str(f))


def _load_tok(name):
    global _tok
    from transformers import AutoTokenizer
    m, rv = TOKENIZERS[name]
    _tok = AutoTokenizer.from_pretrained(m, revision=rv)


def _rebuild_one(args):
    it, orig = args
    _paths()
    import fixes
    import rerender
    from plant_both import plant_both
    from probe_shift.reader import _layout
    new, changed = plant_both(it["messages"], fixes, rerender)
    t = len(_layout(_tok, new)[2]["input_ids"])
    return it | {"arm": ARMS[orig], "messages": new, "tokens": t, "plant": "both", "orig_arm": orig, "changed": [list(c) for c in changed]}


def rebuild(src: Path, out: Path, rec: dict):
    want = [(p["orig_arm"], p["key"]) for p in rec["planted"]]
    wset, items = set(want), {}
    for shard in sorted(src.glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            if (it["arm"], it["key"]) in wset:
                items[(it["arm"], it["key"])] = it
    missing = len(set(want) - set(items))
    if missing:
        raise SystemExit(f"{missing} planted runs are not in the local source copy")
    _load_tok("Qwen3-32B")  # preflight, as in reach()
    with get_context("fork").Pool(initializer=_load_tok, initargs=("Qwen3-32B",)) as pool:
        staged = pool.map(_rebuild_one, [(items[w], w[0]) for w in want], chunksize=8)
    out.mkdir(parents=True, exist_ok=False)
    match = {}
    for s in range(N_SHARD_FILES):
        body = "".join(json.dumps(x) + "\n" for x in staged[s::N_SHARD_FILES]).encode()
        (out / f"shard_{s}.jsonl").write_bytes(body)
        match[f"shard_{s}.jsonl"] = sha(body) == rec["inputs_sha256"][f"{PLANTED_RUN}/shard_{s}.jsonl"]
    return {"source dir": str(src), "runs rebuilt": len(staged), "shard sha256 matches the staging record": match,
            "all shards match": all(match.values())}


def _render(name, msgs):
    """(text, token ids, char spans of the tool outputs in message order) through the reader's extraction rendering."""
    if name == "Olmo-3.1-32B":
        from q2_reply_extract_olmo import END, ENV_OPEN, prep
        msgs = prep(msgs)
        text = _tok.apply_chat_template(msgs, tokenize=False)
        tools = [m for m in msgs if m["role"] == "tool"]  # positions()'s guards, verbatim in effect
        if text.count(ENV_OPEN) != len(tools) or any(ENV_OPEN in m["content"] or END in m["content"] for m in tools):
            raise ValueError("a delimiter quoted in content")
        spans, pos = [], 0
        for m in tools:
            a = text.index(ENV_OPEN, pos) + len(ENV_OPEN); b = a + len(m["content"])
            if text[a:b] != m["content"] or not text.startswith(END, b):
                raise ValueError("a tool output not verbatim")
            spans.append((a, b)); pos = b
        enc = _tok(text, return_offsets_mapping=True, add_special_tokens=False)
    else:
        from probe_shift.reader import _layout
        text, found, enc = _layout(_tok, msgs)  # raises if an output is not verbatim (after Qwen3.5's strip)
        spans = [(a, b) for kind, _, _, _, a, b in found if kind == "tool_output"]
    return text, enc["input_ids"], [s for s, _ in enc["offset_mapping"]], spans


def _reach_one(args):
    name, x = args
    _paths()
    msgs = x["messages"]
    tool_ix = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
    text, ids, starts, spans = _render(name, msgs)
    rows = []
    for i, step, action, slip in x["changed"]:
        if slip != "gutter+newline":
            continue
        assert msgs[i]["content"].endswith("\n")
        a, b = spans[tool_ix.index(i)]
        in_text = b > a and text[b - 1] == "\n"  # the output's span in the rendered text ends with the planted newline
        tok_i = bisect.bisect_right(starts, b - 1) - 1 if in_text else None  # the token covering that character
        m2 = [dict(m) for m in msgs]; m2[i]["content"] = msgs[i]["content"][:-1]
        ids2 = _render(name, m2)[1]
        rows.append({"in rendered text": in_text, "token ids change": ids != ids2,
                     "newline token inside Qwen3-32B's cut": tok_i is not None and tok_i < QWEN3_CUT,
                     "token ids change inside Qwen3-32B's cut": ids[:QWEN3_CUT] != ids2[:QWEN3_CUT]})
    return x["orig_arm"], rows


def reach(name: str, out: Path, rec: dict):
    for s in range(N_SHARD_FILES):
        if sha((out / f"shard_{s}.jsonl").read_bytes()) != rec["inputs_sha256"][f"{PLANTED_RUN}/shard_{s}.jsonl"]:
            raise SystemExit(f"rebuilt shard_{s} does not match the staging record: run rebuild first")
    xs = [json.loads(line) for s in range(N_SHARD_FILES) for line in (out / f"shard_{s}.jsonl").read_text().splitlines()]
    n = Counter()
    _load_tok(name)  # preflight: a tokenizer that fails to load stops here, not in the pool's initializer
    with get_context("fork").Pool(initializer=_load_tok, initargs=(name,)) as pool:
        for orig, rows in pool.imap_unordered(_reach_one, [(name, x) for x in xs], chunksize=4):
            for r in rows:
                n[(orig, "gutter+newline outputs")] += 1
                n[(orig, "final newline in the full rendered text, at the output's end")] += r["in rendered text"]
                n[(orig, "full-transcript token ids change when the final newline is removed")] += r["token ids change"]
                if name == "Qwen3-32B":
                    n[(orig, "newline's token inside the 32,768-token cut")] += r["newline token inside Qwen3-32B's cut"]
                    n[(orig, "token ids change within the first 32,768 tokens")] += r["token ids change inside Qwen3-32B's cut"]
    import transformers
    staged = {(c["arm"], c["count"]): c["n"] for c in rec["counts"]}
    return {"reader": name, "tokenizer": TOKENIZERS[name], "transformers": transformers.__version__,
            "counts": [{"arm": a, "count": c, "n": int(v)} for (a, c), v in sorted(n.items())],
            "staging record's minimal-render count, beside": {a: staged.get((a, f"gutter+newline outputs whose trailing newline reaches the {name} prompt")) for a in ARMS}}


def main(mode: str, *args):
    me = Path(__file__).resolve()
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode:
        raise SystemExit("commit this script before running it")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if sha(STAGE_REC.read_bytes()) != STAGE_REC_SHA:
        raise SystemExit("not the staging record the P4-P6 scores records pin")
    rec = json.loads(STAGE_REC.read_text())
    _paths()
    if mode == "rebuild":
        res, tag = rebuild(Path(args[0]), Path(args[1]), rec), "rebuild"
    elif mode == "reach":
        res, tag = reach(args[0], Path(args[1]), rec), f"reach_{args[0]}"
    else:
        raise SystemExit("mode: rebuild | reach")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"POST HOC": "review thread #1084 rw2pY; not part of RULE.md's reading", "git": head, "mode": mode,
           "staging record": STAGE_REC.name, "staging record sha256": STAGE_REC_SHA} | res
    with open(dest := HERE / "results" / f"reach_full_{tag}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(*sys.argv[1:])
