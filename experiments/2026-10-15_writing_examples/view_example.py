"""One-line viewer for the writing session: row N of an example index -> the command's length and its first characters, read
from the row's source inside Modal (the .eval, or the sha-pinned Sol staged row). The bytes are re-hashed first and nothing is
shown unless the sha256 equals the index's. --check prints only the match and the length (no text). Nothing is written.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-15_writing_examples/view_example.py \\
      --index experiments/2026-10-15_writing_examples/results/examples_index_<A|B>_<stamp>.json --row N [--chars 200] [--args] [--check]
--args also shows the start of the call's full arguments as canonical JSON (useful for text_editor calls, whose 'command' is the
editor subcommand), only when they match the index's arguments_sha256.
"""

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-p2probes_exview_kill-2026-11-13"
VOLS = {
    "/q4": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13",
    "/sol": "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13",
    "/acts": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14",
}
MOUNT = {v: k for k, v in VOLS.items()}
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("zstandard==0.25.0")
    .add_local_file(HERE / "run_examples_index.py", "/root/run_examples_index.py")
)
app = modal.App(APP, image=image)
vols = {k: modal.Volume.from_name(v).read_only() for k, v in VOLS.items()}


@app.function(cpu=1, memory=4096, timeout=600, retries=0, volumes=vols)
def view(e: dict, chars: int, args: bool, check: bool) -> dict:
    err = None
    try:
        return _view(e, chars, args, check)
    except SystemExit as x:
        err = str(x)
    except Exception as x:  # noqa: BLE001 -- parse errors can hold source text: only the type crosses back
        err = f"stopped: {type(x).__name__} (message withheld)"
    raise SystemExit(err)


def _view(e: dict, chars: int, args: bool, check: bool) -> dict:
    import hashlib

    import run_examples_index as rx

    if "staged_row" in e:  # Sol (a): the sha-pinned staged shard line
        sr = e["staged_row"]
        raw = (Path(rx.SOL_STAGED) / sr["shard"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sr["shard_sha256"]:
            raise SystemExit("staged shard sha256 differs from the index")
        row = json.loads(raw.decode().splitlines()[sr["line"]])
        turns = [x for x in row["messages"] if x["role"] == "assistant"]
        tc = turns[e["unit"]["step"]]["tool_calls"][e["unit"]["action"]]
        fn = tc.get("function")
        a = fn.get("arguments") if isinstance(fn, dict) else tc.get("arguments")
        a = json.loads(a) if isinstance(a, str) else a
    else:  # own_frac100_L48 draws (index A) and index B: the .eval model event by x-irid, tool call call_index
        ev = e.get("run_eval") or e["run"]
        u = e.get("unit") if isinstance(e.get("unit"), dict) else {"irid": e["irid"], "call_index": e["call_index"]}
        s = rx.read_member(Path(MOUNT[ev["volume"]]) / ev["eval"], ev["sample_member"])
        model = [x for x in s.get("events") or [] if x.get("event") == "model"]
        hit = model[e["step"]]
        if (((hit.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid") != u["irid"]:
            raise SystemExit("the step's model event does not carry the indexed x-irid")
        a = hit["output"]["choices"][0]["message"]["tool_calls"][u["call_index"]].get("arguments")
    text = a["cmd"] if "cmd" in a else a["command"] if "command" in a else json.dumps(a, sort_keys=True, separators=(",", ":"))
    full = json.dumps(a, sort_keys=True, separators=(",", ":"))
    ok = hashlib.sha256(text.encode()).hexdigest() == e["sha256"] and len(text) == e["length"]
    ok_a = hashlib.sha256(full.encode()).hexdigest() == e.get("arguments_sha256") and len(full) == e.get("arguments_length")
    out = {"sha256_matches_index": ok, "length": len(text), "arguments_sha256_matches_index": ok_a}
    if ok and not check:
        out["first_chars"] = text[:chars]
        if args and ok_a:  # shown only when the whole arguments match the index's arguments_sha256 (Codex r1)
            out["arguments_json_first_chars"] = full[:chars]
    return out


@app.local_entrypoint()
def main(index: str, row: int, chars: int = 200, args: bool = False, check: bool = False):
    entries = json.loads(Path(index).read_text())["entries"]
    if row == -1:  # --row -1 --check: every ok row, match and length only
        if not check:
            raise SystemExit("--row -1 needs --check")
        ok = [i for i, x in enumerate(entries) if x["status"] == "ok"]
        for i, r in zip(ok, view.map([entries[i] for i in ok], kwargs={"chars": 0, "args": False, "check": True})):
            print(
                f"row {i}: length {r['length']} | sha256 matches index: {r['sha256_matches_index']} | arguments sha256 matches: {r['arguments_sha256_matches_index']}"
            )
        return
    e = entries[row]
    if e["status"] != "ok":
        raise SystemExit(f"row {row}: {e['status']}")
    r = view.remote(e, chars, args, check)
    where = e.get("task_key") or e.get("run")
    print(
        f"row {row} | {e.get('probe')} | {e.get('class') or e.get('run')} | {where} | step {e.get('step', e.get('unit'))} | tool {e['tool']} | field {e['field']} | "
        f"length {r['length']} | sha256 matches index: {r['sha256_matches_index']} | arguments sha256 matches: {r['arguments_sha256_matches_index']}"
    )
    for k in ("first_chars", "arguments_json_first_chars"):
        if k in r:
            print(f"{k}: {r[k]!r}")
