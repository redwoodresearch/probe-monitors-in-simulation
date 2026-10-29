"""Stage one example per tool output for the Q1 text baselines, from the cohorts in S3 (read by alias).

For real attack epoch 1, epoch 2 and simulated attack on the shared paired cells, each tool output becomes one row:
  - `text`: the cleaned text, exactly what Qwen3-32B reads for that output (`reader_messages`: the error message
    in place of the output when the call errored, trailing newlines kept).
  - `readout_text`: `text` from its first character through the probe's readout, the last content token `output_end` (a
    final newline that the tokenizer merges with the template's next token is left out, as the probe's activation never
    sees it; NFC-normalized, as Qwen's tokenizer reads it).
  - `raw`: the tool message exactly as stored in the log (output, error record, tool name), IDs dropped.
  - calls.jsonl.gz: one row per tool call, `text` = the call as rendered (`{"name": ..., "arguments": ...}`) through
    its last content token (`call_end`); `kept` if that token is inside the cut.
  - `kept`: the probe's own rule, `probe_shift.reader.segment_end_positions`: the output's last content token
    (`output_end`) exists and lies inside the first `max_tokens` tokens. Outputs with no content token are not kept.
Attack runs are rendered with the honest system prompt of the same main task, as in the activation extraction, so
token positions match the probe's. Keys match the extraction's (`<file>__<sample id>__<epoch>`). Validity, truncation
and each simulated run's reference tag are checked against the shared splits file's per-sample table.

  uv run --with transformers==4.56.2 --with jinja2 python experiments/2026-10-06_q1_text_baselines/run_stage_examples.py
"""

import gzip
import json
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from probe_shift.models import QWEN3_32B
from probe_shift.reader import segment_end_positions
from probe_shift.transcripts import (RUNNER_SUBMIT, eval_files, file_samples, invalidity, main_task_id, reader_messages,
                                     system_prompt, tool_result_text)

HERE = Path(__file__).parent
SPLITS = HERE.parent.parent / "splits"


def side_task_id(sample: dict) -> str | None:
    st = (sample.get("metadata") or {}).get("side_task")
    return st.get("id") if isinstance(st, dict) else st


def stream(alias: str, threads: int):
    files = sorted(eval_files(alias))
    with ThreadPoolExecutor(threads) as ex:
        for f, samples in zip(files, ex.map(lambda f: file_samples(alias, f), files)):
            for s in samples:
                yield f, s


def raw_outputs(sample: dict) -> list[dict]:
    """The log's own tool messages that `reader_messages` keeps, in order, IDs dropped."""
    submit_ids = {tc.get("id") for m in sample["messages"] if m["role"] == "assistant"
                  for tc in m.get("tool_calls") or [] if tc["function"] == RUNNER_SUBMIT}
    return [m for m in sample["messages"] if m["role"] == "tool"
            and m.get("tool_call_id") not in submit_ids and m.get("function") != RUNNER_SUBMIT]


PROBE_CUT = 32_768  # the probe's cut; a run over it is "truncated" for the no_run_over_32k slice, whatever max_tokens


def stage_run(tok, messages: list[dict], max_tokens: int | None, meta: dict, raws: list[dict] | None = None):
    """One run's segment positions, output rows, call rows and run row, from the reader's messages cut to max_tokens
    (None: the whole transcript). `meta` (arm, key, cell, main_task, ...) goes on every row. `raws`: the log's own
    tool messages, for the `raw` field, when staging from the logs."""
    seg = segment_end_positions(tok, messages, max_tokens)
    ends = {(p["step"], p["action"]): p for p in seg["positions"] if p["kind"] == "output_end"}
    reader_tools = [m for m in messages if m["role"] == "tool"]
    key = meta["key"]
    assert len(seg["outputs"]) == len(reader_tools) == len(ends), f"{key}: output counts disagree"
    assert raws is None or len(raws) == len(reader_tools), f"{key}: raw and cleaned output counts disagree"
    rows = []
    for j, (out, m) in enumerate(zip(seg["outputs"], reader_tools)):
        end = ends[(out["step"], out["action"])]
        # The cleaned text up to the probe's readout: the output from its first character through output_end. A
        # final newline merged with the template's next token is left out, as the probe's activation never sees
        # it; leading content merged with the template's opening newline is kept, as the probe has read it.
        # NFC, as Qwen's tokenizer normalizes.
        readout = ""
        if end["kept"]:
            full = unicodedata.normalize("NFC", m["content"])
            inner = tok.decode(seg["ids"][out["start"]:end["token"] + 1], clean_up_tokenization_spaces=False)
            at = full.find(inner)
            assert at >= 0, f"{key}: readout text is not inside the output"
            readout = full[:at + len(inner)]
        row = meta | {"step": out["step"], "action": out["action"], "tool": out["tool"], "n_tokens": out["n_tokens"],
                      "output_end": end["token"], "kept": end["kept"], "text": m["content"], "readout_text": readout}
        if raws is not None:
            assert tool_result_text(raws[j]) == m["content"], f"{key}: raw and cleaned outputs are misaligned"
            row["raw"] = json.dumps({k: v for k, v in raws[j].items() if k not in ("id", "tool_call_id")}, ensure_ascii=False)
        rows.append(row)
    # Tool calls: the `{"name": ..., "arguments": ...}` text as rendered, from after `<tool_call>` through the call's
    # last content token (call_end, the probe's call readout).
    short = {k: meta[k] for k in ("arm", "key", "cell", "main_task", "sim_reference")}
    opens = {(p["step"], p["action"]): p for p in seg["positions"] if p["kind"] == "pre_call"}
    calls = []
    for p in seg["positions"]:
        if p["kind"] != "call_end":
            continue
        text = (tok.decode(seg["ids"][opens[(p["step"], p["action"])]["token"] + 1:p["token"] + 1],
                           clean_up_tokenization_spaces=False).lstrip("\n") if p["kept"] else "")
        assert not p["kept"] or text.startswith(f'{{"name": "{p["tool"]}"'), f"{key}: call text is misaligned"
        calls.append(short | {"step": p["step"], "action": p["action"], "tool": p["tool"], "call_end": p["token"],
                              "kept": p["kept"], "text": text})
    run = short | {"tokens": seg["n_tokens"], "truncated": seg["n_tokens"] > PROBE_CUT, "outputs": len(seg["outputs"]),
                   "kept_outputs": sum(r["kept"] for r in rows)}
    return seg, rows, calls, run


def main():
    cfg = yaml.safe_load((HERE / "config" / "stage.yaml").read_text())
    from transformers import AutoTokenizer

    assert cfg["tokenizer"] == QWEN3_32B, f"config tokenizer {cfg['tokenizer']} is not the shared pin {QWEN3_32B}"
    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])  # as the splits and extraction
    splits = json.loads((SPLITS / cfg["exclusions_file"]).read_text())
    paired = set(splits["paired_attack_cells"]["cells"])
    sim_reference = splits["paired_attack_cells"]["sim_reference"]
    shared = {(r["arm"], r["key"]): r for r in splits["samples"]}

    honest_prompts = {}
    for _, s in stream(cfg["cohorts"]["real-honest"], cfg["threads"]):
        honest_prompts.setdefault(main_task_id(s), set()).add(system_prompt(s))
    ambiguous = [mt for mt, ps in honest_prompts.items() if len(ps) != 1]
    assert not ambiguous, f"{len(ambiguous)} main tasks have more than one honest system prompt"
    honest_prompts = {mt: ps.pop() for mt, ps in honest_prompts.items()}

    rows, calls, runs, dropped = [], [], [], Counter()
    for arm in cfg["arms"]:
        alias = cfg["cohorts"][arm]
        chosen = {}
        for f, s in stream(alias, cfg["threads"]):
            mt, side = main_task_id(s), side_task_id(s)
            cell = f"{mt}:{side}"
            if cell not in paired:
                dropped[(arm, "not_paired")] += 1
                continue
            key = f"{f.removesuffix('.eval')}__{s['id'].replace('/', '_')}__{s['epoch']}"
            reason = invalidity(s)
            assert reason == shared[(arm, key)]["reason"], f"validity disagrees with {cfg['exclusions_file']} for {arm} {key}"
            if reason:
                dropped[(arm, reason)] += 1
                continue
            assert cell not in chosen, f"two valid {arm} runs for cell {cell}"  # as the probe's analysis requires
            chosen[cell] = (key, f, s)
        missing = paired - set(chosen)
        assert not missing, f"{arm}: {len(missing)} paired cells have no valid run, e.g. {sorted(missing)[:3]}"

        for cell, (key, f, s) in sorted(chosen.items()):
            mt = main_task_id(s)
            env = (s.get("metadata") or {}).get("environment")
            meta = {"arm": arm, "key": key, "file": f, "sample_id": s["id"], "epoch": s["epoch"],
                    "environment": env.get("id") if isinstance(env, dict) else env, "main_task": mt, "cell": cell,
                    "sim_reference": sim_reference[cell] if arm == "sim-attack" else None}
            seg, run_rows, run_calls, run = stage_run(tok, reader_messages(s, honest_prompts[mt]), cfg["max_tokens"], meta,
                                                      raw_outputs(s))
            assert seg["truncated"] == shared[(arm, key)]["truncated"], f"{key}: truncation disagrees with the splits"
            rows += run_rows
            calls += run_calls
            runs.append(run)
        print(f"{arm}: {len(chosen)} runs, {sum(r['arm'] == arm for r in rows)} outputs, {sum(c['arm'] == arm for c in calls)} calls", flush=True)
    under_32k = paired - {r["cell"] for r in runs if r["truncated"]}
    assert under_32k == set(splits["paired_attack_cells_under_32k"]["cells"]), "the under-32k cells disagree with the splits"

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / ".run_cache" / f"examples_{stamp}"
    out.mkdir(parents=True)
    with gzip.open(out / "outputs.jsonl.gz", "wt") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    with gzip.open(out / "calls.jsonl.gz", "wt") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in calls)
    (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs))
    (out / "stage.json").write_text(json.dumps({"config": cfg, "dropped": {f"{a}/{r}": n for (a, r), n in sorted(dropped.items())},
                                                "paired_cells": len(paired)}, indent=1))
    print("dropped:", dict(dropped))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
