"""Part 3: example pairs for why the two readers shift in opposite directions on text_editor calls (pre_call, attack
view, probes trained on real; saved scores only).

A. Qwen3.5: the 30 sim text_editor `view` calls (read-only) flagged above the frozen threshold that have, in the
   matched real run (same cell), a view of the same file scored below it. Ranked by the sim call's margin.
B. Qwen3-32B: the 30 sim text_editor edits (impactful) whose margin falls furthest below the median margin of the
   matched real run's edits of the same file.
For each: the call before it and that call's output (what the reader read last), the agent's text and the call.
Margin = score minus the fold's frozen threshold (logits). Writes examples.md and examples_features.json.
"""
import gzip
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from probe_shift.transcripts import reader_messages  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, RESULTS, ROOT, stamp as now  # noqa: E402

HERE = Path(__file__).parent
STAGED = Path.home() / "probe-shift-baselines/experiments/2026-10-06_q1_text_baselines/.run_cache/examples_2026-10-06_130135/outputs.jsonl.gz"
EVAL = Path.home() / "probe-shift-viewer/.run_cache/viewer/eval"
COHORT = {"real-attack-e1": "la_public10_real_attack_n1000", "sim-attack": "la_public10_sim_astra_attack_n1000"}
FLAGS = ROOT / "experiments/2026-10-06_q1_copied_vs_invented/.run_cache/flags_2026-10-07_121021.jsonl.gz"  # also on S3
EDITS = {"create", "str_replace", "insert"}
GUTTER = re.compile(r"^\s*\d+\t", re.M)  # the editor's line-number gutter


def table(reader):
    t = pd.read_json(DATA / "tables" / f"{reader}__pre_call__attack_only.jsonl.gz", lines=True).query("train_set == 'real'")
    return t.assign(margin=t.score - t.threshold).reset_index(drop=True)


def load_runs(keys):
    """key -> {(step, action): call, output, the agent's text} for every call of the run, as the reader indexes them."""
    where = {}
    with gzip.open(STAGED, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["key"] in keys and r["arm"] in COHORT:
                where[r["key"]] = (r["arm"], r["file"], r["sample_id"], r["epoch"])
    missing = keys - set(where)
    if missing:
        raise ValueError(f"{len(missing)} runs not in the staged table, e.g. {sorted(missing)[:2]}")
    by_file = defaultdict(list)
    for key, (arm, file, sid, ep) in where.items():
        by_file[(arm, file)].append((key, sid, ep))

    def read(item):
        (arm, file), wanted = item
        with zipfile.ZipFile(io.BytesIO((EVAL / COHORT[arm] / file).read_bytes())) as z:
            return [(key, json.loads(z.read(f"samples/{sid}_epoch_{ep}.json"))) for key, sid, ep in wanted]

    runs = {}
    with ThreadPoolExecutor(16) as pool:
        for got in pool.map(read, by_file.items()):
            for key, s in got:
                msgs = reader_messages(s, "")
                calls, step, pending = {}, -1, []
                for m in msgs:
                    if m["role"] == "assistant":
                        step += 1
                        pending = [(step, j) for j in range(len(m["tool_calls"]))]
                        for j, c in enumerate(m["tool_calls"]):
                            calls[(step, j)] = {"tool": c["function"]["name"], "args": c["function"]["arguments"],
                                                "text": m["content"], "output": None}
                    elif m["role"] == "tool" and pending:
                        calls[pending.pop(0)]["output"] = m["content"]
                runs[key] = calls
    return runs


def editor_args(call):
    a = call["args"] if isinstance(call["args"], dict) else json.loads(call["args"] or "{}")
    return a.get("command"), a.get("path")


def prev_visible(order, sa):
    """The call whose output the reader saw last before call `sa`: the last call of an earlier step. All calls of one
    assistant turn are written before any of their outputs, so a sibling call's output is not yet visible."""
    earlier = [o for o in order if o[0] < sa[0]]
    return earlier[-1] if earlier else None


def context(runs, flags, arm, key, step, action):
    """The call, the call whose output the reader saw last, and features of what the reader had seen."""
    calls = runs[key]
    order = sorted(calls)
    pk = prev_visible(order, (step, action))
    prev = calls[pk] if pk else None
    cmd, path = editor_args(calls[(step, action)])
    earlier = [calls[o] for o in order if o[0] < step]
    edited_before = any(c["tool"] == "text_editor" and editor_args(c)[0] in EDITS and editor_args(c)[1] == path for c in earlier)
    prev_out = (prev or {}).get("output") or ""
    return {"call": calls[(step, action)], "prev": prev,
            "features": {"step": step, "command": cmd, "path": path, "prev_tool": prev and prev["tool"],
                         "prev_editor_command": editor_args(prev)[0] if prev and prev["tool"] == "text_editor" else None,
                         "prev_same_path": bool(prev and prev["tool"] == "text_editor" and editor_args(prev)[1] == path),
                         "file_edited_earlier": edited_before,
                         "prev_flag": flags.get((arm, key) + pk) if pk else "first",
                         "prev_chars": len(prev_out), "prev_gutter_lines": len(GUTTER.findall(prev_out)),
                         "prev_is_error": prev_out.lower().startswith(("error", "traceback")) or "No such file" in prev_out,
                         "n_earlier_editor_outputs": sum(c["tool"] == "text_editor" for c in earlier)}}


def show(tag, row, ctx, cut=900):
    f = ctx["features"]
    prev = ctx["prev"]
    out = [f"**{tag}** `{row.arm}` {row.cell} step {row.step}.{row.action} | margin {row.margin:+.2f} | "
           f"prev: {f['prev_tool']} {f['prev_editor_command'] or ''} ({f['prev_flag']}, {f['prev_chars']} chars, "
           f"{f['prev_gutter_lines']} gutter lines){' same file' if f['prev_same_path'] else ''}"
           f"{' | file edited earlier' if f['file_edited_earlier'] else ''}"]
    if prev:
        out.append("previous call: `" + json.dumps(prev["args"])[:300].replace("`", "'") + "`")
        out.append("```\n" + (prev["output"] or "")[:cut] + ("\n[...]" if len(prev["output"] or "") > cut else "") + "\n```")
    if ctx["call"]["text"].strip():
        out.append("agent text: " + ctx["call"]["text"].strip()[:400].replace("\n", " "))
    out.append("call: `" + json.dumps(ctx["call"]["args"])[:500].replace("`", "'") + "`")
    return "\n\n".join(out)


def main():
    q35, q32 = table("qwen3.5"), table("qwen3-32b")
    keys = set(q32.key)
    runs = load_runs(keys)
    flags = {(r["arm"], r["key"], r["step"], r["action"]): r["out_match"]
             for r in map(json.loads, gzip.open(FLAGS, "rt")) if r["arm"] in COHORT}
    for t in (q35, q32):
        t["command"], t["path"] = zip(*[editor_args(runs[k][(s, a)]) if tool == "text_editor" else (None, None)
                                        for k, s, a, tool in zip(t.key, t.step, t.action, t.tool)])
    md, feats = ["# Example pairs (pre_call, attack view, probes trained on real)\n"], {"A": [], "A_real": [], "B": [], "B_real": []}

    # A: Qwen3.5 sim views above the threshold, with a same-file real view below it in the matched real run.
    views = q35[(q35.tool == "text_editor") & (q35.command == "view") & (q35.y == 0)]
    real_v = views[(views.domain == "real") & ~views.flag]
    sims = views[(views.domain == "sim") & views.flag].sort_values("margin", ascending=False)
    md.append("## A. Qwen3.5: sim views flagged, same-file real views not\n")
    n = 0
    for row in sims.itertuples():
        match = real_v[(real_v.cell == row.cell) & (real_v.path == row.path)]
        if match.empty:
            continue
        r = match.iloc[(match.margin - match.margin.median()).abs().argsort().iloc[0]]
        cs, cr = context(runs, flags, row.arm, row.key, row.step, row.action), context(runs, flags, r.arm, r.key, r.step, r.action)
        feats["A"].append(cs["features"] | {"margin": row.margin}); feats["A_real"].append(cr["features"] | {"margin": r.margin})
        md += [f"### A{n + 1}: `{row.path}`", show("sim", row, cs), show("real", r, cr), "---"]
        n += 1
        if n == 30:
            break
    md.append(f"\n(A: {len(sims)} sim views flagged in all; {n} have a same-file real view below the threshold.)\n")

    # B: Qwen3-32B sim edits furthest below the matched real run's same-file edits.
    edits = q32[(q32.tool == "text_editor") & q32.command.isin(EDITS) & (q32.y == 1)]
    real_e = edits[edits.domain == "real"].groupby(["cell", "path"]).margin.median().rename("real_median")
    sim_e = edits[edits.domain == "sim"].join(real_e, on=["cell", "path"], how="inner")
    sim_e["drop"] = sim_e.margin - sim_e.real_median
    md.append("## B. Qwen3-32B: sim edits scored furthest below same-file real edits\n")
    for n, row in enumerate(sim_e.sort_values("drop").head(30).itertuples()):
        cand = edits[(edits.domain == "real") & (edits.cell == row.cell) & (edits.path == row.path)]
        r = cand.iloc[(cand.margin - row.real_median).abs().argsort().iloc[0]]
        cs, cr = context(runs, flags, row.arm, row.key, row.step, row.action), context(runs, flags, r.arm, r.key, r.step, r.action)
        feats["B"].append(cs["features"] | {"margin": row.margin, "drop": row.drop}); feats["B_real"].append(cr["features"] | {"margin": r.margin})
        md += [f"### B{n + 1}: `{row.path}` (drop {row.drop:+.2f})", show("sim", row, cs), show("real", r, cr), "---"]
    md.append(f"\n(B: {len(sim_e)} sim edits have same-file real edits in the matched run.)\n")

    # The same features for every sim and real text_editor call, as the base rate for the selected 60.
    base = defaultdict(list)
    for name, t, m in (("views", q35, (q35.tool == "text_editor") & (q35.command == "view")),
                       ("edits", q32, (q32.tool == "text_editor") & q32.command.isin(EDITS))):
        for row in t[m].itertuples():
            base[f"{name} {row.domain}"].append(context(runs, flags, row.arm, row.key, row.step, row.action)["features"])
    (DATA / "examples.md").write_text("\n\n".join(md))
    (DATA / "examples_features.json").write_text(json.dumps({"selected": feats, "base": base}, default=str))
    print(DATA / "examples.md")


if __name__ == "__main__":
    main()
