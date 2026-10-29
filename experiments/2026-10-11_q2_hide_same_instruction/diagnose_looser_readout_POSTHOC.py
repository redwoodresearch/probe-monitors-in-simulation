"""Row 11 pilot, post hoc (after "too few even at k = 24"): would option B's looser readout fix the yield? Option B keeps
any report tokens before the first mention of the changed file (all report tokens in a hid report), first 40, at least
10, with the other usable conditions unchanged. Counts token positions only: never opens scores/, writes no text and no
probe value. Then, as in diagnose_sizing_POSTHOC.py, the chance a full-stage run has both groups at k samples and the
expected count of 176 runs with both. Local, $0.

  uv run --no-project --python 3.12 --with numpy --with "transformers==5.19.0" --with jinja2 python \\
      experiments/2026-10-11_q2_hide_same_instruction/diagnose_looser_readout_POSTHOC.py <pilot stage folder> <readout json>
"""
import hashlib
import importlib.util
import itertools
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
SEL = HERE.parent / "2026-10-11_q2_e1_sentences" / "select_sentences.py"
spec = importlib.util.spec_from_file_location("select_sentences", SEL); sel = importlib.util.module_from_spec(spec); spec.loader.exec_module(sel)
INDEX = HERE.parents[1] / "data" / "hide_pilot" / "index_pilot_2026-10-11_204624.jsonl"
INDEX_SHA = "b1836ddeced3fa0978ee3b80911f306c905778343d98e42d959e0506820c1ef6"
MIN_TOKENS, K0, N_FULL = 10, 8, 176


def main(folder: Path, readout_p: Path):
    for f in (Path(__file__).resolve(), SEL):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if hashlib.sha256(INDEX.read_bytes()).hexdigest() != INDEX_SHA:
        raise SystemExit("pilot index differs")
    ro = json.loads(readout_p.read_text())
    if ro["staged.json sha256"] != hashlib.sha256((folder / "staged.json").read_bytes()).hexdigest():
        raise SystemExit("the readout was made from another stage folder")
    frozen = {(r["key"], r["seq"]): r for r in ro["rows"]}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sel.MODEL, revision=sel.REVISION)
    byte_of = {u: b for b, u in sel.bytes_to_unicode().items()}
    added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    token_bytes = lambda seq: [added[t].encode() if t in added else bytes(byte_of[ch] for ch in tok.convert_ids_to_tokens(t)) for t in seq]
    terminal = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    rows = {r["key"]: r for r in map(json.loads, INDEX.open())}
    flags = json.loads((folder / "flags.json").read_text())
    staged = {(s["key"], s["seq"]): s for s in json.loads((folder / "staged.json").read_text())}
    n, c, seen = Counter(), Counter(), Counter()
    for f in sorted((folder / "generations").glob("*.json")):
        g = json.loads(f.read_text()); key = g["key"]; cre = sel.changed_file_re(rows[key]["path"])
        for i, x in enumerate(g["hide"]["samples"]):
            seq = f"hide#{i}"; fl = flags[f"{key}|{seq}"]; fr = frozen[(key, seq)]; seen[(key, seq)] += 1
            if any(fl[q] != fr[q] for q in ("stop", "think closed", "report non-empty", "tool call")):
                raise SystemExit(f"{key} {seq}: flags differ from the frozen readout's")
            if fr["label"] is None or not (fl["stop"] and fl["think closed"] and fl["report non-empty"] and not fl["tool call"]):
                c["not usable before the readout (empty, unfinished or tool call)"] += 1; continue
            comp = x["completion_ids"][:-1] if x["completion_ids"] and x["completion_ids"][-1] in terminal else x["completion_ids"]
            tb = token_bytes(comp); text = b"".join(tb).decode("utf-8")
            lo_c = text.rfind("</think>") + len("</think>") if "</think>" in text else 0
            hi_c = text.find("<tool_call>", lo_c) if "<tool_call>" in text[lo_c:] else len(text)
            report = text[lo_c:hi_c].strip(); start = lo_c + text[lo_c:].index(report)
            st = staged[(key, seq)]  # the same checks as build_readout: tokens and report as staged
            if hashlib.sha256(np.asarray(g["hide"]["prompt_ids"] + x["completion_ids"], np.int32).tobytes()).hexdigest() != st["ids_sha256"] \
                    or hashlib.sha256(report.encode()).hexdigest() != st["report_sha256"]:
                raise SystemExit(f"{key} {seq}: tokens or report differ from the staged ones")
            first = cre.search(report); label = "named" if first else "hid"
            if label != fr["label"]:
                raise SystemExit(f"{key} {seq}: label differs from the frozen readout's")
            ends = list(itertools.accumulate(len(b) for b in tb)); starts = [0] + ends[:-1]
            lo_r = len(text[:start].encode()); cut = len(text[:start + (first.start() if first else len(report))].encode())
            # report tokens as build_readout counts them (overlapping the report), ending at or before the first mention
            m = sum(1 for s, e in zip(starts, ends) if e > s and e > lo_r and s < cut and (first is None or e <= cut))
            ok = m >= MIN_TOKENS; c[(label, "usable" if ok else "fewer than 10 tokens")] += 1
            n[(key, label)] += ok
    if set(seen) != set(frozen) or max(seen.values()) != 1:
        raise SystemExit("the generations' samples are not exactly the frozen readout's, each once")
    runs = sorted({r["key"] for r in ro["rows"]})  # as diagnose_sizing_POSTHOC.py
    rates = [(n[(k, "hid")] / K0, n[(k, "named")] / K0) for k in runs]
    out = {"plan": "PLAN.md (this folder); post hoc diagnostic of option B, no verdict", "git": head, "readout": readout_p.name,
           "counts": {" | ".join(k) if isinstance(k, tuple) else k: v for k, v in sorted(c.items(), key=str)},
           "usable": sum(n.values()), "pilot runs with both groups": sum(1 for k in runs if n[(k, "hid")] and n[(k, "named")]), "by k": {}}
    for k in (8, 12, 16, 24):
        p = sum(1 - (1 - h) ** k - (1 - m_) ** k + max(0.0, 1 - h - m_) ** k for h, m_ in rates) / len(rates)
        out["by k"][k] = {"P(a run has both groups)": round(p, 4), "expected runs with both (of 176)": round(N_FULL * p, 1)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"looser_readout_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
