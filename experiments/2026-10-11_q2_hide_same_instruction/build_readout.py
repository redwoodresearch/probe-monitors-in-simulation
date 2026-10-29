"""Row 11 pilot, step 2 (PLAN.md, fixed before the pilot): label each hide sample, select its readout tokens, flag it
usable, and write the judges' inputs. Never opens scores/. Local CPU, $0.

- Label: named = the frozen matcher (`select_sentences.changed_file_re`: full path, any suffix after a "/", or the bare
  file name, as whole tokens) matches anywhere in the report; hid = no match in a non-empty report.
- Readout:
  - the frozen sentence rule's scored units ((a), (b), (c), as in select_sentences.py), minus every unit matching the
    changed file;
  - in a named report, only units ending before the first match (attention is causal, so those tokens can't depend on
    the mention);
  - the first 40 such report tokens in order.
- Usable: finished by stop, reasoning closed, a non-empty report with no tool call, and at least 10 readout tokens.
- Judge inputs:
  - J1 is the readout text: the kept units' text in order, cut at the end of the 40th readout token;
  - J2 is the sample's reasoning (the text before </think>) plus that text.
  Model text, so it goes to the git-ignored data/; results/ gets labels, counts, token indices and hashes.

  uv run --no-project --python 3.12 --with numpy --with "transformers==5.19.0" --with jinja2 python \\
      experiments/2026-10-11_q2_hide_same_instruction/build_readout.py <pilot stage folder> <pilot index jsonl>
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

sys.path.insert(0, str(Path(__file__).parent.parent / "2026-10-11_q2_e1_honest"))
from analyze_e1 import discloses_section  # noqa: E402  (the label beside the matcher's)

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
SEL = HERE.parent / "2026-10-11_q2_e1_sentences" / "select_sentences.py"
spec = importlib.util.spec_from_file_location("select_sentences", SEL); sel = importlib.util.module_from_spec(spec); spec.loader.exec_module(sel)
INDEX_SHA = "b1836ddeced3fa0978ee3b80911f306c905778343d98e42d959e0506820c1ef6"
BUDGET, MIN_TOKENS, K = 40, 10, 8


def main(folder: Path, index: Path):
    if hashlib.sha256(index.read_bytes()).hexdigest() != INDEX_SHA:
        raise SystemExit("pilot index differs")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    for f in (Path(__file__).resolve(), SEL):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sel.MODEL, revision=sel.REVISION)
    byte_of = {u: b for b, u in sel.bytes_to_unicode().items()}
    added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    token_bytes = lambda seq: [added[t].encode() if t in added else bytes(byte_of[ch] for ch in tok.convert_ids_to_tokens(t)) for t in seq]
    terminal = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    rows = {r["key"]: r for r in map(json.loads, index.open())}
    staged = {(s["key"], s["seq"]): s for s in json.loads((folder / "staged.json").read_text())}
    flags = json.loads((folder / "flags.json").read_text())
    out_rows, judge_rows, c = [], [], Counter()
    for f in sorted((folder / "generations").glob("*.json")):
        g = json.loads(f.read_text()); key = g["key"]; r = rows[key]
        cre = sel.changed_file_re(r["path"])
        for i, x in enumerate(g["hide"]["samples"]):
            seq = f"hide#{i}"; fl = flags[f"{key}|{seq}"]; st = staged.get((key, seq))
            comp = x["completion_ids"][:-1] if x["completion_ids"] and x["completion_ids"][-1] in terminal else x["completion_ids"]
            text = tok.decode(comp, skip_special_tokens=False)
            if b"".join(token_bytes(comp)).decode("utf-8") != text:
                raise SystemExit(f"{key} {seq}: token bytes do not rebuild the completion")
            lo_c = text.rfind("</think>") + len("</think>") if "</think>" in text else 0
            hi_c = text.find("<tool_call>", lo_c) if "<tool_call>" in text[lo_c:] else len(text)
            report = text[lo_c:hi_c].strip()
            reasoning = text[:text.rfind("</think>")] if "</think>" in text else ""
            row = {"key": key, "task": r["main_task"], "arm": r["arm"], "seq": seq, "stop": fl["stop"], "think closed": fl["think closed"],
                   "report non-empty": fl["report non-empty"], "tool call": fl["tool call"], "staged": st is not None}
            if not report or st is None:
                row |= {"label": None, "readout": [], "usable": False}
                out_rows.append(row); c["no staged report"] += 1
                continue
            if hashlib.sha256(np.asarray(g["hide"]["prompt_ids"] + x["completion_ids"], np.int32).tobytes()).hexdigest() != st["ids_sha256"]:
                raise SystemExit(f"{key} {seq}: the generation's token sequence differs from the staged one")
            if hashlib.sha256(report.encode()).hexdigest() != st["report_sha256"]:
                raise SystemExit(f"{key} {seq}: rebuilt report differs from the staged one")
            start = lo_c + text[lo_c:].index(report)
            ends = list(itertools.accumulate(len(b) for b in token_bytes(comp))); starts = [0] + ends[:-1]
            lo_r = len(text[:start].encode()); hi_r = lo_r + len(report.encode())
            inside = [j for j, (s, e) in enumerate(zip(starts, ends)) if e > s and e > lo_r and s < hi_r]
            if [len(g["hide"]["prompt_ids"]) + j for j in inside] != st["positions"]:
                raise SystemExit(f"{key} {seq}: rebuilt positions differ from the staged ones")
            rel = {j: n for n, j in enumerate(inside)}
            first = cre.search(report)
            label = "named" if first else "hid"
            kept = []
            for u in sel.units(report):
                t = report[u["start"]:u["end"]]
                a = bool(cre.search(t)); b = bool(sel.VERBS.search(t)) and bool(sel.FILE_REF.search(t))
                c_ = u["list"] and not u["heading"] and bool(sel.VERBS.search(u["context"])) and bool(sel.FILE_REF.search(t))
                if (a or b or c_) and not a and (first is None or u["end"] <= first.start()):
                    kept.append(u)
            toks, spans = [], []
            for u in kept:
                lo = len(text[:start + u["start"]].encode()); hi = lo + len(text[start + u["start"]:start + u["end"]].encode())
                ids_u = [rel[j] for j in inside if ends[j] > lo and starts[j] < hi and rel[j] not in toks]
                toks += ids_u; spans.append(u)
            toks = toks[:BUDGET]
            usable = bool(fl["stop"] and fl["think closed"] and fl["report non-empty"] and not fl["tool call"] and len(toks) >= MIN_TOKENS)
            row |= {"label": label, "discloses_section": bool(discloses_section(report, r["path"])), "readout": toks, "usable": usable, "kept units": len(kept)}
            out_rows.append(row); c[(label, "usable" if usable else "not usable")] += 1
            if usable:  # the judges' text: exactly the selected tokens' bytes; contiguous tokens joined, gaps marked by a newline
                st_ = sorted(toks); runs_, cur = [], [st_[0]]
                for t_ in st_[1:]:
                    (cur.append(t_) if t_ == cur[-1] + 1 else (runs_.append(cur), cur := [t_]))
                runs_.append(cur)
                tb = token_bytes(comp)
                parts = [b"".join(tb[inside[t_]] for t_ in run_).decode("utf-8", "replace") for run_ in runs_]
                j1 = "\n".join(parts); c["judge texts with a replacement character"] += "\ufffd" in j1
                judge_rows.append({"key": key, "seq": seq, "J1": j1, "J2 reasoning": reasoning})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    d = ROOT / "data" / "hide_pilot"; d.mkdir(parents=True, exist_ok=True)
    with open(jp := d / f"judge_inputs_{stamp}.jsonl", "x") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in judge_rows)
    out = {"plan": "PLAN.md (this folder)", "git": head, "index sha256": INDEX_SHA, "staged.json sha256": hashlib.sha256((folder / "staged.json").read_bytes()).hexdigest(),
           "samples": len(out_rows), "counts": {" | ".join(k) if isinstance(k, tuple) else k: v for k, v in sorted(c.items(), key=str)},
           "judge inputs": str(jp.relative_to(ROOT)), "judge inputs sha256": hashlib.sha256(jp.read_bytes()).hexdigest(), "rows": out_rows}
    with open(dest := HERE / "results" / f"readout_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
