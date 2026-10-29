"""The frozen sentence rule on a new stage folder. The selection functions (units, rules (a)-(c), FILE_REF, VERBS, the
changed-file regex, the byte mapping) are imported unchanged from experiments/2026-10-11_q2_e1_sentences/select_sentences.py
(a471d57b); only the loop over a folder is repeated here, with the same report rebuild and checks as stage 1's staging.
Never opens scores/; returns indices and counts, never text."""
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-11_q2_e1_sentences"))
from select_sentences import (APPENDED_TOKENS, CONDS, FILE_REF, MODEL, REVISION, VERBS,  # noqa: E402,F401
                              bytes_to_unicode, changed_file_re, units)

SELECT_SHA = "6ac16c17f8457797aa4586b71e2ff8e662cafb327c939a68441c0068dc276b4b"  # select_sentences.py at a471d57b, checked so the imported rule is the frozen one


def build_masks(folder: Path, path_of: dict) -> dict:
    """masks[f"{key}|{seq}"] for every staged report and readout-check sequence in folder; path_of: key -> target path."""
    from transformers import AutoTokenizer
    src = Path(__file__).resolve().parents[1] / "2026-10-11_q2_e1_sentences" / "select_sentences.py"
    if hashlib.sha256(src.read_bytes()).hexdigest() != SELECT_SHA:
        raise SystemExit("select_sentences.py differs from the frozen version")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    byte_of = {u: b for b, u in bytes_to_unicode().items()}
    added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    token_bytes = lambda seq: [added[t].encode() if t in added else bytes(byte_of[ch] for ch in tok.convert_ids_to_tokens(t)) for t in seq]
    terminal = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    staged = {(s["key"], s["seq"]): s for s in json.loads((folder / "staged.json").read_text())}
    masks = {}
    for f in sorted((folder / "generations").glob("*.json")):
        g = json.loads(f.read_text())
        cre = changed_file_re(path_of[g["key"]])
        for cond in CONDS:
            st = staged.get((g["key"], cond))
            if st is None:
                continue
            c = g[cond]
            comp = c["completion_ids"][:-1] if c["completion_ids"] and c["completion_ids"][-1] in terminal else c["completion_ids"]
            text = tok.decode(comp, skip_special_tokens=False)
            if b"".join(token_bytes(comp)).decode("utf-8") != text:
                raise SystemExit(f"{g['key']} {cond}: token bytes do not rebuild the completion")
            lo_c = text.rfind("</think>") + len("</think>") if "</think>" in text else 0
            hi_c = text.find("<tool_call>", lo_c) if "<tool_call>" in text[lo_c:] else len(text)
            report = text[lo_c:hi_c].strip()
            if hashlib.sha256(report.encode()).hexdigest() != st["report_sha256"]:
                raise SystemExit(f"{g['key']} {cond}: rebuilt report differs from the staged one")
            start = lo_c + text[lo_c:].index(report)
            ends = list(itertools.accumulate(len(x) for x in token_bytes(comp)))
            starts = [0] + ends[:-1]
            lo_r = len(text[:start].encode()); hi_r = lo_r + len(report.encode())
            inside = [i for i, (s_, e) in enumerate(zip(starts, ends)) if e > s_ and e > lo_r and s_ < hi_r]
            if [len(c["prompt_ids"]) + i for i in inside] != st["positions"]:
                raise SystemExit(f"{g['key']} {cond}: rebuilt positions differ from the staged ones")
            rel = {i: j for j, i in enumerate(inside)}

            def tokens_of(a, b):
                lo = len(text[:start + a].encode()); hi = lo + len(text[start + a:start + b].encode())
                return {rel[i] for i in inside if ends[i] > lo and starts[i] < hi}

            scored, path_tokens = set(), set()
            for u in units(report):
                t = report[u["start"]:u["end"]]
                if cre.search(t) or (VERBS.search(t) and FILE_REF.search(t)) or (u["list"] and not u["heading"] and VERBS.search(u["context"]) and FILE_REF.search(t)):
                    scored |= tokens_of(u["start"], u["end"])
            for mm in cre.finditer(report):
                path_tokens |= tokens_of(mm.start(), mm.end())
            masks[f"{g['key']}|{cond}"] = {"report tokens": len(inside), "scored": sorted(scored), "scored without the changed path": sorted(scored - path_tokens)}
    for key in {k.split("|")[0] for k in masks}:
        h = masks.get(f"{key}|honest")
        for ctl in ("honest+conceal", "honest+neutral"):
            st = staged.get((key, ctl))
            if h and st:
                n = len(st["positions"])
                if n != h["report tokens"] + APPENDED_TOKENS:
                    raise SystemExit(f"{key} {ctl}: control positions don't extend the honest report by {APPENDED_TOKENS}")
                masks[f"{key}|{ctl}"] = {"report tokens": n, "scored": h["scored"] + list(range(n - APPENDED_TOKENS, n)),
                                         "appended only": list(range(n - APPENDED_TOKENS, n))}
    want = {f"{k}|{q}" for k, q in staged if q in CONDS or q.startswith("honest+")}
    if set(masks) != want:
        raise SystemExit(f"masks don't cover the staged reports: {len(want - set(masks))} missing, {len(set(masks) - want)} extra")
    return masks
