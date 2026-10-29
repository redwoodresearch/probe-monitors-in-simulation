"""Row 11 amendment B, the sizing gate (AMENDMENT_B.md, committed with this script before any score is read under
readout B). The pilot is held out of every reading: this writes no hid-minus-named mean and no per-group mean.
- B tokens per usable sample, counted as in diagnose_looser_readout_POSTHOC.py (token positions only).
- The matched readout: per run, L = min(40, the fewest B tokens among its usable samples); each usable sample's score is
  the mean frozen-probe score over its first L report tokens, in the frozen honest SD.
- The pooled within-run, within-group SD of that readout, entered at its upper 90% confidence bound (chi-square).
- analyze_pilot.power, unchanged, at k = 8. Gate: power >= 0.80. Local, $0.

  uv run --no-project --python 3.12 --with numpy --with scipy --with "transformers==5.19.0" --with jinja2 python \\
      experiments/2026-10-11_q2_hide_same_instruction/analyze_sizing_b.py <pilot stage folder> <readout json> <confirm index jsonl>
"""
import hashlib
import importlib.util
import itertools
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import chi2

HERE = Path(__file__).parent
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
sel = load("select_sentences", HERE.parent / "2026-10-11_q2_e1_sentences" / "select_sentences.py")
AP = load("analyze_pilot", HERE / "analyze_pilot.py")
INDEX = HERE.parents[1] / "data" / "hide_pilot" / "index_pilot_2026-10-11_204624.jsonl"
INDEX_SHA = "b1836ddeced3fa0978ee3b80911f306c905778343d98e42d959e0506820c1ef6"
BUDGET, MIN_TOKENS, K, GOAL, SD_LEVEL = 40, 10, 8, 0.80, 0.90


def main(folder: Path, readout_p: Path, confirm_p: Path):
    for f in (Path(__file__).resolve(), HERE / "analyze_pilot.py", HERE / "AMENDMENT_B.md", HERE.parent / "2026-10-11_q2_e1_sentences" / "select_sentences.py"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if hashlib.sha256(INDEX.read_bytes()).hexdigest() != INDEX_SHA or hashlib.sha256(confirm_p.read_bytes()).hexdigest() != AP.CONFIRM_SHA:
        raise SystemExit("an index differs from its pin")
    ro = json.loads(readout_p.read_text())
    if ro["staged.json sha256"] != hashlib.sha256((folder / "staged.json").read_bytes()).hexdigest():
        raise SystemExit("the readout was made from another stage folder")
    frozen = {(r["key"], r["seq"]): r for r in ro["rows"]}
    if len(frozen) != len(ro["rows"]):
        raise SystemExit("the readout repeats a sample")
    staged = {(s["key"], s["seq"]): s for s in json.loads((folder / "staged.json").read_text())}
    flags = json.loads((folder / "flags.json").read_text())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sel.MODEL, revision=sel.REVISION)
    byte_of = {u: b for b, u in sel.bytes_to_unicode().items()}
    added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    token_bytes = lambda seq: [added[t].encode() if t in added else bytes(byte_of[ch] for ch in tok.convert_ids_to_tokens(t)) for t in seq]
    terminal = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    rows = {r["key"]: r for r in map(json.loads, INDEX.open())}
    usable, seen = defaultdict(list), Counter()  # run -> [(label, B-token count, (key, seq))]
    for f in sorted((folder / "generations").glob("*.json")):
        g = json.loads(f.read_text()); key = g["key"]; cre = sel.changed_file_re(rows[key]["path"])
        for i, x in enumerate(g["hide"]["samples"]):
            seq = f"hide#{i}"; fl = flags[f"{key}|{seq}"]; fr = frozen[(key, seq)]; seen[(key, seq)] += 1
            if any(fl[q] != fr[q] for q in ("stop", "think closed", "report non-empty", "tool call")):
                raise SystemExit(f"{key} {seq}: flags differ from the frozen readout's")
            if fr["label"] is None or not (fl["stop"] and fl["think closed"] and fl["report non-empty"] and not fl["tool call"]):
                continue
            comp = x["completion_ids"][:-1] if x["completion_ids"] and x["completion_ids"][-1] in terminal else x["completion_ids"]
            tb = token_bytes(comp); text = b"".join(tb).decode("utf-8")
            lo_c = text.rfind("</think>") + len("</think>") if "</think>" in text else 0
            hi_c = text.find("<tool_call>", lo_c) if "<tool_call>" in text[lo_c:] else len(text)
            report = text[lo_c:hi_c].strip(); start = lo_c + text[lo_c:].index(report)
            st = staged[(key, seq)]
            if hashlib.sha256(np.asarray(g["hide"]["prompt_ids"] + x["completion_ids"], np.int32).tobytes()).hexdigest() != st["ids_sha256"] \
                    or hashlib.sha256(report.encode()).hexdigest() != st["report_sha256"]:
                raise SystemExit(f"{key} {seq}: tokens or report differ from the staged ones")
            first = cre.search(report); label = "named" if first else "hid"
            if label != fr["label"]:
                raise SystemExit(f"{key} {seq}: label differs from the frozen readout's")
            ends = list(itertools.accumulate(len(b) for b in tb)); starts = [0] + ends[:-1]
            lo_r = len(text[:start].encode()); cut = len(text[:start + (first.start() if first else len(report))].encode())
            m = sum(1 for s, e in zip(starts, ends) if e > s and e > lo_r and s < cut and (first is None or e <= cut))
            if m >= MIN_TOKENS:
                usable[key].append((label, m, (key, seq)))
    expected = {(k, f"hide#{i}") for k in rows for i in range(K)}  # all 40 pinned pilot runs, eight samples each
    if set(seen) != expected or set(frozen) != expected or max(seen.values()) != 1:
        raise SystemExit("the generations and readout don't hold exactly the 40 pinned runs x 8 samples, each once")
    Ls = {key: min(BUDGET, min(m for _, m, _ in us)) for key, us in usable.items()}  # the matched length per run
    scores = {}
    for p_ in (folder / "scores").glob("*.json"):
        r = json.loads(p_.read_text())
        if r["ids_sha256"] != staged[(r["key"], r["seq"])]["ids_sha256"]:
            raise SystemExit(f"{p_.name}: score doesn't match its staged sequence")
        scores[(r["key"], r["seq"])] = r
    per = defaultdict(lambda: {"hid": [], "named": []}); oom = 0
    for key, us in usable.items():
        L = Ls[key]
        for label, m, k in us:
            s = scores.get(k)
            if s is None:
                raise SystemExit(f"{k}: a usable sample has no score file")
            if s.get("oom"):
                oom += 1; continue
            pt = np.asarray(s["per_token"], float)
            if len(pt) != len(staged[k]["positions"]) or len(pt) < L:
                raise SystemExit(f"{k}: per-token scores don't align with the staged report tokens")
            if not np.isfinite(pt[:L]).all():
                raise SystemExit(f"{k}: non-finite probe scores")
            per[key][label].append(float(pt[:L].mean()) / AP.SD0)
    runs = sorted(rows)  # the pinned pilot index
    resid = [v - np.mean(vs) for k in runs for grp in ("hid", "named") if len(vs := per[k][grp]) >= 2 for v in vs]
    dfree = sum(len(per[k][grp]) - 1 for k in runs for grp in ("hid", "named") if len(per[k][grp]) >= 2)
    if dfree <= 0:
        raise SystemExit("no within-run, within-group spread to size from: invalid sizing inputs")
    sd = float(np.sqrt(np.sum(np.square(resid)) / dfree))
    sd_up = sd * float(np.sqrt(dfree / chi2.ppf(1 - SD_LEVEL, dfree)))
    n_h = {k: len(per[k]["hid"]) for k in runs}; n_n = {k: len(per[k]["named"]) for k in runs}
    rates = [(n_h[k] / K, n_n[k] / K) for k in runs]
    pilot = set(runs); full_tasks = [r["main_task"] for r in map(json.loads, confirm_p.open()) if r["key"] not in pilot]
    pw = AP.power(rates, full_tasks, sd_up, K, np.random.default_rng(AP.SEED))
    out = {"amendment": "AMENDMENT_B.md", "git": head, "readout": readout_p.name, "usable samples (B)": sum(len(u) for u in usable.values()),
           "usable named / hid (B)": [sum(n_n.values()), sum(n_h.values())], "scored usable samples out of memory": oom,
           "pilot runs with both groups": sum(1 for k in runs if n_h[k] and n_n[k]),
           "matched length L per run: min / median / max": [min(Ls.values()), float(np.median(list(Ls.values()))), max(Ls.values())],
           "within-run, within-group SD of the matched readout (frozen honest SD units)": sd, "degrees of freedom": dfree,
           "SD used (upper 90% bound)": sd_up, "full-stage runs": len(full_tasks), "full-stage tasks": len(set(full_tasks)),
           "k": K, "true D": AP.TRUE_D, "simulations": AP.SIMS, "P(A detects at 0.3 SD), k = 8": pw,
           "gate": "passes: the full stage may launch (pre-approved, <= $32 with 30%)" if pw >= GOAL else "fails: row 11 closes, no further readout search"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"sizing_b_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
