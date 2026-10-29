"""The tells: which snippets TF-IDF (cleaned text) leans on, with one real and one simulated output where each appears.

Fits one model on all kept outputs of a comparison (no held-out folds: this is for reading, not scoring) and writes a
Markdown file: the top snippets pushing toward each side, each with its weight, how often it occurs per arm, and a short
excerpt around one occurrence in each arm. Comparisons: real epoch 1 vs simulated, and the null, epoch 1 vs epoch 2.
Settings (features, C, examples) come from config/tfidf.yaml.

  uv run python experiments/2026-10-06_q1_text_baselines/analyze_tfidf_tells.py
"""

import gzip
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from analyze_tfidf import E1, E2, SIM, featurize, fit, name_columns, scrub

HERE = Path(__file__).parent
SHOW = {"\n": "⏎", "\t": "⇥", "\r": "␍"}


def visible(s: str) -> str:
    return "".join(SHOW.get(ch, ch) for ch in s).replace("|", "\\|").replace("`", "'")


def excerpt(text: str, gram: str, width: int = 40) -> str:
    i = text.find(gram)
    a, b = max(0, i - width), min(len(text), i + len(gram) + width)
    return f"{'…' if a else ''}{visible(text[a:i])}**{visible(gram)}**{visible(text[i + len(gram):b])}{'…' if b < len(text) else ''}"


def main():
    cfg = yaml.safe_load((HERE / "config" / "tfidf.yaml").read_text())
    n_top = cfg["tells"]["top"]
    spec = cfg["baselines"][cfg["tells"]["baseline"]]
    rows = [json.loads(l) for l in gzip.open(HERE / ".run_cache" / cfg["examples"] / "outputs.jsonl.gz", "rt")]
    rows = [r for r in rows if r["kept"]]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    lines = [f"# TF-IDF tells ({cfg['tells']['baseline']}, {cfg['examples']})", "",
             "One model per comparison, fit on all kept outputs (for reading only; scores come from the out-of-fold runs).",
             f"Snippets are hashed word 1-2-grams or character {cfg['char_ngrams'][0]}-{cfg['char_ngrams'][1]}-grams; "
             "a column can hold more than one snippet (hash collision), listed most frequent first. "
             "⏎ is a newline, ⇥ a tab. Bold marks the snippet in each excerpt. Frequency is the share of each arm's outputs "
             "containing the first snippet. A word bigram is shown with a space between its two tokens (`10 :`), so it may not "
             "occur literally in the text; then its frequency reads 0% and no excerpt is shown.", ""]
    for name, (neg, pos) in {"Real epoch 1 vs simulated": (E1, SIM), "Null: real epoch 1 vs real epoch 2": (E1, E2)}.items():
        sub = [r for r in rows if r["arm"] in (neg, pos)]
        texts = [scrub(r[spec["text"]], spec["scrub"]) for r in sub]
        arm = np.array([r["arm"] for r in sub])
        X = featurize(texts, cfg)
        _, clf = fit(X, (arm == pos).astype(int), cfg, cfg["C"])
        coef = clf.coef_[0]
        order = np.argsort(coef)
        sides = {f"toward {pos}": order[::-1][:n_top], f"toward {neg}": order[:n_top]}
        names = name_columns(X, texts, np.concatenate(list(sides.values())), cfg)
        lines += [f"## {name}", ""]
        for side, cols in sides.items():
            lines += [f"### Top {n_top} {side}", "", f"| # | snippet | weight | in {neg} | in {pos} | example from {neg} | example from {pos} |",
                      "|---|---|---|---|---|---|---|"]
            for k, c in enumerate(cols, 1):
                grams = names[int(c)]
                g = grams[0] if grams else None
                cells = []
                for a in (neg, pos):
                    hits = [t for t, x in zip(texts, arm) if x == a and g and g in t]
                    share = len(hits) / max((arm == a).sum(), 1)
                    cells.append((f"{share:.1%}", excerpt(hits[0], g) if hits else "—"))
                shown = " / ".join(f"`{visible(x)}`" for x in grams) or "?"
                lines.append(f"| {k} | {shown} | {coef[c]:+.2f} | {cells[0][0]} | {cells[1][0]} | {cells[0][1]} | {cells[1][1]} |")
            lines.append("")
        print(f"{name}: done", flush=True)
    out = HERE / "results" / f"tfidf_tells_{stamp}.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
