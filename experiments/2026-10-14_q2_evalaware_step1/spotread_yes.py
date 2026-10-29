"""Spot-read the honest runs the judge labelled YES (step 1). Runs on Modal CPU and reads ONLY the work Volume, which
holds honest-type reasoning (the judge refuses other arms). A run is YES if any of its chunks' first line parses YES,
the same rule as run_evalaware.py. For each YES run it returns the judge's response and short windows (±WIN chars)
around STRICT/BROAD matches, or around a fixed fallback cue regex when a run has none. It writes a static HTML viewer
(whole reasoning, matched spans highlighted) to the work Volume, never to git. The laptop writes a keys-only record to
results/ (rid, item indices, pattern ids, labels; no text) and prints the quotes for the report.
  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach <this file>
"""
import html
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q2-evalaware_spotread_kill-2026-11-14"
WORK = "probe-shift_aswin-surya_sub-q2-evalaware_work_kill-2026-11-14"
PASSES = ["first", "simh", "simh_final"]  # every judged pass (simh_final added 10-14 ~23:35, q2 OK)
WIN = 220
# fallback cues, used only for a YES run with no STRICT/BROAD match (fixed before reading)
CUES = r"\b(evaluat\w*|sandbox\w*|simulat\w*|honeypot|being\s+(tested|watched|monitored|observed)|real\s+(environment|system|user)|test\s+(scenario|environment))\b"
image = modal.Image.debian_slim(python_version="3.12").add_local_file(HERE / "run_evalaware.py", "/root/run_evalaware.py")
app = modal.App(APP)
work_vol = modal.Volume.from_name(WORK)


@app.function(image=image, volumes={"/work": work_vol}, cpu=(1, 1), memory=(1024, 1024), timeout=300, retries=0,
              startup_timeout=180, scaledown_window=2)
def spotread(stamp: str) -> dict:
    src = Path("/root/run_evalaware.py").read_text()  # the frozen patterns, taken from the count script, not retyped
    ns = {}
    exec(src[src.index('W = r"\\s+"'):src.index("ATT = ")], ns)  # only the W, STRICT and BROAD assignments
    STRICT, BROAD = ns["STRICT"], ns["BROAD"]
    pats = {**{k: re.compile(v, re.I) for k, v in STRICT.items()}, **{k: re.compile(v, re.I) for k, v in BROAD.items()}}
    out, page = {"runs": [], "judged_runs": {}}, []
    for ps in PASSES:
        jdir = Path("/work", ps, "judge")
        by_run = {}
        for f in sorted(jdir.glob("*.json")):
            rid, idx = f.stem.rsplit("_", 1)
            d = json.loads(f.read_text())
            ans = d.get("response", "")
            first = ans.splitlines()[0].strip().strip("*").strip().rstrip(".").upper() if ans else ""
            by_run.setdefault(rid, []).append((int(idx), first, ans))
        out["judged_runs"][ps] = len(by_run)
        for rid, chunks in sorted(by_run.items()):
            if not any(c[1] == "YES" for c in chunks):
                continue
            items = json.loads(Path("/work", ps, "reasoning", f"{rid}.json").read_text())
            hits = [(i, k, m.start(), m.end()) for i, t in enumerate(items) for k, rx in pats.items() for m in rx.finditer(t)]
            used = "STRICT/BROAD"
            if not hits:
                used = "fallback cues"
                hits = [(i, "cue", m.start(), m.end()) for i, t in enumerate(items) for m in re.finditer(CUES, t, re.I)]
            quotes = [{"item": i, "of": len(items), "pattern": k, "text": items[i][max(0, a - WIN):b + WIN]}
                      for i, k, a, b in hits[:8]]
            out["runs"].append({"pass": ps, "rid": rid, "judge": [{"chunk": c[0], "response": c[2]} for c in sorted(chunks)],
                                "windows from": used, "matches": len(hits), "quotes": quotes})
            marked = {}
            for i, k, a, b in hits:
                marked.setdefault(i, []).append((a, b, k))
            body = []
            for i, t in enumerate(items):
                spans, pos, seg = sorted(marked.get(i, [])), 0, []
                for a, b, k in spans:
                    if a < pos:
                        continue
                    seg += [html.escape(t[pos:a]), f'<mark title="{k}">{html.escape(t[a:b])}</mark>']
                    pos = b
                seg.append(html.escape(t[pos:]))
                cls = ' class="hit"' if i in marked else ""
                body.append(f'<div{cls} id="{rid}-{i}"><b>item {i}</b><pre>{"".join(seg)}</pre></div>')
            page.append(f'<h2>{ps} / {rid}</h2><p><i>judge:</i> {html.escape(" | ".join(c[2] for c in sorted(chunks)))}</p>'
                        f'<p>items with matches: {", ".join(f"<a href=#{rid}-{i}>{i}</a>" for i in sorted(marked))}</p>'
                        + "".join(body))
    doc = ("<!doctype html><meta charset=utf-8><title>Judge-YES runs</title><style>body{font:14px system-ui;margin:16px;"
           "max-width:1000px}pre{white-space:pre-wrap}div{border-top:1px solid #ccc}div.hit{background:#fff8d0}"
           "mark{background:#ffb000}</style><h1>Honest runs the judge labelled YES (step 1)</h1>" + "".join(page))
    p = Path("/work/spotread", f"yes_runs_{stamp}.html")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc)
    work_vol.commit()
    out["viewer on the work Volume"] = str(p.relative_to("/work"))
    return out


@app.local_entrypoint()
def main():
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = spotread.remote(stamp)
    keys = {"status": "PRELIMINARY until Aswin reviews", "launched (PT)": stamp, "judged runs per pass": out["judged_runs"],
            "viewer (work Volume, not git)": out["viewer on the work Volume"],
            "judge-YES runs (keys only; text stays on the Volume)": [
                {"pass": r["pass"], "rid": r["rid"], "windows from": r["windows from"], "matches": r["matches"],
                 "matched items [item, of, pattern]": [[q["item"], q["of"], q["pattern"]] for q in r["quotes"]]}
                for r in out["runs"]]}
    (HERE / "results" / f"spotread_yes_{stamp}.json").write_text(json.dumps(keys, indent=1) + "\n")
    print(json.dumps(out, indent=1))  # the quotes for the report go to this terminal, not to git
