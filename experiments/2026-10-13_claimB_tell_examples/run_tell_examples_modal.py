"""POST HOC (RULE.md in this folder): seeded random examples of three simulator tells, honest runs only.

One Modal CPU container reads q2's fix-full simulated honest shard, the v4 staged real-honest rows and fork B's ledger
from the activations Volume, draws one example per tell with SEED, and writes an HTML page plus a JSON record back to the
Volume. Nothing is computed on the laptop; the page (raw honest-run tool output) never goes to git.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-13_claimB_tell_examples/run_tell_examples_modal.py
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if modal.is_local() else HERE  # in the container the script sits at /root: no repo above it
APP = "probe-shift_aswin-surya_sub-claimb-tell-examples_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
V4 = "2026-10-06_la_segment-end-v4_qwen3-32b_v1/inputs"
FX12 = ("2026-10-13_q2_fixfull_inputs_v2/inputs/shard_12.jsonl", "94ae36d6d3bbd420d607254fd7c389910aee80b512d2281e19d2f08ae693b799")
LEDGER = ("claimB/tell_examples_v1/ledger_read10_2026-10-13.jsonl", "7e7465bca560898c8c070032ce20f0c237eaeeec3defac86aeefb5beebeca253")
EXCL_SHA = "a0ede1850566593776c9db8781553cadae95fd92d3b55c86f9bf753a304e6606"
OUT = "claimB/tell_examples_v1"
SEED = 61013
SLIP_ROWS = {0: "R01", 20: "R04", 35: "R07", 64: "M2"}  # fork B's ledger rows named in b7's directive (RULE.md)
SLIP_SENTENCE = {
    "R01": "The simulated listing of /app gives file sizes that differ from the real run's, and the simulator's reference run never listed /app, so the sizes were made up.",
    "R04": "The simulator answers as if the shell were already inside the project folder; in the real run the same relative path fails.",
    "R07": "The simulated shell reports /app/model-registry as its working directory; the real shell starts in /app.",
    "M2": "The simulator serves files at paths relative to the project folder; the real run, and the simulator's own reference run, say they don't exist.",
}
SNIPPET = "Here's the result of running `cat -n` on a snippet of"  # fork D, rerender.py:27
DIFF_MARKS = ("+++ edited_version", "Here's a diff showing the changes made:")
REVIEW = "Review the changes and make sure they are as expected"
MAXL, MAXC = 12, 160

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():  # the local file is added at build time; inside the container it is already in place
    image = image.add_local_file(REPO / "splits" / "exclusions_v4.json", "/root/exclusions_v4.json")


def classify(text: str) -> str:
    """Fork D's edit-reply classifier, unchanged (run_edit_formats_modal.py @ 7acd7ecf)."""
    import re

    numbered = re.compile(r"^\s*\d+\|[ +\-] ")
    if SNIPPET in text:
        return "snippet"
    t = text.replace("\r\n", "\n")
    if any(m in t for m in DIFF_MARKS):
        body = t.split("+++ edited_version", 1)[1] if "+++ edited_version" in t else t.split(DIFF_MARKS[1], 1)[1]
        lines = body.split("\n")[1:]
        if lines and lines[-1] == "":
            lines = lines[:-1]
        hunk = [x for x in lines if not x.startswith(("@@", "---", "+++")) and REVIEW not in x]
        return "diff_gutter" if hunk and all(numbered.match(x) for x in hunk) else "diff_plain"
    if text.startswith("File created successfully at:"):
        return "created"
    return "other"


def editor_replies(row):
    """(step, command, path, text) for every text_editor reply; step = 1-based position among all tool replies.
    Pairing as fork D: the k-th tool message after an assistant turn answers that turn's k-th call."""
    out, pending, step = [], [], 0
    for m in row["messages"]:
        if m["role"] == "assistant":
            pending = [tc["function"] for tc in m.get("tool_calls") or []]
        elif m["role"] == "tool":
            step += 1
            call = pending.pop(0) if pending else None
            if call is None or call.get("name") != m.get("tool_name") or call["name"] != "text_editor":
                continue
            args = call.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            text = m["content"] if isinstance(m["content"], str) else ""
            out.append((step, args.get("command"), str(args.get("path") or ""), text))
    return out


def content_tab_lines(text):
    """Indices of lines with a content tab: a tab left after removing one leading cat -n gutter tab (fork D's rule)."""
    import re

    gutter = re.compile(r"^(\s*\d+)\t")
    return [i for i, x in enumerate(text.split("\n")) if "\t" in gutter.sub(r"\1", x, count=1)]


def clip(lines, tabs=False):
    out = []
    for x in lines[:MAXL]:
        x = x.replace("\r", "")
        if tabs:
            x = x.replace("\t", "→")
        out.append(x if len(x) <= MAXC else x[: MAXC - 1] + "…")
    return "\n".join(out)


def real_rows(job):
    path, want = job
    got = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("arm") == "real-honest" and r["key"] in want:
                got[r["key"]] = r
    return got


@app.function(image=image, volumes={"/vol": acts}, cpu=8, memory=16384, timeout=900, retries=0)
def draw(git: str, script_sha256: str) -> dict:
    import html
    import random
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    from zoneinfo import ZoneInfo

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    if sha("/root/exclusions_v4.json") != EXCL_SHA:
        raise SystemExit("exclusions_v4.json sha256 mismatch: stop")
    fx, ld = Path("/vol", FX12[0]), Path("/vol", LEDGER[0])
    if sha(fx) != FX12[1] or sha(ld) != LEDGER[1]:
        raise SystemExit("input sha256 mismatch: stop")

    # real counterpart: the first valid overlap800 real-honest run by key in each simulated task (fork A's cohort_rows)
    ex = [r for r in json.loads(Path("/root/exclusions_v4.json").read_text())["samples"] if not r["reason"]]
    sim = {json.loads(x)["key"]: json.loads(x) for x in fx.read_text().splitlines() if x.strip()}
    by = {}
    for r in sorted((r for r in ex if r["arm"] == "real-honest" and r["key"].startswith("overlap800_")), key=lambda r: r["key"]):
        by.setdefault(r["cell"], []).append(r["key"])
    first = {k: by[s["cell"]][0] for k, s in sim.items() if by.get(s["cell"])}
    shards = sorted(str(p) for p in Path("/vol", V4).glob("shard_*.jsonl"))
    if len(shards) != 64:
        raise SystemExit(f"v4 shards: {len(shards)}")
    real = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for got in pool.map(real_rows, [(p, set(first.values())) for p in shards]):
            real.update(got)
    missing = sorted(set(first.values()) - set(real))
    if missing:
        raise SystemExit(f"{len(missing)} real counterparts missing from the v4 staged rows: stop")

    # pools
    pool_i, pool_ii, pick_info = [], [], {}
    for k in sorted(first):
        s, r = editor_replies(sim[k]), editor_replies(real[first[k]])
        si = [x for x in s if x[1] == "insert" and classify(x[3]) == "diff_plain"]
        ri = [x for x in r if x[1] == "insert" and classify(x[3]) == "snippet"]
        if si and ri:
            pool_i.append(k)
            pick_info[("i", k)] = (si[0], ri[0])
        sg = [x for x in s if x[2].endswith(".go") and content_tab_lines(x[3])]
        rg = [x for x in r if x[2].endswith(".go")]
        if sg and rg:
            pool_ii.append(k)
            pick_info[("ii", k)] = (sg[0], rg[0])
    ledger = [json.loads(x) for x in ld.read_text().splitlines() if x.strip()]
    pool_iii = sorted(SLIP_ROWS)
    for i in pool_iii:
        if ledger[i]["tag"] != SLIP_ROWS[i] or not ledger[i]["class"].startswith("content"):
            raise SystemExit(f"ledger row {i} is not the named content item: stop")

    rng = random.Random(SEED)
    k1, k2, i3 = rng.choice(sorted(pool_i)), rng.choice(sorted(pool_ii)), rng.choice(pool_iii)

    (s1, r1), (s2, r2) = pick_info[("i", k1)], pick_info[("ii", k2)]
    ex1 = (clip(s1[3].split("\n")), clip(r1[3].split("\n")))
    t0 = max(0, content_tab_lines(s2[3])[0] - 2)
    ex2 = (clip(s2[3].split("\n")[t0:], tabs=True), clip(r2[3].split("\n"), tabs=True))
    row3 = ledger[i3]
    keys3 = next((x for x in ledger if x["tag"] == row3["tag"] and "sim" in x), {})
    rows = [
        {"tell": "(i) Insert reply format", "pool": len(pool_i), "sim_key": k1, "real_key": first[k1],
         "sim_step": str(s1[0]), "real_step": str(r1[0]), "sim_path": s1[2], "real_path": r1[2],
         "sim_excerpt": ex1[0], "real_excerpt": ex1[1],
         "differs": "The simulated insert reply is a plain “+” diff (the full fix only appends the review note); the real tool answers an insert with a `cat -n` snippet of the edited file plus the review note."},
        {"tell": "(ii) Go file reply with literal tabs", "pool": len(pool_ii), "sim_key": k2, "real_key": first[k2],
         "sim_step": str(s2[0]), "real_step": str(r2[0]), "sim_path": s2[2], "real_path": r2[2],
         "sim_excerpt": ex2[0], "real_excerpt": ex2[1],
         "differs": "The simulated Go file reply keeps literal tab characters (shown as →) inside code lines; the real tool expands them to spaces (in a `cat -n` view, only the tab after the line number remains)."},
        {"tell": f"(iii) Factual slip (fork B's ledger row {i3}, {row3['tag']})", "pool": len(pool_iii),
         "sim_key": keys3.get("sim", ""), "real_key": keys3.get("real", ""),
         "sim_step": row3["steps_sim"], "real_step": row3["steps_real"], "sim_path": "", "real_path": "",
         "sim_excerpt": clip(row3["quote_sim"].split("\n")), "real_excerpt": clip(row3["quote_real"].split("\n")),
         "differs": SLIP_SENTENCE[row3["tag"]]},
    ]

    e = html.escape
    trs = "".join(
        f"<tr><td>{e(x['tell'])}</td><td>{x['pool']}</td><td>{SEED}</td>"
        f"<td><b>sim</b> {e(x['sim_key'])}<br><b>real</b> {e(x['real_key'])}</td>"
        f"<td><b>sim</b> {e(x['sim_step'])}<br><b>real</b> {e(x['real_step'])}</td>"
        f"<td><pre>{e(x['sim_excerpt'])}</pre></td><td><pre>{e(x['real_excerpt'])}</pre></td><td>{e(x['differs'])}</td></tr>"
        for x in rows)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Simulator tell examples</title><style>
:root{{--bg:#fff;--fg:#1b1b1b;--line:#d0d0d0;--code:#f4f4f4}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#e8e8e8;--line:#3a3a3a;--code:#202020}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:16px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid var(--line);padding:6px;vertical-align:top}}
pre{{background:var(--code);padding:6px;margin:0;white-space:pre-wrap;word-break:break-all;font-size:12px}}
.wrap{{overflow-x:auto}}</style></head><body>
<h1>Seeded random examples of three simulator tells (preliminary)</h1>
<p>Honest runs only: q2's fix-full simulated honest runs (437) against the first overlap800 real honest run on the same task.
One example per tell, drawn at random with seed {SEED} from the stated pool (RULE.md on aswin/claimb-tell-examples).
Excerpts are tool replies only, at most {MAXL} lines a side. Steps for (i) and (ii) are the reply's position among the run's
tool replies; for (iii) they are fork B's step/call notation. Row (iii)'s pool is the 4 named slips in the 12 runs fork B
read (10 drawn from the probe's top 10%, plus 2 of q2's movers), not a census. Built {stamp} PT, git {git[:8]}.</p>
<div class="wrap"><table><tr><th>Tell</th><th>Pool</th><th>Seed</th><th>Runs</th><th>Steps</th><th>Simulated reply</th>
<th>Real reply</th><th>What differs</th></tr>{trs}</table></div></body></html>
"""
    out = Path("/vol", OUT)
    out.mkdir(parents=True, exist_ok=True)
    pname, jname = f"tell_examples_{stamp}.html", f"tell_examples_{stamp}.json"
    with open(out / pname, "x") as fh:
        fh.write(page)
    rec = {"label": "POST HOC seeded random examples, preliminary", "git": git, "script_sha256": script_sha256, "seed": SEED,
           "inputs_sha256": {"fixfull_shard_12": FX12[1], "ledger": LEDGER[1], "exclusions_v4": EXCL_SHA},
           "pool_sizes": {"i": len(pool_i), "ii": len(pool_ii), "iii": len(pool_iii)},
           "picks": [{k: x[k] for k in ("tell", "pool", "sim_key", "real_key", "sim_step", "real_step")} for x in rows],
           "page": f"{OUT}/{pname}", "page_sha256": hashlib.sha256(page.encode()).hexdigest()}
    with open(out / jname, "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    acts.commit()
    return {"page": f"{OUT}/{pname}", "record": f"{OUT}/{jname}", "pool_sizes": rec["pool_sizes"], "page_sha256": rec["page_sha256"],
            "picks": rec["picks"]}


@app.local_entrypoint()
def main():
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--", str(HERE)], capture_output=True, text=True, cwd=HERE).stdout.strip():
        raise SystemExit("commit the script before running it")
    print(json.dumps(draw.remote(git, hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=1))
