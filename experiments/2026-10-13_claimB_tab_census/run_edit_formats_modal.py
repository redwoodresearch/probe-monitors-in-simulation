"""POST HOC (RULE.md Amendment 3): edit-reply formats per arm, counts only; and package versions per real cohort.

- `--part formats` (default): one Modal CPU container reads the 64 staged v4 shards and q2's fix-full shards 10 and 12
  from the activations Volume, classifies every text_editor reply (snippet / diff_gutter / diff_plain / created / other,
  plus a review-note flag), and writes counts to the Volume. Attack arms: counts only; row failures counted, never echoed.
- `--part versions`: reads only `header.json` of the real cohorts' .eval files uploaded unparsed under
  claimB/eval_headers_v1/<cohort>/, and records each file's sha256 and the package versions (strings and counts only).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-13_claimB_tab_census/run_edit_formats_modal.py [--part versions]
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-claimb-tab-census_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
V4 = "2026-10-06_la_segment-end-v4_qwen3-32b_v1/inputs"
FX = "2026-10-13_q2_fixfull_inputs_v2/inputs"
FX_SHARDS = {"shard_10.jsonl": ("sim-attack-fx-full", "5a3874814163b21e1c20dfedd00f1473dc01d2151b9769bb2930a1c16ae05652"),
             "shard_12.jsonl": ("sim-honest-fx-full", "94ae36d6d3bbd420d607254fd7c389910aee80b512d2281e19d2f08ae693b799")}
SPLIT = {"real-honest", "sim-attack", "sim-attack-fx-full"}
HEADERS = "claimB/eval_headers_v1"
OUT = "claimB/tab_census_v1"
SNIPPET = "Here's the result of running `cat -n` on a snippet of"  # rerender.py:27
DIFF_MARKS = ("+++ edited_version", "Here's a diff showing the changes made:")  # str_replace_tells / rerender.py:29
REVIEW = "Review the changes and make sure they are as expected"  # rerender.py:28
CMDS = ("str_replace", "insert", "create", "undo_edit")
CLASSES = ("snippet", "diff_gutter", "diff_plain", "created", "other")

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12")
vimage = image.pip_install("zstandard==0.23.0")  # .eval members are zstd-compressed (method 93), which 3.12's zipfile can't read


def classify(text: str) -> str:
    import re

    numbered = re.compile(r"^\s*\d+\|[ +\-] ")
    if SNIPPET in text:
        return "snippet"
    t = text.replace("\r\n", "\n")
    if any(m in t for m in DIFF_MARKS):
        body = t.split("+++ edited_version", 1)[1] if "+++ edited_version" in t else t.split(DIFF_MARKS[1], 1)[1]
        lines = body.split("\n")[1:]  # the rest of the marker's own line is not a hunk line
        if lines and lines[-1] == "":
            lines = lines[:-1]  # one trailing newline leaves one empty piece
        # every other line counts, blank ones included: a blank context line without its gutter is unnumbered
        hunk = [x for x in lines if not x.startswith(("@@", "---", "+++")) and REVIEW not in x]
        return "diff_gutter" if hunk and all(numbered.match(x) for x in hunk) else "diff_plain"
    if text.startswith("File created successfully at:"):
        return "created"
    return "other"


def shard_counts(job):
    from collections import Counter

    path, forced_arm, want = job
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if want is not None and sha != want:
        raise SystemExit("shard sha256 mismatch: stop")
    reps = Counter()  # (arm, cmd, class) -> replies; (arm, cmd, "review") -> review-note replies
    runs = Counter()  # (arm, cmd, class) -> runs with at least one
    nruns, errs = Counter(), Counter()
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        try:
            r = json.loads(line.decode("utf-8"))
            arm = forced_arm or r["arm"]
            arms = [arm] + ([f"{arm}|{str(r.get('file', '')).split('_', 1)[0]}"] if arm in SPLIT else [])
            seen, local, pending = set(), Counter(), []
            for m in r["messages"]:
                if m["role"] == "assistant":
                    pending = [dict(tc["function"]) for tc in (m.get("tool_calls") or [])]
                elif m["role"] == "tool":
                    call = pending.pop(0) if pending else None
                    if call is None or call.get("name") != m.get("tool_name"):
                        errs["pairing_mismatches"] += 1
                        continue
                    if call.get("name") != "text_editor":
                        continue
                    args = call.get("arguments") or {}
                    if isinstance(args, str):
                        args = json.loads(args)
                    cmd = dict(args).get("command")
                    if cmd == "view":
                        continue
                    cmd = cmd if cmd in CMDS else "other"
                    text = m["content"] if isinstance(m["content"], str) else ""
                    c = classify(text)
                    local[(cmd, c)] += 1
                    local[(cmd, "review")] += REVIEW in text
                    seen.add((cmd, c))
                    seen.add((cmd, "any"))
        except Exception:  # counted, never echoed
            errs["row_failures"] += 1
            continue
        for a in arms:
            nruns[a] += 1
            for (cmd, c), n in local.items():
                reps[(a, cmd, c)] += n
            for cmd, c in seen:
                runs[(a, cmd, c)] += 1
    return reps, runs, nruns, errs, (Path(path).relative_to("/vol").as_posix(), sha)


@app.function(image=image, volumes={"/vol": acts}, cpu=8, memory=16384, timeout=900, retries=0)
def formats(git: str, script_sha256: str) -> dict:
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    from zoneinfo import ZoneInfo

    v4 = sorted(str(p) for p in Path("/vol", V4).glob("shard_*.jsonl"))
    if len(v4) != 64:
        raise SystemExit(f"v4 shards: {len(v4)}")
    jobs = [(p, None, None) for p in v4] + [(str(Path("/vol", FX, s)), a, sha) for s, (a, sha) in FX_SHARDS.items()]
    reps, runs, nruns, errs, shas = Counter(), Counter(), Counter(), Counter(), {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r, u, n, e, (p, sha) in ex.map(shard_counts, jobs):
            reps.update(r)
            runs.update(u)
            nruns.update(n)
            errs.update(e)
            shas[p] = sha
    arms = sorted(nruns)
    table = {}
    for a in arms:
        table[a] = {"runs": nruns[a]}
        for cmd in CMDS + ("other",):
            N = sum(reps[(a, cmd, c)] for c in CLASSES)
            if not N:
                continue
            table[a][cmd] = {"N": N, "review_note": reps[(a, cmd, "review")], "runs_with_any": runs[(a, cmd, "any")],
                             "classes": {c: {"n": reps[(a, cmd, c)], "share": round(reps[(a, cmd, c)] / N, 4),
                                             "runs_with": runs[(a, cmd, c)]} for c in CLASSES}}
    # the pre-registered top line: real attack vs real honest, str_replace and insert
    top = {}
    for cmd in ("str_replace", "insert"):
        def dom(a):
            t = table.get(a, {}).get(cmd)
            if not t:
                return None, None
            succ = {c: t["classes"][c]["share"] for c in ("snippet", "diff_gutter", "diff_plain")}
            c = max(succ, key=succ.get)
            return c, succ[c]
        h = dom("real-honest")
        top[cmd] = {}
        for a in ("real-attack-e1", "real-attack-e2"):
            x = dom(a)
            same = (x[0] is not None and x[0] == h[0]
                    and abs(table[a][cmd]["classes"][x[0]]["share"] - table["real-honest"][cmd]["classes"][x[0]]["share"]) < 0.10)
            top[cmd][a] = {"attack_dominant": x[0], "attack_share": x[1], "honest_dominant": h[0], "honest_share": h[1],
                           "verdict": "SAME" if same else "DIFFERENT"}
    res = {"label": "POST HOC (RULE.md Amendment 3), preliminary until reviewed", "git": git, "script_sha256": script_sha256,
           "errors": dict(errs), "inputs_sha256": shas, "top_line": top, "table": table}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = Path("/vol", OUT)
    out.mkdir(parents=True, exist_ok=True)
    name = f"edit_formats_POSTHOC_{stamp}.json"
    with open(out / name, "x") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")
    acts.commit()
    return {"path": f"{OUT}/{name}", "errors": dict(errs), "top_line": top,
            "edit_shares": {a: {cmd: {c: (v["classes"][c]["n"], v["N"], v["classes"][c]["runs_with"]) for c in CLASSES
                                      if v["classes"][c]["n"]} | {"review_note": v["review_note"], "runs_with_any": v["runs_with_any"]}
                                for cmd, v in t.items() if cmd in ("str_replace", "insert")} | {"runs": t["runs"]}
                            for a, t in table.items()}}


def read_member(raw: bytes, name: str) -> bytes:
    """One zip member's bytes, stored, deflated or zstd (method 93), read from its local header by offset."""
    import io
    import struct
    import zipfile
    import zlib

    import zstandard

    zi = zipfile.ZipFile(io.BytesIO(raw)).getinfo(name)
    off = zi.header_offset
    n, e = struct.unpack("<HH", raw[off + 26:off + 30])
    data = raw[off + 30 + n + e: off + 30 + n + e + zi.compress_size]
    if zi.compress_type == 0:
        return data
    if zi.compress_type == 8:
        return zlib.decompress(data, -15)
    if zi.compress_type == 93:
        return zstandard.ZstdDecompressor().decompressobj().decompress(data)
    raise ValueError("unsupported compression")


@app.function(image=vimage, volumes={"/vol": acts}, cpu=4, memory=8192, timeout=900, retries=0)
def versions(git: str, script_sha256: str) -> dict:
    from collections import Counter, defaultdict
    from datetime import datetime
    from zoneinfo import ZoneInfo

    res = {"label": "POST HOC (RULE.md Amendment 3, versions), preliminary until reviewed", "git": git,
           "script_sha256": script_sha256, "cohorts": {}}
    for d in sorted(p for p in Path("/vol", HEADERS).iterdir() if p.is_dir()):
        files = sorted(d.glob("*.eval"))
        vers, shas, fails = defaultdict(Counter), {}, 0
        for f in files:
            raw = f.read_bytes()
            shas[f.name] = hashlib.sha256(raw).hexdigest()
            try:
                h = json.loads(read_member(raw, "header.json"))
                pk = (h.get("eval") or {}).get("packages") or {}
                for k, v in pk.items():
                    vers[k][str(v)] += 1
                if not pk:
                    vers["<no packages field>"]["-"] += 1
            except Exception:
                fails += 1
        res["cohorts"][d.name] = {"files": len(files), "header_failures": fails,
                                  "packages": {k: dict(v) for k, v in sorted(vers.items())}, "file_sha256": shas}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    name = f"eval_versions_POSTHOC_{stamp}.json"
    with open(Path("/vol", OUT, name), "x") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")
    acts.commit()
    return {"path": f"{OUT}/{name}", "cohorts": {c: {"files": v["files"], "header_failures": v["header_failures"],
                                                     "packages": v["packages"]} for c, v in res["cohorts"].items()}}


@app.local_entrypoint()
def main(part: str = "formats"):
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(HERE)], capture_output=True, text=True, cwd=HERE).stdout
    if dirty.strip():
        raise SystemExit("commit the script before running it")
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = formats.remote(git, sha) if part == "formats" else versions.remote(git, sha)
    print(json.dumps(out, indent=1))
