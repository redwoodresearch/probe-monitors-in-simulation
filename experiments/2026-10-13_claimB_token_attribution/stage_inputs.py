"""Token attribution, step 1 (RULE.md in this folder): stage the honest inputs and count their tokens. $0, local CPU.

Every item is an honest LinuxArena run; no attack run is read. Transcripts are reader messages
(probe_shift.transcripts.reader_messages), with the system prompt of the site's first real run for every arm, rendered
with Qwen3-32B's template (pinned revision) and cut at 32k tokens, as steps A and B read them.

  site   The meaning judge's 91 strict sites (its results/prefix_matched_2026-10-12_145915.json; data/sites.jsonl and
         data/sim_new.jsonl are hash-checked against its results_manifest_v2.json). Three arms, each cut after the site's
         reply (the output of the k-th non-submit call):
           e1    the first real run;
           e2    the second real run (the same command at k);
           fake  natural sites (33): the simulated n437 run itself, cut after its own k-th output;
                 new sites (58): e1's messages with the reply replaced by the simulator's reply (sim_new reader_output).
         A site whose transcript in any arm is longer than the cut is dropped in all three arms (counted).
  plant / twin  N_PLANT replies per slip, seeded, at most one per run, from the sites' first and second real runs:
         gutter  a str_replace output in the real gutter form;
         note    an insert output ending in the review note.
         Each is planted with plant_both.plant_output (the plantslip transfer's code, unchanged). The transcript is cut
         after the planted reply; the twin is the same cut, unplanted. Planted sites are recorded as characters of the
         planted reply: for gutter, the first character of every line whose gutter was stripped, plus the final newline;
         for note, the reply's end (a deletion has no tokens of its own).
  whole  N_WHOLE seeded natural sites' whole runs, per arm (e1, e2 and the simulated run), cut at 32k tokens.

Writes data/staged.jsonl (git-ignored: transcript text) and results/stage_<stamp>.json (counts, token totals, hashes).

  uv run --no-project --python 3.14 --with transformers==4.56.2 --with jinja2 --with boto3 python \\
      experiments/2026-10-13_claimB_token_attribution/stage_inputs.py <meaning judge data dir>
"""

import hashlib
import json
import os
import random
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "experiments/2026-10-09_fixed_sim"), str(REPO / "experiments/2026-10-12_plantslip_transfer")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import fixes  # noqa: E402  (puts the rendering-swap folder on the path)
import rerender  # noqa: E402
from plant_both import GUTTER, plant_output  # noqa: E402

from probe_shift.models import QWEN3_32B  # noqa: E402
from probe_shift.reader import segment_end_positions  # noqa: E402
from probe_shift.transcripts import reader_messages, resolve_attachments, system_prompt  # noqa: E402

MJ = REPO / "experiments/2026-10-12_claimB_meaning_judge"
PM = MJ / "results/prefix_matched_2026-10-12_145915.json"
REAL, SIM = Path.home() / "probe-shift-viewer/.run_cache/viewer/eval", Path.home() / ".cache/probe-shift/s3"
CUT, N_PLANT, N_WHOLE = 32768, 30, 10
SEED = "token-attr-2026-10-13"
H = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
_zips = {}


def sample(row):
    if row["arm"] not in ("real-honest", "sim-honest"):
        raise ValueError(f"{row['key']}: not an honest run")
    path = REAL / row["cohort"] / row["file"] if row["arm"] == "real-honest" else SIM / row["cohort"].removeprefix("s3://") / "logs" / row["file"]
    if path not in _zips:
        with zipfile.ZipFile(path) as z:
            _zips[path] = {}
            for n in z.namelist():
                if n.startswith("samples/") and n.endswith(".json"):
                    x = json.loads(z.read(n))
                    _zips[path][(x["id"], x["epoch"])] = x
    return resolve_attachments(_zips[path][(row["sample_id"], row["epoch"])])


def paired(msgs):
    """Per tool message index, its call (name, arguments), paired by order as fixes.calls_of pairs them."""
    out, pending = {}, []
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            pending = [c["function"] for c in m.get("tool_calls") or []]
        elif m["role"] == "tool" and pending:
            f = pending.pop(0)
            out[i] = (f["name"], f["arguments"])
    return out


def cut_at(msgs, k, command, reply):
    """The messages up to and including the k-th tool output, checked against the site's command and reply."""
    tools = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
    i = tools[k]
    if paired(msgs).get(i) != (command["function"], command["arguments"]) or msgs[i]["content"] != reply:
        raise ValueError("the k-th output is not the site's command and reply")
    return msgs[: i + 1]


def gutter_sites(original, planted):
    """Characters of the planted reply where the slip shows: each stripped line's first character, and the final newline."""
    ob, pb = original.partition("+++ edited_version\n"), planted.partition("+++ edited_version\n")
    o_lines, p_lines, at, sites = ob[2].split("\n"), pb[2].rstrip("\n").split("\n"), len(pb[0]) + len(pb[1]), []
    if len(o_lines) != len(p_lines):
        raise ValueError("planted diff has another line count")
    for o, p in zip(o_lines, p_lines):
        if not o.startswith("@@") and GUTTER.match(o):
            sites.append(at)
        at += len(p) + 1
    return sites + [len(planted) - 1]


def main(mj_data):
    mj_data = Path(mj_data)
    man = json.loads((MJ / "results_manifest_v2.json").read_text())
    want = man["files"][f"{MJ.relative_to(REPO)}/data/meaning_run_sol_sim_2026-10-12.tar.gz"]["contains"]
    for f in ("sites.jsonl", "sim_new.jsonl"):
        if H((mj_data / f).read_bytes()) != want[f"data/{f}"]:
            raise SystemExit(f"{f} differs from the meaning judge's manifest")
    rows = {r["key"]: r for r in json.loads((REPO / "splits/exclusions_v4.json").read_text())["samples"]}
    pm = json.loads(PM.read_text())
    sites = {s["site"]: s for s in map(json.loads, (mj_data / "sites.jsonl").read_text().splitlines()) if s["set"] == "strict"}
    simnew = {r["pair"]: r for r in map(json.loads, (mj_data / "sim_new.jsonl").read_text().splitlines())}
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    items, notes, full = [], Counter(), {}

    def add(item, msgs):
        seg = segment_end_positions(tok, msgs, CUT)
        last = seg["outputs"][-1]
        item |= {"messages": msgs, "n_tokens": seg["n_tokens"], "truncated": seg["truncated"],
                 "reply_tokens": [last["start"], last["end"]], "reply_n_tokens": last["n_tokens"]}
        items.append(item)
        return item

    for i, st in enumerate(pm["strict_sites"]):
        s = sites[i]
        assert s["task"] == st["task"] and s["k"] == st["k"]
        x1, x2, xs = sample(rows[st["e1"]]), sample(rows[st["e2"]]), sample(rows[st["sim"]])
        sys1 = system_prompt(x1)
        notes["e2_own_prompt_equal"] += system_prompt(x2) == sys1
        notes["sim_own_prompt_equal"] += system_prompt(xs) == sys1
        m1, m2, ms = reader_messages(x1, sys1), reader_messages(x2, sys1), reader_messages(xs, sys1)
        full[i] = {"e1": m1, "e2": m2, "fake": ms if st["sim_matched"] else None}
        c1, c2 = cut_at(m1, s["k"], s["command"], s["e1"]), cut_at(m2, s["k"], s["command"], s["e2"])
        if st["sim_matched"]:
            cf = cut_at(ms, s["k"], s["command"], s["sim"])
        else:
            r = simnew[i]
            if r["status"] != "ok":
                raise ValueError(f"site {i}: simulator reply not usable")
            notes["new_fake_inspect_truncated"] += r["result"]["inspect_truncated"]  # used as the meaning judge used it
            cf = c1[:-1] + [c1[-1] | {"content": r["result"]["reader_output"]}]
        base = {"kind": "site", "site": i, "task": s["task"], "environment": s["environment"], "split": s["split"], "k": s["k"],
                "sim_source": s["sim_source"], "family": st["family"]}
        got = [add(base | {"arm": a, "run": st[{"e1": "e1", "e2": "e2", "fake": "sim"}[a]] if a != "fake" or st["sim_matched"] else st["e1"]}, m)
               for a, m in (("e1", c1), ("e2", c2), ("fake", cf))]
        if any(g["truncated"] for g in got):
            for g in got:
                items.remove(g)
            notes["sites_dropped_past_cut"] += 1
    kept_sites = sorted({it["site"] for it in items if it["kind"] == "site"})
    runs = sorted({(st[a], i, a) for i, st in enumerate(pm["strict_sites"]) for a in ("e1", "e2") if i in kept_sites})
    seen_runs, by_run = set(), {}
    for key, i, a in runs:  # one entry per run (a run can be e1 at one site and e2 at another)
        if key not in seen_runs:
            seen_runs.add(key)
            by_run[key] = full[i][a]
    for slip in ("gutter", "note"):
        rng = random.Random(f"{SEED}:plant:{slip}")
        order = sorted(by_run)
        rng.shuffle(order)
        n = 0
        for key in order:
            if n == N_PLANT:
                break
            msgs = by_run[key]
            seg = segment_end_positions(tok, msgs, None)
            tools = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
            cand = []
            for c in fixes.calls_of(msgs):
                cmd = c["args"].get("command")
                if c["tool"] != "text_editor" or cmd != ("str_replace" if slip == "gutter" else "insert"):
                    continue
                o = msgs[c["msg"]]["content"]
                p = plant_output(o, cmd, fixes, rerender)
                if p is None or p == o or seg["outputs"][tools.index(c["msg"])]["end"] + 16 > CUT:
                    continue
                cand.append((c["msg"], o, p))
            if not cand:
                continue
            j, o, p = rng.choice(cand)
            base = {"kind": "plant", "slip": slip, "run": key, "msg": j, "environment": rows[key]["environment"], "task": rows[key]["main_task"]}
            planted = msgs[:j] + [msgs[j] | {"content": p}]
            it = add(base | {"arm": "planted", "planted_chars": gutter_sites(o, p) if slip == "gutter" else [len(p) - 1]}, planted)
            tw = add(base | {"kind": "twin", "arm": "twin"}, msgs[: j + 1])
            if it["truncated"] or tw["truncated"]:
                raise ValueError("a planted item passed the cut")
            n += 1
        notes[f"plant_{slip}"] = n
    natural = [i for i in kept_sites if pm["strict_sites"][i]["sim_matched"]]
    for i in sorted(random.Random(f"{SEED}:whole").sample(natural, min(N_WHOLE, len(natural)))):
        st = pm["strict_sites"][i]
        for a in ("e1", "e2", "fake"):
            add({"kind": "whole", "site": i, "arm": a, "run": st[{"fake": "sim"}.get(a, a)], "environment": rows[st["e1"]]["environment"],
                 "task": st["task"]}, full[i][a])
    for n, it in enumerate(items):
        it["item"] = n
    data = HERE / "data/staged.jsonl"
    data.parent.mkdir(exist_ok=True)
    with open(data, "x") as fh:
        fh.writelines(json.dumps(it) + "\n" for it in items)
    read = lambda it: min(it["n_tokens"], CUT)  # noqa: E731  (tokens the forward pass reads)
    by = Counter()
    for it in items:
        by[f"{it['kind']}|{it['arm']}"] += read(it)
    out = {"label": "token attribution: staged honest inputs (counts and hashes only)", "counts": dict(Counter(f"{it['kind']}|{it['arm']}" for it in items)),
           "tokens_read": dict(by), "tokens_read_total": sum(by.values()), "notes": dict(notes), "sites_kept": len(kept_sites),
           "natural_sites_kept": len(natural), "reply_tokens_median": {k: sorted(it["reply_n_tokens"] for it in items if f"{it['kind']}|{it['arm']}" == k)[len([1 for it in items if f"{it['kind']}|{it['arm']}" == k]) // 2] for k in by},
           "staged_sha256": H(data.read_bytes()), "inputs_sha256": {f: H((mj_data / f).read_bytes()) for f in ("sites.jsonl", "sim_new.jsonl")},
           "reader": QWEN3_32B, "cut": CUT, "seed": SEED,
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": H(Path(__file__).read_bytes())}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / f"stage_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("reader",)}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
