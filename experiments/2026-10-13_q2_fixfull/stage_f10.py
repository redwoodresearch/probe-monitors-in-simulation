"""F10 (PLAN_F10.md), POST HOC: expand literal tabs in simulated text_editor replies as the real tool does, on full + F9.
Modal CPU. Counts, hashes, token counts and the price only; never message text (attack runs included). Staged inputs go to
the Volume, never overwritten.

  --mode survey: gates, positive control, counts, token counts and price; nothing written to the Volume.
  --mode stage:  the same, then `2026-10-13_q2_f10_inputs_v1/inputs/shard_<n>.jsonl` (changed runs only; arms
                 sim-attack-fx-fullF9F10 and sim-honest-fx-fullF9F10) and its stage_record.json.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run \\
      experiments/2026-10-13_q2_fixfull/stage_f10.py --mode survey
"""
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
SWAP, FSIM = REPO / "experiments" / "2026-10-07_q2_rendering_swap", REPO / "experiments" / "2026-10-09_fixed_sim"
APP = "probe-shift_aswin-surya_sub-q2-f10_stage_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
G3OUT = "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"  # Seed metas (forward tokens), read-only
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
F9SRC, DST = "2026-10-13_q2_fixfull_inputs_f9_v1", "2026-10-13_q2_f10_inputs_v1"
BASE = "2026-10-06_la_segment-end_qwen3-32b_v1"  # #36's staged inputs: the real arms (positive control)
F9_STAGE = HERE / "results" / "stage_f9_2026-10-13_145822.json"  # its record_sha256 pins F9SRC's stage record
SEED_MAP = HERE / "results" / "f9_mapping_seed_2026-10-13_154954.json"
RUNS36 = REPO / "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
SRC_ARMS = {"sim-attack-fx-fullF9": ("attack", 939), "sim-honest-fx-fullF9": ("honest", 437)}
REAL = {"real-attack-e1": 939, "real-attack-e2": 939, "real-honest": 990}
SEED_RUNS = {"full": "2026-10-13_fixfull_seed-oss-36b_v1", "fullF9": "2026-10-13_fixfull_f9_seed-oss-36b_v1"}
PRICE = {"qwen3 $/1M forward tokens": 0.78, "qwen3 $/model load": 0.26, "seed $/1M forward tokens (F9 measured, loads included)": 1.59}
MAX_TOKENS = 32768

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
g3out = modal.Volume.from_name(G3OUT, version=2).read_only()
hf_cache = modal.Volume.from_name(HF_CACHE).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==4.56.2", "jinja2==3.1.6", "numpy", "unidiff==0.7.5")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift")
         .add_local_file(SWAP / "rerender.py", "/root/swap/rerender.py")
         .add_local_file(SWAP / "str_replace_tells.py", "/root/swap/str_replace_tells.py")
         .add_local_file(FSIM / "fixes.py", "/root/fsim/fixes.py"))

DIRLIST = "Here are the files and directories up to 2 levels deep"
CAT_N = re.compile(r"^( *\d+)\t(.*)$")
GUTTER = re.compile(r"^( *\d+\|[+\- ] )(.*)$")
ECHO = "old_str `"  # both str_replace errors echo old_str after `expandtabs()` (text_editor.py:91, :98, :106)
COMMANDS = ("view", "create", "str_replace", "insert", "undo_edit")


def f10_text(text: str, c: Counter | None = None) -> str:
    """PLAN_F10.md's six line classes; expansion starts where the real tool's content starts."""
    if not isinstance(text, str) or "\t" not in text or text.startswith(DIRLIST):
        if c is not None and isinstance(text, str) and "\t" in text:
            c["line class|1 directory listing (outputs left unchanged)"] += 1
        return text
    out, hunk = [], False
    for line in text.split("\n"):
        tab = "\t" in line
        if m := CAT_N.match(line):
            cls, new = "2 cat -n", m.group(1) + "\t" + m.group(2).expandtabs()
            tab = "\t" in m.group(2)
        elif m := GUTTER.match(line):  # control_tower rstrips each diff value (text_editor_output.py:41)
            cls, new = "3 diff gutter", m.group(1) + (m.group(2).expandtabs().rstrip() if tab else m.group(2))
        elif line.startswith("@@ "):
            hunk, cls, new = True, "6 other", line.expandtabs()
        elif hunk and line[:1] in (" ", "+", "-", "\\"):  # a hunk runs until a line outside the unified-diff body alphabet
            cls, new = "4 unified diff body", line[0] + line[1:].expandtabs()
        elif (i := line.find(ECHO)) >= 0:
            hunk, j = False, i + len(ECHO)
            cls, new = "5 str_replace error echo", line[:j] + line[j:].expandtabs()
        else:
            hunk = False
            cls, new = "6 other", line.expandtabs()
        if c is not None and tab:
            c[f"line class|{cls} (lines with a content tab)"] += 1
        out.append(new)
    return "\n".join(out)


def content_tabs(text) -> int:
    """Tabs outside a `cat -n` gutter (the census's 'content tabs'); directory listings hold none (the real tool keeps them)."""
    if not isinstance(text, str) or text.startswith(DIRLIST):
        return 0
    return sum(m.group(2).count("\t") if (m := CAT_N.match(x)) else x.count("\t") for x in text.split("\n"))


def fixtures(c: Counter) -> bool:
    """Source-derived fixtures: the real renderers (Inspect's `_make_output`, text_editor.py:231-249; control_tower's diff,
    text_editor_output.py:8-54; the old_str echoes) with tabs kept must map under F10 to the same renderers with tabs
    expanded. Synthetic files only (seed 20261013); no transcript text."""
    import difflib
    import random
    from io import StringIO

    import unidiff

    def make_output(content, desc, init=1, expand=True):
        content = content.expandtabs() if expand else content
        return f"Here's the result of running `cat -n` on {desc}:\n" + "\n".join(f"{i + init:6}\t{x}" for i, x in enumerate(content.split("\n"))) + "\n"

    def diff(before, after, expand=True):
        b, a = (before.expandtabs(), after.expandtabs()) if expand else (before, after)
        text = "\n".join(difflib.unified_diff(b.splitlines(), a.splitlines(), fromfile="old_version", tofile="edited_version", lineterm=""))
        res = []
        for pf in unidiff.PatchSet(StringIO(text)):
            res += [f"--- {pf.source_file}", f"+++ {pf.target_file}"]
            for h in pf:
                res.append(f"@@ -{h.source_start},{h.source_length} +{h.target_start},{h.target_length} @@")
                for ln in h:
                    pre = f"{ln.target_line_no:>4}|+" if ln.is_added else f"{ln.source_line_no:>4}|{'-' if ln.is_removed else ' '}"
                    res.append(f"{pre} {ln.value.rstrip()}")
        return "The file /app/x.go has been edited. Here's a diff showing the changes made:\n" + "\n".join(res)

    rng = random.Random(20261013)
    words = ["func", "x", "if", "return", "err", "nil", "{", "}", "ab", "abcdefghi", "//", "1", "a\tb"]
    for n in range(300):
        lines = ["".join(rng.choice(["\t", "", " ", "  "]) for _ in range(rng.randint(0, 3))) + " ".join(rng.choice(words) for _ in range(rng.randint(1, 5)))
                 + rng.choice(["", "\t", " \t"]) for _ in range(rng.randint(3, 12))]
        f = "\n".join(lines)
        k = rng.randrange(len(lines))
        g = "\n".join(lines[:k] + ["\tadded " + lines[k]] + lines[k + 1:])
        old = lines[k]
        cases = {"view": (make_output(f, "/app/x.go", expand=False), make_output(f, "/app/x.go")),
                 "diff": (diff(f, g, expand=False), diff(f, g)),
                 "not found": (f"No replacement was performed, old_str `{old}` did not appear verbatim in /app/x.go.",
                               f"No replacement was performed, old_str `{old.expandtabs()}` did not appear verbatim in /app/x.go."),
                 "multiple": (f"No replacement was performed. Multiple occurrences of old_str `{old}` in lines [1, 2]. Please ensure it is unique",
                              f"No replacement was performed. Multiple occurrences of old_str `{old.expandtabs()}` in lines [1, 2]. Please ensure it is unique")}
        for name, (kept, real) in cases.items():
            c[f"fixture|{name}|cases"] += 1
            c[f"fixture|{name}|F10 equals the real rendering"] += f10_text(kept) == real
    return all(c[f"fixture|{x}|cases"] == c[f"fixture|{x}|F10 equals the real rendering"] for x in ("view", "diff", "not found", "multiple"))


def msha(msgs):
    return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()


def run_job(mode: str, pins: dict, real_keys: dict, seed_overrides: dict, qwen_rev: str) -> dict:
    sys.path[:0] = ["/root/swap", "/root/fsim"]
    import fixes
    from transformers import AutoTokenizer

    from probe_shift.reader import segment_end_positions

    def rows(run, arms):
        out = {}
        for p in sorted((Path("/acts") / run / "inputs").glob("shard_*.jsonl")):
            for line in p.read_text().splitlines():
                it = json.loads(line)
                if it["arm"] in arms and (it["arm"], it["key"]) in out:
                    raise ValueError("a run twice in a staged run")
                if it["arm"] in arms:
                    out[(it["arm"], it["key"])] = it
        return out

    rec_path = Path("/acts") / F9SRC / "stage_record.json"
    if hashlib.sha256(rec_path.read_bytes()).hexdigest() != pins["f9 record"]:
        raise ValueError("F9's stage record is not the pinned one")
    f9_sha = {(r["arm"], r["key"]): r["messages_sha256"] for r in json.loads(rec_path.read_text())["runs"]}
    src = rows(F9SRC, set(SRC_ARMS))
    c = Counter()
    for arm, (_, n) in SRC_ARMS.items():
        got = [k for (a, k) in src if a == arm]
        if len(got) != n:
            raise ValueError(f"{arm}: {len(got)} runs, not {n}")
    c["gate|base runs equal to F9's stage record (message sha256)"] = sum(msha(it["messages"]) == f9_sha.get(ak) for ak, it in src.items())

    def apply(it, group, tally):
        m = it["messages"]
        te = {x["msg"]: x for x in fixes.calls_of(m) if x["tool"] == "text_editor"}
        new = [dict(x) for x in m]
        for i, x in te.items():
            go = str(x["args"].get("path", "")).endswith(".go")
            o = m[i]["content"]
            if not isinstance(o, str):
                tally[f"{group}|text_editor messages whose content is not a string (left unchanged)"] += 1
            has = content_tabs(o) > 0
            tally[f"{group}|text_editor messages|{'go' if go else 'other'}"] += 1
            tally[f"{group}|text_editor messages with a content tab|{'go' if go else 'other'}"] += has
            new[i] = m[i] | {"content": f10_text(o, tally if group.startswith("sim") else None)}
            if new[i] != m[i]:
                cmd = x["args"].get("command") if x["args"].get("command") in COMMANDS else "unknown"
                tally[f"{group}|messages changed|{'go' if go else 'other'}|text_editor:{cmd}"] += 1
        changed = [i for i, (a, b) in enumerate(zip(m, new)) if a != b]
        return new, changed, te

    staged, record, tok_in = {}, [], []
    for (arm, k), it in sorted(src.items()):
        group = f"sim-{SRC_ARMS[arm][0]}"
        new, changed, te = apply(it, group, c)
        if len(new) != len(it["messages"]) or any(i not in te or set(it["messages"][i]) != set(new[i]) or
                                                  any(it["messages"][i][f] != new[i][f] for f in new[i] if f != "content") for i in changed):
            raise ValueError("F10 changed a non-text_editor message, the message count, or a field other than content")
        c[f"{group}|runs changed"] += bool(changed)
        c[f"{group}|messages changed"] += len(changed)
        c[f"{group}|content tabs left after F10 in text_editor messages"] += sum(content_tabs(new[i]["content"]) for i in te)
        if changed:
            again = [x | {"content": f10_text(x["content"])} if j in te else x for j, x in enumerate(new)]
            c[f"{group}|runs where F10 is idempotent"] += msha(again) == msha(new)
            c[f"{group}|runs changed, go"] += any(str(te[i]["args"].get("path", "")).endswith(".go") for i in changed)
            dst_arm = arm.replace("fullF9", "fullF9F10")
            row = it | {"arm": dst_arm, "messages": new}
            c[f"{group}|staged rows holding attachment://"] += "attachment://" in json.dumps(row)
            staged.setdefault(dst_arm, []).append(row)
            record.append({"arm": dst_arm, "key": k, "changed_messages": changed, "messages_sha256": msha(new),
                           "base_messages_sha256": msha(it["messages"]), "first_changed_message": changed[0]})
            tok_in.append((group, k, it["messages"], new))

    # positive control: F10 on the real arms (counts only)
    real = rows(BASE, set(REAL))
    for arm, n in REAL.items():
        keys = set(real_keys[arm])
        got = {k: it for (a, k), it in real.items() if a == arm and k in keys}
        if len(got) != n:
            raise ValueError(f"{arm}: {len(got)} of {n} runs found")
        for k, it in got.items():
            _, changed, _ = apply(it, arm, c)
            c[f"{arm}|runs changed"] += bool(changed)
            c[f"{arm}|messages changed"] += len(changed)

    # tokens: Qwen3-32B exact (the extractor's segment_end_positions, cut at 32,768), before and after F10; Seed scaled
    # from its full + F9 forward tokens (fix-full's or F9's meta) by the run's Qwen3-32B after/before ratio
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", revision=qwen_rev)
    tokens = Counter()
    for group, k, old, new in tok_in:
        q_old, q_new = (len(segment_end_positions(tok, x, MAX_TOKENS)["ids"]) for x in (old, new))
        side = group.removeprefix("sim-")
        run_, arm_ = (SEED_RUNS["fullF9"], f"sim-{side}-fx-fullF9") if k in seed_overrides[side] else (SEED_RUNS["full"], f"sim-{side}-fx-full")
        s_old = json.loads((Path("/g3") / run_ / "meta" / arm_ / f"{filename(k)}.json").read_text())["forward_tokens"]
        tokens[f"{group}|qwen3 tokens before F10"] += q_old
        tokens[f"{group}|qwen3 tokens after F10"] += q_new
        tokens[f"{group}|seed forward tokens before F10 (full + F9 meta)"] += s_old
        tokens[f"{group}|seed forward tokens after F10 (scaled)"] += round(s_old * q_new / q_old)
    fixture_ok = fixtures(c)
    ok = (fixture_ok and c["gate|base runs equal to F9's stage record (message sha256)"] == len(src)
          and all(c[f"{g}|text_editor messages|go"] > 0 and c[f"{g}|runs changed"] > 0 for g in ("sim-attack", "sim-honest"))
          and all(c[f"{g}|content tabs left after F10 in text_editor messages"] == 0 for g in ("sim-attack", "sim-honest"))
          and all(c[f"{g}|runs where F10 is idempotent"] == c[f"{g}|runs changed"] for g in ("sim-attack", "sim-honest"))
          and all(c[f"{g}|staged rows holding attachment://"] == 0 for g in ("sim-attack", "sim-honest")))
    den = {a: sum(v for kk, v in c.items() if kk.startswith(f"{a}|text_editor messages|")) for a in REAL}
    real_ok = {a: den[a] > 0 and c[f"{a}|messages changed"] < 0.01 * den[a] for a in REAL}
    out = {"counts": dict(sorted(c.items())), "tokens": dict(sorted(tokens.items())), "gates pass": ok, "fixtures pass": fixture_ok,
           "positive control: real arm changes under 1% of its text_editor messages": real_ok,
           "runs by arm": {a: sorted(r["key"] for r in rs) for a, rs in sorted(staged.items())}}
    if mode == "stage":
        if not ok or not all(real_ok.values()):
            raise ValueError("a gate or the positive control failed: nothing is staged")
        dst = Path("/acts") / DST
        if dst.exists():
            raise FileExistsError(f"{DST} exists: a staged run is never overwritten")
        (dst / "inputs").mkdir(parents=True)
        shards = {}
        for n, (arm, rs) in enumerate(sorted(staged.items())):
            body = "".join(json.dumps(x) + "\n" for x in sorted(rs, key=lambda x: x["key"])).encode()
            (dst / "inputs" / f"shard_{n}.jsonl").write_bytes(body)
            shards[f"inputs/shard_{n}.jsonl"] = {"arm": arm, "runs": len(rs), "sha256": hashlib.sha256(body).hexdigest()}
        rec = json.dumps({"runs": record, "shards": shards, "base": F9SRC, "base record sha256": pins["f9 record"]}).encode()
        (dst / "stage_record.json").write_bytes(rec)
        acts.commit()
        out |= {"staged": DST, "shards": shards, "record_sha256": hashlib.sha256(rec).hexdigest()}
    return out


def filename(key: str) -> str:  # the extractors' file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def price(tok: dict, shards: dict) -> dict:
    q = sum(v for k, v in tok.items() if k.endswith("qwen3 tokens after F10"))
    s = sum(v for k, v in tok.items() if k.endswith("seed forward tokens after F10 (scaled)"))
    qr = q / 1e6 * PRICE["qwen3 $/1M forward tokens"] + shards["qwen3"] * PRICE["qwen3 $/model load"]
    sr = s / 1e6 * PRICE["seed $/1M forward tokens (F9 measured, loads included)"]
    return {"basis": PRICE, "shards": shards, "qwen3 forward tokens": q, "seed forward tokens": s,
            "qwen3 $ raw": round(qr, 2), "qwen3 $ with 30%": round(qr * 1.3, 2), "seed $ raw": round(sr, 2), "seed $ with 30%": round(sr * 1.3, 2),
            "total $ raw": round(qr + sr, 2), "total $ with 30%": round((qr + sr) * 1.3, 2)}


@app.function(image=image, volumes={"/acts": acts, "/g3": g3out, "/hf": hf_cache}, cpu=8, memory=65536, timeout=3600, retries=0)
def job(mode: str, pins: dict, real_keys: dict, seed_overrides: dict, qwen_rev: str) -> dict:
    try:
        return run_job(mode, pins, real_keys, seed_overrides, qwen_rev)
    except Exception as e:  # noqa: BLE001 — sanitized (no staged text crosses the boundary) and re-raised
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"stage failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}") from None


@app.local_entrypoint()
def main(mode: str = "survey"):
    if mode not in ("survey", "stage"):
        raise SystemExit("--mode survey|stage")
    for f in (Path(__file__), HERE / "PLAN_F10.md"):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
                or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    from probe_shift.models import QWEN3_32B
    pins = {"f9 record": json.loads(F9_STAGE.read_text())["record_sha256"]}
    runs = json.loads(RUNS36.read_text())
    real_keys = {a: sorted(r["key"] for r in runs if r["arm"] == a) for a in REAL}
    smap = json.loads(SEED_MAP.read_text())["arms"]
    seed_overrides = {"attack": smap["full + F9, attack"]["override keys"], "honest": smap["full + F9, honest"]["override keys"]}
    out = job.remote(mode, pins, real_keys, seed_overrides, QWEN3_32B["revision"])
    n = sum(len(v) for v in out["runs by arm"].values())
    out["price"] = price(out["tokens"], {"qwen3": 8, "seed": 8})  # both configs use 8 shards
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"stage_f10_{mode}_{stamp}.json"
    with open(p, "x") as fh:
        fh.write(json.dumps({"POST HOC": True, "status": "preliminary", "plan": "PLAN_F10.md", "git": head, "mode": mode,
                             "inputs sha256": {"#36 runs.json": hashlib.sha256(RUNS36.read_bytes()).hexdigest(),
                                               F9_STAGE.name: hashlib.sha256(F9_STAGE.read_bytes()).hexdigest(),
                                               SEED_MAP.name: hashlib.sha256(SEED_MAP.read_bytes()).hexdigest()}} | out, indent=1) + "\n")
    print(p)
    print("gates pass:", out["gates pass"], "| positive control:", out["positive control: real arm changes under 1% of its text_editor messages"])
    print({a: len(v) for a, v in out["runs by arm"].items()}, out["price"])
