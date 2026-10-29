"""RULE.md Addition 6, step D, POST HOC: the dose of each candidate tell in the probe's output rows. Per candidate and
simulated arm (attack 939, honest 437; full + F9 + F11 as staged, F12 on top for its own row): share of output rows changed
(F12, F10) or touched (detection proxies), share of runs with >= 1, mean such rows per affected run; whole run and the
first 21 replies. Rows = tool outputs whose output_end_mean position is kept inside the 32,768 cut, read from the metas of
the dump each run's latest staged version was extracted to (fxavg, F11 or F12), matched by (step, action). Modal CPU;
counts only (attack runs: scripted counts only, no text).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_q2_ffm_iter1/dose_count.py
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
FOLDER = "experiments/2026-10-14_q2_ffm_iter1"
if modal.is_local():
    sys.path.insert(0, str(REPO / "src"))
APP = "probe-shift_aswin-surya_sub-q2-ffm1_dose_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
V4 = "2026-10-06_la_segment-end-v4_qwen3-32b_v1"
FXAVG, F11D, F12D = "2026-10-13_q2_fxavg_qwen3-32b_l7_v1", "2026-10-13_q2_f11_qwen3-32b_l7_v1", "2026-10-14_q2_f12_qwen3-32b_l7_v1"
B7_SHA = "8bdaa55d0a395e5a36415bb001f1eff13c9eaabb9ff9f5c71e56bdd26e9e5042"
# side -> staged files (path, sha256, arm, runs): full + F9, F11 overlay, F12 overlay, F10 (on full + F9)
SRC = {"attack": {"base": ("2026-10-13_q2_fxavg_inputs_v1/inputs/shard_2.jsonl", "009466e7b8eb0f3f265a840157c800826ff459cb387f9824f05fabdd6f09fe55", "sim-attack-fx-fullF9", 939),
                  "F11": ("2026-10-13_q2_f11q_inputs_v1/inputs/shard_2.jsonl", "7a97837283ab161820ff78fbb87d645fc8753ab19a13aa694f77b0a78a7bdd6b", "sim-attack-fx-fullF9F11", 427),
                  "F12": ("2026-10-14_q2_f12q_inputs_v1/inputs/shard_2.jsonl", "e79ae7464a0d2db42f5b71a5d761603a70bd90335a2d83ca2cb7d3bd377b9b48", "sim-attack-fx-fullF9F11F12", 176),
                  "F10": ("2026-10-13_q2_fxavg_inputs_v1/inputs/shard_3.jsonl", "1d5cbdfb7de9bd7334e9fd9ddfa7e7966688f3f73885292f9a3fcd75618e8ffd", "sim-attack-fx-fullF9F10", 74)},
       "honest": {"base": ("2026-10-13_q2_fxavg_inputs_v1/inputs/shard_4.jsonl", "eaa15a3c136de18916ae66b4bfd28a9a9779f7d14e94dc9e83cb82e20c7bdd0c", "sim-honest-fx-fullF9", 437),
                  "F11": ("2026-10-13_q2_f11q_inputs_v1/inputs/shard_3.jsonl", "78445677dc32700b49ccd9f9d30e73fe096c895584ce5bce109db0b663df3e91", "sim-honest-fx-fullF9F11", 226),
                  "F12": ("2026-10-14_q2_f12q_inputs_v1/inputs/shard_3.jsonl", "08831846683866b31110597a3af94580823107205ea8f7d758998036eb8ec106", "sim-honest-fx-fullF9F11F12", 114),
                  "F10": ("2026-10-13_q2_fxavg_inputs_v1/inputs/shard_5.jsonl", "b50259e55e4e61cee38c068240cdb7d8ce9bf244863ee527f845dc423cf2fc24", "sim-honest-fx-fullF9F10", 36)}}
DUMP_OF = {"base": FXAVG, "F11": F11D, "F12": F12D}
FIRST, KIND, MIN_REAL, MIN_SHARE = 21, "output_end_mean", 20, 0.5
ANSI, BOX = re.compile(r"\x1b\["), re.compile("[─-╿]")
CANDIDATES = ("F12 CRLF (staged changes)", "F10 tabs (staged changes)", "blank-line (detection proxy)", "ANSI (detection proxy)",
              "box-drawing (detection proxy)")

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy==2.5.4", "scipy==1.18.1", "scikit-learn==1.9.1", "joblib==1.6.0", "boto3==1.43.113", "pyyaml==6.0.3")
         .env({"AUTHOR_SPLIT_STAGE": "/tmp/author_split_stage"})
         .add_local_python_source("probe_shift")
         .add_local_file(REPO / "splits" / "exclusions_v4.json", "/root/splits/exclusions_v4.json")
         .add_local_file(HERE / "analyze_POSTHOC_author_split.py", f"/root/{FOLDER}/analyze_POSTHOC_author_split.py")
         .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/rerender.py", "/root/swap/rerender.py")
         .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/str_replace_tells.py", "/root/swap/str_replace_tells.py")
         .add_local_file(REPO / "experiments/2026-10-09_fixed_sim/fixes.py", "/root/fsim/fixes.py"))


class Gate(Exception):
    """A failed check; its message holds counts only."""


def program(args: dict) -> str:
    """The first word of a bash command (after env assignments and cd ... &&), or '' if none."""
    c = str(args.get("cmd") or args.get("command") or "").strip()
    c = re.sub(r"^(?:cd\s+\S+\s*&&\s*)+", "", c)
    c = re.sub(r"^(?:\w+=\S*\s+)+", "", c)
    m = re.match(r"[\w./-]+", c)
    return m.group(0).rsplit("/", 1)[-1] if m else ""


def filename(key: str) -> str:
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes={"/acts": acts}, cpu=8, memory=32768, timeout=1800, retries=0)
def dose(git: str, stamp: str) -> dict:
    for p in ("/root/swap", "/root/fsim", f"/root/{FOLDER}"):
        sys.path.insert(0, p)
    import fixes

    if hashlib.sha256(Path(f"/root/{FOLDER}/analyze_POSTHOC_author_split.py").read_bytes()).hexdigest() != B7_SHA:
        raise Gate("b7's analysis module differs from its pin")
    import analyze_POSTHOC_author_split as A

    def pinned(spec):
        p, h, arm, n = spec
        raw = (Path("/acts") / p).read_bytes()
        if hashlib.sha256(raw).hexdigest() != h:
            raise Gate(f"{p}: sha256 differs from its pin")
        rows = {r["key"]: r for r in map(json.loads, raw.decode().splitlines())}
        if len(rows) != n or {r["arm"] for r in rows.values()} != {arm}:
            raise Gate(f"{p}: not {n} runs of {arm}")
        return rows

    # REAL1 programs whose outputs mostly end in a blank line
    _, first, _, _ = A.cohort_rows()
    want = {r["key"] for r in first.values()}
    real1 = {}
    for p in sorted((Path("/acts") / V4 / "inputs").glob("shard_*.jsonl")):
        for line in open(p):
            it = json.loads(line)
            if it["arm"] == "real-honest" and it["key"] in want:
                real1[it["key"]] = it["messages"]
    if set(real1) != want:
        raise Gate("REAL1 runs missing")
    tot, dbl = Counter(), Counter()
    for m in real1.values():
        for x in fixes.calls_of(m):
            t = m[x["msg"]].get("content")
            if x["tool"].endswith("bash") and isinstance(t, str):  # all of a program's outputs (Codex r2 required fix 1)
                pr = program(x["args"])
                tot[pr] += 1
                dbl[pr] += t.endswith("\n\n")
    blank_programs = sorted(pr for pr in tot if pr and tot[pr] >= MIN_REAL and dbl[pr] / tot[pr] >= MIN_SHARE)

    out, c = {}, Counter()
    for side, specs in SRC.items():
        S = {k: pinned(v) for k, v in specs.items()}
        f12_changed = {}
        for k_, r in S["F12"].items():
            base = (S["F11"].get(k_) or S["base"][k_])["messages"]
            f12_changed[k_] = {i for i, (a, b) in enumerate(zip(base, r["messages"])) if a != b}
        f10_changed = {k_: {i for i, (a, b) in enumerate(zip(S["base"][k_]["messages"], r["messages"])) if a != b} for k_, r in S["F10"].items()}
        acc = {cand: {w: Counter() for w in ("whole run", f"first {FIRST} replies")} for cand in CANDIDATES}
        for k_ in sorted(S["base"]):
            # Codex r2 required fix 2: F12 is measured on its own overlay; every other candidate on full + F9 + F11
            ver1 = "F11" if k_ in S["F11"] else "base"
            ver2 = "F12" if k_ in S["F12"] else ver1
            rows_of = {}
            for ver in {ver1, ver2}:
                vm = S[ver][k_]["messages"]
                meta = json.loads((Path("/acts") / DUMP_OF[ver] / "meta" / specs[ver][2] / f"{filename(k_)}.json").read_text())
                kept = {(p["step"], p["action"]) for p in meta["positions"] if p["kind"] == KIND and p["kept"]}
                rm = {x["msg"] for x in fixes.calls_of(vm) if (x["step"], x["action"]) in kept}
                c[f"{side}|kept positions not matched to an output"] += len(kept) - len(rm)
                rows_of[ver] = rm
            msgs = S[ver1][k_]["messages"]
            calls = [x for x in fixes.calls_of(msgs)]
            tool_order = [i for i, mm in enumerate(msgs) if mm["role"] == "tool"]
            first21 = set(tool_order[:FIRST])
            c[f"{side}|outputs"] += len(tool_order)
            c[f"{side}|outputs with a kept row (full + F9 + F11)"] += len(rows_of[ver1])
            touched = {"F12 CRLF (staged changes)": f12_changed.get(k_, set()), "F10 tabs (staged changes)": f10_changed.get(k_, set())}
            bl, an, bx = set(), set(), set()
            for x in calls:
                t = msgs[x["msg"]].get("content")
                if not isinstance(t, str):
                    continue
                if x["tool"].endswith("bash") and t.endswith("\n") and not t.endswith("\n\n") and program(x["args"]) in blank_programs:
                    bl.add(x["msg"])
                if ANSI.search(t):
                    an.add(x["msg"])
                if BOX.search(t):
                    bx.add(x["msg"])
            touched |= {"blank-line (detection proxy)": bl, "ANSI (detection proxy)": an, "box-drawing (detection proxy)": bx}
            for cand, ms in touched.items():
                for w, win in (("whole run", None), (f"first {FIRST} replies", first21)):
                    rm = rows_of[ver2] if cand.startswith("F12") else rows_of[ver1]
                    rows = rm if win is None else rm & win
                    hit = len(rows & ms)
                    a = acc[cand][w]
                    a["rows"] += len(rows)
                    a["rows touched"] += hit
                    a["runs"] += 1
                    a["runs with >= 1"] += hit > 0
        out[side] = {cand: {w: {"rows": a["rows"], "rows touched": a["rows touched"],
                                "share of rows": a["rows touched"] / a["rows"] if a["rows"] else None,
                                "runs": a["runs"], "runs with >= 1": a["runs with >= 1"],
                                "share of runs": a["runs with >= 1"] / a["runs"],
                                "mean rows per affected run": a["rows touched"] / a["runs with >= 1"] if a["runs with >= 1"] else None}
                            for w, a in v.items()} for cand, v in acc.items()}
    if any(v for k_, v in c.items() if k_.endswith("not matched to an output")):
        raise Gate(f"kept positions not matched to outputs: {dict(c)}")
    return {"POST HOC": True, "status": "preliminary", "rule": "RULE.md Addition 6, step D", "git": git, "stamp": stamp,
            "row unit": f"tool outputs with a kept {KIND} position (cut 32,768), matched by (step, action)",
            "blank-line programs (REAL1 >= 20 outputs, >= 50% end in a blank line)": {pr: [tot[pr], dbl[pr]] for pr in blank_programs},
            "counts": dict(c), "dose": out}


@app.local_entrypoint()
def main():
    for f in (Path(__file__), HERE / "RULE.md", HERE / "analyze_POSTHOC_author_split.py"):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
                or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = dose.remote(git, stamp)
    p = HERE / "results" / f"dose_count_{stamp}.json"
    with open(p, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print("wrote", p)
