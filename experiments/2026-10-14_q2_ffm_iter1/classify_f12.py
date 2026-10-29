"""RULE.md (c) 1, POST HOC, descriptive: b7's author-split text classifier on simulated honest replies before and after
F12. b7's run_replies_broad_modal.py (aswin/claimb-author-split-classifier @ ea0e8f31 = 54cbb57e, its broad result's commit)
with only the fake arm swapped; variant `broad` (b7's primary: scrub level 1 + the broadened time mask); the analysis
functions are analyze_POSTHOC_author_split.py's, byte-identical (sha256 pinned). One Modal CPU container per comparison:
  repro_b7       b7's own fake arm (fix-full F1-F8, shard_12) vs REAL1: must reproduce b7's 0.7358 / null p95 0.5302 (gate);
  before         BASE = full + F9 + F11 simulated honest vs REAL1;
  after          BASE + F12 (the 114 F12-changed runs replacing theirs) vs REAL1;
  control        REAL2 vs REAL1 (reads no fake run; unchanged by F12);
  positive_control  b7's marker control on REAL1 (reads no fake run).
Numbers and n-grams of at most 5 characters from honest runs only, as b7's results hold; no reply text.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_q2_ffm_iter1/classify_f12.py
"""
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
FOLDER = "experiments/2026-10-14_q2_ffm_iter1"
if modal.is_local():
    sys.path.insert(0, str(REPO / "src"))
APP = "probe-shift_aswin-surya_sub-q2-ffm1_classify_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
V4 = "2026-10-06_la_segment-end-v4_qwen3-32b_v1"
FX_B7 = "2026-10-13_q2_fixfull_inputs_v2/inputs/shard_12.jsonl"  # b7's fake arm; its sha256 is A.FX_SHA
BASE_SRC = [("2026-10-13_q2_fxavg_inputs_v1/inputs/shard_4.jsonl", "eaa15a3c136de18916ae66b4bfd28a9a9779f7d14e94dc9e83cb82e20c7bdd0c", "sim-honest-fx-fullF9", 437),
            ("2026-10-13_q2_f11q_inputs_v1/inputs/shard_3.jsonl", "78445677dc32700b49ccd9f9d30e73fe096c895584ce5bce109db0b663df3e91", "sim-honest-fx-fullF9F11", 226)]
F12_SRC = ("2026-10-14_q2_f12q_inputs_v1/inputs/shard_3.jsonl", "08831846683866b31110597a3af94580823107205ea8f7d758998036eb8ec106", "sim-honest-fx-fullF9F11F12", 114)
B7_SHA = "8bdaa55d0a395e5a36415bb001f1eff13c9eaabb9ff9f5c71e56bdd26e9e5042"
B7_BROAD = {"auroc": 0.7358, "null_p95": 0.5302}  # author_split_replies_broad_POSTHOC_2026-10-13_175920.json fake_vs_real (ea0e8f31)
OUT = "2026-10-14_q2_ffm_iter1/results"
MON = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
BROAD = [  # b7's, verbatim: (name, pattern, placeholder), applied in this order after the level-1 scrub
    ("compact stamps", r"\b20\d{6}(?:T?\d{4,6})?\b", "<date>"),
    ("day-month dates", rf"\b\d{{1,2}}\s+(?:{MON})[a-z]*\b", "<date>"),
    ("relative times", r"\b\d+\s+(?:second|minute|hour|day|week|month|year)s?\s+ago\b", "<ago>"),
    ("durations m-s", r"\b\d+m\d+(?:\.\d+)?s\b", "<dur>"),
    ("durations units", r"\b\d+(?:\.\d+)?\s?(?:ms|s|sec|secs|seconds|min|mins|minutes|h|hrs|hours)\b", "<dur>"),
    ("time zone names", r"\b(?:UTC|GMT|PDT|PST|EDT|EST|CDT|CST|MDT|MST|CET|CEST|BST|IST|JST)\b", "<tz>"),
    ("time zone offsets", r"(?<![\w+-])[+-]\d{4}\b", "<tz>"),
    ("weekday names", r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b", "<wday>"),
    ("month names", (r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|"
                     r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b"), "<month>"),
    ("standalone years", r"\b(?:19|20)\d{2}\b", "<year>"),
]
BROAD = [(n, re.compile(p), r) for n, p, r in BROAD]
COMPARISONS = ("repro_b7", "before", "after", "control", "positive_control")

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy==2.5.4", "scipy==1.18.1", "scikit-learn==1.9.1", "joblib==1.6.0", "boto3==1.43.113", "pyyaml==6.0.3")
         .env({"AUTHOR_SPLIT_STAGE": "/tmp/author_split_stage"})
         .add_local_python_source("probe_shift")
         .add_local_file(REPO / "splits" / "exclusions_v4.json", "/root/splits/exclusions_v4.json")
         .add_local_file(HERE / "analyze_POSTHOC_author_split.py", f"/root/{FOLDER}/analyze_POSTHOC_author_split.py"))


def say(*a):
    print(datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S"), *a, flush=True)


def broad(text: str) -> str:
    for _, rx, rep in BROAD:
        text = rx.sub(rep, text)
    return text


def replies_of(A, msgs):
    """b7's replies document under `broad`: level 1, then the broadened mask, joined by SEP."""
    outs = [m["content"] for m in msgs if m["role"] == "tool"]
    if not all(isinstance(t, str) for t in outs):
        raise SystemExit("a tool message's content is not a string")
    return A.SEP.join(broad(A.scrub(t, 1)) for t in outs)


def load():
    sys.path.insert(0, f"/root/{FOLDER}")
    if hashlib.sha256(Path(f"/root/{FOLDER}/analyze_POSTHOC_author_split.py").read_bytes()).hexdigest() != B7_SHA:
        raise SystemExit("b7's analysis module differs from its pin")
    import analyze_POSTHOC_author_split as A

    def pinned(p, h, arm, n):
        raw = (Path("/acts") / p).read_bytes()
        if hashlib.sha256(raw).hexdigest() != h:
            raise SystemExit(f"{p}: sha256 differs from its pin")
        rows = {r["key"]: r for r in map(json.loads, raw.decode().splitlines())}
        if len(rows) != n or {r["arm"] for r in rows.values()} != {arm}:
            raise SystemExit(f"{p}: not {n} runs of {arm}")
        return {k: r["messages"] for k, r in rows.items()}

    raw = (Path("/acts") / FX_B7).read_bytes()
    if hashlib.sha256(raw).hexdigest() != A.FX_SHA:
        raise SystemExit("b7's fix-full shard differs from its stage record")
    fxb7 = {r["key"]: r["messages"] for r in map(json.loads, raw.decode().splitlines())}
    base = pinned(*BASE_SRC[0]) | pinned(*BASE_SRC[1])
    after = base | pinned(*F12_SRC)
    sim, first, second, _ = A.cohort_rows()
    if not (set(fxb7) == set(base) == set(after) == set(sim)):
        raise SystemExit("a fake arm is not the 437 simulated honest runs")
    want = {r["key"] for r in [*first.values(), *second.values()]}
    real = {}
    for p in sorted((Path("/acts") / V4 / "inputs").glob("shard_*.jsonl")):
        for line in open(p):
            it = json.loads(line)
            if it["arm"] == "real-honest" and it["key"] in want:
                if it["key"] in real:
                    raise SystemExit("a run appears twice in the v4 inputs")
                real[it["key"]] = it["messages"]
    if set(real) != want:
        raise SystemExit("wanted real runs missing from the v4 inputs")
    return A, sim, first, second, {"repro_b7": fxb7, "before": base, "after": after}, real


@app.function(image=image, volumes={"/acts": acts}, cpu=32, memory=98304, timeout=45 * 60, retries=0)
def cell(comparison: str, stamp: str) -> dict:
    A, sim, first, second, fakes, real = load()
    tasks = sorted(first)
    key_of = {r["cell"]: k for k, r in sim.items()}
    real_docs = {t: replies_of(A, real[first[t]["key"]]) for t in tasks}
    if comparison in fakes:
        names = ("fix_full", "real") if comparison == "repro_b7" else (f"sim_{comparison}", "real")
        res = A.classify([replies_of(A, fakes[comparison][key_of[t]]) for t in tasks], [real_docs[t] for t in tasks], A.np.array(tasks), names)
    elif comparison == "control":
        ct = sorted(second)
        res = A.classify([replies_of(A, real[second[t]["key"]]) for t in ct], [real_docs[t] for t in ct], A.np.array(ct), ("real_second", "real_first"))
    else:
        res = A.marker_control(real_docs, tasks, first)
        res["pass"] = bool(res["pass"])
    say(comparison, {k: res.get(k) for k in ("auroc", "null_p95", "p")})
    out = {"comparison": comparison, "result": res}
    dst = Path("/acts") / OUT / "shards" / f"classify_{comparison}_{stamp}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    acts.commit()
    return out


@app.function(image=image, volumes={"/acts": acts}, cpu=0.25, memory=1024, timeout=60 * 60, retries=0)
def orchestrate(git: str, stamp: str) -> dict:
    got = {x["comparison"]: x["result"] for x in cell.starmap([(c, stamp) for c in COMPARISONS])}
    r = got["repro_b7"]
    gate = {"repro_b7 auroc": r["auroc"], "repro_b7 null_p95": r["null_p95"], "b7's": B7_BROAD,
            "pass": r["auroc"] == B7_BROAD["auroc"] and r["null_p95"] == B7_BROAD["null_p95"]}
    if not gate["pass"]:  # Codex r1 required fix 3: nothing is reported without the reproduction
        raise SystemExit(f"b7 reproduction failed: {gate}")
    out = {"label": "POST HOC, descriptive, preliminary until reviewed", "rule": "RULE.md (c) 1 (this folder)", "split": "replies",
           "variant": "broad (b7's primary)", "git": git, "stamp": stamp, "reproduction gate": gate,
           "readings": {"before (full + F9 + F11) vs REAL1": {k: got["before"][k] for k in ("auroc", "null_p95", "p")},
                        "after (+ F12) vs REAL1": {k: got["after"][k] for k in ("auroc", "null_p95", "p")},
                        "delta AUROC (after - before)": round(got["after"]["auroc"] - got["before"]["auroc"], 4),
                        "control REAL2 vs REAL1": {k: got["control"][k] for k in ("auroc", "null_p95", "p")},
                        "marker control pass": got["positive_control"]["pass"]},
           **got}
    dst = Path("/acts") / OUT / f"classify_f12_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    acts.commit()
    return out


@app.local_entrypoint()
def main():
    for f in (Path(__file__), HERE / "RULE.md", HERE / "analyze_POSTHOC_author_split.py"):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
                or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = orchestrate.remote(git, stamp)
    p = HERE / "results" / f"classify_f12_POSTHOC_{stamp}.json"
    with open(p, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print("wrote", p, json.dumps(out["reproduction gate"]), json.dumps(out["readings"]))
