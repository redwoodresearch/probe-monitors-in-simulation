"""RULE.md Amendment 2: the replies split rerun with a broadened time mask (PRIMARY) and that mask plus scrub level 2
(SECONDARY), on Modal CPU. One container per (variant, comparison), plus one for the mask counts; each writes its own
shard file on the Volume, and the orchestrator combines them into one result JSON per variant. The analysis functions
(classify, marker_control, the rows and cohorts) are analyze_POSTHOC_author_split.py's, unchanged.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_claimB_author_split/run_replies_broad_modal.py
"""

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
FOLDER = "experiments/2026-10-13_claimB_author_split"
if modal.is_local():  # add_local_python_source needs probe_shift importable locally
    sys.path.insert(0, str(REPO / "src"))
APP = "probe-shift_aswin-surya_sub-claimb-author-split_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
V4 = "2026-10-06_la_segment-end-v4_qwen3-32b_v1"
FX = "2026-10-13_q2_fixfull_inputs_v2/inputs/shard_12.jsonl"
OUT = "2026-10-13_claimB_author_split_v1/results"
MON = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
BROAD = [  # (name, pattern, placeholder), applied in this order after the level-1 scrub (Amendment 2)
    ("compact stamps", r"\b20\d{6}(?:T?\d{4,6})?\b", "<date>"),
    ("day-month dates", rf"\b\d{{1,2}}\s+(?:{MON})[a-z]*\b", "<date>"),
    (
        "relative times",
        r"\b\d+\s+(?:second|minute|hour|day|week|month|year)s?\s+ago\b",
        "<ago>",
    ),
    ("durations m-s", r"\b\d+m\d+(?:\.\d+)?s\b", "<dur>"),
    (
        "durations units",
        r"\b\d+(?:\.\d+)?\s?(?:ms|s|sec|secs|seconds|min|mins|minutes|h|hrs|hours)\b",
        "<dur>",
    ),
    (
        "time zone names",
        r"\b(?:UTC|GMT|PDT|PST|EDT|EST|CDT|CST|MDT|MST|CET|CEST|BST|IST|JST)\b",
        "<tz>",
    ),
    ("time zone offsets", r"(?<![\w+-])[+-]\d{4}\b", "<tz>"),
    (
        "weekday names",
        r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b",
        "<wday>",
    ),
    (
        "month names",
        (
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|"
            r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b"
        ),
        "<month>",
    ),
    ("standalone years", r"\b(?:19|20)\d{2}\b", "<year>"),
]
BROAD = [(n, re.compile(p), r) for n, p, r in BROAD]
VARIANTS = ("broad", "broad_l2")
COMPARISONS = ("fake_vs_real", "control", "positive_control")

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.5.4",
        "scipy==1.18.1",
        "scikit-learn==1.9.1",
        "joblib==1.6.0",
        "boto3==1.43.113",
        "pyyaml==6.0.3",
    )
    .env({"AUTHOR_SPLIT_STAGE": "/tmp/author_split_stage"})
    .add_local_python_source("probe_shift")
    .add_local_file(
        REPO / "splits" / "exclusions_v4.json", "/root/splits/exclusions_v4.json"
    )
    .add_local_file(
        HERE / "analyze_POSTHOC_author_split.py",
        f"/root/{FOLDER}/analyze_POSTHOC_author_split.py",
    )
)


def say(*a):
    print(
        datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S"),
        *a,
        flush=True,
    )


def broad(text: str, counts=None) -> str:
    for name, rx, rep in BROAD:
        if counts is not None:
            spans = [len(m.group(0)) for m in rx.finditer(text)]
            if spans:
                counts[name]["replies"] += 1
                counts[name]["chars"] += sum(spans)
        text = rx.sub(rep, text)
    return text


def load():
    """The rows, exactly as run_author_split_modal.run_split reads them (sha-checked fix-full; v4 inputs)."""
    sys.path.insert(0, f"/root/{FOLDER}")
    import hashlib

    import analyze_POSTHOC_author_split as A

    raw = (Path("/acts") / FX).read_bytes()
    if hashlib.sha256(raw).hexdigest() != A.FX_SHA:
        raise SystemExit("the fix-full shard's sha256 does not match the stage record")
    fx = {r["key"]: r for r in map(json.loads, raw.decode().splitlines())}
    sim, first, second, ex_sha = A.cohort_rows()
    if set(fx) != set(sim):
        raise SystemExit("the fix-full rows are not the 437 simulated honest runs")
    want = {("real-honest", r["key"]) for r in [*first.values(), *second.values()]}
    rows = {}
    for p in sorted((Path("/acts") / V4 / "inputs").glob("shard_*.jsonl")):
        for line in p.read_text().splitlines():
            it = json.loads(line)
            if (it["arm"], it["key"]) in want:
                if (it["arm"], it["key"]) in rows:
                    raise SystemExit("a run appears twice in the v4 inputs")
                rows[(it["arm"], it["key"])] = it["messages"]
    if set(rows) != want:
        raise SystemExit("wanted real runs missing from the v4 inputs")
    real = {k: m for (_, k), m in rows.items()}
    return A, sim, first, second, ex_sha, fx, real


def replies_of(A, msgs, variant, counts=None):
    """The replies split's document under a variant: level 1, the broadened mask, and for broad_l2 then level 2."""
    outs = [m["content"] for m in msgs if m["role"] == "tool"]
    if not all(isinstance(t, str) for t in outs):
        raise SystemExit("a tool message's content is not a string")
    docs = [broad(A.scrub(t, 1), counts) for t in outs]
    if variant == "broad_l2":
        docs = [A.scrub(t, 2) for t in docs]
    return A.SEP.join(docs)


@app.function(
    image=image,
    volumes={"/acts": acts},
    cpu=32,
    memory=98304,
    timeout=30 * 60,
    retries=0,
)
def cell(variant: str, comparison: str, stamp: str) -> dict:
    A, sim, first, second, _ex_sha, fx, real = load()
    say(variant, comparison, "rows loaded")
    tasks = sorted(first)
    key_of = {r["cell"]: k for k, r in sim.items()}
    real_docs = {t: replies_of(A, real[first[t]["key"]], variant) for t in tasks}
    if comparison == "fake_vs_real":
        fake = [replies_of(A, fx[key_of[t]]["messages"], variant) for t in tasks]
        res = A.classify(
            fake, [real_docs[t] for t in tasks], A.np.array(tasks), ("fix_full", "real")
        )
    elif comparison == "control":
        ct = sorted(second)
        res = A.classify(
            [replies_of(A, real[second[t]["key"]], variant) for t in ct],
            [real_docs[t] for t in ct],
            A.np.array(ct),
            ("real_second", "real_first"),
        )
    else:
        res = A.marker_control(real_docs, tasks, first)
        res["pass"] = bool(res["pass"])
    say(variant, comparison, {k: res[k] for k in ("auroc", "null_p95", "p")})
    out = {"variant": variant, "comparison": comparison, "result": res}
    dst = (
        Path("/acts") / OUT / "shards" / f"replies_{variant}_{comparison}_{stamp}.json"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    acts.commit()
    return out


@app.function(
    image=image,
    volumes={"/acts": acts},
    cpu=8,
    memory=32768,
    timeout=30 * 60,
    retries=0,
)
def mask_counts(stamp: str) -> dict:
    from collections import Counter

    A, _sim, first, second, _ex_sha, fx, real = load()
    arms = {
        "fix_full": [fx[k]["messages"] for k in sorted(fx)],
        "real_first": [real[first[t]["key"]] for t in sorted(first)],
        "real_second": [real[second[t]["key"]] for t in sorted(second)],
    }
    out = {}
    for arm, runs in arms.items():
        c = {n: Counter() for n, _, _ in BROAD}
        for m in runs:
            replies_of(A, m, "broad", c)
        out[arm] = {
            "runs": len(runs),
            "replies": sum(m["role"] == "tool" for r in runs for m in r),
            "rules": {n: dict(v) for n, v in c.items()},
        }
    say(
        "mask counts",
        json.dumps(
            {
                a: {n: v["replies"] for n, v in o["rules"].items()}
                for a, o in out.items()
            }
        ),
    )
    dst = Path("/acts") / OUT / "shards" / f"replies_mask_counts_{stamp}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    acts.commit()
    return out


@app.function(
    image=image,
    volumes={"/acts": acts},
    cpu=0.25,
    memory=1024,
    timeout=45 * 60,
    retries=0,
)
def orchestrate(git: str, stamp: str) -> list[str]:
    import hashlib

    jobs = [(v, c, stamp) for v in VARIANTS for c in COMPARISONS]
    counts = mask_counts.spawn(stamp)
    cells = list(cell.starmap(jobs))
    counts = counts.get()
    paths = []
    for v in VARIANTS:
        got = {x["comparison"]: x["result"] for x in cells if x["variant"] == v}
        out = {
            "label": "POST HOC, descriptive, preliminary until reviewed",
            "rule": "RULE.md Amendment 2 (this folder)",
            "split": "replies",
            "variant": {
                "broad": "PRIMARY: level 1 + broadened time mask",
                "broad_l2": "SECONDARY: broadened time mask + scrub level 2",
            }[v],
            "git": git,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "broad_rules": [(n, rx.pattern, r) for n, rx, r in BROAD],
            "mask_counts_broad_rules": counts,
            **got,
            "feature_lists_readable": got["positive_control"]["pass"],
            "baseline_level1": "results/author_split_replies_POSTHOC_2026-10-13_173448.json (AUROC 0.7414)",
        }
        dst = Path("/acts") / OUT / f"author_split_replies_{v}_POSTHOC_{stamp}.json"
        with open(dst, "x") as fh:
            fh.write(json.dumps(out, indent=1) + "\n")
        paths.append(str(dst))
    acts.commit()
    return paths


@app.local_entrypoint()
def main():
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=HERE,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", FOLDER],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO,
    ).stdout
    if dirty.strip():
        raise SystemExit("commit the experiment folder before launching")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(orchestrate.remote(git, stamp))
