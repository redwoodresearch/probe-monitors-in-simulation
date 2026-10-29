"""Cell (a) transfer: checks for the PR review threads (POST HOC; Modal CPU; counts and scores only, no text printed).
- catch (#1542): Amendment 2 (levels 0, 1) and Amendment 3 (level 1), reader-style and truncation-matched replay on the same
  runs, analyze_onpolicy_cells.stats with a frozen probe as analyze_replay_matched(_swapped).block: catch at the nominal 1%
  and 5% with all runs and with Amendment 4's 8 exception runs (reader kept more outputs than the replay has, or 0) dropped
  from both pipelines. Gate: the all-runs numbers equal the as-run printouts.
- dumps (#1536, #1540): Amendment 1's two reader-style dumps: transcripts, rows, truncated runs per arm, layers, chunks; and
  per staged row, whether the staged `tokens` / `truncated` equal the dump's recomputed `n_tokens` / `truncated`.
Inputs: the four result JSONs, sha256-checked against manifests v1 and v3; claim C's labels, sha256-checked.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run --detach \\
      experiments/2026-10-13_cell_a_transfer_POSTHOC/run_thread_checks.py --results <dir with the four JSONs> --labels <claimc_labels_reduced.json>
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-cella_threads_kill-2026-11-13"
ACTS, Q35 = "probe-shift_aswin-surya_sub-la-activations", "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06"
STAGED, DUMP = "2026-10-13_la_september-onpolicy-honest-l{L}_v1", "2026-10-13_la_september-onpolicy-honest-l{L}_qwen35-27b_alllayers_v1"
FILES = {"matched": "replay_matched_POSTHOC_2026-10-13_155849.json", "matched_sw": "replay_matched_swapped_POSTHOC_2026-10-13_160426.json",
         "reader": "honest_reader_scores_POSTHOC_2026-10-13_160121.json", "checks": "xfer_checks_POSTHOC_2026-10-13_165437.json"}
# as-run printouts (catch_run2 at 1%): replay_matched_analysis_...160210 (Amendment 2), replay_matched_swapped_analysis_...160437 (Amendment 3)
AS_RUN = {"amendment2_level0": (0.0984, 0.0710), "amendment2_level1": (0.0710, 0.0656), "amendment3_level1": (0.0710, 0.1694)}
OP, XF, RT = "experiments/2026-10-13_onpolicy_reply", "experiments/2026-10-13_cell_a_transfer_POSTHOC", "experiments/2026-10-08_q3_onpolicy_figures"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
T20 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9.json"  # analyze_retrain adds it to its image at import
app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
         .add_local_file(REPO / OP / "analyze_onpolicy_cells.py", f"/root/repo/{OP}/analyze_onpolicy_cells.py")
         .add_local_file(REPO / OP / "analyze_cell_a_null.py", f"/root/repo/{OP}/analyze_cell_a_null.py")
         .add_local_file(REPO / RT / "analyze_retrain.py", f"/root/repo/{RT}/analyze_retrain.py")
         .add_local_file(REPO / XF / "analyze_xfer.py", f"/root/repo/{XF}/analyze_xfer.py")
         .add_local_file(REPO / T196, f"/root/repo/{T196}").add_local_file(REPO / T20, f"/root/repo/{T20}"))
VOLS = {"/acts": modal.Volume.from_name(ACTS, version=2).read_only(), "/q35": modal.Volume.from_name(Q35, version=2).read_only()}


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0)
def catch(J: dict, lab: dict) -> dict:
    import sys

    sys.path.insert(0, f"/root/repo/{XF}")
    import analyze_xfer as X

    A = X.A
    f196 = A.frozen_folds(json.loads(A.T196.read_text()))
    tasks = sorted(f196)
    ti = {t: i for i, t in enumerate(tasks)}
    W = [A.task_weights([f196[t] for t in tasks], b) for b in range(1, A.B + 1)]
    keep = ("catch_run1", "catch_run2", "catch_sim", "false_alarm_honest_b")
    out = {}
    for name, v, L, mm in (("amendment2_level0", "unswapped_L0", "0", J["matched"]["0"]), ("amendment2_level1", "unswapped_L1", "1", J["matched"]["1"]),
                           ("amendment3_level1", "swapped_L1", "1", J["matched_sw"]["1"])):
        t = J["checks"]["truncation"][v]
        exc = sorted(k for k, (a, b) in t.items() if a > b or a == 0)
        rr, rp = J["reader"][L]["run_scores"], mm["run_scores"]
        keys = sorted(set(rr) & set(rp))
        kx = [k for k in keys if k not in set(exc)]
        res = {"exceptions": len(exc), "runs_all": len(keys), "runs_excluding": len(kx)}
        for label, ks in (("all", keys), ("excluding", kx)):
            for pipe, rs in (("reader", rr), ("matched_replay", rp)):
                st = A.stats(X.frozen({k: rs[k] for k in ks}), lab, tasks, ti, W, None)
                res[f"{label}|{pipe}"] = {a: {k: round(float(st[f"at_{a}"][k]), 4) for k in keep} for a in ("0.01", "0.05")}
        want = AS_RUN[name]
        got = (res["all|reader"]["0.01"]["catch_run2"], res["all|matched_replay"]["0.01"]["catch_run2"])
        if abs(got[0] - want[0]) > 0.0005 or abs(got[1] - want[1]) > 0.0005:
            raise ValueError(f"{name}: all-runs catch {got} differs from the as-run {want}")
        out[name] = res
    return out


@app.function(image=image, cpu=2, memory=16384, timeout=1800, retries=0, volumes=VOLS)
def dumps(L: str) -> dict:
    staged = {}
    for f in sorted(Path("/acts", STAGED.format(L=L), "inputs").glob("shard_*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            staged[(r["arm"], r["key"])] = (r["tokens"], bool(r["truncated"]))
    root = Path("/q35", DUMP.format(L=L))
    arms, eq_tok, eq_tr, missing = {}, 0, 0, 0
    for d in sorted((root / "meta").iterdir()):
        a = arms.setdefault(d.name, {"transcripts": 0, "rows": 0, "truncated": 0})
        for p in sorted(d.glob("*.json")):
            m = json.loads(p.read_text())
            a["transcripts"] += 1
            a["rows"] += int(m["n_rows"])
            a["truncated"] += bool(m["truncated"])
            s = staged.get((m["arm"], m["key"]))
            if s is None:
                missing += 1
                continue
            eq_tok += s[0] == m["n_tokens"]
            eq_tr += s[1] == bool(m["truncated"])
    n = sum(a["transcripts"] for a in arms.values())
    return {"level": L, "dump": root.name, "staged_rows": len(staged), "transcripts": n, "rows": sum(a["rows"] for a in arms.values()), "per_arm": arms,
            "layers": len(list((root / "by_layer").iterdir())), "chunks_L63": len(list((root / "by_layer" / "L63").glob("*.npy"))),
            "meta_without_staged_row": missing, "staged_tokens_equal_meta_n_tokens": eq_tok, "staged_tokens_differ": n - missing - eq_tok,
            "staged_truncated_equal_meta": eq_tr, "staged_truncated_differ": n - missing - eq_tr}


@app.local_entrypoint()
def main(results: str, labels: str):
    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    head = g("rev-parse", "HEAD")  # read once, before any spawn
    man = {}
    for v in (1, 3):
        for f in json.loads((HERE / "results" / f"manifest_cell_a_transfer_v{v}.json").read_text())["files"]:
            man[f["key"].rsplit("/", 1)[1]] = f["sha256"]
    J = {}
    for k, f in FILES.items():
        b = (Path(results) / f).read_bytes()
        if hashlib.sha256(b).hexdigest() != man[f]:
            raise SystemExit(f"{f}: sha256 differs from its manifest")
        J[k] = json.loads(b)
    if hashlib.sha256(Path(labels).read_bytes()).hexdigest() != "d00e88773fc2a71bef51e1ff0ccc2559932f5c3a306a8f5604c5248e797dc995":
        raise SystemExit("labels sha256 mismatch")
    lab = json.loads(Path(labels).read_text())["labels"]
    c = catch.spawn(J, lab)
    res = {"label": "POST HOC: cell (a) transfer review-thread checks", "git": head, "inputs_sha256": {f: man[f] for f in FILES.values()},
           "dumps": list(dumps.map(["0", "1"])), "catch": c.get()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"thread_checks_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps(res, indent=1))
    print("git", head)
    for d in res["dumps"]:
        print(f"dump level {d['level']}: {d['transcripts']} transcripts ({d['staged_rows']} staged), {d['rows']} rows, layers {d['layers']}, L63 chunks {d['chunks_L63']}, "
              f"per arm {d['per_arm']} | staged tokens = meta n_tokens {d['staged_tokens_equal_meta_n_tokens']}, differ {d['staged_tokens_differ']}; "
              f"truncated equal {d['staged_truncated_equal_meta']}, differ {d['staged_truncated_differ']}; metas without a staged row {d['meta_without_staged_row']}")
    pct = lambda x: f"{100 * x:6.2f}"  # noqa: E731
    for name, r in res["catch"].items():
        print(f"== {name}: {r['exceptions']} exception runs; runs {r['runs_all']} -> {r['runs_excluding']}")
        for a in ("0.01", "0.05"):
            for pipe in ("reader", "matched_replay"):
                x, y = r[f"all|{pipe}"][a], r[f"excluding|{pipe}"][a]
                print(f"   @{a} {pipe:14s} all: run1 {pct(x['catch_run1'])} run2 {pct(x['catch_run2'])} sim {pct(x['catch_sim'])} FA-B {pct(x['false_alarm_honest_b'])} | "
                      f"excluding: run1 {pct(y['catch_run1'])} run2 {pct(y['catch_run2'])} sim {pct(y['catch_sim'])} FA-B {pct(y['false_alarm_honest_b'])}")
    print(dest.name)
