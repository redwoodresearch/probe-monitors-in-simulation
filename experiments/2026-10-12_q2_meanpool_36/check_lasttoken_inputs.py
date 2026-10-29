"""Review check for #1101 (Codex rr2If): the committed reading (meanpool_reading_2026-10-12_160554.json) hashed the
mean-pooled scores but not the last-token ones (#36's run_scores_2026-10-06_181548). This (1) checks the last-token dir's
runs.json, scores_L48.npz and scores_L63.npz against the sha256 values fix939 recorded for #36's scores before this reading
existed, and all seven layers' scores_L*.npz against the per-layer sha256 #36's null-selection run recorded on 10-07
(results/null_selection_2026-10-07_102220.json, selection.saved_scores.files_sha256, base_run 2026-10-06_181548); (2) re-runs analyze_meanpool.py, unchanged, with that dir and the same other
inputs, and asserts every last-token number and every paired readout difference in the new output equals the committed
reading's exactly. (An earlier version said layers 16, 24, 32, 40 and 56 had no earlier record; the 10-07 null-selection
record holds all seven, Codex ru32h on #1109.) The
committed reading is authenticated against s3_manifest_v1.json, and the re-run's input hashes must equal the committed
reading's. Exits nonzero on any mismatch.
  uv run --no-project --with numpy python experiments/2026-10-12_q2_meanpool_36/check_lasttoken_inputs.py \\
      <mean-pool run_scores dir> <fix_scores_meanpool json> <#36 run_scores dir> <fix939 per-run jsonl> <committed reading json>
"""
import glob, hashlib, json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
mp, rec, lt, fixp, ref_p = sys.argv[1:6]
fx = json.loads((HERE.parent / "2026-10-08_q3_swap_cue/results/fix939_POSTHOC_2026-10-11_020519.json").read_text())["inputs"]["scores_dir"]
got = {f"scores_L{l}.npz": sha(Path(lt) / f"scores_L{l}.npz") for l in (16, 24, 32, 40, 48, 56, 63)} | {"runs.json": sha(Path(lt) / "runs.json")}
checks = {"runs.json": fx["runs_json_sha256"], "scores_L48.npz": fx["scores_L48_sha256"], "scores_L63.npz": fx["scores_L63_sha256"]}
for k, v in checks.items():
    assert got[k] == v, f"{k} differs from fix939's recorded sha256"
ns = json.loads(subprocess.run(["git", "show", "HEAD:experiments/2026-10-06_q3_attack_detector/results/null_selection_2026-10-07_102220.json"],
                               capture_output=True, text=True, check=True).stdout)
assert ns["config"]["base_run"] == "2026-10-06_181548", "the null-selection record is not #36's run"
for l, v in ns["selection"]["saved_scores"]["files_sha256"].items():
    assert got[f"scores_L{l}.npz"] == v, f"scores_L{l}.npz differs from the 10-07 null-selection record"
print("last-token dir: runs.json, scores_L48.npz, scores_L63.npz match fix939's recorded sha256; all seven scores_L*.npz match the 10-07 null-selection record (base_run 2026-10-06_181548)")
print("sha256 of all seven layers used:", json.dumps(got, sort_keys=True))
man = json.loads((HERE / "results" / "s3_manifest_v1.json").read_text())["files"]["meanpool_reading_2026-10-12_160554.json"]
assert sha(ref_p) == man["sha256"], "the committed reading differs from its S3 manifest entry"
ref = json.loads(Path(ref_p).read_text())
run = subprocess.run([sys.executable, str(HERE / "analyze_meanpool.py"), mp, rec, lt, fixp], check=True, capture_output=True, text=True)
new_p = run.stdout.strip().splitlines()[-1]  # analyze_meanpool.py prints its own output path last
assert Path(new_p).name.startswith("meanpool_reading_") and Path(new_p).exists(), new_p
new = json.loads(Path(new_p).read_text())
n = 0
for a in ref["rates"]:
    r, w = ref["rates"][a], new["rates"][a]
    assert r["by readout"]["last-token (#36)"] == w["by readout"]["last-token (#36)"], f"last-token rows differ at {a}"
    k = "paired readout difference: mean-pooled minus last-token (points, 95%, same draws)"
    assert r[k] == w[k], f"paired differences differ at {a}"
    n += 1
assert ref["layer per test fold"] == new["layer per test fold"]
assert ref["inputs sha256"] == new["inputs sha256"], "the re-run's input hashes differ from the committed reading's"
print(f"re-run of analyze_meanpool.py reproduces the committed last-token rows and paired differences exactly at {n} rates; layer per fold identical")
print("re-run output (not committed):", Path(new_p).name, "sha256", sha(new_p))
