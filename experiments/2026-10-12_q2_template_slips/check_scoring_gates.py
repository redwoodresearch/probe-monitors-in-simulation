"""Review check for #1045 (Codex threads rpT9A, rpT9E): every committed plant scoring record's gates, and the pilot list
against the pilot run's commit and the staging record. run_score_plants.py records a failed gate as false and exits 0;
this asserts no committed record failed, from its stored flags and from its recorded maxima against the tolerances
(G1 1e-3, G2/G3 0.01), and exits nonzero otherwise. Reads bytes from the as-run commit 503b55e8 (reachable from part 9's merge).
  uv run --no-project python experiments/2026-10-12_q2_template_slips/check_scoring_gates.py
"""
import hashlib, json, subprocess
C, D = "503b55e8", "experiments/2026-10-12_q2_template_slips/"
show = lambda p, c=C: subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True).stdout
names = sorted(set(subprocess.run(["git", "log", C, "--format=", "--name-only", "--diff-filter=A", "--", D + "results/plant_score_*"], capture_output=True, text=True, check=True).stdout.split()))
for n in names:
    r = json.loads(show(n))
    g1 = r["G1: max |re-scored - stored| e2 score by layer [nested, outer] (tolerance 1e-3)"]
    print(n.split("/")[-1], "| mode", r["mode"], "| git", r["git"][:8], "| G1 passes", r["G1 passes"], "max", f"{max(max(v) for v in g1.values()):.2e}",
          "| G2 and G3 pass", r.get("G2 and G3 pass", "n/a (gate mode)"), "| expected absent", r.get("expected runs absent", "n/a"),
          "| scored vs staged", r.get("runs scored vs staged", "n/a"))
    assert r["G1 passes"] is True and max(max(v) for v in g1.values()) <= 1e-3, n
    if r["mode"] != "gate":
        worst = [x for k in ("G2: max |control row - dump row| by layer (tolerance 0.01)", "G3: max |planted prefix row - dump row| by layer (tolerance 0.01)")
                 for x in r[k].values()]
        assert r["G2 and G3 pass"] is True and max(worst) <= 0.01 and r["expected runs absent"] == 0, n
        assert r["rows compared for the controls and both plants"] is True and r["no nonfinite row score or difference"] is True, n
        print("   recomputed: G2/G3 worst", max(worst), "(tolerance 0.01)")
    if r["mode"] == "full":
        assert all(a == b for a, b in r["runs scored vs staged"].values()), n
    if r["mode"] == "pilot":
        pilot_commit = r["git"]
# the pilot list: its bytes at the pilot run's commit vs at the as-run head, and every entry in the staging record
pl_path = D + "config/plants_pilot_transcripts.json"
a, b = show(pl_path, pilot_commit), show(pl_path)
assert a == b
print("pilot list sha256 at the pilot scoring run's commit", pilot_commit[:8], hashlib.sha256(a).hexdigest()[:16], "| at", C, hashlib.sha256(b).hexdigest()[:16], "| identical", a == b)
log = subprocess.run(["git", "log", "--format=%h %ad", "--date=format:%H:%M", C, "--", pl_path], capture_output=True, text=True).stdout.split("\n")
print("pilot list commits:", [x for x in log if x])
pl = json.loads(b)
# the staging record: first copy in git (now on S3 via manifest v3); pin its sha256 and check the pilot list against it
stp = D + "results/plant_stage_2026-10-12_124505.json"
first = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h", C, "--", stp], capture_output=True, text=True, check=True).stdout.split()[-1]
sb = show(stp, first); st = json.loads(sb)
man = json.loads(show(D + "results/s3_manifest_v3.json"))
assert hashlib.sha256(sb).hexdigest() == man["files"]["plant_stage_2026-10-12_124505.json"]["sha256"], "staging record vs manifest v3"
print("staging record first copy", first, "sha256 matches manifest v3:", hashlib.sha256(sb).hexdigest() == man["files"]["plant_stage_2026-10-12_124505.json"]["sha256"])
planted = {(x["plant"], x["key"]) for x in st["per run"]}
ctrl = set(st["controls"])
aff = [tuple(x) for x in pl["affected"]]; con = [tuple(x) for x in pl["controls"]]
assert all(x in planted for x in aff) and all(k in ctrl for _, k in con)
# exact match to the staging record's canonical pilot (#1055): the same (arm, run) entries, no duplicates, controls in the
# control arm, and the same staged inputs run and shard hashes
cfg = sorted(aff + con); canon = sorted(tuple(x) for x in st["pilot"])
assert len(set(cfg)) == len(cfg) and cfg == canon, "pilot list vs the staging record's pilot"
assert all(a == "e2-control" for a, _ in con) and all(a != "e2-control" for a, _ in aff), "pilot arms"
assert pl["inputs_run"] == st["staged"].split(":", 1)[1] and pl["inputs_sha256"] == st["inputs_sha256"], "pilot inputs vs staged"
print("pilot list = the staging record's pilot exactly:", cfg == canon, f"({len(cfg)} entries, no duplicates)",
      "| inputs run and", len(pl["inputs_sha256"]), "shard sha256s match the staging record:", True)
print("pilot list: affected", len(aff), "all in the staging record's planted (plant, run):", all(x in planted for x in aff),
      "| controls", len(con), "all staged controls:", all(k in ctrl for _, k in con))
for n in names:
    r = json.loads(show(n))
    assert r["inputs sha256"].get("plant_stage_2026-10-12_124505.json") == hashlib.sha256(sb).hexdigest(), n
    print(n.split("/")[-1], "staging sha256 in record = first copy's:", r["inputs sha256"].get("plant_stage_2026-10-12_124505.json") == hashlib.sha256(sb).hexdigest())
