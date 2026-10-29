"""Build the four subset copies by exact, asserted replacements on the as-run sources."""
import subprocess
from pathlib import Path
S = "/private/tmp/claude-501/-Users-aswinsurya-probe-shift-q2/802b982d-f788-49b8-9390-11c29e3725f3/scratchpad"
OUT = Path(S) / "wt_sens/experiments/2026-10-12_q2_repair_sensitivity"
REPO = "/Users/aswinsurya/probe-shift-q2"
ROW2 = S + "/wt_row2/experiments/2026-10-12_plantslip_heldout"
PS = S + "/wt_ps/experiments/2026-10-12_plantslip_transfer"
def src(c, p):
    return subprocess.run(["git", "show", f"{c}:{p}"], cwd=REPO, capture_output=True, text=True, check=True).stdout
def patch(t, subs, name):
    for a, b in subs:
        n = t.count(a)
        assert n == 1, (name, n, a[:70])
        t = t.replace(a, b)
    return t
def head(orig, c):
    return f'"""SUBSET COPY (repair-rule sensitivity, PLAN.md here) of {orig} @ {c}, changed only to filter which cells enter the\ncatch averages (all | out217 | in217); on "all" its own gates must pass unchanged. The original docstring follows.\n\n'

# 1. per-repair Part B (fix939 and per-repair shares)
P = "experiments/2026-10-12_q2_per_repair/analyze_per_repair_939.py"
t = patch(src("04823b06", P), [
 ('"""Part B\'s reading (PLAN.md here;', head("../2026-10-12_q2_per_repair/analyze_per_repair_939.py", "04823b06") + 'Part B\'s reading (PLAN.md there;'),
 ("import subprocess\n", "import subprocess\nimport sys\n"),
 ("HERE = Path(__file__).parent\n", "HERE = Path(__file__).parent\nsys.path.insert(0, str(HERE))\nfrom subset_cells import SUBSETS, keeper  # noqa: E402\nSUBSET, KEEP = \"all\", (lambda c: True)\n"),
 ('    sk = [runs[i]["key"] for i in ii["sim"]]\n', '    sk = [runs[i]["key"] for i in ii["sim"]]\n    km = np.array([KEEP(c) for c in cells])  # the subset: only these cells enter the catch averages\n'),
 ("        Cpt = {n: catch(S[n], F, T_un[r_]) for n in S}\n", "        Cpt = {n: catch(S[n], F, T_un[r_])[km] for n in S}\n"),
 ("            w = ca[tix].astype(float)\n", "            w = ca[tix].astype(float) * km\n"),
 ('        for key, x in (("as_is_sim_minus_e2", "as simulated"), ("fixed_sim_minus_e2 (primary)", "both repairs")):  # the gate\n',
  '        for key, x in ((("as_is_sim_minus_e2", "as simulated"), ("fixed_sim_minus_e2 (primary)", "both repairs")) if SUBSET == "all" else ()):  # the gate (all cells)\n'),
 ('    out["gate"] = "as simulated and both repairs reproduce fix939_POSTHOC_2026-10-11_020519.json exactly (points and intervals)"\n    return out\n\n\ndef olmo(a):',
  '    out["cells in subset"] = int(km.sum())\n    out["gate"] = "as simulated and both repairs reproduce fix939_POSTHOC_2026-10-11_020519.json exactly (points and intervals)" if SUBSET == "all" else "not applied (subset)"\n    return out\n\n\ndef olmo(a):'),
 ("    cells = sorted(set(sim_i) & set(e2_i))\n", "    cells = sorted(set(sim_i) & set(e2_i))\n    cells = [c for c in cells if KEEP(c)]  # the subset (this reader's draws resample all tasks, independent of cells)\n"),
 ("    e1i = np.array([i for i in range(len(runs)) if e1[i] and np.isfinite(np.delete(S_all[i], fold[i])).all()])\n",
  "    e1i = np.array([i for i in range(len(runs)) if e1[i] and KEEP(cell[i]) and np.isfinite(np.delete(S_all[i], fold[i])).all()])\n"),
 ('        for key, x in (("real e2 - as simulated", "as simulated"), ("real e2 - fixed", "both repairs")):  # the gate\n',
  '        for key, x in ((("real e2 - as simulated", "as simulated"), ("real e2 - fixed", "both repairs")) if SUBSET == "all" else ()):  # the gate (all cells)\n'),
 ('    out["gate"] = "as simulated and both repairs reproduce edit_effect_olmo_2026-10-12_115023.json exactly (points and intervals)"\n',
  '    out["gate"] = "as simulated and both repairs reproduce edit_effect_olmo_2026-10-12_115023.json exactly (points and intervals)" if SUBSET == "all" else "not applied (subset)"\n'),
 ("    a = ap.parse_args()\n", "    ap.add_argument(\"--subset\", required=True, choices=SUBSETS); ap.add_argument(\"--swap-sets\", required=True); ap.add_argument(\"--committed\", required=True)\n    a = ap.parse_args()\n    global SUBSET, KEEP\n    SUBSET, KEEP = a.subset, keeper(a.subset, a.swap_sets)\n"),
 ('    out = {"plan": "PLAN.md Part B (this folder)",', '    out = {"plan": "PLAN.md (this folder): subset copy of the per-repair Part B reading", "subset": a.subset,'),
 ('    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")\n    (HERE / "results").mkdir(exist_ok=True)\n    with open(dest := HERE / "results" / f"per_repair_939_{stamp}.json", "x") as fh:', '    if a.subset == "all":  # the full gate: every row, catch and share of both readers equals the committed Part B result\n        ref = json.loads(Path(a.committed).read_text()); o = json.loads(json.dumps(out))\n        for rd in ("Qwen3-32B", "Olmo-3.1-32B"):\n            if ref[rd]["rates"] != o[rd]["rates"]:\n                raise SystemExit(f"all cells, {rd}: does not reproduce the committed Part B result")\n        out["full gate"] = f"every row, catch and share of both readers reproduces {Path(a.committed).name} exactly"\n    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")\n    (HERE / "results").mkdir(exist_ok=True)\n    with open(dest := HERE / "results" / f"subset_per_repair_{a.subset}_{stamp}.json", "x") as fh:'),
], "per_repair")
(OUT / "analyze_subset_per_repair.py").write_text(t)

# 2. P4/P5 reply delta
P = "experiments/2026-10-12_plantslip_transfer/q2_reply_delta.py"
t = patch(src("a5a895e8", P), [
 ('"""Planted-slip transfer, q2\'s reply pairs P4-P6 (RULE.md, Amendment 1): the response.', head("../2026-10-12_plantslip_transfer/q2_reply_delta.py", "a5a895e8") + 'Planted-slip transfer, q2\'s reply pairs P4-P6 (RULE.md, Amendment 1): the response.'),
 ("sys.path.insert(0, str(HERE))\nfrom q2_reply_score import READERS  # noqa: E402  the pairs' readers, layers and stored scores\n",
  f"sys.path.insert(0, str(HERE)); AS_RUN = \"{PS}\"; sys.path.insert(1, AS_RUN)\nfrom q2_reply_score import READERS  # noqa: E402  the pairs' readers, layers and stored scores (as-run copy)\nfrom subset_cells import keeper  # noqa: E402\n"),
 ("def main(pair: str, rec_p: Path):\n", "def main(pair: str, rec_p: Path, subset: str, swap: str, committed: Path):\n    KEEP = keeper(subset, swap)\n"),
 ('    for f in (me, HERE / "RULE.md", HERE / "q2_reply_score.py"):\n', '    for f in (me, HERE / "PLAN.md", HERE / "subset_cells.py"):\n'),
 ('        raise SystemExit(f"{len(cells)} cells, not 939")\n', '        raise SystemExit(f"{len(cells)} cells, not 939")\n    cells = [c for c in cells if KEEP(c)]  # the subset (draws resample all tasks; thresholds fixed)\n'),
 ('    out = {"plan": "RULE.md, Amendment 1 (this folder)",', '    out = {"plan": "PLAN.md (this folder): subset copy of q2_reply_delta.py", "subset": subset, "q2_reply_score.py sha256": sha(Path(AS_RUN) / "q2_reply_score.py"),'),
 ('    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")\n    with open(dest := HERE / "results" / f"q2_reply_delta_{pair}_{stamp}.json", "x") as fh:',
  '    if subset == "all":  # the gate: the committed result reproduces exactly\n        ref = json.loads(Path(committed).read_text())\n        if ref["rates"] != json.loads(json.dumps(out["rates"])) or ref["layer per test fold"] != json.loads(json.dumps(out["layer per test fold"])):\n            raise SystemExit("all cells: does not reproduce the committed result")\n        out["gate"] = f"reproduces {Path(committed).name} exactly (rates and layers)"\n    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")\n    with open(dest := HERE / "results" / f"subset_reply_delta_{pair}_{subset}_{stamp}.json", "x") as fh:'),
 ("    main(sys.argv[1], Path(sys.argv[2]))\n", "    main(sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5]))\n"),
], "reply_delta")
(OUT / "analyze_subset_reply_delta.py").write_text(t)

# 3. Llama plant delta
P = "experiments/2026-10-12_q2_llama_plant/analyze_plant_llama.py"
t0 = src("652bce10", P)
t = patch(t0, [
 (t0.split("\n")[0] + "\n", head("../2026-10-12_q2_llama_plant/analyze_plant_llama.py", "652bce10") + t0.split("\n")[0][3:] + "\n"),
 ("HERE = Path(__file__).parent\n", "HERE = Path(__file__).parent\nsys.path.insert(0, str(HERE))\nfrom subset_cells import keeper  # noqa: E402\n"),
 ("def main(d: Path, rec_p: Path):\n", "def main(d: Path, rec_p: Path, subset: str, swap: str, committed: Path):\n    KEEP = keeper(subset, swap)\n"),
 ('    for f in (Path(__file__).resolve(), HERE / "PLAN.md", HERE / "run_score_plant_llama.py"):\n', '    for f in (Path(__file__).resolve(), HERE / "PLAN.md", HERE / "subset_cells.py"):\n'),
 ('        raise SystemExit(f"{len(cells)} cells, not 939")\n', '        raise SystemExit(f"{len(cells)} cells, not 939")\n    cells = [c for c in cells if KEEP(c)]  # the subset (draws resample all tasks; thresholds fixed)\n'),
], "llama")
# output name and gate: find the stamp/dest lines
import re
m = re.search(r'    stamp = datetime\.now\(ZoneInfo\("America/Los_Angeles"\)\)\.strftime\("%Y-%m-%d_%H%M%S"\)\n    with open\(dest := HERE / "results" / f"(llama_plant_delta)_\{stamp\}\.json", "x"\) as fh:', t)
assert m, "llama dest"
t = t.replace(m.group(0), '    out["subset"] = subset\n    if subset == "all":  # the gate: the committed result reproduces exactly\n        ref = json.loads(Path(committed).read_text())\n        if ref["rates"] != json.loads(json.dumps(out["rates"])):\n            raise SystemExit("all cells: does not reproduce the committed result")\n        out["gate"] = f"reproduces {Path(committed).name} exactly (rates)"\n    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")\n    with open(dest := HERE / "results" / f"subset_llama_plant_delta_{subset}_{stamp}.json", "x") as fh:')
t = patch(t, [("    main(Path(sys.argv[1]), Path(sys.argv[2]))\n", "    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5]))\n")], "llama main")
(OUT / "analyze_subset_plant_llama.py").write_text(t)

# 4. row 2's y v2
P = "experiments/2026-10-12_plantslip_heldout/q2_y.py"
t0 = src("4c4df611", P)
V2 = ("q2_y_2026-10-12_220622.json", "cf843108d0728c4838c78a8515b390bd88f7fbf28749e76c63c693cf546407cb")
t = patch(t0, [
 (t0.split("\n")[0] + "\n", head("../2026-10-12_plantslip_heldout/q2_y.py", "4c4df611 (version 2)") + t0.split("\n")[0][3:] + "\n"),
 ('HERE = Path(__file__).parent; R_ = HERE / "results"\n',
  f'HERE = Path(__file__).parent; ROW2 = Path("{ROW2}"); R_ = ROW2 / "results"  # row 2\'s worktree (as-run 62edb8e8)\nsys.path.insert(0, str(HERE))\nfrom subset_cells import keeper  # noqa: E402\nV2 = {V2!r}  # version 2, the final result: the all-cells gate\n'),
 ("def main(step2d_p, stage_p, real_only, qwen_fit, qplant_p, oplant_p, lplant_p, qsim_p, olmo_mixed, llama_mixed, part1_p):\n",
  "def main(step2d_p, stage_p, real_only, qwen_fit, qplant_p, oplant_p, lplant_p, qsim_p, olmo_mixed, llama_mixed, part1_p, subset, swap):\n    KEEP = keeper(subset, swap)\n"),
 ('    for f in (Path(__file__).resolve(), HERE / "RULE.md"):\n', '    for f in (Path(__file__).resolve(), HERE / "PLAN.md", HERE / "subset_cells.py"):\n'),
 ('        if subprocess.run(["git", "merge-base", "--is-ancestor", a, "HEAD"], cwd=HERE).returncode:\n', '        if subprocess.run(["git", "merge-base", "--is-ancestor", a, "HEAD"], cwd=ROW2).returncode:\n'),
 ('capture_output=True, text=True, check=True, cwd=HERE).stdout)  # noqa: E731\n', 'capture_output=True, text=True, check=True, cwd=ROW2).stdout)  # noqa: E731\n'),
 ('        cells = c["retained cell ids"]; cR, cP, cS, tk, bad = [], [], [], [], []\n', '        cells = [cl for cl in c["retained cell ids"] if KEEP(cl)]; cR, cP, cS, tk, bad = [], [], [], [], []  # the subset\n'),
 ('        if abs(pt[k]["x"] - pred[k]["x"]) > 1e-9:\n', '        if subset == "all" and abs(pt[k]["x"] - pred[k]["x"]) > 1e-9:\n'),
 ('    if sha(R_ / V1[0]) != V1[1]:\n        raise SystemExit("not version 1\'s result file")\n    v1 = json.loads((R_ / V1[0]).read_text())\n    if v1["primary"] != out["primary"] or any({kk: vv for kk, vv in v1["configurations"][k].items() if kk != "split-half"} !=\n                                              {kk: vv for kk, vv in out["configurations"][k].items() if kk != "split-half"} for k in out["configurations"]) \\\n            or {k: v for k, v in v1["secondary"].items() if k != "split-half Spearman"} != {k: v for k, v in out["secondary"].items() if k != "split-half Spearman"}:\n        raise SystemExit("a number other than the split-half block differs from version 1\'s file")\n    out["version"] = {"version": 2, "supersedes for the split-half block only": {"file": V1[0], "sha256": V1[1]},\n                      "split-half Spearman, version 1": v1["secondary"]["split-half Spearman"]}\n',
  '    out["subset"] = subset\n    if subset == "all":  # the gate: version 2\'s final result reproduces exactly\n        if sha(R_ / V2[0]) != V2[1]:\n            raise SystemExit("not version 2\'s result file")\n        v2 = json.loads((R_ / V2[0]).read_text()); o = json.loads(json.dumps(out))\n        if v2["primary"] != o["primary"] or v2["configurations"] != o["configurations"] or v2["secondary"] != o["secondary"]:\n            raise SystemExit("all cells: does not reproduce version 2\'s result")\n        out["gate"] = f"reproduces {V2[0]} exactly (primary, configurations, secondary)"\n'),
 ('    with open(dest := R_ / f"q2_y_{stamp}.json", "x") as fh:', '    with open(dest := HERE / "results" / f"subset_y_{subset}_{stamp}.json", "x") as fh:'),
 ('            p_, a_ = bool(pred[k][fk]), pt[k]["y"] >= X5\n', '            p_ = bool(pred[k][fk]) if (subset == "all" or not name.startswith("planting")) else bool(pt[k]["x"] + (OFFSET if "calibrated" in name else 0.0) >= X5)  # subsets: planting flags from the subset x\n            a_ = pt[k]["y"] >= X5\n'),
 ("    main(*sys.argv[1:12])\n", "    main(*sys.argv[1:14])\n"),
], "y")
(OUT / "analyze_subset_y.py").write_text(t)
print("built 4 copies")
