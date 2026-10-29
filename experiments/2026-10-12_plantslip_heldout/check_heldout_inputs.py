"""Review check for seven Codex threads on row 2: did its as-run scoring, step 2(d), x and y use the right inputs?
  #1162 rwbQs  q2_step2d.py doesn't check the Qwen3-32B fit's local runs.json against the fit's archived data/runs.json
  #1162 rwbQz  --real-only accepts any self-consistent folder: its manifest part 1 sha256 isn't checked against the pin
  #1163 rwSMq  planted scoring checks only the gate's score npz, not the gate's staging and real-only record sha256
  #1163 rwbMV  the scorers don't check the frozen probes against the frozen-artifact manifest
  #1164 rwS2y  q2_x.py doesn't check the step-2 record against manifest part 2
  #1164 rweqe  q2_step2d_llama.py trusts every manifest_part2_*.json its glob finds in results/
  #1168 rwzoc  q2_score_sim_qwen3.py accepts any --fit-run, not only the preregistered 2026-10-12_192842
The scripts can't be re-run, so this checks what they read. Every JSON and npz it reads is downloaded from S3 by its key in
this folder's results_manifest_v*.json and must match that manifest's sha256 first; the exceptions are the Olmo and Llama fit
records (git show at the commits manifest part 1 pins) and, if an as-run results folder is given, its local files (the
Qwen3-32B fit's runs.json, which step 2(d) read but didn't hash). Exits nonzero on any mismatch. Prints hashes, file names and
layer lists only: no score, threshold or rate.
  uv run --no-project --with numpy python experiments/2026-10-12_plantslip_heldout/check_heldout_inputs.py [<as-run results dir>]
"""
import hashlib
import io
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
REC = "experiments/2026-10-12_plantslip_heldout/results/"
P2, P2S = "2ff64db39ebd", "0911c962646e"  # 09's re-checked prefixes of manifest part 2 and its supplement
PART1 = re.search(r'PART1_SHA = "([0-9a-f]{64})"', (HERE / "q2_prepare_real_only.py").read_text())[1]  # the committed pin
STAGE_PIN = re.search(r'STAGE_SHA = "([0-9a-f]+)"', (HERE / "q2_step2d_llama.py").read_text())[1]
PT = ZoneInfo("America/Los_Angeles")
ASRUN = Path(sys.argv[1]) if len(sys.argv) > 1 else None
sha = lambda b: hashlib.sha256(b).hexdigest()
MAN, BUCKETS = {}, set()
for p in sorted(HERE.glob("results_manifest_v*.json")):
    m = json.loads(p.read_text()); BUCKETS.add(m["bucket"])
    MAN |= {k: v["sha256"] for k, v in m.get("files", {}).items()}
assert len(BUCKETS) == 1, "results manifests name more than one bucket"
BUCKET = BUCKETS.pop()
bad = []


def raw(key):
    """an S3 object's bytes, sha256-checked against the results manifest when it lists the key"""
    b = subprocess.run(["aws", "--profile", "redwood-sandbox", "s3", "cp", f"s3://{BUCKET}/{key}", "-"], capture_output=True, check=True).stdout
    assert key not in MAN or sha(b) == MAN[key], f"{key}: S3 copy differs from its manifest sha256"
    return b, sha(b)


def get(name):
    assert REC + name in MAN, f"{name}: not in a results manifest"
    b, h = raw(REC + name)
    return json.loads(b), h


def chk(thread, label, ok):
    print(("match   " if ok else "MISMATCH"), f"[{thread}]", label); ok or bad.append(label)


m1, h1 = get("manifest_part2_2026-10-12_201334.json"); m2, h2 = get("manifest_part2_2026-10-12_212356.json"); p1, hp1 = get("manifest_part1_2026-10-12_185811.json")
chk("all", f"manifest part 2 sha256 {h1[:12]} = 09's {P2}; supplement {h2[:12]} = 09's {P2S}", h1.startswith(P2) and h2.startswith(P2S))
chk("all", f"manifest part 1 sha256 {hp1[:12]} = results_manifest_v1's pin = q2_prepare_real_only.py's PART1_SHA", hp1 == PART1)
RO = m1["real-only artifacts"]["record"]["sha256"]; GATE = m1["gate outcomes"]["score-level gate"]
chk("all", "the supplement keeps manifest part 2's real-only record and score-level gate", RO == m2["real-only artifacts"]["record"]["sha256"] and GATE == m2["gate outcomes"]["score-level gate"])
FIT = {rd: p1["existing_detector_artifacts"][rd] for rd in ("olmo", "llama")}
FITREC = {rd: json.loads(subprocess.run(["git", "show", "{1}:{0}".format(*FIT[rd]["run_record"].split("@"))], capture_output=True, text=True, check=True).stdout) for rd in FIT}
QARCH = {k: v["sha256"] for k, v in m1["qwen3-32b detector"]["archived artifacts (from the run record's receipt)"].items()}
PROBES = {"qwen3-32b": QARCH} | {rd: {k: v["sha256"] for k, v in FIT[rd]["objects"].items()} for rd in FIT}  # the frozen-artifact pins
for rd in FIT:
    chk("rwbMV", f"{rd}: manifest part 1's probe pins equal the fit record's archive receipt ({FIT[rd]['run_record'][-49:]})",
        all(FITREC[rd]["archive"]["objects"][k]["sha256"] == v for k, v in PROBES[rd].items()))

# --- #1163 (rwSMq, rwbMV): each planted run against its gate, the staging record and the frozen probes
PAIRS = {"qwen3-32b": ("q2_frozen_scores_gate_2026-10-12_193504.json", "q2_frozen_scores_planted_2026-10-12_202420.json", "fit run"),
         "olmo": ("q2_olmo_scores_gate_2026-10-12_201619.json", "q2_olmo_scores_planted_2026-10-12_203343.json", "frozen run"),
         "llama": ("q2_llama_scores_gate_2026-10-12_204411.json", "q2_llama_scores_planted_2026-10-12_212315.json", "frozen run")}
planted_sha, STAGE = {}, None
for rd, (gf, pf, runkey) in PAIRS.items():  # a planted record doesn't name its gate (that was --gate), so the pairing must be unique
    pre = gf.split("gate_")[0]
    listed = sorted(k for k in MAN if re.fullmatch(re.escape(REC + pre) + r"(gate|planted)_[0-9_-]+\.json", k))
    chk("rwSMq", f"{rd}: the results manifests list exactly one gate and one planted record ({gf}, {pf})", listed == sorted([REC + gf, REC + pf]))
    g, hg = get(gf); p, hp = get(pf); planted_sha[pf] = hp
    if rd == "qwen3-32b":
        chk("rwSMq", "qwen3-32b: the gate record is manifest part 2's score-level gate (record and npz sha256)", hg == GATE["sha256"] and g["scores sha256"] == GATE["scores file"]["sha256"])
        STAGE = g["staging sha256"]  # independent of step 2: the staging hash inside the manifest-pinned gate record
        chk("rwSMq", f"the manifest-pinned gate's staging sha256 {STAGE[:12]} starts with the scripts' pinned STAGE_SHA {STAGE_PIN}", STAGE.startswith(STAGE_PIN))
    chk("rwSMq", f"{rd}: gate is mode gate, planted is mode planted, same frozen detector ({g.get(runkey)})", g["mode"] == "gate" and p["mode"] == "planted" and g.get(runkey) is not None and g.get(runkey) == p.get(runkey))
    chk("rwSMq", f"{rd}: gate and planted staging sha256 equal the manifest-pinned gate's", g["staging sha256"] == p["staging sha256"] == STAGE)
    if rd != "qwen3-32b":
        chk("rwSMq", f"{rd}: gate and planted real-only record sha256 equal manifest part 2's ({RO[:12]})", g["real-only record sha256"] == p["real-only record sha256"] == RO)
    layers = [str(l) for l in p["layers scored"]]
    chk("rwSMq", f"{rd}: gate failed no layer and passed every planted layer {layers}", bool(layers) and not g["layers failing"] and set(layers) <= set(map(str, g["layers passing"])))
    chk("rwbMV", f"{rd}: gate and planted probe sha256 equal the frozen-artifact pins ({'manifest part 2' if rd == 'qwen3-32b' else 'manifest part 1'}) at every scored layer",
        all(g["probes sha256"][l] == PROBES[rd][f"probes_L{l}.npz"] for l in map(str, g["layers scored"])) and all(p["probes sha256"][l] == PROBES[rd][f"probes_L{l}.npz"] for l in layers))
    chk("rwSMq", f"{rd}: gate and planted score npz sha256 equal their manifest entries", MAN[REC + g["scores file"]] == g["scores sha256"] and MAN[REC + p["scores file"]] == p["scores sha256"])

# --- #1164 (rwS2y) and #1162 (rwbQz): the step-2 records and the real-only record they read
s2a, hs2a = get("q2_step2d_2026-10-12_200953.json"); s2b, hs2b = get("q2_step2d_2026-10-12_211734.json")
for name, rec, h, man in (("q2_step2d_2026-10-12_200953.json", s2a, hs2a, m1), ("q2_step2d_2026-10-12_211734.json", s2b, hs2b, m2)):
    chk("rwS2y", f"{name}: sha256 {h[:12]} = its manifest's frozen thresholds and masks", h == man["frozen thresholds and masks"]["sha256"])
    i = rec["inputs sha256"]
    chk("rwS2y", f"{name}: records manifest part 2's real-only record and score-level gate, and the pinned staging record", i["real_only_record.json"] == RO and i[GATE["file"]] == GATE["sha256"] and i["q2_reply_stage_2026-10-12_135318.json"] == STAGE)
ro_rec, hro = get("real_only_2026-10-12_191044/real_only_record.json")
chk("rwbQz", f"the real-only record step 2(d) read ({RO[:12]}) is the S3 copy, and it records manifest part 1's pinned sha256", hro == RO and ro_rec["manifest part 1 sha256"] == hp1 == PART1)
for rd in FIT:
    rr = ro_rec["readers"][rd]
    chk("rwbQz", f"{rd}: the real-only record's source score files equal manifest part 1's pins at layers {sorted(rr['layers'], key=int)}",
        sorted(rr["layers"], key=int) == list(map(str, FIT[rd]["layers"])) and all(v["source sha256"] == FIT[rd]["objects"][f"scores_L{l}.npz"]["sha256"] for l, v in rr["layers"].items()))
    chk("rwbQs", f"{rd}: the real-only record's source runs.json equals the fit record's archived data/runs.json ({rr['source runs.json sha256'][:12]})",
        rr["source runs.json sha256"] == FITREC[rd]["archive"]["objects"]["data/runs.json"]["sha256"])
    runs, hr = raw(REC + f"real_only_2026-10-12_191044/{rd}_runs.json"); src = [r["source_index"] for r in json.loads(runs)]
    order = True
    for v in rr["layers"].values():
        b, hb = raw(REC + f"real_only_2026-10-12_191044/{v['file']}")
        order &= hb == v["sha256"] == m1["real-only artifacts"]["files"][f"real_only_2026-10-12_191044/{v['file']}"] and np.array_equal(np.load(io.BytesIO(b))["source_index"], src)
    chk("rwbQs", f"{rd}: the real-only runs file and score files (each sha256 = the record's and manifest part 2's) share one source_index order",
        hr == rr["runs file"]["sha256"] == m1["real-only artifacts"]["files"][f"real_only_2026-10-12_191044/{rd}_runs.json"] and order)

# --- #1162 (rwbQs): the Qwen3-32B fit's runs.json, which q2_step2d.py read from results/run_scores_<fit>/ without hashing it
qrec, hq = get("run_2026-10-12_192842.json")
chk("rwbQs", "qwen3-32b: the fit record's S3 copy is manifest part 2's run record and the one step 2(d) records", hq == m1["qwen3-32b detector"]["run record"]["sha256"] == s2a["inputs sha256"]["run_2026-10-12_192842.json"])
chk("rwbQs", f"qwen3-32b: manifest part 2's archived data/runs.json ({QARCH['data/runs.json'][:12]}) and scores equal the fit record's receipt",
    all(qrec["archive"]["objects"][k]["sha256"] == v for k, v in QARCH.items()))
_, harch = raw(m1["qwen3-32b detector"]["archive"]["prefix"].split("/", 3)[3] + "data/runs.json")
chk("rwbQs", "qwen3-32b: the archived data/runs.json on S3 hashes to that receipt", harch == QARCH["data/runs.json"])
QS = {k.split(" ", 1)[1]: v for k, v in s2a["inputs sha256"].items() if k.startswith("qwen3-32b scores_L")}
chk("rwbQs", f"qwen3-32b: the {len(QS)} score files step 2(d) recorded equal the fit's archived ones", len(QS) == 6 and all(QARCH[k] == v for k, v in QS.items()))
if ASRUN:
    d = ASRUN / "run_scores_2026-10-12_192842"; t2 = datetime.strptime("2026-10-12_200953", "%Y-%m-%d_%H%M%S").replace(tzinfo=PT)
    chk("rwbQs", "qwen3-32b, as-run folder: its six score files are the ones step 2(d) hashed (the folder it read)",
        all(sha((d / k).read_bytes()) == v for k, v in QS.items()))
    chk("rwbQs", "qwen3-32b, as-run folder: its runs.json equals the fit's archived data/runs.json, last written before step 2(d)'s output stamp"
        " (corroboration, not a recorded input hash: it assumes the folder is untouched since)",
        sha((d / "runs.json").read_bytes()) == QARCH["data/runs.json"] and datetime.fromtimestamp((d / "runs.json").stat().st_mtime, PT) < t2)
else:
    print("not run  [rwbQs] qwen3-32b: the as-run local runs.json (pass the as-run results folder)"); bad.append("as-run folder not given")

# --- #1164 (rweqe): which manifest q2_step2d_llama.py's glob could find, and what it pinned
listed = sorted(k.rsplit("/", 1)[1] for k in MAN if re.fullmatch(re.escape(REC) + r"manifest_part2_[0-9_-]+\.json", k))
chk("rweqe", f"the results manifests list two manifest_part2 files {listed}", listed == ["manifest_part2_2026-10-12_201334.json", "manifest_part2_2026-10-12_212356.json"])
if ASRUN:
    chk("rweqe", "the as-run results folder holds the same two", sorted(p.name for p in ASRUN.glob("manifest_part2_*.json")) == listed)
chk("rweqe", "the Llama run's recorded --step2d input is the step-2(d) record manifest part 2 (2ff64db3..., 09-checked) pins",
    m1["frozen thresholds and masks"]["sha256"] == hs2a == s2b["supplements"]["sha256"] == s2b["inputs sha256"]["q2_step2d_2026-10-12_200953.json"])
chk("rweqe", "the supplement pins the Llama run's own output (be3739a7...), so it postdates that run (chronology, assuming no manifest_part2 file was"
    " written and later removed; it can't prove the folder's past contents)", m2["frozen thresholds and masks"]["sha256"] == hs2b)
LL = {"retained cells", "removed, by reason", "retained cell ids", "masks final", "eligible"}
ca, cb = s2a["configurations"], s2b["configurations"]
chk("rweqe", f"the Llama run carried over every field of the {len(ca)} configurations except Llama's masks (thresholds, false-alarm and convergence gates unchanged)",
    sorted(ca) == sorted(cb) and all({f: v for f, v in ca[k].items() if not (k.startswith("llama") and f in LL)} == {f: v for f, v in cb[k].items() if not (k.startswith("llama") and f in LL)} for k in ca)
    and all(set(cb[k]["retained cell ids"]) <= set(ca[k]["retained cell ids"]) for k in ca if k.startswith("llama")))
y2, _ = get("q2_y_2026-10-12_220622.json"); iy = y2["inputs sha256"]  # inputs only
chk("rweqe", "y v2 records the supplement (sha256), its step-2(d) record and manifest part 1", iy["manifest_part2_2026-10-12_212356.json"] == h2 and iy["q2_step2d_2026-10-12_211734.json"] == hs2b and iy["manifest part 1"] == hp1)

# --- #1168 (rwzoc): the fit run behind the as-run simulated scoring, and the thresholds y applied to it (hash fields only)
FITRUN = m1["qwen3-32b detector"]["fit run"]; QREC = m1["qwen3-32b detector"]["run record"]["sha256"]
listed = sorted(k.rsplit("/", 1)[1] for k in MAN if re.fullmatch(re.escape(REC) + r"q2_sim_scores_qwen3_[0-9_-]+\.json", k))
chk("rwzoc", f"the results manifests list one simulated scoring record {listed}", listed == ["q2_sim_scores_qwen3_2026-10-12_215648.json"])
sim, hsim = get("q2_sim_scores_qwen3_2026-10-12_215648.json")
chk("rwzoc", f"its fit run is the preregistered {FITRUN} (manifest part 2's), and the fit record it hashed is manifest part 2's run record",
    sim["fit run"] == FITRUN == "2026-10-12_192842" and sim["inputs sha256"][f"run_{FITRUN}.json"] == QREC)
QL = sorted(k for k in QARCH if k.startswith("probes_L"))
chk("rwzoc", f"its probe sha256 equal the frozen fit's archived {', '.join(QL)}", sorted(f"probes_L{l}.npz" for l in sim["probes sha256"]) == QL
    and all(QARCH[f"probes_L{l}.npz"] == v for l, v in sim["probes sha256"].items()))
chk("rwzoc", "its score npz sha256 equals its manifest entry", MAN[REC + sim["scores file"]] == sim["scores sha256"])
for yf in ("q2_y_2026-10-12_215741.json", "q2_y_2026-10-12_220622.json"):
    iy = get(yf)[0]["inputs sha256"]  # inputs only
    chk("rwzoc", f"{yf} (y) records this simulated scoring record and, for thresholds, the step-2(d) record the supplement pins, built from the same fit record",
        iy["q2_sim_scores_qwen3_2026-10-12_215648.json"] == hsim and iy["q2_step2d_2026-10-12_211734.json"] == hs2b and s2b["inputs sha256"][f"run_{FITRUN}.json"] == QREC)

# --- #1164 (rwS2y): what each x file records as its inputs
for xf, s2name, hs2, planted in (("q2_x_2026-10-12_203432.json", "q2_step2d_2026-10-12_200953.json", hs2a, ("qwen3-32b", "olmo")),
                                 ("q2_x_2026-10-12_212448.json", "q2_step2d_2026-10-12_211734.json", hs2b, ("qwen3-32b", "olmo", "llama"))):
    x, _ = get(xf); i = x["inputs sha256"]  # inputs only
    chk("rwS2y", f"{xf}: its step-2 record is {s2name} with the checked sha256", x["step 2 record"] == s2name and i[s2name] == hs2)
    chk("rwS2y", f"{xf}: its staging record and planted records ({', '.join(planted)}) have the checked sha256",
        i["q2_reply_stage_2026-10-12_135318.json"] == STAGE and all(i[PAIRS[rd][1]] == planted_sha[PAIRS[rd][1]] for rd in planted))
print(f"{len(bad)} mismatches or checks not run")
sys.exit(1 if bad else 0)
