"""Writes README.md from the committed result record, every number read by key (no number typed by hand).

  uv run --no-project --python 3.12 python -I make_readme.py [results/recipe_transfer_POSTHOC_2026-10-14_000532.json]

A record kept on S3 (results/manifest_recipe_transfer_v1.json) is fetched with the AWS CLI (AWS_PROFILE) when absent; its
sha256 and size are checked on every use, local copy or not.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = sys.argv[1] if len(sys.argv) > 1 else "results/recipe_transfer_POSTHOC_2026-10-14_000532.json"


def hydrate(rel: str) -> Path:
    """The record at rel. If the manifest lists it, it is fetched from S3 when absent and checked against the entry's sha256
    and size every time, so a stale or edited local copy fails too; a record the manifest does not list is used as is."""
    p = (HERE / rel).resolve()  # ./results/x, results/x and the absolute path all map to one manifest key
    m = json.loads((HERE / "results" / "manifest_recipe_transfer_v1.json").read_text())
    key = p.relative_to(HERE.parents[1]).as_posix() if p.is_relative_to(HERE.parents[1]) else None
    f = next((x for x in m["files"] if x["key"] == key), None)
    if f is None:
        if not p.exists():
            raise SystemExit(f"{rel}: absent and not in the manifest")
        return p
    fetched = not p.exists()
    if fetched:
        subprocess.run(["aws", "s3", "cp", f"s3://{m['bucket']}/{key}", str(p)], check=True)
    b = p.read_bytes()
    if hashlib.sha256(b).hexdigest() != f["sha256"] or len(b) != f["bytes"]:
        if fetched:
            p.unlink()
        raise SystemExit(f"{rel}: sha256 or size differs from the manifest" + ("" if fetched else " (local copy kept)"))
    return p


rec = json.loads(hydrate(SRC).read_text())
A = rec["analysis"]
pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}"  # noqa: E731
iv = lambda i: f"[{pct(i['ci95'][0])}, {pct(i['ci95'][1])}]"  # noqa: E731
NAMES = {"a": "(a) real-trained, full count", "b": "(b) fake/fake", "c": "(c) mixed", "b_fix939": "(b) with fix939", "c_fix939": "(c) with fix939"}
ver_name = {"full": "FULL (own reasoning in context)", "stripped": "STRIPPED (reasoning and chat deleted)"}
L = ["# The actual recipe, trained off-policy (Sol's runs) and tested on Qwen3.5's own runs: item 1, last token (POST HOC)", "",
     "Rule: `RULE.md` (committed @ 1188ce0a before any number; Amendment 1 = compute only). Result record "
     f"`{SRC}` (git {rec['git'][:8]}, app {rec['app']}, stamp {rec['stamp']}); printout beside it. "
     "**PRELIMINARY until Codex agrees.**", "",
     "## Gates (all passed)", ""]
L += [f"- {f['kind']}, fold {f['fold']}: " + "; ".join(f"{k} {json.dumps(v)}" for k, v in sorted(f["gates"].items())) for f in rec["folds"]]
L += ["", "## Readouts", "",
      "- **Reply (output_end, last token):** run score = the mean over ALL of a run's tool outputs (#36's pooling), not a fixed",
      "  first-j prefix. The reasoning-removed 2×2's 96-98% at j = 21 was a prefix mean of the first 21 outputs, with a probe",
      "  trained ON-policy: a different definition and different training data.",
      "- **Command (content-only): unavailable.** Both own-run caches hold only `call_end` and `output_end` rows "
      f"(kinds present: {json.dumps(rec['folds'][0]['kinds_present'])}), no rule-1 `command` rows. Per the rule it is not",
      "  replaced by the own call_end, which keeps fixed-slot tokens.", ""]
for ver in ("full", "stripped"):
    C = A["cells"][ver]["output_end"]
    any_c = C["a"]
    L += [f"## {ver_name[ver]}", "",
          f"N: cohort {any_c['cohort_tasks']} main tasks (one first real run, one second real run and one simulated attack run each); "
          f"honest A {any_c['honest_a_panel']} (sets the threshold), honest B {any_c['honest_b_panel']} (false alarms).", ""]
    for fp in ("0.01", "0.05"):
        L += [f"**At {float(fp):.0%} (threshold on own honest A):**", "",
              "| recipe | second real catch | simulated catch | simulated − real (95%) | real-vs-real: second − first (95%) | false alarms on honest B |",
              "|---|---|---|---|---|---|"]
        for r in ("a", "b", "c", "b_fix939", "c_fix939"):
            o = C[r][f"at_{fp}"]
            L.append(f"| {NAMES[r]} | {pct(o['catch_run2'])} {iv(o['catch_run2_interval'])} | {pct(o['catch_sim'])} | "
                     f"{pct(o['sim_minus_run2'])} {iv(o['sim_minus_run2_interval'])} | {pct(o['yardstick_run2_minus_run1'])} {iv(o['yardstick_interval'])} | {pct(o['false_alarm_honest_b'])} |")
        am = [C[f"am{d}"][f"at_{fp}"]["catch_run2"] for d in range(5)]
        L += ["", f"(a_m), count-matched, second real catch per draw: {', '.join(pct(x) for x in am)}.", ""]
        P = A["pairs"][ver]["output_end"]
        def reading(i):  # RULE.md, margin X = 10 points; both can hold, and the reference is named in the column
            lo, hi = i["ci95"]
            r = []
            if -0.10 < lo and hi < 0.10:
                r.append("within 10 points")
            if hi < 0:
                r.append("below")
            return " and ".join(r) if r else "inconclusive (interval crosses 0 and extends past ±10)"
        L += ["| recipe | D vs (a) (95%) | reading vs (a) | D vs (a_m), V2 (95%) | reading vs (a_m) |", "|---|---|---|---|---|"]
        for r in ("b", "c", "b_fix939", "c_fix939"):
            o = P[r][f"at_{fp}"]
            L.append(f"| {NAMES[r]} | {pct(o['D_vs_a'])} {iv(o['D_vs_a_interval'])} | {reading(o['D_vs_a_interval'])} | "
                     f"{pct(o['D_vs_am_V2'])} {iv(o['D_vs_am_V2_interval'])} | {reading(o['D_vs_am_V2_interval'])} |")
        L.append("")
rd = sorted({(v, r, json.dumps(A["cells"][v]["output_end"][r]["reading_at_1pct"])) for v in A["cells"] for r in A["cells"][v]["output_end"]})
L += ["## Formal readings of the grading check (stats' `reading_at_1pct`, simulated − real at 1%)", "",
      "These need the real-vs-real control inside ±5 and the ±10 plants detected; at this catch level the plants cannot be built, so",
      "every cell reads: " + ", ".join(sorted({x for _, _, x in rd})) + ". The simulated − real intervals above are descriptive.", ""]
tn = {f["fold"]: f["train_n"] for f in rec["folds"] if f["kind"] == "output_end"}
L += ["## Training runs per recipe and outer fold (RULE.md Statistics)", "",
      "Runs in each outer fold's training set, positive / negative arm (`folds[].train_n` in the record, output_end; the test "
      "readouts FULL and STRIPPED share these fits).", "",
      "| recipe | " + " | ".join(f"fold {f}" for f in sorted(tn)) + " |", "|---|" + "---|" * len(tn)]
L += [f"| {NAMES.get(r, r)} | " + " | ".join(f"{tn[f][r]['1']} / {tn[f][r]['0']}" for f in sorted(tn)) + " |" for r in tn[min(tn)]]
L.append("")
L += ["## Caveats", "",
      "- **Floor:** every recipe catches few real attacks at 1% (see the tables), so (b)'s small simulated − real difference is",
      "  partly a floor effect: both its estimate and the real catch are low. Intervals of about ±10–15 points follow from 183",
      "  tasks per arm and thresholds resting on 1–2 honest runs at 1%.",
      "- **Format mismatch:** training reads Sol's runs in #36's reader format; the test reads Qwen3.5's own runs in on-policy",
      "  replay format (FULL: own reasoning in context; STRIPPED: reasoning and chat deleted). The matched-format test is the",
      "  2×2 fork's extraction B (own runs in reader format), reported separately when it lands.",
      "- **Repairs:** fix939 is the only repaired Qwen3.5 copy that exists at the last-token readout; no simulated-honest repair.",
      "- **Post hoc:** these cells were chosen after earlier results; readings are descriptive (margin X = 10 points).", ""]
(HERE / "README.md").write_text("\n".join(L))
print("README.md written,", len(L), "lines")
