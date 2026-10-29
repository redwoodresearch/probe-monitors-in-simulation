"""Row 2, step 2's end (RULE.md here): manifest part 2. Local, $0; prints no score. It pins, with sha256:
- the new Qwen3-32B detector (the real-only fit's run record and its archived probes, scores and fits, from the record's S3
  receipt), and the frozen thresholds (in the step-2(d) record);
- the real-only Olmo and Llama artifacts (step 2(c));
- the retained-cell masks and the gate outcomes: the activation gate, the score-level gate, and step 2(d)'s false-alarm and
  convergence gates;
- the two Qwen3-32B extractions' summaries (the controls pilot and the planted e2 run).

  uv run --no-project --python 3.12 python experiments/2026-10-12_plantslip_heldout/q2_manifest_part2.py <results dir names...>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent; R = HERE / "results"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def main(fit_run, real_only, act_gate, score_gate, step2d, ctl_summary, planted_summary):
    for f in (Path(__file__).resolve(), HERE / "RULE.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} is not committed as is")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    run_rec = R / f"run_{fit_run}.json"; rr = json.loads(run_rec.read_text())
    s2 = json.loads((R / step2d).read_text()); sg = json.loads((R / score_gate).read_text()); ag = json.loads((R / act_gate).read_text())
    if s2["inputs sha256"][f"run_{fit_run}.json"] != sha(run_rec) or s2["inputs sha256"][act_gate] != sha(R / act_gate) or s2["inputs sha256"][score_gate] != sha(R / score_gate):
        raise SystemExit("step 2(d) used other inputs")
    ro = R / real_only; rorec = json.loads((ro / "real_only_record.json").read_text())
    if s2["inputs sha256"]["real_only_record.json"] != sha(ro / "real_only_record.json"):
        raise SystemExit("step 2(d) used another real-only record")
    objs = rr["archive"]["objects"]
    for l in (20, 28, 36, 44, 52, 60):  # the archived detector is the one step 2(d) and the score gate used
        if objs[f"scores_L{l}.npz"]["sha256"] != s2["inputs sha256"][f"qwen3-32b scores_L{l}.npz"] or objs[f"probes_L{l}.npz"]["sha256"] != sg["probes sha256"][str(l)]:
            raise SystemExit(f"L{l}: archived scores or probes differ from step 2(d)'s or the score gate's")
    if sha(R / sg["scores file"]) != sg["scores sha256"]:
        raise SystemExit("the score gate's npz differs from its record")
    expected = {x["file"]: x["sha256"] for rd in rorec["readers"].values() for x in [rd["runs file"]] + list(rd["layers"].values())}
    if any(not (ro / f).exists() or sha(ro / f) != h for f, h in expected.items()):
        raise SystemExit("a real-only file is missing or differs from its record")
    m = {"rule": "RULE.md step 2, manifest part 2 (this folder)", "git": head, "made": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
         "qwen3-32b detector": {"fit run": fit_run, "run record": {"file": run_rec.name, "sha256": sha(run_rec)},
                                "archived artifacts (from the run record's receipt)": {k: v for k, v in rr["archive"]["objects"].items() if k.startswith(("probes_L", "scores_L", "fits_L", "data/runs.json"))},
                                "archive": {k: v for k, v in rr["archive"].items() if k != "objects"},
                                "fits that may have hit the cap": sum(x["may_have_hit_cap"] for L in rr["probe_fits"].values() for x in L["fits"])},
         "real-only artifacts": {"record": {"file": f"{real_only}/real_only_record.json", "sha256": sha(ro / "real_only_record.json")},
                                 "files": {f"{real_only}/{f}": h for f, h in sorted(expected.items())}},
         "frozen thresholds and masks": {"file": step2d, "sha256": sha(R / step2d)},
         "gate outcomes": {"activation gate": {"file": act_gate, "sha256": sha(R / act_gate), "layers passing": ag["layers passing"], "layers failing": ag["layers failing"]},
                           "score-level gate": {"file": score_gate, "sha256": sha(R / score_gate), "layers passing": sg["layers passing"], "layers failing": sg["layers failing"],
                                                "scores file": {"file": sg["scores file"], "sha256": sha(R / sg["scores file"])}},
                           "false-alarm and convergence gates, eligibility, masks": {k: {kk: v[kk] for kk in ("false-alarm rate (held-out real honest, pooled)", "false-alarm gate passes",
                                                                                                          "convergence gate passes", "retained cells", "masks final", "eligible")}
                                                                                    for k, v in s2["configurations"].items()}},
         "qwen3-32b extractions": {"controls pilot": {"file": ctl_summary, "sha256": sha(R / ctl_summary)}, "planted e2": {"file": planted_summary, "sha256": sha(R / planted_summary)}},
         "pending": {"configurations": sorted(k for k, v in s2["configurations"].items() if not v["masks final"]),
                     "why": "Llama-3.3-70B's planted dump comes from row 3 (aswin/q2-llama-plant-row3), not yet extracted; its masks and eligibility go in a supplement to this manifest, made the same way, before any Llama prediction is scored"}}
    if any(not v["masks final"] for k, v in s2["configurations"].items() if not k.startswith("llama")):
        raise SystemExit("a non-Llama configuration's masks are not final")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := R / f"manifest_part2_{stamp}.json", "x") as fh:
        fh.write(json.dumps(m, indent=1) + "\n")
    print(dest)


if __name__ == "__main__":
    main(*sys.argv[1:8])
