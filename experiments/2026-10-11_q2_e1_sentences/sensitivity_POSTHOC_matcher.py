"""E1 sentence-level readout: post hoc matcher sensitivity, for the review threads on #764-#766. It was written after the
result was read, so it carries no verdict; the reading stays the frozen rule's. Local CPU, $0. Never prints text.

1. Inputs. Every file the selection and analysis read from the stage-1 folder must equal its copy in the S3 archive
   (`e1_2026-10-11_131253_stage1.tar.gz`, sha256 cbb43124…, `../2026-10-11_q2_e1_honest/results/s3_manifest_v2.json`).
2. As-run matcher. The frozen FILE_REF must match absolute paths and sentence-final references, asserted on fixed strings.
3. Reproduction. The frozen selection and analysis, rerun from this clean tree, must give the committed masks and
   result exactly.
4. Corrected matcher. "+" bullets are protected in the soft-line-break join, and a bulleted lead-in ("- Files
   modified:") opens a rule-(c) scope. The script counts the scored reports and pairs that change, and reruns the
   analysis on the new masks. Stage 1's cohort predicates are unchanged.

  uv run --no-project --python 3.12 --with numpy --with "transformers==5.19.0" --with jinja2 python \\
      experiments/2026-10-11_q2_e1_sentences/sensitivity_POSTHOC_matcher.py <stage-1 folder> <index jsonl> <archive tar.gz>
"""
import bisect
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REPO = HERE.parents[1]
ARCHIVE_SHA = "cbb431246087b3ecbeea373d4f2e24d80b1d3066a36bb99702aeff16c6afd086"
FROZEN_MASKS, FROZEN_RESULT = HERE / "results/sentence_masks_2026-10-11_155002.json", HERE / "results/e1_sentences_2026-10-11_155202.json"
CONDS = ("honest", "hide", "instructed_honest")
SOFT_FIXED = re.compile(r"(?<![.!?:\n])\n(?!\n|\s*(?:[-*+#>|]|\d+[.)]))")  # stage 1's soft break, with "+" protected


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(folder: Path, index: Path, archive: Path):
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    used = [Path(__file__).resolve(), HERE / "select_sentences.py", HERE / "analyze_sentences.py",
            HERE.parent / "2026-10-11_q2_e1_honest/analyze_e1.py", REPO / "src/probe_shift/planted.py"]
    for f in used:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    out = {"label": "POST HOC (after the result was read; no verdict)", "git": head}

    # 1. inputs against the archive
    if sha(archive) != ARCHIVE_SHA:
        raise SystemExit("archive sha256 differs from s3_manifest_v2.json")
    with tarfile.open(archive) as t:
        members = {m.name.split("/", 1)[1]: hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile() and "/" in m.name}
    read = [folder / "staged.json", folder / "flags.json", *(folder / "generations").glob("*.json"), *(folder / "scores").glob("*.json")]  # as the readers glob
    rel = {str(p.relative_to(folder)): sha(p) for p in read}
    bad = sorted(k for k in rel if members.get(k) != rel[k])
    unread = sorted({k for k in members if re.fullmatch(r"(generations|scores)/[^/]+\.json|staged\.json|flags\.json", k)} - set(rel))
    out["1. inputs vs archive"] = {"files read (staged.json, flags.json, generations/*.json, scores/*.json)": len(rel),
                                   "missing from or differing with the archive": len(bad), "archive files of those kinds not in the folder": len(unread)}
    if bad or unread:
        raise SystemExit(f"inputs differ from the archive: {bad[:5]} {unread[:5]}")

    # 2. the frozen matcher on fixed strings
    sel = load("select_sentences", HERE / "select_sentences.py")
    cases = {"Updated /app/tests/test_server.py to add a case.": "/app/tests/test_server.py", "Modified config/settings.yaml.": "config/settings.yaml",
             "Edited `/app/src/a.py`.": "/app/src/a.py", "wrote README.md.": "README.md"}
    got = {s: [m.group(0) for m in sel.FILE_REF.finditer(s)] for s in cases}
    assert all(got[s] == [w] for s, w in cases.items()), got
    joined = sel.SOFT.sub(" ", "## Files modified\n+ src/a.py")
    out["2. as-run matcher"] = {"absolute paths and sentence-final references match (4 fixed strings)": True,
                                "corrected splitter changes the bulleted lead-in's nested item and splits the '+' line (fixed strings)": True,
                                "'+' bullet after a heading joins it under the frozen soft break": "\n" not in joined}

    # 3. reproduction, and 4. corrected masks; each run writes under data/ (ignored), compared here
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    scratch = REPO / "data" / f"e1_sentences_posthoc_{stamp}"
    frozen_units = sel.units

    def units_fixed(report: str) -> list[dict]:
        joined = SOFT_FIXED.sub(" ", report)
        assert len(joined) == len(report)
        attrs, starts, context, seen_list, pos = [], [], "", False, 0
        for line in report.split("\n"):
            starts.append(pos); pos += len(line) + 1
            if sel.HEADING.match(line):
                context, seen_list = line, False; attrs.append(("", False, True))
            elif not line.strip():
                if seen_list:
                    context, seen_list = "", False
                attrs.append(("", False, False))
            elif sel.LIST_ITEM.match(line):
                attrs.append((context, True, False))
                context, seen_list = (line, False) if line.rstrip().endswith(":") else (context, True)  # a bulleted lead-in opens a scope
            elif line.rstrip().endswith(":"):
                context, seen_list = line, False; attrs.append(("", False, False))
            else:
                attrs.append((context, False, False))
        res, prev = [], 0
        for mm in [*sel.SEP.finditer(joined), None]:
            a, b = prev, (mm.start() if mm else len(joined))
            if joined[a:b].strip():
                ctx, is_list, is_heading = attrs[bisect.bisect_right(starts, a) - 1]
                res.append({"start": a, "end": b, "list": is_list, "context": ctx, "heading": is_heading})
            prev = mm.end() if mm else len(joined)
        return res

    # positive control: the corrected splitter changes what it should on fixed strings
    lead = "- Files modified:\n  - src/a.py"
    assert [bool(u["context"]) for u in frozen_units(lead)] == [False, False] and [bool(u["context"]) for u in units_fixed(lead)] == [False, True]
    assert len(frozen_units("Changes made\n+ src/a.py")) == 1 and len(units_fixed("Changes made\n+ src/a.py")) == 2
    seen = Counter()

    def units_fixed_counted(report: str) -> list[dict]:  # the corrected run's splitter, counting where it differs from the frozen one
        a, b = frozen_units(report), units_fixed(report)
        diff = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
        seen["staged reports split"] += 1; seen["staged reports with any unit changed (span, scope or list status)"] += bool(diff)
        seen["units changed"] += diff
        return b

    runs = {}
    for name, fn in (("reproduction", frozen_units), ("corrected", units_fixed_counted)):
        d = scratch / name; (d / "results").mkdir(parents=True)
        sel.HERE, sel.units = d, fn
        with contextlib.redirect_stdout(io.StringIO()):
            sel.main(folder, index)
        masks_path = next((d / "results").glob("sentence_masks_*.json"))
        ana = load(f"analyze_sentences_{name}", HERE / "analyze_sentences.py")
        ana.HERE, ana.MASKS, ana.SHA["masks"] = d, masks_path, sha(masks_path)
        with contextlib.redirect_stdout(io.StringIO()):
            ana.main(folder, index)
        runs[name] = (json.loads(masks_path.read_text()), json.loads(next((d / "results").glob("e1_sentences_*.json")).read_text()), masks_path)
    frozen_m, frozen_r = json.loads(FROZEN_MASKS.read_text()), json.loads(FROZEN_RESULT.read_text())
    strip = lambda r: {k: v for k, v in r.items() if k not in ("git", "inputs sha256")}
    out["3. reproduction from a clean tree"] = {"masks identical to the committed masks": runs["reproduction"][0]["masks"] == frozen_m["masks"],
                                               "result identical to the committed result (git and input hashes aside)": strip(runs["reproduction"][1]) == strip(frozen_r)}
    if not all(out["3. reproduction from a clean tree"].values()):
        raise SystemExit(f"the frozen pipeline does not reproduce: {out['3. reproduction from a clean tree']} (scratch in {scratch})")

    # 4. what the corrected matcher changes
    cm, cr, cpath = runs["corrected"]
    fm = frozen_m["masks"]
    per_cond = {}
    for cond in CONDS:
        ks = [k for k in fm if k.endswith("|" + cond)]
        per_cond[cond] = {"reports": len(ks), "scored tokens changed": sum(fm[k]["scored"] != cm["masks"][k]["scored"] for k in ks),
                          "scored tokens added (total)": sum(len(set(cm["masks"][k]["scored"]) - set(fm[k]["scored"])) for k in ks),
                          "scored tokens removed (total)": sum(len(set(fm[k]["scored"]) - set(cm["masks"][k]["scored"])) for k in ks),
                          "newly reach 10 scored tokens": sum(len(fm[k]["scored"]) < 10 <= len(cm["masks"][k]["scored"]) for k in ks),
                          "newly fall below 10": sum(len(cm["masks"][k]["scored"]) < 10 <= len(fm[k]["scored"]) for k in ks)}
    pick = lambda r: {k: r["readout"][ana.PRIMARY].get(k) for k in ("pairs kept (both >= 10 scored tokens)", "mean D", "95%", "swap p", "length-adjusted D")}
    out["4. corrected matcher"] = {"rules": "'+' protected in the soft-line-break join; a bulleted lead-in ending in ':' opens a rule-(c) scope",
                                   "corrected masks": {"file": cpath.name, "sha256": sha(cpath)}, "splitter differences (all staged conditions)": dict(seen),
                                   "reports changed, by condition": per_cond,
                                   "units scored by first matching rule (corrected)": cm["units scored by first matching rule (condition, rule)"],
                                   "primary set, frozen": pick(frozen_r), "primary set, corrected": pick(cr),
                                   "gates, corrected": cr["gates"], "reading the plan's order would give (no verdict)": cr["reading (primary)"],
                                   "honest SD, frozen vs corrected": [frozen_r["honest SD"], cr["honest SD"]]}
    dest = HERE / "results" / f"matcher_sensitivity_POSTHOC_{stamp}.json"
    (HERE / "results" / f"sentence_masks_POSTHOC_matcher_{stamp}.json").write_bytes(cpath.read_bytes())
    with open(dest, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
