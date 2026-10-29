"""Exposure audit for the held-out planting test: in every version of every committed results JSON, on every remote
branch, list each place that names a candidate layer of a candidate reader. Prints key paths and file@blob only, never
values. A hit is then classed by hand (metadata, or a possible gap). $0, git only.

usage: python3 -I audit_exposure.py <repo> > out.json"""

import json
import re
import subprocess
import sys

REPO = sys.argv[1]
CAND = {"qwen3-32b": [20, 28, 36, 44, 52, 60], "olmo": [16, 24, 48, 63], "llama": [20, 30, 40, 50, 79]}
# A results file belongs to a reader if its path names the reader, or it sits in a folder that only that reader uses.
READER_OF_PATH = [("olmo", re.compile(r"olmo", re.I)), ("llama", re.compile(r"llama", re.I)),
                  ("qwen3-32b", re.compile(r"qwen3-32b|qwen3_32b|q3_attack_detector|q3_swap_cue|meanpool_36|pinned_layers|"
                                           r"per_repair|reverse_plant|plantP3|plantslip_reply|q2_reply", re.I))]
SEALED = ("oct11c", "lean230")
TAG = r"(?:^|[^0-9A-Za-z])(?:L|layer[ _]?|l|@|_L|fits_L|scores_L|probes_L)({})(?![0-9])"


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, check=True).stdout


def walk(obj, path, keyhit, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            if keyhit(ks):
                out.append("/".join(path + (ks,)))
            if ks.lower() in ("layer", "layers", "chosen_layer", "layer_per_fold", "layer per test fold") and isinstance(v, (int, list)):
                vals = v if isinstance(v, list) else [v]
                if any(isinstance(x, int) and keyhit(str(x), bare=True) for x in vals):
                    out.append("/".join(path + (ks,)) + " (value)")
            walk(v, path + (ks,), keyhit, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, path + (str(i),), keyhit, out)
    elif isinstance(obj, str) and keyhit(obj):
        out.append("/".join(path) + " (string)")


paths = sorted({f for f in git("log", "--all", "--name-only", "--format=").split("\n")
                if f.endswith(".json") and "/results/" in f and not any(s in f.lower() for s in SEALED)})
report = {r: {} for r in CAND}
scanned = 0
failures = []
skipped_no_reader = []
for f in paths:
    readers = [r for r, rx in READER_OF_PATH if rx.search(f)]
    if not readers:
        skipped_no_reader.append(f)
        continue
    blobs = {}
    for h in git("log", "--all", "--format=%h", "--", f).split():
        r = subprocess.run(["git", "-C", REPO, "rev-parse", f"{h}:{f}"], capture_output=True, text=True)
        if r.returncode == 0:
            blobs.setdefault(r.stdout.strip(), h)
    for blob, h in blobs.items():
        try:
            d = json.loads(git("cat-file", "-p", blob))
        except (json.JSONDecodeError, subprocess.CalledProcessError) as e:
            failures.append({"file": f, "blob": blob[:12], "error": type(e).__name__})
            continue
        scanned += 1
        for rd in readers:
            layers = CAND[rd]
            rx = re.compile(TAG.format("|".join(map(str, layers))), re.I)
            bare = re.compile(r"^({})$".format("|".join(map(str, layers))))

            def keyhit(s, bare_ok=False, bare_=False):
                return bool(rx.search(s)) or bool(bare.match(s)) if not bare_ok else bool(bare.match(s))
            out = []
            walk(d, (), lambda s, bare=False: (bool(re.compile(r"^({})$".format("|".join(map(str, layers)))).match(s)) if bare
                                               else bool(rx.search(s)) or bool(re.compile(r"^({})$".format("|".join(map(str, layers)))).match(s))), out)
            if out:
                report[rd].setdefault(f, []).append({"blob": blob[:12], "commit": h, "hits": out})
refs = [l.split(" ", 1) for l in git("for-each-ref", "--format=%(objectname) %(refname)").splitlines()] + [[git("rev-parse", "HEAD").strip(), "HEAD"]]
json.dump({"candidates": CAND, "coverage": {
    "refs_scanned": len(refs), "refs": {r[1]: r[0] for r in refs}, "note_refs": "every ref git log --all traverses (all refs plus HEAD), with its tip commit", "fetch_head_mtime": subprocess.run(
        ["stat", "-f", "%Sm", git("rev-parse", "--path-format=absolute", "--git-common-dir").strip() + "/FETCH_HEAD"],
        capture_output=True, text=True).stdout.strip(),
    "scope": "JSON files under any results/ folder, every blob on the locally fetched remote refs; reader assigned by path pattern",
    "reader_patterns": {r: rx.pattern for r, rx in READER_OF_PATH}, "excluded_substrings": list(SEALED),
    "results_json_paths_total": len(paths), "paths_scanned": sorted(set(paths) - set(skipped_no_reader)),
    "paths_skipped_no_reader": skipped_no_reader,
    "parse_failures": failures, "not_covered": "non-JSON outputs, logs, uncommitted files, Modal Volumes and S3 objects. Owner dispositions cover score files and run directories; logs and other non-JSON outputs remain unverified"},
    "scanned_blobs": scanned, "report": report}, sys.stdout, indent=1)
