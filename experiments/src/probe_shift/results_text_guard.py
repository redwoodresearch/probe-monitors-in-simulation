"""Fail when a change adds model text to an experiment's results/.

Model text (completions, reasoning, self-reports, judge explanations, call arguments, coder paraphrases) goes to data/
and S3; results/ keeps lengths and hashes (AGENTS.md). This scans every JSON/JSONL(.gz) blob under
experiments/*/results/ that any commit on a branch adds or changes relative to a base (so text added and later removed is
still caught; for a merge commit, only blobs that differ from every parent, i.e. conflict resolutions), and flags:
- prose-shaped strings (80+ characters, 8+ spaces) under keys that hold model text (keys like note or rule also hold our
  own prose, so they are not matched: this is a backstop, not a proof);
- any string over MAX_CHARS, whatever its key;
- in text results (.csv, .tsv, .txt, .log, .md, .patch), any line over MAX_CHARS (prose there is often ours, so only the
  size check applies).
Commits listed in the base branch's docs/raw_text_exception_manifest.json (the 2026-10-08 one-time exception) are
skipped. It does not compare against transcripts (the 6-word overlap check needs them locally; see each experiment's scrubber).

  PYTHONPATH=src uv run --no-project python -m probe_shift.results_text_guard --base origin/main
"""

import argparse
import gzip
import json
import re
import subprocess
import sys

MODEL_TEXT_KEY = re.compile(
    r"reasoning|completion|response|messages?$|^content$|_content$|^text$|_text$|self_report|explanation|evidence"
    r"|paraphrase|arm_guess|approach|tail|tokens$|argument|^command$|^cmd$|transcript|excerpt|quote|answer|output$|attack_description"
)
MAX_CHARS = 5000
PATHSPEC = ":(glob)experiments/*/results/**"


def prose(s: str) -> bool:
    return len(s) >= 80 and s.count(" ") >= 8


def findings(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from findings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for v in obj:
            yield from findings(v, f"{path}[]")
    elif isinstance(obj, str):
        key = re.sub(r"\[\]$", "", path.rsplit(".", 1)[-1])
        if len(obj) > MAX_CHARS:
            yield path, len(obj), "longer than %d characters" % MAX_CHARS
        elif prose(obj) and MODEL_TEXT_KEY.search(key):
            yield path, len(obj), f"prose under a model-text key ({key})"


TEXT_SUFFIX = re.compile(r"\.(csv|tsv|txt|log|md|patch)(\.gz)?$")


def text_findings(data: bytes, name: str):
    if name.endswith(".gz"):
        data = gzip.decompress(data)
    for i, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
        if len(line) > MAX_CHARS:
            yield f"line {i}", len(line), "line longer than %d characters" % MAX_CHARS


def records(data: bytes, name: str):
    if name.endswith(".gz"):
        data, name = gzip.decompress(data), name[:-3]
    text = data.decode("utf-8", "replace")
    if not text.strip():  # an empty or whitespace-only file holds no text to scan; any other parse failure still raises
        return []
    if name.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def excepted_commits(base: str) -> set[str]:
    """Read from the base branch, not the PR, so a PR cannot exempt its own commits."""
    try:
        files = json.loads(git("show", f"{base}:docs/raw_text_exception_manifest.json"))["files"]
    except subprocess.CalledProcessError:
        return set()
    return {f["commit"] for f in files}


def changed_blobs(c: str):
    """(path, blob) for results files commit c adds or changes; for a merge, only blobs that match no parent."""
    parents = git("rev-list", "--parents", "-n", "1", c).decode().split()[1:]
    first = [parents[0]] if parents else ["--root"]
    out = git("diff-tree", "-r", "-z", "--no-commit-id", "--no-renames", "--diff-filter=AM", *first, c, "--", PATHSPEC)
    fields = out.decode().split("\0")
    for meta, name in zip(fields[0::2], fields[1::2]):
        old, blob = meta.split()[2:4]
        if old == blob:
            continue  # mode-only change: the content is the first parent's
        if any(blob_at(p, name) == blob for p in parents[1:]):
            continue  # taken unchanged from another parent, which is scanned on its own or already on the base
        yield name, blob


def blob_at(commit: str, name: str):
    try:
        return git("rev-parse", f"{commit}:{name}").decode().strip()
    except subprocess.CalledProcessError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/main", help="scan the commits in base..head")
    ap.add_argument("--head", default="HEAD")
    args = ap.parse_args(argv)
    skip = excepted_commits(args.base)
    commits = git("rev-list", f"{args.base}..{args.head}").decode().split()
    seen, bad, n_files = set(), [], 0
    for c in commits:
        if any(c.startswith(s) for s in skip):
            continue
        for name, blob in changed_blobs(c):
            is_json, is_text = bool(re.search(r"\.jsonl?(\.gz)?$", name)), bool(TEXT_SUFFIX.search(name))
            if (blob, name) in seen or not (is_json or is_text):
                continue
            seen.add((blob, name))  # keyed by path too: the format (JSON, text, gzip) comes from the name
            n_files += 1
            data = git("cat-file", "blob", blob)
            if is_text:
                bad += [(c[:7], name, *f) for f in text_findings(data, name)]
                continue
            for rec in records(data, name):
                bad += [(c[:7], name, *f) for f in findings(rec)]
    for c, name, path, n, why in bad:
        print(f"{c} {name}: {path} ({n} chars): {why}")
    print(f"checked {n_files} results blob(s) in {len(commits)} commit(s); {len(bad)} finding(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
