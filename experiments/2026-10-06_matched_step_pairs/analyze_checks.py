"""Checks on every simulated pair: did the simulator get the right history, and could it have seen step k's real output?

History: the simulator's memory must be the reference system prompt (checked at run time), then the k prefix actions in
order, each with its real output (as the simulator stores it: the first 6,000 characters inside <OUTPUT> tags), then
step k's call. Here: memory length is 2k + 2, every prefix call's command or path is in its user turn, every prefix
output is in its assistant turn, and step k's command is in the last user turn.

Leak: no input carries step k's real output (its call ID is not in the prefix; the reference shares no call ID with
the run). As a behavioral check, collect the real output's high-entropy tokens (8+ characters with a letter and 3+
digits, or 5+ digit numbers: hashes, temp names, PIDs, byte counts, timestamps) that appear in no input (prefix calls
and outputs, step k's call, the reference trajectory, the stored system prompt, the environment's codebase files and
paths), and count how many the simulated output reproduces. A simulator that saw the real output would reproduce them.

Also per pair: exact match, same length, a character similarity, and whether the simulated output copies a reference
output verbatim. Writes results/checks_<stamp>.json with no transcript text: a reproduced token is stored and printed only
as a keyed short digest (HMAC-SHA256 with a key kept in git-ignored data/), its type (hash, ip, temp name, key-like, number,
path, other), length and position, never its value.

  python experiments/2026-10-06_matched_step_pairs/analyze_checks.py --pairs data/pairs_<stamp>.jsonl --sims a.jsonl,b.jsonl
  python experiments/2026-10-06_matched_step_pairs/analyze_checks.py --pairs data/pairs_<stamp>.jsonl --rehash <old checks>
      rewrites a checks file from before 2026-10-06 20:30 (raw `reproduced_examples`) into the hashed form, as
      results/<old name>_hmac.json
"""

import argparse
import difflib
import hashlib
import hmac
import json
import re
import secrets
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
SETTINGS = Path.home() / "simulated-control-evals-sept" / ".settings" / "linuxarena"
TOKEN = re.compile(r"(?<![A-Za-z0-9])(?:(?=(?:[A-Za-z0-9+/_.=-]*\d){3})(?=[A-Za-z0-9+/_.=-]*[A-Za-z])[A-Za-z0-9+/_.=-]{8,}|\d{5,})(?![A-Za-z0-9])")
OUTPUT_CAP = 6000


# Simulator files written before rows recorded their build: only these verified files, with the build both of their
# launches logged (run_simulate_pairs.LEGACY_BUILDS; sim_pilot20 is a copy of the first 20 rows).
LEGACY_BUILDS = {name: {"simulator_model": "openai/gpt-6-astra", "sims_commit": "ba990399bfbd71aebf7636c04252d22c05720156",
                        "sims_src_diff_sha256": "7d2b6021d1301cc1b084441905bae9592d74cae2c201d808c322ae1763702db4"}
                 for name in ("sim_pilot_2026-10-06.jsonl", "sim_pilot20_2026-10-06.jsonl")}


def build_of(path, row) -> str:
    """The simulator build a row was made by, as a comparable string; legacy rows only from the verified files."""
    if "build" in row:
        return json.dumps(row["build"], sort_keys=True)
    assert Path(path).name in LEGACY_BUILDS, f"{path} has rows with no recorded simulator build"
    return json.dumps(LEGACY_BUILDS[Path(path).name], sort_keys=True)


def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)


def kind_of(token: str) -> str:
    """A coarse type for a token, so a reader can judge a reproduced token without seeing it."""
    tail = re.split(r"[=:]", token)[-1]
    if re.search(r"\d{1,3}(?:\.\d{1,3}){3}", token):
        return "ip"
    if "tmp" in token.lower():
        return "temp name"
    if token.isdigit():
        return "number"
    if re.fullmatch(r"[0-9a-fA-F]{8,}", tail):
        return "hash"
    if "/" in token and not re.search(r"[+=]", token):
        return "path"
    if len(tail) >= 16 and re.search(r"[A-Z]", tail) and re.search(r"[a-z]", tail) or re.search(r"[+/]=*$|==$", tail):
        return "key-like"
    return "other"


def hmac_key() -> bytes:
    """A random key kept only in git-ignored data/: a plain hash of a short token (a 5-digit number, a path) can be
    reversed by enumeration, a keyed one cannot without the key. Created on first use."""
    path = HERE / "data" / "checks_hmac.key"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(secrets.token_bytes(32))
        path.chmod(0o600)
    return path.read_bytes()


def describe(token: str, text: str) -> dict:
    """A token as it may be stored or printed: a keyed short digest (HMAC-SHA256, key not committed), its type, length
    and first position in the real output. Never the value: LinuxArena environments plant secrets that real outputs
    can contain."""
    digest = hmac.new(hmac_key(), token.encode(), hashlib.sha256).hexdigest()[:12]
    return {"hmac_sha256_12": digest, "type": kind_of(token), "length": len(token), "position": text.find(token)}


_env_tokens = {}


def env_tokens(env: str) -> set[str]:
    """Every high-entropy token in the environment's codebase files and paths (the simulator's read tools serve these)."""
    if env not in _env_tokens:
        found = set()
        root = SETTINGS / env / "codebase"
        for p in root.rglob("*"):
            found.update(TOKEN.findall(str(p.relative_to(root))))
            if p.is_file() and p.stat().st_size < 5_000_000:
                found.update(TOKEN.findall(p.read_bytes().decode("utf-8", "replace")))
        _env_tokens[env] = found
    return _env_tokens[env]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--sims", default=None)
    ap.add_argument("--rehash", type=Path, default=None, help="an old checks file with raw reproduced tokens")
    args = ap.parse_args()
    pairs = {json.loads(l)["pair"]: json.loads(l) for l in args.pairs.read_text().splitlines()}
    if args.rehash:
        old = json.loads(args.rehash.read_text())
        assert old["pairs_file"] == args.pairs.name, "the checks file was made from another pairs file"
        for r in old["pairs"]:
            r["reproduced_examples"] = [describe(t, pairs[r["pair"]]["real_output"]) for t in r["reproduced_examples"]]
        old["rehashed_from"] = args.rehash.name
        with open(HERE / "results" / f"{args.rehash.stem}_hmac.json", "x") as f:
            f.write(json.dumps(old, indent=1))
        return
    assert args.sims, "--sims is required unless --rehash"
    pairs_sha = hashlib.sha256(args.pairs.read_bytes()).hexdigest()
    sims, builds = {}, set()
    for path in args.sims.split(","):
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            assert r["pairs_sha256"] == pairs_sha, f"{path} holds simulator results for another pairs file"
            if r["status"] == "ok":
                builds.add(build_of(path, r))
                sims.setdefault(r["pair"], r["result"])
    assert len(builds) <= 1, f"--sims mixes {len(builds)} simulator builds; check one build at a time"
    rows = []
    for n, res in sorted(sims.items()):
        p = pairs[n]
        k, mem = p["k"], res["memory"]
        hist = len(mem) == 2 * k + 2
        def key(a):  # bash calls carry `cmd` (or, from some agents, `command`); editor calls a `path`
            args = a["arguments"]
            return str(args.get("cmd") or (args.get("command") if a["function"] != "text_editor" else None)
                       or args.get("path") or "")[:200]

        def rendered(a, text):  # bash and text_editor render the command raw; other tools render `cmd=<repr>`
            return key(a) in text or repr(key(a))[1:-1][:150] in text

        if hist:  # the slot checks only make sense, and only index safely, when the memory has the expected length
            for i, a in enumerate(p["prefix"]):
                hist &= rendered(a, mem[2 * i]["content"]) and a["output"][:OUTPUT_CAP].strip()[:2000] in mem[2 * i + 1]["content"]
            hist &= rendered(p["step"], mem[2 * k]["content"])
        inputs = "\n".join([*(s for a in p["prefix"] for s in strings(a)), *strings(p["step"]["arguments"]),
                            *(s for a in p["sim_metadata"]["orig_trajectory"] for s in strings(a)), p["stored_system_prompt"]])
        known = set(TOKEN.findall(inputs)) | env_tokens(p["environment"])
        real, sim = p["real_output"], res["output"]
        novel = set(TOKEN.findall(real)) - known
        reproduced = sorted(t for t in novel if t in sim)
        ref_outputs = {a.get("output") for a in p["sim_metadata"]["orig_trajectory"]}
        rows.append({"pair": n, "arm": p["arm"], "environment": p["environment"], "history_ok": bool(hist),
                     "memory_messages": len(mem), "k": k, "exact_match": sim == real, "same_length": len(sim) == len(real),
                     "similarity": round(difflib.SequenceMatcher(None, real[:5000], sim[:5000], autojunk=False).ratio(), 4),
                     "real_chars": len(real), "sim_chars": len(sim), "real_novel_tokens": len(novel),
                     "novel_tokens_reproduced": len(reproduced),
                     "reproduced_examples": [describe(t, real) for t in reproduced[:5]],
                     "sim_copies_reference_output": sim in ref_outputs, "real_equals_reference_output": real in ref_outputs})
    summary = {"n": len(rows), "history_ok": sum(r["history_ok"] for r in rows),
               "exact_match": sum(r["exact_match"] for r in rows), "same_length": sum(r["same_length"] for r in rows),
               "pairs_with_novel_real_tokens": sum(r["real_novel_tokens"] > 0 for r in rows),
               "novel_real_tokens": sum(r["real_novel_tokens"] for r in rows),
               "novel_real_tokens_reproduced": sum(r["novel_tokens_reproduced"] for r in rows),
               "sim_copies_reference_output": sum(r["sim_copies_reference_output"] for r in rows),
               "real_equals_reference_output": sum(r["real_equals_reference_output"] for r in rows),
               "median_similarity": sorted(r["similarity"] for r in rows)[len(rows) // 2],
               "exact_match_by_arm": dict(Counter(r["arm"] for r in rows if r["exact_match"]))}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"checks_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps({"pairs_file": args.pairs.name, "sims_files": [Path(s).name for s in args.sims.split(",")],
                            "summary": summary, "pairs": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:  # counts only; the hashed token descriptions are in the results file
        if not r["history_ok"] or r["novel_tokens_reproduced"]:
            print(f"CHECK pair {r['pair']}: history_ok={r['history_ok']} reproduced {r['novel_tokens_reproduced']} "
                  f"of {r['real_novel_tokens']} novel tokens")


if __name__ == "__main__":
    main()
