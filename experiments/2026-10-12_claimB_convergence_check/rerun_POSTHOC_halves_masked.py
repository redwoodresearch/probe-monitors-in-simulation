"""POST HOC (RULE.md in this folder): the halves summaries re-run with amendment 3's convergence mask on the selection,
to see whether any reported number moves. Column 0 converged everywhere (convergence_POSTHOC_2026-10-12_120752.json),
so only the nulls that read the shuffle columns can move: plant 2's refit-null p, the sim catch null (p and p95), and
the AUROC delta's grouped-null margin X.

Each summary is run twice from a git archive of its own commit, reading the Volume as it did (numbers only, $0):
(a) unchanged, which must reproduce the committed result's "results" exactly (the gate); (b) with the one selection line
`best = V.argmax(axis=0)  # per column` replaced by the masked argmax (entries whose max(grad_full, grad_val) exceeds
1e-4·min(sqrt(C), 1) get -inf; a column left without a converged entry stops the run, since none exist here).
Reported: both "results" and every leaf that differs.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy \\
      --with matplotlib python experiments/2026-10-12_claimB_convergence_check/rerun_POSTHOC_halves_masked.py --tree <scratch dir>
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
SUMMARIES = {  # name: (commit, analyzer path, committed result path)
    "halves_qwen3_32b": ("53da7f48", "experiments/2026-10-11_claimB_halves/analyze_halves.py",
                         "experiments/2026-10-11_claimB_halves/results/halves_2026-10-11_164208.json"),
    "halves_qwen35": ("770518ea", "experiments/2026-10-11_claimB_halves_qwen35/analyze_halves_qwen35.py",
                      "experiments/2026-10-11_claimB_halves_qwen35/results/halves_qwen35_2026-10-12_005827.json"),
}
OLD = "best = V.argmax(axis=0)  # per column"
NEW = ("best = np.where(np.maximum(z['grad'][:, 0, :], z['grad'][:, 1, :]) <= np.array([1e-4 * min(C**0.5, 1.0) for _, C in "
       "json.loads(str(z['subs']))])[:, None], V, -np.inf).argmax(axis=0)  # amendment 3's mask (convergence check)\n"
       "{ind}if not np.isfinite(np.where(np.maximum(z['grad'][:, 0, :], z['grad'][:, 1, :]) <= np.array([1e-4 * min(C**0.5, 1.0) for _, C in "
       "json.loads(str(z['subs']))])[:, None], V, -np.inf).max(axis=0)).all():\n{ind}    raise SystemExit('a column has no converged entry')")


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=ROOT).stdout


def run_one(path, tree):
    """Run one analyzer's main() in a fresh interpreter (so the two commits' runner modules never share sys.modules)."""
    code = ("import importlib.util, sys; p = sys.argv[1]; s = importlib.util.spec_from_file_location('m', p); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); m.main()")
    gitdir = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    env = dict(__import__("os").environ, PYTHONPATH=str(tree / "src"), GIT_DIR=gitdir)  # archives have no .git: the analyzers' git calls use the repo's
    r = subprocess.run([sys.executable, "-c", code, str(path)], cwd=path.parent, env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{path.name} failed:\n{r.stderr[-3000:]}")


def diff(a, b, p=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [x for k in sorted(set(a) | set(b)) for x in diff(a.get(k), b.get(k), f"{p}/{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in diff(u, v, f"{p}[{i}]")]
    return [] if a == b else [{"path": p, "unmasked": a, "masked": b}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    a = ap.parse_args()
    a.tree = a.tree.resolve()
    out = {"label": "POST HOC: halves summaries re-run with amendment 3's convergence mask (RULE.md)", "summaries": {}}
    for name, (commit, analyzer, result) in SUMMARIES.items():
        tree = a.tree / commit
        if not tree.exists():
            tree.mkdir(parents=True)
            subprocess.run(["tar", "-x", "-C", str(tree)], input=git("archive", commit, "experiments", "splits", "src"), check=True)
        src = (tree / analyzer).read_text()
        stored_doc = json.loads(git("show", f"{commit}:{result}"))
        if src.encode() != git("show", f"{commit}:{analyzer}") or hashlib.sha256(src.encode()).hexdigest() != stored_doc["script_sha256"]:
            raise SystemExit(f"{analyzer}: the archive copy differs from {commit} or from the result's script_sha256")
        line = next(ln for ln in src.splitlines() if ln.strip() == OLD)
        ind = line[: len(line) - len(line.lstrip())]
        masked = src.replace(line, ind + NEW.format(ind=ind), 1)
        if masked.count("amendment 3's mask") != 1:
            raise SystemExit(f"{analyzer}: selection line not found once")
        mpath = (tree / analyzer).with_name(Path(analyzer).stem + "_MASKED_POSTHOC.py")
        mpath.write_text(masked)
        got = {}
        for tag, path in (("unmasked", tree / analyzer), ("masked", mpath)):
            res_dir = path.parent / "results"
            before = set(res_dir.glob("*.json")) if res_dir.exists() else set()
            run_one(path, tree)
            new = sorted(set(res_dir.glob("*.json")) - before)
            if len(new) != 1:
                raise SystemExit(f"{name} {tag}: expected one new result file, found {len(new)}")
            got[tag] = json.loads(new[0].read_text())["results"]
            if tag == "unmasked" and got[tag] != stored_doc["results"]:  # the gate, before the masked run
                raise SystemExit(f"{name}: the unchanged re-run does not reproduce {result}@{commit}: {diff(stored_doc['results'], got[tag])[:5]}")
        out["summaries"][name] = {"commit": commit, "analyzer": analyzer, "stored_result": result, "gate_reproduced": True,
                                  "differences_masked_vs_unmasked": diff(got["unmasked"], got["masked"]),
                                  "masked_results": got["masked"]}
    out["git"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"halves_masked_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    for name, s in out["summaries"].items():
        print(name, "gate reproduced; leaves that differ:", len(s["differences_masked_vs_unmasked"]))
        for d in s["differences_masked_vs_unmasked"]:
            print("  ", d["path"], d["unmasked"], "->", d["masked"])


if __name__ == "__main__":
    main()
