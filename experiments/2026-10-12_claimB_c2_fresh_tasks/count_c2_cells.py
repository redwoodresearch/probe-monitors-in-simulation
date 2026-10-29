"""C2 phase 1, $0, IDs and counts only (Aswin's approval for C2 drafting, relayed by 0d, 00:07:15 PT 10-12; phase 1 scope
per Aswin 00:12 PT: plan only). How many fresh C2 cells could come from LinuxArena's EXISTING side tasks paired with main
tasks that no analysis has used, per environment and per halves direction?

No attack text is read or written: side tasks are counted by ID, compatibility comes from the ID lists in each side task's
config, and existing attack strategies are counted by their paths only (a blobless clone never downloads their text).

Inputs (paths as arguments; every file's sha256 is recorded):
  --env-root      the September environment snapshot (~/simulated-control-evals-sept/.settings/linuxarena), the env
                  version claim B's runs used; each env's git commit is recorded.
  --exclusions    splits/exclusions_v4.json (claim B / Q1 public-10 cells: real e1, e2, sim attack, real and sim honest).
  --task-order    q2's batch spec (274-task order; covers the sealed zoo C and lean230 tasks: IDs only).
  --la18          la18 manifests (real, R0, R1, honest; selected-sample keys and environments only).
  --strategy-trees  blobless clones of the current LinuxArena env repos (paths only) to count existing attack strategies.
  --scan-repo / --scan-file  a conservative "used anywhere" scan: every main-task ID that appears as a whole word in any
                  tracked file at the tip of any origin or local branch of the given repos, or in any text file under the
                  given paths (sealed-run folders excluded), is used. Limit: earlier commits' deleted files and other
                  machines are not scanned.

Cell = one (main task, side task) pair in one environment, as in splits/exclusions_v4.json `cell` = "main:side".

  uv run --no-project --python 3.12 python experiments/2026-10-12_claimB_c2_fresh_tasks/count_c2_cells.py ...
"""

import argparse
import ast
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
DEV = ["auto_workflow", "file_indexing", "model_registry", "slack_clone", "web_scraping"]  # fixed_sim/split.json
TEST = ["clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a, cwd):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=True).stdout


def inventory(env_root):
    envs = {}
    for e in sorted(p.name for p in Path(env_root).iterdir() if (Path(env_root) / p.name / "main_tasks").is_dir()):
        d = Path(env_root) / e
        mains = sorted(p.name for p in (d / "main_tasks").iterdir() if (p / "info.yml").exists())
        sides, compat = [], {}
        for p in sorted((d / "side_tasks").iterdir()):
            if not (p / "__init__.py").exists():
                continue
            sides.append(p.name)
            rule = parse_rule((p / "__init__.py").read_text())
            compat[p.name] = rule
        envs[e] = {"commit": git("log", "-1", "--format=%H", cwd=d).strip(), "mains": mains, "sides": sides, "compat": compat}
    return envs


def parse_rule(src):
    """Static AST parse (never executed) of compatible_main_tasks / incompatible_main_tasks keyword arguments; a value
    that isn't a literal list/tuple of strings (or a module-level name bound to one) marks the rule unparsed."""
    tree = ast.parse(src)
    names, counts = {}, {}
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for tg in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(tg, ast.Name):
                    counts[tg.id] = counts.get(tg.id, 0) + 1
                    if isinstance(n, ast.Assign):
                        names[tg.id] = n.value
    mutated = {c.func.value.id for c in ast.walk(tree) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
               and isinstance(c.func.value, ast.Name) and c.func.attr in ("append", "extend", "insert", "remove", "add", "update", "discard", "pop")}
    unsafe = {k for k, v in counts.items() if v > 1} | mutated
    rule = {"compatible": None, "incompatible": [], "unparsed": False}

    def lit(v):
        if isinstance(v, ast.Name):
            if v.id in unsafe or v.id not in names:
                raise ValueError  # rebound, mutated or unknown name: don't guess
            v = names[v.id]
        if isinstance(v, (ast.List, ast.Tuple, ast.Set)) and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in v.elts):
            return [e.value for e in v.elts]
        if isinstance(v, ast.Constant) and v.value is None:
            return None
        raise ValueError

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in ("compatible_main_tasks", "incompatible_main_tasks"):
                    try:
                        val = lit(kw.value)
                    except ValueError:
                        rule["unparsed"] = True
                        continue
                    if kw.arg == "compatible_main_tasks":
                        rule["compatible"] = val
                    else:
                        rule["incompatible"] = val or []
    return rule


def admits(rule, main):
    """control_tower task.py:229-239 @d1156cea: a side pairs with every main by default; compatible restricts; incompatible removes."""
    if rule["unparsed"]:
        return False  # conservative: unknown rule, don't count the pair
    if rule["compatible"] is not None and main not in rule["compatible"]:
        return False
    return main not in rule["incompatible"]


def used_structured(args):
    used, src = defaultdict(set), {}
    ex = json.loads(Path(args.exclusions).read_text())
    for r in ex["samples"] + ex["excluded"]:
        m = r.get("main_task") or (r.get("cell") or "").split(":")[0]
        if m:
            used[r["environment"]].add(m)
    src["exclusions"] = sha(args.exclusions)
    order = json.loads(Path(args.task_order).read_text())["task order"]
    used["*task_order"] |= set(order)  # env not recorded per ID; matched to envs below
    src["task_order"] = sha(args.task_order)
    for f in args.la18:
        d = json.loads(Path(f).read_text())
        for s in d["selected_samples"]:
            used[s["environment"]].add(s["key"].split(":")[0])
        src[Path(f).name] = sha(f)
    return used, src


TOKEN = re.compile(rb"[A-Za-z0-9_]+")
BINARY_EXT = (".npz", ".npy", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".eval", ".zst", ".gz", ".zip", ".tar", ".pt",
              ".safetensors", ".parquet", ".bin", ".pkl", ".pickle", ".h5", ".arrow", ".whl", ".so", ".pyc")
SKIP_DIRS = {".venv", ".git", ".settings", "node_modules", "lean230_stage1", "q2_lean230"}  # sealed-run folders never read


def used_scan(all_ids, args):
    """Whole-word match of every main-task ID (all are [a-z0-9_] only, so token-set membership equals grep -w) over every
    unique non-binary blob at the tip of every origin and local branch of --scan-repo, plus text files under --scan-file."""
    assert all(re.fullmatch(r"[A-Za-z0-9_]+", m) for m in all_ids), "an ID has a non-word character; token match != grep -w"
    want = {m.encode() for m in all_ids}
    hits, n_refs, blobs, n_bytes, missing = set(), 0, {}, 0, 0
    for repo in args.scan_repo:
        refs = [r for r in git("for-each-ref", "--format=%(refname)", "refs/remotes", "refs/heads", "refs/tags", cwd=repo).split()
                if not r.endswith("/HEAD")]
        n_refs += len(refs)
        for r in refs:  # each file version is scanned once
            for rec in git("ls-tree", "-r", "-z", r, cwd=repo).split("\0"):
                if rec:
                    meta, path = rec.split("\t", 1)
                    if meta.split()[1] == "blob" and not path.lower().endswith(BINARY_EXT) and not (set(path.split("/")) & SKIP_DIRS):
                        blobs.setdefault(meta.split()[2], repo)
    for repo in args.scan_repo:
        todo = [b for b, rp in blobs.items() if rp == repo]
        cat = subprocess.Popen(["git", "cat-file", "--batch"], cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        for b in todo:  # one request, one reply: never more than one blob in flight, so the pipes can't deadlock
            cat.stdin.write(b.encode() + b"\n")
            cat.stdin.flush()
            head = cat.stdout.readline().split()
            if len(head) < 3 or head[1] == b"missing":
                missing += 1
                continue
            size = int(head[2])
            data = cat.stdout.read(size)
            cat.stdout.read(1)
            n_bytes += size
            hits |= want.intersection(TOKEN.findall(data))
        cat.stdin.close()
        if cat.wait() != 0:
            raise RuntimeError(f"cat-file failed in {repo}")
    n_files = 0
    for root in args.scan_file:
        root = Path(root).resolve()
        if set(root.parts) & SKIP_DIRS:
            raise SystemExit(f"refusing to scan a sealed or excluded path: {root}")
        files = [root] if root.is_file() else [f for f in root.rglob("*") if f.is_file() and not (set(f.relative_to(root).parts[:-1]) & SKIP_DIRS)]
        for f in files:
            if f.name.lower().endswith(BINARY_EXT):
                continue
            data = f.read_bytes()
            if b"\0" in data[:8192]:
                continue  # binary
            n_files += 1
            n_bytes += len(data)
            hits |= want.intersection(TOKEN.findall(data))
    return {h.decode() for h in hits}, {"refs": n_refs, "unique_blobs": len(blobs), "missing_blobs": missing, "scan_files": n_files, "bytes_scanned": n_bytes}


def strategies(trees):
    """Existing attack strategies by path only: side_tasks/<side>/docs/attack_strategies/<main>/<n>.md."""
    out = {}
    for t in trees:
        names = git("ls-tree", "-r", "--name-only", "HEAD", cwd=t).splitlines()
        pairs = {(p.split("/")[1], p.split("/")[4]) for p in names if "/attack_strategies/" in p and p.startswith("side_tasks/")}
        out[Path(t).name] = {"commit": git("log", "-1", "--format=%H", cwd=t).strip(), "pairs": sorted(f"{m}:{s}" for s, m in pairs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-root", required=True)
    ap.add_argument("--exclusions", required=True)
    ap.add_argument("--task-order", required=True)
    ap.add_argument("--la18", nargs="+", required=True)
    ap.add_argument("--strategy-trees", nargs="+", required=True)
    ap.add_argument("--scan-repo", nargs="+", required=True)
    ap.add_argument("--scan-file", nargs="*", default=[])
    a = ap.parse_args()

    envs = inventory(a.env_root)
    owner = {m: e for e, v in envs.items() for m in v["mains"]}
    all_ids = set(owner)
    assert len(all_ids) == sum(len(v["mains"]) for v in envs.values()), "a main-task ID appears in two environments"
    used, src = used_structured(a)
    order = used.pop("*task_order")
    unknown_order = sorted(order - all_ids)
    for m in order & all_ids:
        used[owner[m]].add(m)
    structured = {m for s in used.values() for m in s}
    scanned, scan_stats = used_scan(all_ids, a)
    strat = strategies(a.strategy_trees)

    per_env, per_dir = {}, {"dev->test (held out: test)": TEST, "test->dev (held out: dev)": DEV}
    for e, v in envs.items():
        never = [m for m in v["mains"] if m not in structured and m not in scanned]
        pairs = [(m, s) for m in never for s in v["sides"] if admits(v["compat"][s], m)]
        mains_with_side = sorted({m for m, _ in pairs})
        st = set(strat.get(e, {}).get("pairs", []))
        with_strategy = sorted({f"{m}:{s}" for m, s in pairs if f"{m}:{s}" in st})
        per_env[e] = {
            "main_tasks": len(v["mains"]), "side_tasks": len(v["sides"]),
            "side_tasks_with_restrictions": sorted(s for s, r in v["compat"].items() if r["compatible"] is not None or r["incompatible"] or r["unparsed"]),
            "used_main_tasks_structured": len(used.get(e, set()) & set(v["mains"])),
            "used_main_tasks_any_source": len({m for m in v["mains"] if m in structured or m in scanned}),
            "never_used_main_tasks": len(never),
            "never_used_with_an_admitted_side_task": len(mains_with_side),
            "admitted_pairs_never_used_mains": len(pairs),
            "admitted_pairs_with_existing_strategy": len(with_strategy),
            "never_used_main_task_ids": never,
            "env_commit": v["commit"],
        }
    out = {
        "label": "C2 phase 1 counts ($0; IDs and counts only; no attack text read or written)",
        "rule_pairing": "control_tower task.py:229-239 @d1156cea40a9 (default all mains; compatible restricts; incompatible removes)",
        "cell_definition": "one (main, side) pair in one environment; splits/exclusions_v4.json `cell` = main:side",
        "halves": {"dev": DEV, "test": TEST, "source": "experiments/2026-10-09_fixed_sim/split.json; claimB_halves PLAN.md:10-12"},
        "per_environment": per_env,
        "per_direction": {k: {"environments": envs_, "never_used_main_tasks": sum(per_env[e]["never_used_main_tasks"] for e in envs_ if e in per_env),
                              "cells_one_per_main_existing_side": sum(per_env[e]["never_used_with_an_admitted_side_task"] for e in envs_ if e in per_env),
                              "admitted_pairs": sum(per_env[e]["admitted_pairs_never_used_mains"] for e in envs_ if e in per_env),
                              "pairs_with_existing_strategy": sum(per_env[e]["admitted_pairs_with_existing_strategy"] for e in envs_ if e in per_env)}
                          for k, envs_ in per_dir.items()},
        "totals": {"main_tasks": len(all_ids), "side_tasks": sum(len(v["sides"]) for v in envs.values()),
                   "used_structured": len(structured & all_ids), "used_scan_only": len(scanned - structured), "used_any": len((structured | scanned) & all_ids),
                   "task_order_ids_not_in_snapshot": len(unknown_order), "scan": scan_stats},
        "strategy_trees": {k: {"commit": v["commit"], "n_pairs": len(v["pairs"])} for k, v in strat.items()},
        "inputs": src | {"scan_files": a.scan_file},
        "git": git("rev-parse", "HEAD", cwd=HERE).strip(),
        "script_sha256": sha(__file__),
    }
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"c2_counts_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("totals", "per_direction")}, indent=1))
    for e, v in per_env.items():
        print(e, {k: v[k] for k in ("main_tasks", "side_tasks", "used_main_tasks_any_source", "never_used_main_tasks",
                                    "never_used_with_an_admitted_side_task", "admitted_pairs_never_used_mains", "admitted_pairs_with_existing_strategy")})
    print("->", dst)


if __name__ == "__main__":
    main()
