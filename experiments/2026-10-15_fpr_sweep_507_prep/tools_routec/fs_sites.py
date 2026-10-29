"""PATHGATE by code (09's item 1 after the census side a STOP, 10-17): enumerate, from the runner and its modules AS COMMITTED at a
commit, every file-system call site reachable from run_peraction311.main() (and dispatch) or from module import, by a static call
graph over the source (ast). Reads code only (git show <commit>:<module>); no record, config or Volume is read; nothing is executed.

  uv run --no-project --python 3.12 python -I tools_routec/fs_sites.py <commit> > <out.json>

Output: {"commit", "modules": {name: blob}, "targets", "sites": [{"site": "<module>@<blob8>:L<line>", "module", "line", "func", "call",
"in_step", "text", "target"?}], "n_sites", "n_in_step", "imports_unexpected", "main_defaults", "reachable_funcs", "step_funcs"}; "text"
is the code line (stripped, at most 200 chars). A site is a call to open(), np.load(), a method named in FS_METHODS (read_bytes,
read_text, write_bytes, write_text, open, mkdir, exists, is_file, is_dir, is_symlink, iterdir, glob, rglob, stat, unlink, rename, a
one-argument replace, touch, listdir; modal Volume calls are listed too and classified by hand in the table), or a NON-git subprocess
(subprocess.run / check_output / check_call / Popen / call, os.system / os.popen whose argv does not start with "git"; a0b05-63478's
R-1, 10-17), call "subprocess", with its "target": the repo script named by a ".py" constant in its argv. Each target script is then
analysed the same way, as __main__ (its main guard runs), and its sites inherit the subprocess site's in_step.
Reachability: rp.main and rp.dispatch, every module's top-level code (it runs at import), and every function a reachable function names
(a bare name in the same module, an imported name, or <alias>.<name> for the module aliases below; nested defs and lambdas count as
part of their enclosing function). Over-inclusive by design: a name reached is a function reached. A site reached only through CUT is
"in_step": false.
Imports (a63478's O1): every import in reachable code and at module top level must be the standard library, one of MODULES or a
subprocess target, modal or numpy (installed in the image), or gate_spec_diff_routec inside a CUT function only; any other is listed
in "imports_unexpected" (pathgate_code.py STOPs on it: its file reads are not analysed here).
"""
import ast
import json
import subprocess
import sys

E = "experiments/2026-10-15_fpr_sweep_507_prep"
MODULES = ["run_peraction311", "run_fprsweep_507", "run_common455", "run_common710", "cutfull_decisive", "full_override", "il_arms",
           "inputs_full", "inputs_final", "inputs_final_v2", "copies"]  # D's DCS list: the runner and its 11 modules at the census head
ALIASES = {"FO": "full_override", "C": "run_common455", "CD": "cutfull_decisive", "CP": "copies", "C7": "run_common710", "ILA": "il_arms",
           "IF": "inputs_final", "IF2": "inputs_final_v2", "IFU": "inputs_full"}  # rp's `import x as Y` (L34-38, L1408, L1528, L5377-5378,
# L5888, L5990, L6235); a module's own aliases are read from its import statements below
FS_METHODS = {"read_bytes", "read_text", "write_bytes", "write_text", "open", "mkdir", "exists", "is_file", "is_dir", "is_symlink", "iterdir",
              "glob", "rglob", "stat", "unlink", "rename", "replace", "touch", "listdir"}
SUBPROCESS = {("subprocess", f) for f in ("run", "check_output", "check_call", "Popen", "call")} | {("os", "system"), ("os", "popen")}
ROOTS = [("run_peraction311", "main"), ("run_peraction311", "dispatch")]
CUT = {("run_peraction311", f) for f in ("row", "drive", "run_rows", "finish_rows", "launch_from", "run_cutfull")}  # Modal row / driver functions,
# the local row runner and the laptop launch: none runs inside a records-wrapper step (alltasks with --spec-only or --dryrun: rp L5380-5382,
# L6545-6582, L5346-5352); a site reached only through them is "in_step": false
THIRD_PARTY_OK = {"modal", "numpy"}  # installed in the wrapper's rc image / imported by the runner
CUT_ONLY_OK = {"gate_spec_diff_routec"}  # rp launch_from L5280 (a laptop launch, in CUT)


def src(commit, mod):
    return subprocess.run(["git", "show", f"{commit}:{E}/{mod}.py"], capture_output=True, text=True, check=True).stdout


def blob(commit, mod):
    return subprocess.run(["git", "rev-parse", f"{commit}:{E}/{mod}.py"], capture_output=True, text=True, check=True).stdout.strip()


def outer_funcs(tree):
    """Top-level defs (and methods, as Class.method) -> node; nested defs belong to their enclosing top-level def."""
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[n.name] = n
        elif isinstance(n, ast.ClassDef):
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{n.name}.{m.name}"] = m
    return out


def aliases_of(tree):
    al, names = dict(ALIASES), {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name in MODULES:
                    al[a.asname or a.name] = a.name
        elif isinstance(n, ast.ImportFrom) and n.module in MODULES:
            for a in n.names:
                names[a.asname or a.name] = (n.module, a.name)
    return al, names


def is_main_guard(n):
    t = getattr(n, "test", None)
    return isinstance(n, ast.If) and isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__" \
        and len(t.comparators) == 1 and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value == "__main__"


def site_call(c):
    f = c.func
    if isinstance(f, ast.Name) and f.id == "open":
        return "open"
    if isinstance(f, ast.Attribute):
        if isinstance(f.value, ast.Name) and (f.value.id, f.attr) in SUBPROCESS:
            a0 = c.args[0] if c.args else None
            if isinstance(a0, (ast.List, ast.Tuple)) and a0.elts and isinstance(a0.elts[0], ast.Constant) and a0.elts[0].value == "git":
                return None  # a git read (a pinned object, not a working-tree file): covered by the bundle's commit set
            return "subprocess"
        if f.attr == "replace" and len(c.args) != 1:  # str.replace(old, new): not Path.replace(target)
            return None
        if f.attr in FS_METHODS:
            return f.attr
        if f.attr == "load" and isinstance(f.value, ast.Name) and f.value.id == "np":
            return "np.load"
    return None


def target_of(c):
    """A non-git subprocess's target: the repo script named by a ".py" string constant in its argv (e.g. HERE / "make_waitlist.py")."""
    a0 = c.args[0] if c.args else None
    for n in ast.walk(a0) if a0 is not None else []:
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith(".py"):
            return n.value.rsplit("/", 1)[-1][:-3]
    return None


def main(commit):
    mods, trees, funcs, als, names, lines = {}, {}, {}, {}, {}, {}

    def load(m):
        s = src(commit, m)
        mods[m], trees[m], lines[m] = blob(commit, m), ast.parse(s), s.splitlines()
        funcs[m] = outer_funcs(trees[m])
        als[m], names[m] = aliases_of(trees[m])

    for m in MODULES:
        load(m)

    def callees(m, node):
        out = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Call) or isinstance(n, ast.Name) or isinstance(n, ast.Attribute):
                f = n.func if isinstance(n, ast.Call) else n
                if isinstance(f, ast.Name):
                    if f.id in funcs[m]:
                        out.add((m, f.id))
                    elif f.id in names[m] and names[m][f.id][1] in funcs[names[m][f.id][0]]:
                        out.add(names[m][f.id])
                elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in als[m] and als[m][f.value.id] in funcs:
                    tm = als[m][f.value.id]
                    if f.attr in funcs[tm]:
                        out.add((tm, f.attr))
        return out

    def top(m, as_main):  # module-level statements: an `if __name__ == "__main__":` block runs only when the module is __main__
        return [n for n in trees[m].body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and (as_main or not is_main_guard(n))]

    def reach(roots, mods_, cut, as_main):
        seen, todo = set(), list(roots)
        for m in mods_:
            for n in top(m, as_main):
                todo += sorted(callees(m, n))
        while todo:
            k = todo.pop()
            if k in seen or k in cut or k[0] not in funcs or k[1] not in funcs[k[0]]:
                continue
            seen.add(k)
            todo += sorted(callees(k[0], funcs[k[0]][k[1]]))
        return seen

    seen, step = reach(ROOTS, MODULES, set(), False), reach(ROOTS, MODULES, CUT, False)
    sites, imports, targets = [], [], {}

    def scan(m, reached, in_step_of, as_main):
        spans = [(f"{m}.{name}", node.lineno, node.end_lineno, (m, name)) for name, node in funcs[m].items()]
        guards = [] if as_main else [(n.lineno, n.end_lineno) for n in trees[m].body if is_main_guard(n)]
        fn_of = lambda ln: next(((nm, k) for nm, a, b, k in spans if a <= ln <= b), (f"{m}.<module>", None))  # noqa: E731
        for n in ast.walk(trees[m]):
            if any(a <= getattr(n, "lineno", 0) <= b for a, b in guards):
                continue
            if isinstance(n, ast.Call) and site_call(n):
                fn = fn_of(n.lineno)
                if fn[1] is None or fn[1] in reached:
                    s = {"site": f"{m}@{mods[m][:8]}:L{n.lineno}", "module": m, "line": n.lineno, "func": fn[0], "call": site_call(n),
                         "in_step": in_step_of(fn[1]), "text": lines[m][n.lineno - 1].strip()[:200]}
                    if s["call"] == "subprocess":
                        s["target"] = target_of(n)
                    sites.append(s)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                fn = fn_of(n.lineno)
                if fn[1] is None or fn[1] in reached:
                    tops = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else \
                        ([n.module.split(".")[0]] if n.module and not n.level else [])
                    for t in tops:
                        imports.append((m, n.lineno, fn[1], t))

    for m in MODULES:
        scan(m, seen, lambda k: k is None or k in step, False)
    todo = [s for s in sites if s["call"] == "subprocess" and s.get("target")]
    while todo:  # each target script runs as __main__ in a child process; its sites inherit the calling site's in_step
        s0 = todo.pop()
        t = s0["target"]
        if t in targets:
            targets[t] = targets[t] or s0["in_step"]
            continue
        targets[t] = s0["in_step"]
        load(t)
        rt = reach([], [t], set(), True)
        n0 = len(sites)
        scan(t, rt, lambda k, st=s0["in_step"]: st, True)
        todo += [s for s in sites[n0:] if s["call"] == "subprocess" and s.get("target")]
    std = set(sys.stdlib_module_names)
    ok = std | set(MODULES) | set(targets) | THIRD_PARTY_OK
    unexpected = sorted({f"{m}:L{ln} import {t}" for m, ln, k, t in imports
                         if t not in ok and not (t in CUT_ONLY_OK and k is not None and k in seen and k not in step)})
    mdef = {}
    mn = funcs["run_peraction311"]["main"]
    for a, d in zip(mn.args.args[len(mn.args.args) - len(mn.args.defaults):], mn.args.defaults):
        if isinstance(d, ast.Constant):
            mdef[a.arg] = d.value
    sites.sort(key=lambda s: ((MODULES + sorted(targets)).index(s["module"]), s["line"], s["call"]))
    print(json.dumps({"commit": commit, "modules": mods, "targets": sorted(targets), "n_sites": len(sites),
                      "n_in_step": sum(s["in_step"] for s in sites), "imports_unexpected": unexpected, "main_defaults": mdef,
                      "reachable_funcs": sorted(f"{a}.{b}" for a, b in seen), "step_funcs": sorted(f"{a}.{b}" for a, b in step),
                      "sites": sites}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
