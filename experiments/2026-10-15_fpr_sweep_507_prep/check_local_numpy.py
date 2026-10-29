"""Code reading only (507-prep, 10-17): which numpy imports can a LOCAL entrypoint reach? The named modal161 client has no numpy.

ADVISORY: a best-effort static scan, NOT sound (2e ruling 10-17; its known false-CLEAN limits R1-R8 are in
NOTE_routec_launch_skeleton.md). Reachable = the entry function, plus every loaded module's import-time code (module-level statements, class
bodies, decorators, default values and annotations: Astra signoff1 R4), plus every function with a non-Modal decorator (it may sit in a
registry). From those it follows every reference to a function or class in --dir: by name, by EVERY import binding of that name in the
module, whatever its scope (R6), by module alias attribute, and any `.attr` that names a method of a loaded class. Functions named by a
module-level binding (`HANDLERS = {"x": work}`) are reached through the import-time code (R5). It never enters a Modal function body
(`@app.function` / `@app.cls`: it runs in the container). `--cut mod.fn` stops at a function the caller has shown is not entered on the
path in question (say why in the NOTE). Dynamic lookups it cannot follow (getattr with a non-literal name, globals/locals/vars,
eval/exec/compile, __import__/import_module, __dict__, pickle.load(s), a module passed as a value) make the result INCONCLUSIVE unless
listed with `--allow mod.py:line` after review.

Result line: CLEAN (exit 0), NUMPY (exit 1) or INCONCLUSIVE (exit 2).

    uv run --no-project --python 3.12 python -I check_local_numpy.py run_peraction311:main --cut run_peraction311.run_rows,run_peraction311.finish_rows \
        --allow run_peraction311.py:4542
"""

import argparse
import ast
import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUS = {"np", "np_", "numpy", "pd", "pandas", "scipy", "sklearn", "torch"}
REMOTE_DECOS = ("app.function", "app.cls")
DYN_CALLS = {"globals", "locals", "vars", "eval", "exec", "compile", "__import__"}


class Index:
    def __init__(self, root):
        self.root, self.trees, self.funcs, self.aliases = root, {}, {}, {}
        self.classes, self.methods, self.remote, self.roots = {}, collections.defaultdict(set), set(), set()

    def add_func(self, mod, name, node):
        k = (mod, name)
        self.funcs[k] = node
        decos = [ast.unparse(d) for d in node.decorator_list]
        if any(d.startswith(REMOTE_DECOS) for d in decos):
            self.remote.add(k)
        elif any(not d.startswith("app.local_entrypoint") for d in decos):
            self.roots.add(k)  # a non-Modal decorator may register it anywhere
        return k

    def load(self, mod):
        if mod in self.trees or not (self.root / f"{mod}.py").exists():
            return
        tree = self.trees[mod] = ast.parse((self.root / f"{mod}.py").read_text())
        al = self.aliases[mod] = collections.defaultdict(set)  # R6: every binding of a name, from any scope
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    al[a.asname or a.name.split(".")[0]].add(("mod", a.name if a.asname else a.name.split(".")[0]))
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    al[a.asname or a.name].add(("fn", n.module, a.name) if n.module else ("mod", a.name))

        def head(f):  # evaluated when the def statement runs
            a = f.args
            anns = [x.annotation for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg] if x is not None and x.annotation]
            return f.decorator_list + a.defaults + [d for d in a.kw_defaults if d] + anns + ([f.returns] if f.returns else [])

        init = []
        for st in tree.body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.add_func(mod, st.name, st)
                init += head(st)
            elif isinstance(st, ast.ClassDef):
                ms = self.classes[(mod, st.name)] = []
                init += st.decorator_list + st.bases + [k.value for k in st.keywords]
                for b in st.body:
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        ms.append(self.add_func(mod, f"{st.name}.{b.name}", b))
                        self.methods[b.name].add(ms[-1])
                        init += head(b)
                    else:
                        init.append(b)
            else:
                init.append(st)
        self.funcs[(mod, "<init>")] = ast.Module(body=[x if isinstance(x, ast.stmt) else ast.Expr(value=x) for x in init], type_ignores=[])
        self.roots.add((mod, "<init>"))
        for ts in list(al.values()):
            for t in ts:
                self.load(t[1])

    def targets(self, mod, name):
        out = [(mod, name)] if (mod, name) in self.funcs or (mod, name) in self.classes else []
        out += [(t[1], t[2]) for t in self.aliases[mod].get(name, ()) if t[0] == "fn"]
        return out

    def expand(self, k):
        return [k] if k in self.funcs else self.classes.get(k, [])

    def refs(self, mod, node):
        """-> (reached keys with lines, dynamic sites)."""
        out, dyn = [], []
        bases = {id(n.value) for n in ast.walk(node) if isinstance(n, ast.Attribute)}
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                for t in self.targets(mod, n.id):
                    out += [(k, n.lineno) for k in self.expand(t)]
                if id(n) not in bases and any(t[0] == "mod" for t in self.aliases[mod].get(n.id, ())) and (mod, n.id) not in self.funcs:
                    dyn.append((n.lineno, f"module {n.id} used as a value"))
            elif isinstance(n, ast.Attribute):
                if isinstance(n.value, ast.Name):
                    for t in self.aliases[mod].get(n.value.id, ()):
                        if t[0] == "mod":
                            out += [(k, n.lineno) for k in self.expand((t[1], n.attr))]
                        if t == ("mod", "pickle") and n.attr in ("load", "loads"):
                            dyn.append((n.lineno, "pickle.load(s) can import any module"))
                out += [(k, n.lineno) for k in self.methods.get(n.attr, ())]
                if n.attr in ("import_module", "__dict__", "__getattribute__"):
                    dyn.append((n.lineno, f".{n.attr}"))
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                if n.func.id in DYN_CALLS:
                    dyn.append((n.lineno, f"{n.func.id}()"))
                if n.func.id == "getattr" and not (len(n.args) >= 2 and isinstance(n.args[1], ast.Constant)):
                    dyn.append((n.lineno, "getattr with a non-literal name"))
        return out, dyn


def has_np(node):
    lines = [n.lineno for n in ast.walk(node) if isinstance(n, (ast.Import, ast.ImportFrom))
             and ("numpy" in [a.name.split(".")[0] for a in n.names] or (getattr(n, "module", "") or "").split(".")[0] == "numpy")]
    return lines, sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in SUS})


def check(entry, cut=(), allow=(), root=HERE):
    """-> (verdict, report lines)."""
    mod, fn = entry.split(":")
    ix = Index(Path(root))
    ix.load(mod)
    start = sorted({(mod, fn)} | ix.roots)
    par = {k: None for k in start}
    q, dyn = collections.deque(start), []
    while q:
        k = q.popleft()
        out, d_ = ix.refs(k[0], ix.funcs[k])
        dyn += [(f"{k[0]}.py:{ln}", what, k) for ln, what in d_]
        for r, ln in out:
            if r not in par and r not in ix.remote and f"{r[0]}.{r[1]}" not in cut:
                par[r] = (k, ln)
                q.append(r)
    lines, hits = [], 0
    for k in sorted(par):
        nl, used = has_np(ix.funcs[k])
        if nl or used:
            hits += 1
            chain, c = [], k
            while par[c]:
                c, ln = par[c]
                chain.append(f"{c[0]}.{c[1]}:{ln}")
            lines.append(f"NUMPY {k[0]}.py:{nl[0] if nl else '-'} {k[1]} uses {used} <- {' <- '.join(chain) or '(root)'}")
    open_ = [x for x in dyn if x[0] not in allow]
    lines += [f"{'ALLOWED' if x[0] in allow else 'DYNAMIC'} {x[0]} {x[1]} (in {x[2][0]}.{x[2][1]})" for x in dyn]
    verdict = "NUMPY" if hits else "INCONCLUSIVE" if open_ else "CLEAN"
    lines.append(f"entry {entry} cut {sorted(cut)} allow {sorted(allow)}: modules {len(ix.trees)}, roots {len(ix.roots)}, reachable {len(par)}, "
                 f"Modal functions skipped {len(ix.remote)}, numpy-using {hits}, dynamic {len(dyn)} ({len(open_)} not allowed): result {verdict}")
    return verdict, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", help="module:function, e.g. run_peraction311:main")
    ap.add_argument("--cut", default="")
    ap.add_argument("--allow", default="", help="reviewed dynamic sites, mod.py:line,...")
    ap.add_argument("--dir", default=str(HERE))
    a = ap.parse_args()
    verdict, lines = check(a.entry, {c for c in a.cut.split(",") if c}, {x for x in a.allow.split(",") if x}, a.dir)
    print(*lines, sep="\n")
    sys.exit({"CLEAN": 0, "NUMPY": 1, "INCONCLUSIVE": 2}[verdict])


if __name__ == "__main__":
    main()
