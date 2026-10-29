"""fc_reply.py against a fake `modal` module only: synthetic, no network, no temp files. Each case runs the script in a child process
(this file re-run with -I and --child, which installs the fake module and runs fc_reply.py's text as __main__), so the exit codes and
stdout are real.

1. A reply: exit 0; stdout is exactly json.dumps(the value) plus a newline; the fake saw from_id(<fc-id>) and get(timeout=<int>) by keyword.
2. Each failure exits non-zero with stdout empty: get raises (a timeout), from_id raises, a value JSON cannot encode; one or three
   arguments and a timeout that is not an integer, all three before from_id is called.
3. The script imports json, sys and modal only.
4. Mutants of the script (exceptions swallowed; default=str; an extra stdout line) each fail their case: the cases are load-bearing.

  uv run --offline --no-project --python 3.12 python -I experiments/2026-10-15_fpr_sweep_507_prep/test_fc_reply.py
"""
import ast
import json
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "fc_reply.py"
SRC = SCRIPT.read_text()
FC = "fc-01SYNTHETIC00000000000000"
VALUE = {"record": "xfer/fpr507prep/records/results/pertask_full_synthetic.json", "sha256": "a" * 64, "npz": None, "npz_sha256": None}
MUTANTS = {
    "none": None,
    "swallow": ("    main(sys.argv[1:])\n", "    try:\n        main(sys.argv[1:])\n    except Exception:\n        pass\n"),
    "default_str": (".get(timeout=timeout_s)))", ".get(timeout=timeout_s), default=str))"),
    "extra_line": ("    print(json.dumps(", '    print("reply")\n    print(json.dumps('),
}


def child(mode, mutant, args):
    class FunctionCall:
        @staticmethod
        def from_id(fc):
            print(f"CALL from_id {fc}", file=sys.stderr)
            if mode == "from_id_raises":
                raise ValueError("synthetic")
            return FunctionCall()

        def get(self, *, timeout):
            print(f"CALL get timeout={timeout!r} {type(timeout).__name__}", file=sys.stderr)
            if mode == "get_raises":
                raise TimeoutError("synthetic")
            return {"bad": object()} if mode == "nonjson" else VALUE

    fake = types.ModuleType("modal")
    fake.FunctionCall = FunctionCall
    sys.modules["modal"] = fake
    src = SRC
    if MUTANTS[mutant]:
        old, new = MUTANTS[mutant]
        assert SRC.count(old) == 1, mutant
        src = SRC.replace(old, new)
    sys.argv = [str(SCRIPT), *args]
    exec(compile(src, str(SCRIPT), "exec"), {"__name__": "__main__"})


def run(mode, args, mutant):
    p = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--child", mode, mutant, *args],
                       capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    return p.returncode, p.stdout, p.stderr


def cases(mutant):
    out = {}
    rc, so, se = run("ok", [FC, "7200"], mutant)
    out["ok"] = rc == 0 and so == json.dumps(VALUE) + "\n" and f"CALL from_id {FC}\n" in se and "CALL get timeout=7200 int\n" in se
    for mode, seen in (("get_raises", "CALL get "), ("from_id_raises", "CALL from_id "), ("nonjson", "CALL get ")):
        rc, so, se = run(mode, [FC, "7200"], mutant)
        out[mode] = rc != 0 and so == "" and seen in se
    for name, args in (("one_arg", [FC]), ("three_args", [FC, "7200", "x"]), ("bad_timeout", [FC, "72s"])):
        rc, so, se = run("ok", args, mutant)
        out[name] = rc != 0 and so == "" and "CALL" not in se
    return out


def imports(src):
    names = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            names |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            names.add(f"from {n.module}")
    return names


if __name__ == "__main__" and sys.argv[1:2] == ["--child"]:
    child(sys.argv[2], sys.argv[3], sys.argv[4:])
elif __name__ == "__main__":
    res = {}
    base = cases("none")
    for k, v in base.items():
        res[f"1-2. {k}: as stated (exit code, exact stdout, the fake's calls)"] = v
    res["3. imports are json, sys and modal only"] = imports(SRC) == {"json", "sys", "modal"}
    for mutant, case in (("swallow", "get_raises"), ("swallow", "from_id_raises"), ("default_str", "nonjson"), ("extra_line", "ok")):
        res[f"4. mutant {mutant} fails case {case}"] = not cases(mutant)[case]
    for k, v in res.items():
        print(("PASS " if v else "FAIL ") + k)
    print(f"PASS {all(res.values())} ({sum(res.values())}/{len(res)})")
    sys.exit(0 if all(res.values()) else 1)
