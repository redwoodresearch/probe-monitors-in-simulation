"""Synthetic tests of check_local_numpy.check (tiny module fixtures in a temp dir; no repo code, no record): each Astra signoff1 blind spot
(R4 import-time code, R5 module-level bindings, R6 scoped imports) must give NUMPY, a dynamic lookup INCONCLUSIVE, a Modal body nothing.

  uv run --no-project --python 3.12 python experiments/2026-10-15_fpr_sweep_507_prep/test_check_local_numpy.py
"""

import tempfile
from pathlib import Path

import check_local_numpy as K

NP = "def work():\n    import numpy as np\n    return np\n"
SAFE = "def work():\n    return 1\n"
CASES = {  # name: ({module: source}, entry, expected verdict)
    "clean: main calls a numpy-free helper": ({"m": "import safe\n\ndef main():\n    return safe.work()\n", "safe": SAFE}, "m:main", "CLEAN"),
    "direct: main calls a numpy helper": ({"m": "import needs_np\n\ndef main():\n    return needs_np.work()\n", "needs_np": NP}, "m:main", "NUMPY"),
    "R4: module-level import numpy in an imported module": ({"m": "import lib\n\ndef main():\n    return 1\n", "lib": "import numpy\n"}, "m:main", "NUMPY"),
    "R4: a decorator that calls a numpy function": (
        {"m": "from needs_np import work\n\n\ndef deco(f):\n    work()\n    return f\n\n\n@deco\ndef helper():\n    return 1\n\n\ndef main():\n    return 1\n",
         "needs_np": NP}, "m:main", "NUMPY"),
    "R4: a default value that calls a numpy function": (
        {"m": "from needs_np import work\n\n\ndef main(x=work()):\n    return x\n", "needs_np": NP}, "m:main", "NUMPY"),
    "R5: a module-level registry indexed at run time": (
        {"m": "from needs_np import work\n\nHANDLERS = {'x': work}\n\n\ndef main():\n    return HANDLERS['x']()\n", "needs_np": NP}, "m:main", "NUMPY"),
    "R5: a registry filled by a non-Modal decorator": (
        {"m": "REG = {}\n\n\ndef register(f):\n    REG[f.__name__] = f\n    return f\n\n\n@register\ndef g():\n    import numpy\n    return numpy\n\n\n"
              "def main():\n    return REG['g']()\n"}, "m:main", "NUMPY"),
    "R6: a function-local import of a safe name does not hide the module-level one": (
        {"m": "from needs_np import work\n\n\ndef other():\n    from safe import work\n    return work()\n\n\ndef main():\n    return work()\n",
         "needs_np": NP, "safe": SAFE}, "m:main", "NUMPY"),
    "a method call on an object whose class method imports numpy": (
        {"m": "class R:\n    def run(self):\n        import numpy\n        return numpy\n\n\ndef main(o):\n    return o.run()\n"}, "m:main", "NUMPY"),
    "dynamic: getattr with a non-literal name": (
        {"m": "import safe\n\n\ndef main(name):\n    return getattr(safe, name)()\n", "safe": SAFE}, "m:main", "INCONCLUSIVE"),
    "dynamic: globals() lookup": ({"m": "def main(name):\n    return globals()[name]()\n"}, "m:main", "INCONCLUSIVE"),
    "dynamic: pickle.loads can import any module": ({"m": "import pickle\n\n\ndef main(b):\n    return pickle.loads(b)\n"}, "m:main", "INCONCLUSIVE"),
    "a Modal function body is not entered": (
        {"m": "import modal\napp = modal.App('x')\n\n\n@app.function()\ndef remote():\n    import numpy\n    return numpy\n\n\n"
              "def main():\n    return remote.remote()\n"}, "m:main", "CLEAN"),
    "--cut stops at a function shown not to be entered": (
        {"m": "from needs_np import work\n\n\ndef local_branch():\n    return work()\n\n\ndef main(remote=True):\n    return None if remote else local_branch()\n",
         "needs_np": NP}, "m:main", "CLEAN"),
}
CUT = {"--cut stops at a function shown not to be entered": {"m.local_branch"}}

results = {}
for name, (mods, entry, want) in CASES.items():
    with tempfile.TemporaryDirectory() as d:
        for m, src in mods.items():
            (Path(d) / f"{m}.py").write_text(src)
        got, lines = K.check(entry, CUT.get(name, set()), set(), d)
        results[name] = got == want
        print(("PASS " if got == want else "FAIL ") + f"{name}: {got} (want {want})")
with tempfile.TemporaryDirectory() as d:  # an allowed dynamic site is reported but does not make the result INCONCLUSIVE
    (Path(d) / "m.py").write_text("def main(a=1):\n    return dict(locals())\n")
    ok = K.check("m:main", set(), {"m.py:2"}, d)[0] == "CLEAN" and K.check("m:main", set(), set(), d)[0] == "INCONCLUSIVE"
    results["--allow m.py:2 clears only that site"] = ok
    print(("PASS " if ok else "FAIL ") + "--allow m.py:2 clears only that site")
assert all(results.values()), "check_local_numpy tests failed"
print(f"{len(results)} checks pass")
