"""09's sim2 gap-fill s4r regex: positive and negative cases (09 requires that it rejects ordinary ilnm_s4r names). Pure string checks.

  python3 -I test_folder_regex.py
"""

import ast
import importlib.util
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _const(path, name):
    """A module-level re.compile(r"...") pattern, read from the file without importing it (the scorer imports modal)."""
    for node in ast.parse((HERE / path).read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return re.compile(node.value.args[0].value)
    raise SystemExit(f"{name} not found in {path}")


VERBATIM = r"^ilnm_s4rgap_sim2(_s[0-9]+)?_v1(_try[0-9]+)?__sim_attack$"
YES = ["ilnm_s4rgap_sim2_v1__sim_attack", "ilnm_s4rgap_sim2_s1_v1__sim_attack", "ilnm_s4rgap_sim2_v1_try2__sim_attack", "ilnm_s4rgap_sim2_s12_v1_try3__sim_attack"]
NO = ["ilnm_s4r_sim2_v1__sim_attack", "ilnm_s4r_run3_v1__attack", "ilnm_s4r_sim2_s1_v1__sim_attack", "ilnm_s4d_sim2_v1__sim_attack", "ilnm_s4d_sim2_v2__sim_attack",
      "ilnm_s1_sim2_v1__sim_attack", "ilnm_s2r_sim2_v1__sim_attack", "ilnm_s4rgap_run3_v1__attack", "ilnm_s4rgap_sim2_v1__attack", "ilnm_s4rgap_sim2_v2__sim_attack",
      "ilnm_s4rgap_sim2_v1__sim_attack@september", "ilnm_s4rgap_sim2_cutfull_v1__sim_attack", "ilnm_s4rgap_sim2_cutfull_gate_v1__sim_attack",
      "ilnm_s4rgap_sim2_v1_try__sim_attack", "ilnm_s4rgap_sim2_v1__sim_attack ", "xilnm_s4rgap_sim2_v1__sim_attack", "ilold_oldsel_sim2_v1__sim_attack"]
for path in ("score_step3_il.py", "build_il_message.py"):
    rx = _const(path, "S4RGAP_RE")
    assert rx.pattern == VERBATIM, (path, rx.pattern)
    bad = [n for n in YES if not rx.fullmatch(n)] + [n for n in NO if rx.fullmatch(n)]
    assert not bad, (path, bad)
sel = _const("score_step3_il.py", "FOLDER_RE")
assert not any(sel.fullmatch(n) for n in YES) and sel.fullmatch("ilold_oldsel_sim2_v1__sim_attack")  # the two grammars stay disjoint
spec = importlib.util.spec_from_file_location("build_il_message", HERE / "build_il_message.py")  # the adapter's world check (Codex s4rgap r1)
bim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bim)
assert all(bim.world_ok(n, "sim2") and not bim.world_ok(n, "run3") for n in YES if n not in bim.SUPERSEDED)
assert bim.world_ok("ilold_oldsel_sim2_v1__sim_attack", "sim2") and bim.world_ok("ilold_oldsel_run3_v1__attack", "run3")
assert not any(bim.world_ok(n, w) for n in NO if not n.startswith("ilold_") for w in ("sim2", "run3"))
assert not bim.world_ok("ilnm_s4rgap_sim2_v1_try2__sim_attack", "sim2") and bim.world_ok("ilnm_s4rgap_sim2_v1_try3__sim_attack", "sim2")  # 09: _try2 SUPERSEDED
print(f"PASS: {len(YES)} accepted, {len(NO)} rejected, in both files; verbatim; disjoint from the sel97 grammar; adapter world check")
