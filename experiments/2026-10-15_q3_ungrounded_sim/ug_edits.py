"""The four text edits of the ungrounded arm (RULE.md), each an exact replacement asserted to match ONCE.

  ug_edits.py tree <tree>/src/runner/settings.py            # the sims patch (in the COPY of the September tree)
  ug_edits.py runyaml <harness copy>/config/run.yaml <src_sha256>   # the simulator entry september_ungrounded
  ug_edits.py preflight <harness copy>/check_sim_samples.py         # the negative control for `ungrounded: true`
  ug_edits.py tasks <harness copy>/tasks.py                          # la_sim_attack_sept's real == honest assertion

Never run on a frozen file: build_ug.sh runs it only on the copies.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    n = text.count(old)
    assert n == 1, f"{path}: expected the anchor exactly once, found {n}: {old[:80]!r}"
    path.write_text(text.replace(old, new))


TREE_HELPER_ANCHOR = '''def _orig_trajectory_mode(name: str = "BA_HONEST_LOG_MODE") -> OrigTrajectoryMode:
'''
TREE_HELPER = '''def _no_honest_reference(name: str = "BA_NO_HONEST_REFERENCE") -> bool:
    """The UNGROUNDED arm (probe-shift experiments/2026-10-15_q3_ungrounded_sim): "1" drops the original-trajectory
    reference (BA_HONEST_LOG) entirely, so the simulator never sees the real honest run and the manifest's
    `condition.affordances` loses `original_trajectory`. Unset or "0" is the frozen behaviour; any other value is an
    error rather than a silent fallback."""
    value = os.environ.get(name, "0").strip()
    assert value in ("0", "1"), f"{name} must be 0 or 1, got {value!r}"
    return value == "1"


'''
TREE_CALL_OLD = '''            honest_logs=_items("BA_HONEST_LOG"),
'''
TREE_CALL_NEW = '''            honest_logs=() if _no_honest_reference() else _items("BA_HONEST_LOG"),
'''

RUNYAML_ANCHOR = '''  # The Qwen3-32B native-window arm's tree (Aswin, 10-09): September's plus config/policy_context_guard.patch, which
'''
RUNYAML_ENTRY = '''  # UNGROUNDED September (experiments/2026-10-15_q3_ungrounded_sim/RULE.md; Aswin ~00:10 via 0d): the september tree
  # plus config/ungrounded_no_honest_reference.patch, under which BA_NO_HONEST_REFERENCE=1 drops the original-trajectory
  # reference (BA_HONEST_LOG); BA_REAL_LOG still defines each cell (forked at 0: no real action is seeded). Only this
  # harness copy has the entry. check_sim_samples.py runs the NEGATIVE control for it (`ungrounded: true`).
  september_ungrounded:
    base_tree: ~/sims/simulated-control-evals-sept
    base_src_sha256: 4bcdc038063939cee1874f55dd730cef96ebc8f8e49b1bb97f5639e04aeb5937
    tree: ~/phase2/sims-sept-ungrounded
    extra_patches: [config/policy_strict_tools_off.patch, config/ungrounded_no_honest_reference.patch]
    venv: ~/sims/simulated-control-evals-sept/.venv
    src_sha256: {sha}
    env: {{BA_NO_HONEST_REFERENCE: "1"}}
    ungrounded: true
'''

TASKS_OLD = '''    assert settings.real_logs == settings.honest_logs, "the reference is the fork source, as in September"
'''
TASKS_NEW = '''    # The UNGROUNDED arm (experiments/2026-10-15_q3_ungrounded_sim, BA_NO_HONEST_REFERENCE=1) drops the reference:
    # honest_logs is empty and the fork source alone defines the cell. Every other run: September's assertion, unchanged.
    if os.environ.get("BA_NO_HONEST_REFERENCE") == "1":
        assert settings.real_logs and not settings.honest_logs, settings
    else:
        assert settings.real_logs == settings.honest_logs, "the reference is the fork source, as in September"
'''

PREFLIGHT_OLD = '''                # No reasoning blocks, or no output marker found: the test would prove nothing, so it doesn't pass.
                if row["reasoning_marker_in_prompt"] or not blocks or not row["output_marker_in_prompt"]:
                    failed.append(f"{cell['cell']} ({mode})")
'''
PREFLIGHT_NEW = '''                # No reasoning blocks, or no output marker found: the test would prove nothing, so it doesn't pass.
                # An UNGROUNDED simulator (run.yaml `ungrounded: true`; experiments/2026-10-15_q3_ungrounded_sim) is the
                # NEGATIVE control instead: neither marker may reach the prompt (the reference is dropped), and the copy
                # must carry both kinds of marker, or the test proves nothing.
                row["ungrounded"] = bool(simulator.get("ungrounded"))
                if row["ungrounded"]:
                    if row["reasoning_marker_in_prompt"] or row["output_marker_in_prompt"] or not blocks or not outputs:
                        failed.append(f"{cell['cell']} ({mode})")
                elif row["reasoning_marker_in_prompt"] or not blocks or not row["output_marker_in_prompt"]:
                    failed.append(f"{cell['cell']} ({mode})")
'''


def main() -> None:
    what, path = sys.argv[1], Path(sys.argv[2])
    if what == "tree":
        replace_once(path, TREE_HELPER_ANCHOR, TREE_HELPER + TREE_HELPER_ANCHOR)
        replace_once(path, TREE_CALL_OLD, TREE_CALL_NEW)
    elif what == "runyaml":
        sha = sys.argv[3]
        assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), sha
        assert "september_ungrounded:" not in path.read_text()
        replace_once(path, RUNYAML_ANCHOR, RUNYAML_ENTRY.format(sha=sha) + RUNYAML_ANCHOR)
    elif what == "preflight":
        replace_once(path, PREFLIGHT_OLD, PREFLIGHT_NEW)
    elif what == "tasks":
        replace_once(path, TASKS_OLD, TASKS_NEW)
    else:
        raise SystemExit(f"unknown edit {what!r}")
    print(f"edited {what}: {path}")


if __name__ == "__main__":
    main()
