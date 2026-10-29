"""Which calls the off-policy call_end readout and the on-policy command readout read at the same token: the matched
subset for the on-policy-versus-off-policy comparison (shared by the off-policy refit and the on-policy replay).

The on-policy readout (`reader.command_end_positions`, on aswin/q3-stage1-awareness) is the last token inside the value
of the call's command parameter (`command_param`, copied verbatim below until that branch merges; then import it).
The off-policy readout is call_end, the last token of the call's serialized arguments. They pick the same token only
when the call's last argument is its non-empty command parameter (a1's tokenizer test, 2026-10-08). So the matched
rule keeps a call iff `command_param(tool, arguments)` is not None and is the last argument.
Excluded, among others: a trailing numeric timeout (call_end reads a digit), text_editor view with a trailing
view_range (call_end reads "]"), a str_replace deleting text (no command value), and any tool whose last argument is
not a string.
`call_end_kind_from_arguments` keeps q3's coarser rule 1 (call_end_kind 0) for reference.
Coverage depends on the denominator; the README of experiments/2026-10-08_impactful_refit_bootstrap reports it per
world on Q2's attack calls.
"""

TEXT_EDITOR_PARAMS = ("file_text", "new_str", "insert_text", "path")
BASH_PARAMS = ("cmd", "command")


def command_param(name: str, arguments) -> str | None:
    """Copied verbatim from probe_shift.reader on aswin/q3-stage1-awareness (a1's on-policy readout).
    The parameter holding a call's command or code: the readout pre-registered for Phase 2 probes (rule 1).
    text_editor by its command: create -> file_text, str_replace -> new_str, insert -> insert_text or new_str, view and
    undo_edit -> path (any other command: the first of those present). bash and `*_bash` tools: cmd, then command. Any
    other tool: its last string-valued parameter. Never a numeric parameter. None when the chosen value is missing or
    empty (e.g. a str_replace deleting text): the call has no readout and callers drop and count it."""
    if not isinstance(arguments, dict):
        return None
    if name == "text_editor":
        by_command = {"create": ("file_text",), "str_replace": ("new_str",), "insert": ("insert_text", "new_str"),
                      "view": ("path",), "undo_edit": ("path",)}
        order = by_command.get(arguments.get("command"), TEXT_EDITOR_PARAMS)
        key = next((k for k in order if k in arguments), None)
    elif name == "bash" or name.endswith("_bash"):
        key = next((k for k in BASH_PARAMS if k in arguments), None)
    else:
        strs = [k for k, v in arguments.items() if isinstance(v, str)]
        key = strs[-1] if strs else None
    if key is None or not isinstance(arguments[key], str) or not arguments[key].strip():
        return None
    return key


def matched_call(name: str, arguments) -> bool:
    """The matched subset: the call's last argument is its non-empty command parameter."""
    param = command_param(name, arguments)
    return param is not None and param == list(arguments)[-1]


def call_end_kind_from_arguments(arguments) -> int:
    """q3's rule 1 by arguments: 0 content; 1 a trailing numeric timeout; 2 a trailing string ending in a newline."""
    if isinstance(arguments, dict) and arguments:
        key = list(arguments)[-1]
        value = arguments[key]
        if key == "timeout" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return 1
        if isinstance(value, str) and value.endswith("\n"):
            return 2
    return 0
