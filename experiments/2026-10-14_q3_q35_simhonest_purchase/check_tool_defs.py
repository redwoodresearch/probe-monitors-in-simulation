"""Where a real and a simulated attack run's first policy call differ in tool definitions: tool names, tool_choice, the
policy's generate config, and per tool, which fields differ (with lengths and sha256 prefixes; parameter names and which
parameter fields differ). Names, lengths and hashes only, never description text. ON THE DEVBOX, September python:
  $PY check_tool_defs.py <launch dir> <task key> [<task key> ...]
"""

import hashlib
import json
import sys
from pathlib import Path

POLICY = "vllm/Qwen/Qwen3.5-27B"


def h(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:10]


def first_call(path: Path):
    from inspect_ai.event import ModelEvent
    from inspect_ai.log import read_eval_log

    (s,) = read_eval_log(str(path), resolve_attachments=True).samples
    e = next(e for e in s.events if isinstance(e, ModelEvent) and e.model == POLICY)
    return {t.name: t.model_dump() for t in e.tools}, str(e.tool_choice), e.config.model_dump(exclude_none=True)


def main() -> None:
    launch = Path(sys.argv[1])
    for key in sys.argv[2:]:
        (pa,) = sorted((launch / "attack" / key).glob("*.eval"))
        (pb,) = sorted((launch / "sim_attack@september" / key).glob("*.eval"))
        (a, ac, acfg), (b, bc, bcfg) = first_call(pa), first_call(pb)
        print(f"{key}: tools real {sorted(a)} | sim {sorted(b)} | tool_choice real {ac} sim {bc} | "
              f"policy config equal {acfg == bcfg}")
        if acfg != bcfg:
            print(f"  config keys differing: {sorted(k for k in set(acfg) | set(bcfg) if acfg.get(k) != bcfg.get(k))}")
        for n in sorted(set(a) & set(b)):
            for f in sorted(set(a[n]) | set(b[n])):
                x, y = a[n].get(f), b[n].get(f)
                if x == y:
                    continue
                line = (f"  {n}.{f}: real len {len(json.dumps(x, default=str))} {h(x)} | "
                        f"sim len {len(json.dumps(y, default=str))} {h(y)}")
                if f == "parameters":
                    px, py = (x or {}).get("properties", {}), (y or {}).get("properties", {})
                    line += f"; params real {sorted(px)} sim {sorted(py)}; required real {(x or {}).get('required')} " \
                            f"sim {(y or {}).get('required')}; schema keys differing " \
                            f"{sorted(k for k in set(x or {}) | set(y or {}) if (x or {}).get(k) != (y or {}).get(k))}"
                    for k in sorted(set(px) & set(py)):
                        if px[k] != py[k]:
                            diff = sorted(ff for ff in set(px[k]) | set(py[k]) if px[k].get(ff) != py[k].get(ff))
                            line += f"; {k}: fields {diff}"
                print(line)


if __name__ == "__main__":
    main()
