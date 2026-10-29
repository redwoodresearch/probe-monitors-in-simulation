"""Print one Modal FunctionCall's return value as JSON: the route C reply read (Stage D / kappa REPLYW4; S7R).

  "$PY" -I "$E/fc_reply.py" <fc-id> <timeout_s> < /dev/null

stdout is exactly json.dumps(modal.FunctionCall.from_id(fc).get(timeout=timeout_s)) plus a newline, and nothing else. Stdlib and
modal only. Any exception (a bad argument, the call failing or timing out, a value JSON cannot encode) exits non-zero with stdout empty.
"""
import json
import sys

import modal


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: fc_reply.py <fc-id> <timeout_s>")
    fc, timeout_s = argv[0], int(argv[1])
    print(json.dumps(modal.FunctionCall.from_id(fc).get(timeout=timeout_s)))


if __name__ == "__main__":
    main(sys.argv[1:])
