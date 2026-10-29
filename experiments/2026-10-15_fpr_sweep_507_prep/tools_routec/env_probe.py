"""Route C launch-line environment probes (a0b05's REQUIRED, 10-18: no inline `python -c` in a launch block; scripts live in files).
Each subcommand does exactly what one former inline check did; its exit status (and, for modal-version, its one printed line) is the
check. Run with -I, so neither this file's folder nor the working directory is on sys.path, as with the former `-I -c`.

  "$PY"  -I tools_routec/env_probe.py modal-version     prints modal.__version__   (was: -I -c 'import modal; print(modal.__version__)')
  "$PYM" -I tools_routec/env_probe.py import-modal      exit 0 iff modal imports   (was: -I -c 'import modal'; the caller keeps 2>/dev/null)
  "$PYM" -I tools_routec/env_probe.py no-numpy DIR...   exit 0 iff numpy is not findable with DIR... prepended to sys.path, else 3
                                                        (was: -I -c 'import sys, importlib.util; sys.path[:0] = sys.argv[1:];
                                                         sys.exit(0 if importlib.util.find_spec("numpy") is None else 3)' DIR...)
Any other subcommand or argument count exits 2.
"""
import sys


def main(argv):
    if argv == ["modal-version"]:
        import modal

        print(modal.__version__)
        return 0
    if argv == ["import-modal"]:
        import modal  # noqa: F401  (an import error exits non-zero, as the inline form did)

        return 0
    if len(argv) >= 2 and argv[0] == "no-numpy":
        import importlib.util

        sys.path[:0] = argv[1:]
        return 0 if importlib.util.find_spec("numpy") is None else 3
    print("env_probe: unknown subcommand or arguments", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
