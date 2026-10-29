"""A new version of a counted-attempt manifest or run table whose launch ids are bare (d9 10-15: the T196 layout's runs are relayed
as runs/<id>, and box_files.py / chain.py wrote that path as the launch id; every scorer keys on the bare id, the run folder's name).
Only `launch_id` and `replacement_launch_id` values change; the new file names the file it supersedes, and the original stays
(append-only) with a SUPERSEDED note beside it. Keys only: never opens a transcript.

  python3 make_bare_id_v2.py <src.json> <out.json> <reason>
"""
import hashlib
import json
import sys
from pathlib import Path

ID_KEYS = ("launch_id", "replacement_launch_id")


def bare(x, path, changed):
    """x with every prefixed launch id made bare; each change appended to `changed` as (path, old, new)."""
    if isinstance(x, list):
        return [bare(v, f"{path}[{i}]", changed) for i, v in enumerate(x)]
    if not isinstance(x, dict):
        return x
    out = {}
    for k, v in x.items():
        if k in ID_KEYS and isinstance(v, str) and "/" in v:
            out[k] = v.rsplit("/", 1)[-1]
            changed.append((f"{path}.{k}", v, out[k]))
        else:
            out[k] = bare(v, f"{path}.{k}", changed)
    return out


def diff_paths(a, b, path=""):
    """Every path where a and b differ (structure must match)."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), f"{path}: keys differ"
        return [p for k in a for p in diff_paths(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{path}: lengths differ"
        return [p for i in range(len(a)) for p in diff_paths(a[i], b[i], f"{path}[{i}]")]
    return [] if a == b else [path]


def main():
    src, out, reason = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    if out.exists():
        sys.exit(f"{out} exists (append-only)")
    old = json.loads(src.read_text())
    changed = []
    new = bare(old, "", changed)
    if not changed:
        sys.exit(f"{src}: no prefixed launch id")
    assert sorted(diff_paths(old, new)) == sorted(p for p, _, _ in changed), "a field other than a launch id changed"
    assert all(o.rsplit("/", 1)[-1] == n and "/" not in n for _, o, n in changed)
    assert "supersedes" not in new or new["supersedes"] != src.name
    new["supersedes"] = src.name  # combine_final drops the run table a later version names
    new["supersedes_reason"] = reason
    out.write_text(json.dumps(new, indent=1) + "\n")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    pairs = sorted({(o, n) for _, o, n in changed})
    note = src.with_name(src.stem + ".SUPERSEDED.md")
    if note.exists():
        sys.exit(f"{note} exists")
    note.write_text(f"# SUPERSEDED\n\n`{src.name}` is superseded by `{out.name}` (sha256 {sha}): {reason}. Only these launch-id values changed "
                    f"({len(changed)} fields): " + "; ".join(f"`{o}` -> `{n}`" for o, n in pairs) + ". Every other field is identical; "
                    "the original stays unchanged (append-only).\n")
    print(json.dumps({"src": str(src), "out": str(out), "out_sha256": sha, "fields_changed": len(changed),
                      "by_key": {k: sum(p.endswith("." + k) for p, _, _ in changed) for k in ID_KEYS}, "values": pairs}))


if __name__ == "__main__":
    main()
