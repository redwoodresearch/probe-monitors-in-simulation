"""Carry the constants decisions (class, status, sources, note) from one constants file's overrides to a newer constants file BY
LITERAL (sha256 of each row's text): every row of the new file gets its literal's decision; one literal, one decision (else STOP).
No literal is printed: counts only.

  uv run --no-project --python 3.12 python -I tools/carry_overrides_by_literal.py <old constants.jsonl> <old overrides.json> <new constants.jsonl> <out.json> <note>
"""
import hashlib
import json
import sys


def rows(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def main(old_c, old_o, new_c, out, note):
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
    ov = json.load(open(old_o, encoding="utf-8"))
    by_lit = {}
    for r in rows(old_c):
        d = ov["rows"][str(r["row"])]
        k = sha(r["text"])
        if k in by_lit and by_lit[k] != d:
            raise SystemExit("STOP: one literal has two decisions in the old overrides")
        by_lit[k] = d
    new = rows(new_c)
    miss = [r["row"] for r in new if sha(r["text"]) not in by_lit]
    if miss:
        raise SystemExit(f"STOP: {len(miss)} new rows have a literal with no old decision")
    res = {"what": ov.get("what", "") + " | " + note, "rows": {str(r["row"]): by_lit[sha(r["text"])] for r in new}}
    json.dump(res, open(out, "x", encoding="utf-8"), indent=1, sort_keys=True)
    cls = {}
    for d in res["rows"].values():
        cls[d["class"]] = cls.get(d["class"], 0) + 1
    print(json.dumps({"old_rows": len(ov["rows"]), "new_rows": len(new), "literals": len(by_lit), "by_class": cls}))


if __name__ == "__main__":
    main(*sys.argv[1:6])
