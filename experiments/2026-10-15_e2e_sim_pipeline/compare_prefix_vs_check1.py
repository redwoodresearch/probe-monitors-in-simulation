"""Release-gate item 1 (09 via 2e 20:0x; laptop, KEYS AND COUNTS ONLY): did the token-prefix check (results/check_token_prefix_2026-10-16_192643.json,
c7b0da2f..., @ 571883e36) load the same staged rows as provenance check 1 (results/check_nm172_provenance_2026-10-16_185935.json, b6132b04...)?
Per run of the 4 natural pairs and per line: the staged file path, the line index and n_tokens in the token-prefix export against check 1's staged
file (path, sha256), its nm172 row (line, ids length) and its meta (n_tokens). Check 1 scanned nm172 only: for the nm305 runs it can confirm the staged
FILE (same path, its sha256), not the row. The token-prefix record holds no staged-file sha256 (it holds path, line index, n_tokens and the ids'
sha256 as int64 bytes; check 1 hashes the ids as canonical JSON), so the file sha256 comes from check 1 and the rows are matched by path + line.
Usage: python compare_prefix_vs_check1.py --prefix <json> --check1 <json> --out <new json>"""
import argparse
import hashlib
import json

PREFIX_SHA = "c7b0da2f42cfece95ce1675983173d2c6383c80c92325d152fa10e0dbe06f7a6"
CHECK1_SHA = "b6132b04ddb3b038261c518166df7ecca45273a0e071c40d70e3c0ad4ba73f87"
MOUNT_LINE = {"/a32": "32k", "/aU": "uncut"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--check1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pb, cb = open(a.prefix, "rb").read(), open(a.check1, "rb").read()
    assert hashlib.sha256(pb).hexdigest() == PREFIX_SHA and hashlib.sha256(cb).hexdigest() == CHECK1_SHA, "STOP: a record differs from its pinned sha256"
    tp, c1 = json.loads(pb), json.loads(cb)
    files1 = {f"{s['mount']}:{f['file']}": f for s in c1["staged"] for f in s["files"]}
    metas1 = {(MOUNT_LINE[r["mount"]], h["fields"].get("key")): h["fields"] for r in c1["extraction_hits"] for h in r["hits"]}
    labs = sorted({x for p in tp["pairs"] if p["verdict"].startswith("(1)") for x in p["pair"]})
    rows = []
    for lab in labs:
        r = tp["runs_export"][lab]
        for line, s in sorted(r["staged"].items()):
            f1 = files1.get(s["file"]) if s else None
            row1 = next((x for x in (f1 or {}).get("nm172_rows", []) if x["fields"].get("key") == r["eval_ref"]), None)
            m1 = metas1.get((line, r["eval_ref"]))
            rows.append({"run": lab, "line": line, "file": s["file"] if s else None, "file_in_check1": f1 is not None,
                         "file_sha256_check1": f1["sha256"] if f1 else None, "row_in_check1": row1 is not None,
                         "line_index_equal": None if row1 is None else row1["line"] == s["line_index"],
                         "n_tokens_prefix": s["n_tokens"] if s else None,
                         "ids_len_check1": None if row1 is None else row1["other_fields"].get("ids", {}).get("len"),
                         "meta_n_tokens_check1": None if m1 is None else m1.get("n_tokens"),
                         "n_tokens_equal": None if row1 is None else (row1["other_fields"].get("ids", {}).get("len") == s["n_tokens"]
                                                                       and (m1 is None or m1.get("n_tokens") == s["n_tokens"]))})
    ok = all(x["file_in_check1"] for x in rows) and all(x["line_index_equal"] is not False and x["n_tokens_equal"] is not False for x in rows)
    res = {"prefix": [a.prefix, PREFIX_SHA], "check1": [a.check1, CHECK1_SHA], "rows": rows, "pass": ok,
           "rows_checked_in_full": sum(1 for x in rows if x["row_in_check1"])}
    with open(a.out, "x") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"pass": ok, "runs": len(labs), "rows": len(rows), "rows_checked_in_full": res["rows_checked_in_full"],
                      "files_only": [f"{x['run']} {x['line']}" for x in rows if x["file_in_check1"] and not x["row_in_check1"]]}))


if __name__ == "__main__":
    main()
