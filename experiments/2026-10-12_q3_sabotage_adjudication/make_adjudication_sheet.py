"""Draw the blinded 100-call adjudication sample for the stage-1 sabotage labels, as fixed in RULE.md. Local, $0.

Parses the inputs whole but keeps only IDs, labels and vote counts; never prints or writes a note, reason, call or any
other field:
- labels_stage1_filled.csv (opaque_id, label), vote_audit.json (opaque_id, primary_label, sure_votes, valid_votes) and
  comparisons.json (identifiers.opaque_id, kind, label, panel_label, agree), from aswin/sabotage-stage1-judge-labels
  b90cd42a;
- stage1_rows.csv (opaque_id, world, side): key_stage1.csv from S3 (URI in RULE.md), renamed so its hash entry doesn't
  trip the gitleaks key rule.
Writes:
- sheet (--sheet-dir): audit_id, label, confidence, note; the last three empty; sorted by audit_id. audit_id is the
  S-number the existing label-audit viewer gives the row under `prepare --seed 20261012 --size 1457`.
- key (--key-dir, a private folder outside the repository; goes to S3 only): audit_id -> opaque_id, stratum, side, world,
  panel label, vote counts, cell size, draws, inclusion probability.
- a summary JSON beside each; counts per stratum go on the key side only.
Then publish_key.py uploads the key to S3 and writes the committed manifest.

  uv run --no-project python make_adjudication_sheet.py --inputs <dir> --key-dir <private dir>
"""

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
SEED = 20261012
INPUT_SHA = {
    "labels_stage1_filled.csv": "6aca0a5a8f7954e2de4a84c72c17d45e65a178abad276b35aecf309d1dadc767",
    "vote_audit.json": "93c65dfd138c6c52c94aee764f4631ae6a7c74b2559fa4c53308d3cac5333dfd",
    "comparisons.json": "b59c92748cea9f1678538fb61cea7e8124fc25e86f1e75f11ac8132905ada596",
    "stage1_rows.csv": "626b1a4f4b8b8e46cb5e63dc6da2e48a9f2e3f57506346c123068f837d9cb544",
}
# (stratum, side) -> (rows in the population, draws). Row counts were taken from labels, votes and sides before
# this script was written and are asserted here; draws are RULE.md's allocation (sum 100).
CELLS = {
    ("selfaudit_disagree", "held_out"): (5, 5),
    ("selfaudit_disagree", "train"): (6, 6),
    ("sabotage_unanimous", "held_out"): (22, 12),
    ("sabotage_unanimous", "train"): (38, 4),
    ("sabotage_split", "held_out"): (14, 12),
    ("sabotage_split", "train"): (12, 3),
    ("unclear_split", "held_out"): (41, 15),
    ("unclear_split", "train"): (38, 4),
    ("not_sabotage_split", "held_out"): (34, 10),
    ("not_sabotage_split", "train"): (42, 3),
    ("not_sabotage_unanimous", "held_out"): (442, 24),
    ("not_sabotage_unanimous", "train"): (763, 2),
}


def h(salt, opaque_id):
    return hashlib.sha256(f"{SEED}|{salt}|{opaque_id}".encode()).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()


def main(inputs, sheet_dir, key_dir):
    for name, want in INPUT_SHA.items():
        if hashlib.sha256((inputs / name).read_bytes()).hexdigest() != want:
            raise SystemExit(f"input sha256 mismatch: {name}")
    for path in (HERE / "RULE.md", Path(__file__)):
        committed = subprocess.run(["git", "show", f"HEAD:./{path.name}"], check=False, cwd=HERE, capture_output=True)
        if committed.returncode or committed.stdout != path.read_bytes():
            raise SystemExit(f"{path.name} differs from HEAD; commit the rule and script before drawing")
    head, top = git("rev-parse", "HEAD"), Path(git("rev-parse", "--show-toplevel")).resolve()
    if key_dir.resolve().is_relative_to(top) or key_dir.resolve() == sheet_dir.resolve():
        raise SystemExit("--key-dir must be a private folder outside the repository and the sheet folder")

    with (inputs / "labels_stage1_filled.csv").open() as f:
        label = {r["opaque_id"]: r["label"] for r in csv.DictReader(f)}
    votes = {
        v["opaque_id"]: {k: v[k] for k in ("primary_label", "sure_votes", "valid_votes")}
        for v in json.loads((inputs / "vote_audit.json").read_text())
    }
    with (inputs / "stage1_rows.csv").open() as f:
        where = {r["opaque_id"]: (r["side"], r["world"]) for r in csv.DictReader(f)}
    audit = {
        c["identifiers"]["opaque_id"]: {k: c[k] for k in ("kind", "label", "panel_label", "agree")}
        for c in json.loads((inputs / "comparisons.json").read_text())
    }
    assert len(label) == 1457 and set(label) == set(votes) == set(where)
    assert len(audit) == 50 and set(audit) <= set(label) and all(c["kind"] == "sabotage" for c in audit.values())
    assert all(audit[o]["panel_label"] == label[o] for o in audit)

    rows = {}
    for o, lab in label.items():
        v = votes[o]
        assert v["primary_label"] == lab and v["valid_votes"] in (4, 5)
        assert set(v["sure_votes"]) <= {"sabotage", "not_sabotage"}
        unanimous = any(n == v["valid_votes"] for n in v["sure_votes"].values())
        forced = o in audit and not audit[o]["agree"]
        stratum = "selfaudit_disagree" if forced else f"{lab}_{'unanimous' if unanimous else 'split'}"
        rows[o] = {"opaque_id": o, "stratum": stratum, "side": where[o][0], "world": where[o][1], "panel_label": lab,
                   "valid_votes": v["valid_votes"], "sure_sabotage": v["sure_votes"].get("sabotage", 0),
                   "sure_not_sabotage": v["sure_votes"].get("not_sabotage", 0),
                   "selfaudit_label": audit[o]["label"] if forced else ""}  # fmt: skip
    cell_rows = Counter((r["stratum"], r["side"]) for r in rows.values())
    assert dict(cell_rows) == {c: n for c, (n, _) in CELLS.items()}, dict(cell_rows)
    assert sum(d for _, d in CELLS.values()) == 100

    viewer_rank = {o: i + 1 for i, o in enumerate(sorted(label, key=lambda o: h("sabotage", o)))}  # prepare's order
    drawn = []
    for (stratum, side), (n_rows, n_draw) in CELLS.items():
        cell = sorted((o for o, r in rows.items() if (r["stratum"], r["side"]) == (stratum, side)),
                      key=lambda o: h("adjudicate", o))  # fmt: skip
        for o in cell[:n_draw]:
            drawn.append({"audit_id": f"S{viewer_rank[o]:04d}", **rows[o], "cell_rows": n_rows, "cell_draws": n_draw,
                          "p_incl": round(n_draw / n_rows, 6)})  # fmt: skip
    drawn.sort(key=lambda r: r["audit_id"])
    assert len(drawn) == len({r["audit_id"] for r in drawn}) == 100

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    sheet_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    s_path, k_path = sheet_dir / f"adjudication_sheet_{stamp}.csv", key_dir / f"adjudication_key_{stamp}.csv"
    with s_path.open("x", newline="") as fs, k_path.open("x", newline="") as fk:
        ws = csv.DictWriter(fs, fieldnames=["audit_id", "label", "confidence", "note"])
        wk = csv.DictWriter(fk, fieldnames=list(drawn[0]))
        ws.writeheader()
        wk.writeheader()
        for r in drawn:
            ws.writerow({"audit_id": r["audit_id"], "label": "", "confidence": "", "note": ""})
            wk.writerow(r)
    counts = {f"{s}|{side}": sum((r["stratum"], r["side"]) == (s, side) for r in drawn) for s, side in CELLS}
    common = {"git": head, "seed": SEED, "input_sha256": INPUT_SHA, "rows": len(drawn)}
    s_sum = {**common, "sheet": s_path.name, "sheet_sha256": hashlib.sha256(s_path.read_bytes()).hexdigest()}
    k_sum = {**common, "key": k_path.name, "key_sha256": hashlib.sha256(k_path.read_bytes()).hexdigest(),
             "sheet": s_path.name, "sheet_sha256": s_sum["sheet_sha256"], "draws_per_cell": counts}  # fmt: skip
    (sheet_dir / f"{s_path.stem}.summary.json").write_text(json.dumps(s_sum, indent=1) + "\n")
    (key_dir / f"{k_path.stem}.summary.json").write_text(json.dumps(k_sum, indent=1) + "\n")
    print(json.dumps({"sheet": str(s_path), "key": str(k_path), **{k: k_sum[k] for k in ("sheet_sha256", "key_sha256")},
                      "draws_per_cell": counts}, indent=1))  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", type=Path, required=True, help="folder holding the four pinned inputs")
    ap.add_argument("--sheet-dir", type=Path, default=HERE / "results")
    ap.add_argument("--key-dir", type=Path, required=True, help="private folder outside the repository")
    a = ap.parse_args()
    main(a.inputs, a.sheet_dir, a.key_dir)
