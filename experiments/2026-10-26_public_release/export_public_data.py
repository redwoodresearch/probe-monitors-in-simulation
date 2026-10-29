"""Internal: write the public release's plain data table for Figures 1 and 5 (release/data/catch_rates.csv) from the post's result files.

The public release has no internal labels or internal field names, so its data is exported once from the result files the post was
written from (post/records, the reproducibility folder), with every value checked:
  E1  each input's sha256 (and, for .gz, its decompressed sha256) equals post/MANIFEST.json;
  E2  the six probe names equal the post's display names in the Figure 1 run's name map (figs156 render log);
  E3  the group records' members are exactly the six (and the four) probes;
  E4  every exported number equals the stored value it came from, x100, as Python floats (no rounding);
  E5  every exported number that Figure 1 (commands) or Figure 5 (tool outputs) plotted equals the plotted value in that figure's
      source file (post/expected/figures/*_source.json), x100; every plotted per-probe or average value is covered.
  E6  reported, not a stop: how closely the README's definitions hold (difference = simulated minus real; real = the mean of the two
      runs; rerun difference = run 2 minus run 1; an average row's difference = the mean of its probes' differences).
  E7  release/numbers.csv lists every rebuilt number on the slice's post lines (Figure 1's caption, the paragraphs after it, the footnote
      on the four probes), except those named in DEFERRED, and each one's printed text equals the post's.
Any failure stops before anything is written. Units: catch rates and honest-run alarm shares in %, differences in percentage points.

  cd <worktree> && M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
  "$M" run experiments/2026-10-26_public_release/export_public_data.py
SPEND, worst-case bound: 1 call x (startup_timeout 300 + timeout 600 + scaledown_window 2) s at 1 core and 4 GiB =
902 x ($0.0000131 + 4 x $0.00000222) = $0.01982596. Typical: under $0.01.
"""

import csv
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve()
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE.parent  # /root in the container: no parents[2] there
POST = REPO / "post"
APP = "probe-shift_aswin-surya_sub-public-release_export_kill-2026-11-25"

INPUTS = {"A": "record", "group_catch": "group", "pw_shares": "pw"}  # MANIFEST name -> the figure source files' input name
FIG_SOURCES = {"C": "expected/figures/fig1_main_figs156_routecA_source.json",
               "R": "expected/figures/fig1_main_R_figs156_routecA_source.json"}
NAME_LOG = "figures/figs156/render_log_figs156_routecA_012618.json"
PROBES = [("own_all_rs_uncut_L48", "On-policy probe"), ("sol_a_uncut", "Off-policy probe"), ("FIBS|uncut", "Deception probe"),
          ("harmfulness|uncut", "Harmfulness direction"), ("refusal|uncut", "Refusal direction"),
          ("high-stakes|uncut", "High-stakes probe")]
FOUR = ["own_all_rs_uncut_L48", "sol_a_uncut", "FIBS|uncut", "harmfulness|uncut"]
GROUPS = [("six", "headline_six_full", "Six-probe average", [p for p, _ in PROBES]),
          ("auroc_subset", "auroc06_subset", "Four-probe average", FOUR)]
SCORES = {"C": "commands", "R": "tool outputs"}
LEAD, MODE, LEVEL = "pooled_all_origins", "joint", "0.01"
SETS = ["S_old", "S_new", "S_newmains", "S_paper493"]  # the four task sets; tasks = the sum of their cohort cells (rule L46.8)
NUMBER_IDS = {"1": "L46.8", "2": "L48.1", "3": "L48.3", "4": "L48.4", "5": "L48.5", "6": "L48.9", "7": "L48.11", "8": "L50.1",
              "9": "L100.2", "10": "L100.3", "11": "L100.4", "12": "L100.5"}  # release/numbers.csv id -> the post token id
SLICE_LINES = {46, 48, 50, 100}  # Figure 1's caption, the paragraphs after it, and the footnote on the four probes
DEFERRED = {"L50.2": "planted-error size; comes with the planted-error table"}
COLUMNS = ["scores", "alarm_level", "row", "real_caught", "real_caught_low", "real_caught_high", "simulated_caught", "simulated_caught_low",
           "simulated_caught_high", "simulated_minus_real", "simulated_minus_real_low", "simulated_minus_real_high", "run2_minus_run1",
           "run2_minus_run1_low", "run2_minus_run1_high", "honest_runs_alarmed", "tasks"]

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    man = json.loads((POST / "MANIFEST.json").read_text())
    recs = {r["name"]: r for r in man["records"]}
    for name in INPUTS:
        image = image.add_local_file(str(POST / recs[name]["file"]), f"/in/{name}")
    for f in ["MANIFEST.json", NAME_LOG, *FIG_SOURCES.values(), "expected/expected_post_numbers.json"]:
        image = image.add_local_file(str(POST / f), f"/in/post/{f}")
    image = image.add_local_file(str(REPO / "release" / "numbers.csv"), "/in/numbers.csv")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@app.function(image=image, cpu=(1.0, 1.0), memory=(4096, 4096), timeout=600, startup_timeout=300, scaledown_window=2, retries=0)
def export() -> dict:
    import gzip

    man = json.loads(Path("/in/post/MANIFEST.json").read_bytes())
    recs = {r["name"]: r for r in man["records"]}
    data = {}
    for name in INPUTS:  # E1
        b, r = Path(f"/in/{name}").read_bytes(), recs[name]
        assert sha(b) == r["sha256"], f"STOP E1: {name} sha256"
        if r["file"].endswith(".gz"):
            b = gzip.decompress(b)
            assert sha(b) == r["decompressed_sha256"], f"STOP E1: {name} decompressed sha256"
        data[name] = json.loads(b)
    copied = {c["file"]: c["sha256"] for c in man["copied_files"]}
    for f in [NAME_LOG, *FIG_SOURCES.values()]:  # copied byte for byte; pinned in MANIFEST copied_files
        assert f in copied and sha(Path(f"/in/post/{f}").read_bytes()) == copied[f], f"STOP E1: {f} sha256"
    A, G, PW = data["A"], data["group_catch"], data["pw_shares"]

    log = Path(f"/in/post/{NAME_LOG}").read_text()  # E2: the name map is a JSON string inside the run log's plan arguments
    maps = [json.loads(s) for s in _strings(json.loads(log)) if s.startswith("{") and '"sol_a_uncut":"Off-policy probe"' in s]
    assert maps and all(m == maps[0] for m in maps), "STOP E2: no single display-name map in the figure run log"
    for lab, shown in PROBES:
        assert maps[0].get(lab) == shown, f"STOP E2: {lab} is not shown as {shown!r}"

    rows, checks = [], []

    def pct(x, src):  # E4: the exported number is the stored value x100, as computed here
        assert isinstance(x, float | int) and not isinstance(x, bool), f"STOP E4: {src} is not a number"
        checks.append(src)
        return 100 * x

    def summ(node, src):
        assert node.get("ci95") is not None and len(node["ci95"]) == 2, f"STOP E4: {src} has no 95% interval"
        return [pct(node["point"], src), pct(node["ci95"][0], src + ".lo"), pct(node["ci95"][1], src + ".hi")]

    for ro, scores in SCORES.items():
        coh = A["coheadline_signed"][LEAD][MODE][LEVEL]
        for g, h, shown, members in GROUPS:  # E3
            gr = G["groups"][f"{g}|{ro}"]
            assert sorted(gr["members"]) == sorted(f"{m} {ro}" for m in members), f"STOP E3: {g}|{ro} members"
            assert float(gr["level"]) == 0.01, f"STOP E3: {g}|{ro} level"
            c = coh[f"{h}|{ro}"]
            assert sorted(c["rows"]) == sorted(f"{m} {ro}" for m in members), f"STOP E3: {h}|{ro} rows"
            rows.append([scores, "1%", shown, *summ(gr["catch"]["real_mean"], f"group:{g}|{ro}.real_mean"),
                         *summ(gr["catch"]["sim"], f"group:{g}|{ro}.sim"), *summ(c["S"], f"A:coheadline.{h}|{ro}.S"),
                         *summ(c["Y"], f"A:coheadline.{h}|{ro}.Y"), "", ""])
        for lab, shown in PROBES:
            n = A["pooled"][f"{lab} {ro}"][LEAD][MODE][LEVEL]
            fl = PW["shares"][f"{lab} {ro}"][LEVEL]["B"]["registered_with_no_unit_runs"]["run_weighted"]
            cov = A["coverage"][f"{lab} {ro}"]
            ks = [k for k in cov if k.split(" ")[0] in SETS and f"[pooled: {LEAD}]" in k]
            assert sorted(k.split(" ")[0] for k in ks) == sorted(SETS), f"STOP E4: {lab} {ro} coverage sets"
            tasks = sum(cov[k]["coverage"]["cohort_cells"] for k in ks)
            assert isinstance(tasks, int), f"STOP E4: {lab} {ro} task count"
            rows.append([scores, "1%", shown, *summ(n["catch"]["real_mean"], f"A:{lab} {ro}.real_mean"),
                         *summ(n["catch"]["sim"], f"A:{lab} {ro}.sim"), *summ(n["gap"]["delta_full"], f"A:{lab} {ro}.delta_full"),
                         *summ(n["gap"]["run2_minus_run1_at_tauB"], f"A:{lab} {ro}.run2_minus_run1"), pct(fl, f"pw:{lab} {ro}"), tasks])

    # E5: every per-probe and average value the two figures plotted equals the exported number
    exported = {}
    for r in rows:
        exported[(r[0], r[2])] = dict(zip(COLUMNS, r, strict=True))
    col = {("catch", "real_mean"): "real_caught", ("catch", "sim"): "simulated_caught", ("gap", "delta_full"): "simulated_minus_real",
           ("gap", "run2_minus_run1_at_tauB"): "run2_minus_run1", "S": "simulated_minus_real", "Y": "run2_minus_run1"}
    by_label = {f"{lab} {ro}": (SCORES[ro], shown) for lab, shown in PROBES for ro in SCORES}
    by_group = {f"{g}|{ro}": (SCORES[ro], shown) for g, _, shown, _ in GROUPS for ro in SCORES}
    by_coh = {f"{h}|{ro}": (SCORES[ro], shown) for _, h, shown, _ in GROUPS for ro in SCORES}
    n_cmp, n_plotted = 0, 0
    for ro, f in FIG_SOURCES.items():
        covered = set()
        for v in json.loads(Path(f"/in/post/{f}").read_bytes())["values"]:
            k, src = v["key"], v["input"]
            if src == "record" and k[0] == "pooled" and k[5] in ("catch", "gap"):
                row, c = by_label[k[1]], col[(k[5], k[6])]
            elif src == "record" and k[0] == "coheadline_signed":
                row, c = by_coh[k[4]], col[k[5]]
            elif src == "group" and k[0] == "groups":
                row, c = by_group[k[1]], col[("catch", k[3])]
            elif src == "pw" and k[0] == "shares":
                row, c = by_label[k[1]], "honest_runs_alarmed"
            elif src == "record" and k[0] == "coverage":
                row, c = by_label[k[1]], None  # the task count is checked as a sum below
                covered.add((row, "tasks"))
            else:
                continue
            n_plotted += 1
            if c is None:
                continue
            covered.add((row, c))
            e = exported[row]
            if "point" in v:
                assert e[c] == 100 * v["point"], f"STOP E5: {f} {k} point"
                assert [e[c + "_low"], e[c + "_high"]] == [100 * x for x in v["ci95"]], f"STOP E5: {f} {k} interval"
            else:
                assert e[c] == 100 * v["value"], f"STOP E5: {f} {k} value"
            n_cmp += 1
        other = [s for r_, s in SCORES.items() if r_ != ro][0]  # the right panel's shared scale also reads the other readout
        want = {((SCORES[ro], shown), c) for _, shown in PROBES
                for c in ("real_caught", "simulated_caught", "simulated_minus_real", "run2_minus_run1", "honest_runs_alarmed", "tasks")}
        want |= {((SCORES[ro], shown), c) for _, _, shown, _ in GROUPS
                 for c in ("real_caught", "simulated_caught", "simulated_minus_real", "run2_minus_run1")}
        want |= {((other, shown), c) for shown in [s for _, s in PROBES] + [s for _, _, s, _ in GROUPS]
                 for c in ("simulated_minus_real", "run2_minus_run1")}
        assert want <= covered, f"STOP E5: {f} plots no value for {sorted(want - covered)[:3]}"
        nb = json.loads(Path(f"/in/post/{f}").read_bytes())["n_by_probe"]
        for lab, shown in PROBES:
            assert exported[(SCORES[ro], shown)]["tasks"] == nb[f"{lab} {ro}"], f"STOP E5: {f} task count for {lab}"

    # E6 (reported, not a stop): do the README's definitions hold for the stored points? Largest absolute gap, in percentage points.
    e6 = {"simulated_minus_real = sim - real_mean": 0.0, "real_mean = (run1 + run2) / 2": 0.0, "run2_minus_run1 = run2 - run1": 0.0,
          "average rows: simulated_minus_real = mean of the members' values": 0.0}
    for ro in SCORES:
        for lab, _ in PROBES:
            q = A["pooled"][f"{lab} {ro}"][LEAD][MODE][LEVEL]
            c, g = {k: q["catch"][k]["point"] for k in ("real_mean", "sim", "run1", "run2")}, q["gap"]
            for k, gap in (("simulated_minus_real = sim - real_mean", g["delta_full"]["point"] - (c["sim"] - c["real_mean"])),
                           ("real_mean = (run1 + run2) / 2", c["real_mean"] - (c["run1"] + c["run2"]) / 2),
                           ("run2_minus_run1 = run2 - run1", g["run2_minus_run1_at_tauB"]["point"] - (c["run2"] - c["run1"]))):
                e6[k] = max(e6[k], 100 * abs(gap))
        for _, h, _, members in GROUPS:
            s = A["coheadline_signed"][LEAD][MODE][LEVEL][f"{h}|{ro}"]["S"]["point"]
            m = sum(A["pooled"][f"{x} {ro}"][LEAD][MODE][LEVEL]["gap"]["delta_full"]["point"] for x in members) / len(members)
            e6["average rows: simulated_minus_real = mean of the members' values"] = max(
                e6["average rows: simulated_minus_real = mean of the members' values"], 100 * abs(s - m))

    # E7: numbers.csv covers every rebuilt (class R) number on this slice's lines, each with the post's printed text
    pinned = {p["file"]: p["sha256"] for p in man["pinned_files"]}
    ef = "expected/expected_post_numbers.json"
    assert sha(Path(f"/in/post/{ef}").read_bytes()) == pinned[ef], f"STOP E1: {ef} sha256"
    exp = json.loads(Path("/in/post/expected/expected_post_numbers.json").read_bytes())
    toks = {t["id"]: t for t in exp["tokens"]}
    pub = list(csv.DictReader(io.StringIO(Path("/in/numbers.csv").read_text(encoding="utf-8"))))
    assert [n["id"] for n in pub] == list(NUMBER_IDS), "STOP E7: numbers.csv ids differ from the slice's map"
    for n in pub:
        assert toks[NUMBER_IDS[n["id"]]]["printed"] == n["printed"], f"STOP E7: number {n['id']} printed text differs from the post"
    in_slice = {t["id"] for t in exp["tokens"] if t["class"] == "R" and t["line"] in SLICE_LINES}
    listed = set(NUMBER_IDS.values()) | set(DEFERRED)
    assert in_slice == listed, f"STOP E7: slice tokens differ: {sorted(in_slice ^ listed)}"

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    w.writerows(rows)
    return {"csv": buf.getvalue(), "n_rows": len(rows), "n_values_checked_against_records": len(checks),
            "n_plotted_values_compared": n_cmp, "n_plotted_values_seen": n_plotted, "e6_largest_gap_points": e6,
            "e7_numbers_checked": len(pub), "e7_deferred": DEFERRED,
            "inputs": {name: recs[name]["sha256"] for name in INPUTS}}


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)


@app.local_entrypoint()
def main():
    res = export.remote()
    out = REPO / "release" / "data" / "catch_rates.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as fh:
        fh.write(res.pop("csv"))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = HERE.parent / "results" / f"export_check_{stamp}.json"
    rec.parent.mkdir(exist_ok=True)
    with rec.open("x") as fh:
        json.dump(res | {"output": str(out.relative_to(REPO)), "output_sha256": sha(out.read_bytes()), "stamp_pt": stamp}, fh, indent=1)
    print("EXPORT_OK", out, sha(out.read_bytes()), json.dumps({k: v for k, v in res.items() if k != "inputs"}))
