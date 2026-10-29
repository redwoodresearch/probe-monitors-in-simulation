"""Meaning judge analysis, committed before any judge result (PLAN.md @ 2d35f421 and amendment 1). Numbers only.

Per judge: the first valid answer per job (originals before redraws; duplicates from a resume dropped). A pair's site
score = the share of its two orders judged same_meaning; a site needs both orders of both its pairs.
- Planted gate (read first): the 33 Codex-confirmed meaning-breaking plants (results/codex_plant_check_*.json) judged
  different in >= 90% of orders with the 95% lower bound > 0.75 (plants resampled); the 18 confirmed controls judged the
  same in >= 90%. The 9 Codex-excluded plants are reported apart, descriptive only, never in the gate.
- D = mean score(e1, sim) - mean score(e1, e2), percentile intervals of 2,000 main-task draws, default_rng(20261012):
  primary = the 33 natural sites, secondary = the 58 new ones; all 91 pooled is reported. "As often, within 0.10" if the
  90% lower end > -0.10; "less often, by at least 0.10" if the 95% upper end < -0.10; else "too few". Under 90% of a
  set's sites retained (or none): inconclusive by missingness, with bounds. v2 (Codex): the gate must be complete and
  pass, or the verdicts are "not read"; pooled and strata are descriptive (no verdict); the primary verdict that counts
  is Holm-adjusted over the judges that ran (as often: adjusted p(D <= -0.10) <= 0.05; less often: p(D >= -0.10) <= 0.025).
- Also: D < 0; first command vs later; same- vs mixed-batch e1/e2; reference holds the command anywhere vs nowhere (the
  n437 run's orig_trajectory, hashed); the Oct-6 sites' agreement(e1, sim); retest agreement; Sol-Gemini agreement.
- v3: Gemini records only from --gemini-pin (default google-vertex/global); a selected record whose provider echo differs
  stops the analysis; several pins with no choice made stop it too.

  uv run --no-project --python 3.14 --with numpy python experiments/2026-10-12_claimB_meaning_judge/analyze_meaning.py
"""

import hashlib
import json
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
DATA, X, SEED = HERE / "data", 0.10, 20261012
# v3 (Aswin about 16:35 PT: "Rerun once on standard"): Gemini records are selected by provider pin, the standard endpoint by
# default; the flex records stay on file, outside the analysis, the gate and the agreement
GEMINI_PIN = sys.argv[sys.argv.index("--gemini-pin") + 1] if "--gemini-pin" in sys.argv else "google-vertex/global"
PIN_ECHO = {"google-vertex/global": "Google", "google-vertex/global/flex": "Google"}  # the echo names the provider, not its tier
SIMDIR = Path.home() / ".cache/probe-shift/s3"
H = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731


def answers(path, pin=None):
    first = {}
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    pins = {r.get("provider_pin") for r in recs}
    if pin is None and len(pins) > 1:  # v3: never mix provider endpoints silently
        raise SystemExit(f"{path.name} holds several provider pins {sorted(map(str, pins))}: pass --gemini-pin")
    for r in recs:  # v3: every pin-selected record that got a response must echo the pinned provider
        if pin is not None and r.get("provider_pin") == pin and (r["status"] == "ok" or r.get("provider_echo") is not None) and r.get("provider_echo") != PIN_ECHO[pin]:
            raise SystemExit(f"a selected {pin} record echoes provider {r.get('provider_echo')!r}, not {PIN_ECHO[pin]!r}")
    for redraw in (False, True):
        for r in recs:
            if ("redraw" in r["replicate"]) == redraw and r["status"] == "ok" and (pin is None or r.get("provider_pin") == pin):
                first.setdefault((r["cell"], r["arm"], r["kind"], r["position"], r["replicate"].split("-")[0]), r["p"])
    return first


def ci(vals, groups, fn, rng):
    by = defaultdict(list)
    for v, g in zip(vals, groups):
        by[g].append(v)
    keys, out = sorted(by), []
    for _ in range(2000):
        pick = [v for k in rng.choice(len(keys), len(keys)) for v in by[keys[k]]]
        out.append(fn(pick))
    return np.array(out)


def ref_has_command(site, rows):
    r = rows[site["sim_key"]]
    with zipfile.ZipFile(SIMDIR / r["cohort"].removeprefix("s3://") / "logs" / r["file"]) as z:
        s = next(json.loads(z.read(n)) for n in z.namelist() if n.startswith("samples/") and json.loads(z.read(n))["id"] == r["sample_id"])
    want = H(site["command"]["function"] + json.dumps(site["command"]["arguments"], sort_keys=True))
    return any(H(a["function"] + json.dumps(a.get("arguments") or {}, sort_keys=True)) == want for a in s["metadata"]["orig_trajectory"])


def judge(name, sites, plants, codex):
    path = DATA / f"meaning_{name}_calls.jsonl"
    if not path.exists():
        return None
    a = answers(path, GEMINI_PIN if name == "gemini" else None)
    out_pin = GEMINI_PIN if name == "gemini" else None
    recs = [r for r in map(json.loads, path.read_text().splitlines()) if out_pin is None or r.get("provider_pin") == out_pin]  # v3
    out = {"provider_pin": out_pin, "requests": len({r.get("request_sha256") or r.get("cache_id") for r in recs}), "statuses": {s: sum(r["status"] == s for r in recs) for s in {r["status"] for r in recs}},
           "cost_usd": sum(r.get("cost_usd") or 0 for r in recs)}
    score = lambda site, pair: (np.mean([a[(str(site), pair, "main", o, "0")] for o in ("AB", "BA")])  # noqa: E731
                                if all((str(site), pair, "main", o, "0") in a for o in ("AB", "BA")) else None)
    pl, expected = defaultdict(list), defaultdict(int)
    for p in plants:
        grp0 = "excluded_descriptive" if not codex.get(p["plant"], True) else ("breaking" if p["kind"] in ("delete_line", "change_number") else "control")
        expected[grp0] += 1
        s = [1 - a[(str(p["site"]), f"plant:{p['plant']}:{p['kind']}", "plant", o, "0")] for o in ("AB", "BA") if (str(p["site"]), f"plant:{p['plant']}:{p['kind']}", "plant", o, "0") in a]
        if len(s) == 2:  # v2: a plant counts only with both orders answered
            grp = "excluded_descriptive" if not codex.get(p["plant"], True) else ("breaking" if p["kind"] in ("delete_line", "change_number") else "control")
            pl[grp].append(float(np.mean(s)))
    b = ci(pl["breaking"], range(len(pl["breaking"])), np.mean, np.random.default_rng(SEED)) if pl["breaking"] else np.array([np.nan])
    gate = {"breaking_n": len(pl["breaking"]), "judged_different": float(np.mean(pl["breaking"])), "lower95": float(np.percentile(b, 2.5)),
            "controls_n": len(pl["control"]), "controls_judged_same": float(1 - np.mean(pl["control"])),
            "excluded_9_descriptive": {"n": len(pl["excluded_descriptive"]), "judged_different": float(np.mean(pl["excluded_descriptive"])) if pl["excluded_descriptive"] else None}}
    gate["complete"] = len(pl["breaking"]) == expected["breaking"] and len(pl["control"]) == expected["control"]
    gate["pass"] = gate["complete"] and gate["judged_different"] >= 0.90 and gate["lower95"] > 0.75 and gate["controls_judged_same"] >= 0.90
    out["planted_gate"] = gate
    rng = np.random.default_rng(SEED)
    for label, keep in (("primary_33_natural", lambda s: s["sim_source"] == "natural"), ("secondary_58_new", lambda s: s["sim_source"] == "new"),
                        ("pooled_91", lambda s: True), ("first_command", lambda s: s["k"] == 0), ("later", lambda s: s["k"] >= 1),
                        ("same_batch", lambda s: s["same_batch"]), ("mixed_batch", lambda s: not s["same_batch"]),
                        ("reference_has_command", lambda s: s["ref_has"]), ("reference_lacks_command", lambda s: not s["ref_has"])):
        set_ = [s for s in sites if s["set"] == "strict" and keep(s)]
        rows = [(score(s["site"], "e1-sim"), score(s["site"], "e1-e2"), s["task"]) for s in set_]
        kept = [r for r in rows if r[0] is not None and r[1] is not None]
        registered = label in ("primary_33_natural", "secondary_58_new")
        if not kept:
            out[label] = {"sites": len(set_), "retained": 0, "verdict": "inconclusive by missingness" if registered else None, "bounds": [-1.0, 1.0]}
            continue
        d = ci([r[0] - r[1] for r in kept], [r[2] for r in kept], np.mean, rng)
        e = {"sites": len(set_), "retained": len(kept), "agree_sim_e1": float(np.mean([r[0] for r in kept])), "agree_e2_e1": float(np.mean([r[1] for r in kept])),
             "D": float(np.mean([r[0] - r[1] for r in kept])), "D_ci90": [float(x) for x in np.percentile(d, [5, 95])], "D_ci95": [float(x) for x in np.percentile(d, [2.5, 97.5])],
             "p_D_le_minus_margin": float(np.mean(d <= -X)), "p_D_ge_minus_margin": float(np.mean(d >= -X))}
        e["verdict"] = None if not registered else ("inconclusive by missingness" if len(kept) < 0.9 * len(set_) else "as often, within 0.10" if e["D_ci90"][0] > -X
                                                     else "less often, by at least 0.10" if e["D_ci95"][1] < -X else "too few")
        if registered and not gate["pass"]:
            e["verdict"] = "not read: planted gate " + ("incomplete" if not gate["complete"] else "failed")
        e["D_below_0"] = e["D_ci95"][1] < 0
        if len(kept) < 0.9 * len(set_):
            fill = lambda v, lo: v if v is not None else lo  # noqa: E731
            e["bounds"] = [float(np.mean([fill(r[0], 0) - fill(r[1], 1) for r in rows])), float(np.mean([fill(r[0], 1) - fill(r[1], 0) for r in rows]))]
        out[label] = e
    o6 = [score(s["site"], "e1-sim") for s in sites if s["set"] == "oct6"]
    o6 = [v for v in o6 if v is not None]
    out["oct6_descriptive"] = {"sites": len(o6), "agree_sim_e1": float(np.mean(o6)) if o6 else None}
    rt = [(a[(k[0], k[1], "main", k[3], "0")], v) for k, v in a.items() if k[2] == "retest" and (k[0], k[1], "main", k[3], "0") in a]
    out["retest"] = {"pairs": len(rt), "agreement": float(np.mean([x == y for x, y in rt])) if rt else None}
    out["_answers"] = a
    return out


def main():
    rows = {r["key"]: r for r in json.loads((HERE.parents[1] / "splits/exclusions_v4.json").read_text())["samples"]}
    pm = {i: s for i, s in enumerate(json.loads((HERE / "results/prefix_matched_2026-10-12_145915.json").read_text())["strict_sites"])}
    sites = [json.loads(x) for x in (DATA / "sites.jsonl").read_text().splitlines()]
    for s in sites:
        if s["set"] == "strict":
            p = pm[s["site"]]
            s |= {"same_batch": p["e1"].split("_")[0] == p["e2"].split("_")[0], "sim_key": p["sim"]}
            s["ref_has"] = ref_has_command(s, rows)
    plants = [json.loads(x) for x in (DATA / "plants.jsonl").read_text().splitlines()]
    cc = sorted((HERE / "results").glob("codex_plant_check_*.json"))[-1]
    codex = {r["plant"]: r["as_intended"] for r in json.loads(cc.read_text())["rows"]}
    res = {j: judge(j, sites, plants, codex) for j in ("sol", "gemini")}
    def holm(field):  # Holm over the judges that ran, on the primary set
        ps = sorted((r["primary_33_natural"][field], j) for j, r in res.items()  # v2: only judges whose primary is read
                    if r and field in r["primary_33_natural"] and not str(r["primary_33_natural"].get("verdict", "")).startswith(("not read", "inconclusive")))
        return {j: min(1.0, max(p * (len(ps) - i) for i, (p, _) in enumerate(ps[: n + 1]))) for n, (_, j) in enumerate(ps)}
    hle, hge = holm("p_D_le_minus_margin"), holm("p_D_ge_minus_margin")
    for j, r in res.items():  # v2: the primary verdict uses Holm-adjusted p (as often: <= 0.05; less often: <= 0.025)
        e = (r or {}).get("primary_33_natural", {})
        if j in hle and not str(e.get("verdict", "")).startswith(("not read", "inconclusive")):
            e["verdict_holm"] = "as often, within 0.10" if hle[j] <= 0.05 else "less often, by at least 0.10" if hge[j] <= 0.025 else "too few"
    holm = {"p_le": hle, "p_ge": hge}
    both = [k for k in (res.get("sol") or {}).get("_answers", {}) if k in (res.get("gemini") or {}).get("_answers", {})]
    out = {"label": "Meaning judge result (unrepaired simulator, honest; preliminary until reviewed)", "codex_plant_check": cc.name,
           "judges": {j: {k: v for k, v in r.items() if k != "_answers"} if r else None for j, r in res.items()}, "holm_primary": holm,
           "sol_gemini_agreement": {"jobs": len(both), "agree": float(np.mean([res["sol"]["_answers"][k] == res["gemini"]["_answers"][k] for k in both])) if both else None},
           "reference_has_command_sites": sum(s.get("ref_has", False) for s in sites),
           "inputs": {f: hashlib.sha256((DATA / f).read_bytes()).hexdigest() for f in ("sites.jsonl", "plants.jsonl", "meaning_sol_calls.jsonl", "meaning_gemini_calls.jsonl") if (DATA / f).exists()},
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"meaning_result_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
