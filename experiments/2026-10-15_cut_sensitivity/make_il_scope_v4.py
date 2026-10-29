"""IL admission scope v4 = v3 with nm145 EXCLUDED by key for every family (09's ruling via 2e, 10-16 ~04:5x): "nm145: 32k pack accepted only
by a narrow launcher copy with 5 known verify FAILs; excluded by key". Every other row unchanged; admitted keys 1,587 -> 1,586. Keys only.
r2 (Codex freeze round scope_v3_v4): every summary (counts, families) is recomputed from the rows as v2 builds them; the first v4 record
(il_scope_v4_2026-10-16_044741.json, 79b59a45...) kept counts.excluded and the s4rgap family count from v3. Admitted keys asserted equal to it."""
import hashlib, json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
V3 = (HERE / "results" / "il_scope_v3_2026-10-16_040038.json", "c1006e344a9341081476ecf65bf11e2a6f749595afabbb2a5e1c726fc996be97")
CAVEAT = "nm145: 32k pack accepted only by a narrow launcher copy with 5 known verify FAILs; excluded by key"
b = V3[0].read_bytes()
if hashlib.sha256(b).hexdigest() != V3[1]:
    raise SystemExit("scope v3 sha256")
d = json.loads(b)
hit = [x for x in d["rows"] if x["cell"] == "nm145_model_registry" and x["world"] == "sim2" and x["admitted"]]
if len(hit) != 1:
    raise SystemExit(f"{len(hit)} admitted nm145 sim2 rows")
before = set(d["admitted_keys"])
hit[0] |= {"admitted": False, "class": "excluded by key (caveat)", "caveat": CAVEAT}
d["admitted_keys"] = sorted(x["key"] for x in d["rows"] if x["admitted"])
if before - set(d["admitted_keys"]) != {hit[0]["key"]} or set(d["admitted_keys"]) - before:
    raise SystemExit("more than the one nm145 key changed")
adm = [x for x in d["rows"] if x["admitted"]]
d["counts"]["admitted"] = len(adm)
d["counts"]["admitted_by_set_arm"] = {f"{s}|{w}": n for (s, w), n in sorted(Counter((x["set"], x["world"]) for x in adm).items())}
d["counts"]["admitted_by_replacement_kind"] = dict(Counter(x["replacement_kind"] for x in adm))
d["counts"]["excluded_by_class"] = dict(Counter(x["class"] for x in d["rows"] if not x["admitted"]))
d["counts"]["excluded"] = len(d["rows"]) - len(adm)
d["counts"]["admitted_zero_tool_structural"] = sum(1 for x in adm if x["zero_tool"])
fam = Counter((x["family"], x["world"], x["set"], x["in_append32_regex"]) for x in adm)
d["families"] = [{"family": f, "arm": w, "set": s, "admitted": n, "in_append32_regex": rx, "flag_for_09": not rx} for (f, w, s, rx), n in sorted(fam.items())]
V4A = (HERE / "results" / "superseded" / "il_scope_v4_2026-10-16_044741.json", "79b59a455b13046124c4d8a5277f2985f4e865b3850fdf8a442da08a05cc45b7")
old = json.loads(V4A[0].read_bytes())
if hashlib.sha256(V4A[0].read_bytes()).hexdigest() != V4A[1] or old["admitted_keys"] != d["admitted_keys"]:
    raise SystemExit("admitted keys differ from the first v4 record")
d |= {"label": d["label"].replace("v3", "v4", 1), "supersedes": {"path": V3[0].name, "sha256": V3[1]}, "v4_fix": CAVEAT,
      "v4_r2": {"replaces": V4A[0].name, "fix": "summaries recomputed from the rows (Codex freeze round scope_v3_v4); admitted keys equal"}}
blob = json.dumps(d, indent=1).encode()
dest = HERE / "results" / f"il_scope_v4_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
dest.open("xb").write(blob)
print(json.dumps({"record": dest.name, "sha256": hashlib.sha256(blob).hexdigest(), "admitted": len(adm), "S_newmains|sim2": d["counts"]["admitted_by_set_arm"].get("S_newmains|sim2")}))
