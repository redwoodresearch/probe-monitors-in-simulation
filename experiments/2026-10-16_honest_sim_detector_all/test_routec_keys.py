"""Lean synthetic test of routec_keys (route C rerun's S_old sim key check). No I/O, no git, no Modal.
Run: uv run --no-project python test_routec_keys.py   (prints PASS n/n, rc 0; any failure raises)."""

import copy
import hashlib

import routec_keys as rk

N = rk.EXTRA_N
TASKS = [f"t{i:02d}_task" for i in range(N)]


def manifest():
    return {"n": N, "keys": [{"world": rk.EXTRA_WORLD, "stratum": "S_old", "task": t, "eval_sha256": hashlib.sha256(t.encode()).hexdigest(),
                              "run_key": f"sim_honest@x/{t}"} for t in TASKS]}


def inv_pair():
    base = [{"key": f"old/{i}", "task_set": "T196", "world": "honest_b", "task": f"c{i}"} for i in range(5)]
    new = [{"key": f"pilot/{t}#e0", "task_set": "T196", "world": "sim_honest", "task": t, "eval_sha256": hashlib.sha256(t.encode()).hexdigest()}
           for t in TASKS]
    v1 = {"status": "COMPLETE", "s4_orphans_out": [{"target": "x", "arm": "a"}], "runs": base}
    v2 = {"status": "COMPLETE", "s4_orphans_out": [{"target": "x", "arm": "a"}], "runs": base + new}
    return v1, v2


def kept_for(new_keys, n_reg=rk.SOLD_SIM_REGISTERED, extra=()):
    k = [{"key": f"reg/{i}", "batch": "sim", "set": "S_old"} for i in range(n_reg)]
    k += [{"key": x, "batch": "sim", "set": "S_old"} for x in sorted(new_keys)]
    k += [{"key": "b/1", "batch": "B", "set": "S_old"}, {"key": "s/1", "batch": "sim", "set": "S_new"}]
    return k + list(extra)


def raises(f, *a):
    try:
        f(*a)
    except ValueError:
        return True
    return False


# the pin hash is the real census pin, so the synthetic keys cannot match it: test pilot_keys' hash rule with a patched pin
_REAL_PIN = rk.PILOT_LIST_DIGEST
rk.PILOT_LIST_DIGEST = hashlib.sha256("\n".join(sorted(f"{rk.EXTRA_WORLD}|{t}" for t in TASKS)).encode()).hexdigest()

checks = []
# 1. the good path
keys, pairs = rk.pilot_keys(manifest())
v1, v2 = inv_pair()
new_keys = rk.new_rows_are_pilot(rk.inventory_delta(v1, v2), pairs)
got = rk.sold_sim(kept_for(new_keys), new_keys)
checks.append(("good path counts", got == {"sold_sim": 191, "registered": 171, "pilot": 20}))
checks.append(("real census pin unchanged in module", _REAL_PIN == "e7f62cca627f1dfea225ee921fc532c7ed5234a140a7a8bd07e112229eabd25a"))
# 2. pilot_keys: a missing key, an extra key, a wrong key, a wrong world, a wrong hash
m = manifest(); m["keys"].pop(); checks.append(("pilot: missing key", raises(rk.pilot_keys, m)))
m = manifest(); m["keys"].append(dict(m["keys"][0], task="t99_extra")); m["n"] = N + 1; checks.append(("pilot: extra key", raises(rk.pilot_keys, m)))
m = manifest(); m["keys"][3]["task"] = "t03_wrong"; checks.append(("pilot: wrong key (hash)", raises(rk.pilot_keys, m)))
m = manifest(); m["keys"][0]["world"] = "sim_honest"; checks.append(("pilot: wrong world", raises(rk.pilot_keys, m)))
# 3. pilot_pin: exact INPUTS_FULL pin
spec = {"commit": "344829af6dc4cd0473757398d542bb8e1f72f512", "path": "p.json", "sha256": "ab" * 32}
full = {"s_old_simh_extra_roots": {rk.EXTRA_ROOT: {"commit": "344829af6", "path": "p.json", "sha256": "ab" * 32}}}
rk.pilot_pin(full, spec)
checks.append(("pin: good", True))
f2 = copy.deepcopy(full); f2["s_old_simh_extra_roots"][rk.EXTRA_ROOT]["sha256"] = "cd" * 32; checks.append(("pin: wrong sha", raises(rk.pilot_pin, f2, spec)))
f2 = copy.deepcopy(full); f2["s_old_simh_extra_roots"]["other"] = {}; checks.append(("pin: extra root", raises(rk.pilot_pin, f2, spec)))
# 4. inventory_delta: a changed old row, a dropped old row, a changed orphan list
a, b = inv_pair(); b["runs"][0] = dict(b["runs"][0], task="c_changed"); checks.append(("delta: changed old row", raises(rk.inventory_delta, a, b)))
a, b = inv_pair(); b["runs"].pop(0); checks.append(("delta: dropped old row", raises(rk.inventory_delta, a, b)))
a, b = inv_pair(); b["s4_orphans_out"] = []; checks.append(("delta: orphans changed", raises(rk.inventory_delta, a, b)))
# 5. new_rows_are_pilot: a missing pilot row, an extra row, a wrong row (eval sha), a non-sim row
a, b = inv_pair(); b["runs"].pop(); checks.append(("rows: missing pilot row", raises(lambda: rk.new_rows_are_pilot(rk.inventory_delta(a, b), pairs))))
a, b = inv_pair(); b["runs"].append({"key": "x#e0", "task_set": "T196", "world": "sim_honest", "task": "t99", "eval_sha256": "0" * 64})
checks.append(("rows: extra row", raises(lambda: rk.new_rows_are_pilot(rk.inventory_delta(a, b), pairs))))
a, b = inv_pair(); b["runs"][-1] = dict(b["runs"][-1], eval_sha256="1" * 64); checks.append(("rows: wrong eval sha", raises(lambda: rk.new_rows_are_pilot(rk.inventory_delta(a, b), pairs))))
a, b = inv_pair(); b["runs"][-1] = dict(b["runs"][-1], world="honest_a"); checks.append(("rows: non-sim row", raises(lambda: rk.new_rows_are_pilot(rk.inventory_delta(a, b), pairs))))
# 6. sold_sim: a pilot key missing from the class, an extra registered key, a missing registered key, a pilot key in the wrong set
nk = sorted(new_keys)
checks.append(("sold: pilot key not kept", raises(rk.sold_sim, [e for e in kept_for(new_keys) if e["key"] != nk[0]], new_keys)))
checks.append(("sold: extra registered key", raises(rk.sold_sim, kept_for(new_keys, n_reg=172), new_keys)))
checks.append(("sold: missing registered key", raises(rk.sold_sim, kept_for(new_keys, n_reg=170), new_keys)))
k = kept_for(new_keys); [e.update(set="S_new") for e in k if e["key"] == nk[1]]; checks.append(("sold: pilot key in another set", raises(rk.sold_sim, k, new_keys)))
# 09's control (d): a copy with ONE pilot key removed everywhere (inventory, counted keys' consequences, built class) must STOP the 191 check
nk_minus = set(nk[1:])
checks.append(("(d) one pilot key removed everywhere -> S_old != 191", raises(rk.sold_sim, kept_for(nk_minus), nk_minus)))
a, b = inv_pair(); b["runs"] = [r for r in b["runs"] if r["key"] != nk[0]]
checks.append(("(d) one pilot row removed from fix_v2 -> STOP", raises(lambda: rk.new_rows_are_pilot(rk.inventory_delta(a, b), pairs))))


# 7. prior_match (09's (b) + (c)): the 10-16 run table vs this build, pilot excluded
def built(new_keys, folds=None, extra=()):
    rows = [{"name": f"A|c{i}|e1|{i:08d}", "key": f"a/{i}", "batch": "A", "set": "S_newmains", "fold": i % 5, "task": f"m{i % 7}", "cell": f"c{i}",
             "epoch": 1, "source": "S_newmains"} for i in range(30)]
    rows += [{"name": f"B|c{i}|e2|{i:08d}", "key": f"b/{i}", "batch": "B", "set": "S_old", "fold": i % 5, "task": f"m{i % 7}", "cell": f"c{i}",
              "epoch": 2, "source": "ep2"} for i in range(30)]
    fold_of = {f"m{j}": j % 5 for j in range(7)}
    for r in rows:
        r["fold"] = fold_of[r["task"]]
    pil = [{"name": f"sim|p{j}|e1|{j:08d}", "key": x, "batch": "sim", "set": "S_old", "task": f"m{j % 7}", "cell": f"p{j}", "epoch": 1,
            "source": "S_old"} for j, x in enumerate(sorted(new_keys))]
    for r in pil:
        r["fold"] = fold_of[r["task"]] if folds is None else folds
    return rows + pil + list(extra)


def prior_of(kept_rows, new_keys):
    old = [r for r in kept_rows if r["key"] not in new_keys]
    f = {"run_keys": "key", "run_set": "set", "run_folds": "fold", "run_main": "task", "run_cell": "cell", "run_epoch": "epoch", "run_source": "source"}
    return {k: {r["name"]: r[v] for r in old} for k, v in f.items()}


kb = built(new_keys)
pr = prior_of(kb, new_keys)
checks.append(("prior: good", rk.prior_match(pr, kb, new_keys) == {"equal": True, "n_prior": 60, "n_pilot": 20, "n_mains": 7}))
k2 = copy.deepcopy(kb); k2[0]["key"] = "a/changed"; checks.append(("prior: an A key changed", raises(rk.prior_match, pr, k2, new_keys)))
k2 = copy.deepcopy(kb); k2[31]["fold"] = (k2[31]["fold"] + 1) % 5; checks.append(("prior: a B run re-folded", raises(rk.prior_match, pr, k2, new_keys)))
k2 = copy.deepcopy(kb); k2.pop(5); checks.append(("prior: a 10-16 run missing", raises(rk.prior_match, pr, k2, new_keys)))
k2 = built(new_keys, extra=[{"name": "A|cx|e1|x", "key": "a/x", "batch": "A", "set": "S_newmains", "fold": 0, "task": "m0", "cell": "cx", "epoch": 1,
                             "source": "S_newmains"}])
checks.append(("prior: an extra non-pilot run", raises(rk.prior_match, pr, k2, new_keys)))
k2 = built(new_keys, folds=4); checks.append(("prior: pilot main re-folded (map differs)", raises(rk.prior_match, pr, k2, new_keys)))
k2 = copy.deepcopy(kb); k2[2]["set"] = "S_old"; checks.append(("prior: an S_newmains run's set changed", raises(rk.prior_match, pr, k2, new_keys)))

bad = [n for n, ok in checks if not ok]
if bad:
    raise SystemExit(f"FAIL {len(bad)}/{len(checks)}: {bad}")
print(f"PASS {len(checks)}/{len(checks)}")
