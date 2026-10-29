# Restart-18 state edit, part 1 (d9 10-15 ~23:40 OK; 09's nm145 staging-retry ruling): il-s4rgap-t2 takes its box read from il-s4rgap's final read,
# because xbox2 (i-00db08777bb29d430) is stopped. Derivation: boxfiles_il-s4rgap-t2.json = chain.only_cells_view(boxfiles_il-s4rgap.json, {nm145});
# the full read is kept beside it. Asserts (each checked at 23:4x, recorded in the restart-18 notes): (1) that read (file mtime 19:07:03 PT) is after
# the launch's finished time 2026-10-15T18:17:15-07:00; (2) nm145's .eval and launch.json sha256 on the Volume equal the read's (hash_on_volume:
# both EQUAL). Re-asserted here from the files: the read's finished time, and the nm145 row's sha in the committed manifest @ 5cbd3354c.
import copy, datetime, json, os, subprocess, sys, time
st_dir = sys.argv[1]
sys.path.insert(0, sys.argv[2])  # the run worktree's chain dir (chain.only_cells_view, the reviewed code)
import chain as C  # noqa: E402
p = os.path.join(st_dir, "state.json")
d = json.load(open(p))
b = d["batches"]["il-s4rgap-t2"]
assert b["status"] != "STOPPED" and not b["done"] and b.get("only_cells") == ["nm145_model_registry"], (b["status"], b["done"])
src = os.path.join(st_dir, "boxfiles_il-s4rgap.json")
full = json.load(open(src))
fin = datetime.datetime.fromisoformat(full["facts"]["finished"])
read_t = datetime.datetime.fromtimestamp(os.path.getmtime(src)).astimezone()
assert read_t > fin, (read_t, fin)
man = json.loads(subprocess.run(["git", "show", "5cbd3354c:experiments/2026-10-14_q3_overnight_chain/config/manifest_ilnm_s4rgap_sim2.json"],
                                capture_output=True, check=True, cwd=sys.argv[2]).stdout)
want = {f["path"]: f["sha256"] for f in man["files"] if f["task"] == "nm145_model_registry" and f["arm_dir"] == "sim_attack@september"}
got = {f["path"]: f["sha256"] for m in full["manifests"].values() for f in m["files"] if f["path"] in want}
assert len(want) == 1 and got == want, (want, got)
view = C.only_cells_view(copy.deepcopy(full), {"nm145_model_registry"})
assert view["members"] == ["sim_attack@september/nm145_model_registry"], view["members"]
dst = os.path.join(st_dir, "boxfiles_il-s4rgap-t2.json")
assert not os.path.exists(dst)
open(os.path.join(st_dir, "boxfiles_il-s4rgap-t2.full.json"), "x").write(json.dumps(full))
open(dst, "x").write(json.dumps(view))
b["done"] = ["ended", "boxfiles"]
b["members"] = view["members"]
b["files"], b["bytes"], b["counted_without_eval"] = len(view["files"]), sum(f["bytes"] for f in view["files"]), len(view["counted_without_eval"])
b["box_read_from"] = {"source": "boxfiles_il-s4rgap.json (chain state; read 19:07:03 PT, after finished 18:17:15)", "view": "chain.only_cells_view",
                      "asserts": ["read after launch.json finished", "nm145 .eval + launch.json sha256 on the Volume == the read's (hash_on_volume, EQUAL)"],
                      "why": "xbox2 stopped; d9 OK 10-15 ~23:40"}
b["status"] = "waiting"
open(p + ".tmp", "w").write(json.dumps(d, indent=1)); os.replace(p + ".tmp", p)
print("il-s4rgap-t2: ended + boxfiles from il-s4rgap's final read;", len(view["files"]), "files; members", view["members"], time.strftime("%H:%M:%S"))
