"""Build the S_paper493 sim L1/L2 post-start record and the two start rollout rows from the box reads."""
import json, sys
SP, OUT, ROWS = sys.argv[1:]
def first_read(k):
    lines = open(f'{SP}/sp493/postlaunch_{k}_read.txt').read().splitlines()
    return [l for l in lines if l.startswith('PID')][-1].split()[1], json.loads([l for l in lines if l.startswith('POST ')][-1][5:])
L = {
 'L1': dict(box='box3 i-0cc7555acb95c694b', started='2026-10-15T19:08:30-07:00', set='a015ee', servers='sp493/srv/results/servers_2026-10-15_185847_a015ee.json', servers_sha='5b68648574bf1e3655151a4d2d2cfb0f6dc697fab8013a1c06c6f6791e587135',
   gate='rr sp493L1 ALL CONDITIONS HOLD 19:07:57 (servers a015ee 6/6 live; census sim_error_census_2026-10-15_1859.json @ c5bca8847: 18:59:03 RL 0/27, running 3 <= 65)',
   freeze='IL frozen +311 manifest 71fd27724a045d3df5d8dba9b6e63d5f86514377c025a11a57fd78f6bf1aa287 (140 files) except freeze_exempt_box3.tsv cd0fa40708e7a7d30fb31d3765bddef5387a567ffad65c35b8040004dea63797: exempt paths absent or pinned',
   settings='results/gate_settings_sp493_sim_L1_1792116480.json (sha256 34b37d0209c7e48d...)', guards='scsp493simL1 (sim_cap_stop 4346), gpsp493simL1 (gate_pause_311_rl.py); verified 19:09:33 (GUARDS 1 1)', cap=4246),
 'L2': dict(box='box4 i-07bce857a818b6f69', started='2026-10-15T19:21:02-07:00', set='78e85d', servers='sp493/srv/results/servers_2026-10-15_185850_78e85d.json', servers_sha='9a697c8c92597af09c22295376a7a56902ef19437d386961448f4b9d7b49dc26',
   gate='rr sp493L2 ALL CONDITIONS HOLD 19:20:30 (servers 78e85d 6/6 live; census sim_error_census_2026-10-15_1909.json @ 9a63d783e: 19:09:54 RL 0/6, running 1 <= 65)',
   freeze='IL frozen +311 manifest 71fd27724a045d3df5d8dba9b6e63d5f86514377c025a11a57fd78f6bf1aa287 (137 files) except freeze_exempt_box4.tsv 536bf688318304c04e31db436b447036906adb0d0a3d88f162b20a56adf0fec0: exempt paths absent or pinned',
   settings='results/gate_settings_sp493_sim_L2_1792117232.json (sha256 34b37d0209c7e48d...)', guards='scsp493simL2 (sim_cap_stop 3978), gpsp493simL2 (gate_pause_311_rl.py); verified 19:22:05 (GUARDS 1 1)', cap=3878,
   stale_canary_exemption=[
     {'run_dir': '/home/aswin/phase2/runs/2026-10-14_153435_canary_a', 'launch_json_sha256': '031d6972d0f7c5c179e18419faa2bdde09ee225ebf82349f13250918dfbcf37a', 'launch_json_mtime': 1792017901},
     {'run_dir': '/home/aswin/phase2/runs/2026-10-14_153435_canary_b', 'launch_json_sha256': 'e99b2af227b7acf0186644aff2b24b7ddb0a6a1ff345db83485246a9a682aafb', 'launch_json_mtime': 1792017845}],
   stale_canary_reason='dead 10-14 canary launches (finished null) on box4; d9 ~19:12 relayed the acceptance by 09 (no committed 09 file): exactly these two paths, matched by full path, pinned by launch.json sha256 + mtime, and no python process naming them (sp493sim_box.sh v2 @ 6adacd661, starter v16); both accepted 19:20:32; any other unfinished launch.json still blocks; the L2 run must never reference them (post-start count 0)'),
}
rec = {'what': 'S_paper493 sim-attack L1/L2 post-start checks: consumed references vs the refindex pins by key; server checks cited by canonical sha256; recorded inputs; exempt-file and canary references. Read-only on box3/box4; reference/run-state/server-check fields only, no outcome field', 'launches': {}}
rows = []
for k, m in L.items():
    pid, fr = first_read(k)
    bk = json.load(open(f'{SP}/sp493/postlaunch_{k}.byKey.json'))
    refs = bk['refs']; neq = sum(r['equal'] for r in refs)
    assert len(refs) == neq == {'L1': 238, 'L2': 217}[k], (k, len(refs), neq)
    assert all(r['reference_state'] == 'usable' for r in refs)
    assert fr['server_checks']['n'] == fr['server_checks']['exit0'] == 6
    ex = fr.get('exempt_refs') or fr.get('exempt_and_canary_refs'); assert not any(ex.values()), ex
    rec['launches'][k] = {**{x: m[x] for x in m if x not in ('servers',)}, 'servers_record': m['servers'], 'run': bk['run'], 'launcher_pid': int(pid),
      'first_read': {'launch_json_sha256': fr['launch_json_sha256'], 'server_checks': fr['server_checks'], 'inputs': fr['inputs'], 'exempt_or_canary_ref_counts': ex},
      'second_read_utc': bk['read_utc'], 'launch_json_sha256_at_second_read': bk['launch_json_sha256_at_read'], 'refindex': bk['refindex'], 'refindex_sha256': bk['refindex_sha256'],
      'equal_by_key': f'{neq}/{len(refs)}', 'refs': refs}
    sc = fr['server_checks']['sha256_canonical'][:8]; inp = fr['inputs']; cfg = [v for kk, v in inp.items() if kk.startswith('config/pilot_')][0][:8]; tsk = [v for kk, v in inp.items() if kk.startswith('config/tasks_')][0][:8]
    reason = (f"S_paper493 sim-attack {k} (sims only, real_arms [], max_runs_per_server 5; run list results/sp493_sim_runlist_{k}_2026-10-15.json; 09 ack as relayed by d9 in this session, no committed 09 file); approval as relayed: SPEND row FINAL @ 66790e7c1 (Aswin via 0d ~18:57: \"i said okay for all, go 493, replay, okay\", taken as GO by 0d). "
      f"Cap ${m['cap']:,}. {m['gate']}. Box script sp493sim_box.sh: freeze = {m['freeze']}; settings PASS {m['settings']}; guards {m['guards']}; STARTED {m['started'][11:19]}; launcher PID {pid}. "
      + (f"Stale-canary exemption: {m['stale_canary_reason']}. " if k == 'L2' else '')
      + f"Post-start (results/sp493_sim_postlaunch_checks_2026-10-15.json): consumed references = the {neq} refindex pins by key (refindex sha256 {bk['refindex_sha256'][:8]}...), all usable; server checks 6/6 exit 0 (canonical sha256 {sc}...); inputs config {cfg} / tasks {tsk}; exempt{' and canary' if k == 'L2' else ''} files referenced 0 times. 22:45 PAUSE_SIMS guard armed (sp493/pause_2245.sh, PID 91713). Outcomes sealed until 09's read gates")
    row = {'time_pt': m['started'], 'action': 'launch_start', 'box': m['box'], 'launch': f"sp493_sim_{k} ({bk['run']}; session sp493sim{k}; launcher PID {pid})", 'il_launch': False, 'reason': reason,
      'source': f"scratchpad il_watch/rr_sp493{k}.log; {m['box'].split()[0]} reads 19:2x-19:30", 'outcomes_read': False, 'outcome_info_available': 'none', 'reason_class': 'registered', 'reported_by': 'design fork'}
    if k == 'L2': row['stale_canary_exemption'] = m['stale_canary_exemption']
    rows.append(row)
json.dump(rec, open(OUT, 'w'), indent=1); open(OUT, 'a').write('\n')
open(ROWS, 'w').write(''.join(json.dumps(r) + '\n' for r in rows))
print('ok', {k: v['equal_by_key'] for k, v in rec['launches'].items()})
