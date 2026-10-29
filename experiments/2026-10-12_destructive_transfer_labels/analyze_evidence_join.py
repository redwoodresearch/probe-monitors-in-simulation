"""Independently verify every private packet join against the hashed blind bundle."""
import argparse
import json
from pathlib import Path
from probe_shift.dt_label_prepare import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('commands','definitions','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepared',type=Path,action='append',required=True)
    a=p.parse_args()
    commands_hash,definitions_hash=sha(a.commands),sha(a.definitions)
    with a.commands.open() as f:source={r['blind_id']:r for r in map(json.loads,f)}
    with a.definitions.open() as f:definitions={r['tool_def_id']:r['definition'] for r in map(json.loads,f)}
    report={'commands_sha256':commands_hash,'definitions_sha256':definitions_hash,'stages':[]}
    for prepared in a.prepared:
        metadata=json.loads((prepared/'preparation.json').read_text())
        if metadata['commands_sha256']!=commands_hash or metadata['definitions_sha256']!=definitions_hash:
            raise ValueError('Prepared stage refers to another source bundle')
        join=json.loads((prepared/'join.json').read_text())
        mapping={(r['packet'],r['id']):r['blind_id'] for r in join}
        if len(mapping)!=len(join) or len(set(mapping.values()))!=len(join):raise ValueError('Join is not unique')
        seen=set()
        for packet in json.loads((prepared/'packets.json').read_text()):
            path=prepared/'inputs'/(packet['packet']+'.json')
            if sha(path)!=packet['sha256']:raise ValueError('Packet hash mismatch')
            payload=json.loads(path.read_text())
            if [c['id'] for c in payload['calls']]!=packet['ids']:raise ValueError('Packet IDs changed')
            for call in payload['calls']:
                k=(packet['packet'],call['id']);original=source[mapping[k]]
                if original['arguments']!=call['args'] or original['tool']!=call['tool']:
                    raise ValueError('Command evidence is joined to the wrong blinded row')
                if definitions[original['tool_def_id']]!=payload['tool_definitions'][call['definition']]:
                    raise ValueError('Tool definition is joined to the wrong blinded row')
                seen.add(k)
        if seen!=set(mapping):raise ValueError('Join contains unmatched calls')
        report['stages'].append({'prepared':str(prepared),'verified_calls':len(seen),
                               'join_sha256':sha(prepared/'join.json'),'packets_sha256':sha(prepared/'packets.json')})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
