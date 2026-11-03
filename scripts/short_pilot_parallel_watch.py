"""Observe approved overlapping pilots and release frozen Task09 final evaluations."""

import json
import subprocess
import time

import short_pilot_sequence as S

NAMES=('task08-swarm-02','task08-astra','task09-swarm','task09-astra','task07')


def main():
    records={name:S.read(S.SETUP/f'{name}-process.json') for name in NAMES}
    if not all(records.values()):raise ValueError('Missing parallel launch records')
    reported=set()
    results=[]
    barrier=S.SETUP/'task09-final-release.json'
    while time.time()<S.STOP:
        active={name:S.running(record) for name,record in records.items()}
        if not barrier.exists():
            if all((S.output(records[name])/'selection-frozen.json').exists() or not active[name]
                   for name in ('task09-swarm','task09-astra')):
                S.write(barrier,{'released_at':time.time(),'all_conditions_frozen_or_stopped':True})
        for name,record in records.items():
            if name in reported:continue
            summary=S.read(S.output(record)/'summary.json')
            if summary or not active[name]:
                reported.add(name)
                row={'name':name,'output':str(S.output(record)),'summary':summary,'at':time.time()}
                results.append(row)
                S.write(S.SETUP/'parallel-results.json',results)
                if summary:
                    text=f"{name} finished: status={summary.get('status')}, score={summary.get('score')}, API estimate=${summary.get('api',{}).get('estimated_cost_usd',0):.4f}. Shortened/exploratory; cumulative series API estimate=${S.spend():.4f}."
                else:text=f'{name} stopped without summary; see operator log. No score claimed. Cumulative API estimate=${S.spend():.4f}.'
                S.message(text)
        S.write(S.SETUP/'parallel-progress.json',{'at':time.time(),'active':active,'completed':sorted(reported),
            'total_estimated_spend_usd':S.spend(),'combined_ceiling_usd':26,'schedule':'overlapping_tasks',
            'task08_processes_and_configuration_untouched':True})
        if not any(active.values()):break
        try:
            stats=subprocess.run(['docker','stats','--no-stream','--format','{{json .}}'],
                                 capture_output=True,text=True,timeout=5,check=True)
            rows=[json.loads(line) for line in stats.stdout.splitlines() if line.strip()]
            with (S.SETUP/'docker-observations.jsonl').open('a') as handle:
                handle.write(json.dumps({'at':time.time(),'containers':rows})+'\n')
        except (subprocess.SubprocessError,ValueError):pass
        time.sleep(10)
    S.write(S.SETUP/'parallel-summary.json',{'at':time.time(),'results':results,
        'total_estimated_api_spend_usd':S.spend(),'combined_ceiling_usd':26,
        'status':'finished_or_deadline','recorded_conditions':len(results)})
    S.message(f'Overlapping shortened pilots finished/deadline reached. Total recorded API estimate=${S.spend():.4f}; operator report outputs/short-pilots-20261102/parallel-summary.json.')


if __name__=='__main__':main()
