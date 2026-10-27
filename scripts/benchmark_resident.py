#!/usr/bin/env python3
"""Measure repeated requests to the resident WebUI, preserving actual run records."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:7860')
    p.add_argument('--width',type=int,default=512)
    p.add_argument('--height',type=int,default=512)
    p.add_argument('--runs',type=int,default=2)
    p.add_argument('--seed',default='42')
    p.add_argument('--steps',type=int,choices=[4,6,8],default=8)
    p.add_argument('--step-cache',action='store_true')
    p.add_argument('--prompt',default='A small red ceramic teapot on a wooden table beside a window, soft morning sunlight, realistic photograph.')
    p.add_argument('--report',default='reports/resident-benchmark.json')
    args=p.parse_args()
    if args.runs<1:p.error('--runs must be positive')
    def get(route):
        with urllib.request.urlopen(args.url+route,timeout=10) as r:return json.load(r)
    for _ in range(180):
        s=get('/api/state')
        if s['engine_state']=='failed':raise RuntimeError(s.get('engine_error'))
        if s['engine_state']=='ready':break
        time.sleep(1)
    else:raise TimeoutError('Engine did not become ready')
    if s['active']:raise RuntimeError('Another generation is already running')
    records=[]
    for i in range(args.runs):
        body=dict(prompt=args.prompt,seed=args.seed,mode='resolution',width=args.width,height=args.height,steps=args.steps,step_cache=args.step_cache)
        request=urllib.request.Request(args.url+'/api/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=10) as r:key=json.load(r)['id']
        while True:
            state=get('/api/state')
            job=next(j for j in state['jobs'] if j['id']==key)
            if job['status'] not in ('running','cancelling'):break
            time.sleep(1)
        if job['status']!='completed':raise RuntimeError(job.get('error',job['stage']))
        result=get(job['metadata_url']);records.append(result)
        print(f"Run {i+1}: {result['total_seconds']}s; prompt cache hit={result['prompt_cache_hit']}; {result['timings_seconds']}",flush=True)
    report=Path(args.report);report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps({'runs':records},ensure_ascii=False,indent=2)+'\n')
    print(f'Saved {report}')

if __name__=='__main__':main()
