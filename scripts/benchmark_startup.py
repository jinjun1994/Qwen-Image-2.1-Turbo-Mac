#!/usr/bin/env python3
"""Interleave source/prepared startup measurements; does not purge OS file caches."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace


def main():
    import mqwen
    assets=Path(mqwen.__file__).parent/'assets'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',default=str(Path.home()/'Documents/ComfyUI2/models'))
    p.add_argument('--extra-models',action='append',default=[])
    p.add_argument('--workflow',default=str(assets/'turbo_t2i.json'))
    p.add_argument('--tokenizer',default=str(assets/'tokenizer'))
    p.add_argument('--weight-cache',default=str(Path.home()/'Library/Caches/mqwen-infer'))
    p.add_argument('--runs',type=int,default=2,help='Number of source/prepared pairs')
    p.add_argument('--report',default='reports/startup-benchmark.json')
    args=p.parse_args()
    if args.runs<1:p.error('--runs must be positive')
    import mlx.core as mx
    from mqwen.native import NativeEngine
    rows=[]
    target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True)
    for run in range(args.runs):
        for mode in ('source','prepared'):
            options=SimpleNamespace(**vars(args),dit=None,text_encoder=None,vae=None)
            options.weight_cache=None if mode=='source' else args.weight_cache
            mx.reset_peak_memory()
            start=time.perf_counter()
            engine=NativeEngine(options)
            row=dict(run=run+1,mode=mode,seconds=round(time.perf_counter()-start,3),
                     engine_load_seconds=engine.load_seconds,weight_cache=engine.weight_cache,
                     active_bytes=mx.get_active_memory(),peak_bytes=mx.get_peak_memory())
            engine.close()
            row['released_active_bytes']=mx.get_active_memory()
            rows.append(row)
            target.write_text(json.dumps(dict(note='OS file cache not purged; imports excluded; startup only, no graph warmup',runs=rows),indent=2)+'\n')
            print('STARTUP',row,flush=True)
            if mode=='prepared' and set(row['weight_cache'].values())!={'hit'}:
                raise RuntimeError('Prepared measurement did not hit every weight cache')
    print(f'Saved {target}')

if __name__=='__main__':main()
