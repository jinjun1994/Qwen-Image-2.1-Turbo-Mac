#!/usr/bin/env python3
"""Compare rotation reuse or tuned GEMM on one resident model, same sampler."""
import argparse
import json
from pathlib import Path
import re
import statistics
import time
from types import SimpleNamespace


def main():
    import mqwen
    assets=Path(mqwen.__file__).parent/'assets'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',default=str(Path.home()/'Documents/ComfyUI2/models'))
    p.add_argument('--extra-models',action='append',default=[])
    p.add_argument('--weight-cache',default=str(Path.home()/'Library/Caches/mqwen-infer'))
    p.add_argument('--workflow',default=str(assets/'turbo_t2i.json'))
    p.add_argument('--tokenizer',default=str(assets/'tokenizer'))
    p.add_argument('--width',type=int,default=512)
    p.add_argument('--height',type=int,default=512)
    p.add_argument('--steps',type=int,choices=[4,6,8],default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--runs',type=int,default=1,help='Number of legacy/shared pairs')
    p.add_argument('--optimization',choices=['rotation','gemm'],default='rotation')
    p.add_argument('--denoise-only',action='store_true',help='Compare production sampler latents without VAE decoding; standard path only')
    p.add_argument('--step-cache',action='store_true',help='Compare both paths with the existing balanced cache enabled')
    p.add_argument('--prompt',default='A small red ceramic teapot on a wooden table beside a window, soft morning sunlight, realistic photograph.')
    p.add_argument('--report',default='reports/denoise-benchmark.json')
    p.add_argument('--output-dir',default='outputs')
    args=p.parse_args()
    if args.runs<1:p.error('--runs must be positive')
    if args.denoise_only and args.step_cache:p.error('--denoise-only does not support --step-cache')
    if args.step_cache and args.steps!=8:p.error('--step-cache requires 8 steps')
    if not 0<=args.seed<2**64:p.error('--seed must be an unsigned 64-bit integer')
    from mqwen.generate import output_size
    output_size({},args.width,args.height)
    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from mqwen.native import NativeEngine,denoise_latents
    from mqwen.tokenizer import PROMPT_TEMPLATE,SYSTEM_PREFIX
    from mqwen.workflow import Workflow
    import mqwen.native_weights as weights
    target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True)
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    logs=[]
    def log(*items,**kw):
        line=' '.join(map(str,items));logs.append(line);print(line,flush=kw.get('flush',False))
    options=SimpleNamespace(**vars(args),dit=None,text_encoder=None,vae=None,gemm_backend='auto' if args.optimization=='gemm' else 'mlx')
    shared=weights.rotate
    def legacy(x,group):
        return (x.reshape(-1,group)@weights.hadamard(group).astype(x.dtype)).reshape(x.shape)
    engine=NativeEngine(options,log=log)
    rows=[];reference=None
    tuned_layers=[l for _,l in engine.transformer.named_modules() if getattr(l,'_tuned_gemm',False)]
    tuned_backend=engine.gemm_backend
    modes=('legacy','shared') if args.optimization=='rotation' else ('mlx','tuned')
    try:
        if args.optimization=='gemm' and not tuned_layers:
            raise ValueError('Tuned GEMM is unavailable on this device/version or failed preflight')
        if args.denoise_only:
            ids=engine.tokenizer.encode(PROMPT_TEMPLATE.format(args.prompt))
            drop=len(engine.tokenizer.encode(SYSTEM_PREFIX))
            positions=mx.broadcast_to(mx.arange(len(ids))[None,None,:],(3,1,len(ids)))
            embeds=engine.text(engine.text.embed_tokens(mx.array([ids],dtype=mx.int32)),positions)[:,drop:]
            mx.eval(embeds)
            schedule=Workflow(args.workflow).plan()['sigmas']
            if args.steps!=8:schedule=np.interp(np.linspace(0,8,args.steps+1),np.arange(9),schedule).tolist()
        for run in range(args.runs):
            for mode in (modes if run%2==0 else tuple(reversed(modes))):
                logs.clear()
                weights.rotate=legacy if mode=='legacy' else shared
                engine.denoise_optimizations=[] if mode=='legacy' else ['shared_convrot_constants']
                if args.optimization=='gemm':
                    for layer in tuned_layers:layer._tuned_gemm=mode=='tuned'
                    engine.gemm_backend=tuned_backend if mode=='tuned' else 'mlx'
                    engine.gemm_layers=len(tuned_layers) if mode=='tuned' else 0
                    if mode=='tuned':engine.denoise_optimizations.append(tuned_backend)
                engine.transformer._step_fn=engine.transformer._image_step_fn=None
                # Balanced-cache compiled functions also capture the chosen rotation.
                engine._step_cache=None
                options.output=str(output/f'denoise-{mode}-{time.time_ns()}.png')
                if args.denoise_only:
                    started=time.perf_counter()
                    latents=denoise_latents(engine.transformer,embeds,args.width,args.height,args.seed,schedule,log=log)
                    mx.synchronize()
                    row=dict(timings_seconds={'denoise':time.perf_counter()-started},width=args.width,height=args.height,
                             steps=args.steps,seed=args.seed,prompt=args.prompt,mlx=mx.__version__,
                             weight_cache=engine.weight_cache,engine_load_seconds=engine.load_seconds,
                             gemm_backend=engine.gemm_backend,gemm_layers=engine.gemm_layers,
                             denoise_optimizations=list(engine.denoise_optimizations))
                else:
                    row=engine.generate(options)
                row.update(mode=mode,run=run+1)
                steps=[float(m.group(1)) for line in logs if (m:=re.search(r'denoise \d+/\d+: ([\d.]+)s',line))]
                row['step_seconds']=steps
                row['warm_step_mean']=statistics.mean(steps[1:])
                pixels=np.asarray(latents.astype(mx.float32)) if args.denoise_only else np.array(Image.open(row['output']))
                if reference is None:reference=pixels
                exact_key='latent_exact' if args.denoise_only else 'pixel_exact'
                row[exact_key]=bool(np.array_equal(pixels,reference))
                if not row[exact_key]:raise AssertionError('Optimization changed sampler output')
                rows.append(row)
                target.write_text(json.dumps(dict(note='One resident model; OS/thermal state uncontrolled; step 1 includes graph setup',denoise_only=args.denoise_only,optimization=args.optimization,runs=rows),indent=2)+'\n')
                print('DENOISE',mode,row['timings_seconds']['denoise'],'warm step',row['warm_step_mean'],flush=True)
    finally:
        weights.rotate=shared
        tuned_layers.clear()
        layer=None
        engine.close()
    print(f'Saved {target}')

if __name__=='__main__':main()
