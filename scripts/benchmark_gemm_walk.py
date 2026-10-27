#!/usr/bin/env python3
"""Compare output-tile walks on real DiT projections. Stop WebUI inference first."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
from gemm_walk import SCHEDULES,matmul as scheduled


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dit',default=str(Path.home()/'ComfyUI-Shared/models/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors'))
    p.add_argument('--weight-cache',default=str(Path.home()/'Library/Caches/mqwen-infer'))
    p.add_argument('--tokens',type=int,nargs='+',default=[1024,4096])
    p.add_argument('--schedules',nargs='+',choices=SCHEDULES,default=list(SCHEDULES))
    p.add_argument('--trials',type=int,default=4)
    p.add_argument('--iterations',type=int,default=4)
    p.add_argument('--report',default='reports/gemm-walk-benchmark.json')
    args=p.parse_args()
    if min(*args.tokens,args.trials,args.iterations)<1:p.error('Counts must be positive')
    import mlx.core as mx
    from mqwen.native import Qwen21Transformer
    from mqwen.native_weights import get_path,dit_key
    from mqwen.prepared import load_or_convert
    from mqwen.metal_gemm import matmul
    target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True)
    mx.set_cache_limit(512*1024**2)
    model,status=load_or_convert(Qwen21Transformer,args.dit,'dit',dit_key,root=args.weight_cache)
    names=['transformer_blocks.0.attn.to_q','transformer_blocks.0.img_mlp.gate_layer','transformer_blocks.0.img_mlp.out']
    layers={n:get_path(model,n) for n in names}
    del model;gc.collect();mx.clear_cache();results=[]
    for name,layer in layers.items():
        for tokens in args.tokens:
            x=mx.random.normal((1,tokens,layer.in_features),key=mx.random.key(42)).astype(mx.bfloat16);mx.eval(x)
            fns={'current':mx.compile(lambda x:matmul(x,layer))}
            fns.update({s:mx.compile(lambda x,s=s:scheduled(x,layer,s)) for s in args.schedules})
            expected=fns['current'](x);mx.eval(expected);records={}
            for key,fn in fns.items():
                y=fn(x);mx.eval(y)
                row=dict(layer=name,m=tokens,n=layer.out_features,k=layer.in_features,schedule=key,
                    exact=bool(mx.all(y==expected).item()),max_error=float(mx.max(mx.abs(y.astype(mx.float32)-expected.astype(mx.float32))).item()),samples_ms=[])
                if not row['exact']:raise AssertionError(f'Incorrect traversal: {key}, {row}')
                records[key]=row;results.append(row)
            for trial in range(args.trials):
                keys=list(fns) if trial%2==0 else list(reversed(fns))
                for key in keys:
                    fn=fns[key];mx.eval(fn(x));start=time.perf_counter()
                    for _ in range(args.iterations):mx.eval(fn(x))
                    records[key]['samples_ms'].append((time.perf_counter()-start)*1000/args.iterations)
            baseline=statistics.median(records['current']['samples_ms'])
            for row in records.values():row['median_ms']=statistics.median(row['samples_ms']);row['ratio']=row['median_ms']/baseline
            for row in sorted(records.values(),key=lambda r:r['median_ms']):print(json.dumps(row),flush=True)
            target.write_text(json.dumps(dict(mlx=mx.__version__,device=mx.device_info(),weight_cache=status,
                note='Real weights, synthetic BF16 activations; QMM only, not full generation. Alternating trial order; OS and thermal state uncontrolled.',results=results),indent=2)+'\n')
            del fns,x,y,expected;gc.collect();mx.clear_cache()
    print(f'Saved {target}',flush=True)


if __name__=='__main__':main()
