#!/usr/bin/env python3
"""Tune Metal INT8 MMA tiles on real DiT weights; stop WebUI inference first."""
import argparse
import gc
import itertools
import json
from pathlib import Path
import statistics
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dit',default=str(Path.home()/'ComfyUI-Shared/models/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors'))
    p.add_argument('--weight-cache',default=str(Path.home()/'Library/Caches/mqwen-infer'))
    p.add_argument('--tokens',type=int,nargs='+',default=[1024,4096])
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--iterations',type=int,default=4)
    p.add_argument('--report',default='reports/metal-gemm-tiles.json')
    args=p.parse_args()
    if min(*args.tokens,args.trials,args.iterations)<1:p.error('Counts must be positive')
    import mlx.core as mx
    from mqwen.native import Qwen21Transformer
    from mqwen.native_weights import get_path,dit_key
    from mqwen.prepared import load_or_convert
    from mqwen.metal_gemm import _kernel
    target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True)
    def qmm(x,layer,bm,bn,bk):
        m=x.size//x.shape[-1];k=x.shape[-1];n=layer.out_features
        return _kernel()(inputs=[x,layer.weight,layer.scales[:,:1].reshape(-1),layer.biases[:,:1].reshape(-1)],
            template=[('T',x.dtype),('M',m),('N',n),('K',k),('BM',bm),('BN',bn),('BK',bk)],
            grid=((n+bn-1)//bn*128,(m+bm-1)//bm,1),threadgroup=(128,1,1),
            output_shapes=[(*x.shape[:-1],n)],output_dtypes=[x.dtype])[0]

    mx.set_cache_limit(512*1024**2)
    m,status=load_or_convert(Qwen21Transformer,args.dit,'dit',dit_key,root=args.weight_cache)
    layers={n:get_path(m,n) for n in ['transformer_blocks.0.attn.to_q','transformer_blocks.0.img_mlp.gate_layer','transformer_blocks.0.img_mlp.out']}
    del m;gc.collect();mx.clear_cache();rows=[]
    for name,l in layers.items():
     for tokens in args.tokens:
      x=mx.random.normal((1,tokens,l.in_features),key=mx.random.key(42)).astype(mx.bfloat16);mx.eval(x)
      native=mx.compile(lambda x:mx.quantized_matmul(x,l.weight,l.scales,l.biases,transpose=True,group_size=64,bits=8))
      ref=native(x);mx.eval(ref)
      fns={'native':native}
      for bm,bn,bk in itertools.product((16,32,64),(32,64,128),(32,64)):
       fns[f'{bm}x{bn}x{bk}']=mx.compile(lambda x,bm=bm,bn=bn,bk=bk:qmm(x,l,bm,bn,bk))
      records={}
      for tile,fn in fns.items():
       y=fn(x);mx.eval(y)
       r=dict(layer=name,m=tokens,n=l.out_features,k=l.in_features,tile=tile,
        max_error=float(mx.max(mx.abs(ref.astype(mx.float32)-y.astype(mx.float32))).item()),
        different_fraction=float(mx.mean((ref!=y).astype(mx.float32)).item()),samples_ms=[])
       records[tile]=r;rows.append(r)
      for trial in range(args.trials):
       order=list(fns) if trial%2==0 else list(reversed(fns))
       for tile in order:
        fn=fns[tile]
        for _ in range(2):mx.eval(fn(x))
        t=time.perf_counter()
        for _ in range(args.iterations):mx.eval(fn(x))
        records[tile]['samples_ms'].append((time.perf_counter()-t)*1000/args.iterations)
      base=statistics.median(records['native']['samples_ms'])
      for r in records.values():
       r['median_ms']=statistics.median(r['samples_ms']);r['ratio']=r['median_ms']/base
      for r in sorted(records.values(),key=lambda r:r['median_ms']):print(json.dumps(r),flush=True)
      target.write_text(json.dumps(dict(mlx=mx.__version__,device=mx.device_info(),note='Real weights, synthetic BF16 activations; quantized GEMM only; no ConvRot or whole-image timing; system and thermal state uncontrolled',results=rows),indent=2)+'\n')
      del fns,ref,x,y;gc.collect();mx.clear_cache()

if __name__=='__main__':main()
