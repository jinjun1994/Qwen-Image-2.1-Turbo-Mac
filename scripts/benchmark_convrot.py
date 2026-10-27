#!/usr/bin/env python3
"""Benchmark real INT8 ConvRot projection shapes, including exact output checks.

Stop the WebUI model engine before running. Results do not imply end-to-end speed.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time


def tiled_projection(layer, x, rows=0, channels=0):
    import mlx.core as mx
    from mqwen.native_weights import rotate
    shape=x.shape
    x=rotate(x,layer.group) if layer.group else x
    x=x.reshape(-1,layer.in_features)
    row_size=rows or x.shape[0]
    channel_size=channels or layer.out_features
    outputs=[]
    for start in range(0,x.shape[0],row_size):
        parts=[]
        for channel in range(0,layer.out_features,channel_size):
            end=min(channel+channel_size,layer.out_features)
            parts.append(mx.quantized_matmul(x[start:start+row_size],layer.weight[channel:end],
                layer.scales[channel:end],layer.biases[channel:end],transpose=True,group_size=64,bits=8))
        outputs.append(parts[0] if len(parts)==1 else mx.concatenate(parts,axis=-1))
    y=outputs[0] if len(outputs)==1 else mx.concatenate(outputs,axis=0)
    y=y.reshape(*shape[:-1],layer.out_features)
    return y if layer.bias is None else y+layer.bias


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dit',default=str(Path.home()/'ComfyUI-Shared/models/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors'))
    p.add_argument('--weight-cache',default=str(Path.home()/'Library/Caches/mqwen-infer'))
    p.add_argument('--tokens',type=int,nargs='+',default=[1024,4096])
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--iterations',type=int,default=4)
    p.add_argument('--report',default='reports/convrot-tiling.json')
    a=p.parse_args()
    if min(*a.tokens,a.trials,a.iterations)<1:p.error('Counts must be positive')
    import mlx.core as mx
    from mqwen.native import Qwen21Transformer
    from mqwen.native_weights import dit_key,get_path,prime_rotations,rotate,clear_rotations
    from mqwen.prepared import load_or_convert
    mx.set_cache_limit(512*1024**2)
    model,status=load_or_convert(Qwen21Transformer,a.dit,'dit',dit_key,root=a.weight_cache)
    names=['transformer_blocks.0.attn.to_q','transformer_blocks.0.img_mlp.gate_layer','transformer_blocks.0.img_mlp.out']
    layers={name:get_path(model,name) for name in names}
    del model;gc.collect();mx.clear_cache()
    mx.eval([layer.parameters() for layer in layers.values()]);prime_rotations({l.group for l in layers.values() if l.group})
    rows=[]
    report=Path(a.report);report.parent.mkdir(parents=True,exist_ok=True)
    def save():
        report.write_text(json.dumps(dict(device=mx.device_info(),mlx=mx.__version__,weight_cache=status,
            note='Real projection weights, synthetic BF16 activations; compiled per-op timings; not full denoising; uncontrolled system/thermal state',results=rows),indent=2)+'\n')
    def measure(fn,x):
        for _ in range(2):mx.eval(fn(x))
        start=time.perf_counter()
        for _ in range(a.iterations):mx.eval(fn(x))
        return (time.perf_counter()-start)*1000/a.iterations
    for name,layer in layers.items():
        for tokens in a.tokens:
            x=mx.random.normal((1,tokens,layer.in_features),key=mx.random.key(42)).astype(mx.bfloat16);mx.eval(x)
            baseline=mx.compile(lambda x:layer(x))
            expected=baseline(x);mx.eval(expected)
            rotation=mx.compile(lambda x:rotate(x,layer.group))
            rotated=rotation(x);mx.eval(rotated)
            quantized=mx.compile(lambda x:mx.quantized_matmul(x,layer.weight,layer.scales,layer.biases,transpose=True,group_size=64,bits=8))
            components=dict(rotation_ms=measure(rotation,x),quantized_matmul_ms=measure(quantized,rotated))
            variants={(0,0):baseline}
            tiles=[(0,1024),(0,2048),(0,4096),(256,0),(512,0),(1024,0),(2048,0),(512,2048),(1024,4096)]
            for rm,cn in tiles:
                if (rm==0 or rm<tokens) and (cn==0 or cn<layer.out_features):
                    variants[(rm,cn)]=mx.compile(lambda x,rm=rm,cn=cn:tiled_projection(layer,x,rm,cn))
            records={}
            for tile,fn in variants.items():
                y=fn(x);mx.eval(y)
                record=dict(layer=name,m=tokens,n=layer.out_features,k=layer.in_features,rows=tile[0],channels=tile[1],
                    exact=bool(mx.all(y==expected).item()),max_error=float(mx.max(mx.abs(y.astype(mx.float32)-expected.astype(mx.float32))).item()),
                    different_fraction=float(mx.mean((y!=expected).astype(mx.float32)).item()),samples_ms=[],**components)
                records[tile]=record;rows.append(record)
            for trial in range(a.trials):
                order=list(variants) if trial%2==0 else list(reversed(variants))
                for tile in order:
                    records[tile]['samples_ms'].append(measure(variants[tile],x))
            base=statistics.median(records[(0,0)]['samples_ms'])
            for record in records.values():
                record['median_ms']=statistics.median(record['samples_ms'])
                record['relative_to_baseline']=record['median_ms']/base
                print(json.dumps(record),flush=True)
            save()
            del baseline,rotation,quantized,variants,expected,rotated,x,y;gc.collect();mx.clear_cache()
    clear_rotations()
    print(f'Saved {report}',flush=True)


if __name__=='__main__':main()
