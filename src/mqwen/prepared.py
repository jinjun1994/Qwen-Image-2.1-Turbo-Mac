"""Offline, immutable MLX-ready weights. Reads never create or rewrite a cache."""
from __future__ import annotations
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from .native_weights import ConvRotEmbedding, ConvRotLinear, get_path, set_path
from .weights import WeightFile

DEFAULT_CACHE = Path.home()/'Library/Caches/mqwen-infer'
SCHEMA = 1
ADAPTER = 'convrot-int8-affine64-row-bf16-vae-f32-v1'


def _digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(path):
    path=Path(path).expanduser().resolve()
    stat=path.stat()
    with path.open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8:raise ValueError('Truncated source weights')
        size=struct.unpack('<Q',raw)[0]
        if not 2<=size<=min(100_000_000,stat.st_size-8):raise ValueError('Invalid source header')
        header=f.read(size)
    return dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns,device=stat.st_dev,inode=stat.st_ino,
                header_sha256=hashlib.sha256(raw+header).hexdigest())


def identity(path,role):
    return dict(schema=SCHEMA,adapter=ADAPTER,role=role,source=fingerprint(path),
                mlx=mx.__version__,mflux=importlib.metadata.version('mflux'))


def location(root,ident):
    key=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()[:24]
    return Path(root).expanduser().resolve()/f"{ident['role']}-{key}"


def _validate(folder,ident,verify=False):
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['identity']!=ident:raise ValueError('Prepared weight identity mismatch')
    payload=folder/'weights.safetensors'
    stat=payload.stat()
    if stat.st_size!=manifest['payload_bytes'] or stat.st_mtime_ns!=manifest['payload_mtime_ns']:
        raise ValueError('Prepared payload changed or is incomplete')
    index=WeightFile(payload)
    if index.tensors!=manifest['tensors']:raise ValueError('Prepared tensor index mismatch')
    if not isinstance(manifest['layers'],dict):raise ValueError('Invalid prepared layers')
    if verify:
        if _digest(payload)!=manifest['payload_sha256']:raise ValueError('Prepared payload SHA256 mismatch')
        if _digest(ident['source']['path'])!=manifest['source_sha256']:raise ValueError('Source SHA256 mismatch')
    return manifest


def load_prepared(module,path,role,root):
    ident=identity(path,role)
    folder=location(root,ident)
    if not (folder/'manifest.json').exists():return None
    manifest=_validate(folder,ident)
    arrays=mx.load(str(folder/'weights.safetensors'))
    original=dict(tree_flatten(module.parameters()))
    for name,desc in manifest['layers'].items():
        if name+'.weight' not in original:raise ValueError(f'Unknown prepared layer: {name}')
        previous=get_path(module,name)
        weight=arrays[name+'.weight']
        group=desc['group']
        if type(group) is not int or group<0 or (group and (group<4 or group&(group-1) or (group.bit_length()-1)%2)):
            raise ValueError('Invalid ConvRot group')
        if desc['type']=='linear' and type(previous) is nn.Linear:
            if desc.get('scale_layout')!='row-broadcast':raise ValueError('Unknown prepared scale layout')
            geometry=(weight.shape[0],weight.shape[1]*4//64)
            for field in ('scales','biases'):
                key=name+'.'+field
                if arrays[key].shape!=(geometry[0],1):raise ValueError('Prepared row scale mismatch')
                arrays[key]=mx.broadcast_to(arrays[key],geometry)
            layer=ConvRotLinear.from_packed(weight,arrays[name+'.scales'],arrays[name+'.biases'],group,
                                            arrays.get(name+'.bias'))
            if (layer.out_features,layer.in_features)!=tuple(original[name+'.weight'].shape):
                raise ValueError(f'Prepared linear shape mismatch: {name}')
        elif desc['type']=='embedding' and type(previous) is nn.Embedding:
            if weight.shape!=original[name+'.weight'].shape or weight.dtype!=mx.int8:
                raise ValueError('Prepared embedding shape or dtype mismatch')
            scale=arrays[name+'.scale']
            if scale.shape!=(weight.shape[0],1):
                raise ValueError('Prepared embedding scale mismatch')
            layer=ConvRotEmbedding(weight,scale,dict(convrot=bool(group),convrot_groupsize=group))
        else:raise ValueError(f'Unknown prepared layer type: {name}')
        set_path(module,name,layer)
    expected=dict(tree_flatten(module.parameters()))
    if set(arrays)!=set(expected):raise ValueError('Prepared parameter set mismatch')
    for name,value in arrays.items():
        if value.shape!=expected[name].shape:raise ValueError(f'Prepared parameter shape mismatch: {name}')
    module.load_weights(list(arrays.items()),strict=True)
    return module


def load_or_convert(factory,path,role,key_transform,*,root=None,vae=False,log=print):
    from .native_weights import load_module
    status='disabled' if root is None else 'miss'
    if root is not None:
        try:
            module=load_prepared(factory(),path,role,root)
            if module is not None:
                log(f'[mqwen] prepared weight cache hit: {role}',flush=True)
                return module,'hit'
        except (ValueError,OSError,KeyError,TypeError,AttributeError) as exc:
            # Retry from a fresh module: rejected manifests may have replaced layers.
            status='rejected'
            log(f'[mqwen] prepared cache rejected ({role}): {exc}; using source',flush=True)
    return load_module(factory(),path,key_transform,vae=vae,log=log),status


def save_prepared(module,path,role,root):
    ident=identity(path,role)
    folder=location(root,ident)
    if folder.exists():
        manifest=_validate(folder,ident)
        return dict(role=role,path=str(folder),bytes=manifest['payload_bytes'],reused=True)
    folder.parent.mkdir(parents=True,exist_ok=True)
    arrays=dict(tree_flatten(module.parameters()))
    required=sum(v.nbytes for v in arrays.values())
    if shutil.disk_usage(folder.parent).free<required+1024**3:
        raise ValueError(f'Insufficient cache disk space: need at least {required+1024**3} bytes')
    layers={}
    for name,layer in module.named_modules():
        if type(layer) in (ConvRotLinear,ConvRotEmbedding):
            layers[name]=dict(type='linear' if type(layer) is ConvRotLinear else 'embedding',group=layer.group)
            if type(layer) is ConvRotLinear:
                layers[name]['scale_layout']='row-broadcast'
                # Tensorwise quantization repeats each row's scale across K groups.
                # Store one value and restore a view, avoiding expanded scale storage.
                for field in ('scales','biases'):
                    key=name+'.'+field
                    values=arrays[key]
                    if not bool(mx.all(values==values[:,:1]).item()):
                        raise ValueError('Prepared format requires tensorwise row scales')
                    arrays[key]=values[:,:1]
    staging=Path(tempfile.mkdtemp(prefix='.building-',dir=folder.parent))
    try:
        payload=staging/'weights.safetensors'
        mx.save_safetensors(str(payload),arrays,metadata={'mqwen_adapter':ADAPTER,'role':role})
        manifest=dict(identity=ident,layers=layers,tensors=WeightFile(payload).tensors,
                      source_sha256=_digest(path),payload_sha256=_digest(payload),
                      payload_bytes=payload.stat().st_size,payload_mtime_ns=payload.stat().st_mtime_ns)
        if fingerprint(path)!=ident['source']:raise ValueError('Source changed during preparation')
        (staging/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        os.rename(staging,folder)
    finally:
        if staging.exists():shutil.rmtree(staging)
    return dict(role=role,path=str(folder),bytes=manifest['payload_bytes'],reused=False)


def prepare(args):
    from .native import TEXT_CONFIG,LanguageModel,Qwen21Transformer,Qwen21VAE
    from .native_weights import load_module,text_key,dit_key,vae_key
    from .generate import model_paths,validate_plan
    from .workflow import Workflow
    plan=Workflow(args.workflow).plan();validate_plan(plan)
    paths=model_paths(plan,[args.models]+args.extra_models,{r:getattr(args,r) for r in ('dit','text_encoder','vae')})
    specs=[('text_encoder',lambda:LanguageModel(TEXT_CONFIG,text_mode=False),text_key),
           ('dit',Qwen21Transformer,dit_key),('vae',Qwen21VAE,vae_key)]
    started=time.perf_counter();results=[]
    mx.set_cache_limit(512*1024**2)
    for role,factory,mapping in specs:
        ident=identity(paths[role],role);folder=location(args.output_dir,ident)
        if folder.exists():
            try:
                manifest=_validate(folder,ident,verify=getattr(args,'verify',False))
            except (ValueError,OSError,KeyError,TypeError) as exc:
                raise ValueError(f'Invalid prepared cache at {folder}: {exc}; remove this derived directory and prepare again') from exc
            results.append(dict(role=role,path=str(folder),bytes=manifest['payload_bytes'],reused=True))
            continue
        print(f'[mqwen] preparing {role}',flush=True)
        module=load_module(factory(),paths[role],mapping,vae=role=='vae')
        result=save_prepared(module,paths[role],role,args.output_dir);results.append(result)
        print(f"[mqwen] prepared {role}: {result['bytes']/1024**3:.2f} GiB",flush=True)
        del module;gc.collect();mx.clear_cache();mx.synchronize()
    return dict(cache_root=str(Path(args.output_dir).expanduser().resolve()),models=results,
                total_seconds=round(time.perf_counter()-started,3))
