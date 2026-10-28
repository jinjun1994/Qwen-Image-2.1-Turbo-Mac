# Copyright 2026 The Qwen Team and The HuggingFace Team.
# SPDX-License-Identifier: Apache-2.0
# Reference orchestration adapted from MFLUX QwenImage21Edit (0.23.0).
"""Reference-conditioned MLX editing using the resident Qwen-Image 2.1 weights."""
from pathlib import Path
import inspect
import json
import math

import mlx.core as mx
from mlx import nn
import numpy as np
from PIL import Image, ImageOps
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel
from mflux.models.qwen21.model.qwen21_text_encoder.text_encoder import QwenImage21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_vae.blocks import VAEOperations
from .native_weights import load_module
from .tokenizer import PROMPT_TEMPLATE, SYSTEM_PREFIX

ASSETS=Path(__file__).parent/'assets'


def decode_rgba(vae,latents):
    """Same full VAE decoder as RGB, retaining its fourth channel."""
    mean=mx.array(vae._mean,dtype=mx.float32).reshape(1,64,1,1)
    std=mx.array(vae._std,dtype=mx.float32).reshape(1,64,1,1)
    return vae.decoder(VAEOperations.conv(vae.post_quant_conv,latents*std+mean,True))


def reference_size(image,budget=512):
    ratio=image.width/image.height
    return max(32,round(budget*math.sqrt(ratio)/32)*32),max(32,round(budget/math.sqrt(ratio)/32)*32)


def vision_patches(image):
    """Qwen3-VL 16px patches, 2x2 merge ordering, duplicated temporal frame."""
    white=Image.new('RGB',image.size,'white');white.paste(image,mask=image.getchannel('A'))
    array=np.asarray(white,dtype=np.float32)/255.0
    array=((array-.5)/.5).transpose(2,0,1)
    h,w=image.height//16,image.width//16
    frames=np.stack([array,array],axis=0)
    patches=frames.reshape(1,2,3,h//2,2,16,w//2,2,16)
    patches=patches.transpose(0,3,6,4,7,2,1,5,8).reshape(h*w,3*2*16*16)
    return patches,(1,h,w)


def load_vision(engine):
    if getattr(engine,'vision_encoder',None) is not None:return engine.vision_encoder
    config=json.loads((ASSETS/'qwen21_vl_config.json').read_text())
    vision={k:v for k,v in config['vision_config'].items() if k in inspect.signature(Qwen3VLVisionModel).parameters}
    visual=Qwen3VLVisionModel(**vision,preserve_input_dtype=True)
    for block in visual.blocks:block.mlp.act_fn=QwenImage21TextEncoder._tanh_gelu
    for merger in [visual.merger,*visual.deepstack_merger_list]:merger.act_fn=QwenImage21TextEncoder._exact_gelu
    def key(name):
        name=name.removeprefix('model.')
        return name.removeprefix('visual.') if name.startswith('visual.') else None
    def weight(name,value):
        return value.transpose(0,2,3,4,1) if name=='patch_embed.proj.weight' else value
    engine.log('[mqwen] loading resident vision tower for reference editing',flush=True)
    load_module(visual,engine.paths['text_encoder'],key,weight_transform=weight,log=engine.log)
    # Share the already resident language stack; no second language model or lm_head.
    encoder=QwenImage21TextEncoder.__new__(QwenImage21TextEncoder);nn.Module.__init__(encoder)
    encoder.language_model=engine.text;encoder.visual=visual
    encoder.image_token_id=config['image_token_id'];encoder.merge_size=2
    mx.eval(visual.parameters())
    engine.vision_encoder=encoder
    return encoder


def encode_references(engine,prompt,images):
    encoder=load_vision(engine)
    prefix=' '.join(f'<image{i+1}><|vision_start|><|image_pad|><|vision_end|>' for i in range(len(images)))
    text=PROMPT_TEMPLATE.format(prefix+prompt)
    patches=[];grids=[]
    for image in images:
        values,grid=vision_patches(image);patches.append(values);grids.append(grid)
        text=text.replace('<|image_pad|>','<|placeholder|>'*(grid[1]*grid[2]//4),1)
    text=text.replace('<|placeholder|>','<|image_pad|>')
    ids=engine.tokenizer.encode(text)
    if len(ids)>8192:raise ValueError('Reference prompt exceeds 8192 tokens; use fewer images or a shorter prompt')
    ids=mx.array([ids],dtype=mx.int32)
    hidden=encoder(ids,mx.array(np.concatenate(patches)),mx.array(grids,dtype=mx.int32))
    drop=len(engine.tokenizer.encode(SYSTEM_PREFIX))
    return hidden[:,drop:],ids[0,drop:]==encoder.image_token_id


def composite_mask(generated,source,mask):
    """White edits, black preserves, including all RGBA bytes outside the mask."""
    original=np.asarray(source.convert('RGBA').resize(generated.size,Image.Resampling.LANCZOS))
    edited=np.asarray(generated.convert('RGBA'))
    keep=np.asarray(mask.convert('L').resize(generated.size,Image.Resampling.BILINEAR),dtype=np.float32)[...,None]/255
    return Image.fromarray(np.rint(edited*keep+original*(1-keep)).astype(np.uint8))


def denoise_edit(engine,prompt,width,height,seed,schedule,image_paths,mask_path,check_cancel):
    images=[]
    for path in image_paths:
        check_cancel()
        with Image.open(path) as image:
            if image.width*image.height>16*1024**2 or max(image.size)/min(image.size)>8:
                raise ValueError('Images must be at most 16 MP with aspect ratio between 1:8 and 8:1')
            images.append(ImageOps.exif_transpose(image).convert('RGBA'))
    source=images[0]
    refs=[img.resize(reference_size(img),Image.Resampling.LANCZOS) for img in images]
    embeds,slots=encode_references(engine,prompt,refs)
    mx.eval(embeds)
    if not bool(mx.all(mx.isfinite(embeds)).item()):raise RuntimeError('Non-finite text features')
    layout=QwenImage21Layout.create(slots,[(1,img.height//16,img.width//16) for img in refs]+[(1,height//16,width//16)],engine.transformer.axes)
    conditions=[]
    for image in refs:
        check_cancel()
        pixels=mx.array(np.asarray(image).astype(np.float32)/127.5-1).transpose(2,0,1)[None]
        z=engine.vae.encode(pixels)
        conditions.append(z.transpose(0,2,3,1).reshape(1,-1,64).astype(mx.bfloat16));mx.eval(conditions[-1])
    conditions=mx.concatenate(conditions,axis=1)
    key=mx.random.fold_in(mx.random.key(seed & 0xffffffff),seed>>32) if seed>>32 else mx.random.key(seed)
    latents=mx.random.normal((1,(height//16)*(width//16),64),key=key).astype(mx.bfloat16)
    initial=latents
    mask=blend=source_z=None
    if mask_path:
        with Image.open(mask_path) as opened:mask=opened.convert('L').resize((width,height),Image.Resampling.BILINEAR)
        pixels=mx.array(np.asarray(source.resize((width,height),Image.Resampling.LANCZOS)).astype(np.float32)/127.5-1).transpose(2,0,1)[None]
        source_z=engine.vae.encode(pixels).transpose(0,2,3,1).reshape(1,-1,64).astype(latents.dtype)
        blend=mx.array(np.asarray(mask.resize((width//16,height//16),Image.Resampling.BOX),dtype=np.float32).reshape(1,-1,1)/255)
    import time
    sigmas=mx.array(schedule,dtype=mx.float32);cache=[]
    try:
        for step in range(len(schedule)-1):
            check_cancel();begin=time.perf_counter()
            model_input=mx.concatenate([conditions,latents],axis=1) if not cache else latents
            timestep=(sigmas[step:step+1]*1000).astype(latents.dtype)/1000
            prediction=engine.transformer.forward_reference(model_input,embeds,timestep,layout,cache)
            latents=(latents.astype(mx.float32)+(sigmas[step+1]-sigmas[step])*prediction.astype(mx.float32)).astype(latents.dtype)
            if blend is not None:
                noised=source_z+sigmas[step+1]*(initial-source_z)
                latents=(blend*latents+(1-blend)*noised).astype(latents.dtype)
            mx.eval(latents)
            engine.log(f'[mqwen] denoise {step+1}/{len(schedule)-1}: {time.perf_counter()-begin:.3f}s',flush=True)
    finally:cache.clear()
    return latents,source,mask
